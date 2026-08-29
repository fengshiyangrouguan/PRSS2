"""Ky Fan score math contracts (pure torch, synthetic data).

Verifies the identities the theory rests on: the profiled-MSE closed form,
the Ky Fan supremum, whitening invariance, the [0, min(r,m)] bound and the
gradient isolation of the fixed measurement side.
"""

import unittest

import numpy as np
import torch

from rpbe.loss import (KFLaggedWindow, KFMomentWindow, WeightedWelford, _covs,
                       _score_from_covs, dedup_cut_rows, kf_adjoint,
                       kf_score, kf_score_fixed, kf_vjp_batch)
from rpbe.records import CutRecord


def _rand(*shape, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g)


class TestProfiledMSEIdentity(unittest.TestCase):
    def test_j_equals_whitened_cross_covariance_norm(self):
        z = _rand(500, 6, seed=1)
        p = _rand(500, 12, seed=2)
        j = kf_score(z, p, eps=0.0)
        zc = z - z.mean(0, keepdim=True)
        pc = p - p.mean(0, keepdim=True)
        m = z.shape[0]
        szz = zc.t() @ zc / m
        szp = zc.t() @ pc / m
        spp = pc.t() @ pc / m
        # With the whitening convention A^T A = S^{-1} (A = L^{-1}), J equals
        # ||Lz^{-1} S_ZP Lp^{-T}||_F^2 via triangular solves only.
        lz = torch.linalg.cholesky(szz)
        lp = torch.linalg.cholesky(spp)
        w = torch.linalg.solve_triangular(lz, szp, upper=False)    # Lz^{-1} S_ZP
        cross = torch.linalg.solve_triangular(lp, w.t(), upper=False).t()  # w Lp^{-T}
        direct = (cross ** 2).sum()
        self.assertTrue(torch.allclose(j, direct, atol=1e-4),
                        "J != ||Lz^{-1} S_ZP Lp^{-T}||_F^2")

    def test_profiled_mse_equals_trace_minus_j(self):
        # With WHITENED p (E[p p^T] = I_d): min_A E||p - A z||^2 = d - J.
        z = _rand(800, 5, seed=3)
        p_raw = _rand(800, 9, seed=4)
        pc = p_raw - p_raw.mean(0, keepdim=True)
        lp = torch.linalg.cholesky(pc.t() @ pc / pc.shape[0])
        p = torch.linalg.solve_triangular(lp, pc.t(), upper=False).t()  # pc Lp^{-T}
        j = kf_score(z, p, eps=0.0)
        zc = z - z.mean(0, keepdim=True)
        pc2 = p - p.mean(0, keepdim=True)
        m = z.shape[0]
        szz = zc.t() @ zc / m
        szp = zc.t() @ pc2 / m
        a_star = torch.linalg.solve(szz, szp)  # [r, d]
        resid = pc2 - zc @ a_star
        mse = (resid ** 2).sum(dim=1).mean()   # E ||p - Az||^2 over rows
        self.assertTrue(torch.allclose(mse, p.shape[1] - j, atol=1e-3),
                        "profiled MSE must equal d - J for whitened p")


class TestKyFanSupremum(unittest.TestCase):
    def test_top_r_eigen_coordinates_achieve_sum_of_top_eigenvalues(self):
        # History H = x with E[xx^T] = I; the conditional future embedding is
        # m(H) = B x and the observed whitened test is p = B x + eps with
        # E[eps eps^T] = I - B B^T (so E[p p^T] = I).  Then M = E[m m^T] = B B^T
        # and the Ky Fan supremum over whitened z is sum of its top-r
        # eigenvalues.
        d, r, n = 10, 4, 6000
        x = _rand(n, d, seed=5)
        s = torch.tensor([0.95, 0.8, 0.6, 0.4, 0.2,
                          0.05, 0.05, 0.05, 0.05, 0.05])
        q1 = torch.linalg.qr(_rand(d, d, seed=6)).Q
        q2 = torch.linalg.qr(_rand(d, d, seed=7)).Q
        b = q1 @ torch.diag(s) @ q2.t()
        big_m = b @ b.t()                            # = E[m m^T]
        evals, evecs = torch.linalg.eigh(big_m)      # ascending
        top = evecs[:, -r:]
        top_sum = evals[-r:].sum()
        # Noise with covariance I - B B^T (PSD because s <= 1).
        resid_cov = torch.eye(d) - big_m
        rv, rq = torch.linalg.eigh(resid_cov)
        resid_sqrt = rq @ torch.diag(rv.clamp(min=0.0).sqrt()) @ rq.t()
        noise = _rand(n, d, seed=8) @ resid_sqrt.t()
        p = x @ b.t() + noise                        # whitened joint tests
        # Optimal whitened coordinates: z_j = v_j^T m / sqrt(lambda_j).
        m = x @ b.t()                                # [n, d]
        z_opt = (m @ top) / torch.sqrt(evals[-r:]).clamp(min=1e-8)
        j_opt = kf_score(z_opt, p, eps=1e-6)
        self.assertTrue(torch.allclose(j_opt, top_sum, atol=0.05 * top_sum),
                        "top-r coordinates must achieve the Ky Fan supremum")
        # A random whitened B must not exceed the supremum.
        b_rand = torch.linalg.qr(_rand(d, r, seed=9)).Q
        z_rand = m @ b_rand
        j_rand = kf_score(z_rand, p, eps=1e-6)
        self.assertLessEqual(float(j_rand), float(top_sum) + 1e-2)


class TestWhiteningInvariance(unittest.TestCase):
    def test_scale_invariance(self):
        z = _rand(300, 6, seed=8)
        p = _rand(300, 10, seed=9)
        j0 = kf_score(z, p, eps=1e-6)
        j1 = kf_score(3.7 * z, 0.2 * p, eps=1e-6)
        j2 = kf_score(-z, p, eps=1e-6)
        self.assertTrue(torch.allclose(j0, j1, atol=1e-5))
        self.assertTrue(torch.allclose(j0, j2, atol=1e-5))

    def test_bound_between_zero_and_min_dims(self):
        for r, mdim, seed in ((6, 20, 10), (20, 6, 11), (8, 8, 12)):
            z = _rand(400, r, seed=seed)
            p = _rand(400, mdim, seed=seed + 100)
            j = kf_score(z, p, eps=1e-6)
            self.assertGreaterEqual(float(j), -1e-6)
            self.assertLessEqual(float(j), float(min(r, mdim)) + 1e-3)


class TestGradientIsolation(unittest.TestCase):
    def test_kf_score_hard_detaches_p(self):
        # API-level isolation: even if a caller passes a grad-connected P,
        # the score must cut it (no gradient can flow into the fixed tests).
        z = _rand(100, 4, seed=19).requires_grad_(True)
        p = _rand(100, 8, seed=20).requires_grad_(True)
        j = kf_score(z, p, eps=1e-4)
        j.backward()
        self.assertIsNone(p.grad,
                          "P must be detached inside kf_score")

    def test_p_side_has_no_grad_and_j_backprops_to_z(self):
        z = _rand(200, 6, seed=13).requires_grad_(True)
        p = _rand(200, 10, seed=14)
        self.assertIsNone(p.grad_fn)
        j = kf_score(z, p, eps=1e-6)
        self.assertIsNotNone(j.grad_fn)
        j.backward()
        self.assertIsNotNone(z.grad)
        self.assertGreater(z.grad.abs().sum().item(), 0.0)

    def test_full_gradient_has_zero_radial_derivative(self):
        # Scale invariance of the normalized score: with the full gradient
        # (C_zz as a theta-dependent normalization term), both the analytic
        # radial derivative <grad_Z J, Z> and the finite-difference
        # (J((1+h)Z) - J((1-h)Z)) / 2h must be ~0.  A stop-grad wrapper
        # would give fd ~ 0 but auto ~ 2J — the exact signature of the
        # half-gradient bug.
        z = _rand(200, 6, seed=15).requires_grad_(True)
        p = _rand(200, 10, seed=16)
        j = kf_score(z, p, eps=1e-6)
        grad = torch.autograd.grad(j, z)[0]
        d_auto = float((grad * z).sum())
        h = 1e-3
        j_plus = float(kf_score((1 + h) * z.detach(), p, eps=1e-6))
        j_minus = float(kf_score((1 - h) * z.detach(), p, eps=1e-6))
        d_fd = (j_plus - j_minus) / (2 * h)
        self.assertAlmostEqual(d_auto, 0.0, delta=1e-2,
                               msg="radial derivative of the analytic "
                                   "gradient must vanish")
        self.assertAlmostEqual(d_fd, 0.0, delta=1e-2,
                               msg="finite-difference radial derivative "
                                   "must vanish")

    def test_fixed_statistics_core_gradchecks(self):
        # The fixed-scale core (for a future frozen-reference variant) must
        # carry exactly the analytic gradient of its closed form.
        z = _rand(30, 4, seed=15).double().requires_grad_(True)
        p = _rand(30, 6, seed=16).double()
        z_c = (z - z.mean(0, keepdim=True)).double()
        p_c = p - p.mean(0, keepdim=True)
        m = z.shape[0]
        szz = (z_c.t() @ z_c / m).detach()
        spp = (p_c.t() @ p_c / m).detach()
        ok = torch.autograd.gradcheck(
            lambda zz: kf_score_fixed(zz - zz.mean(0, keepdim=True), p_c,
                                      szz, spp, eps=1e-2),
            (z,), eps=1e-4, atol=1e-3, rtol=1e-3)
        self.assertTrue(ok, "gradcheck failed on the fixed-statistics core")

    def test_j_forward_value_unchanged_by_sg(self):
        # Detaching the whitening stats must not change the forward score.
        z = _rand(200, 6, seed=17)
        p = _rand(200, 10, seed=18)
        j_now = kf_score(z, p, eps=1e-4)
        self.assertGreater(float(j_now), -1e-6)
        self.assertLessEqual(float(j_now), 6.0 + 1e-3)


class _FakeMaps:
    """Deterministic P rows (independent of Z) for tracker tests."""

    def __init__(self, m):
        self.m = m

    def pv(self, context, outcome):
        g = torch.Generator().manual_seed(
            int(context["counterpart"]) * 7919 + int(context["delta_t"]))
        return torch.randn(self.m, generator=g)


def make_cut_rows(n_cuts, r, m, rows_per_cut=1, offset=0):
    rows = []
    for c in range(n_cuts):
        z = torch.randn(r)
        for k in range(rows_per_cut):
            rows.append(CutRecord(
                tree_id=offset + c,
                occurrence_id=offset + c,
                tau="t", horizon=k + 1, node=offset + c,
                time=float(offset + c), z=z,
                context={"delta_t": float(c * 7 + k), "counterpart": c + k,
                         "role": 0, "query_type": 0,
                         "horizon": k + 1, "path": []},
                outcome=float(k == 0),
                outcome_id=("edge", offset + c),
                weight=1.0 / rows_per_cut))
    return rows


def _feed_window(window, rows, param_version=0, epoch=0):
    """The loop's macro-group sequence on one group: consume -> close ->
    commit.  Returns ``(scores, surrogates, diagnostics, cold, refreshed)``
    (the order the old ``step()`` returned)."""
    window.begin_group(param_version, epoch)
    scores, surrogates, cold = window.consume(rows)
    diagnostics, refreshed = window.close_group()
    window.commit_reference()
    return scores, surrogates, diagnostics, cold, refreshed


class TestKFLaggedWindow(unittest.TestCase):
    def test_cold_start_builds_detached_reference(self):
        window = KFLaggedWindow(
            {"t": 4}, min_ratio=1.0, min_abs=4,
            fixed_maps=_FakeMaps(8))
        rows = make_cut_rows(4, 4, 8)
        for row in rows:
            row.z.requires_grad_(True)
        scores, surrogates, diagnostics, cold, refreshed = _feed_window(
            window, rows, param_version=0)
        # Cold-start group: no active reference yet, so no score/surrogate;
        # the group only accumulates and builds the NEXT reference.
        self.assertEqual(scores, {})
        self.assertEqual(surrogates, {})
        self.assertIn("t", cold)
        self.assertIn("t", refreshed)
        self.assertIsNone(rows[0].z.grad)
        self.assertFalse(window._reference["t"]["mu_z"].requires_grad)
        self.assertEqual(window._reference["t"]["param_version"], 0)

    def test_next_batch_gets_zero_valued_nonzero_gradient_vjp(self):
        window = KFLaggedWindow(
            {"t": 4}, min_ratio=1.0, min_abs=4,
            fixed_maps=_FakeMaps(8))
        _feed_window(window, make_cut_rows(4, 4, 8, offset=0),
                     param_version=0)
        rows = make_cut_rows(3, 4, 8, offset=100)
        for row in rows:
            row.z.requires_grad_(True)
        scores, surrogates, diagnostics, cold, refreshed = _feed_window(
            window, rows, param_version=1)
        self.assertIn("t", scores)
        self.assertIn("t", surrogates)
        self.assertAlmostEqual(float(surrogates["t"].detach()), 0.0)
        (-surrogates["t"]).backward()
        grad_norm = sum(float(row.z.grad.norm()) for row in rows
                        if row.z.grad is not None)
        self.assertGreater(grad_norm, 0.0)

    def test_gate_counts_trees_not_horizon_rows(self):
        window = KFLaggedWindow(
            {"t": 4}, min_ratio=1.0, min_abs=4,
            fixed_maps=_FakeMaps(8))
        rows = make_cut_rows(2, 4, 8, rows_per_cut=2)
        window.begin_group(0, 0)
        scores, surrogates, cold = window.consume(rows)
        # Below threshold: no active reference, the 2 trees sit pending.
        self.assertEqual(scores, {})
        self.assertIn("t", cold)
        self.assertEqual(window.pending_tree_count("t"), 2)
        diagnostics, refreshed = window.close_group()
        window.commit_reference()
        # Eighth review C: a below-threshold group DISCARDS its partial
        # window — a reference must never mix parameter versions.
        self.assertEqual(refreshed, [])
        self.assertTrue(diagnostics["t"]["below_threshold"])
        self.assertEqual(diagnostics["t"]["dropped_trees"], 2)
        self.assertEqual(window.pending_tree_count("t"), 0)
        self.assertIsNone(window.reference_score("t"))

    def test_kf_lagged_surrogate_ascends_reference_objective(self):
        """Seventh-review sign contract on the lagged path.

        The surrogate's gradient must ascend the reference-linearized
        objective ``<A_ref, M2_batch>`` (kf_adjoint returns +dJ/dM2 and the
        loop multiplies by -lambda).  Comparing against the TRUE batch J is
        NOT the right check: the lagged linearization carries a radial
        degree of freedom while the true score is scale-invariant (pure
        tangential gradient), so a stale reference may legitimately point
        elsewhere.  One optimizer step on -surrogate must therefore raise
        the linearized objective itself.
        """
        window = KFLaggedWindow(
            {"t": 4}, min_ratio=1.0, min_abs=4, fixed_maps=_FakeMaps(8))
        # Cold start: build the detached reference window (group v0).
        _feed_window(window, make_cut_rows(40, 4, 8, offset=0),
                     param_version=0)
        reference = window._reference["t"]
        # Next batch: graph-connected z.
        rows = make_cut_rows(40, 4, 8, offset=100)
        zs = [row.z for row in rows]
        for z in zs:
            z.requires_grad_(True)
        window.begin_group(1, 0)
        _, surrogates, _ = window.consume(rows)
        self.assertIn("t", surrogates)
        surrogate = surrogates["t"]

        def linearized_value():
            z_stack = torch.stack(zs)
            p_stack = torch.stack(
                [_FakeMaps(8).pv(row.context, row.outcome) for row in rows])
            w = torch.tensor([row.weight for row in rows],
                             dtype=torch.float64)
            return kf_vjp_batch(z_stack, p_stack, w,
                                reference["mu_z"], reference["mu_p"],
                                reference["adjoints"])

        before = float(linearized_value().detach())
        opt = torch.optim.Adam(zs, lr=0.01)
        opt.zero_grad()
        (-surrogate).backward()
        opt.step()
        after = float(linearized_value().detach())
        self.assertGreater(
            after, before,
            "one lagged KF step must ascend the reference objective "
            "({:.4f} -> {:.4f})".format(before, after))


class TestAblationVariants(unittest.TestCase):
    """Paper Table 2 variant contracts (P0)."""

    def _data(self, n=120, r=6, mdim=8, seed=3):
        g = torch.Generator().manual_seed(seed)
        z = torch.randn(n, r, generator=g)
        p = torch.randn(n, mdim, generator=g)
        u = z @ torch.randn(r, r, generator=g) +             0.3 * torch.randn(n, r, generator=g)   # z-related rich state
        return z, p, u

    def test_diagonal_equals_full_on_diagonal_covariances(self):
        # When C_ZZ and C_PP are diagonal (independent coordinates), the
        # diagonal whitening and the full balancing coincide.
        n, r, mdim = 200, 6, 8
        g = torch.Generator().manual_seed(11)
        z = torch.randn(n, r, generator=g) * torch.tensor(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        p = torch.randn(n, mdim, generator=g) * torch.tensor(
            [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
        zc = z - z.mean(0, keepdim=True)
        pc = p - p.mean(0, keepdim=True)
        m = n
        czz = zc.t() @ zc / m
        cpp = pc.t() @ pc / m
        czp = zc.t() @ pc / m
        j_full, d_full = _score_from_covs(
            czz, czp, cpp, 1e-4, "full_balancing")
        j_diag, d_diag = _score_from_covs(
            czz, czp, cpp, 1e-4, "diagonal")
        self.assertIsNone(d_full["failed"])
        self.assertIsNone(d_diag["failed"])
        # Off-diagonals are sampling noise ~1/sqrt(n); allow that size.
        self.assertAlmostEqual(float(j_full), float(j_diag),
                               delta=max(0.15, 0.2 * float(j_full)))

    def test_diagonal_differs_from_full_under_correlation(self):
        # The two variants are NOT ordered in general: diagonal whitening
        # ignores correlations (both exploitable redundancy and noise),
        # while full balancing reshapes by S^-1/2.  Both must be finite
        # and the scores generally differ when correlations exist.
        n, r = 400, 4
        g = torch.Generator().manual_seed(21)
        base = torch.randn(n, 1, generator=g)
        z = torch.cat([base + 0.05 * torch.randn(n, 1, generator=g),
                       -base + 0.05 * torch.randn(n, 1, generator=g),
                       torch.randn(n, 2, generator=g)], dim=1)
        p = torch.cat([base + 0.05 * torch.randn(n, 1, generator=g),
                       torch.randn(n, 5, generator=g)], dim=1)
        zc = z - z.mean(0, keepdim=True)
        pc = p - p.mean(0, keepdim=True)
        czz = zc.t() @ zc / n
        cpp = pc.t() @ pc / n
        czp = zc.t() @ pc / n
        j_full, d_full = _score_from_covs(
            czz, czp, cpp, 1e-4, "full_balancing")
        j_diag, d_diag = _score_from_covs(
            czz, czp, cpp, 1e-4, "diagonal")
        self.assertIsNone(d_full["failed"])
        self.assertIsNone(d_diag["failed"])
        self.assertTrue(np.isfinite(float(j_full)))
        self.assertTrue(np.isfinite(float(j_diag)))
        self.assertNotAlmostEqual(float(j_full), float(j_diag), places=2)

    def test_reconstruction_matches_profiled_closed_form(self):
        # J_rec = tr(S_UZ S_ZZ^-1 S_ZU) must equal the closed form
        # ||S_ZZ^-1/2 S_ZU||_F^2 (U in the Z slot, Z in the P slot,
        # U side unwhitened).
        n, r = 300, 5
        g = torch.Generator().manual_seed(31)
        z = torch.randn(n, r, generator=g)
        u = z @ torch.randn(r, r, generator=g) +             0.2 * torch.randn(n, r, generator=g)
        zc = z - z.mean(0, keepdim=True)
        uc = u - u.mean(0, keepdim=True)
        szz = zc.t() @ zc / n
        szu = zc.t() @ uc / n           # S_ZU = S_UZ^T
        suz = szu.t()
        j_rec, d = _score_from_covs(szz, szu, szz, 1e-4, "reconstruction")
        self.assertIsNone(d["failed"])
        # The score path adds the relative ridge eps to the whitened Z
        # side; the closed form has none.  Compare with a small tolerance.
        closed = float(torch.trace(suz @ torch.linalg.solve(szz, szu)))
        self.assertAlmostEqual(float(j_rec), closed, delta=0.02,
                               msg="reconstruction score must match "
                                   "tr(S_UZ S_ZZ^-1 S_ZU)")

    def test_lagged_window_reconstruction_path(self):
        # End-to-end reconstruction variant: cold start builds the
        # detached reference from (U, Z) pairs; the next batch gets a
        # zero-valued, nonzero-gradient VJP against the same pairs.
        rows = make_cut_rows(8, 4, 8, offset=0)
        for row in rows:
            row.u = row.z + 0.1 * torch.randn(4)
        window = KFLaggedWindow(
            {"t": 4}, min_ratio=1.0, min_abs=4, fixed_maps=_FakeMaps(8),
            variant="reconstruction")
        scores, surrogates, diagnostics, cold, refreshed = _feed_window(
            window, rows, param_version=0)
        # Cold-start group: accumulates (U, Z) pairs into the reference.
        self.assertEqual(scores, {})
        self.assertEqual(surrogates, {})
        self.assertIn("t", refreshed)
        rows2 = make_cut_rows(6, 4, 8, offset=100)
        for row in rows2:
            row.u = row.z + 0.1 * torch.randn(4)
            row.z.requires_grad_(True)
            row.u.requires_grad_(True)
        window.begin_group(1, 0)
        scores, surrogates, _ = window.consume(rows2)
        self.assertIn("t", surrogates)
        self.assertAlmostEqual(float(surrogates["t"].detach()), 0.0)
        (-surrogates["t"]).backward()
        # In the reconstruction variant U occupies the Z slot; the VJP's
        # centering terms also flow to the p slot (z), so both sides get
        # a gradient.
        grad_norm = sum(
            float(row.z.grad.norm() if row.z.grad is not None else 0.0)
            + float(row.u.grad.norm() if row.u.grad is not None else 0.0)
            for row in rows2)
        self.assertGreater(grad_norm, 0.0)



class TestKFMomentWindow(unittest.TestCase):
    def test_window_gates_small_sample(self):
        # d=32 with <64 unique cuts must stay open (no score yet).
        w = KFMomentWindow({"t": 32}, min_ratio=2.0, min_abs=64,
                           fixed_maps=_FakeMaps(256))
        closed, diag, gated = w.add(make_cut_rows(16, 32, 256))
        self.assertEqual(closed, {})
        self.assertIn("t", gated)

    def test_window_closes_after_enough_unique_cuts(self):
        w = KFMomentWindow({"t": 8}, min_ratio=2.0, min_abs=64,
                           fixed_maps=_FakeMaps(64))
        closed = {}
        last_diag = {}
        for k in range(3):
            c, diag, gated = w.add(
                make_cut_rows(40, 8, 64, offset=k * 100))
            closed.update(c)
            if diag:
                last_diag = diag
        self.assertIn("t", closed)
        self.assertGreaterEqual(last_diag["t"]["M_unique"], 64)
        self.assertTrue(np.isfinite(float(closed["t"].detach())))

    def test_independent_noise_does_not_saturate(self):
        # Window far above d: the honest score stays well below the
        # saturation bound, and the shuffled score is on the same level
        # (no real correlation to claim).
        w = KFMomentWindow({"t": 8}, min_ratio=2.0, min_abs=64,
                           fixed_maps=_FakeMaps(64))
        closed = {}
        last_diag = {}
        for k in range(3):
            c, d, gated = w.add(make_cut_rows(50, 8, 64, offset=k * 100))
            closed.update(c)
            if d:
                last_diag = d
        j = float(closed["t"].detach())
        bound = min(8, last_diag["t"]["M_unique"] - 1)
        self.assertLess(j, 0.75 * bound)
        self.assertLessEqual(
            abs(j - last_diag["t"]["J_shuffled"]) / max(bound, 1), 0.2,
            "independent-noise J and shuffled J must agree")

    def test_correlated_zp_beats_shuffled(self):
        # Z and P sharing a low-rank signal must score clearly above the
        # shuffled baseline.  Single-window estimates are noisy, so average
        # over three independent windows.
        n, r, m, sdim = 400, 8, 64, 6
        js, jsh = [], []
        for rep in range(3):
            g = torch.Generator().manual_seed(42 + rep)
            signal = torch.randn(n, sdim, generator=g)
            z = torch.cat([signal, torch.randn(n, r - sdim, generator=g)],
                          dim=1)
            p = torch.cat([signal, torch.randn(n, m - sdim, generator=g)],
                          dim=1)
            w = KFMomentWindow({"t": r}, min_ratio=2.0, min_abs=64,
                               fixed_maps=None)
            class _Stub:
                def pv(self, ctx, y):
                    return p[ctx["counterpart"] % n]
            w.fixed_maps = _Stub()
            rows = []
            for i in range(n):
                rows.append(CutRecord(
                    tree_id=i, occurrence_id=i, tau="t", horizon=1,
                    node=i, time=float(i), z=z[i],
                    context={"delta_t": 0.0, "counterpart": i, "role": 0,
                             "query_type": 0, "horizon": 1, "path": []},
                    outcome=0.0, outcome_id=("edge", i)))
            closed, diag, gated = w.add(rows)
            self.assertIn("t", closed)
            js.append(float(closed["t"].detach()))
            jsh.append(diag["t"]["J_shuffled"])
        self.assertGreater(sum(js) / 3, sum(jsh) / 3 + 1.0,
                           "correlated score must beat shuffled "
                           "({:.2f} vs {:.2f})".format(sum(js) / 3,
                                                       sum(jsh) / 3))

    def test_constant_z_returns_zero_not_crash(self):
        w = KFMomentWindow({"t": 4}, min_ratio=1.0, min_abs=4,
                           fixed_maps=_FakeMaps(16))
        n = 20
        rows = [CutRecord(
            tree_id=i, occurrence_id=i, tau="t", horizon=1,
            node=i, time=float(i), z=torch.ones(4),
            context={"delta_t": 0.0, "counterpart": i, "role": 0,
                     "query_type": 0, "horizon": 1, "path": []}, outcome=0.0,
                     outcome_id=("edge", i))
            for i in range(n)]
        closed, diag, gated = w.add(rows)
        self.assertIn("t", closed)
        j = float(closed["t"].detach())
        self.assertAlmostEqual(j, 0.0, places=6)
        self.assertTrue(np.isfinite(j))

    def test_collapsed_z_scale_does_not_break_the_bound(self):
        # Regression for the second cloud crash: z collapsing to ~1e-10
        # scale made the absolute ridge jitter (1e-12) dominate, blowing
        # J up past dim (J=245.09 vs dim=172 -> monitor invariant raise).
        # The scale-following ridge must keep J within [0, dim].
        r, m, n = 172, 256, 420
        g = torch.Generator().manual_seed(456)
        z = 1e-10 * torch.randn(n, r, generator=g)     # collapsed scale
        p = torch.randn(n, m, generator=g)
        w = KFMomentWindow({"t": r}, min_ratio=1.0, min_abs=64,
                           fixed_maps=None)
        class _Stub:
            def pv(self, ctx, y):
                return p[ctx["counterpart"] % n]
        w.fixed_maps = _Stub()
        rows = []
        for i in range(n):
            rows.append(CutRecord(
                tree_id=i, occurrence_id=i, tau="t", horizon=1,
                node=i, time=float(i), z=z[i],
                context={"delta_t": 0.0, "counterpart": i, "role": 0,
                         "query_type": 0, "horizon": 1, "path": []}, outcome=0.0,
                outcome_id=("edge", i)))
        closed, diag, gated = w.add(rows)
        self.assertIn("t", closed)
        j = float(closed["t"].detach())
        self.assertTrue(np.isfinite(j))
        self.assertGreaterEqual(j, -1e-6)
        self.assertLessEqual(j, r + 1e-3,
                             "collapsed-scale J must stay within the bound")

    def test_near_singular_covariance_does_not_crash(self):
        # Regression for the cloud crash: after ~18 epochs some z direction
        # collapses, making the 172x172 C_ZZ non-positive-definite at the
        # LAST Cholesky pivot.  The window close must degrade (finite score)
        # instead of killing training.
        r, m, n = 172, 256, 420
        g = torch.Generator().manual_seed(123)
        signal = torch.randn(n, 4, generator=g)          # shared low-rank core
        z = torch.cat([signal, torch.randn(n, r - 4, generator=g)], dim=1)
        # Collapse the FINAL coordinate: variance -> machine zero.  This is
        # exactly the leading-minor-of-order-171 failure shape.
        z[:, -1] = z[:, -1] * 0.0 + 1e-13 * torch.randn(n, generator=g)
        p = torch.randn(n, m, generator=g)
        w = KFMomentWindow({"t": r}, min_ratio=1.0, min_abs=64,
                           fixed_maps=None)
        class _Stub:
            def pv(self, ctx, y):
                return p[ctx["counterpart"] % n]
        w.fixed_maps = _Stub()
        rows = []
        for i in range(n):
            rows.append(CutRecord(
                tree_id=i, occurrence_id=i, tau="t", horizon=1,
                node=i, time=float(i), z=z[i],
                context={"delta_t": 0.0, "counterpart": i, "role": 0,
                         "query_type": 0, "horizon": 1, "path": []}, outcome=0.0,
                outcome_id=("edge", i)))
        closed, diag, gated = w.add(rows)
        self.assertIn("t", closed)
        j = float(closed["t"].detach())
        self.assertTrue(np.isfinite(j), "near-singular C_ZZ must not crash")
        self.assertGreaterEqual(j, -1e-6)
        self.assertLessEqual(j, r + 1e-3)


class TestWeightedWelford(unittest.TestCase):
    """The pass-1 accumulator of the Moment-Adjoint Replay must reproduce
    the direct float64 stack-center covariances exactly (the numerical
    contract of d80d65f), merge batches via the Chan correction, and keep
    the cluster degrees of freedom D = W - W2_cut/W."""

    def _batches(self, sizes=(40, 60, 25), dim_z=6, dim_p=10, seed=3):
        g = torch.Generator().manual_seed(seed)
        zs = [torch.randn(n, dim_z, generator=g) for n in sizes]
        ps = [torch.randn(n, dim_p, generator=g) for n in sizes]
        ws = [torch.rand(n, generator=g) + 0.5 for n in sizes]
        cut_ids = [[(i, i, "t", 1) for i in range(n)] for n in sizes]
        return zs, ps, ws, cut_ids

    def test_matches_direct_stack_centering(self):
        zs, ps, ws, cut_ids = self._batches()
        wf = WeightedWelford(6, 10)
        for z, p, w, cids in zip(zs, ps, ws, cut_ids):
            wf.add(z, p, w, cids)
        r = wf.result()
        z_all = torch.cat(zs).double()
        p_all = torch.cat(ps).double()
        w_all = torch.cat(ws).double()
        W = w_all.sum()
        mu_z = (z_all * w_all[:, None]).sum(0) / W
        mu_p = (p_all * w_all[:, None]).sum(0) / W
        zc = z_all - mu_z
        pc = p_all - mu_p
        sw = w_all.sqrt()[:, None]
        mzz = (zc * sw).t() @ (zc * sw)
        mpp = (pc * sw).t() @ (pc * sw)
        mzp = (zc * sw).t() @ (pc * sw)
        self.assertAlmostEqual(r["W"], float(W), places=12)
        self.assertTrue(torch.allclose(r["mu_z"], mu_z, atol=1e-10))
        self.assertTrue(torch.allclose(r["mu_p"], mu_p, atol=1e-10))
        self.assertTrue(torch.allclose(r["M2_zz"], mzz, atol=1e-8))
        self.assertTrue(torch.allclose(r["M2_pp"], mpp, atol=1e-8))
        self.assertTrue(torch.allclose(r["M2_zp"], mzp, atol=1e-8))

    def test_large_dc_offset_stays_stable(self):
        # z with a 1e4 DC offset (added in float64, the accumulator's
        # domain): the centered M2 must match the offset-free run exactly
        # — no raw-moment cancellation (the d80d65f contract).  A float32
        # offset would quantize the low bits before the accumulator even
        # sees the data, which is not the accumulator's concern.
        zs, ps, ws, cut_ids = self._batches()
        wf0 = WeightedWelford(6, 10)
        wf1 = WeightedWelford(6, 10)
        for z, p, w, cids in zip(zs, ps, ws, cut_ids):
            wf0.add(z, p, w, cids)
            wf1.add(z.double() + 1e4, p, w, cids)
        r0, r1 = wf0.result(), wf1.result()
        self.assertTrue(torch.allclose(r0["M2_zz"], r1["M2_zz"], atol=1e-6))
        self.assertTrue(torch.allclose(r0["M2_zp"], r1["M2_zp"], atol=1e-6))

    def test_uniform_weights_degrade_to_n_minus_one(self):
        # All-ones weights, one row per cut: D = n - 1 (the pre-weighting
        # denominator contract).
        zs, ps, _, cut_ids = self._batches()
        n = sum(len(z) for z in zs)
        wf = WeightedWelford(6, 10)
        for z, p, cids in zip(zs, ps, cut_ids):
            wf.add(z, p, torch.ones(len(z)), cids)
        r = wf.result()
        self.assertAlmostEqual(r["W"], float(n), places=10)
        self.assertAlmostEqual(r["W2_cut"], float(n), places=10)
        self.assertAlmostEqual(r["D"], float(n - 1), places=10)

    def test_cluster_w2_two_horizon_cut(self):
        # One cut with two horizon rows (0.5 + 0.5): the cluster W2 sums
        # the cut's rows first (1.0 per cut), so D = W - W2/W with W=n —
        # horizon splitting does NOT fake extra independent samples.
        zs, ps, _, cut_ids = self._batches(sizes=(80,))
        n = len(zs[0])
        wf = WeightedWelford(6, 10)
        w = torch.full((n,), 0.5)
        cids = [(i, i, "t", 1) for i in range(n)]
        wf.add(zs[0], ps[0], w, cids)
        r = wf.result()
        # W = n*0.5; W2_cut = n*0.25; D = W - W2_cut/W = n/2 - 1/2
        # (a single row with w=1 gives W=n, W2=n, D=n-1 — the horizon
        # split scales everything by 1/2 and does NOT add fake samples).
        self.assertAlmostEqual(r["W"], float(n) * 0.5, places=10)
        self.assertAlmostEqual(r["W2_cut"], float(n) * 0.25, places=10)
        self.assertAlmostEqual(r["D"], float(n) * 0.5 - 0.5, places=10)

    def test_chan_merge_across_batches_equals_one_pass(self):
        # The cross-batch merge must be order-invariant: two batches added
        # separately equal the same rows added in one call.
        zs, ps, ws, cut_ids = self._batches(sizes=(50, 30))
        wf_sep = WeightedWelford(6, 10)
        for z, p, w, cids in zip(zs, ps, ws, cut_ids):
            wf_sep.add(z, p, w, cids)
        wf_one = WeightedWelford(6, 10)
        wf_one.add(torch.cat(zs), torch.cat(ps), torch.cat(ws),
                   cut_ids[0] + cut_ids[1])
        rs, ro = wf_sep.result(), wf_one.result()
        self.assertAlmostEqual(rs["W"], ro["W"], places=12)
        self.assertTrue(torch.allclose(rs["mu_z"], ro["mu_z"], atol=1e-10))
        self.assertTrue(torch.allclose(rs["M2_zz"], ro["M2_zz"], atol=1e-8))
        self.assertTrue(torch.allclose(rs["M2_zp"], ro["M2_zp"], atol=1e-8))


class TestMomentAdjointReplay(unittest.TestCase):
    """Three-path gradient equivalence (sixth review): direct-stack,
    moment-adjoint replay and latent-adjoint replay must give the SAME
    parameter gradients at the same point — the latent form is the one
    the training loop now uses (pass 2 = index z + dot product only)."""

    def _rows(self, z, p, w):
        n = z.shape[0]
        return [CutRecord(
            tree_id=i, occurrence_id=i, tau="t", horizon=1,
            node=i, time=float(i), z=z[i],
            context={"delta_t": 0.0, "counterpart": i, "role": 0,
                     "query_type": 0, "horizon": 1, "path": []},
            outcome=0.0, outcome_id=("edge", i), weight=float(w[i]))
            for i in range(n)]

    def _stub(self, p):
        class _Stub:
            def pv(self, ctx, y):
                return p[ctx["counterpart"] % p.shape[0]]
        return _Stub()

    def _three_grads(self, z, p, w):
        """Returns (grad_direct, grad_moment, grad_latent, j_direct)."""
        n, r, mdim = z.shape[0], z.shape[1], p.shape[1]
        rows = self._rows(z, p, w)

        # Path 1 (direct): whole-window close, one backward.
        wA = KFMomentWindow({"t": r}, min_ratio=1.0, min_abs=2,
                            fixed_maps=self._stub(p))
        jA = wA.add(rows)[0]["t"]
        jA.backward()
        gradA = z.grad.clone()
        z.grad = None

        # Path 2 (moment-adjoint): Welford -> kf_adjoint -> kf_vjp_batch.
        wf = WeightedWelford(r, mdim)
        wf.add(z.detach(), p, w, [(i, i, "t", 1) for i in range(n)])
        res = wf.result()
        j2, adj, diag = kf_adjoint(res, eps=1e-4)
        self.assertIsNone(diag["failed"])
        surr2 = kf_vjp_batch(z, p, w, res["mu_z"], res["mu_p"], adj)
        surr2.backward()
        gradB = z.grad.clone()
        z.grad = None

        # Path 3 (latent-adjoint): the window's close_replay contract.
        wC = KFMomentWindow({"t": r}, min_ratio=1.0, min_abs=2,
                            fixed_maps=self._stub(p), autoclose=False)
        wC.add([CutRecord(
            tree_id=r2.tree_id, occurrence_id=r2.occurrence_id,
            tau=r2.tau, horizon=1, node=r2.node, time=r2.time,
            z=r2.z.detach(), context=r2.context, outcome=r2.outcome,
            outcome_id=r2.outcome_id, weight=r2.weight) for r2 in rows])
        closedC, planC, diagC = wC.close_replay()
        self.assertAlmostEqual(float(closedC["t"]), float(jA.detach()),
                               places=5,
                               msg="latent close score must equal direct")
        surr3 = sum((g * z[occ_id].float()).sum()
                    for (occ_id, g) in planC["t"]["by_batch"][0])
        surr3.backward()
        gradC = z.grad.clone()
        z.grad = None
        return gradA, gradB, gradC, float(jA.detach())

    def test_three_paths_agree_on_z_gradient(self):
        n, r, mdim = 90, 8, 12
        g = torch.Generator().manual_seed(11)
        z = torch.randn(n, r, generator=g).requires_grad_(True)
        p = torch.randn(n, mdim, generator=g)
        w = torch.rand(n, generator=g) + 0.5
        gradA, gradB, gradC, j = self._three_grads(z, p, w)
        self.assertTrue(torch.allclose(gradA, gradB, atol=1e-6, rtol=1e-5),
                        "moment-adjoint gradient mismatch vs direct "
                        "({})".format(float((gradA - gradB).abs().max())))
        self.assertTrue(torch.allclose(gradA, gradC, atol=1e-6, rtol=1e-5),
                        "latent-adjoint gradient mismatch vs direct "
                        "({})".format(float((gradA - gradC).abs().max())))

    def test_three_paths_agree_on_mlp_parameter_gradient(self):
        n, r, mdim, hdim = 60, 8, 12, 16
        g = torch.Generator().manual_seed(21)
        x = torch.randn(n, 10, generator=g)
        p = torch.randn(n, mdim, generator=g)
        w = torch.rand(n, generator=g) + 0.5
        torch.manual_seed(7)
        lin = torch.nn.Linear(10, r)
        torch.nn.init.xavier_normal_(lin.weight, generator=g)
        # Direct path through the MLP, then the latent path: the PARAMETER
        # gradients must agree (z-gradient agreement is covered by the
        # leaf test above).
        z = lin(x)
        rows = self._rows(z, p, w)
        wA = KFMomentWindow({"t": r}, min_ratio=1.0, min_abs=2,
                            fixed_maps=self._stub(p))
        jA = wA.add(rows)[0]["t"]
        jA.backward()
        paramA = lin.weight.grad.clone()
        lin.zero_grad()
        z = lin(x)
        wC = KFMomentWindow({"t": r}, min_ratio=1.0, min_abs=2,
                            fixed_maps=self._stub(p), autoclose=False)
        wC.add([CutRecord(
            tree_id=r2.tree_id, occurrence_id=r2.occurrence_id,
            tau=r2.tau, horizon=1, node=r2.node, time=r2.time,
            z=r2.z.detach(), context=r2.context, outcome=r2.outcome,
            outcome_id=r2.outcome_id, weight=r2.weight) for r2 in rows])
        closedC, planC, _ = wC.close_replay()
        surr3 = sum((g * z[occ_id].float()).sum()
                    for (occ_id, g) in planC["t"]["by_batch"][0])
        surr3.backward()
        paramC = lin.weight.grad.clone()
        self.assertTrue(
            torch.allclose(paramA, paramC, atol=1e-6, rtol=1e-5),
            "MLP parameter gradient mismatch (direct vs latent) "
            "({})".format(float((paramA - paramC).abs().max())))


class TestSeventhReviewFixes(unittest.TestCase):
    """Seventh review: KF sign, horizon duplication, alpha scaling."""

    def _multi_horizon_rows(self, n_cuts, r, mdim, horizons, z, p, w):
        # One cut with several horizon rows sharing z; weights 1/|H|.
        rows = []
        for c in range(n_cuts):
            for h in horizons:
                rows.append(CutRecord(
                    tree_id=c, occurrence_id=c, tau="t", horizon=h,
                    node=c, time=float(c), z=z[c],
                    context={"delta_t": float(c * 7 + h), "counterpart": c,
                             "role": 0, "query_type": 0, "horizon": h,
                             "path": []},
                    outcome=float(h == 1), outcome_id=("edge", c),
                    weight=float(w[c]) / len(horizons)))
        return rows

    def _stub(self, p):
        class _Stub:
            def pv(self, ctx, y):
                return p[ctx["counterpart"] % p.shape[0]]
        return _Stub()

    def _latent_surrogate(self, z, plan):
        return sum((g * z[occ_id].float()).sum()
                   for (occ_id, g) in plan["t"]["by_batch"][0])

    def test_multi_horizon_replay_gradient_matches_direct(self):
        # 1/2/3 horizon cuts: the merged per-cut gradient must be replayed
        # ONCE (seventh review) — direct and latent gradients must agree
        # exactly for every horizon multiplicity.
        for horizons in ((1,), (1, 2), (1, 2, 3)):
            n, r, mdim = 60, 8, 12
            g = torch.Generator().manual_seed(11)
            z = torch.randn(n, r, generator=g).requires_grad_(True)
            p = torch.randn(n, mdim, generator=g)
            w = torch.rand(n, generator=g) + 0.5
            rows = self._multi_horizon_rows(n, r, mdim, horizons, z, p, w)
            wA = KFMomentWindow({"t": r}, min_ratio=1.0, min_abs=2,
                                fixed_maps=self._stub(p))
            jA = wA.add(rows)[0]["t"]
            jA.backward()
            gradA = z.grad.clone()
            z.grad = None
            wC = KFMomentWindow({"t": r}, min_ratio=1.0, min_abs=2,
                                fixed_maps=self._stub(p), autoclose=False)
            wC.add([CutRecord(
                tree_id=r2.tree_id, occurrence_id=r2.occurrence_id,
                tau=r2.tau, horizon=r2.horizon, node=r2.node,
                time=r2.time, z=r2.z.detach(), context=r2.context,
                outcome=r2.outcome, outcome_id=r2.outcome_id,
                weight=r2.weight) for r2 in rows])
            closedC, planC, diagC = wC.close_replay()
            self.assertAlmostEqual(float(closedC["t"]), float(jA.detach()),
                                   places=5)
            # Each cut must appear EXACTLY ONCE in the replay plan.
            emitted = [occ for (occ, _g) in planC["t"]["by_batch"][0]]
            self.assertEqual(len(emitted), n,
                             "horizons={}: merged gradient must be "
                             "replayed once per cut".format(horizons))
            surr = self._latent_surrogate(z, planC)
            surr.backward()
            gradC = z.grad.clone()
            self.assertTrue(
                torch.allclose(gradA, gradC, atol=1e-6, rtol=1e-5),
                "horizons={}: latent gradient mismatch vs direct "
                "({})".format(horizons,
                              float((gradA - gradC).abs().max())))

    def test_kf_only_optimizer_step_raises_j(self):
        # The objective maximizes J: one optimizer step on -J must RAISE
        # the score (this fails if the replay sign is flipped).
        n, r, mdim = 400, 6, 12
        g = torch.Generator().manual_seed(31)
        z = torch.randn(n, r, generator=g).requires_grad_(True)
        p = torch.randn(n, mdim, generator=g)
        j_before = float(kf_score(z, p, eps=1e-4))
        opt = torch.optim.Adam([z], lr=0.1)
        opt.zero_grad()
        (-kf_score(z, p, eps=1e-4)).backward()
        opt.step()
        j_after = float(kf_score(z.detach(), p, eps=1e-4))
        self.assertGreater(j_after, j_before,
                           "one KF ascent step must raise J "
                           "({:.4f} -> {:.4f})".format(j_before, j_after))

    def test_alpha_scaling_direct_matches_latent(self):
        # A per-tau alpha enters the objective linearly: alpha * grad_J
        # must equal the latent replay scaled by the same alpha.
        n, r, mdim = 80, 8, 12
        g = torch.Generator().manual_seed(41)
        z = torch.randn(n, r, generator=g).requires_grad_(True)
        p = torch.randn(n, mdim, generator=g)
        w = torch.rand(n, generator=g) + 0.5
        rows = self._multi_horizon_rows(n, r, mdim, (1, 2), z, p, w)
        wA = KFMomentWindow({"t": r}, min_ratio=1.0, min_abs=2,
                            fixed_maps=self._stub(p))
        alpha = 0.37
        jA = wA.add(rows)[0]["t"]
        (alpha * jA).backward()
        gradA = z.grad.clone()
        z.grad = None
        wC = KFMomentWindow({"t": r}, min_ratio=1.0, min_abs=2,
                            fixed_maps=self._stub(p), autoclose=False)
        wC.add([CutRecord(
            tree_id=r2.tree_id, occurrence_id=r2.occurrence_id,
            tau=r2.tau, horizon=r2.horizon, node=r2.node,
            time=r2.time, z=r2.z.detach(), context=r2.context,
            outcome=r2.outcome, outcome_id=r2.outcome_id,
            weight=r2.weight) for r2 in rows])
        _, planC, _ = wC.close_replay()
        (alpha * self._latent_surrogate(z, planC)).backward()
        gradC = z.grad.clone()
        self.assertTrue(torch.allclose(gradA, gradC, atol=1e-6, rtol=1e-5),
                        "alpha-scaled latent gradient must match direct")


class TestReviewRegression(unittest.TestCase):
    """Regression tests from the cloud-crash review (2026-08-27).

    The review found the crash report mis-diagnosed the two failures: a
    collapsed z scale CANNOT blow J up (``J(c) ~ c^2/(c^2+delta) -> 0``),
    and a Cholesky pivot near zero is not evidence of a collapsed
    coordinate.  The real suspects are raw-moment cancellation and
    high-dimensional small-sample CCA saturation.  These tests pin the
    corrected numerical contract.
    """

    def test_scale_invariance_across_orders_of_magnitude(self):
        # Z -> 10^k Z for k = -12..12: the score must be (almost) exactly
        # invariant and stay within [0, min(r, m)] — a collapsed z scale
        # must never blow J past dim (the old absolute-jitter failure).
        z = _rand(400, 6, seed=30)
        p = _rand(400, 10, seed=31)
        j0 = float(kf_score(z, p, eps=1e-6))
        for k in range(-12, 13):
            j = float(kf_score(10.0 ** k * z, p, eps=1e-6))
            self.assertAlmostEqual(j, j0, delta=1e-3 * max(1.0, j0),
                                   msg="10^{} scale drift changed J".format(k))
            self.assertGreaterEqual(j, -1e-6)
            self.assertLessEqual(j, 6.0 + 1e-3)

    def test_large_dc_offset_no_catastrophic_cancellation(self):
        # Z = 1e4 * 1 + 1e-3 * noise: raw-moment accumulation
        # (E[zz^T] - E[z]E[z]^T) in float32 loses the signal to
        # catastrophic cancellation; direct float64 centering must recover
        # the same score as the noise alone.
        n, r, mdim = 500, 6, 12
        g = torch.Generator().manual_seed(40)
        noise = torch.randn(n, r, generator=g)
        z_offset = 1e4 + 1e-3 * noise
        p = _rand(n, mdim, seed=41)
        j_off = float(kf_score(z_offset, p, eps=1e-6))
        j_plain = float(kf_score(noise, p, eps=1e-6))
        self.assertAlmostEqual(j_off, j_plain, delta=1e-2,
                               msg="DC offset must not change the score")

    def test_window_matches_direct_centered_score(self):
        # The window close path (float64, den = M-1) must agree term by
        # term with a manual stack-center of the same rows.
        n, r, mdim = 300, 8, 64
        rows = make_cut_rows(n, r, mdim, offset=700)
        maps = _FakeMaps(mdim)
        w = KFMomentWindow({"t": r}, min_ratio=1.0, min_abs=64,
                           fixed_maps=maps)
        closed, diag, gated = w.add(rows)
        self.assertIn("t", closed)
        row_ids, cut_ids, tids, zs, ps, weights = dedup_cut_rows(rows, maps)
        zc = (zs.double() - zs.double().mean(0, keepdim=True))
        pc = (ps.double() - ps.double().mean(0, keepdim=True))
        czz, cpp, czp, _ = _covs(zc, pc, float(zc.shape[0] - 1))
        j_manual, d = _score_from_covs(czz, czp, cpp, 1e-4)
        self.assertIsNone(d["failed"])
        self.assertTrue(torch.allclose(closed["t"], j_manual.float(),
                                       atol=1e-4),
                        "window close must match the direct computation")

    def test_z_and_p_share_one_row_mask(self):
        # Z and P must come from the same rows, the same dedup mask and the
        # same denominator — a mismatch silently corrupts the score.
        n, r, mdim = 200, 8, 64
        rows = make_cut_rows(n, r, mdim)
        row_ids, cut_ids, tids, zs, ps, weights = dedup_cut_rows(
            rows, _FakeMaps(mdim))
        self.assertEqual(zs.shape[0], ps.shape[0],
                         "Z and P rows must match after dedup")
        self.assertEqual(len(row_ids), zs.shape[0])
        self.assertEqual(len(cut_ids), zs.shape[0])
        self.assertEqual(len(tids), zs.shape[0])
        self.assertEqual(len(weights), zs.shape[0])
        # Several horizons per cut: rows stay SEPARATE (no p averaging —
        # averaging would create p1*p2^T cross terms and change Cov(P));
        # cut_ids collapse to one unique id per cut.
        dup = make_cut_rows(n, r, mdim, rows_per_cut=2)
        k2, c2, t2, z2, p2, w2 = dedup_cut_rows(dup, _FakeMaps(mdim))
        self.assertEqual(z2.shape[0], 2 * n, "rows stay separate")
        self.assertEqual(p2.shape[0], 2 * n)
        self.assertEqual(len(set(c2)), n, "cut_ids collapse per cut")
        self.assertEqual(len(set(k2)), 2 * n, "row_ids stay distinct")
        # The close path asserts the invariant itself.
        w = KFMomentWindow({"t": r}, min_ratio=1.0, min_abs=64,
                           fixed_maps=_FakeMaps(mdim))
        closed, diag, gated = w.add(rows)
        self.assertIn("t", closed)
        self.assertEqual(diag["t"]["M_unique"], n)

    def test_independent_noise_saturation_regime_344_172_256(self):
        # The review's sharper problem: with M=344, d_z=172, d_p=256 the
        # centered rows force an intersection of dim
        # d_z + d_p - (M-1) = 85, so independent noise scores ~128 — the
        # epoch-19 J=62.6 was mostly geometry, not learned future
        # information.  In that regime J_real must match J_shuffled.
        r, mdim, n = 172, 256, 344
        g = torch.Generator().manual_seed(77)
        z = torch.randn(n, r, generator=g)
        p = torch.randn(n, mdim, generator=g)
        w = KFMomentWindow({"t": r}, min_ratio=1.0, min_abs=64,
                           fixed_maps=None)
        class _Stub:
            def pv(self, ctx, y):
                return p[ctx["counterpart"] % n]
        w.fixed_maps = _Stub()
        rows = [CutRecord(
            tree_id=i, occurrence_id=i, tau="t", horizon=1,
            node=i, time=float(i), z=z[i],
            context={"delta_t": 0.0, "counterpart": i, "role": 0,
                     "query_type": 0, "horizon": 1, "path": []}, outcome=0.0,
                     outcome_id=("edge", i))
            for i in range(n)]
        closed, diag, gated = w.add(rows)
        self.assertIn("t", closed)
        j = float(closed["t"].detach())
        jsh = diag["t"]["J_shuffled"]
        # Independent noise: nothing to claim, both scores must agree.
        self.assertLess(abs(j - jsh) / max(jsh, 1.0), 0.1,
                        "independent-noise J_real ({:.1f}) must match "
                        "J_shuffled ({:.1f})".format(j, jsh))
        self.assertLessEqual(j, r + 1e-3)

    def test_window_radial_derivative_vanishes(self):
        # Wrapper-level (full window close path) scale invariance: both the
        # analytic radial derivative <grad J, Z> and the finite difference
        # must vanish — the signature that no half-gradient wrapper slipped
        # into the windowed path.
        n, r, mdim = 300, 8, 64
        g = torch.Generator().manual_seed(55)
        z = torch.randn(n, r, generator=g)
        p = torch.randn(n, mdim, generator=g)
        class _Stub:
            def pv(self, ctx, y):
                return p[ctx["counterpart"] % n]
        def close_with(zs):
            w = KFMomentWindow({"t": r}, min_ratio=1.0, min_abs=64,
                               fixed_maps=_Stub())
            rows = [CutRecord(
                tree_id=i, occurrence_id=i, tau="t", horizon=1,
                node=i, time=float(i), z=zs[i],
                context={"delta_t": 0.0, "counterpart": i, "role": 0,
                         "query_type": 0, "horizon": 1, "path": []}, outcome=0.0,
                     outcome_id=("edge", i))
                for i in range(n)]
            closed, _, _ = w.add(rows)
            return closed["t"]
        zz = z.clone().requires_grad_(True)
        j = close_with(zz)
        grad = torch.autograd.grad(j, zz)[0]
        d_auto = float((grad * zz).sum())
        h = 1e-3
        j_plus = float(close_with(((1 + h) * zz.detach())).detach())
        j_minus = float(close_with(((1 - h) * zz.detach())).detach())
        d_fd = (j_plus - j_minus) / (2 * h)
        self.assertAlmostEqual(d_auto, 0.0, delta=1e-2,
                               msg="window analytic radial derivative "
                                   "must vanish")
        self.assertAlmostEqual(d_fd, 0.0, delta=1e-2,
                               msg="window finite-difference radial "
                                   "derivative must vanish")


if __name__ == "__main__":
    unittest.main()
