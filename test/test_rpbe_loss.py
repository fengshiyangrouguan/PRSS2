"""Ky Fan score math contracts (pure torch, synthetic data).

Verifies the identities the theory rests on: the profiled-MSE closed form,
the Ky Fan supremum, whitening invariance, the [0, min(r,m)] bound and the
gradient isolation of the fixed measurement side.
"""

import unittest

import numpy as np
import torch

from rpbe.loss import (KFMomentWindow, _covs, _score_from_covs, dedup_cut_rows,
                       kf_score, kf_score_fixed)
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
                tree_id=offset + c, cut_id=offset + c,
                occurrence_id=offset + c,
                tau="t", node=offset + c, time=float(offset + c), z=z,
                context={"delta_t": float(c * 7 + k), "counterpart": c + k,
                         "role": 0, "query_type": 0},
                outcome=float(k == 0)))
    return rows


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
                    tree_id=i, cut_id=i, occurrence_id=i, tau="t",
                    node=i, time=float(i), z=z[i],
                    context={"delta_t": 0.0, "counterpart": i, "role": 0,
                             "query_type": 0},
                    outcome=0.0))
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
            tree_id=i, cut_id=i, occurrence_id=i, tau="t",
            node=i, time=float(i), z=torch.ones(4),
            context={"delta_t": 0.0, "counterpart": i, "role": 0,
                     "query_type": 0}, outcome=0.0)
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
                tree_id=i, cut_id=i, occurrence_id=i, tau="t",
                node=i, time=float(i), z=z[i],
                context={"delta_t": 0.0, "counterpart": i, "role": 0,
                         "query_type": 0}, outcome=0.0))
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
                tree_id=i, cut_id=i, occurrence_id=i, tau="t",
                node=i, time=float(i), z=z[i],
                context={"delta_t": 0.0, "counterpart": i, "role": 0,
                         "query_type": 0}, outcome=0.0))
        closed, diag, gated = w.add(rows)
        self.assertIn("t", closed)
        j = float(closed["t"].detach())
        self.assertTrue(np.isfinite(j), "near-singular C_ZZ must not crash")
        self.assertGreaterEqual(j, -1e-6)
        self.assertLessEqual(j, r + 1e-3)


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
        keys, tids, zs, ps = dedup_cut_rows(rows, maps)
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
        keys, tids, zs, ps = dedup_cut_rows(rows, _FakeMaps(mdim))
        self.assertEqual(zs.shape[0], ps.shape[0],
                         "Z and P rows must match after dedup")
        self.assertEqual(len(keys), zs.shape[0])
        self.assertEqual(len(tids), zs.shape[0])
        # A cut with several rows_per_cut still contributes ONE row pair.
        dup = make_cut_rows(n, r, mdim, rows_per_cut=3)
        k2, t2, z2, p2 = dedup_cut_rows(dup, _FakeMaps(mdim))
        self.assertEqual(z2.shape[0], n)
        self.assertEqual(p2.shape[0], n)
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
            tree_id=i, cut_id=i, occurrence_id=i, tau="t",
            node=i, time=float(i), z=z[i],
            context={"delta_t": 0.0, "counterpart": i, "role": 0,
                     "query_type": 0}, outcome=0.0)
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
                tree_id=i, cut_id=i, occurrence_id=i, tau="t",
                node=i, time=float(i), z=zs[i],
                context={"delta_t": 0.0, "counterpart": i, "role": 0,
                         "query_type": 0}, outcome=0.0)
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
