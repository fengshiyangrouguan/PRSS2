"""Ky Fan score math contracts (pure torch, synthetic data).

Verifies the identities the theory rests on: the profiled-MSE closed form,
the Ky Fan supremum, whitening invariance, the [0, min(r,m)] bound and the
gradient isolation of the fixed measurement side.
"""

import unittest

import torch

from rpbe.loss import KyFanTracker, kf_score, kf_score_fixed
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


def make_cut_rows(n_cuts, r, m, rows_per_cut=1):
    rows = []
    for c in range(n_cuts):
        z = torch.randn(r)
        for k in range(rows_per_cut):
            rows.append(CutRecord(
                tree_id=0, cut_id=c, occurrence_id=c, tau="t",
                node=c, time=float(c), z=z,
                context={"delta_t": float(c * 7 + k), "counterpart": c + k,
                         "role": 0, "query_type": 0},
                outcome=float(k == 0)))
    return rows


class TestKyFanTracker(unittest.TestCase):
    def test_gate_blocks_small_sample_saturation(self):
        # The audit counterexample: r=32 with 16 unique cuts per batch would
        # saturate on independent noise (J -> min(r, M-1)); the gate
        # (min_ratio*r = 64) must hold the score back.
        tr = KyFanTracker({"t": 32}, min_ratio=2.0, min_abs=64,
                          fixed_maps=_FakeMaps(256))
        scores, skipped = tr.update(make_cut_rows(16, 32, 256))
        self.assertEqual(scores, {})
        self.assertIn("t", skipped)

    def test_score_appears_once_enough_unique_cuts_accumulate(self):
        tr = KyFanTracker({"t": 8}, ema_rho=0.2, min_ratio=2.0, min_abs=64,
                          fixed_maps=_FakeMaps(64))
        scores = {}
        skipped = []
        for _ in range(6):
            s, sk = tr.update(make_cut_rows(60, 8, 64))
            scores.update(s)
            skipped.append(sk)
        self.assertIn("t", scores)
        n_eff = tr.effective_n("t")
        self.assertGreaterEqual(n_eff, 64)

    def test_independent_noise_does_not_saturate(self):
        # With M_unique >> r and independent P, J stays far below its
        # saturation bound min(r, M-1) — the small-sample false maximum is
        # gone by construction (gated) and the accumulated score is honest.
        tr = KyFanTracker({"t": 8}, ema_rho=0.1, min_ratio=2.0, min_abs=64,
                          fixed_maps=_FakeMaps(64))
        scores = {}
        for _ in range(8):
            s, _ = tr.update(make_cut_rows(50, 8, 64))
            scores.update(s)
        j = float(scores["t"].detach())
        n_eff = tr.effective_n("t")
        bound = min(8, max(1, int(n_eff) - 1))
        self.assertLess(j, 0.5 * bound,
                        "independent noise score {} too close to the "
                        "saturation bound {}".format(j, bound))

    def test_dedup_weights_unique_cuts(self):
        # 5 rows per cut (1 pos + 4 neg) must count as ONE unique sample:
        # the effective n accumulates per unique cut.
        tr = KyFanTracker({"t": 4}, min_ratio=1.0, min_abs=2,
                          fixed_maps=_FakeMaps(16))
        tr.update(make_cut_rows(20, 4, 16, rows_per_cut=5))
        self.assertAlmostEqual(tr.effective_n("t"), 20.0, places=1)


if __name__ == "__main__":
    unittest.main()
