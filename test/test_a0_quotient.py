"""A0 conditional-moment quotient: numeric contracts + the two theory
counterexamples (doc 4.3 / 11.1).

Pure-torch throughout (no numpy bridge needed), so this file runs on the
local Windows box as well as the GPU box.

The mod-8 test pins the central claim of the A0 R computation: a quotient
supervised by MULTI-step future conditional moments separates carry states
that one-step supervision provably merges; the XOR test pins the
conditional-vs-marginal claim.
"""

import unittest

import torch

from prss.a0.audit import (closure_residual, evaluate_gates, path_gain_report,
                           prediction_residuals, relative_residual)
from prss.a0.operators import OperatorRidge, chi_sigma, chi_width
from prss.a0.quotient import A0Quotient, randomized_svd


def _seeded(seed=0):
    return torch.Generator().manual_seed(seed)


# ---------------------------------------------------------------- numeric


class TestA0QuotientNumeric(unittest.TestCase):
    def test_accumulate_batched_equals_manual(self):
        gen = _seeded(0)
        x = torch.randn(20, 5, generator=gen)
        u = torch.randn(20, 6, generator=gen)
        batched = A0Quotient("t", p=5, m=6)
        batched.accumulate(x, u)
        manual = A0Quotient("t", p=5, m=6)
        for i in range(x.shape[0]):
            manual.accumulate(x[i:i + 1], u[i:i + 1])
        self.assertEqual(batched.n, manual.n)
        self.assertTrue(torch.allclose(batched.c_ux, manual.c_ux, atol=1e-12))
        self.assertTrue(torch.allclose(batched.c_xx, manual.c_xx, atol=1e-12))
        self.assertTrue(torch.allclose(batched.s_x, manual.s_x, atol=1e-12))
        self.assertTrue(torch.allclose(batched.s_u, manual.s_u, atol=1e-12))

    def test_centering_matches_manual(self):
        gen = _seeded(1)
        x = torch.randn(100, 8, generator=gen)
        u = torch.randn(100, 10, generator=gen)
        q = A0Quotient("t", p=8, m=10)
        q.accumulate(x, u)
        cxx, cux = q.centered_moments()
        xc = x.double() - x.double().mean(dim=0, keepdim=True)
        uc = u.double() - u.double().mean(dim=0, keepdim=True)
        self.assertTrue(torch.allclose(cxx, xc.T @ xc / x.shape[0], atol=1e-10))
        self.assertTrue(torch.allclose(cux, uc.T @ xc / x.shape[0], atol=1e-10))

    def test_solve_shapes_and_finite(self):
        gen = _seeded(2)
        x = torch.randn(200, 8, generator=gen)
        u = torch.randn(200, 10, generator=gen)
        q = A0Quotient("t", p=8, m=10)
        q.accumulate(x, u)
        tail = q.solve(rank_r=3, lambda_x=1e-4)
        self.assertEqual(tuple(q.r_matrix.shape), (3, 8))
        self.assertEqual(tuple(q.sigma.shape), (3,))
        self.assertTrue(q.solved)
        self.assertTrue(torch.isfinite(q.r_matrix).all())
        self.assertTrue(torch.isfinite(torch.as_tensor(tail)))

    def test_whitened_orthonormality(self):
        """R (C_xx + λI) Rᵀ must be the identity: Vh W (C+λI) W Vhᵀ = I."""
        gen = _seeded(3)
        x = torch.randn(300, 8, generator=gen)
        u = torch.randn(300, 10, generator=gen)
        q = A0Quotient("t", p=8, m=10)
        q.accumulate(x, u)
        q.solve(rank_r=4, lambda_x=1e-4)
        cxx, _ = q.centered_moments()
        eye = torch.eye(8, dtype=torch.float64)
        r = q.r_matrix
        gram = r @ (cxx + 1e-4 * eye) @ r.T
        self.assertTrue(torch.allclose(gram, torch.eye(4, dtype=torch.float64),
                                       atol=1e-6))

    def test_rank_tail_matches_manual(self):
        gen = _seeded(4)
        x = torch.randn(500, 8, generator=gen)
        u = torch.randn(500, 10, generator=gen)
        q = A0Quotient("t", p=8, m=10)
        q.accumulate(x, u)
        tail = q.solve(rank_r=3, lambda_x=1e-6)
        cxx, cux = q.centered_moments()
        eye = torch.eye(8, dtype=torch.float64)
        vals, vecs = torch.linalg.eigh(cxx + 1e-6 * eye)
        w = (vecs * (1.0 / vals.clamp_min(1e-12).sqrt())) @ vecs.T
        s = torch.linalg.svdvals(cux @ w)
        manual = float((s[3:] ** 2).sum() / (s ** 2).sum())
        self.assertAlmostEqual(tail, manual, places=6)

    def test_randomized_svd_matches_dense(self):
        gen = _seeded(5)
        # A gapped spectrum (predictive signal + small noise), not pure noise:
        # randomized SVD is a range finder, its guarantee needs a spectral gap.
        u0, _ = torch.linalg.qr(torch.randn(30, 30, generator=gen))
        v0, _ = torch.linalg.qr(torch.randn(300, 300, generator=gen))
        s = 1.0 / torch.arange(1, 31, dtype=torch.float64)
        m = (u0.double() * s) @ v0[:, :30].double().T + 1e-3 * torch.randn(
            30, 300, generator=gen, dtype=torch.float64)
        u_r, s_r, _ = randomized_svd(m, rank=10, n_iter=3)
        s_dense = torch.linalg.svdvals(m)[:10]
        rel = ((s_r - s_dense).abs() / s_dense.clamp_min(1e-12)).max()
        self.assertLess(float(rel), 1e-3)
        # Orthonormal left factor.
        self.assertTrue(torch.allclose(u_r.T @ u_r,
                                       torch.eye(10, dtype=torch.float64),
                                       atol=1e-5))

    def test_project_requires_solve(self):
        q = A0Quotient("t", p=4, m=6)
        with self.assertRaises(RuntimeError):
            q.project(torch.randn(2, 4))

    def test_accumulate_after_solve_raises(self):
        q = A0Quotient("t", p=4, m=6)
        gen = _seeded(6)
        q.accumulate(torch.randn(50, 4, generator=gen),
                     torch.randn(50, 6, generator=gen))
        q.solve(rank_r=2)
        with self.assertRaises(RuntimeError):
            q.accumulate(torch.randn(10, 4), torch.randn(10, 6))


# ---------------------------------------------------------- mod-8 carry tree


def _mod8_stream(n, p_lift, horizon, gen, one_step=False):
    """Synthetic cut-tree stream for the mod-8 carry counterexample.

    History h = state s in Z_8; context c = the next ``horizon`` increments
    o_i ~ Bernoulli(0.5); outcome Y = g((s + Σo) mod 8) with g = 1{· >= 4}.

    ``one_step=True`` supervises the one-step future Y' = g((s + o_1) mod 8)
    with the one-step context probe — the "one-step supervision" baseline the
    theory says must merge states 0/1/2.
    """
    s = torch.randint(0, 8, (n,), generator=gen)
    os_ = torch.randint(0, 2, (n, horizon), generator=gen)
    if one_step:
        y = (((s + os_[:, 0]) % 8) >= 4).float()
        a = torch.nn.functional.one_hot(os_[:, 0], 2).float()
    else:
        y = (((s + os_.sum(dim=1)) % 8) >= 4).float()
        a = torch.cat([torch.nn.functional.one_hot(os_[:, i], 2).float()
                       for i in range(horizon)], dim=-1)
    # History lift: fixed random injective map of the state one-hot.
    lift_gen = torch.Generator().manual_seed(99)
    p_state = torch.randn(8, p_lift, generator=lift_gen)
    x = torch.nn.functional.one_hot(s, 8).float() @ p_state
    return x, a, y, s


def _pairwise_dists(z0, z1, z2, sigma=None):
    """Predictive-metric distances (theory doc 5.5 eq. 8): ‖Σ(z−z')‖ equals
    the difference of the predicted conditional moments (U is orthonormal),
    so separability is judged by prediction behavior, not bare coordinates —
    a rank-r truncation may split coordinates without splitting predictions."""
    if sigma is not None:
        z0, z1, z2 = (sigma * z0), (sigma * z1), (sigma * z2)
    d01 = float((z0 - z1).norm())
    d02 = float((z0 - z2).norm())
    d12 = float((z1 - z2).norm())
    return d01, d02, d12


class TestMod8Counterexample(unittest.TestCase):
    """A0 multi-step conditional moments separate carry states; one-step
    supervision merges them (doc 4.3)."""

    def setUp(self):
        self.p_lift = 12
        self.n = 20000

    def _state_embeddings(self, q):
        lift_gen = torch.Generator().manual_seed(99)
        p_state = torch.randn(8, self.p_lift, generator=lift_gen)
        x = torch.nn.functional.one_hot(
            torch.arange(8), 8).float() @ p_state
        return q.project(x), q.sigma

    def test_multistep_separates_carry_states(self):
        gen = _seeded(7)
        x, a, y, s = _mod8_stream(self.n, self.p_lift, horizon=3, gen=gen)
        phi = torch.stack([1 - y, y], dim=-1)
        u = torch.cat([a * phi[:, :1], a * phi[:, 1:]], dim=-1)
        q = A0Quotient("t", p=self.p_lift, m=u.shape[-1])
        q.accumulate(x, u)
        q.solve(rank_r=3)
        z, sigma = self._state_embeddings(q)
        z0, z1, z2 = z[0], z[1], z[2]
        d01, d02, d12 = _pairwise_dists(z0, z1, z2, sigma)
        self.assertGreater(min(d01, d02, d12), 0.05)
        # Nearest-centroid state classification on fresh samples: perfect.
        test_gen = _seeded(8)
        x_t, _, _, s_t = _mod8_stream(2000, self.p_lift, horizon=3,
                                      gen=test_gen)
        z_t = q.project(x_t)
        centroids = sigma * z[:8]
        pred = torch.cdist(sigma * z_t, centroids).argmin(dim=-1)
        self.assertEqual(float((pred == s_t).float().mean()), 1.0)

    def test_onestep_baseline_collapses_carry_states(self):
        """One-step future Y' = g(s+o_1) is identical for s in {0,1,2}, so the
        conditional moment cannot separate them — the merged-state failure the
        multi-step supervision fixes.  Judged in the predictive metric (eq. 8):
        rank-r coordinates may split on bare L2 without splitting predictions."""
        gen = _seeded(9)
        x, a, y, s = _mod8_stream(self.n, self.p_lift, horizon=3, gen=gen,
                                  one_step=True)
        phi = torch.stack([1 - y, y], dim=-1)
        u = torch.cat([a * phi[:, :1], a * phi[:, 1:]], dim=-1)
        q = A0Quotient("t", p=self.p_lift, m=u.shape[-1])
        q.accumulate(x, u)
        q.solve(rank_r=3)
        z, sigma = self._state_embeddings(q)
        d01, d02, d12 = _pairwise_dists(z[0], z[1], z[2], sigma)
        self.assertLess(max(d01, d02, d12), 0.05)


# --------------------------------------------------------- XOR / context switch


class TestXorContextSwitch(unittest.TestCase):
    """Y = X xor C: conditional moments keep X, marginal moments drop it
    (doc 4.3 second counterexample)."""

    def setUp(self):
        self.n = 8000
        self.p_lift = 8

    def _stream(self, gen, with_context):
        x_v = torch.randint(0, 2, (self.n,), generator=gen)
        c_v = torch.randint(0, 2, (self.n,), generator=gen)
        y = (x_v != c_v).float()
        lift_gen = torch.Generator().manual_seed(123)
        p_x = torch.randn(2, self.p_lift, generator=lift_gen)
        x = torch.nn.functional.one_hot(x_v, 2).float() @ p_x
        if with_context:
            a = torch.nn.functional.one_hot(c_v, 2).float()
        else:
            a = torch.ones(self.n, 1)
        return x, a, y, x_v

    def _solve(self, x, a, y, r=1):
        phi = torch.stack([1 - y, y], dim=-1)
        u = torch.cat([a * phi[:, :1], a * phi[:, 1:]], dim=-1)
        q = A0Quotient("t", p=self.p_lift, m=u.shape[-1])
        q.accumulate(x, u)
        q.solve(rank_r=r)
        lift_gen = torch.Generator().manual_seed(123)
        p_x = torch.randn(2, self.p_lift, generator=lift_gen)
        x_both = torch.nn.functional.one_hot(torch.arange(2), 2).float() @ p_x
        z = q.project(x_both)
        # Predictive-metric distance (theory doc 5.5 eq. 8).
        return float((q.sigma * (z[0] - z[1])).norm()), q

    def test_with_context_probe_separates(self):
        gen = _seeded(10)
        x, a, y, _ = self._stream(gen, with_context=True)
        dist, _ = self._solve(x, a, y)
        self.assertGreater(dist, 0.05)

    def test_without_context_collapses(self):
        gen = _seeded(11)
        x, a, y, _ = self._stream(gen, with_context=False)
        dist, _ = self._solve(x, a, y)
        self.assertLess(dist, 0.01)


# ------------------------------------------------- recursive operators (phase B)


class TestOperatorRidge(unittest.TestCase):
    def test_chi_layout(self):
        gen = _seeded(20)
        r, d_c = 4, 6
        z_s = torch.randn(r, generator=gen)
        z_n = torch.randn(r, generator=gen)
        a = torch.randn(d_c, generator=gen)
        chi = chi_sigma(z_s, z_n, a)
        self.assertEqual(chi.shape[-1], chi_width(r, d_c))
        self.assertAlmostEqual(float(chi[0]), 1.0)
        self.assertTrue(torch.allclose(chi[1:1 + r], z_s))
        self.assertTrue(torch.allclose(chi[1 + r:1 + 2 * r], z_n))
        self.assertTrue(torch.allclose(chi[1 + 2 * r:1 + 3 * r], z_s * z_n))

    def test_ridge_recovers_known_linear_operator(self):
        gen = _seeded(21)
        r, s = 3, 5
        b_true = torch.randn(r, s, generator=gen)
        op = OperatorRidge("tjo:layer0", "tjo:layer1", s=s, r=r)
        n = 5000
        phi = torch.randn(n, s, generator=gen)
        z_rich = phi @ b_true.T
        op.accumulate(phi, z_rich)
        op.solve(lambda_gamma=1e-6)
        # Ridge recovers the exact linear map up to the shrinkage.
        pred = op.predict(phi[:100])
        target = phi[:100].double() @ b_true.double().T
        rel = float(((pred - target).abs().max() / target.abs().max()))
        self.assertLess(rel, 1e-3)
        self.assertTrue(op.condition_number >= 1.0)

    def test_gain_is_source_block_spectral_norm(self):
        gen = _seeded(22)
        r, s = 3, 1 + 3 * 3 + 2
        b_true = torch.randn(r, s, generator=gen)
        op = OperatorRidge("tjo:layer0", "tjo:layer1", s=s, r=r)
        op.accumulate(torch.randn(500, s, generator=gen),
                      torch.randn(500, r, generator=gen))
        op.solve()
        manual = float(torch.linalg.norm(op.b_matrix[:, 1:1 + r], ord=2))
        self.assertAlmostEqual(op.gain(), manual, places=5)


# ----------------------------------------------------------------- audit (C)


class TestAudit(unittest.TestCase):
    def test_relative_residual_perfect_fit_is_zero(self):
        gen = _seeded(23)
        x = torch.randn(300, 5, generator=gen)
        u = x @ torch.randn(5, 4, generator=gen)  # exact linear fit
        self.assertLess(relative_residual(x, u, 1e-6), 1e-6)

    def test_prediction_gap_nonnegative(self):
        gen = _seeded(24)
        x = torch.randn(500, 10, generator=gen)
        u = torch.randn(500, 8, generator=gen)
        q = A0Quotient("t", p=10, m=8)
        q.accumulate(x, u)
        q.solve(rank_r=3)
        z = q.project(x)
        res = prediction_residuals(u, x, z, lambda_audit=1e-3)
        self.assertGreaterEqual(res["prediction_gap"], 0.0)
        self.assertGreaterEqual(res["rank_r_ridge_residual"],
                                res["unrestricted_ridge_residual"] - 1e-9)

    def test_closure_residual_tracks_error(self):
        gen = _seeded(25)
        r = 3
        z_rich = torch.randn(200, r, generator=gen)
        sigma = torch.tensor([2.0, 1.0, 0.5], dtype=torch.float64)
        small = closure_residual(z_rich, z_rich, sigma)
        self.assertLess(small["closure_residual"], 1e-12)
        large = closure_residual(z_rich, z_rich + 1.0, sigma)
        self.assertGreater(large["closure_residual"], 1.0)

    def test_gates_direction(self):
        audit = {"ess": 100.0, "rank_tail_max": 0.2,
                 "closure_residual_max": 0.05,
                 "path_gain_product": 0.7, "auc_delta": 0.01}
        gates = {"G0": 50.0, "G1": 0.5, "G2": 0.1, "G3": 1.0, "G4": 0.0}
        out = evaluate_gates(audit, gates, mode="stop")
        self.assertTrue(out["gates_passed"])
        gates["G1"] = 0.1  # rank tail 0.2 > 0.1: compressibility fails
        out = evaluate_gates(audit, gates, mode="stop")
        self.assertFalse(out["gates_passed"])
        self.assertEqual(out["failed_gates"], ["G1"])
        # Unset thresholds are not enforced.
        out = evaluate_gates(audit, {"G1": None}, mode="stop")
        self.assertEqual(out["gate_results"], {})

    def test_path_gain_report(self):
        gen = _seeded(26)
        r, s = 2, 1 + 3 * 2 + 3
        ops = []
        for parent in ("tjo:layer1", "tjo:layer2"):
            op = OperatorRidge("tjo:layer0", parent, s=s, r=r)
            op.accumulate(torch.randn(300, s, generator=gen),
                          torch.randn(300, r, generator=gen))
            op.solve()
            ops.append(op)
        report = path_gain_report(ops)
        self.assertEqual(set(report["gain_by_parent_layer"]), {1, 2})
        self.assertGreater(report["path_gain_product"], 0.0)


if __name__ == "__main__":
    unittest.main()
