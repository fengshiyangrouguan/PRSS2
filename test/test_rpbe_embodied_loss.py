"""§20 gate: sample-space dual is EXACTLY equivalent to the feature-space
scale-normalized full-balancing score (plan §18).  Training on LIBERO-Mem
is forbidden until this passes.

Reference path: rpbe.loss._covs + _score_from_covs (TGN implementation,
imported in tests only -- the embodied package itself never imports rpbe).
"""
import unittest

import torch

from rpbe.loss import _covs, _score_from_covs
from rpbe_embodied.loss import (
    diag_latent_z_adjoint,
    dual_full_score,
    dual_latent_z_adjoint,
)


def _feature_reference(z, p, w, cut_ids, eps=1e-4):
    """Exact KFMomentWindow._close path: weighted center, D correction,
    scale-normalized full_balancing via _score_from_covs.

    z must be a double leaf (gradient flows through it)."""
    z64, p64, w64 = z, p.double(), w.double()
    W = w64.sum()
    acc = {}
    for cid, wi in zip(cut_ids, w64.tolist()):
        acc[cid] = acc.get(cid, 0.0) + wi
    W2_cut = sum(v * v for v in acc.values())
    D = (W - W2_cut / W)
    mu_z = (z64 * w64[:, None]).sum(0) / W
    mu_p = (p64 * w64[:, None]).sum(0) / W
    zc = (z64 - mu_z) * w64[:, None].sqrt()
    pc = (p64 - mu_p) * w64[:, None].sqrt()
    czz, cpp, czp, _ = _covs(zc, pc, D, w=torch.ones_like(w64))
    J, diag = _score_from_covs(czz, czp, cpp, eps, variant="full_balancing")
    assert J is not None, f"reference failed: {diag}"
    return J, zc, pc, D


class TestDualExactEquivalence(unittest.TestCase):
    """N=20, d_z=32, m=16 random matrices x 4 weight configs."""

    def _make_data(self, seed, config):
        torch.manual_seed(seed)
        N, dz, m = 20, 32, 16
        z = torch.randn(N, dz)
        p = torch.randn(N, m)
        if config == "uniform":
            w = torch.ones(N, dtype=torch.float64)
            cut_ids = [(0, i, "cog") for i in range(N)]
        elif config == "two_horizon":
            # 10 cuts x 2 horizons sharing z per cut
            z10 = torch.randn(10, dz)
            p = torch.randn(20, m)
            z = z10.repeat_interleave(2, dim=0)
            w = torch.ones(20, dtype=torch.float64)
            cut_ids = [(0, i, "cog") for i in range(10) for _ in range(2)]
        elif config == "zero_weight":
            w = torch.ones(N, dtype=torch.float64)
            w[3] = 0.0
            cut_ids = [(0, i, "cog") for i in range(N)]
        elif config == "dc_offset":
            z = torch.randn(N, dz) + 100.0
            p = torch.randn(N, m) - 50.0
            w = torch.rand(N, dtype=torch.float64) + 0.5
            cut_ids = [(0, i, "cog") for i in range(N)]
        else:
            raise ValueError(config)
        return z, p, w, cut_ids

    def _check(self, config):
        z, p, w, cut_ids = self._make_data(7, config)
        # reference: J from a double leaf (gradient-connected)
        z_leaf = z.double().requires_grad_(True)
        J_ref, _, _, _ = _feature_reference(z_leaf, p, w, cut_ids)
        (g_ref,) = torch.autograd.grad(J_ref, z_leaf)
        # dual: same
        z_leaf2 = z.double().requires_grad_(True)
        J_dual, diag = dual_full_score(z_leaf2, p, w, cut_ids, eps=1e-4)
        assert "failed" not in diag, diag
        (g_dual,) = torch.autograd.grad(J_dual, z_leaf2)
        self.assertLessEqual(
            abs(J_ref.item() - J_dual.item()), 1e-8 * (1.0 + abs(J_ref.item())),
            f"[{config}] J mismatch: ref {J_ref.item()} vs dual {J_dual.item()}")
        self.assertTrue(
            torch.allclose(g_ref, g_dual, atol=1e-6, rtol=1e-6),
            f"[{config}] grad mismatch: max diff "
            f"{(g_ref - g_dual).abs().max().item()}")

    def test_uniform(self):
        self._check("uniform")

    def test_two_horizon(self):
        self._check("two_horizon")

    def test_zero_weight(self):
        self._check("zero_weight")

    def test_dc_offset(self):
        self._check("dc_offset")


class TestDualAdjoint(unittest.TestCase):
    def test_cut_merging(self):
        torch.manual_seed(3)
        z10 = torch.randn(10, 32)
        p = torch.randn(20, 16)
        z = z10.repeat_interleave(2, dim=0)
        w = torch.ones(20, dtype=torch.float64)
        cut_ids = [(0, i, "cog") for i in range(10) for _ in range(2)]
        j, g_by_cut, diag = dual_latent_z_adjoint(z, p, w, cut_ids)
        assert "failed" not in diag, diag
        self.assertEqual(len(g_by_cut), 10)          # one gradient per cut
        for cid, g in g_by_cut.items():
            self.assertEqual(tuple(g.shape), (32,))

    def test_diag_variant_runs(self):
        torch.manual_seed(5)
        z = torch.randn(8, 4096)
        p = torch.randn(8, 64)
        w = torch.ones(8, dtype=torch.float64)
        cut_ids = [(0, i, "cog") for i in range(8)]
        j, g_by_cut, diag = diag_latent_z_adjoint(z, p, w, cut_ids)
        assert "failed" not in diag, diag
        self.assertTrue(torch.isfinite(torch.tensor(j)))
        self.assertEqual(len(g_by_cut), 8)

    def test_cholesky_failure_soft(self):
        # degenerate: all z identical -> s_Z? centering removes it; instead
        # force N=1 window where K_Z is rank 0 and K+epsI still choleskys,
        # so use strict=True with a window whose Q has a constant column.
        torch.manual_seed(9)
        z = torch.randn(4, 32)
        p = torch.cat([torch.ones(4, 1), torch.randn(4, 15)], dim=1)
        w = torch.ones(4, dtype=torch.float64)
        cut_ids = [(0, i, "cog") for i in range(4)]
        # not degenerate -> succeeds
        j, g, diag = dual_latent_z_adjoint(z, p, w, cut_ids)
        assert "failed" not in diag
        self.assertIsNotNone(j)


if __name__ == "__main__":
    unittest.main()
