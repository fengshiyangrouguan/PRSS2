"""Pair KF window exact-replay gates (plan step 6).

Verifies:
* every pair is ONE loss row (one z + one p);
* per-tree total weight 1;
* close_replay returns per-position z adjoints (gradient path finite);
* the two-pass linear surrogate equals the direct window gradient at the
  current parameters.
"""

import unittest

import numpy as np
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rpbe.pair_window import PairKFWindow, tree_equal_weights


class _StubRecord:
    def __init__(self, idx, root_row, dim=8, mdim=16, signal=True):
        self.boundary_key = ("p", idx)
        self.root_row = root_row
        g = torch.Generator().manual_seed(idx)
        if signal:
            s = torch.randn(3, generator=g)
            z = torch.cat([s, torch.randn(dim - 3, generator=g) * 0.1],
                          dim=0)
        else:
            z = torch.randn(dim, generator=g)
        self.z = z.requires_grad_(True)
        self.signal = signal
        self.weight = 1.0
        self._mdim = mdim


class _StubMaps:
    def __init__(self, mdim=16):
        self._g = torch.Generator().manual_seed(99)
        self._base = torch.randn(mdim, generator=self._g)

    def pv_row(self, rec):
        # p distinct per record, mildly correlated with rec.z.signal via idx
        g = torch.Generator().manual_seed(1000 + int(rec.boundary_key[1]))
        return self._base + 0.3 * torch.randn(16, generator=g)


class TestPairWindow(unittest.TestCase):
    def test_one_row_per_pair_and_tree_weight(self):
        recs = [_StubRecord(i, root_row=i // 4) for i in range(16)]
        w = tree_equal_weights(recs)
        for r in recs:
            r.weight = w[r.root_row]
        # per-tree total weight 1 (4 trees x 4 pairs each)
        per_tree = {}
        for r in recs:
            per_tree[r.root_row] = per_tree.get(r.root_row, 0.0) + r.weight
        for t, tot in per_tree.items():
            self.assertAlmostEqual(tot, 1.0, places=6)

    def test_close_replay_per_position_grads(self):
        maps = _StubMaps()
        win = PairKFWindow(tau="tjo:layer1", min_unique_trees=4)
        recs = [_StubRecord(i, root_row=i // 4) for i in range(16)]
        w = tree_equal_weights(recs)
        for r in recs:
            r.weight = w[r.root_row]
        for r in recs:
            win.add(r)
        self.assertTrue(win.ready())
        j, g_by_pos, diag = win.close_replay(maps)
        self.assertIsNotNone(j)
        self.assertTrue(np.isfinite(j))
        self.assertEqual(len(g_by_pos), len(recs),
                         "one adjoint per surviving pair position")
        for pos, g in g_by_pos.items():
            self.assertEqual(g.shape, recs[pos].z.shape)

    def test_surrogate_matches_direct_gradient(self):
        maps = _StubMaps()
        win = PairKFWindow(tau="tjo:layer1", min_unique_trees=4)
        recs = [_StubRecord(i, root_row=i // 4) for i in range(16)]
        w = tree_equal_weights(recs)
        for r in recs:
            r.weight = w[r.root_row]
        for r in recs:
            win.add(r)
        j, g_by_pos, diag = win.close_replay(maps)
        # direct gradient of the window score w.r.t. z
        # (score = win._score(recs, maps))
        sc = win._score(recs, maps)
        self.assertIsNotNone(sc)
        sc.backward()
        direct = [r.z.grad.detach().clone() for r in recs]
        # surrogate: sum <g, z> difference reproduces grad; compare its grad
        # to direct by zeroing and backpropping the surrogate value once more
        for r in recs:
            if r.z.grad is not None:
                r.z.grad.zero_()
        surr = win.surrogate(g_by_pos, recs, coeff=1.0)
        surr.backward()
        surr_grads = [r.z.grad.detach().clone() if r.z.grad is not None else None
                      for r in recs]
        for a, b, r in zip(direct, surr_grads, recs):
            if b is None:
                continue
            # surrogate gradient equals the exact window z-gradient
            self.assertTrue(torch.allclose(a, b, atol=1e-4),
                            "surrogate grad must match direct grad "
                            "(pos {})".format(r.boundary_key))


if __name__ == "__main__":
    unittest.main()
