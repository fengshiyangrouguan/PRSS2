"""PendingMergeQueue maturation rules (review ruling 1: strict causal
cut; both futures required) + EmbodiedFixedMaps fixedness contract."""
import unittest

import torch

from rpbe_embodied.config import EmbodiedRPBConfig
from rpbe_embodied.maps import EmbodiedFixedMaps
from rpbe_embodied.records import MergeRecord, PendingMergeQueue


def _mk_merge(eid, mid, tau, d=4096):
    return MergeRecord(
        episode_id=eid, merge_id=mid, merge_decision_time=tau,
        left_state=torch.randn(d), right_state=torch.randn(d),
        merged_state=torch.randn(d), left_id=mid * 2, right_id=mid * 2 + 1,
        node_id=1000 + mid, depth=1, start_step=tau, end_step=tau,
        param_version=0)


class TestPendingMergeQueue(unittest.TestCase):
    def test_strict_future_and_maturation(self):
        q = PendingMergeQueue()
        m = _mk_merge(0, 0, tau=5)
        q.register(m)
        # futures at 6 and 7 -> maturation deferred to drain
        self.assertEqual(q.offer(0, 6, {"h": 1}, torch.randn(112)), [])
        self.assertEqual(q.offer(0, 7, {"h": 2}, torch.randn(112)), [])
        rows = q.drain_episode(0, n_merges=1)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0].cut_id, rows[1].cut_id)
        self.assertEqual([r.horizon for r in rows], [1, 2])
        self.assertIs(rows[0].z, m.merged_state)   # shared detached state

    def test_causal_skip(self):
        """Rows at or before tau are not futures: skipped silently."""
        q = PendingMergeQueue()
        m = _mk_merge(0, 0, tau=5)
        q.register(m)
        self.assertEqual(q.offer(0, 4, {}, torch.randn(112)), [])   # before
        self.assertEqual(q.offer(0, 5, {}, torch.randn(112)), [])   # equal
        rows = q.drain_episode(0, n_merges=1)                       # no future
        self.assertEqual(rows, [])
        self.assertEqual(q.n_censored, 1)

    def test_censored_on_missing_future(self):
        q = PendingMergeQueue()
        q.register(_mk_merge(0, 0, tau=10))
        q.offer(0, 11, {}, torch.randn(112))        # only Y1
        rows = q.drain_episode(0, n_merges=1)       # episode ends
        self.assertEqual(rows, [])
        self.assertEqual(q.n_censored, 1)

    def test_drain_weights(self):
        q = PendingMergeQueue()
        q.register(_mk_merge(0, 0, tau=3))
        q.offer(0, 4, {}, torch.randn(112))
        q.offer(0, 5, {}, torch.randn(112))
        rows = q.drain_episode(0, n_merges=4)       # tree_weight = 1/4
        self.assertEqual(len(rows), 2)
        for r in rows:
            self.assertAlmostEqual(r.weight, 0.25 * 0.5, places=6)


class TestEmbodiedFixedMaps(unittest.TestCase):
    def _cfg(self):
        return EmbodiedRPBConfig(rpbe_seed=0)

    def test_deterministic_and_fixed(self):
        cfg = self._cfg()
        m1 = EmbodiedFixedMaps(cfg)
        m2 = EmbodiedFixedMaps(cfg)
        ctx = {"vision_feat": torch.randn(2176), "instruction": "lift bowl 3 times",
               "delta_s": 1.0, "horizon": 1}
        y = torch.randn(112)
        p1, p2 = m1.pv(ctx, y), m2.pv(ctx, y)
        self.assertTrue(torch.equal(p1, p2))
        self.assertEqual(m1.fingerprint(), m2.fingerprint())
        self.assertEqual(tuple(p1.shape), (cfg.m,))

    def test_pv_batch_matches_rowwise(self):
        m = EmbodiedFixedMaps(self._cfg())
        ctxs = [{"vision_feat": torch.randn(2176),
                 "instruction": f"task {i}", "delta_s": float(i), "horizon": 1 + i % 2}
                for i in range(5)]
        ys = torch.randn(5, 112)
        pb = m.pv_batch(ctxs, ys)
        for i in range(5):
            self.assertTrue(torch.equal(pb[i], m.pv(ctxs[i], ys[i])))

    def test_fixedness_against_perturbation(self):
        """P must not depend on any trainable state: maps module itself has
        no parameters, and pv runs under no_grad."""
        m = EmbodiedFixedMaps(self._cfg())
        self.assertEqual(sum(p.numel() for p in m.parameters()), 0)
        ctx = {"vision_feat": torch.randn(2176), "instruction": "x",
               "delta_s": 2.0, "horizon": 2}
        y = torch.randn(112)
        p1 = m.pv(ctx, y).clone()
        p2 = m.pv(ctx, y).clone()
        self.assertTrue(torch.equal(p1, p2))


if __name__ == "__main__":
    unittest.main()
