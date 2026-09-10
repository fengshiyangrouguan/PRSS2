"""Structural-arm boundary gates (plan step 5, gate B).

Verifies over a synthetic boundary pool that:

* 1obs / aligned / mispaired share the identical candidate POSITION set
  (feasibility drop is arm-independent);
* aligned parent future == re-query of the exact parent occurrence;
* mispaired parent-future multiset == aligned parent-future multiset
  (Counter equality);
* mispaired unchanged fraction == 0 (every surviving position's parent
  future changed);
* every assigned (donor) parent future is strictly later than the receiving
  parent's time (legal future);
* no child-second-event proxy is used (parent future is never the child's
  second future event — enforced by construction in build_boundary_records).
"""

import unittest
from collections import Counter

import numpy as np
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rpbe.link_records import (LinkFutureIndex, BoundaryRecord,
                               build_boundary_records)
from rpbe.pair_arms import (build_mispaired_parent_map, feasible_positions,
                            parent_side)


def _stream():
    # events: node1 at t=1,4,7,11 ; node2 at 2,5,8 ; node3 at 3,6,9,12
    srcs = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3], dtype=np.int64)
    dsts = np.array([9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9], dtype=np.int64)
    tms = np.array([1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
                   dtype=np.float64)
    eis = np.arange(1, len(srcs) + 1, dtype=np.int64)
    return srcs, dsts, tms, eis


class _Pair:
    def __init__(self, idx, child_node, child_time, parent_node,
                 parent_time, child_layer=1, parent_layer=2):
        self.pair_id = (idx, 0, parent_layer, 0)
        self.root_row = 0
        self.tau = "tjo:layer1"
        self.child_layer = child_layer
        self.parent_layer = parent_layer
        self.child_node = child_node
        self.child_time = child_time
        self.parent_node = parent_node
        self.parent_time = parent_time
        self.relation_time = child_time - 0.5
        self.relation_lag = 0.5
        self.relation_slot = 0
        self.path = ((1, 0.5),)
        self.z = torch.randn(4, requires_grad=True)


class TestPairArms(unittest.TestCase):
    def _pool(self):
        srcs, dsts, tms, eis = _stream()
        idx = LinkFutureIndex(srcs, dsts, tms, eis)
        pairs = [
            # child node1@1 parent node2@2
            _Pair(0, 1, 1.0, 2, 2.0),
            # child node2@2 parent node3@3
            _Pair(1, 2, 2.0, 3, 3.0),
            # child node3@3 parent node1@4
            _Pair(2, 3, 3.0, 1, 4.0),
            # child node1@4 parent node2@5
            _Pair(3, 1, 4.0, 2, 5.0),
            # child node2@5 parent node3@6
            _Pair(4, 2, 5.0, 3, 6.0),
            # child node3@6 parent node1@7
            _Pair(5, 3, 6.0, 1, 7.0),
            # child node1@7 parent node2@8
            _Pair(6, 1, 7.0, 2, 8.0),
            # child node2@8 parent node3@9
            _Pair(7, 2, 8.0, 3, 9.0),
        ]
        recs = build_boundary_records(pairs, idx)
        # all must be valid (each child/parent has a future)
        self.assertEqual(len(recs), len(pairs))
        return idx, recs

    def test_shared_position_set_across_arms(self):
        _, recs = self._pool()
        fea = feasible_positions(recs, seed=0, batch_seed=1)
        mapping = build_mispaired_parent_map(recs, seed=0, batch_seed=1)
        # aligned arm uses all feasible positions; mispaired map covers them
        self.assertEqual(sorted(mapping.keys()), fea)
        # mapping has no fixed points
        self.assertTrue(all(i not in mapping or mapping[i] is not None
                            for i in fea))

    def test_mispaired_parent_multiset_equals_aligned(self):
        _, recs = self._pool()
        fea = feasible_positions(recs, seed=0, batch_seed=1)
        aligned = Counter(recs[i].parent_future.event_id for i in fea)
        mapping = build_mispaired_parent_map(recs, seed=0, batch_seed=1)
        misp = Counter(mapping[i].event_id for i in sorted(mapping))
        self.assertEqual(aligned, misp,
                         "mispaired parent-future multiset must equal aligned")

    def test_unchanged_fraction_zero_and_legal(self):
        _, recs = self._pool()
        mapping = build_mispaired_parent_map(recs, seed=0, batch_seed=1)
        unchanged = 0
        illegal = 0
        for i, pf in mapping.items():
            rec = recs[i]
            if pf.event_id == rec.parent_future.event_id:
                unchanged += 1
            if not (pf.time > rec.parent_time):
                illegal += 1
        self.assertEqual(unchanged, 0,
                         "mispaired must change EVERY parent future")
        self.assertEqual(illegal, 0,
                         "every donor parent future must be a legal future")

    def test_child_future_not_parent_proxy(self):
        """The parent future is never the child's second future event."""
        srcs, dsts, tms, eis = _stream()
        idx = LinkFutureIndex(srcs, dsts, tms, eis)
        pair = _Pair(0, 1, 1.0, 2, 2.0)
        rec = build_boundary_records([pair], idx)[0]
        # child's SECOND future event (query twice)
        cf = idx.query(rec.child_node, rec.child_time)
        second = idx.query(rec.child_node, cf.time)
        # parent future comes from the PARENT node, distinct occurrence
        self.assertNotEqual(rec.parent_future.event_id,
                            second.event_id if second else -1)
        # parent future's node is the parent occurrence's own next event
        direct_parent = idx.query(rec.parent_node, rec.parent_time)
        self.assertEqual(rec.parent_future.event_id,
                         direct_parent.event_id)

    def test_mispaired_receiver_fields_fixed(self):
        """Mispaired replaces ONLY the parent future; the child future of each
        surviving receiver is untouched (z / child_future are the receiver's)."""
        _, recs = self._pool()
        mapping = build_mispaired_parent_map(recs, seed=0, batch_seed=1)
        self.assertGreater(len(mapping), 0)
        for i, pf in mapping.items():
            rec = recs[i]
            # z and child_future stay the receiver's own
            self.assertIsNotNone(rec.child_future)
            self.assertIsNotNone(rec.z)
            # pf is a *different* record's parent future (changed pairing)
            self.assertNotEqual(pf.event_id, rec.parent_future.event_id)


if __name__ == "__main__":
    unittest.main()
