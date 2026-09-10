"""Train-only link future index + boundary records (plan step 3, gate B part).

Verifies on synthetic streams that:

* Y(n,t) = first real train event of n strictly after t (both roles);
* a real consumed pair gives Y_v from the child and Y_p(v) from the PARENT
  occurrence (no child-second-event proxy);
* parent future == re-query of the exact parent occurrence's first strict
  future;
* pairs with no legal child or parent future are dropped (shared set);
* all futures are strictly later than the corresponding cut time.
"""

import unittest
from types import SimpleNamespace

import numpy as np
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rpbe.link_records import (BoundaryRecord, LinkFutureIndex,
                               build_boundary_records)


def _stream():
    # node events: node 1 at t=1,4,7 ; node 2 at t=2,5 ; node 3 at t=3,6,9
    srcs = np.array([1, 2, 3, 1, 2, 3, 1, 3], dtype=np.int64)
    dsts = np.array([9, 1, 9, 9, 9, 1, 9, 9], dtype=np.int64)
    tms = np.array([1., 2., 3., 4., 5., 6., 7., 9.], dtype=np.float64)
    eis = np.arange(1, len(srcs) + 1, dtype=np.int64)
    return srcs, dsts, tms, eis


class _Pair:
    def __init__(self, child_node, child_time, parent_node, parent_time,
                 pid=(0, 0, 1, 0), child_layer=1, parent_layer=2):
        self.pair_id = pid
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


class TestLinkFutureIndex(unittest.TestCase):
    def test_first_strict_future_per_node(self):
        srcs, dsts, tms, eis = _stream()
        idx = LinkFutureIndex(srcs, dsts, tms, eis)
        # node 1 events: roles 0 at t=1,4,7 (as src) — also role1 when dst
        # at t=... check node 1: appears src at t=1,4,7; dst of node2@2, node3@6
        # -> incident times for node1: 1(src),2(dst),4(src),6(dst),7(src)
        ev = idx.query(1, 1.0)
        self.assertIsNotNone(ev)
        self.assertGreater(ev.time, 1.0)
        self.assertEqual(ev.time, 2.0)   # first strictly after t=1
        # exact equality is not a future
        ev2 = idx.query(1, 2.0)
        self.assertGreater(ev2.time, 2.0)
        # node with no later event
        self.assertIsNone(idx.query(3, 9.0))

    def test_both_roles_see_event(self):
        srcs, dsts, tms, eis = _stream()
        idx = LinkFutureIndex(srcs, dsts, tms, eis)
        # the edge (3,1) at t=6: node 1's future and node 3's future both see
        # it as the next incident event when queried just before 6.
        ev1 = idx.query(1, 5.9)
        ev3 = idx.query(3, 5.9)
        self.assertEqual(ev1.time, 6.0)
        self.assertEqual(ev3.time, 6.0)
        # roles differ by endpoint
        # node1 is dst at t=6 -> role 1, counterpart src=3
        self.assertEqual(ev1.role, 1)
        self.assertEqual(ev1.counterpart, 3)
        # node3 is src at t=6 -> role 0
        self.assertEqual(ev3.role, 0)

    def test_message_idx_is_edge_id(self):
        srcs, dsts, tms, eis = _stream()
        idx = LinkFutureIndex(srcs, dsts, tms, eis)
        ev = idx.query(2, 2.0)
        self.assertEqual(ev.message_idx, ev.event_id)


class TestBoundaryRecords(unittest.TestCase):
    def test_yv_and_y_pv_from_child_and_parent(self):
        srcs, dsts, tms, eis = _stream()
        idx = LinkFutureIndex(srcs, dsts, tms, eis)
        # child occurrence node 3 @ t=3 ; parent node 1 @ t=4
        pair = _Pair(child_node=3, child_time=3.0,
                     parent_node=1, parent_time=4.0)
        recs = build_boundary_records([pair], idx)
        self.assertEqual(len(recs), 1)
        r = recs[0]
        self.assertEqual(r.child_future.time, 6.0)   # node3 next after 3
        self.assertEqual(r.parent_future.time, 6.0)  # node1 next after 4 is 6
        # Y_v^(2) := Y_p(v)^(1) -> parent future is the second-observation
        self.assertTrue(r.valid)

    def test_no_legal_future_drops_pair(self):
        srcs, dsts, tms, eis = _stream()
        idx = LinkFutureIndex(srcs, dsts, tms, eis)
        pair = _Pair(child_node=3, child_time=9.0,
                     parent_node=1, parent_time=10.0)   # no futures
        recs = build_boundary_records([pair], idx)
        self.assertEqual(recs, [])

    def test_parent_future_is_exact_parent_requery(self):
        srcs, dsts, tms, eis = _stream()
        idx = LinkFutureIndex(srcs, dsts, tms, eis)
        pair = _Pair(child_node=1, child_time=1.0,
                     parent_node=2, parent_time=2.0)
        recs = build_boundary_records([pair], idx)
        r = recs[0]
        direct = idx.query(pair.parent_node, pair.parent_time)
        self.assertEqual(r.parent_future.event_id, direct.event_id)
        self.assertEqual(r.parent_future.time, direct.time)


if __name__ == "__main__":
    unittest.main()
