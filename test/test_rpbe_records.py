"""FutureIndex / JodieCutBuilder contracts on a synthetic event stream.

Pure numpy + torch, no host model: traces are built by hand so the data-pipeline
semantics (censored != y=0, searchsorted, per-stage rows, forbidden fields) are
checked in isolation.
"""

import unittest

import numpy as np
import torch

from rpbe.records import FutureIndex, JodieCutBuilder, LINK, NODE_CLASS
from rpbe.state import OccurrenceState, RecursiveOccurrence, RecursiveTrace


def make_trace(root_rows=(0, 2), as_of_times=(10.0, 12.0)):
    """Two-root trace; each root has one child (layer0)."""
    trace = RecursiveTrace()
    nxt = [0]

    def add(tau, node, time, children):
        oid = nxt[0]
        nxt[0] += 1
        trace.add(RecursiveOccurrence(
            occurrence_id=oid, tau=tau,
            state=OccurrenceState(tau=tau, z=torch.randn(4)),
            children=list(children),
            child_relations=[0] * len(children),
            child_delta_t=[0.0] * len(children),
            metadata={"layer": int(tau.split(":")[1][len("layer"):]),
                      "node": node, "time": time,
}))
        return oid

    for i, (row, t) in enumerate(zip(root_rows, as_of_times)):
        child = add("tjo:layer0", node=5 + i, time=t, children=[])
        root = add("tjo:layer1", node=5 + i, time=t, children=[child])
        trace.roots.append(root)
        trace.root_rows.append(row)
    return trace


def make_index():
    # Nodes 1..6; split boundaries: val at t<=20, test at t<=30.
    sources = np.array([1, 1, 2, 2, 3, 4, 4, 5, 6, 6])
    destinations = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
    timestamps = np.array([1.0, 5.0, 3.0, 9.0, 8.0, 11.0, 25.0, 15.0, 12.0, 40.0])
    labels = np.array([1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    return FutureIndex(sources, destinations, timestamps, labels,
                       val_time=20.0, test_time=30.0)


class TestFutureIndex(unittest.TestCase):
    def setUp(self):
        self.idx = make_index()

    def test_searchsorted_first_event_strictly_after(self):
        # Node 1: events at 1.0 and 5.0.  t=1.0 -> next is 5.0 (strictly after).
        hit = self.idx.query(1, 1.0)
        self.assertTrue(hit["valid"])
        self.assertEqual(hit["time"], 5.0)
        self.assertEqual(hit["counterpart"], 11)
        self.assertEqual(hit["outcome"], 0.0)

    def test_no_future_event_returns_none(self):
        # Node 3's only event is at t=8.0; after it there is nothing.
        self.assertIsNone(self.idx.query(3, 9.0))
        # Unknown node entirely.
        self.assertIsNone(self.idx.query(99, 1.0))

    def test_censored_when_next_event_is_in_val_test(self):
        # Node 4: events at 11.0 (train) and 25.0 (val).  From t=12 the next
        # event is in val -> censored, content not readable.
        hit = self.idx.query(4, 12.0)
        self.assertFalse(hit["valid"])
        self.assertIsNone(hit["outcome"])
        self.assertIsNone(hit["counterpart"])
        # From t=26 the next event (40.0) is in test -> censored too.
        hit = self.idx.query(6, 26.0)
        self.assertFalse(hit["valid"])

    def test_dual_role_indexing(self):
        # Node 10 only appears as a destination (of node 1 at t=1): its next
        # event must be found with role=1 and counterpart = the source.
        hit = self.idx.query(10, 0.0)
        self.assertTrue(hit["valid"])
        self.assertEqual(hit["role"], 1)
        self.assertEqual(hit["counterpart"], 1)

    def test_neg_pool_is_train_region_only(self):
        # Destination 19 appears only at t=40 (test region) -> excluded.
        self.assertNotIn(19, self.idx.neg_pool.tolist())
        # Destination 11 appears at t=5 (train) -> included.
        self.assertIn(11, self.idx.neg_pool.tolist())


class TestJodieCutBuilder(unittest.TestCase):
    def setUp(self):
        self.idx = make_index()

    def test_stage2_one_row_per_cut_with_label_outcome(self):
        trace = make_trace(as_of_times=(4.0, 9.0))
        # Every occurrence is a cut: node 5 (t=4) has next event at 15.0,
        # label 0.0; node 6 (t=9) has next event at 12.0, label 1.0.
        rows = JodieCutBuilder(self.idx, stage=NODE_CLASS, seed=0).build(
            trace, batch_seed=0)
        by_cut = {r.cut_id: r for r in rows}
        self.assertEqual(len(rows), 4)  # 2 cuts per tree x 2 trees
        self.assertEqual(by_cut[0].outcome, 0.0)   # node 5 next label
        self.assertEqual(by_cut[1].outcome, 0.0)
        self.assertEqual(by_cut[2].outcome, 1.0)   # node 6 next label
        self.assertEqual(by_cut[3].outcome, 1.0)
        for r in rows:
            self.assertEqual(r.context["query_type"], 1)
            self.assertGreaterEqual(r.context["delta_t"], 0.0)
            self.assertIn("counterpart", r.context)
            self.assertTrue(r.valid)

    def test_stage1_one_real_continuation_per_cut(self):
        # Ky Fan measurement uses REAL futures only: exactly one row per
        # cut, no fabricated negatives (those belong to the link TASK loss).
        trace = make_trace(as_of_times=(4.0, 9.0))
        rows = JodieCutBuilder(self.idx, stage=LINK, seed=7).build(
            trace, batch_seed=0)
        self.assertEqual(len(rows), 4)  # 2 cuts per tree x 2 trees, 1 row each
        for r in rows:
            self.assertEqual(r.context["query_type"], 0)
            self.assertIn(r.outcome, (0.0, 1.0))  # the observed event label
            self.assertNotIn("is_positive", r.context)
        # Node 5 (t=4) next event has label 0.0; node 6 (t=9) next 1.0.
        by_cut = {r.cut_id: r for r in rows}
        self.assertEqual(by_cut[0].outcome, 0.0)
        self.assertEqual(by_cut[3].outcome, 1.0)

    def test_depth_balanced_sampling_caps_per_tau(self):
        # Root layers produce fewer real cuts than leaf layers; with a cap
        # every interface contributes at most cuts_per_tau rows.
        trace = make_trace(as_of_times=(4.0, 9.0))
        rows = JodieCutBuilder(self.idx, stage=NODE_CLASS, cuts_per_tau=1,
                               seed=3).build(trace, batch_seed=0)
        from collections import Counter
        counts = Counter(r.tau for r in rows)
        for tau, n in counts.items():
            self.assertLessEqual(n, 1)

    def test_censored_cuts_never_become_y0_rows(self):
        # Node 5 from t=4 is valid (next 15.0); use a node whose next event is
        # in val: node 4 has events 11.0 (train) / 25.0 (val).  Build a trace
        # rooted at node 4 with t=12 -> censored; no rows may appear for it.
        trace = RecursiveTrace()
        trace.add(RecursiveOccurrence(
            occurrence_id=0, tau="tjo:layer1",
            state=OccurrenceState(tau="tjo:layer1", z=torch.randn(4)),
            children=[],
            child_relations=[], child_delta_t=[],
            metadata={"layer": 1, "node": 4, "time": 12.0,
}))
        trace.roots.append(0)
        trace.root_rows.append(0)
        rows = JodieCutBuilder(self.idx, stage=NODE_CLASS, seed=0).build(
            trace, batch_seed=0)
        self.assertEqual(rows, [])

    def test_forbidden_fields_absent_from_context(self):
        trace = make_trace(as_of_times=(4.0,))
        rows = JodieCutBuilder(self.idx, stage=LINK, seed=1).build(
            trace, batch_seed=0)
        for r in rows:
            for bad in ("outcome", "y", "label", "valid", "is_positive"):
                self.assertNotIn(bad, r.context)
            self.assertIn("delta_t", r.context)
            self.assertIn("counterpart", r.context)
            self.assertIn("role", r.context)
            self.assertIn("query_type", r.context)

    def test_tree_ids_are_global_across_batches(self):
        b = JodieCutBuilder(self.idx, stage=NODE_CLASS, seed=0)
        t1 = make_trace(as_of_times=(4.0, 9.0))
        t2 = make_trace(as_of_times=(4.0, 9.0))
        r1 = b.build(t1, batch_seed=0)
        r2 = b.build(t2, batch_seed=1)
        ids1 = {r.tree_id for r in r1}
        ids2 = {r.tree_id for r in r2}
        self.assertEqual(ids1, {0, 1})
        self.assertEqual(ids2, {2, 3})

    def test_pseudo_duplicate_node_time_soft_deduped(self):
        # Two roots, same (node, time, tau) pair: NO hard dedupe — both
        # rows survive with their tree weights (the window dilutes them by
        # the context-overlap weight instead).  Each tree has exactly one
        # cut, so both rows carry the tree equal-weight 1.0.
        trace = RecursiveTrace()
        for k in range(2):
            trace.add(RecursiveOccurrence(
                occurrence_id=k, tau="tjo:layer1",
                state=OccurrenceState(tau="tjo:layer1", z=torch.randn(4)),
                children=[],
                child_relations=[], child_delta_t=[],
                metadata={"layer": 1, "node": 5, "time": 4.0,
    }))
            trace.roots.append(k)
            trace.root_rows.append(k)
        rows = JodieCutBuilder(self.idx, stage=NODE_CLASS, seed=0).build(
            trace, batch_seed=0)
        self.assertEqual(len(rows), 2)          # both duplicates kept
        self.assertEqual({(r.node, r.time) for r in rows}, {(5, 4.0)})
        for r in rows:
            self.assertEqual(r.weight, 1.0)     # one cut per tree

    def test_tree_equal_weight_and_sampling_correction(self):
        # Document §四: every tree contributes total weight 1 (each row
        # carries 1 / #cuts of its tree), and the sampling-probability
        # correction keeps the total unchanged when a tau is subsampled.
        # make_trace gives 2 trees x 2 cuts (layer0 + layer1, same node).
        trace = make_trace(as_of_times=(4.0, 9.0))
        builder = JodieCutBuilder(self.idx, stage=NODE_CLASS, seed=0)
        rows = builder.build(trace, batch_seed=0)
        self.assertEqual(len(rows), 4)
        for r in rows:
            self.assertAlmostEqual(r.weight, 0.5, places=9)  # 1/2 cuts
        self.assertAlmostEqual(sum(r.weight for r in rows), 2.0,
                               places=6)  # = number of trees
        # With a sampling cap each tau is subsampled with correction
        # w_sample = total/sampled: the total weight is preserved.
        rows2 = JodieCutBuilder(self.idx, stage=NODE_CLASS, seed=3,
                                cuts_per_tau=1).build(trace, batch_seed=0)
        self.assertAlmostEqual(sum(r.weight for r in rows2), 2.0,
                               places=6,
                               msg="sampling correction must preserve "
                                   "the total tree weight")


if __name__ == "__main__":
    unittest.main()
