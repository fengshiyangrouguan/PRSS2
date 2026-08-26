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
            local_features=torch.zeros(1),
            children=list(children),
            child_relations=[0] * len(children),
            child_delta_t=[0.0] * len(children),
            metadata={"layer": int(tau.split(":")[1][len("layer"):]),
                      "node": node, "time": time,
                      "own_raw": torch.randn(4)}))
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

    def test_query_batch_matches_query_per_row(self):
        nodes = np.array([1, 1, 2, 3, 4, 6, 99, 2], dtype=np.int64)
        times = np.array([1.0, 6.0, 2.0, 9.0, 12.0, 26.0, 1.0, 10.0])
        found, valid, ht, hd, hy = self.idx.query_batch(nodes, times)
        for i, (u, t) in enumerate(zip(nodes, times)):
            hit = self.idx.query(int(u), float(t))
            self.assertEqual(bool(found[i]), hit is not None,
                             "found mismatch at row {}".format(i))
            if found[i]:
                self.assertEqual(bool(valid[i]), hit["valid"])
                if valid[i]:
                    self.assertEqual(ht[i], hit["time"])
                    self.assertEqual(hd[i], hit["counterpart"])
                    self.assertEqual(hy[i], hit["outcome"])


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

    def test_stage1_positive_plus_negatives(self):
        trace = make_trace(as_of_times=(4.0, 9.0))
        rows = JodieCutBuilder(self.idx, stage=LINK, neg_per_cut=3,
                               seed=7).build(trace, batch_seed=0)
        # Four cuts x (1 pos + 3 neg).
        self.assertEqual(len(rows), 4 * 4)
        self.assertEqual(sum(1 for r in rows if r.outcome == 1.0), 4)
        self.assertEqual(sum(1 for r in rows if r.outcome == 0.0), 12)
        for r in rows:
            self.assertEqual(r.context["query_type"], 0)
        # z repeats across the rows of one cut (same tensor object).
        pos = next(r for r in rows if r.cut_id == 0 and r.outcome == 1.0)
        negs = [r for r in rows if r.cut_id == 0 and r.outcome == 0.0]
        self.assertEqual(len(negs), 3)
        for n in negs:
            self.assertTrue(torch.equal(n.z, pos.z))
            self.assertNotEqual(n.context["counterpart"], pos.context["counterpart"])

    def test_censored_cuts_never_become_y0_rows(self):
        # Node 5 from t=4 is valid (next 15.0); use a node whose next event is
        # in val: node 4 has events 11.0 (train) / 25.0 (val).  Build a trace
        # rooted at node 4 with t=12 -> censored; no rows may appear for it.
        trace = RecursiveTrace()
        trace.add(RecursiveOccurrence(
            occurrence_id=0, tau="tjo:layer1",
            state=OccurrenceState(tau="tjo:layer1", z=torch.randn(4)),
            local_features=torch.zeros(1), children=[],
            child_relations=[], child_delta_t=[],
            metadata={"layer": 1, "node": 4, "time": 12.0,
                      "own_raw": torch.randn(4)}))
        trace.roots.append(0)
        trace.root_rows.append(0)
        rows = JodieCutBuilder(self.idx, stage=NODE_CLASS, seed=0).build(
            trace, batch_seed=0)
        self.assertEqual(rows, [])

    def test_forbidden_fields_absent_from_context(self):
        trace = make_trace(as_of_times=(4.0,))
        rows = JodieCutBuilder(self.idx, stage=LINK, neg_per_cut=1,
                               seed=1).build(trace, batch_seed=0)
        for r in rows:
            for bad in ("outcome", "y", "label", "valid", "is_positive"):
                self.assertNotIn(bad, r.context)
            self.assertIn("delta_t", r.context)
            self.assertIn("counterpart", r.context)
            self.assertIn("role", r.context)
            self.assertIn("query_type", r.context)

    def test_negatives_deterministic_per_batch_seed(self):
        trace = make_trace(as_of_times=(4.0, 9.0))
        b = JodieCutBuilder(self.idx, stage=LINK, neg_per_cut=2, seed=3)
        a1 = [(r.context["counterpart"], r.outcome) for r in b.build(trace, batch_seed=11)]
        a2 = [(r.context["counterpart"], r.outcome) for r in b.build(trace, batch_seed=11)]
        b1 = [(r.context["counterpart"], r.outcome) for r in b.build(trace, batch_seed=12)]
        self.assertEqual(a1, a2)
        self.assertNotEqual(a1, b1)


if __name__ == "__main__":
    unittest.main()
