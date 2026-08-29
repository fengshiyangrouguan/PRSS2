"""Strict-future outcome and compact-cut construction contracts."""

import unittest
from types import SimpleNamespace

import numpy as np
import torch

from rpbe.records import (JodieCutBuilder, JodieFutureIndex, LINK,
                          NODE_CLASS, build_edge_tables)
from rpbe.state import CompactCutTrace, CutCandidate


def stream():
    # Chronological interactions.  Node 1 is a source; node 10 is a
    # destination, so both role directions can be checked.
    return SimpleNamespace(
        sources=np.asarray([1, 2, 1, 3, 1, 4]),
        destinations=np.asarray([10, 10, 11, 10, 12, 10]),
        timestamps=np.asarray([1.0, 2.0, 3.0, 3.0, 5.0, 6.0]),
        edge_idxs=np.asarray([101, 102, 103, 104, 105, 106]),
        labels=np.asarray([0.0, 1.0, 1.0, 0.0, 0.0, 1.0]))


def candidate(oid=7, root_row=0, node=1, time=1.0, tau="layer1"):
    return CutCandidate(
        occurrence_id=oid, root_row=root_row, tau=tau, node=node,
        time=time, z=torch.randn(4, requires_grad=True), path=[(0, 0.0)])


class TestFutureIndex(unittest.TestCase):
    def test_first_two_are_strictly_later_real_events(self):
        index = JodieFutureIndex(stream())
        rows = index.query(1, 1.0)
        self.assertEqual([row.time for row in rows], [3.0, 5.0])
        self.assertEqual([row.outcome for row in rows], [1.0, 0.0])
        self.assertEqual([row.outcome_id for row in rows],
                         [("future", 103), ("future", 105)])
        self.assertTrue(all(row.role == 0 for row in rows))

    def test_equal_timestamp_is_not_future(self):
        index = JodieFutureIndex(stream())
        rows = index.query(10, 3.0)
        self.assertEqual([row.time for row in rows], [6.0])
        self.assertEqual(rows[0].counterpart, 4)
        self.assertEqual(rows[0].role, 1)

    def test_missing_future_is_empty_not_padded(self):
        index = JodieFutureIndex(stream())
        self.assertEqual(index.query(1, 5.0), [])
        self.assertEqual(index.query(999, 0.0), [])

    def test_index_cannot_see_an_unsupplied_split(self):
        train = stream()
        index = JodieFutureIndex(train)
        # A hypothetical validation event is not in the constructed index.
        self.assertEqual(index.query(1, 5.0), [])


class TestCutBuilder(unittest.TestCase):
    def test_two_real_future_rows_and_no_root_supervisor(self):
        trace = CompactCutTrace(root_rows=[0], cuts=[candidate()])
        builder = JodieCutBuilder(
            JodieFutureIndex(stream()), stage=NODE_CLASS, seed=0)
        rows = builder.build(trace, batch_seed=0)
        self.assertEqual(len(rows), 2)
        self.assertEqual([row.horizon for row in rows], [1, 2])
        self.assertEqual([row.outcome_time for row in rows], [3.0, 5.0])
        self.assertTrue(all(row.outcome_time > row.time for row in rows))
        self.assertEqual([row.outcome_id for row in rows],
                         [("future", 103), ("future", 105)])
        self.assertFalse(any(row.outcome_id[0] == "root" for row in rows))
        self.assertAlmostEqual(sum(row.weight for row in rows), 1.0)
        self.assertTrue(all(row.z.requires_grad for row in rows))

    def test_missing_second_future_is_masked_and_weight_renormalized(self):
        trace = CompactCutTrace(
            root_rows=[0], cuts=[candidate(node=10, time=3.0)])
        stats = {}
        rows = JodieCutBuilder(
            JodieFutureIndex(stream()), stage=LINK).build(trace, stats=stats)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].horizon, 1)
        self.assertEqual(rows[0].outcome_time, 6.0)
        self.assertEqual(rows[0].context["role"], 1)
        self.assertEqual(rows[0].context["query_type"], 0)
        self.assertAlmostEqual(rows[0].weight, 1.0)
        self.assertEqual(stats["missing_horizons"][("layer1", 2)], 1)

    def test_no_future_produces_no_rows(self):
        trace = CompactCutTrace(
            root_rows=[0], cuts=[candidate(node=1, time=5.0)])
        rows = JodieCutBuilder(
            JodieFutureIndex(stream()), stage=NODE_CLASS).build(trace)
        self.assertEqual(rows, [])

    def test_builder_reads_candidates_directly_without_tree_contract(self):
        trace = CompactCutTrace(
            root_rows=[2, 5],
            cuts=[candidate(1, 2, node=1, time=1.0, tau="layer1"),
                  candidate(2, 5, node=10, time=2.0, tau="layer2")])
        rows = JodieCutBuilder(
            JodieFutureIndex(stream()), stage=NODE_CLASS).build(trace)
        self.assertEqual({row.tree_id for row in rows}, {0, 1})
        self.assertEqual({row.tau for row in rows}, {"layer1", "layer2"})
        self.assertFalse(hasattr(trace, "occurrences"))
        self.assertFalse(hasattr(trace, "roots"))

    def test_cut_cap_samples_whole_cuts(self):
        cuts = [candidate(oid=i, root_row=i, node=1, time=1.0)
                for i in range(5)]
        trace = CompactCutTrace(root_rows=list(range(5)), cuts=cuts)
        rows = JodieCutBuilder(
            JodieFutureIndex(stream()), stage=NODE_CLASS,
            cuts_per_tau=2, seed=3).build(trace, batch_seed=4)
        self.assertEqual(len({row.cut_id for row in rows}), 2)
        counts = {}
        for row in rows:
            counts[row.cut_id] = counts.get(row.cut_id, 0) + 1
        self.assertEqual(set(counts.values()), {2})

    def test_invalid_old_edge_table_argument_is_rejected(self):
        with self.assertRaises(TypeError):
            JodieCutBuilder(({}, {}, set(), set()), stage=NODE_CLASS)

    def test_one_observation_mode_emits_only_y1(self):
        trace = CompactCutTrace(
            root_rows=[0], cuts=[candidate(7, 0, node=1, time=1.0)])
        builder = JodieCutBuilder(
            JodieFutureIndex(stream()), stage=NODE_CLASS, seed=0,
            n_observations=1)
        rows = builder.build(trace, batch_seed=0)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].horizon, 1)
        self.assertAlmostEqual(rows[0].weight, 1.0,
                               msg="per-tree total weight stays 1")



class TestEdgeTableAudit(unittest.TestCase):
    def test_maps_and_conflict_counts_are_audit_only(self):
        data = SimpleNamespace(full=stream())
        endpoints, labels, users, pages, stats = build_edge_tables(data)
        self.assertEqual(endpoints[101], (1, 10))
        self.assertEqual(labels[106], 1.0)
        self.assertIn(1, users)
        self.assertIn(10, pages)
        self.assertEqual(stats["endpoint_conflicts"], 0)
        self.assertEqual(stats["label_conflicts"], 0)


if __name__ == "__main__":
    unittest.main()
