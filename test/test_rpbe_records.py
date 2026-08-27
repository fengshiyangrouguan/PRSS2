"""JodieCutBuilder same-tree consumption contracts on a synthetic trace.

Pure numpy + torch, no host model: a four-layer tree is built by hand with
explicit ``metadata["consumption"]`` records, so the shifted-horizon walk,
the alignment mask, the SELF-step skipping, the four identities and the
by-cut cap are all checked in isolation.
"""

import unittest

import torch

from rpbe.records import (JodieCutBuilder, LINK, NODE_CLASS, build_edge_tables)
from rpbe.state import OccurrenceState, RecursiveOccurrence, RecursiveTrace


def make_occurrence(oid, tau, node, children, relations, deltas, consumption):
    return RecursiveOccurrence(
        occurrence_id=oid, tau=tau,
        state=OccurrenceState(tau=tau, z=torch.randn(4)),
        children=list(children),
        child_relations=list(relations),
        child_delta_t=list(deltas),
        metadata={"layer": int(tau.split(":")[1][len("layer"):]),
                  "node": node, "time": 100.0,
                  "consumption": consumption})


def make_trace(n_trees=1, mutate=None):
    """Four-layer tree(s): leaf(layer0,n10) -> mid1(layer1,n20) ->
    mid2(layer2,n30) -> root(layer3,n40), all at as-of time 100.

    Consumption edges carry DIFFERENT labels so the shifted-horizon
    off-by-one is pinned: leaf's edge idx 11 (label 0.0), mid1's 22
    (1.0), mid2's 33 (0.0); the root task record label is 1.0.
    """
    trace = RecursiveTrace()
    for t in range(n_trees):
        base = t * 10
        leaf_cons = {"kind": "edge", "edge_idx": 11 + base, "edge_time": 90.0,
                     "endpoint_role": 1, "label_owner": 10 + base,
                     "counterpart": 20 + base}
        mid1_cons = {"kind": "edge", "edge_idx": 22 + base, "edge_time": 80.0,
                     "endpoint_role": 1, "label_owner": 20 + base,
                     "counterpart": 30 + base}
        mid2_cons = {"kind": "edge", "edge_idx": 33 + base, "edge_time": 70.0,
                     "endpoint_role": 1, "label_owner": 30 + base,
                     "counterpart": 40 + base}
        if mutate is not None:
            leaf_cons, mid1_cons, mid2_cons = mutate(
                t, leaf_cons, mid1_cons, mid2_cons)
        oid_leaf = base + 0
        oid_mid1 = base + 1
        oid_mid2 = base + 2
        oid_root = base + 3
        trace.add(make_occurrence(
            oid_leaf, "tjo:layer0", 10 + base, [], [], [], leaf_cons))
        trace.add(make_occurrence(
            oid_mid1, "tjo:layer1", 20 + base, [oid_leaf], [1], [10.0],
            mid1_cons))
        trace.add(make_occurrence(
            oid_mid2, "tjo:layer2", 30 + base, [oid_mid1], [1], [20.0],
            mid2_cons))
        trace.add(make_occurrence(
            oid_root, "tjo:layer3", 40 + base, [oid_mid2], [1], [30.0],
            {"kind": "self"}))
        trace.roots.append(oid_root)
        trace.root_rows.append(t)
    return trace


def make_labels(n_trees=1):
    d = {}
    for t in range(n_trees):
        base = t * 10
        d[11 + base] = 0.0   # leaf's own consumption edge
        d[22 + base] = 1.0   # mid1's edge
        d[33 + base] = 0.0   # mid2's edge
    return d


def make_root_events(n_trees=1):
    return {t: {"dst": 50 + t, "label": 1.0, "time": 100.0,
                "event_idx": 999 + t} for t in range(n_trees)}


def build_rows(n_trees=1, stage=NODE_CLASS, cuts_per_tau=32, seed=0,
               mutate=None, root_events=None, labels=None):
    trace = make_trace(n_trees=n_trees, mutate=mutate)
    if labels is None:
        labels = make_labels(n_trees=n_trees)
    if root_events is None:
        root_events = make_root_events(n_trees=n_trees)
    b = JodieCutBuilder(({}, labels), stage=stage, cuts_per_tau=cuts_per_tau,
                        seed=seed)
    return trace, b, b.build(trace, root_events=root_events, batch_seed=0)


def rows_of_cut(rows, oid, tau):
    return [r for r in rows if r.occurrence_id == oid and r.tau == tau]


class TestShiftedHorizonWalk(unittest.TestCase):
    def test_h1_reads_parent_consumption_not_own(self):
        # Pin the off-by-one: leaf's h1 Y == mid1's edge label (1.0), NOT
        # the leaf's own edge (0.0).  h2 == mid2's edge (0.0).
        _, _, rows = build_rows()
        cut = sorted(rows_of_cut(rows, 0, "tjo:layer0"),
                     key=lambda r: r.horizon)
        self.assertEqual(len(cut), 2)
        self.assertEqual(cut[0].horizon, 1)
        self.assertEqual(cut[0].outcome, 1.0, "leaf h1 must read mid1's edge")
        self.assertEqual(cut[0].outcome_id, ("edge", 22))
        self.assertEqual(cut[1].horizon, 2)
        self.assertEqual(cut[1].outcome, 0.0, "leaf h2 must read mid2's edge")
        self.assertEqual(cut[1].outcome_id, ("edge", 33))

    def test_per_layer_row_matrix(self):
        # layer0: 2 rows (h1 mid1 edge, h2 mid2 edge); layer1: 2 rows
        # (h1 mid2 edge, h2 root record); layer2: 1 row (h1 root record);
        # layer3 (root): no upward walk -> 0 rows.  Total 5.
        _, _, rows = build_rows()
        self.assertEqual(len(rows_of_cut(rows, 0, "tjo:layer0")), 2)
        self.assertEqual(len(rows_of_cut(rows, 1, "tjo:layer1")), 2)
        self.assertEqual(len(rows_of_cut(rows, 2, "tjo:layer2")), 1)
        self.assertEqual(len(rows_of_cut(rows, 3, "tjo:layer3")), 0)
        self.assertEqual(len(rows), 5)

    def test_root_record_reached_by_layer2_h1(self):
        _, _, rows = build_rows()
        cut = rows_of_cut(rows, 2, "tjo:layer2")
        self.assertEqual(len(cut), 1)
        r = cut[0]
        self.assertEqual(r.horizon, 1)
        self.assertEqual(r.outcome, 1.0)          # root task label
        self.assertEqual(r.outcome_id, ("root", 999))
        self.assertEqual(r.context["counterpart"], 50)  # dst
        self.assertEqual(r.context["delta_t"], 0.0)     # t_root - t_root
        self.assertEqual(r.context["path"], [(1, 30.0)])

    def test_layer1_h2_reaches_root(self):
        _, _, rows = build_rows()
        cut = sorted(rows_of_cut(rows, 1, "tjo:layer1"),
                     key=lambda r: r.horizon)
        self.assertEqual(cut[1].horizon, 2)
        self.assertEqual(cut[1].outcome_id, ("root", 999))
        self.assertEqual(cut[1].context["path"], [(1, 20.0), (1, 30.0)])


class TestAlignmentAndSelfSkip(unittest.TestCase):
    def test_unaligned_edge_probe_skipped_upward(self):
        # mid1's label belongs to node 99, not to mid1's node (20): the
        # probe is unusable; the walk skips to mid2's edge.  The skipped
        # step stays in the path.
        def mutate(t, l, m1, m2):
            m1["label_owner"] = 99
            return l, m1, m2
        _, _, rows = build_rows(mutate=mutate)
        cut = sorted(rows_of_cut(rows, 0, "tjo:layer0"),
                     key=lambda r: r.horizon)
        self.assertEqual(len(cut), 2)
        self.assertEqual(cut[0].outcome_id, ("edge", 33),
                         "unaligned mid1 must be skipped")
        self.assertEqual(cut[0].context["path"], [(1, 10.0), (1, 20.0)])
        self.assertEqual(cut[1].outcome_id, ("root", 999))

    def test_self_step_skipped_upward(self):
        # mid1 has no interaction of its own (SELF step): same skip.
        def mutate(t, l, m1, m2):
            m1["kind"] = "self"
            return l, m1, m2
        _, _, rows = build_rows(mutate=mutate)
        cut = sorted(rows_of_cut(rows, 0, "tjo:layer0"),
                     key=lambda r: r.horizon)
        self.assertEqual(len(cut), 2)
        self.assertEqual(cut[0].outcome_id, ("edge", 33))
        self.assertEqual(cut[1].outcome_id, ("root", 999))


class TestIdentitiesAndWeights(unittest.TestCase):
    def test_four_identities(self):
        _, _, rows = build_rows()
        leaf_h1 = sorted(rows_of_cut(rows, 0, "tjo:layer0"),
                         key=lambda r: r.horizon)[0]
        self.assertEqual(leaf_h1.cut_id, (0, 0, "tjo:layer0"))
        self.assertEqual(leaf_h1.row_id, (0, 0, "tjo:layer0", 1))
        self.assertEqual(leaf_h1.overlap_id, (10, 100.0, "tjo:layer0"))
        self.assertEqual(leaf_h1.outcome_id, ("edge", 22))
        # Horizons of one cut share cut_id, differ in row_id.
        leaf_h2 = sorted(rows_of_cut(rows, 0, "tjo:layer0"),
                         key=lambda r: r.horizon)[1]
        self.assertEqual(leaf_h1.cut_id, leaf_h2.cut_id)
        self.assertNotEqual(leaf_h1.row_id, leaf_h2.row_id)

    def test_horizon_weights_sum_one_per_cut(self):
        _, _, rows = build_rows()
        for cut_rows in (rows_of_cut(rows, 0, "tjo:layer0"),
                         rows_of_cut(rows, 1, "tjo:layer1"),
                         rows_of_cut(rows, 2, "tjo:layer2")):
            self.assertAlmostEqual(sum(r.weight for r in cut_rows), 1.0)
        for r in rows_of_cut(rows, 0, "tjo:layer0"):
            self.assertAlmostEqual(r.weight, 0.5)

    def test_same_node_time_across_trees_is_not_dropped(self):
        # Two identical trees: the same (node, time, tau) overlap does NOT
        # delete rows (overlap_id is a correlation grouping, never a
        # delete key) — both trees contribute their full rows.
        _, _, rows = build_rows(n_trees=2)
        self.assertEqual(len(rows), 10)
        self.assertEqual(len(rows_of_cut(rows, 0, "tjo:layer0")), 2)
        self.assertEqual(len(rows_of_cut(rows, 10, "tjo:layer0")), 2)
        # Nodes are offset per tree (10/20/30/40 vs 20/30/40/50), so the
        # overlap ids are all distinct here; with shared nodes they would
        # repeat and still keep every row (the overlap is a group key).
        overlap_ids = {r.overlap_id for r in rows}
        self.assertEqual(len(overlap_ids), 6)

    def test_shared_node_overlap_keeps_all_rows(self):
        # Same (node, time, tau) in two trees: both rows survive.
        trace = make_trace(n_trees=2, mutate=lambda t, l, m1, m2: (
            dict(l, label_owner=10, edge_idx=11, counterpart=20),
            dict(m1, label_owner=20, edge_idx=22, counterpart=30),
            dict(m2, label_owner=30, edge_idx=33, counterpart=40)))
        # Rebuild with nodes pinned across trees: overwrite metadata nodes
        # so both trees share node ids 10/20/30/40 (mutate already pinned
        # the label owners to the same ids, keeping the probes aligned).
        node_pin = {0: 10, 1: 20, 2: 30, 3: 40}
        for oid, occ in trace.occurrences.items():
            occ.metadata["node"] = node_pin[int(occ.metadata["layer"])]
        labels = {11: 0.0, 22: 1.0, 33: 0.0}
        b = JodieCutBuilder(({}, labels), stage=NODE_CLASS, seed=0)
        rows = b.build(trace, root_events=make_root_events(2), batch_seed=0)
        self.assertEqual(len(rows), 10)
        overlap_ids = {r.overlap_id for r in rows}
        self.assertEqual(len(overlap_ids), 3, "shared nodes collapse the "
                         "overlap grouping only, not the rows")

    def test_tree_ids_global_across_batches(self):
        trace = make_trace()
        b = JodieCutBuilder(({}, make_labels()), stage=NODE_CLASS, seed=0)
        r1 = b.build(trace, root_events=make_root_events(1), batch_seed=0)
        t2 = make_trace()
        r2 = b.build(t2, root_events=make_root_events(1), batch_seed=1)
        self.assertEqual({r.tree_id for r in r1}, {0})
        self.assertEqual({r.tree_id for r in r2}, {1})


class TestCapAndErrors(unittest.TestCase):
    def test_cap_samples_by_cut_keeps_all_horizons(self):
        # cuts_per_tau=1 with two trees: layer0 has 2 cuts; the sampled
        # cut keeps BOTH its horizon rows.
        _, _, rows = build_rows(n_trees=2, cuts_per_tau=1, seed=3)
        layer0 = [r for r in rows if r.tau == "tjo:layer0"]
        self.assertEqual(len(layer0), 2)
        oids = {r.occurrence_id for r in layer0}
        self.assertEqual(len(oids), 1, "cap must sample by cut")
        self.assertEqual(len({r.horizon for r in layer0}), 2,
                         "a sampled cut keeps all horizons")

    def test_parent_map_asserts_single_parent(self):
        trace = make_trace()
        # Give leaf (oid 0) a second parent: oid 1 already parents it,
        # append it to the root's children as well.
        trace.occurrences[3].children.append(0)
        trace.occurrences[3].child_relations.append(1)
        trace.occurrences[3].child_delta_t.append(5.0)
        b = JodieCutBuilder(({}, make_labels()), stage=NODE_CLASS, seed=0)
        with self.assertRaises(ValueError):
            b.build(trace, root_events=make_root_events(), batch_seed=0)

    def test_missing_root_events_removes_root_records_only(self):
        # Without root events, the walk to the root yields nothing: layer2
        # has no h1 (0 rows), but lower layers keep their edge probes.
        _, _, rows = build_rows(root_events={})
        self.assertEqual(len(rows_of_cut(rows, 2, "tjo:layer2")), 0)
        self.assertEqual(len(rows_of_cut(rows, 1, "tjo:layer1")), 1)
        self.assertEqual(len(rows_of_cut(rows, 0, "tjo:layer0")), 2)

    def test_stage_query_type(self):
        _, _, rows = build_rows(stage=LINK)
        for r in rows:
            self.assertEqual(r.context["query_type"], 0)
        _, _, rows2 = build_rows(stage=NODE_CLASS)
        for r in rows2:
            self.assertEqual(r.context["query_type"], 1)


class TestEdgeTables(unittest.TestCase):
    def test_build_edge_tables_maps_and_counts_conflicts(self):
        from types import SimpleNamespace
        import numpy as np
        data = SimpleNamespace(full=SimpleNamespace(
            edge_idxs=np.array([3, 3, 4]),
            sources=np.array([1, 5, 2]),
            destinations=np.array([9, 9, 8]),
            labels=np.array([1.0, 1.0, 0.0])))
        endpoints, labels, stats = build_edge_tables(data)
        self.assertEqual(endpoints[3], (5, 9))          # last write wins
        self.assertEqual(labels[3], 1.0)
        self.assertEqual(labels[4], 0.0)
        self.assertEqual(stats["endpoint_conflicts"], 1)
        self.assertEqual(stats["label_conflicts"], 0)


if __name__ == "__main__":
    unittest.main()
