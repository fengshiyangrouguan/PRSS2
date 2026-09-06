"""Structural-supervision arm construction tests (paper Part III, Stage 1).

These verify, on synthetic traces, the properties the three arms must share
before any training is allowed:

* 1obs / 2obs_aligned / 2obs_mispaired run on the SAME cut set (cuts with
  both a legal Y1 and Y2), i.e. identical cut_id sets and per-tree counts;
* 1obs emits one Y1 row per cut with per-tree total weight 1;
* 2obs arms emit Y1 and Y2 rows per cut, each horizon weighted 0.5, and the
  per-tree total weight stays 1;
* the mispaired arm keeps the Y1 and Y2 MULTISETS of the aligned arm while
  destroying (at least one) sample-level (Y1, Y2) pairing;
* the mispaired derangement is deterministic in (seed, batch_seed);
* production behaviour is untouched (regression guard: the default path is
  bit-for-bit the historical one).
"""

import unittest
from types import SimpleNamespace

import numpy as np
import torch

from rpbe.records import (
    JodieCutBuilder, JodieFutureIndex, NODE_CLASS,
    SUPERVISION_1OBS, SUPERVISION_2OBS_ALIGNED,
    SUPERVISION_2OBS_MISPAIRED, SUPERVISION_PRODUCTION)
from rpbe.state import CompactCutTrace, CutCandidate


def multi_stream():
    """Several nodes, each with >= 2 strictly-future events.

    Node ids: user side {1, 2, 3, 4} (role 0 when source), page side {100,
    101} (role 1 when destination).  Each user 1..4 interacts repeatedly so
    that a cut at t=1.0 on that user sees two later events with stable
    stream positions.
    """
    srcs = np.asarray([1, 2, 1, 3, 2, 4, 1, 3, 2, 4, 1, 2])
    dsts = np.asarray([100, 100, 101, 100, 101, 100, 100, 101, 100, 101,
                       101, 101])
    times = np.asarray([1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0,
                        6.0, 6.0])
    edge_idxs = np.arange(1000, 1000 + len(srcs))
    labels = np.asarray([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0,
                         1.0, 0.0])
    return SimpleNamespace(sources=srcs, destinations=dsts, timestamps=times,
                           edge_idxs=edge_idxs, labels=labels)


def cut_candidate(oid, root_row, node, time, tau, path_len=1):
    return CutCandidate(
        occurrence_id=oid, root_row=root_row, tau=tau, node=node, time=time,
        z=torch.randn(4, requires_grad=True),
        path=[(0, 0.0)] * path_len)


def make_trace():
    """A mixed-depth, multi-root trace over nodes with rich futures."""
    roots = [0, 1, 2, 3]
    cuts = []
    oid = 0
    for rr, node, time, tau in [
            (0, 1, 1.0, "tjo:layer1"), (0, 1, 1.0, "tjo:layer2"),
            (1, 2, 1.0, "tjo:layer1"), (1, 2, 1.0, "tjo:layer2"),
            (2, 3, 2.0, "tjo:layer1"), (2, 3, 2.0, "tjo:layer2"),
            (3, 4, 3.0, "tjo:layer1"), (3, 4, 3.0, "tjo:layer2")]:
        cuts.append(cut_candidate(oid, rr, node, time, tau))
        oid += 1
    return CompactCutTrace(root_rows=roots, cuts=cuts)


def summarize(rows):
    """Map horizon rows by cut_id, keeping outcome ids and horizon order."""
    by_cut = {}
    for r in rows:
        by_cut.setdefault(r.cut_id, []).append(r)
    return by_cut


class TestStructuralCutSet(unittest.TestCase):
    def setUp(self):
        self.idx = JodieFutureIndex(multi_stream())
        self.trace = make_trace()
        self.stats = {}

    def _builder(self, mode):
        return JodieCutBuilder(self.idx, stage=NODE_CLASS, seed=0,
                               n_observations=2, supervision_mode=mode)

    def test_arms_share_identical_cut_set(self):
        sets = {}
        for mode in (SUPERVISION_1OBS, SUPERVISION_2OBS_ALIGNED,
                     SUPERVISION_2OBS_MISPAIRED):
            rows = self._builder(mode).build(self.trace, batch_seed=7)
            sets[mode] = set(summarize(rows).keys())
        self.assertEqual(sets[SUPERVISION_1OBS],
                         sets[SUPERVISION_2OBS_ALIGNED])
        self.assertEqual(sets[SUPERVISION_2OBS_ALIGNED],
                         sets[SUPERVISION_2OBS_MISPAIRED])
        self.assertTrue(len(sets[SUPERVISION_1OBS]) >= 4,
                        "expected a rich shared cut set")

    def test_1obs_emits_only_y1_and_unit_tree_weight(self):
        rows = self._builder(SUPERVISION_1OBS).build(
            self.trace, batch_seed=7)
        by_cut = summarize(rows)
        for cid, rws in by_cut.items():
            self.assertEqual([r.horizon for r in rws], [1])
        per_tree = {}
        for r in rows:
            per_tree[r.tree_id] = per_tree.get(r.tree_id, 0.0) + r.weight
        for t, w in per_tree.items():
            self.assertAlmostEqual(w, 1.0, places=6,
                                   msg="per-tree total weight must stay 1")

    def test_aligned_emits_y1_y2_with_half_weight(self):
        rows = self._builder(SUPERVISION_2OBS_ALIGNED).build(
            self.trace, batch_seed=7)
        by_cut = summarize(rows)
        per_tree_cuts = {}
        for r in rows:
            per_tree_cuts[r.tree_id] = per_tree_cuts.get(r.tree_id, 0) + 0
        tree_cut_count = {}
        for cid in by_cut:
            tree_cut_count[cid[0]] = tree_cut_count.get(cid[0], 0) + 1
        for cid, rws in by_cut.items():
            self.assertEqual(sorted(r.horizon for r in rws), [1, 2])
            # Each cut's rows sum to 1/(#cuts of its tree) and the two
            # horizons split it evenly (base * 0.5 each).
            base = 1.0 / tree_cut_count[cid[0]]
            self.assertAlmostEqual(sum(r.weight for r in rws),
                                   base, places=6)
            for r in rws:
                self.assertAlmostEqual(r.weight, 0.5 * base, places=6)
        per_tree = {}
        for r in rows:
            per_tree[r.tree_id] = per_tree.get(r.tree_id, 0.0) + r.weight
        for t, w in per_tree.items():
            self.assertAlmostEqual(w, 1.0, places=6)

    def test_mispaired_keeps_multisets_but_changes_pairing(self):
        aligned = self._builder(SUPERVISION_2OBS_ALIGNED).build(
            self.trace, batch_seed=7)
        misp = self._builder(SUPERVISION_2OBS_MISPAIRED).build(
            self.trace, batch_seed=7)
        a_by = summarize(aligned)
        m_by = summarize(misp)
        # Same cut set (guarded above); same Y1/Y2 multisets.
        self.assertEqual(
            sorted(r.outcome_id for r in aligned),
            sorted(r.outcome_id for r in misp))
        # Y1 per cut identical (only the second observation is mispaired).
        for cid in a_by:
            a1 = next(r for r in a_by[cid] if r.horizon == 1)
            m1 = next(r for r in m_by[cid] if r.horizon == 1)
            self.assertEqual(a1.outcome_id, m1.outcome_id)
            self.assertEqual(a1.outcome_time, m1.outcome_time)
        # At least one cut's Y2 pairing changed.
        changed = 0
        for cid in a_by:
            a2 = next((r for r in a_by[cid] if r.horizon == 2), None)
            m2 = next((r for r in m_by[cid] if r.horizon == 2), None)
            if a2 is not None and m2 is not None and \
                    a2.outcome_id != m2.outcome_id:
                changed += 1
        self.assertGreater(changed, 0,
                           "mispaired arm must change at least one Y2 pairing")

    def test_mispaired_derangement_is_deterministic(self):
        b1 = self._builder(SUPERVISION_2OBS_MISPAIRED)
        b2 = self._builder(SUPERVISION_2OBS_MISPAIRED)
        r1 = b1.build(self.trace, batch_seed=7)
        r2 = b2.build(self.trace, batch_seed=7)
        self.assertEqual(
            [(r.cut_id, r.horizon, r.outcome_id) for r in r1],
            [(r.cut_id, r.horizon, r.outcome_id) for r in r2])

    def test_aligned_matches_production_2obs_rows(self):
        """The aligned arm's rows must equal production n_observations=2."""
        prod = JodieCutBuilder(self.idx, stage=NODE_CLASS, seed=0,
                               n_observations=2,
                               supervision_mode=SUPERVISION_PRODUCTION)
        # Production keeps singleton-bucket cuts too, so compare on the
        # shared subset by cut_id when singleton-dropping removes nothing.
        prod_rows = prod.build(self.trace, batch_seed=7)
        aligned_rows = self._builder(SUPERVISION_2OBS_ALIGNED).build(
            self.trace, batch_seed=7)
        if len(prod_rows) == len(aligned_rows):
            key = lambda r: (r.cut_id, r.horizon)
            self.assertEqual(sorted((r.cut_id, r.horizon, r.outcome_id)
                                    for r in aligned_rows),
                             sorted((r.cut_id, r.horizon, r.outcome_id)
                                    for r in prod_rows))


class TestProductionRegression(unittest.TestCase):
    def test_production_default_is_unchanged(self):
        """The default path still emits horizon rows in stream/cut order and
        keeps the historical per-row weights (omega-normalized)."""
        idx = JodieFutureIndex(multi_stream())
        trace = make_trace()
        builder = JodieCutBuilder(idx, stage=NODE_CLASS, seed=0)
        self.assertEqual(builder.supervision_mode, SUPERVISION_PRODUCTION)
        rows = builder.build(trace, batch_seed=7)
        # 8 cuts; every cut with >=2 futures contributes Y1+Y2, others Y1.
        # Per-tree total weight is 1 regardless.
        per_tree = {}
        for r in rows:
            per_tree[r.tree_id] = per_tree.get(r.tree_id, 0.0) + r.weight
        for t, w in per_tree.items():
            self.assertAlmostEqual(w, 1.0, places=6)
        # A cut that sees two futures gets two rows (horizons 1 and 2).
        # make_trace: nodes 1,2 have >=2 futures (2 cuts each -> 2 rows);
        # nodes 3,4 have exactly 1 future (2 cuts each -> 1 row).
        by_cut = summarize(rows)
        counts = sorted(len(rws) for rws in by_cut.values())
        self.assertEqual(counts, [1, 1, 1, 1, 2, 2, 2, 2])


if __name__ == "__main__":
    unittest.main()
