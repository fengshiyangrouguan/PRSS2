"""Compact cut rows paired with *strictly future* observed outcomes.

The supervision protocol is intentionally simple and leakage-safe:

* the outcome index is built from the chronological **training split only**;
* for a cut ``(node, cut_time)``, Y1/Y2 are the first two incident training
  events with ``event_time > cut_time``;
* a missing future is masked by omitting that row;
* historical neighbor edges, ancestor states, and the current root label are
  never outcomes.

The host query already emitted a bounded ``CompactCutTrace``.  This module
therefore performs no tree walk and no second model query.
"""

import math
from collections import Counter

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

LINK = "link"
NODE_CLASS = "node_class"

HORIZON_OMEGA = (0.5, 0.5)
MAX_HORIZONS = 2


def _fixed_bipartite_derangement(times_from, times_to, seed_int):
    """One derangement of the horizon-2 EVENTS among a bucket of cuts.

    We need a permutation ``pi`` of the bucket's cut indices such that for
    every source cut ``s``, the target cut ``pi(s) != s`` AND the target
    cut's own Y2 event is a STRICTLY-LATER future of the source cut
    (``target.Y2.time > source.cut.time``).  This is a bipartite perfect
    matching between "receivers" (cuts, on the left) and "donors"
    (their Y2 events, on the right); a cut may receive any donor's Y2 whose
    event time is strictly after the receiver's cut time, except it must not
    receive its own event (that would keep the pairing unchanged).

    Existence is not guaranteed for an arbitrary bucket; callers must drop
    (in ALL arms) any bucket for which no such matching exists, so the three
    structural arms keep an identical cut set.  Returns a list of donor
    indices (``receiver i gets donors[perm[i]]``) or None when no perfect
    matching exists.

    Implementation: a greedy augmenting-path (Kuhn) maximum matching over the
    legal edges, biased by a deterministic per-(bucket, seed) order so the
    result is reproducible across the two passes of exact replay.  On an
    equal-sized bipartite graph a maximum matching of size n is perfect.
    """
    n = len(times_from)
    rs = np.random.RandomState(int(seed_int))
    order = rs.permutation(n)
    # legal donor set per receiver: donor j's Y2 strictly later than
    # receiver i's cut time, and donor != receiver.
    adj = []
    for i in range(n):
        ci = times_from[i]
        row = [j for j in order
               if j != i and times_to[j] > ci]
        adj.append(row)
    # Kuhn's algorithm: match donors to receivers.
    match_donor = [-1] * n      # donor -> receiver
    match_receiver = [-1] * n   # receiver -> donor

    def try_augment(i, seen):
        for j in adj[i]:
            if j in seen:
                continue
            seen.add(j)
            if match_donor[j] == -1 or try_augment(match_donor[j], seen):
                match_donor[j] = i
                match_receiver[i] = j
                return True
        return False

    matched = 0
    recv_order = sorted(range(n), key=lambda i: rs.rand())
    for i in recv_order:
        if match_receiver[i] == -1:
            if try_augment(i, set()):
                matched += 1
    if matched < n:
        return None
    return [match_receiver[i] for i in range(n)]



# Structural-supervision modes (paper Part III).  ``production`` reproduces
# the historical behaviour exactly (``n_observations`` decides how many future
# events a cut may use).  The structural modes are ablation-only: they put
# every arm on the SAME cut set (cuts with both a legal Y1 and Y2) and differ
# only in how the second observation is used:
#   ``1obs``             — emit Y1 only, per-tree total weight 1 on Y1;
#   ``2obs_aligned``     — emit Y1 and Y2 (the production 2Obs target);
#   ``2obs_mispaired``   — emit Y1 and Y2, but derange the Y2 pairing inside
#                          (tau, role, coarse time) buckets, keeping the Y2
#                          marginal, count, mask and cut set identical.
# ``2obs_mispaired`` destroys the SAMPLE-LEVEL joint pairing of the two future
# observations of a cut.  It is NOT an ablation of a parent/child recursive
# alignment Y_v^(2)=Y_p(v)^(1): the current trace (a SELF spine per query
# root) never materialises such cross-layer occurrences.  Name it
# "shuffled/mispaired 2nd observation" in any writeup.
SUPERVISION_PRODUCTION = "production"
SUPERVISION_1OBS = "1obs"
SUPERVISION_2OBS_ALIGNED = "2obs_aligned"
SUPERVISION_2OBS_MISPAIRED = "2obs_mispaired"
SUPERVISION_MODES = (SUPERVISION_PRODUCTION, SUPERVISION_1OBS,
                     SUPERVISION_2OBS_ALIGNED, SUPERVISION_2OBS_MISPAIRED)


@dataclass(frozen=True)
class ObservedOutcome:
    """One real event viewed as a future outcome of one endpoint node.

    ``role`` identifies where the JODIE label owner lies relative to the
    queried node: 0 when the queried node is the event source (owner on the
    cut side), 1 when it is the destination (owner is the counterpart).
    """

    time: float
    counterpart: int
    role: int
    outcome: float
    outcome_id: tuple


class JodieFutureIndex:
    """Per-node chronological lookup over one explicitly supplied stream.

    Construct this object from ``dataset.train``.  Keeping the constructor
    stream-shaped (rather than accepting a split container) makes accidental
    train/validation/test crossing visible at the call site.
    """

    def __init__(self, train_stream):
        fields = (train_stream.sources, train_stream.destinations,
                  train_stream.timestamps, train_stream.edge_idxs,
                  train_stream.labels)
        lengths = {len(x) for x in fields}
        if len(lengths) != 1:
            raise ValueError("future-index stream fields must have equal length")

        by_node: Dict[int, List[Tuple[float, int, ObservedOutcome]]] = {}
        for pos, (src, dst, time, edge_idx, label) in enumerate(zip(*fields)):
            src_i, dst_i = int(src), int(dst)
            time_f = float(time)
            outcome_id = ("future", int(edge_idx))
            by_node.setdefault(src_i, []).append((
                time_f, pos, ObservedOutcome(
                    time=time_f, counterpart=dst_i, role=0,
                    outcome=float(label), outcome_id=outcome_id)))
            by_node.setdefault(dst_i, []).append((
                time_f, pos, ObservedOutcome(
                    time=time_f, counterpart=src_i, role=1,
                    outcome=float(label), outcome_id=outcome_id)))

        self._events: Dict[int, Tuple[ObservedOutcome, ...]] = {}
        self._times: Dict[int, np.ndarray] = {}
        for node, rows in by_node.items():
            # Stable stream position resolves equal timestamps, while the
            # query's side="right" still enforces strictly greater time.
            rows.sort(key=lambda x: (x[0], x[1]))
            events = tuple(x[2] for x in rows)
            self._events[node] = events
            self._times[node] = np.asarray(
                [x.time for x in events], dtype=np.float64)
        self.n_events = next(iter(lengths), 0)

    def query(self, node: int, cut_time: float,
              limit: int = MAX_HORIZONS) -> List[ObservedOutcome]:
        """Return at most ``limit`` real events with time strictly after cut."""
        if int(limit) < 0:
            raise ValueError("limit must be nonnegative")
        times = self._times.get(int(node))
        if times is None or limit == 0:
            return []
        start = int(np.searchsorted(times, float(cut_time), side="right"))
        out = list(self._events[int(node)][start:start + int(limit)])
        if any(event.time <= float(cut_time) for event in out):
            raise AssertionError("future index returned a non-future event")
        return out


@dataclass
class CutRecord:
    tree_id: int
    occurrence_id: int
    tau: str
    horizon: int
    node: int
    time: float                 # cut/query as-of time
    z: torch.Tensor             # graph-connected internal compressed state
    context: Dict[str, Any]
    outcome: float
    outcome_id: tuple
    # Pre-compression rich state; only the reconstruction ablation uses it.
    u: Optional[torch.Tensor] = None
    # Builders always fill this with an actual later timestamp.  The default
    # keeps low-level synthetic loss tests (which do not model time) concise.
    outcome_time: float = float("nan")
    weight: float = 1.0
    valid: bool = True

    @property
    def cut_id(self):
        return (int(self.tree_id), int(self.occurrence_id), str(self.tau))

    @property
    def row_id(self):
        return self.cut_id + (int(self.horizon),)

    @property
    def overlap_id(self):
        return (int(self.node), float(self.time), str(self.tau))

    def to(self, device):
        self.z = self.z.to(device)
        return self


def build_edge_tables(dataset):
    """Build full-stream edge maps for dataset audits only.

    This legacy helper remains available to callers that inspect endpoint or
    label-index consistency.  It is deliberately not used by
    ``JodieCutBuilder``: historical edge labels are not future supervision.
    """
    endpoints: Dict[int, Tuple[int, int]] = {}
    labels: Dict[int, float] = {}
    endpoint_conflicts = 0
    label_conflicts = 0
    user_nodes = set()
    page_nodes = set()
    stream = dataset.full if hasattr(dataset, "full") else dataset
    for idx, src, dst, label in zip(stream.edge_idxs, stream.sources,
                                    stream.destinations, stream.labels):
        edge_idx = int(idx)
        pair = (int(src), int(dst))
        user_nodes.add(pair[0])
        page_nodes.add(pair[1])
        if edge_idx in endpoints and endpoints[edge_idx] != pair:
            endpoint_conflicts += 1
        if edge_idx in labels and labels[edge_idx] != float(label):
            label_conflicts += 1
        endpoints[edge_idx] = pair
        labels[edge_idx] = float(label)
    if user_nodes & page_nodes:
        raise AssertionError(
            "JODIE graph must be bipartite; {} nodes appear on both sides"
            .format(len(user_nodes & page_nodes)))
    return (endpoints, labels, user_nodes, page_nodes,
            {"endpoint_conflicts": endpoint_conflicts,
             "label_conflicts": label_conflicts})


class JodieCutBuilder:
    """Turn bounded query-time candidates into Y1/Y2 CutRecord rows."""

    def __init__(self, future_index: JodieFutureIndex, *, stage: str,
                 cuts_per_tau: int = 32, seed: int = 0,
                 n_observations: int = 2,
                 supervision_mode: str = SUPERVISION_PRODUCTION):
        if stage not in (LINK, NODE_CLASS):
            raise ValueError("unknown stage {}".format(stage))
        if not hasattr(future_index, "query"):
            raise TypeError(
                "JodieCutBuilder requires a train-only JodieFutureIndex")
        if n_observations not in (1, 2):
            raise ValueError("n_observations must be 1 or 2")
        if supervision_mode not in SUPERVISION_MODES:
            raise ValueError("unknown supervision_mode {}".format(
                supervision_mode))
        # The structural arms are two-future constructions by definition:
        # they put every arm on the cuts that have BOTH a legal Y1 and Y2.
        if supervision_mode != SUPERVISION_PRODUCTION and \
                int(n_observations) != 2:
            raise ValueError(
                "supervision_mode={} requires n_observations=2".format(
                    supervision_mode))
        self.future_index = future_index
        self.stage = stage
        self.n_observations = int(n_observations)
        self.supervision_mode = str(supervision_mode)
        self.cuts_per_tau = int(cuts_per_tau)
        if self.cuts_per_tau < 1:
            raise ValueError("cuts_per_tau must be at least one")
        self.seed = int(seed)
        self._tree_counter = 0
        self._build_calls = 0

    def build(self, trace, batch_seed: int = 0, stats=None):
        """Turn a query trace into CutRecord rows.

        ``supervision_mode`` selects the structural-supervision protocol:
        ``production`` runs the historical path unchanged (bit-for-bit);
        the structural arms run a parallel, isolation-safe path that never
        shares code with the production branch.
        """
        if self.supervision_mode != SUPERVISION_PRODUCTION:
            return self._build_structural(trace, batch_seed, stats)
        return self._build_production(trace, batch_seed, stats)

    def _build_production(self, trace, batch_seed: int = 0, stats=None):
        """The historical path (bit-for-bit unchanged)."""
        if self.stage == LINK:
            raise NotImplementedError(
                "RPBE link supervision is not supported yet: the future "
                "index stores node-classification labels, which must not "
                "masquerade as link outcomes (paper spec section 2).  "
                "Use vanilla link pretraining.")
        if trace is None or not trace.root_rows or not trace.cuts:
            return []
        self._build_calls += 1
        if stats is None:
            stats = {}
        rng = np.random.RandomState((self.seed * 1000003) ^ int(batch_seed))

        if len(set(trace.root_rows)) != len(trace.root_rows):
            raise ValueError("trace.root_rows must be unique")
        tree_by_row = {
            int(row): self._tree_counter + local
            for local, row in enumerate(trace.root_rows)}
        self._tree_counter += len(trace.root_rows)

        per_tau_cuts: Dict[str, List[Tuple[tuple, List[CutRecord]]]] = {}
        raw = stats.setdefault("raw_candidates", {})
        valid = stats.setdefault("valid_rows", {})
        missing = stats.setdefault("missing_horizons", {})
        overlap = stats.setdefault("overlap_groups", {})
        outcome_use = stats.setdefault("outcome_use", {})

        for cut in trace.cuts:
            if int(cut.root_row) not in tree_by_row:
                raise ValueError(
                    "cut root_row {} is absent from trace.root_rows"
                    .format(cut.root_row))
            raw[cut.tau] = raw.get(cut.tau, 0) + 1
            future = self.future_index.query(
                cut.node, cut.time, limit=self.n_observations)
            for horizon in range(len(future) + 1,
                                 self.n_observations + 1):
                key = (cut.tau, horizon)
                missing[key] = missing.get(key, 0) + 1
            if not future:
                continue

            tree_id = tree_by_row[int(cut.root_row)]
            cut_key = (tree_id, int(cut.occurrence_id), str(cut.tau))
            rows = []
            for horizon, event in enumerate(future, start=1):
                delta_t = float(event.time) - float(cut.time)
                if not delta_t > 0.0:
                    raise AssertionError("future outcome must be strictly later")
                record = CutRecord(
                    tree_id=tree_id,
                    occurrence_id=int(cut.occurrence_id),
                    tau=str(cut.tau),
                    horizon=horizon,
                    node=int(cut.node),
                    time=float(cut.time),
                    outcome_time=float(event.time),
                    z=cut.z,
                    u=cut.u,
                    context={
                        "horizon": horizon,
                        "delta_t": delta_t,
                        "counterpart": int(event.counterpart),
                        "role": int(event.role),
                        "query_type": 0 if self.stage == LINK else 1,
                        "path": list(cut.path),
                    },
                    outcome=float(event.outcome),
                    outcome_id=tuple(event.outcome_id))
                rows.append(record)
                key = (cut.tau, horizon)
                valid[key] = valid.get(key, 0) + 1
                overlap[record.overlap_id] = \
                    overlap.get(record.overlap_id, 0) + 1
                outcome_use[record.outcome_id] = \
                    outcome_use.get(record.outcome_id, 0) + 1
            per_tau_cuts.setdefault(cut.tau, []).append((cut_key, rows))

        # Every query tree has total pre-cap cut weight one.  Thus a tree
        # with more internal layers cannot dominate simply by having more cuts.
        tree_cut_counts: Dict[int, int] = {}
        for cut_rows in per_tau_cuts.values():
            for cut_key, _ in cut_rows:
                tree_cut_counts[cut_key[0]] = \
                    tree_cut_counts.get(cut_key[0], 0) + 1
        tree_weight = {
            tree_id: 1.0 / float(count)
            for tree_id, count in tree_cut_counts.items()}

        out = []
        for tau, cut_rows in per_tau_cuts.items():
            if len(cut_rows) <= self.cuts_per_tau:
                selected = cut_rows
                sample_correction = 1.0
            else:
                indices = rng.choice(len(cut_rows), size=self.cuts_per_tau,
                                     replace=False)
                selected = [cut_rows[i] for i in sorted(indices)]
                sample_correction = \
                    float(len(cut_rows)) / float(self.cuts_per_tau)
            for cut_key, rows in selected:
                omega_sum = sum(
                    HORIZON_OMEGA[row.horizon - 1]
                    for row in rows[:self.n_observations])
                base = tree_weight[cut_key[0]] * sample_correction
                for row in rows:
                    row.weight = base * HORIZON_OMEGA[row.horizon - 1] \
                        / omega_sum
                out.extend(rows)
        return out

    # ------------------------------------------------------------------ structural
    @staticmethod
    def _row_from_event(tree_id, cut_key, cut, event, horizon, cut_time):
        """One CutRecord row for a (cut, event) pair (structural path)."""
        delta_t = float(event.time) - cut_time
        if not delta_t > 0.0:
            raise AssertionError("future outcome must be strictly later")
        return CutRecord(
            tree_id=tree_id,
            occurrence_id=cut_key[1],
            tau=cut_key[2],
            horizon=horizon,
            node=int(cut.node),
            time=cut_time,
            outcome_time=float(event.time),
            z=cut.z,
            u=cut.u,
            context={
                "horizon": horizon,
                "delta_t": delta_t,
                "counterpart": int(event.counterpart),
                "role": int(event.role),
                "query_type": 1,
                "path": list(cut.path),
            },
            outcome=float(event.outcome),
            outcome_id=tuple(event.outcome_id))

    def _build_structural(self, trace, batch_seed: int = 0, stats=None):
        """Structural-supervision arms (paper Part III).

        Every arm runs on the SAME cut set — the cuts whose node has at least
        two strictly-future training events (Y1 and Y2 both legal).  Arms
        differ only in how the second observation is used:

        * ``1obs``            emit Y1 only (weight 1 on the Y1 row);
        * ``2obs_aligned``    emit Y1 and the true Y2 of the same cut;
        * ``2obs_mispaired``  emit Y1 and a Y2 whose sample-level pairing is
                              deranged within a (tau, role2, coarse-time)
                              bucket.  Y2 marginal, count, mask and the cut
                              set are unchanged; the joint (Y1, Y2) pairing
                              is destroyed.  Singleton buckets (fewer than two
                              members) are dropped from ALL arms so the three
                              arms keep identical cut sets.

        Determinism: the derangement is a deterministic function of
        ``(self.seed, batch_seed, bucket)`` only, so the two passes of exact
        replay reconstruct the identical permutation.
        """
        if self.stage == LINK:
            raise NotImplementedError(
                "RPBE link supervision is not supported yet (structural "
                "arms use the node-classification future index).")
        if trace is None or not trace.root_rows or not trace.cuts:
            return []
        self._build_calls += 1
        if stats is None:
            stats = {}
        raw = stats.setdefault("raw_candidates", {})
        valid = stats.setdefault("valid_rows", {})
        missing = stats.setdefault("missing_horizons", {})
        overlap = stats.setdefault("overlap_groups", {})
        outcome_use = stats.setdefault("outcome_use", {})
        dropped_singletons = stats.setdefault("dropped_singleton_buckets", 0)
        dropped_match = 0

        if len(set(trace.root_rows)) != len(trace.root_rows):
            raise ValueError("trace.root_rows must be unique")
        tree_by_row = {
            int(row): self._tree_counter + local
            for local, row in enumerate(trace.root_rows)}
        self._tree_counter += len(trace.root_rows)

        # ------------------------------------------------------------------ pass 1
        # Keep only cuts with BOTH legal futures; bucket by
        # (tau, role-of-Y2, coarse cut-time bin) for the later derangement.
        units = []            # (tree_id, cut_key, cut, e1, e2) at original index
        buckets = {}          # (tau, role2, tbin) -> [unit index]
        two_future_cuts = []  # (cut, e2) among candidates (for bin sizing)
        for cut in trace.cuts:
            if int(cut.root_row) not in tree_by_row:
                raise ValueError(
                    "cut root_row {} is absent from trace.root_rows"
                    .format(cut.root_row))
            raw[cut.tau] = raw.get(cut.tau, 0) + 1
            future = self.future_index.query(cut.node, cut.time, limit=2)
            for horizon in range(len(future) + 1, 3):
                key = (cut.tau, horizon)
                missing[key] = missing.get(key, 0) + 1
            if len(future) < 2:
                continue
            two_future_cuts.append((cut, future))
        # A time-bucket count that adapts to the population: with very few
        # cuts we use few bins so each bucket has enough members to
        # derange; with real data the count caps at 16.
        n_two = len(two_future_cuts)
        n_tbins = 1 if n_two <= 1 else min(16, max(1, int(math.ceil(
            n_two / 5.0))))
        times = [float(c.time) for c, _f in two_future_cuts]
        tmin = min(times) if times else 0.0
        tspan = (max(times) - tmin) if times else 0.0
        for cut, future in two_future_cuts:
            tree_id = tree_by_row[int(cut.root_row)]
            cut_key = (tree_id, int(cut.occurrence_id), str(cut.tau))
            e1, e2 = future[0], future[1]
            tbin = 0 if tspan <= 0.0 else int(
                (float(cut.time) - tmin) / tspan * n_tbins)
            bkey = (str(cut.tau), int(e2.role), int(tbin))
            idx = len(units)
            units.append((tree_id, cut_key, cut, e1, e2))
            buckets.setdefault(bkey, []).append(idx)

        # Singleton derangement buckets are dropped from EVERY arm.
        surviving = set()
        for idxs in buckets.values():
            if len(idxs) < 2:
                dropped_singletons += len(idxs)
                continue
            surviving.update(idxs)
        # A bucket whose cuts cannot be put on a constrained perfect
        # derangement of Y2 (every assignment legal, every Y2 moved) must
        # also be dropped from EVERY arm — otherwise the mispaired arm would
        # either break the Y2 marginal or share cuts with the other arms
        # only by keeping some pairings fixed.  The feasibility depends only
        # on the bucket contents (cut times vs own Y2 times), never on the
        # arm, so it is computed once and applied identically to all three
        # arms.
        by_index_full = {i: u for i, u in enumerate(units)}
        for idxs in buckets.values():
            ids = sorted(i for i in idxs if i in surviving)
            if len(ids) < 2:
                continue
            bucket_seed = 0
            for i in ids:
                bucket_seed = (bucket_seed * 31 + i) % (2 ** 31)
            times_from = [by_index_full[i][2].time for i in ids]
            times_to = [by_index_full[i][4].time for i in ids]
            perm = _fixed_bipartite_derangement(
                times_from, times_to,
                (self.seed * 104729) ^ int(batch_seed) ^ bucket_seed)
            if perm is None:
                dropped_match += len(ids)
                for i in ids:
                    surviving.discard(i)
        by_index = {i: u for i, u in enumerate(units) if i in surviving}
        units = by_index

        # ------------------------------------------------------------------ pass 2
        # Y2 mispairing for the mispaired arm only: reuse the (deterministic)
        # constrained perfect matching of each surviving bucket.  Every
        # surviving cut receives a DIFFERENT cut's Y2 that is a strictly
        # later legal future, so the aligned and mispaired arms share the
        # Y2 multiset exactly and no pairing is left unchanged.
        y2_for_unit = {}
        if self.supervision_mode == SUPERVISION_2OBS_MISPAIRED:
            for idxs in buckets.values():
                ids = sorted(i for i in idxs if i in surviving)
                if len(ids) < 2:
                    continue
                bucket_seed = 0
                for i in ids:
                    bucket_seed = (bucket_seed * 31 + i) % (2 ** 31)
                times_from = [by_index_full[i][2].time for i in ids]
                times_to = [by_index_full[i][4].time for i in ids]
                perm = _fixed_bipartite_derangement(
                    times_from, times_to,
                    (self.seed * 104729) ^ int(batch_seed) ^ bucket_seed)
                # A surviving bucket always has a matching (checked above);
                # perm is therefore never None here.
                assert perm is not None
                for pos, i in enumerate(ids):
                    j = ids[perm[pos]]
                    assert j != i
                    y2_for_unit[i] = by_index_full[j][4]
        stats["mispaired_bucket_dropped"] = stats.get(
            "mispaired_bucket_dropped", 0) + dropped_match

        # ------------------------------------------------------------------ rows
        tree_cut_counts = {}
        for tree_id, _cut_key, _cut, _e1, _e2 in units.values():
            tree_cut_counts[tree_id] = tree_cut_counts.get(tree_id, 0) + 1
        tree_weight = {t: 1.0 / float(c)
                       for t, c in tree_cut_counts.items()}

        per_tau_cuts = {}
        rng = np.random.RandomState((self.seed * 1000003) ^ int(batch_seed))
        for i in sorted(units):
            tree_id, cut_key, cut, e1, e2 = units[i]
            rows = [self._row_from_event(
                tree_id, cut_key, cut, e1, 1, float(cut.time))]
            if self.supervision_mode != SUPERVISION_1OBS:
                y2 = y2_for_unit.get(i, e2)
                rows.append(self._row_from_event(
                    tree_id, cut_key, cut, y2, 2, float(cut.time)))
            for row in rows:
                key = (cut_key[2], row.horizon)
                valid[key] = valid.get(key, 0) + 1
                overlap[row.overlap_id] = overlap.get(row.overlap_id, 0) + 1
                outcome_use[row.outcome_id] = \
                    outcome_use.get(row.outcome_id, 0) + 1
            per_tau_cuts.setdefault(cut_key[2], []).append((cut_key, rows))

        out = []
        for tau, cut_rows in per_tau_cuts.items():
            if len(cut_rows) <= self.cuts_per_tau:
                selected = cut_rows
                sample_correction = 1.0
            else:
                indices = rng.choice(len(cut_rows), size=self.cuts_per_tau,
                                     replace=False)
                selected = [cut_rows[i] for i in sorted(indices)]
                sample_correction = \
                    float(len(cut_rows)) / float(self.cuts_per_tau)
            for cut_key, rows in selected:
                base = tree_weight[cut_key[0]] * sample_correction
                if self.supervision_mode == SUPERVISION_1OBS:
                    rows[0].weight = base
                else:
                    for row in rows:
                        row.weight = base * HORIZON_OMEGA[row.horizon - 1]
                out.extend(rows)
        return out
