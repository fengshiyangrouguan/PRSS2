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

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

LINK = "link"
NODE_CLASS = "node_class"

HORIZON_OMEGA = (0.5, 0.5)
MAX_HORIZONS = 2


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
                 n_observations: int = 2):
        if stage not in (LINK, NODE_CLASS):
            raise ValueError("unknown stage {}".format(stage))
        if not hasattr(future_index, "query"):
            raise TypeError(
                "JodieCutBuilder requires a train-only JodieFutureIndex")
        if n_observations not in (1, 2):
            raise ValueError("n_observations must be 1 or 2")
        self.future_index = future_index
        self.stage = stage
        self.n_observations = int(n_observations)
        self.cuts_per_tau = int(cuts_per_tau)
        if self.cuts_per_tau < 1:
            raise ValueError("cuts_per_tau must be at least one")
        self.seed = int(seed)
        self._tree_counter = 0
        self._build_calls = 0

    def build(self, trace, batch_seed: int = 0, stats=None):
        """Build rows without walking a tree or invoking the host again."""
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
