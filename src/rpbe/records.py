"""CutRecord rows and the same-tree consumption probes.

A cut is one compression call inside a host computation tree.  Its supervised
pairing does NOT come from the data timeline (no ``find_next_event``): for
horizon ``h`` the record is the h-th *valid consumption probe* along the
tree's upward continuation — the real interaction edge through which an
ancestor occurrence consumed its child state, or the root task record when
the walk reaches the tree root.

Consumption records are written by the host adapter at tree construction
time (``metadata["consumption"]``); the builder only walks the trace.
Censoring is tree-level: a tree rooted at a train event is entirely inside
the train region, so there is no per-cut val/test filter here.

Four identities are kept separate:

* ``cut_id``     = (tree_id, occurrence_id, tau) — unique-cut counting
* ``row_id``     = cut_id + horizon — window row dedupe
* ``overlap_id`` = (node, time, tau) — correlation grouping (never a delete key)
* ``outcome_id`` = ("edge", edge_idx) / ("root", event_idx) — label reuse stats
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

LINK = "link"
NODE_CLASS = "node_class"


@dataclass
class CutRecord:
    tree_id: int
    occurrence_id: int       # global per-adapter counter (unique across trees)
    tau: str
    horizon: int             # 1 / 2 (the star horizon is reserved)
    node: int                # the cut occurrence's node
    time: float              # as-of time of the cut (= tree query time)
    z: torch.Tensor          # [r_tau], graph-connected
    context: Dict[str, Any]  # raw C: horizon, delta_t, counterpart, role,
                             # query_type, path [(rel, dt), ...]
    outcome: float           # raw Y
    outcome_id: tuple        # ("edge", edge_idx) | ("root", event_idx)
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
    """Explicit ``idx -> (source, destination)`` and ``idx -> label`` maps.

    ``graph_df.idx`` is a 1-based edge-feature index whose convention we do
    NOT assume (no ``labels[edge_idx]`` indexing anywhere): the maps are
    built from the data itself.  Conflicts (one idx shared by different
    endpoints or labels) are counted for the audit.
    """
    endpoints: Dict[int, Tuple[int, int]] = {}
    labels: Dict[int, float] = {}
    endpoint_conflicts = 0
    label_conflicts = 0
    for idx, s, d, y in zip(dataset.full.edge_idxs, dataset.full.sources,
                            dataset.full.destinations, dataset.full.labels):
        i = int(idx)
        pair = (int(s), int(d))
        if i in endpoints and endpoints[i] != pair:
            endpoint_conflicts += 1
        endpoints[i] = pair
        if i in labels and labels[i] != float(y):
            label_conflicts += 1
        labels[i] = float(y)
    return endpoints, labels, {"endpoint_conflicts": endpoint_conflicts,
                               "label_conflicts": label_conflicts}


class JodieCutBuilder:
    """Walks a host trace into per-horizon CutRecord rows for one stage.

    Stage 1 (link) and stage 2 (node_class) differ only in ``query_type``;
    every row carries ONE real consumption probe, and fabricated negatives
    belong to the link TASK loss only and never enter the Ky Fan measurement.
    """

    def __init__(self, edge_tables, *, stage: str,
                 cuts_per_tau: int = 32, seed: int = 0):
        if stage not in (LINK, NODE_CLASS):
            raise ValueError("unknown stage {}".format(stage))
        self.endpoints, self.labels = edge_tables
        self.stage = stage
        self.cuts_per_tau = int(cuts_per_tau)
        self.seed = int(seed)
        self._tree_counter = 0

    def build(self, trace, root_events=None, batch_seed: int = 0,
              stats=None):
        """One call per batch; ``batch_seed`` keeps sampling deterministic.

        ``root_events`` maps ``trace.root_rows`` entries to the real task
        event: ``{"dst", "label", "time", "event_idx"}`` — the root task
        record is the tree-level supervisor reached at the end of the walk.

        ``stats`` (optional dict, audit path) accumulates the cut funnel:
        raw occurrence counts per tau, consumption kind coverage, probe
        alignment, skipped steps, valid rows, overlap groups and outcome
        reuse.  All counters are dicts/counts the caller can aggregate.
        """
        if trace is None or not trace.roots:
            return []
        if stats is None:
            stats = {}
        rng = np.random.RandomState((self.seed * 1000003) ^ int(batch_seed))
        # Map occurrence -> its GLOBAL tree id (a per-builder counter, so
        # tree-level statistics stay meaningful across batches).
        occ_to_tree = {}
        for local_tree_id, root in enumerate(trace.roots):
            tree_id = self._tree_counter + local_tree_id
            stack = [root]
            while stack:
                oid = stack.pop()
                if oid in occ_to_tree:
                    continue
                occ_to_tree[oid] = tree_id
                stack.extend(trace.occurrences[oid].children)
        self._tree_counter += len(trace.roots)

        # Parent inverse map; the JODIE trace must be a tree (one parent
        # per occurrence — an occurrence is created per (row, neighbor
        # slot), never shared).
        parent_of: Dict[int, int] = {}
        for oid, occ in trace.occurrences.items():
            for child in occ.children:
                if child in parent_of:
                    raise ValueError(
                        "occurrence {} has multiple parents; the JODIE "
                        "trace must be a tree".format(child))
                parent_of[child] = oid

        root_cons: Dict[int, dict] = {}
        for row, oid in zip(trace.root_rows, trace.roots):
            ev = (root_events or {}).get(int(row))
            if ev is not None:
                root_cons[oid] = {
                    "outcome": float(ev["label"]),
                    "outcome_id": ("root", int(ev["event_idx"])),
                    "counterpart": int(ev["dst"]),
                    "edge_time": float(ev["time"]),
                    "role": 1}

        per_tau_cuts: Dict[str, List[Tuple[tuple, List[CutRecord]]]] = {}
        stats_raw = stats.setdefault("raw_occurrences", {})
        stats_kind = stats.setdefault("consumption_kind", {})
        stats_rows = stats.setdefault("valid_rows", {})
        stats_overlap = stats.setdefault("overlap_groups", {})
        stats_outcome = stats.setdefault("outcome_use", {})
        for oid in trace.postorder():
            occ = trace.occurrences[oid]
            node = int(occ.metadata.get("node", -1))
            time = float(occ.metadata.get("time", -1.0))
            if node < 0:
                continue
            stats_raw[occ.tau] = stats_raw.get(occ.tau, 0) + 1
            cons = occ.metadata.get("consumption")
            kind = cons.get("kind") if isinstance(cons, dict) else "none"
            stats_kind[(occ.tau, kind)] = \
                stats_kind.get((occ.tau, kind), 0) + 1
            cut_key = (occ_to_tree[oid], int(oid), str(occ.tau))
            probes, walk = self._walk_up(oid, parent_of, trace, root_cons)
            if walk["aligned"]:
                stats["aligned_probes"] = \
                    stats.get("aligned_probes", 0) + walk["aligned"]
            if walk["unaligned"]:
                stats["unaligned_probes"] = \
                    stats.get("unaligned_probes", 0) + walk["unaligned"]
            if walk["self_steps"]:
                stats["self_steps_skipped"] = \
                    stats.get("self_steps_skipped", 0) + walk["self_steps"]
            if walk["hit_root"]:
                stats["root_records_used"] = \
                    stats.get("root_records_used", 0) + 1
            if walk["terminated_by_depth"]:
                stats["depth_terminated"] = \
                    stats.get("depth_terminated", 0) + 1
            rows = []
            for h, (rec, path) in enumerate(probes, start=1):
                r = self._record(occ, cut_key, h, time, rec, path)
                rows.append(r)
                stats_rows[(occ.tau, h)] = \
                    stats_rows.get((occ.tau, h), 0) + 1
                stats_overlap[r.overlap_id] = \
                    stats_overlap.get(r.overlap_id, 0) + 1
                stats_outcome[r.outcome_id] = \
                    stats_outcome.get(r.outcome_id, 0) + 1
            if rows:
                w = 1.0 / len(rows)      # horizon mixing: w_{v,h} = w_v/|H_v|
                for r in rows:
                    r.weight = w
                per_tau_cuts.setdefault(occ.tau, []).append((cut_key, rows))

        # Depth-balanced sampling BY CUT: a sampled cut keeps ALL its
        # horizon rows (sampling-probability correction arrives together
        # with tree equal weights, task 3).
        out = []
        for tau, cut_rows in per_tau_cuts.items():
            if len(cut_rows) <= self.cuts_per_tau:
                picks = cut_rows
            else:
                idx = rng.choice(len(cut_rows), size=self.cuts_per_tau,
                                 replace=False)
                picks = [cut_rows[i] for i in sorted(idx)]
            for _, rows in picks:
                out.extend(rows)
        return out

    # ------------------------------------------------------------- walk logic
    def _walk_up(self, oid, parent_of, trace, root_cons):
        """Upward walk: up to two valid probes as ``[(record, path), ...]``.

        A step is SKIPPED when the ancestor has no usable probe — a SELF
        recursion step (no interaction of its own) or a historical edge
        whose label does not belong to the ancestor's node (unaligned).
        Skipped steps stay in ``path`` so C keeps the walk structure.

        Returns ``(probes, walk)`` where ``walk`` carries the audit counts
        (aligned/unaligned probes inspected, SELF steps skipped, whether
        the root record was reached, whether the walk ran out of tree).
        """
        probes = []
        path: List[tuple] = []
        walk = {"aligned": 0, "unaligned": 0, "self_steps": 0,
                "hit_root": False, "terminated_by_depth": False}
        anc = parent_of.get(oid)
        while anc is not None and len(probes) < 2:
            a_occ = trace.occurrences[anc]
            i = a_occ.children.index(oid)
            rel = int(a_occ.child_relations[i]) \
                if i < len(a_occ.child_relations) else 0
            dt = float(a_occ.child_delta_t[i]) \
                if i < len(a_occ.child_delta_t) else 0.0
            path.append((rel, dt))
            rec, kind = self._probe_of(anc, trace, root_cons)
            if kind == "aligned":
                walk["aligned"] += 1
            elif kind == "unaligned":
                walk["unaligned"] += 1
            elif kind == "self":
                walk["self_steps"] += 1
            elif kind == "root":
                walk["aligned"] += 1
                walk["hit_root"] = True
            if rec is not None:
                probes.append((rec, list(path)))
            oid, anc = anc, parent_of.get(anc)
        if len(probes) < 2:
            walk["terminated_by_depth"] = True
        return probes, walk

    def _probe_of(self, anc, trace, root_cons):
        """The usable probe record of one ancestor, plus its kind.

        Kinds: ``root`` (task record), ``aligned`` (historical edge whose
        label belongs to the ancestor's node), ``unaligned`` (historical
        edge, label belongs elsewhere — skipped), ``self`` (SELF step or
        no record — skipped).  Only root/aligned yield a record.
        """
        a_occ = trace.occurrences[anc]
        if anc in root_cons:
            rec = root_cons[anc]
            return ({"outcome": rec["outcome"],
                     "outcome_id": rec["outcome_id"],
                     "counterpart": rec["counterpart"],
                     "edge_time": rec["edge_time"],
                     "role": rec["role"]}, "root")
        cons = a_occ.metadata.get("consumption")
        if not isinstance(cons, dict) or cons.get("kind") != "edge":
            return None, "self"
        e_idx = int(cons["edge_idx"])
        owner = int(cons.get("label_owner", -1))
        node = int(a_occ.metadata.get("node", -1))
        if owner != node:
            return None, "unaligned"
        if e_idx not in self.labels:
            return None, "unaligned"
        return ({"outcome": float(self.labels[e_idx]),
                 "outcome_id": ("edge", e_idx),
                 "counterpart": int(cons.get("counterpart", -1)),
                 "edge_time": float(cons.get("edge_time", 0.0)),
                 "role": int(cons.get("endpoint_role", 1))}, "aligned")

    def _record(self, occ, cut_key, horizon: int, as_of: float, rec, path):
        return CutRecord(
            tree_id=cut_key[0], occurrence_id=cut_key[1], tau=cut_key[2],
            horizon=horizon,
            node=int(occ.metadata.get("node", -1)),
            time=float(as_of),
            z=occ.state.z,
            context={
                "horizon": int(horizon),
                "delta_t": max(0.0, float(as_of) - float(rec["edge_time"])),
                "counterpart": int(rec["counterpart"]),
                "role": int(rec["role"]),
                "query_type": 0 if self.stage == LINK else 1,
                "path": list(path),
            },
            outcome=float(rec["outcome"]),
            outcome_id=tuple(rec["outcome_id"]))
