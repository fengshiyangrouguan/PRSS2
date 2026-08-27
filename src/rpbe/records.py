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

# Horizon weights (method_delta.md: omega_1 = omega_2 = 1/4, omega_star =
# 1/2): the star horizon (root task result) dominates the joint test.
# Within a cut the surviving horizons are renormalized so the cut's total
# horizon weight is 1 (tree equal weights then act on cuts).
HORIZON_OMEGA = (0.25, 0.25, 0.5)
MAX_HORIZONS = 3


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

    JODIE semantics (fifth review): the label is the USER's state change
    (``state_label``), so ``label_owner_node`` is ALWAYS the source node —
    never inferred from whatever the carrier happens to be.  Wikipedia is
    bipartite (users edit pages), so the builder also returns the user /
    page node sets for the structural audit.
    """
    endpoints: Dict[int, Tuple[int, int]] = {}
    labels: Dict[int, float] = {}
    endpoint_conflicts = 0
    label_conflicts = 0
    user_nodes = set()
    page_nodes = set()
    for idx, s, d, y in zip(dataset.full.edge_idxs, dataset.full.sources,
                            dataset.full.destinations, dataset.full.labels):
        i = int(idx)
        pair = (int(s), int(d))
        user_nodes.add(int(s))
        page_nodes.add(int(d))
        if i in endpoints and endpoints[i] != pair:
            endpoint_conflicts += 1
        endpoints[i] = pair
        if i in labels and labels[i] != float(y):
            label_conflicts += 1
        labels[i] = float(y)
    # Bipartite structure: a node is either a user or a page, never both
    # (the audit measured overlap 0 on wikipedia; the assert pins it).
    assert not (user_nodes & page_nodes), \
        "JODIE graph must be bipartite; {} nodes appear on both sides" \
        .format(len(user_nodes & page_nodes))
    return (endpoints, labels, user_nodes, page_nodes,
            {"endpoint_conflicts": endpoint_conflicts,
             "label_conflicts": label_conflicts})


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
        (self.endpoints, self.labels,
         self.user_nodes, self.page_nodes) = edge_tables
        self.stage = stage
        self.cuts_per_tau = int(cuts_per_tau)
        self.seed = int(seed)
        self._tree_counter = 0
        # Pass-2 audit counter (sixth review): the replay pass must not
        # re-walk; the loop asserts this does not move during pass 2.
        self._build_calls = 0

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
        self._build_calls += 1
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
                # ``role`` encodes the owner position of the task label:
                # 0 for TASK_SRC roots (the label belongs to the traced
                # source user = the cut-side node itself) and 1 for
                # POS_DST shadow roots (the owner is the event's source,
                # the consumer side).  ``counterpart`` is the OTHER
                # endpoint in both cases.
                root_cons[oid] = {
                    "outcome": float(ev["label"]),
                    "outcome_id": ("root", int(ev["event_idx"])),
                    "counterpart": int(ev["counterpart"]),
                    "edge_time": float(ev["time"]),
                    "role": int(ev.get("role", 0))}

        per_tau_cuts: Dict[str, List[Tuple[tuple, List[CutRecord]]]] = {}
        stats_raw = stats.setdefault("raw_occurrences", {})
        stats_kind = stats.setdefault("consumption_kind", {})
        stats_rows = stats.setdefault("valid_rows", {})
        stats_overlap = stats.setdefault("overlap_groups", {})
        stats_outcome = stats.setdefault("outcome_use", {})
        stats_ctype = stats.setdefault("cut_node_type", {})
        stats_orole = stats.setdefault("owner_position", {})
        for oid in trace.postorder():
            occ = trace.occurrences[oid]
            node = int(occ.metadata.get("node", -1))
            time = float(occ.metadata.get("time", -1.0))
            if node < 0:
                continue
            stats_raw[occ.tau] = stats_raw.get(occ.tau, 0) + 1
            # Structural audit (fifth review, spec E): bipartite types
            # must alternate with depth ALONG NEIGHBOR edges.  The SELF
            # recursion chain (relation 0) recurses on the SAME node and
            # is same-type by construction — a neighbor edge (relation 1)
            # joining two same-type nodes is a collector bug, counted and
            # asserted separately.
            is_user = node in self.user_nodes
            ctype = "user" if is_user else ("page" if node in self.page_nodes
                                            else "unknown")
            stats_ctype[(occ.tau, ctype)] = \
                stats_ctype.get((occ.tau, ctype), 0) + 1
            par = parent_of.get(oid)
            if par is not None:
                pnode = int(trace.occurrences[par].metadata.get("node", -1))
                if pnode >= 0:
                    p_user = pnode in self.user_nodes
                    if p_user == is_user:
                        p_occ = trace.occurrences[par]
                        i = p_occ.children.index(oid)
                        rel = int(p_occ.child_relations[i]) \
                            if i < len(p_occ.child_relations) else 0
                        key = ("neighbor_same_type" if rel == 1
                               else "self_chain_same_type")
                        stats[key] = stats.get(key, 0) + 1
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
                stats_orole[(occ.tau, int(r.context["role"]))] = \
                    stats_orole.get((occ.tau, int(r.context["role"])), 0) + 1
            if rows:
                per_tau_cuts.setdefault(occ.tau, []).append((cut_key, rows))

        # Tree equal weights (doc section 4): every tree contributes total
        # weight 1, counted over its PRE-CAP cuts across all interfaces —
        # a leaf flood cannot starve the root layers.
        tree_cut_counts: Dict[int, int] = {}
        for cut_rows in per_tau_cuts.values():
            for cut_key, _ in cut_rows:
                tree_cut_counts[cut_key[0]] = \
                    tree_cut_counts.get(cut_key[0], 0) + 1
        w_tree = {t: 1.0 / float(n) for t, n in tree_cut_counts.items()}

        # Depth-balanced sampling BY CUT: a sampled cut keeps ALL its
        # horizon rows.  When the cap samples, the surviving rows are
        # up-weighted by n_total / n_sampled (sampling-probability
        # correction): total weight is conserved per tau.
        out = []
        for tau, cut_rows in per_tau_cuts.items():
            if len(cut_rows) <= self.cuts_per_tau:
                picks = cut_rows
                w_sample = 1.0
            else:
                idx = rng.choice(len(cut_rows), size=self.cuts_per_tau,
                                 replace=False)
                picks = [cut_rows[i] for i in sorted(idx)]
                w_sample = float(len(cut_rows)) / float(self.cuts_per_tau)
            for cut_key, rows in picks:
                # Horizon mixing with the document weights: omega_h =
                # (1/4, 1/4, 1/2), renormalized over the SURVIVING
                # horizons of this cut (missing horizons are masked, the
                # cut's total horizon weight stays 1 for tree equal
                # weights).
                omega_sum = sum(HORIZON_OMEGA[:len(rows)])
                w = w_tree[cut_key[0]] * w_sample
                for r in rows:
                    r.weight = w * HORIZON_OMEGA[r.horizon - 1] / omega_sum
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
        while anc is not None and len(probes) < MAX_HORIZONS:
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
        if len(probes) < MAX_HORIZONS:
            walk["terminated_by_depth"] = True
        return probes, walk

    def _probe_of(self, anc, trace, root_cons):
        """The usable probe record of one ancestor, plus its kind.

        Alignment rule (fifth review): the label belongs to the
        interaction's SOURCE user (JODIE ``state_label``), never to
        "whoever the carrier is".  A probe is valid when its edge's label
        owner lies ON the continuation path — which is automatic for a
        historical edge, whose owner is one of its two endpoints (the
        cut-side node or the consumer-side parent).  ``role`` encodes the
        owner POSITION: 0 = owner is the cut-side node, 1 = owner is the
        consumer side.  ``carrier == label_owner`` is NOT a gate (it is a
        descriptive statistic): a page's history affecting a user's state
        is exactly the cross-node signal we keep.

        Kinds: ``root`` (task record; the label belongs to the traced
        source user, the cut-side node itself — role 0), ``aligned``
        (usable historical edge), ``self`` (SELF step, missing record, or
        owner off the path — skipped).
        """
        a_occ = trace.occurrences[anc]
        if anc in root_cons:
            rec = root_cons[anc]
            return ({"outcome": rec["outcome"],
                     "outcome_id": rec["outcome_id"],
                     "counterpart": rec["counterpart"],
                     "edge_time": rec["edge_time"],
                     "role": int(rec["role"])}, "root")
        cons = a_occ.metadata.get("consumption")
        if not isinstance(cons, dict) or cons.get("kind") != "edge":
            return None, "self"
        e_idx = int(cons["edge_idx"])
        if e_idx not in self.labels:
            return None, "self"
        owner = int(cons.get("label_owner", -1))
        node = int(a_occ.metadata.get("node", -1))
        counterpart = int(cons.get("counterpart", -1))
        if owner == node:
            role = 0      # CUT_SIDE: label belongs to the cut-side node
        elif owner == counterpart:
            role = 1      # CONSUMER_SIDE: label belongs to the parent
        else:
            return None, "self"   # owner off the path: data anomaly
        return ({"outcome": float(self.labels[e_idx]),
                 "outcome_id": ("edge", e_idx),
                 "counterpart": counterpart,
                 "edge_time": float(cons.get("edge_time", 0.0)),
                 "role": role}, "aligned")

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
