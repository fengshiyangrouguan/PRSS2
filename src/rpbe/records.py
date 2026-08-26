"""CutRecord rows and the node-grouped, time-sorted future index.

Each cut of a host computation tree pairs the compressed state ``z_v`` (with
gradient) with one or more joint tests ``(c, y)`` built from *real observed*
future continuations of the cut's node.  ``c`` describes how the future asks
the cut ("what, at what delay, in which role"), ``y`` is the data's actual
answer.  Neither ever enters the compressor.

A cut whose next future event falls in val/test is censored: the builder emits
nothing for it (``valid=False`` is the *absence* of rows, never a negative
outcome — censored != y=0).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

LINK = "link"
NODE_CLASS = "node_class"


@dataclass
class CutRecord:
    tree_id: int            # root row index of the computation tree
    cut_id: int             # occurrence id (unique per trace)
    occurrence_id: int
    tau: str
    node: int
    time: float
    z: torch.Tensor         # [r_tau], graph-connected
    context: Dict[str, Any]  # raw C: delta_t, counterpart id, role, query_type
    outcome: float          # raw Y: 0/1 for both stages
    valid: bool = True

    def to(self, device):
        self.z = self.z.to(device)
        return self


class FutureIndex:
    """Per-node, time-sorted event table with split flags.

    Events are grouped by *source* node (the JODIE protocol reads a node's
    future through the interactions it initiates).  ``query`` returns the first
    event with ``time > t``; the result is censored when that event belongs to
    the val/test region.
    """

    def __init__(self, sources: np.ndarray, destinations: np.ndarray,
                 timestamps: np.ndarray, labels: np.ndarray,
                 val_time: float, test_time: float):
        self.val_time = float(val_time)
        self.test_time = float(test_time)
        self.unique_destinations = np.unique(np.asarray(destinations))
        per_node: Dict[int, List] = {}
        for u, dst, t, y in zip(sources, destinations, timestamps, labels):
            per_node.setdefault(int(u), []).append(
                (float(t), int(dst), float(y)))
        self._events: Dict[int, np.ndarray] = {}
        for u, events in per_node.items():
            arr = np.asarray(events, dtype=np.float64)  # [n, 3]: time, dst, y
            arr = arr[np.argsort(arr[:, 0])]
            self._events[u] = arr

    def _split_of(self, t: float) -> str:
        if t <= self.val_time:
            return "train"
        if t <= self.test_time:
            return "val"
        return "test"

    def query(self, node: int, t: float) -> Optional[dict]:
        """First source-event of ``node`` after time ``t``.

        Returns ``{"time", "counterpart", "outcome", "valid"}``; ``None`` when
        the node has no future event at all, ``valid=False`` when the next
        event lies in val/test (censored — its content must not be read).
        """
        events = self._events.get(int(node))
        if events is None:
            return None
        idx = int(np.searchsorted(events[:, 0], float(t), side="right"))
        if idx >= len(events):
            return None
        nxt = events[idx]
        split = self._split_of(float(nxt[0]))
        if split != "train":
            return {"time": float(nxt[0]), "counterpart": None,
                    "outcome": None, "valid": False}
        return {"time": float(nxt[0]), "counterpart": int(nxt[1]),
                "outcome": float(nxt[2]), "valid": True}

    def query_batch(self, nodes: np.ndarray, times: np.ndarray):
        """Vectorized ``query`` over aligned (node, time) pairs.

        Returns ``(found, valid, hit_time, hit_dst, hit_outcome)`` arrays:
        ``found`` marks pairs whose node has ANY future event (beyond
        val/test it is ``valid=False``, content left unread); censored or
        missing pairs carry 0 / nan placeholders.
        """
        nodes = np.asarray(nodes, dtype=np.int64)
        times = np.asarray(times, dtype=np.float64)
        n = len(nodes)
        found = np.zeros(n, dtype=bool)
        valid = np.zeros(n, dtype=bool)
        hit_time = np.full(n, np.nan, dtype=np.float64)
        hit_dst = np.zeros(n, dtype=np.int64)
        hit_outcome = np.zeros(n, dtype=np.float64)
        for u in np.unique(nodes):
            events = self._events.get(int(u))
            if events is None:
                continue
            rows = np.flatnonzero(nodes == u)
            pos = np.searchsorted(events[:, 0], times[rows], side="right")
            has_next = pos < len(events)
            rows_next = rows[has_next]
            if rows_next.size == 0:
                continue
            nxt = events[pos[has_next]]
            in_train = nxt[:, 0] <= self.val_time
            found[rows_next] = True
            valid[rows_next] = in_train
            rows_ok = rows_next[in_train]
            if rows_ok.size == 0:
                continue
            nxt_ok = nxt[in_train]
            hit_time[rows_ok] = nxt_ok[:, 0]
            hit_dst[rows_ok] = nxt_ok[:, 1].astype(np.int64)
            hit_outcome[rows_ok] = nxt_ok[:, 2]
        return found, valid, hit_time, hit_dst, hit_outcome


class JodieCutBuilder:
    """Turns an adapter trace into CutRecord rows for one training stage.

    Stage 1 (link): one positive row (the next real interaction, y=1) plus
    ``neg_per_cut`` random-candidate rows (y=0) per cut; ``z`` repeats across
    the rows of one cut.  Stage 2 (node_class): one row per cut whose outcome
    is the ``state_label`` of the next real interaction.
    """

    def __init__(self, future_index: FutureIndex, *, stage: str,
                 neg_per_cut: int = 4, seed: int = 0):
        if stage not in (LINK, NODE_CLASS):
            raise ValueError("unknown stage {}".format(stage))
        self.future_index = future_index
        self.stage = stage
        self.neg_per_cut = int(neg_per_cut)
        self.seed = int(seed)

    def build(self, trace, batch_seed: int = 0) -> List[CutRecord]:
        """One call per batch; ``batch_seed`` keeps negative sampling
        deterministic per global step (no internal state mutation).

        The per-cut Python loop is reduced to Record construction only: the
        future lookups run through ``FutureIndex.query_batch`` (one
        searchsorted per unique node) and stage-1 negatives are drawn with a
        single vectorized ``choice``.
        """
        if trace is None or not trace.roots:
            return []
        # Map occurrence -> its root row for tree_id: walk each root's own
        # subtree (occurrences are never shared between roots).
        occ_to_tree = {}
        for tree_id, root in enumerate(trace.roots):
            stack = [root]
            while stack:
                oid = stack.pop()
                if oid in occ_to_tree:
                    continue
                occ_to_tree[oid] = tree_id
                stack.extend(trace.occurrences[oid].children)

        oids = list(trace.postorder())
        occs = [trace.occurrences[oid] for oid in oids]
        nodes = np.asarray([int(o.metadata.get("node", -1)) for o in occs],
                           dtype=np.int64)
        times = np.asarray([float(o.metadata.get("time", -1.0)) for o in occs],
                           dtype=np.float64)
        keep = nodes >= 0
        oids = [oid for oid, k in zip(oids, keep) if k]
        occs = [o for o, k in zip(occs, keep) if k]
        nodes = nodes[keep]
        times = times[keep]
        if len(nodes) == 0:
            return []

        found, valid, hit_time, hit_dst, hit_outcome = \
            self.future_index.query_batch(nodes, times)
        sel = found & valid
        n_valid = int(sel.sum())
        rows: List[CutRecord] = []

        if self.stage == NODE_CLASS:
            for occ, t, d, y in zip(np.asarray(occs)[sel],
                                    hit_time[sel], hit_dst[sel],
                                    hit_outcome[sel]):
                base_ctx = {
                    "delta_t": float(t - occ.metadata["time"]),
                    "counterpart": int(d),
                    "role": 0,
                    "query_type": 1,
                }
                rows.append(self._record(occ, occ_to_tree, base_ctx,
                                         outcome=float(y)))
            return rows

        # LINK: one positive row per cut + neg_per_cut negative rows; the
        # negative counterpart pool is drawn once, vectorized.
        rng = np.random.RandomState((self.seed * 1000003) ^ int(batch_seed))
        pool = self.future_index.unique_destinations
        cand = rng.choice(pool, size=n_valid * self.neg_per_cut, replace=True)
        k = 0
        for occ, t, d in zip(np.asarray(occs)[sel], hit_time[sel],
                             hit_dst[sel]):
            base_ctx = {
                "delta_t": float(t - occ.metadata["time"]),
                "counterpart": int(d),
                "role": 0,
                "query_type": 0,
            }
            rows.append(self._record(occ, occ_to_tree, dict(base_ctx),
                                     outcome=1.0))
            for _ in range(self.neg_per_cut):
                ctx = dict(base_ctx)
                ctx["counterpart"] = int(cand[k])
                k += 1
                rows.append(self._record(occ, occ_to_tree, ctx, outcome=0.0))
        return rows

    def _record(self, occ, occ_to_tree: Dict[int, int],
                context: Dict[str, Any], outcome: float) -> CutRecord:
        return CutRecord(
            tree_id=occ_to_tree.get(occ.occurrence_id, 0),
            cut_id=int(occ.occurrence_id),
            occurrence_id=int(occ.occurrence_id),
            tau=str(occ.tau),
            node=int(occ.metadata.get("node", -1)),
            time=float(occ.metadata.get("time", -1.0)),
            z=occ.state.z,
            context=context,
            outcome=outcome,
            valid=True)
