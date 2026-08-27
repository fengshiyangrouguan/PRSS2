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
        # Both roles are indexed: a node's future continuation includes the
        # interactions it initiates (role 0) and the ones it receives
        # (role 1) — the role is part of the context C.
        per_node: Dict[int, List] = {}
        for u, dst, t, y in zip(sources, destinations, timestamps, labels):
            per_node.setdefault(int(u), []).append(
                (float(t), int(dst), float(y), 0))
            per_node.setdefault(int(dst), []).append(
                (float(t), int(u), float(y), 1))
        self._events: Dict[int, np.ndarray] = {}
        for u, events in per_node.items():
            arr = np.asarray(events, dtype=np.float64)  # [n,4]: t,other,y,role
            arr = arr[np.argsort(arr[:, 0])]
            self._events[u] = arr
        # Stage-1 negative pool: train-region destinations only (no val/test
        # support leakage).
        ts = np.asarray(timestamps)
        self.neg_pool = np.unique(np.asarray(destinations)[ts <= self.val_time])

    def _split_of(self, t: float) -> str:
        if t <= self.val_time:
            return "train"
        if t <= self.test_time:
            return "val"
        return "test"

    def query(self, node: int, t: float) -> Optional[dict]:
        """First event of ``node`` (either role) after time ``t``.

        Returns ``{"time", "counterpart", "outcome", "role", "valid"}``;
        ``None`` when the node has no future event at all, ``valid=False``
        when the next event lies in val/test (censored — its content must not
        be read).
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
                    "outcome": None, "role": None, "valid": False}
        return {"time": float(nxt[0]), "counterpart": int(nxt[1]),
                "outcome": float(nxt[2]), "role": int(nxt[3]),
                "valid": True}


class JodieCutBuilder:
    """Turns an adapter trace into CutRecord rows for one training stage.

    Stage 1 (link): one positive row (the next real interaction, y=1) plus
    Stage 2 (node_class): one row per cut whose outcome is the
    ``state_label`` of the next real interaction.  Every cut carries ONE
    real continuation; fabricated negatives belong to the link TASK loss
    only and never enter the Ky Fan measurement.
    """

    def __init__(self, future_index: FutureIndex, *, stage: str,
                 cuts_per_tau: int = 32, seed: int = 0):
        if stage not in (LINK, NODE_CLASS):
            raise ValueError("unknown stage {}".format(stage))
        self.future_index = future_index
        self.stage = stage
        self.cuts_per_tau = int(cuts_per_tau)
        self.seed = int(seed)
        self._tree_counter = 0

    def build(self, trace, batch_seed: int = 0) -> List[CutRecord]:
        """One call per batch; ``batch_seed`` keeps negative sampling
        deterministic per global step (no internal state mutation)."""
        if trace is None or not trace.roots:
            return []
        rng = np.random.RandomState((self.seed * 1000003) ^ int(batch_seed))
        # Map occurrence -> its GLOBAL tree id (a per-builder counter, so
        # tree-level cross-fitting stays meaningful across batches).
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

        seen_pairs = set()
        per_tau_candidates: Dict[str, list] = {}
        for oid in trace.postorder():
            occ = trace.occurrences[oid]
            node = int(occ.metadata.get("node", -1))
            time = float(occ.metadata.get("time", -1.0))
            if node < 0:
                continue
            # Pseudo-replication guard: the same (node, as-of time) pair at
            # the SAME interface may appear in several trees of one batch;
            # keep a single cut for it (different taus are different cuts).
            pair = (node, time, occ.tau)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            hit = self.future_index.query(node, time)
            if hit is None:
                continue
            if not hit["valid"]:
                continue  # censored: never becomes a y=0 row
            base_ctx = {
                "delta_t": float(hit["time"] - time),
                "counterpart": int(hit["counterpart"]),
                "role": int(hit["role"]),       # 0 source / 1 destination
                "query_type": 0 if self.stage == LINK else 1,
            }
            # Ky Fan measurement uses REAL continuations only: exactly one
            # row per cut.  Stage-1 negatives belong to the link TASK loss
            # and never enter the KF moments (they are not observed futures).
            per_tau_candidates.setdefault(
                occ.tau, []).append(
                self._record(occ, occ_to_tree, base_ctx,
                             outcome=float(hit["outcome"])))

        # Depth-balanced sampling: every interface contributes the same
        # number of cuts per batch (uniform over the interface's real
        # occurrences), so the root layer is not starved by the leaf flood.
        rows = []
        for tau, cands in per_tau_candidates.items():
            if len(cands) <= self.cuts_per_tau:
                rows.extend(cands)
            else:
                idx = rng.choice(len(cands), size=self.cuts_per_tau,
                                 replace=False)
                rows.extend(cands[i] for i in sorted(idx))
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
