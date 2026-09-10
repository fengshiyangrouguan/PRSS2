"""Train-only future-event index and aligned child-parent boundary records
for the TGB link-prediction recursive-closure experiment.

Semantics (paper):
    occurrence(n, t) -> Y(n,t) = the FIRST real train incident event of node
    ``n`` with time > t.

For a real consumed pair (child, parent) recorded by the adapter:
    Y_v    = Y(child_node, child_time)
    Y_p(v) = Y(parent_node, parent_time)
    Y_v^(2) := Y_p(v)^(1)

Both futures come from the TRAIN positive stream only.  Negatives never enter
Y_v or any PRBE statistic.  A pair whose child or parent has no legal strict
future is dropped from the shared candidate set (all arms identically).

Every event's ``message`` is the fixed 172-d TGB msg (row 0 never used here;
internal edge id = global row + 1, so message index = edge_id - 1).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


@dataclass(frozen=True)
class ObservedLinkEvent:
    """One real positive train event viewed as a future outcome of a node."""

    event_id: int          # internal edge id (row+1), used for audit/hash
    time: float
    src: int               # internal (+1) node id
    dst: int
    counterpart: int       # the other endpoint relative to the queried node
    role: int              # 0 = queried node is src, 1 = queried node is dst
    message_idx: int       # index into the edge feature table (row+1)


@dataclass
class BoundaryRecord:
    """One supervised child-parent boundary (single p row in the KF window)."""

    pair_id: tuple
    root_row: int
    tau: str
    child_layer: int
    parent_layer: int
    child_node: int
    child_time: float
    parent_node: int
    parent_time: float
    relation_time: float
    relation_lag: float
    relation_slot: int
    path: Tuple[int, float]
    z: torch.Tensor                       # child compressed state
    child_future: Optional[ObservedLinkEvent]
    parent_future: Optional[ObservedLinkEvent]
    weight: float = 1.0
    u: Optional[torch.Tensor] = None      # child pre-Gamma aggregate (P1)

    @property
    def valid(self) -> bool:
        return self.child_future is not None and self.parent_future is not None

    @property
    def boundary_key(self):
        return (self.pair_id,)


class LinkFutureIndex:
    """Per-node chronological lookup over the train positive stream.

    ``train`` is a JodieData-like stream (internal +1 ids, timestamps,
    edge_idxs); ``sources``/``destinations`` give both endpoint roles for
    every event, so both nodes see the same event as a future.
    """

    def __init__(self, sources, destinations, timestamps, edge_idxs):
        by_node: Dict[int, List[Tuple[float, int, ObservedLinkEvent]]] = {}
        n = len(sources)
        for pos in range(n):
            src_i = int(sources[pos])
            dst_i = int(destinations[pos])
            time_f = float(timestamps[pos])
            eid = int(edge_idxs[pos])
            # queried node is the source -> counterpart is dst, role 0
            by_node.setdefault(src_i, []).append(
                (time_f, pos, ObservedLinkEvent(
                    event_id=eid, time=time_f, src=src_i, dst=dst_i,
                    counterpart=dst_i, role=0, message_idx=eid)))
            by_node.setdefault(dst_i, []).append(
                (time_f, pos, ObservedLinkEvent(
                    event_id=eid, time=time_f, src=src_i, dst=dst_i,
                    counterpart=src_i, role=1, message_idx=eid)))
        self._events: Dict[int, Tuple[ObservedLinkEvent, ...]] = {}
        self._times: Dict[int, np.ndarray] = {}
        for node, rows in by_node.items():
            rows.sort(key=lambda x: (x[0], x[1]))
            self._events[node] = tuple(x[2] for x in rows)
            self._times[node] = np.asarray([x[0] for x in rows],
                                           dtype=np.float64)
        self.n_events = n

    def query(self, node: int, cut_time: float) -> Optional[ObservedLinkEvent]:
        """First real train incident event of ``node`` strictly after cut."""
        times = self._times.get(int(node))
        if times is None:
            return None
        start = int(np.searchsorted(times, float(cut_time), side="right"))
        if start >= len(times):
            return None
        ev = self._events[int(node)][start]
        if ev.time <= float(cut_time):
            raise AssertionError("future index returned a non-future event")
        return ev


def build_boundary_records(pairs, future_index: LinkFutureIndex,
                           *, weight: float = 1.0) -> List[BoundaryRecord]:
    """Map adapter ConsumedPairCandidates to BoundaryRecords.

    Drops any pair whose child or parent has no legal strict future, so the
    surviving candidate set is identical across the structural arms.
    """
    out = []
    for p in pairs:
        cf = future_index.query(p.child_node, p.child_time)
        pf = future_index.query(p.parent_node, p.parent_time)
        rec = BoundaryRecord(
            pair_id=p.pair_id, root_row=p.root_row, tau=p.tau,
            child_layer=p.child_layer, parent_layer=p.parent_layer,
            child_node=p.child_node, child_time=p.child_time,
            parent_node=p.parent_node, parent_time=p.parent_time,
            relation_time=p.relation_time, relation_lag=p.relation_lag,
            relation_slot=p.relation_slot, path=p.path, z=p.z,
            child_future=cf, parent_future=pf, weight=weight,
            u=getattr(p, "u", None))
        if rec.valid:
            out.append(rec)
    return out
