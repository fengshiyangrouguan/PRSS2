"""Pair-row seam: BoundaryRecord -> (ctx C_v, child/parent event) -> p_v.

C_v is the structural context known at the cut (child/parent layer, relation
lag, neighbor slot, coarse path/depth, root role).  Future fields
(counterpart / delta_t / message / event id) NEVER enter C_v — they belong to
Y_v / Y_p(v).  For each record one p row is produced through BoundaryMaps.
"""

import numpy as np
import torch


def build_ctx_vector(rec, d_ctx: int, device=None) -> torch.Tensor:
    """Fixed structural-context chi(C_v) -> [d_ctx] (frozen, deterministic).

    Encodes only cut-time-known recursion structure via a stable bin index
    folded into a fixed pseudo-random direction; no future fields.
    """
    bin_rel_lag = int(np.clip(float(rec.relation_lag), 0, 63)) % 8
    sid = (int(rec.child_layer) * 1009 + int(rec.parent_layer) * 1003
           + int(rec.relation_slot) * 997 + bin_rel_lag * 991
           + len(rec.path) * 983) % (2 ** 31)
    # CPU generator for determinism; result moved to device after.
    base = torch.randn(d_ctx, generator=torch.Generator().manual_seed(0))
    dirv = torch.randn(d_ctx,
                       generator=torch.Generator().manual_seed(int(sid)))
    out = base + 0.5 * torch.tanh(dirv)
    if device is not None:
        out = out.to(device)
    return out


class PairRowProjector:
    """Builds one p_v per BoundaryRecord via BoundaryMaps.

    ``edge_table`` is the fixed float edge-feature matrix (internal edge id =
    row index; row 0 = padding).  ``maps`` is a ``BoundaryMaps``.
    """

    def __init__(self, maps, edge_table, *, use_parent: int = 1,
                 d_ctx: int = 32):
        self.maps = maps
        self.edge_table = edge_table
        self.use_parent = int(use_parent)
        self.d_ctx = int(d_ctx)

    def _message(self, edge_id):
        dev = self.maps.rff_w.device
        row = self.edge_table[int(edge_id)]
        return torch.as_tensor(row, dtype=torch.float32, device=dev)

    def _event(self, future, cut_time):
        return {
            "counterpart": future.counterpart,
            "role": future.role,
            "delta_t": float(future.time) - float(cut_time),
            "message": self._message(future.message_idx),
        }

    def pv_row(self, rec):
        """One p_v for one BoundaryRecord."""
        ctx = build_ctx_vector(rec, self.d_ctx, device=self.maps.rff_w.device)
        child = self._event(rec.child_future, rec.child_time)
        parent = self._event(rec.parent_future, rec.parent_time)
        return self.maps.pv(ctx, child, parent,
                            use_parent=self.use_parent)
