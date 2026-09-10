"""Joint child-parent boundary fixed maps (paper recursive-closure).

Single-row-per-pair measurement.  For one consumed pair the row is::

    p_v = TS_ctx( [1; chi(C_v)]  (x)  phi_S )

where ``chi(C_v)`` encodes only the structural context known at the cut
(child/parent layers, relation type, relation lag, neighbor slot, path/depth,
root src/dst role) and ``phi_S`` is the joint child-parent boundary::

    phi_S = TS_pair( [1; phi_E(Y_v)]  (x)  [1; m_p; m_p * phi_E(Y_p(v))] )

with ``m_p`` the parent-mask (0 for 1Obs, 1 for 2Obs), and for one real
event::

    phi_E(Y) = [ hash(counterpart, role); RFF(log1p(delta_t));
                 fixed_projection(message) ]

Everything is frozen: hashes / RFF / sketches come from an independent seed
(never shared with the context tables in rpbe.maps.FixedMaps, so the event
identity is not double-counted as context).  ``event_id`` is never used as a
numeric input — only for audit / multiset hash.

Geometry:
    d_E  = d_hash + d_rff + d_msg
    phi_S has a FIXED output dimension across arms (the pair tensor sketch
    dim), and p_v a FIXED output dimension ``m``.  1Obs (m_p=0) and 2Obs
    (m_p=1) therefore share identical dimensions.

No gradient flows through any buffer here.
"""

import hashlib
import math

import numpy as np
import torch
from torch import nn

# seed offsets for the independent measurement stream
SEED_BASE = 1000


def _fixed_binary(shape, seed, device=None, dtype=None):
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    return torch.randint(0, 2, shape, generator=g,
                         dtype=torch.int64).to(device=device) * 2 - 1


class BoundaryMaps(nn.Module):
    """Frozen single-row-per-pair measurement for the pair KF loss."""

    def __init__(self, *, d_ctx: int, d_event: int, m: int, d_msg: int,
                 delta_t_scale: float, msg_scale: float = 1e-3,
                 num_counter_bins: int = 4096, m_p_cap: int = 1,
                 seed: int = 0):
        super().__init__()
        self.d_ctx = int(d_ctx)
        self.d_event = int(d_event)
        self.m = int(m)
        self.d_msg = int(d_msg)
        self.delta_t_scale = float(delta_t_scale)
        self.msg_scale = float(msg_scale)
        self.num_counter_bins = int(num_counter_bins)
        self.seed = int(seed)
        self.m_p_cap = 1.0  # parent mask value used at 2Obs

        # partition d_event into hash / rff / msg blocks
        d_msg = min(self.d_event // 3, self.d_msg)
        rem = self.d_event - d_msg
        d_rff = rem // 2
        d_hash = rem - d_rff
        self.d_hash = d_hash
        self.d_rff = d_rff
        # message block width = whatever remains so the sum is exactly d_event
        self.d_msg_pad = self.d_event - (d_hash + d_rff)
        self.d_msg_use = min(d_msg, self.d_msg_pad)

        s = int(seed)
        # counterpart+role hash -> d_hash (row = 2*role*num_bins? use two
        # blocks: role 0 -> [0,bins), role 1 -> [bins, 2bins))
        self.register_buffer("event_hash_table", _fixed_binary(
            (2 * self.num_counter_bins, self.d_hash), SEED_BASE + s + 1),
            persistent=True)
        # RFF for log1p(delta_t)
        self.register_buffer("rff_w", torch.randn(
            (1, self.d_rff),
            generator=torch.Generator().manual_seed(SEED_BASE + s + 2)),
            persistent=True)
        self.register_buffer("rff_b", torch.rand(
            (self.d_rff,),
            generator=torch.Generator().manual_seed(SEED_BASE + s + 3)) * 6.2832,
            persistent=True)
        # fixed message projection: d_msg_pad x d_event_pad? We project the
        # full message to d_msg_pad using a fixed ±1 matrix.
        self.register_buffer("msg_proj", _fixed_binary(
            (self.d_msg_pad, self.d_msg), SEED_BASE + s + 4),
            persistent=True)

        # context sketch: d_ctx (chi) -> p_v
        # Pair tensor sketch geometry: TS_pair maps
        #   [(1+d_event) * (2 + d_event)] -> d_pair
        # then TS_ctx maps [(1 + d_ctx) * d_pair] -> m.
        # We implement as two sparse sketches with count-sketch guarantees.
        self.d_pair = max(64, int((1 + self.d_event) * 2))  # upper bound
        # actually choose a fixed smaller pair dim for memory
        self.d_pair = int(2 * (1 + self.d_event))
        self._build_pair_sketch(SEED_BASE + s + 5)
        self._build_ctx_sketch(SEED_BASE + s + 6)

    # ------------------------------------------------------------ sketches
    def _count_sketch(self, full_dim, out_dim, seed):
        """(rows_idx, col_idx, sign) covering every coordinate at least once."""
        g = torch.Generator().manual_seed(int(seed))
        base_rows = torch.arange(full_dim)
        base_cols = torch.arange(full_dim) % out_dim
        base_signs = torch.ones(full_dim, dtype=torch.int64)
        k = 2
        extra_n = (k - 1) * full_dim
        extra_rows = torch.randint(0, full_dim, (extra_n,), generator=g)
        extra_cols = torch.randint(0, out_dim, (extra_n,), generator=g)
        extra_signs = torch.randint(0, 2, (extra_n,), generator=g) * 2 - 1
        return (torch.cat([base_rows, extra_rows]),
                torch.cat([base_cols, extra_cols]),
                torch.cat([base_signs, extra_signs]))

    def _build_pair_sketch(self, seed):
        full_pair = (1 + self.d_event) * (2 + self.d_event)
        self._pair_full = full_pair
        rows, cols, signs = self._count_sketch(full_pair, self.d_pair, seed)
        self.register_buffer("pair_rows", rows, persistent=True)
        self.register_buffer("pair_cols", cols, persistent=True)
        self.register_buffer("pair_signs", signs, persistent=True)

    def _build_ctx_sketch(self, seed):
        full_ctx = (1 + self.d_ctx) * self.d_pair
        self._ctx_full = full_ctx
        rows, cols, signs = self._count_sketch(full_ctx, self.m, seed)
        self.register_buffer("ctx_rows", rows, persistent=True)
        self.register_buffer("ctx_cols", cols, persistent=True)
        self.register_buffer("ctx_signs", signs, persistent=True)

    # ------------------------------------------------------- event feature
    def event_vector(self, counterpart: int, role: int, delta_t: float,
                     message: torch.Tensor) -> torch.Tensor:
        """phi_E(Y) for one real event -> [d_event], frozen."""
        with torch.no_grad():
            cp = int(counterpart) % self.num_counter_bins
            ro = 1 if int(role) > 0 else 0
            h = self.event_hash_table[self.num_counter_bins * ro + cp]
            dt = math.log1p(max(float(delta_t), 0.0)) / self.delta_t_scale
            t = torch.tensor(dt, dtype=self.rff_w.dtype,
                             device=self.rff_w.device).reshape(1, 1)
            rff = torch.cos(t @ self.rff_w + self.rff_b)[0]
            msg = message.to(self.rff_w.dtype).reshape(-1)
            m_proj = (self.msg_proj.to(self.rff_w.dtype) @ msg) * self.msg_scale
            m_use = m_proj[:self.d_msg_use]
            pad = torch.zeros(self.d_msg_pad - self.d_msg_use,
                              dtype=self.rff_w.dtype,
                              device=self.rff_w.device)
            m_full = torch.cat([m_use, pad], dim=-1)
            return torch.cat([h.to(self.rff_w.dtype), rff, m_full], dim=-1)

    # ------------------------------------------------------- boundary p_v
    def _sketch_tensor(self, body, rows, cols, signs, out_dim, dev, dtype):
        """sparse scatter of ``body`` (1d) via (rows, cols, signs)."""
        out = torch.zeros(out_dim, dtype=dtype, device=dev)
        out.index_add_(0, cols,
                       body[rows] * signs.to(dtype))
        return out

    def pv(self, ctx_vec: torch.Tensor, child_event, parent_event,
           use_parent: int, msg_cache: dict = None) -> torch.Tensor:
        """One pair row -> [m].

        ``ctx_vec`` is chi(C_v) [d_ctx] (structural context only).
        ``child_event`` / ``parent_event`` are dicts or objects exposing
        ``counterpart``, ``role``, ``delta_t`` (event.time - cut time), and
        ``message`` (a 1-d tensor of the fixed edge message).
        """
        with torch.no_grad():
            c = ctx_vec.to(self.rff_w.dtype).reshape(-1)
            fe_c = self.event_vector(
                child_event["counterpart"], child_event["role"],
                child_event["delta_t"], child_event["message"])
            if use_parent:
                fe_p = self.event_vector(
                    parent_event["counterpart"], parent_event["role"],
                    parent_event["delta_t"], parent_event["message"])
                right = torch.cat([torch.ones(1, dtype=fe_c.dtype,
                                              device=fe_c.device),
                                   torch.tensor([self.m_p_cap],
                                                dtype=fe_c.dtype,
                                                device=fe_c.device),
                                   self.m_p_cap * fe_p])
            else:
                right = torch.cat([torch.ones(1, dtype=fe_c.dtype,
                                              device=fe_c.device),
                                   torch.zeros(1, dtype=fe_c.dtype,
                                               device=fe_c.device),
                                   torch.zeros_like(fe_c)])
            left = torch.cat([torch.ones(1, dtype=fe_c.dtype,
                                         device=fe_c.device), fe_c])
            prod = torch.outer(left, right).reshape(-1)
            s_pair = self._sketch_tensor(
                prod, self.pair_rows, self.pair_cols, self.pair_signs,
                self.d_pair, fe_c.device, fe_c.dtype)
            ctx_body = torch.cat([torch.ones(1, dtype=c.dtype,
                                             device=c.device), c])
            ctx_prod = torch.outer(ctx_body, s_pair).reshape(-1)
            p = self._sketch_tensor(
                ctx_prod, self.ctx_rows, self.ctx_cols, self.ctx_signs,
                self.m, c.device, c.dtype)
            return p.float()

    def isolation_fingerprint(self) -> dict:
        h = hashlib.sha256()
        h.update(str(self.seed).encode())
        h.update(str(self.m).encode())
        h.update(str(self.d_event).encode())
        h.update(str(self.delta_t_scale).encode())
        for name in ("event_hash_table", "rff_w", "rff_b", "msg_proj",
                     "pair_rows", "pair_cols", "pair_signs",
                     "ctx_rows", "ctx_cols", "ctx_signs"):
            buf = getattr(self, name)
            h.update(buf.detach().cpu().reshape(-1).contiguous()
                     .numpy().tobytes())
        return {"seed": int(self.seed),
                "d_event": int(self.d_event),
                "d_ctx": int(self.d_ctx),
                "m": int(self.m),
                "sha256": h.hexdigest()}
