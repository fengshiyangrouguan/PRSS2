"""Gamma_theta -- the four-slot soft fusion (task book v4.1 §2.2).

Frozen formula, no terms added or removed:

    Q_v = Q_0 + B_c A_c q_emb                      # [4, 256]
    K   = X_v W_K,  V = X_v W_V,  g = sigmoid(X_v w_g)
    A_v = softmax_j( Q_v K^T / sqrt(d_e) + 1_4 log(g + eps)^T + M_v )
    Z_v = LN( Q_v + A_v V )

with low-rank W_K = A_K B_K, W_V = A_V B_V (h = 64) and low-rank
Cond(q) = B_c A_c q (cond_rank = 32). Parameter count is frozen at 108 288
(v4.1 §1.2) and asserted at construction.

M_v masks PADDING ONLY: every item is legal for every slot and the legal
bias is exactly 0. The four slots carry no predefined semantics.
"""

from __future__ import annotations

import torch
from torch import nn

from meta_n.rpbe import config as C


def _randn(shape, gen, scale):
    return torch.randn(*shape, generator=gen) * scale


class SlottedFusion(nn.Module):
    """Gamma_theta. Input X_v is FROZEN (no grad); q_emb is FROZEN."""

    def __init__(self, d_e: int = C.D_E, h: int = C.H_LORA,
                 n_slots: int = C.N_SLOTS, cond_rank: int = C.COND_RANK,
                 init_seed: int = C.GAMMA_INIT_SEED,
                 init_scale: float = C.GAMMA_INIT_SCALE):
        super().__init__()
        self.d_e = int(d_e)
        self.h = int(h)
        self.n_slots = int(n_slots)
        self.cond_rank = int(cond_rank)
        self.eps = 1e-8

        # One generator, drawn in a fixed order -> reproducible initialisation
        # (v4.1 §1.3).
        gen = torch.Generator().manual_seed(int(init_seed))

        # Cond(q) = B_c A_c q, low rank. nn.Linear convention: [out, in].
        self.A_c = nn.Parameter(_randn((cond_rank, d_e), gen, init_scale))
        self.B_c = nn.Parameter(
            _randn((n_slots * d_e, cond_rank), gen, init_scale))

        # Low-rank keys / values.
        self.A_K = nn.Parameter(_randn((d_e, h), gen, init_scale))
        self.B_K = nn.Parameter(_randn((h, d_e), gen, init_scale))
        self.A_V = nn.Parameter(_randn((d_e, h), gen, init_scale))
        self.B_V = nn.Parameter(_randn((h, d_e), gen, init_scale))

        # Item gate: exactly 0, so g = sigmoid(0) = 0.5 and log(g + eps) is a
        # constant across items (v4.1 §1.3).
        self.w_g = nn.Parameter(torch.zeros(d_e))

        # Slot queries.
        self.Q_0 = nn.Parameter(_randn((n_slots, d_e), gen, init_scale))

        # LayerNorm defaults are weight=1, bias=0 -- the frozen init.
        self.ln = nn.LayerNorm(d_e)

    # -- frozen low-rank factors -------------------------------------------
    @property
    def W_K(self) -> torch.Tensor:
        return self.A_K @ self.B_K                      # [d_e, d_e]

    @property
    def W_V(self) -> torch.Tensor:
        return self.A_V @ self.B_V                      # [d_e, d_e]

    def cond(self, q_emb: torch.Tensor) -> torch.Tensor:
        """Cond(q) = B_c A_c q -> [..., n_slots, d_e]."""
        z = q_emb @ self.A_c.t()                        # [..., cond_rank]
        z = z @ self.B_c.t()                            # [..., n_slots*d_e]
        return z.view(*q_emb.shape[:-1], self.n_slots, self.d_e)

    def queries(self, q_emb: torch.Tensor) -> torch.Tensor:
        """Q_v = Q_0 + Cond(q_emb) -> [..., n_slots, d_e]."""
        return self.Q_0 + self.cond(q_emb)

    # -- padding-only mask helper ------------------------------------------
    @staticmethod
    def pad_mask(valid: torch.Tensor, n_slots: int,
                 dtype=torch.float32) -> torch.Tensor:
        """Additive mask from a boolean `valid` (True = real item).

        `valid` [n_v] or [B, n_v] -> [n_slots, n_v] or [B, n_slots, n_v].
        Legal entries are exactly 0; padding is -inf.
        """
        valid = valid.bool()
        m = torch.zeros_like(valid, dtype=dtype)
        m = m.masked_fill(~valid, float("-inf"))
        shape = (1,) * (m.dim() - 1) + (n_slots,) + (m.shape[-1],)
        return m.unsqueeze(-2).expand(*m.shape[:-1], n_slots, m.shape[-1])

    # -- forward -----------------------------------------------------------
    def forward(self, X_v: torch.Tensor, q_emb: torch.Tensor,
                mask: torch.Tensor | None = None):
        """-> (Z_v [..., 4, d_e], A_v [..., 4, n_v]).

        X_v   : [n_v, d_e] or [B, n_v, d_e]  (frozen encoder output)
        q_emb : [d_e] or [B, d_e]
        mask  : optional additive float mask [n_slots, n_v] or [B, n_slots, n_v]
        """
        squeeze = X_v.dim() == 2
        if squeeze:
            X_v = X_v.unsqueeze(0)
            q_emb = q_emb.unsqueeze(0)
        if mask is not None and mask.dim() == 2:
            mask = mask.unsqueeze(0)

        K = X_v @ self.W_K                              # [B, n_v, d_e]
        V = X_v @ self.W_V
        g = torch.sigmoid(X_v @ self.w_g)               # [B, n_v]

        Q = self.queries(q_emb)                         # [B, 4, d_e]
        logits = (Q @ K.transpose(-1, -2)) / (self.d_e ** 0.5)   # [B, 4, n_v]
        logits = logits + torch.log(g + self.eps).unsqueeze(1)
        if mask is not None:
            logits = logits + mask.to(logits.dtype)

        A_v = torch.softmax(logits, dim=-1)             # [B, 4, n_v]
        Z_v = self.ln(Q + A_v @ V)                      # [B, 4, d_e]

        if squeeze:
            return Z_v[0], A_v[0]
        return Z_v, A_v
