"""J_mem: the fixed lift from the SUM-token K/V memory into z space.

The CCM merge memory M_v is the pair (K, V) of every SUM token slot at
every layer/head at cut turn v: ``n_layers * n_heads * n_slots * 2``
vectors of ``head_dim`` scalars (524288 scalars for the 7B backbone).
J_mem compresses that fixed layout into ``z_dim`` coordinates with a
structured CountSketch: every input coordinate is mapped at least once to
a z bin with a fixed +-1 sign (same construction guarantee as the TGN
FixedMaps), plus repetitions drawn from a fixed generator.

Everything here is FROZEN (no learnable parameters; all tables are
buffers from ``seed``).  The INPUT carries gradient — z_v must back-
propagate through the merge (and therefore through Gamma) in pass 2 —
but the measurement itself never trains.
"""

import torch
from torch import nn


def _mem_layout(n_layers, n_heads, n_slots, kv_pairs, head_dim):
    """Flat coordinate order: (layer, head, slot, kv, dim), kv 0=K 1=V."""
    return int(n_layers) * int(n_heads) * int(n_slots) * int(kv_pairs) \
        * int(head_dim)


class JMemLift(nn.Module):
    """z_v = CountSketch(M_v); fixed measurement, differentiable input."""

    def __init__(self, *, n_layers: int, n_heads: int, n_slots: int = 2,
                 kv_pairs: int = 2, head_dim: int = 128, z_dim: int = 128,
                 seed: int = 0, repeats: int = 3):
        super().__init__()
        self.n_layers = int(n_layers)
        self.n_heads = int(n_heads)
        self.n_slots = int(n_slots)
        self.kv_pairs = int(kv_pairs)
        self.head_dim = int(head_dim)
        self.z_dim = int(z_dim)
        self.full_dim = _mem_layout(n_layers, n_heads, n_slots, kv_pairs,
                                    head_dim)
        # Base pass: every input coordinate maps at least once.
        rows = torch.arange(self.full_dim)
        cols = rows % self.z_dim
        signs = torch.ones(self.full_dim)
        # Repetitions from a fixed generator (no RNG state leakage).
        g = torch.Generator()
        g.manual_seed(int(seed))
        extra_n = int(repeats) * self.full_dim
        extra_rows = torch.randint(0, self.full_dim, (extra_n,), generator=g)
        extra_cols = torch.randint(0, self.z_dim, (extra_n,), generator=g)
        extra_signs = torch.randint(0, 2, (extra_n,), generator=g) * 2 - 1
        rows = torch.cat([rows, extra_rows])
        cols = torch.cat([cols, extra_cols])
        signs = torch.cat([signs, extra_signs])
        self.register_buffer("sketch_rows", rows, persistent=True)
        self.register_buffer("sketch_cols", cols, persistent=True)
        self.register_buffer("sketch_signs", signs, persistent=True)
        # Fixed scale: each bin averages full_dim/z_dim coordinates, so a
        # +-1 sum has magnitude ~sqrt of that; rescale to O(input scale).
        self.register_buffer(
            "scale", torch.tensor((self.z_dim / self.full_dim) ** 0.5),
            persistent=True)
        self._nnz = int(rows.numel())

    def forward(self, mem: torch.Tensor) -> torch.Tensor:
        """mem: [B, full_dim] in the (layer, head, slot, kv, dim) layout."""
        if mem.dim() != 2 or mem.shape[1] != self.full_dim:
            raise ValueError(
                "JMemLift expects [B, {}], got {}".format(
                    self.full_dim, tuple(mem.shape)))
        cols = self.sketch_cols.to(mem.device)
        rows = self.sketch_rows.to(mem.device)
        signs = self.sketch_signs.to(dtype=mem.dtype, device=mem.device)
        out = torch.zeros(mem.shape[0], self.z_dim, dtype=mem.dtype,
                          device=mem.device)
        out.index_add_(1, cols, mem[:, rows] * signs)
        return out * self.scale

    @staticmethod
    def pack_sum_mem(k_rows, v_rows):
        """Pack per-layer SUM-row K/V into the flat [B, full_dim] layout.

        Args:
            k_rows/v_rows: lists (one entry per layer) of
                [B, n_heads, n_sum_rows, head_dim] merged SUM K/V tensors.
                The list must be ordered by layer and every entry must
                have the same n_sum_rows; the v-th SUM block corresponds
                to index pair (2v, 2v+1) of that shared row axis.
        """
        B = k_rows[0].shape[0]
        H = k_rows[0].shape[1]
        parts = []
        for k, v in zip(k_rows, v_rows):
            # [B, H, R, D] -> [B, H, R, 2, D] (K then V)
            pair = torch.stack([k, v], dim=3)  # [B, H, R, 2, D]
            parts.append(pair.reshape(B, -1))
        return torch.cat(parts, dim=1)  # [B, full_dim]
