"""CCM host adapter: extract z_v = J_mem(M_v) from the merged SUM K/V.

The vendored attention calls ``mem_callback`` right after the merge (and
the Gamma scan, when attached) with the CURRENT sequence's merged
key/value states plus the sum masks.  This adapter keeps a per-layer
reference cache and assembles the cut memory on demand:

    z_v = JMemLift(concat_layers M_v)   (M_v = SUM-block K/V of turn v)

The cache holds references only (no copy): every batch consumes its own
forward graph before the next batch replaces it.  Pass 1 (no_grad)
consumes through ``KFMomentWindow.add`` (detached); pass 2 keeps the
graph connected z for the exact surrogate (plan L5).

The main experiment has NO witness: z_v is the only memory product.
"""

from typing import List, Optional

import torch

from rpbe.llm.mem_lift import JMemLift

# One cut samples the (S0, S1) SUM pair of its block.
SUM_PAIR = 2


class CCMHostAdapter:
    """Grab merged SUM K/V per layer and lift one cut's memory to z_v."""

    def __init__(self, model, *, n_layers: int, n_heads: int,
                 n_slots: int = 2, kv_pairs: int = 2, head_dim: int = 128,
                 z_dim: int = 128, seed: int = 0):
        from .ccm_patch import _base_model
        base = _base_model(model)
        if len(getattr(base, "layers", [])) != int(n_layers):
            raise ValueError("n_layers does not match the model")
        self.n_layers = int(n_layers)
        self.n_heads = int(n_heads)
        self.head_dim = int(head_dim)
        self.j_mem = JMemLift(n_layers=n_layers, n_heads=n_heads,
                              n_slots=n_slots, kv_pairs=kv_pairs,
                              head_dim=head_dim, z_dim=z_dim, seed=seed)
        self.z_dim = int(z_dim)
        self._cache: List[Optional[tuple]] = [None] * self.n_layers

        def make_cb(index):
            def cb(key_states, value_states, sum_mask, sum_count):
                self._cache[index] = (key_states, value_states, sum_mask,
                                      sum_count)
            return cb

        for i, layer in enumerate(base.layers):
            layer.self_attn.mem_callback = make_cb(i)

    def clear(self) -> None:
        """Drop the cached forward-graph references (batch/epoch drain)."""
        self._cache = [None] * self.n_layers

    def extract_z(self, sum_positions: torch.Tensor) -> torch.Tensor:
        """Lift the memory of the cut's SUM block to z_v.

        Args:
            sum_positions: [B, 2] sequence positions of the cut block's
                (S0, S1) tokens (per batch row; the collator metadata
                provides these after padding).

        Returns:
            z_v [B, z_dim], gradient-connected to the merged K/V (and
            therefore to Gamma and the backbone projections).
        """
        if sum_positions.dim() != 2 or sum_positions.shape[1] != SUM_PAIR:
            raise ValueError("sum_positions must be [B, 2]")
        k_parts: List[torch.Tensor] = []
        v_parts: List[torch.Tensor] = []
        for entry in self._cache:
            if entry is None:
                raise RuntimeError(
                    "memory cache empty: run the model forward first "
                    "(or a batch had no SUM tokens)")
            k, v, _smask, _scount = entry
            idx = sum_positions.unsqueeze(1).expand(-1, k.shape[1], -1)
            idx = idx.unsqueeze(-1).expand(-1, -1, -1, k.shape[-1])
            k_parts.append(torch.gather(k, 2, idx))
            v_parts.append(torch.gather(v, 2, idx))
        mem = JMemLift.pack_sum_mem(k_parts, v_parts)
        return self.j_mem(mem)

    def detach_mem_lift(self) -> None:
        """J_mem is fixed by construction; nothing to detach."""
