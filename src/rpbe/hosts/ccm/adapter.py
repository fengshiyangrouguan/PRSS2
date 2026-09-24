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
                 z_dim: int = 128, seed: int = 0, per_layer_dims=None,
                 unique_layer_ids=None):
        from .ccm_patch import _base_model
        base = _base_model(model)
        if len(getattr(base, "layers", [])) != int(n_layers):
            raise ValueError("n_layers does not match the model")
        self.n_layers = int(n_layers)
        self.n_heads = int(n_heads)
        self.n_slots = int(n_slots)
        self.head_dim = int(head_dim)
        # Unique-provider measurement (review 2026-09-24): KV-sharing
        # hosts (gemma4) reference ONE physical K/V from several logical
        # layers — sketching every alias counts the same state multiple
        # times and distorts the predictive gradient (norm AND
        # direction).  unique_layer_ids restricts extract_z to the
        # physical providers; None = every layer (llama/qwen3).
        if unique_layer_ids is not None:
            self.unique_ids = [int(x) for x in unique_layer_ids]
            self.n_used = len(self.unique_ids)
        else:
            self.unique_ids = None
            self.n_used = int(n_layers)
        _pld = per_layer_dims
        if _pld is not None and self.unique_ids is not None:
            _pld = [_pld[i] for i in self.unique_ids]
        # The CountSketch index tables (2.1M entries x 3) must live on the
        # model device: this class is NOT an nn.Module, so nothing moves
        # them otherwise and every extract_z() would re-upload them from
        # CPU (L6.5 review performance fix).
        device = next(base.parameters()).device
        self.j_mem = JMemLift(n_layers=self.n_used, n_heads=n_heads,
                              n_slots=n_slots, kv_pairs=kv_pairs,
                              head_dim=head_dim, z_dim=z_dim,
                              seed=seed,
                              per_layer_dims=_pld).to(device)
        self.z_dim = int(z_dim)
        self._cache: List[Optional[tuple]] = [None] * self.n_layers

        def make_cb(index):
            def cb(key_states, value_states, sum_mask, sum_row_pos):
                self._cache[index] = (key_states, value_states, sum_mask,
                                      sum_row_pos)
            return cb

        for i, layer in enumerate(base.layers):
            layer.self_attn.mem_callback = make_cb(i)

    def clear(self) -> None:
        """Drop the cached forward-graph references (batch/epoch drain)."""
        self._cache = [None] * self.n_layers

    def extract_z(self, sum_positions: torch.Tensor) -> torch.Tensor:
        """Lift the memory of the cut's SUM block to z_v.

        Args:
            sum_positions: [B, n_slots] sequence positions of the cut
                block's SUM tokens (per batch row; the collator metadata
                provides these after padding).  n_slots = 2 for the
                dialog line, 4 for the LaMP one-shot merge.

        Returns:
            z_v [B, z_dim], gradient-connected to the merged K/V (and
            therefore to Gamma and the backbone projections).
        """
        if sum_positions.dim() != 2 or sum_positions.shape[1] != self.n_slots:
            raise ValueError("sum_positions must be [B, {}]".format(
                self.n_slots))
        k_parts: List[torch.Tensor] = []
        v_parts: List[torch.Tensor] = []
        _layers = (self.unique_ids if self.unique_ids is not None
                   else range(self.n_layers))
        for _li in _layers:
            entry = self._cache[_li]
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
