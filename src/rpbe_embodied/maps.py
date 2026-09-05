"""rpbe_embodied.maps — EmbodiedFixedMaps.

Fixed measurement P = psi(C, Y) for embodied samples (plan §14/§15):
  C = (frozen vision raw features, instruction hash, delta_s, horizon)
  Y = future action chunk A*_{s} [112] (normalized)
  P = CountSketch([1; phi_C(C)] tensor phi_Y(Y))

All tables are generated ONCE from cfg.rpbe_seed and registered as
persistent buffers -- P is bit-identical across training regardless of
LoRA/Gamma parameter updates (fixedness contract, Task 6 gate).

CountSketch construction copies the TGN pattern (src/rpbe/maps.py):
base coverage of every coordinate + (k-1) random repeats, stored as
[2, nnz] indices + signs, batched via index_add_.
"""
from __future__ import annotations

import hashlib
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .config import EmbodiedRPBConfig

_VISION_RAW_DIM = 2176        # DINO(1024) + SigLIP(1152) fused patches
_ACTION_DIM = 112             # 16 x 7 flattened future chunk


class EmbodiedFixedMaps(nn.Module):
    def __init__(self, cfg: EmbodiedRPBConfig):
        super().__init__()
        self.cfg = cfg
        self.d_c = cfg.d_c
        self.d_cv = cfg.d_cv
        self.d_f = cfg.d_f
        self.m = cfg.m

        seed = cfg.rpbe_seed
        g = torch.Generator().manual_seed(seed)

        # --- vision fixed projection: 2176 -> d_cv (random ±1/sqrt) ---
        self.register_buffer(
            "vision_proj",
            (torch.randn(_VISION_RAW_DIM, cfg.d_cv, generator=g)
             / (_VISION_RAW_DIM ** 0.5)),
            persistent=True)

        # --- instruction categorical bins (±1 table) ---
        self.instr_bins = cfg.num_counter_bins
        self.instr_dim = cfg.d_c - cfg.d_cv - 4 - 4      # 24
        self.register_buffer(
            "instr_table",
            (2 * torch.randint(0, 2, (self.instr_bins + 1, self.instr_dim),
                               generator=g) - 1).float(),
            persistent=True)

        # --- delta_s RFF (4 dims) ---
        rff_w = torch.randn(4, generator=g) * cfg.delta_s_scale
        rff_b = torch.rand(4, generator=g) * 2 * np.pi
        self.register_buffer("ds_rff_w", rff_w.float(), persistent=True)
        self.register_buffer("ds_rff_b", rff_b.float(), persistent=True)

        # --- horizon one-hot (4 dims, h in {1,2}) ---
        self.register_buffer(
            "horizon_table",
            torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=torch.float32),
            persistent=True)

        # --- action RFF: [112] -> d_f ---
        rff_y_w = torch.randn(cfg.d_f, _ACTION_DIM, generator=g) * cfg.action_rff_scale
        rff_y_b = torch.rand(cfg.d_f, generator=g) * 2 * np.pi
        self.register_buffer("rff_y_w", rff_y_w.float(), persistent=True)
        self.register_buffer("rff_y_b", rff_y_b.float(), persistent=True)

        # --- CountSketch: k=10, full_dim = (d_c+1)*d_f, out m ---
        k = 10
        full_dim = (self.d_c + 1) * self.d_f
        base_rows = torch.arange(full_dim, dtype=torch.int64)
        base_cols = base_rows % self.m
        base_signs = torch.ones(full_dim)
        extra_rows = torch.randint(0, full_dim, ((k - 1) * full_dim,),
                                   generator=g, dtype=torch.int64)
        extra_cols = torch.randint(0, self.m, ((k - 1) * full_dim,),
                                   generator=g, dtype=torch.int64)
        extra_signs = (2 * torch.randint(0, 2, ((k - 1) * full_dim,),
                                         generator=g) - 1).float()
        self.register_buffer(
            "sketch_indices",
            torch.stack([torch.cat([base_rows, extra_rows]),
                         torch.cat([base_cols, extra_cols])]),
            persistent=True)
        self.register_buffer(
            "sketch_signs",
            torch.cat([base_signs, extra_signs]).float(),
            persistent=True)
        self._sketch_full_dim = full_dim

    def _instruction_vec(self, instruction: str) -> torch.Tensor:
        h = int(hashlib.sha256(instruction.encode()).hexdigest()[:16], 16)
        return self.instr_table[h % self.instr_bins]          # [instr_dim]

    def _context_vector(self, context: Dict) -> torch.Tensor:
        """C -> [d_c] fixed feature vector."""
        vf = context["vision_feat"].float().flatten()          # [2176]
        v_proj = vf @ self.vision_proj                          # [d_cv]
        i_vec = self._instruction_vec(context["instruction"])
        ds = float(context["delta_s"])
        ds_rff = torch.cos(self.ds_rff_w * ds + self.ds_rff_b)  # [4]
        h = int(context["horizon"])
        h_vec = self.horizon_table[h - 1]                       # [4]
        return torch.cat([v_proj, i_vec, ds_rff, h_vec])        # [d_c]

    def _future_vector(self, y: torch.Tensor) -> torch.Tensor:
        """Y [112] -> [d_f] fixed RFF cosine features."""
        yf = y.float().flatten()
        return torch.cos(yf @ self.rff_y_w.T + self.rff_y_b)    # [d_f]

    @torch.no_grad()
    def pv(self, context: Dict, y: torch.Tensor) -> torch.Tensor:
        """Single row P = psi(C, Y) [m]."""
        c = self._context_vector(context)
        f = self._future_vector(y)
        body = torch.cat([torch.ones(1, device=f.device), c])   # [1+d_c]
        prod = torch.outer(body, f).reshape(-1)                 # [full_dim]
        out = torch.zeros(self.m, device=f.device)
        out.index_add_(0, self.sketch_indices[1],
                       prod[self.sketch_indices[0]] * self.sketch_signs)
        return out

    @torch.no_grad()
    def pv_batch(self, contexts: list, ys: torch.Tensor) -> torch.Tensor:
        """[N, m]; loops over pv() for BIT-identical rows (fixedness
        contract: the same (C, Y) must map to the same P everywhere)."""
        return torch.stack([self.pv(c, ys[i]) for i, c in enumerate(contexts)])

    def fingerprint(self) -> str:
        h = hashlib.sha256()
        h.update(str(self.cfg.rpbe_seed).encode())
        for name, buf in self.named_buffers():
            h.update(name.encode())
            h.update(buf.cpu().numpy().tobytes())
        return h.hexdigest()[:16]
