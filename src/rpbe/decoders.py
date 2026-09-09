"""decoders.py — A1 auxiliary decoders (Reconstruction / LocalPred).

Spec §5 (A1 信息保存原则): two supervised baselines against the RPBE
statistical objective, sharing the SAME cuts, SAME context C_v and SAME
training protocol as ours (aligned 2Obs).

* ``ReconDecoder``  — D_recon(z, C_v) -> pre-compression child state u
  (normalized MSE, one contribution per canonical valid cut).
* ``LocalPredDecoder`` — D_pred(z, C_v) -> fixed future witness p_v
  (normalized MSE against sg(p_v); p_v built by the SAME fixed BoundaryMaps
  as ours, aligned 2Obs, m_p=1).

Both decoders are fixed small MLPs with IDENTICAL width/depth (spec §5.2:
"hidden width 与 LocalPred decoder 匹配").  They are train-time only; no
decoder touches the eval path.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class _DecoderBase(nn.Module):
    def __init__(self, d_z: int, d_c: int, d_out: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(d_z) + int(d_c), int(hidden)),
            nn.GELU(),
            nn.Linear(int(hidden), int(hidden)),
            nn.GELU(),
            nn.Linear(int(hidden), int(d_out)),
        )
        self.d_out = int(d_out)

    def forward(self, z: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, ctx], dim=-1))


class ReconDecoder(_DecoderBase):
    """Reconstruct the pre-compression child state from the compressed one."""

    def __init__(self, d_z: int, d_c: int, d_u: int, hidden: int):
        super().__init__(d_z, d_c, d_u, hidden)


class LocalPredDecoder(_DecoderBase):
    """Directly predict the fixed future witness p_v."""

    def __init__(self, d_z: int, d_c: int, d_p: int, hidden: int):
        super().__init__(d_z, d_c, d_p, hidden)
