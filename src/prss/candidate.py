"""Host-agnostic candidate builder: vanilla output plus a learned residual."""

import torch
from torch import nn


class GenericResidualCandidateBuilder(nn.Module):
    """h = [vanilla_output; phi(flat_preagg)] with the first host_dim coordinates untouched.

    The host adapter packs its own exact aggregate inputs into ``flat_preagg``; the core
    builder never sees host-specific tensor layouts.  Because the first ``host_dim``
    coordinates are the untouched host state, the identity-like initialization
    R=[I,0] reproduces the vanilla host forward exactly.
    """

    def __init__(self, host_dim: int, preagg_dim: int, candidate_dim: int, hidden_dim: int = 128):
        super().__init__()
        if candidate_dim < host_dim:
            raise ValueError("candidate_dim must be >= host_dim")
        self.host_dim = int(host_dim)
        self.preagg_dim = int(preagg_dim)
        self.candidate_dim = int(candidate_dim)
        self.residual_dim = self.candidate_dim - self.host_dim
        if self.residual_dim > 0:
            self.encoder = nn.Sequential(
                nn.Linear(self.preagg_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, self.residual_dim),
                nn.LayerNorm(self.residual_dim),
            )
        else:
            self.encoder = None

    def forward(self, vanilla_output: torch.Tensor, flat_preagg: torch.Tensor) -> torch.Tensor:
        if self.encoder is None:
            return vanilla_output
        if vanilla_output.shape[-1] != self.host_dim:
            raise ValueError("vanilla_output width mismatch")
        if flat_preagg.shape[-1] != self.preagg_dim:
            raise ValueError("flat_preagg width {} != expected {}".format(
                flat_preagg.shape[-1], self.preagg_dim))
        residual = self.encoder(flat_preagg)
        return torch.cat([vanilla_output, residual], dim=-1)
