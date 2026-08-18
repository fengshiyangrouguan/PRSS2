"""Vanilla compressor: identity-compatible [I, 0] projection, never updated.

The first k candidate coordinates are the untouched host state, so z = R h reproduces
the vanilla host forward exactly.  This is both the no-compression control and the
quotient used by base interfaces with d == k.
"""

import torch

from prss.config import InterfaceSpec, PRSSConfig
from prss.spectral import identity_like_projection

from .base import Compressor, register_variant


@register_variant
class VanillaCompressor(Compressor):
    name = "vanilla"
    statistic = "none"
    use_response_loss = False
    use_spectral_loss = False
    update_projection = False

    def __init__(self, spec: InterfaceSpec, config: PRSSConfig):
        super().__init__(spec, config)
        r = identity_like_projection(spec.host_dim, spec.candidate_dim)
        self.register_buffer("R", r)

    def projection(self) -> torch.Tensor:
        return self.R

    def project(self, candidate: torch.Tensor) -> torch.Tensor:
        return candidate @ self.R.transpose(0, 1)
