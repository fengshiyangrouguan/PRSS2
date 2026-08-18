"""Random compressor: frozen random semi-orthogonal projection.

Keeps the same future-response supervision as the full method so the comparison
isolates *how R is chosen* — everything else is identical.
"""

import torch

from prss.config import InterfaceSpec, PRSSConfig
from prss.spectral import random_semi_orthogonal

from .base import Compressor, register_variant


@register_variant
class RandomCompressor(Compressor):
    name = "random"
    statistic = "none"
    use_response_loss = True
    use_spectral_loss = False
    update_projection = False

    def __init__(self, spec: InterfaceSpec, config: PRSSConfig):
        super().__init__(spec, config)
        if not spec.dimensional_compression:
            raise ValueError("RandomCompressor requires dimensional_compression")
        r = random_semi_orthogonal(spec.host_dim, spec.candidate_dim)
        self.register_buffer("R", r)

    def projection(self) -> torch.Tensor:
        return self.R

    def project(self, candidate: torch.Tensor) -> torch.Tensor:
        return candidate @ self.R.transpose(0, 1)
