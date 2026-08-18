"""PCA compressor: static candidate-covariance principal subspace.

A static variance baseline, not the damped alternating PRSS update: the exact
principal subspace is deployed (spectral_step_size == 1.0) from a *centered*
candidate covariance, so the ablation means what its name says.
"""

from typing import Dict

import torch

from prss.config import InterfaceSpec, PRSSConfig
from prss.spectral import SpectralQuotient, centered_candidate_gram

from .base import Compressor, InterfaceData, register_variant


@register_variant
class PCACompressor(Compressor):
    name = "pca"
    statistic = "pca"
    use_response_loss = True
    use_spectral_loss = False
    update_projection = True

    def __init__(self, spec: InterfaceSpec, config: PRSSConfig):
        super().__init__(spec, config)
        if not spec.dimensional_compression:
            raise ValueError("PCACompressor requires dimensional_compression")
        self.quotient = SpectralQuotient(
            name=spec.name,
            host_dim=spec.host_dim,
            candidate_dim=spec.candidate_dim,
            gram_ema=config.gram_ema_rho,
            eps=config.ridge_eps,
            spectral_step_size=1.0,
        )

    def projection(self) -> torch.Tensor:
        return self.quotient.R

    def project(self, candidate: torch.Tensor) -> torch.Tensor:
        return self.quotient.project(candidate)

    @torch.no_grad()
    def update_statistics(self, step: int, data: InterfaceData) -> None:
        if data is None or data.candidates is None:
            return
        self.quotient.accumulate_covariance(centered_candidate_gram(data.candidates))

    @torch.no_grad()
    def maybe_update(self, step: int) -> bool:
        return self.quotient.update(step)

    def snapshot(self) -> Dict:
        snap = self.quotient.snapshot()
        snap["variant"] = self.name
        return snap
