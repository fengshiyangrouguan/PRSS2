"""Spectral compressor: the full PRSS predictive quotient.

B(C) reader matrices feed the predictive Gram; periodic eigh solves the rank-k
spectral quotient; the normalized spectral-tail loss keeps the reader concentrated
inside the deployed subspace.
"""

from typing import Dict

import torch

from prss.config import InterfaceSpec, PRSSConfig
from prss.spectral import SpectralQuotient

from .base import Compressor, InterfaceData, register_variant


@register_variant
class SpectralCompressor(Compressor):
    name = "spectral"
    statistic = "reader"
    use_response_loss = True
    use_spectral_loss = True
    update_projection = True

    def __init__(self, spec: InterfaceSpec, config: PRSSConfig):
        super().__init__(spec, config)
        if not spec.dimensional_compression:
            raise ValueError("SpectralCompressor requires dimensional_compression")
        self.quotient = SpectralQuotient(
            name=spec.name,
            host_dim=spec.host_dim,
            candidate_dim=spec.candidate_dim,
            gram_ema=config.gram_ema_rho,
            eps=config.ridge_eps,
            spectral_step_size=config.spectral_step_size,
        )

    def projection(self) -> torch.Tensor:
        return self.quotient.R

    def project(self, candidate: torch.Tensor) -> torch.Tensor:
        return self.quotient.project(candidate)

    @torch.no_grad()
    def update_statistics(self, step: int, data: InterfaceData) -> None:
        if data is None or data.reader_matrices is None:
            return
        self.quotient.accumulate(data.reader_matrices)

    @torch.no_grad()
    def maybe_update(self, step: int) -> bool:
        return self.quotient.update(step)

    def spectral_loss(self, reader_matrix: torch.Tensor) -> torch.Tensor:
        return self.quotient.spectral_loss(reader_matrix)

    def snapshot(self) -> Dict:
        snap = self.quotient.snapshot()
        snap["variant"] = self.name
        return snap
