"""Direct compressor: R is a learnable parameter trained end-to-end by the task loss.

This is the "no spectral mechanism" control: same response supervision, same
candidate lift, but the projection is learned by gradient descent instead of being
solved analytically.  R is deliberately unconstrained during training; geometry
diagnostics use the orthonormalized span basis.
"""

import math
from typing import Dict

import torch
from torch import nn

from prss.config import InterfaceSpec, PRSSConfig
from prss.spectral import identity_like_projection, random_semi_orthogonal, row_orthonormalize

from .base import Compressor, register_variant


@register_variant
class DirectCompressor(Compressor):
    name = "direct"
    statistic = "none"
    use_response_loss = True
    use_spectral_loss = False
    update_projection = False

    def __init__(self, spec: InterfaceSpec, config: PRSSConfig):
        super().__init__(spec, config)
        if not spec.dimensional_compression:
            raise ValueError("DirectCompressor requires dimensional_compression")
        if config.initialization == "random":
            init = random_semi_orthogonal(spec.host_dim, spec.candidate_dim)
        else:
            init = identity_like_projection(spec.host_dim, spec.candidate_dim)
        self.R = nn.Parameter(init)

    def projection(self) -> torch.Tensor:
        return self.R

    def project(self, candidate: torch.Tensor) -> torch.Tensor:
        return candidate @ self.R.transpose(0, 1)

    def set_projection_trainable(self, trainable: bool) -> None:
        if not trainable:
            raise RuntimeError(
                "DirectCompressor is the trainable-projection ablation; freezing it is not allowed")

    def snapshot(self) -> Dict:
        with torch.no_grad():
            rr = self.R.detach().double() @ self.R.detach().double().T
            eye_k = torch.eye(self.spec.host_dim, device=rr.device, dtype=rr.dtype)
            orth = float(torch.linalg.norm(rr - eye_k, ord="fro").item() /
                         max(math.sqrt(self.spec.host_dim), 1.0))
            span_basis = row_orthonormalize(self.R.detach())
            span_orth = float(torch.linalg.norm(
                span_basis @ span_basis.T - torch.eye(self.spec.host_dim, device=span_basis.device,
                                                      dtype=span_basis.dtype), ord="fro").item() /
                max(math.sqrt(self.spec.host_dim), 1.0))
        return {
            "variant": self.name,
            "row_orthogonality_relative": orth,
            "span_basis_orthogonality_relative": span_orth,
            # R is learned end-to-end and deliberately unconstrained; the monitor must
            # not enforce row orthonormality on it.
            "projection_expected_orthogonal": False,
        }
