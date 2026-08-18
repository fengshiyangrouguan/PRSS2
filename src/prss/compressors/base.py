"""Pluggable compressor abstraction: how the candidate h is reduced to host-width z.

The pure data contract mirrors the archived AblationPolicy: a variant declares which
statistic feeds the quotient, whether response/spectral supervision is used, and whether
projection updates happen at all.  ``configure`` is the one-time module-surgery hook.
"""

from dataclasses import dataclass
from typing import ClassVar, Dict, Optional, Type

import torch
from torch import nn

from prss.config import InterfaceSpec, PRSSConfig


@dataclass
class InterfaceData:
    """Per-tau training-only statistics for one update step (all detached)."""
    candidates: Optional[torch.Tensor] = None        # [N, d]
    reader_matrices: Optional[torch.Tensor] = None   # [N, p, d]


class Compressor(nn.Module):
    """How R is chosen and applied.  The contract fields are the archived
    AblationPolicy dataclass, lifted into class-level declarations."""

    name: ClassVar[str] = ""
    statistic: ClassVar[str] = "none"            # "none" | "pca" | "reader"
    use_response_loss: ClassVar[bool] = False
    use_spectral_loss: ClassVar[bool] = False
    update_projection: ClassVar[bool] = False

    def __init__(self, spec: InterfaceSpec, config: PRSSConfig):
        super().__init__()
        self.spec = spec
        self.config = config

    def projection(self) -> torch.Tensor:
        """Current quotient rows R (k x d)."""
        raise NotImplementedError

    def project(self, candidate: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def update_statistics(self, step: int, data: InterfaceData) -> None:
        return None

    @torch.no_grad()
    def maybe_update(self, step: int) -> bool:
        return False

    def spectral_loss(self, reader_matrix: torch.Tensor) -> torch.Tensor:
        # Zero but graph-connected, so no-gradient variants never break the autograd graph.
        return reader_matrix.sum() * 0.0

    def set_projection_trainable(self, trainable: bool) -> None:
        raise RuntimeError(
            "Compressor '{}' does not support a trainable projection".format(self.name))

    def configure(self, core) -> None:
        return None

    def snapshot(self) -> Dict:
        return {"variant": self.name}


VARIANT_REGISTRY: Dict[str, Type[Compressor]] = {}


def build_compressor(variant: str, spec: InterfaceSpec, config: PRSSConfig) -> Compressor:
    if variant not in VARIANT_REGISTRY:
        raise ValueError("Unknown PRSS compressor variant: {}".format(variant))
    return VARIANT_REGISTRY[variant](spec, config)


def register_variant(cls: Type[Compressor]) -> Type[Compressor]:
    if not cls.name:
        raise ValueError("Compressor subclass must define a non-empty name")
    if cls.name in VARIANT_REGISTRY:
        raise ValueError("Duplicate compressor variant: {}".format(cls.name))
    VARIANT_REGISTRY[cls.name] = cls
    return cls
