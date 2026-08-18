"""Tau-indexed PRSS assembly.  Host-agnostic by construction.

Every interface is keyed by an opaque type string ``tau`` supplied by a host adapter;
this module contains no host-specific names or tensor layouts.  The compressor variant
is injected through :func:`prss.compressors.build_compressor`.
"""

from typing import Dict, Mapping, Optional

import torch
from torch import nn

from prss.candidate import GenericResidualCandidateBuilder
from prss.compressors import Compressor, InterfaceData, build_compressor
from prss.config import PRSSConfig
from prss.outside import OutsideContextEncoder
from prss.reader import ConditionalMatrixReader, UnrestrictedReader


class PRSSCore(nn.Module):
    """Per-tau quotients, candidate builders, readers and the training-only outside encoder.

    Base interfaces (d == k) receive an identity-compatible vanilla quotient and train
    neither a reader nor a candidate builder; non-trivial PRSS interfaces begin where
    dimensional compression exists.
    """

    def __init__(self, config: PRSSConfig, variant: Optional[str] = None):
        super().__init__()
        self.config = config
        self.variant = variant if variant is not None else config.variant
        self.quotients = nn.ModuleDict()
        self.builders = nn.ModuleDict()
        self.readers = nn.ModuleDict()
        self.unrestricted = nn.ModuleDict()
        for tau, spec in config.interfaces.items():
            compressor_variant = self.variant if spec.dimensional_compression else "vanilla"
            self.quotients[tau] = build_compressor(compressor_variant, spec, config)
            # A builder exists only when the host raw state is genuinely widened:
            # h = [vanilla; phi(preagg)].  Hosts whose raw state already IS the
            # candidate (raw_dim == candidate_dim, e.g. the synthetic tree task)
            # pass it through untouched.
            if spec.candidate_dim > spec.raw_dim:
                self.builders[tau] = GenericResidualCandidateBuilder(
                    host_dim=spec.host_dim,
                    preagg_dim=config.parent_local_dim,
                    candidate_dim=spec.candidate_dim,
                    hidden_dim=config.candidate_hidden_dim,
                )
            # Readers supervise every compressive interface, whether or not the host
            # widened the raw state through a builder.
            if spec.dimensional_compression:
                self.readers[tau] = ConditionalMatrixReader(
                    context_dim=config.context_dim,
                    candidate_dim=spec.candidate_dim,
                    response_dim=spec.response_dim,
                    hidden_dim=config.reader_hidden_dim,
                )
                self.unrestricted[tau] = UnrestrictedReader(
                    context_dim=config.context_dim,
                    candidate_dim=spec.candidate_dim,
                    response_dim=spec.response_dim,
                    hidden_dim=config.reader_hidden_dim,
                )
        self.outside = OutsideContextEncoder(
            config.interfaces,
            root_metadata_dim=config.root_metadata_dim,
            parent_local_dim=config.parent_local_dim,
            context_dim=config.context_dim,
            relation_count=config.relation_count,
            relation_dim=config.relation_dim,
            layers=config.outside_layers,
            detach_siblings=config.detach_siblings,
        )
        self._spectral_updates_allowed = True

    # ------------------------------------------------------------------ interfaces
    @property
    def compressive_interfaces(self) -> list:
        return [tau for tau, spec in self.config.interfaces.items()
                if spec.dimensional_compression]

    def has_reader(self, tau: str) -> bool:
        return tau in self.readers

    def aux_contract(self) -> tuple[bool, bool]:
        """(use_response_loss, use_spectral_loss) from the first compressive quotient.

        All compressive interfaces share one variant in V2, so one contract suffices.
        """
        for tau in self.compressive_interfaces:
            q = self.quotients[tau]
            return bool(q.use_response_loss), bool(q.use_spectral_loss)
        return False, False

    # ------------------------------------------------------------------ main path
    def make_candidate(self, tau: str, vanilla_output: torch.Tensor,
                       flat_preagg: Optional[torch.Tensor] = None) -> torch.Tensor:
        # raw == candidate interfaces pass the host state through untouched.
        if tau not in self.builders:
            return vanilla_output
        return self.builders[tau](vanilla_output, flat_preagg)

    def project(self, tau: str, candidate: torch.Tensor) -> torch.Tensor:
        return self.quotients[tau].project(candidate)

    # --------------------------------------------------------------- training-only
    def structured_read(self, tau: str, context: torch.Tensor, candidate: torch.Tensor):
        reader = self.readers[tau]
        B, b = reader(context)
        return reader.logits(B, b, candidate), B, b

    def unrestricted_read(self, tau: str, context: torch.Tensor,
                          candidate: torch.Tensor) -> torch.Tensor:
        return self.unrestricted[tau](context.detach(), candidate.detach())

    def spectral_loss(self, tau: str, reader_matrix: torch.Tensor) -> torch.Tensor:
        return self.quotients[tau].spectral_loss(reader_matrix)

    @torch.no_grad()
    def update_statistics(self, step: int,
                          interfaces_by_tau: Mapping[str, InterfaceData]) -> None:
        if not self._spectral_updates_allowed:
            return
        for tau, data in interfaces_by_tau.items():
            self.quotients[tau].update_statistics(step, data)

    @torch.no_grad()
    def maybe_update(self, step: int) -> Dict[str, bool]:
        out = {}
        if not self._spectral_updates_allowed:
            return {tau: False for tau in self.quotients}
        for tau, q in self.quotients.items():
            if q.update_projection:
                out[tau] = q.maybe_update(step)
        return out

    def set_spectral_updates_allowed(self, flag: bool) -> None:
        # Hard gate: validation/test must never touch Gram or R.
        self._spectral_updates_allowed = bool(flag)

    def set_projection_trainable(self, tau: str, trainable: bool) -> None:
        self.quotients[tau].set_projection_trainable(trainable)

    def snapshots(self) -> Dict:
        return {tau: q.snapshot() for tau, q in self.quotients.items()}
