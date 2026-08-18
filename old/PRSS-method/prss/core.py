from typing import Dict

import torch
from torch import nn

from .candidate import ExactPreAggregationCandidate
from .outside import OutsideContextEncoder
from .reader import ConditionalMatrixReader, UnrestrictedReader
from .spectral import SpectralQuotient


class PRSSCore(nn.Module):
    def __init__(self, host_dim: int, edge_dim: int, time_dim: int, n_neighbors: int,
                 n_layers: int, candidate_dim: int = 256, candidate_hidden: int = 128,
                 context_dim: int = 64, reader_hidden: int = 128, gram_ema: float = 0.05,
                 spectral_step_size: float = 0.25):
        super().__init__()
        self.host_dim = int(host_dim)
        self.n_layers = int(n_layers)
        self.candidate_dims = {0: self.host_dim}
        for l in range(1, self.n_layers + 1):
            self.candidate_dims[l] = int(candidate_dim)
        self.builders = nn.ModuleDict()
        self.quotients = nn.ModuleDict()
        self.readers = nn.ModuleDict()
        self.unrestricted = nn.ModuleDict()
        for l in range(self.n_layers + 1):
            d = self.candidate_dims[l]
            self.quotients[str(l)] = SpectralQuotient(
                name=f"tgn_layer_{l}", host_dim=self.host_dim, candidate_dim=d, gram_ema=gram_ema,
                spectral_step_size=spectral_step_size)
            # Layer 0 is the upstream leaf/base state with d_0 == k_0.  It has no
            # learned dimensional quotient and therefore must not train a future reader or
            # contribute to the operator bank.  Non-trivial PRSS interfaces begin at l>=1.
            if l > 0:
                self.readers[str(l)] = ConditionalMatrixReader(context_dim, d, hidden_dim=reader_hidden)
                self.unrestricted[str(l)] = UnrestrictedReader(context_dim, d, hidden_dim=reader_hidden)
                self.builders[str(l)] = ExactPreAggregationCandidate(
                    host_dim=self.host_dim, edge_dim=edge_dim, time_dim=time_dim,
                    n_neighbors=n_neighbors, candidate_dim=d, hidden_dim=candidate_hidden)
        # Exact parent-local constructor metadata excluding child states themselves:
        # source-time embedding + ordered neighbor time embeddings + edge features + padding mask.
        self.parent_local_dim = (
            int(time_dim) + int(n_neighbors) * int(time_dim) +
            int(n_neighbors) * int(edge_dim) + int(n_neighbors)
        )
        self.outside = OutsideContextEncoder(
            self.candidate_dims, parent_local_dim=self.parent_local_dim, context_dim=context_dim)
        self.context_dim = context_dim

    def make_candidate(self, layer: int, vanilla_output, source_lower=None, source_time=None,
                       neighbor_lower=None, edge_time=None, edge_features=None, mask=None):
        if layer == 0:
            return vanilla_output
        return self.builders[str(layer)](
            vanilla_output, source_lower, source_time, neighbor_lower, edge_time, edge_features, mask)

    def project(self, layer: int, candidate):
        return self.quotients[str(layer)].project(candidate)

    def snapshots(self) -> Dict:
        return {f"tgn_layer_{l}": self.quotients[str(l)].snapshot()
                for l in range(self.n_layers + 1)}
