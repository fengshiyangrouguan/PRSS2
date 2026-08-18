import torch
from torch import nn


class ExactPreAggregationCandidate(nn.Module):
    """Build h=[vanilla aggregate; learned residual(exact host aggregate inputs)].

    The residual consumes exactly the information passed to the official TGN ``aggregate`` call:
    source lower quotient, source-time embedding, ordered temporal-neighbor lower quotients,
    neighbor-time embeddings, edge features, and the padding mask.  Lower states are *already*
    quotient states, so this builder cannot bypass an earlier recursive quotient.

    The first ``host_dim`` coordinates are the untouched official TGN aggregate.  With the
    identity-like initialization R=[I,0], PRSS therefore starts numerically equal to vanilla TGN.
    """
    def __init__(self, host_dim: int, edge_dim: int, time_dim: int, n_neighbors: int,
                 candidate_dim: int, hidden_dim: int = 128):
        super().__init__()
        if candidate_dim < host_dim:
            raise ValueError("candidate_dim must be >= host_dim")
        self.host_dim = int(host_dim)
        self.edge_dim = int(edge_dim)
        self.time_dim = int(time_dim)
        self.n_neighbors = int(n_neighbors)
        self.candidate_dim = int(candidate_dim)
        self.residual_dim = self.candidate_dim - self.host_dim
        self.preagg_dim = (
            self.host_dim + self.time_dim +
            self.n_neighbors * self.host_dim +
            self.n_neighbors * self.time_dim +
            self.n_neighbors * self.edge_dim +
            self.n_neighbors
        )
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

    def exact_preagg(self, source_lower, source_time, neighbor_lower, edge_time,
                     edge_features, mask):
        b = source_lower.shape[0]
        if neighbor_lower.shape[1] != self.n_neighbors:
            raise ValueError("n_neighbors mismatch in PRSS candidate builder")
        values = [
            source_lower,
            source_time.reshape(b, -1),
            neighbor_lower.reshape(b, -1),
            edge_time.reshape(b, -1),
            edge_features.reshape(b, -1),
            mask.to(source_lower.dtype).reshape(b, -1),
        ]
        x = torch.cat(values, dim=-1)
        if x.shape[-1] != self.preagg_dim:
            raise ValueError(f"preagg width {x.shape[-1]} != expected {self.preagg_dim}")
        return x

    def forward(self, vanilla_output, source_lower, source_time, neighbor_lower,
                edge_time, edge_features, mask):
        if self.residual_dim == 0:
            return vanilla_output
        x = self.exact_preagg(
            source_lower, source_time, neighbor_lower, edge_time, edge_features, mask)
        residual = self.encoder(x)
        return torch.cat([vanilla_output, residual], dim=-1)
