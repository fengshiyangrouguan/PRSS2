import torch
from torch import nn


class ConditionalMatrixReader(nn.Module):
    def __init__(self, context_dim: int, candidate_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.context_dim = int(context_dim)
        self.candidate_dim = int(candidate_dim)
        self.trunk = nn.Sequential(
            nn.Linear(self.context_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.matrix_head = nn.Linear(hidden_dim, self.candidate_dim)
        self.bias_head = nn.Linear(hidden_dim, 1)

    def forward(self, context):
        h = self.trunk(context)
        B = self.matrix_head(h).unsqueeze(1)   # batch x 1 x d
        b = self.bias_head(h).squeeze(-1)      # batch
        return B, b

    @staticmethod
    def logits(B, b, candidate):
        return b + torch.einsum("bpd,bd->bp", B, candidate).squeeze(-1)


class UnrestrictedReader(nn.Module):
    """Monitoring-only comparator. Never contributes to the quotient Gram."""
    def __init__(self, context_dim: int, candidate_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(context_dim + candidate_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, context, candidate):
        return self.net(torch.cat([context, candidate], dim=-1)).squeeze(-1)
