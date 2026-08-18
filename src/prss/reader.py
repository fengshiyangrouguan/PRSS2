"""Conditional future-reading matrices and the monitoring-only unrestricted reader."""

import torch
from torch import nn


class ConditionalMatrixReader(nn.Module):
    """Structured reader: B(C) in R^{p x d} plus bias b(C), linear in the candidate.

    logits = b + B h.  The linear-in-history structure is what makes the predictive
    Gram G = E[B^T B] and its top-k eigenspace semantically meaningful.
    """

    def __init__(self, context_dim: int, candidate_dim: int, response_dim: int = 1,
                 hidden_dim: int = 128):
        super().__init__()
        self.context_dim = int(context_dim)
        self.candidate_dim = int(candidate_dim)
        self.response_dim = int(response_dim)
        self.trunk = nn.Sequential(
            nn.Linear(self.context_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.matrix_head = nn.Linear(hidden_dim, self.response_dim * self.candidate_dim)
        self.bias_head = nn.Linear(hidden_dim, self.response_dim)

    def forward(self, context):
        h = self.trunk(context)
        B = self.matrix_head(h).reshape(*h.shape[:-1], self.response_dim, self.candidate_dim)
        b = self.bias_head(h)
        return B, b

    @staticmethod
    def logits(B, b, candidate):
        # B [..., p, d], candidate [..., d] -> logits [..., p]
        return b + torch.einsum("...pd,...d->...p", B, candidate)


class UnrestrictedReader(nn.Module):
    """Monitoring-only comparator: MLP(h, C).  Never contributes to the quotient Gram."""

    def __init__(self, context_dim: int, candidate_dim: int, response_dim: int = 1,
                 hidden_dim: int = 128):
        super().__init__()
        self.response_dim = int(response_dim)
        self.net = nn.Sequential(
            nn.Linear(context_dim + candidate_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.response_dim),
        )

    def forward(self, context, candidate):
        return self.net(torch.cat([context, candidate], dim=-1))
