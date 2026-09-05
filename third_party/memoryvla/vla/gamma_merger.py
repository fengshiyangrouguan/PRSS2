"""gamma_merger.py — residual low-rank Gamma merger for cognitive memory.

Plan §5:
    m_avg = 0.5 * (m_a + m_b)
    h = f_theta(P m_a, P m_b)         (low-rank bottleneck)
    Gamma(m_a, m_b) = m_avg + alpha * U h

Zero-init guarantees the training start is EXACTLY the official AvgMerge
(alpha = 0, U = 0): the plugin cannot destroy the pretrained latent
geometry at step 0.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class GammaMerger(nn.Module):
    def __init__(self, dim: int = 4096, rank: int = 64, alpha_init: float = 0.0):
        super().__init__()
        self.dim = dim
        self.rank = rank

        self.proj_in = nn.Linear(dim, rank, bias=False)          # P
        nn.init.normal_(self.proj_in.weight, std=dim ** -0.5)

        self.mlp = nn.Sequential(
            nn.Linear(2 * rank, 4 * rank),
            nn.GELU(),
            nn.Linear(4 * rank, rank),
        )
        # zero-init the last MLP layer + U
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

        self.proj_out = nn.Linear(rank, dim, bias=False)         # U
        nn.init.zeros_(self.proj_out.weight)

        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def forward(self, m_a: torch.Tensor, m_b: torch.Tensor) -> torch.Tensor:
        """Merge a pair of [..., dim] cognitive states.

        Expects detached inputs when called from CogMemBank (the bank's
        no_grad context is preserved; all Gamma gradients come from the
        trainer's replay).
        """
        m_avg = 0.5 * (m_a + m_b)
        h = self.mlp(torch.cat([self.proj_in(m_a), self.proj_in(m_b)], dim=-1))
        return m_avg + self.alpha * self.proj_out(h)
