"""gamma_merger.py — residual low-rank Gamma merger for cognitive memory.

Plan §5:
    m_avg = 0.5 * (m_a + m_b)
    h = f_theta(P m_a, P m_b)         (low-rank bottleneck)
    Gamma(m_a, m_b) = m_avg + alpha * U h

Init (review ruling B1, 2026-09-05): alpha = 1, U = 0, MLP fully random.
The training start is EXACTLY the official AvgMerge (1 * 0 * h = 0), but
the learning path is open: dL/dU = alpha * h (x) dL/dz is nonzero at step
0, so U leaves zero on the first repr step and the MLP gradient chain
opens right after.  (alpha=0 or a zero MLP last layer would freeze all
parameters at zero gradient forever.)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class GammaMerger(nn.Module):
    def __init__(self, dim: int = 4096, rank: int = 64, alpha_init: float = 1.0,
                 seed: int = None):
        super().__init__()
        self.dim = dim
        self.rank = rank
        # local generator keeps Gamma init DETERMINISTIC and decoupled from
        # the global RNG stream (review ruling B2: shared module init must be
        # identical across arms regardless of Gamma creation)
        g = torch.Generator()
        if seed is not None:
            g.manual_seed(seed)

        self.proj_in = nn.Linear(dim, rank, bias=False)          # P
        nn.init.normal_(self.proj_in.weight, std=dim ** -0.5, generator=g)

        self.mlp = nn.Sequential(
            nn.Linear(2 * rank, 4 * rank),
            nn.GELU(),
            nn.Linear(4 * rank, rank),
        )
        # MLP stays fully RANDOM; re-init every layer from the LOCAL
        # generator so no global RNG is consumed here
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5),
                                         generator=g)
                nn.init.uniform_(
                    layer.bias, -1.0 / math.sqrt(layer.weight.shape[1]),
                    1.0 / math.sqrt(layer.weight.shape[1]), generator=g)

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
