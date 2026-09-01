"""Gamma residual for the CCM merge_recur memory update (plan v2 L2).

The official merge_recur memory is the arithmetic mean of the same-slot
COMP-token K/Vs seen so far:

    M_t = mean(h_1..h_t)

Gamma attaches a small learned residual on top of that mean:

    M_t = mean(h_1..h_t) + R_theta(M_{t-1}, h_t, t)

with R_theta(x) = s . U tanh(V [prev; cur; time(t)]).  Safe zero-init:
U/V are small random and only the scalar gate ``s`` starts at zero, so the
merged memory starts as the exact official arithmetic mean (Test A) while
``s`` still receives gradient at the first optimizer step (closure 6).

The time encoding is a fixed log-spaced sinusoid with no learned
parameters, so the recurrence supports any turn count k (training random
prefixes may exceed 12 turns; plan L2).
"""

import math

import torch
from torch import nn

TIME_FREQS = 8  # log-spaced frequencies -> time_dim = 2 * TIME_FREQS


def time_features(t: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Deterministic log-spaced sinusoidal encoding of turn counts.

    Args:
        t: turn counts, any integer or float shape ending in the sequence
            dim (e.g. [B, L] or [B, 1, L, 1]).
        freqs: 1-D buffer of TIME_FREQS angular frequencies.

    Returns:
        [*t.shape, 2 * TIME_FREQS] with sin/cos pairs per frequency.
    """
    t = t.float()
    ang = t.unsqueeze(-1) * freqs.to(device=t.device, dtype=t.dtype)
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


class GammaResidual(nn.Module):
    """R_theta(prev, cur, t) = s * U tanh(V [prev; cur; time(t)]).

    One instance is shared across the K and V merges and across the two
    COMP/SUM slots of its layer.  With head_dim=128, hidden=64,
    time_dim=16: ~25.7k params per layer, ~821k for the full 32-layer
    7B backbone (< 1M, plan L2 acceptance).
    """

    def __init__(self, head_dim: int, time_dim: int = 16, hidden: int = 64,
                 init_scale: float = 0.02):
        super().__init__()
        if time_dim % 2:
            raise ValueError("time_dim must be even (sin/cos pairs)")
        self.head_dim = int(head_dim)
        self.time_dim = int(time_dim)
        self.V = nn.Linear(2 * self.head_dim + self.time_dim, hidden)
        self.U = nn.Linear(hidden, self.head_dim, bias=False)
        self.s = nn.Parameter(torch.zeros(()))
        # U/V small random; only the gate s is zero-initialized.
        with torch.no_grad():
            nn.init.normal_(self.V.weight, mean=0.0, std=init_scale)
            nn.init.normal_(self.V.bias, mean=0.0, std=init_scale)
            nn.init.normal_(self.U.weight, mean=0.0, std=init_scale)
        freqs = 2.0 * math.pi / torch.pow(2.0, torch.arange(TIME_FREQS)).float()
        self.register_buffer("time_freqs", freqs, persistent=False)

    def forward(self, prev, cur, t):
        """Residual for the memory update at every position.

        Args:
            prev: previous memory M_{t-1}, [B, H, L, D].
            cur: current COMP-token state h_t, [B, H, L, D].
            t: per-position turn counts, [B, L] (any integer dtype).

        Returns:
            Residual [B, H, L, D]; exactly zero while the gate s == 0.
        """
        feat = time_features(t, self.time_freqs).unsqueeze(1)  # [B, 1, L, T]
        feat = feat.expand(-1, prev.size(1), -1, -1)           # [B, H, L, T]
        x = torch.cat([prev, cur, feat], dim=-1)
        return self.s * self.U(torch.tanh(self.V(x)))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
