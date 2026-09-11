"""Gamma residual for the LaMP merge (one-shot aggregation, L2).

The official LaMP merge copies each COMP slot's K/V to its paired SUM
slot (slot pairing, weight 1).  GammaOnetime adds a learned residual
conditioned on the 4 COMP slots' pooled statistics:

    SUM_k = COMP_k + R_theta(COMP_k, pool(COMP_1..4))

with R_theta(x, pool) = U tanh(V [x; mean_pool; max_pool]).  U is
zero-initialized (Review round 7 init rule), so at step 0 the merge is
exactly the official copy bit-for-bit while every parameter receives
gradient from the first optimizer step.

One instance is shared across the K and V merges of its layer (dialog
line convention).  head_dim=128, hidden=64: ~43k params per layer,
~1.4M for 32 layers.
"""

import torch
from torch import nn


class GammaOnetime(nn.Module):
    """R_theta(x, pool) = U tanh(V [x; mean(pool); max(pool)]).

    Zero-output projection init: U starts at exactly zero, so step 0
    reproduces the official one-shot copy bit-for-bit.
    """

    def __init__(self, head_dim: int, hidden: int = 64,
                 init_scale: float = 0.02):
        super().__init__()
        self.head_dim = int(head_dim)
        self.V = nn.Linear(3 * self.head_dim, hidden)
        self.U = nn.Linear(hidden, self.head_dim, bias=False)
        with torch.no_grad():
            nn.init.normal_(self.V.weight, mean=0.0, std=init_scale)
            nn.init.normal_(self.V.bias, mean=0.0, std=init_scale)
            nn.init.zeros_(self.U.weight)

    def forward(self, x, pool):
        """Residual for each SUM slot.

        Args:
            x:    the paired COMP state, [B, H, N, D] (N = 4 slots).
            pool: all COMP states of the layer's merge block, [B, H, N, D].

        Returns:
            Residual [B, H, N, D]; exactly zero at init.

        The V/U linears run with autocast DISABLED (review 2 fix): the
        main forward executes under the trainer's autocast where Linear
        may run fp16, while the local replay runs outside autocast — the
        two paths would drift numerically.  Pinning the Gamma math to
        fp32 on BOTH paths removes the mismatch.
        """
        with torch.cuda.amp.autocast(enabled=False):
            mean_p = pool.mean(dim=2, keepdim=True).expand_as(x)
            max_p = pool.amax(dim=2, keepdim=True).expand_as(x)
            wd = self.V.weight.dtype
            feat = torch.cat([x.to(dtype=wd), mean_p.to(dtype=wd),
                              max_p.to(dtype=wd)], dim=-1)
            return self.U(torch.tanh(self.V(feat)))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
