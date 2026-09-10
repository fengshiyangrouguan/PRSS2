"""Per-tau Reconstruction / LocalPred training heads (spec §5.1/§5.2).

Two supervision families over the SAME traced pairs as the canonical Ky-Fan
witness (R0).  Each is a per-tau ``Linear(input, 128)-GELU-Linear(128, target)``
decoder that exists ONLY during training:

* P1 Reconstruction : ``u_hat = D_rec_tau([z, chi(C)])``, target = the child's
  pre-Gamma vanilla aggregate ``u`` (recorded in the same forward);
* P2 LocalPred-2Obs : ``p_hat = D_pred_tau([z, chi(C)])``, target = the R0
  aligned-2Obs ``BoundaryMaps.pv`` output for the same cut / C / Y1 / Y2
  (bitwise equal to R0's p).

Targets are normalized by per-tau statistics that are FITTED ONCE (first
detached target batch) and then frozen; targets always stop-gradient.  The
heads form one module so the runner owns them as one disjoint ``aux_head``
parameter group.  Neither decoder sees any trainable future encoder.
"""

from typing import Dict, Iterable, List, Optional

import torch
import torch.nn.functional as F
from torch import nn


def _safe(tau: str) -> str:
    return str(tau).replace(":", "_").replace(".", "_").replace("/", "_")


class _Decoder(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(d_in), 128), nn.GELU(), nn.Linear(128, int(d_out)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AuxHeads(nn.Module):
    """Per-tau decoders + frozen target normalizers for one aux family."""

    def __init__(self, kind: str, *, taus: Iterable[str], z_dim: int,
                 d_ctx: int, d_out: int, eps: float = 1e-6):
        super().__init__()
        if kind not in ("rec", "pred"):
            raise ValueError("unknown aux kind {}".format(kind))
        self.kind = str(kind)
        self.z_dim = int(z_dim)
        self.d_ctx = int(d_ctx)
        self.d_out = int(d_out)
        self.eps = float(eps)
        taus = list(taus)
        self.taus = taus
        self.heads = nn.ModuleDict({
            tau: _Decoder(self.z_dim + self.d_ctx, self.d_out)
            for tau in taus})
        for tau in taus:
            self.register_buffer("mean_" + _safe(tau), torch.zeros(self.d_out))
            self.register_buffer("scale_" + _safe(tau),
                                 torch.ones(self.d_out))
        self._fitted = set()

    # ------------------------------------------------------------- norm stats
    def _mean(self, tau: str) -> torch.Tensor:
        return getattr(self, "mean_" + _safe(tau))

    def _scale(self, tau: str) -> torch.Tensor:
        return getattr(self, "scale_" + _safe(tau))

    def fit(self, tau: str, target: torch.Tensor) -> None:
        """Fit (frozen once) per-coordinate mean/std from a detached batch."""
        if tau in self._fitted:
            return
        with torch.no_grad():
            t = target.detach()
            mean = t.mean(dim=0)
            var = t.var(dim=0, unbiased=False)
            scale = (var + self.eps).sqrt().clamp_min(1e-6)
            self._mean(tau).copy_(mean)
            self._scale(tau).copy_(scale)
        self._fitted.add(tau)

    def fit_weighted(self, tau: str, target: torch.Tensor,
                     weights: torch.Tensor) -> None:
        """Frozen once, PER-TREE-WEIGHTED stats over the whole window."""
        if tau in self._fitted:
            return
        with torch.no_grad():
            t = target.detach()
            w = weights.detach().to(t.dtype)
            w = w / w.sum().clamp_min(1e-30)
            mean = (t * w.unsqueeze(-1)).sum(0)
            d = t - mean
            var = (d * d * w.unsqueeze(-1)).sum(0)
            scale = (var + self.eps).sqrt().clamp_min(1e-6)
            self._mean(tau).copy_(mean)
            self._scale(tau).copy_(scale)
        self._fitted.add(tau)

    def normalize(self, tau: str, target: torch.Tensor) -> torch.Tensor:
        return (target.detach() - self._mean(tau)) / self._scale(tau)

    # ------------------------------------------------------------ supervision
    def decode(self, tau: str, z: torch.Tensor,
               ctx: torch.Tensor) -> torch.Tensor:
        """Head output for ``[z, chi(C)]`` rows."""
        return self.heads[tau](torch.cat([z, ctx], dim=-1))

    def regression_loss(self, tau: str, z: torch.Tensor, ctx: torch.Tensor,
                        target: torch.Tensor) -> torch.Tensor:
        """``mean || D([z, chi(C)]) - sg(normalize(target)) ||^2``."""
        self.fit(tau, target)
        y = self.decode(tau, z, ctx)
        tgt = self.normalize(tau, target)
        return F.mse_loss(y, tgt, reduction="mean")
