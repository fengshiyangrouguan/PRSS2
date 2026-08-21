"""A0 context-overlap weights (theory doc 5.4).

Observed contexts follow pi(C|H); the conditional moment wants the
history-independent reference rho(C).  With rho = the observed marginal, the
importance ratio is

    w(h, c) = d rho / d pi(c|h) = p(h) p(c) / p(h, c) = 1 / r(h, c),

so a paired-vs-shuffled logistic density-ratio fit gives the weights
directly.  Truncated weights keep the estimator bounded, and the ESS

    ESS = (sum w)^2 / sum w^2

is the G0 identifiability check (no common support / wild weights -> the
distributional quotient is all the data can identify).
"""

from typing import Dict

import torch


def _logistic_ridge(features, targets, lambda_reg=1e-3, n_iter=20):
    """IRLS fit of sigmoid(F w) for the paired-vs-shuffled discriminator."""
    features = features.detach().to(dtype=torch.float64)
    targets = targets.detach().to(dtype=torch.float64)
    d = features.shape[-1]
    w = torch.zeros(d, dtype=torch.float64, device=features.device)
    eye = torch.eye(d, dtype=torch.float64, device=features.device)
    for _ in range(int(n_iter)):
        eta = features @ w
        p = torch.sigmoid(eta).clamp_min(1e-6)
        wt = (p * (1.0 - p)).clamp_min(1e-6)
        working = eta + (targets - p) / wt
        fw = features * wt.sqrt().unsqueeze(-1)
        gram = fw.transpose(0, 1) @ fw + lambda_reg * eye
        rhs = (fw.transpose(0, 1) @ (working * wt.sqrt())).unsqueeze(-1)
        chol = torch.linalg.cholesky(gram)
        w = torch.cholesky_solve(rhs, chol).squeeze(-1)
    return w


class DensityRatioWeights:
    """Linear density-ratio weights w = 1 / r(h, c) with truncation."""

    def __init__(self, w_min: float = 0.1, w_max: float = 10.0):
        self.w_min = float(w_min)
        self.w_max = float(w_max)
        self.coef = None
        self._feat_fn = None

    @torch.no_grad()
    def fit(self, h: torch.Tensor, c: torch.Tensor, lambda_reg=1e-3,
            n_iter=20) -> Dict:
        """h (n, p) history rows, c (n, d_c) context rows.

        Positive pairs are (h_i, c_i); negatives pair h_i with a shuffled
        context.  r = P(paired)/P(shuffled) = p(h,c)/(p(h)p(c)).

        The paired-vs-shuffled signal lives in the h-c INTERACTION (shuffling
        leaves both marginals untouched, so a first-order-only discriminator
        has a zero gradient at its origin).  Small widths use the full outer
        product; wide histories use a fixed random interaction sketch.
        """
        n = int(h.shape[0])
        gen = torch.Generator().manual_seed(2026)
        perm = torch.randperm(n, generator=gen)
        d_h, d_c = int(h.shape[-1]), int(c.shape[-1])

        def _feats(x, y):
            if d_h * d_c <= 64:
                inter = (x[:, :, None] * y[:, None, :]).reshape(x.shape[0], -1)
                return torch.cat([x, y, inter], dim=-1)
            ph = torch.randn(d_h, 8, generator=gen) / (d_h ** 0.5)
            pc = torch.randn(d_c, 8, generator=gen) / (d_c ** 0.5)
            xr, yr = x @ ph, y @ pc
            return torch.cat([xr, yr, xr * yr], dim=-1)

        feats = torch.cat([_feats(h, c), _feats(h, c[perm])],
                          dim=0).detach().to(dtype=torch.float64)
        targets = torch.cat([torch.ones(n), torch.zeros(n)],
                            dim=0).to(dtype=torch.float64)
        self.coef = _logistic_ridge(feats, targets, lambda_reg, n_iter)
        self._feat_fn = _feats
        return {"n_pairs": n}

    @torch.no_grad()
    def weights(self, h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """w = 1/r = exp(-logit(h, c)), truncated to [w_min, w_max]."""
        if self.coef is None:
            raise RuntimeError("weights not fitted")
        feats = self._feat_fn(h, c).detach().to(dtype=torch.float64)
        logit = feats @ self.coef
        w = torch.exp(-logit).clamp(self.w_min, self.w_max)
        return w

    @staticmethod
    def ess(w: torch.Tensor) -> float:
        s = float(w.sum().item())
        s2 = float((w * w).sum().item())
        return (s * s) / s2 if s2 > 0 else 0.0
