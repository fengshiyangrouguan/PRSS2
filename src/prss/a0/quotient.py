"""A0 phase A: per-interface conditional-moment low-rank estimation.

Streaming centered cross/self second moments of (U = vec(a(C) ⊗ φ_Y(Y)),
X = x(H)); one-shot whitened rank-r SVD:

    M_tau = C_ux (C_xx + λI)^{-1/2} ≈ U_r Σ_r V_r^T
    R_tau = V_r^T (C_xx + λI)^{-1/2},    z = R_tau x(H)

R is frozen after ``solve`` (a buffer, never a gradient parameter).  The SVD
is only a solver; the "future-valid" definition is the conditional moment
itself (theory doc 5.5).
"""

import torch


@torch.no_grad()
def randomized_svd(matrix: torch.Tensor, rank: int, oversample: int = 10,
                   n_iter: int = 2):
    """Halko-Martinsson-Tropp style range finder (theory doc 5.5).

    Dense ``torch.linalg.svd`` is preferred for p <= 200; this exists for
    wide history lifts.  Returns (U[:, :rank], S[:rank], Vh[:rank]) on the
    same device/dtype as the input.
    """
    m, p = matrix.shape
    k = min(int(rank) + int(oversample), p)
    dtype = matrix.dtype
    device = matrix.device
    omega = torch.randn(p, k, device=device, dtype=dtype)
    y = matrix @ omega
    for _ in range(int(n_iter)):
        y = matrix @ (matrix.transpose(0, 1) @ y)
    q, _ = torch.linalg.qr(y, mode="reduced")
    b = q.transpose(0, 1) @ matrix
    uq, s, vh = torch.linalg.svd(b, full_matrices=False)
    u = q @ uq
    return u[:, :rank], s[:rank], vh[:rank]


class A0Quotient:
    """One interface's conditional-moment accumulator + frozen rank-r map.

    All moments accumulate in float64 regardless of the stream dtype.
    """

    def __init__(self, tau: str, p: int, m: int):
        self.tau = tau
        self.p = int(p)          # history lift width x(H)
        self.m = int(m)          # u width = 2 * d_context
        self.n = 0
        self.c_ux = torch.zeros(self.m, self.p, dtype=torch.float64)
        self.c_xx = torch.zeros(self.p, self.p, dtype=torch.float64)
        self.s_x = torch.zeros(self.p, dtype=torch.float64)
        self.s_u = torch.zeros(self.m, dtype=torch.float64)
        self.r_matrix = None     # (r, p) frozen coordinate map
        self.sigma = None        # (r,) top singular values
        self.rank_tail = None    # eps_rank(r) = sum_{j>r} s_j^2 / sum s_j^2
        self.solved = False

    @torch.no_grad()
    def accumulate(self, x: torch.Tensor, u: torch.Tensor,
                   w: torch.Tensor = None) -> None:
        """Batch-accumulate (x rows, u rows) with optional row weights.

        Unweighted rows count n += batch size; weighted rows count the weight
        sum (importance-corrected moments, theory doc 5.4).
        """
        if self.solved:
            raise RuntimeError("quotient {} already frozen".format(self.tau))
        x = x.detach().to(dtype=torch.float64)
        u = u.detach().to(dtype=torch.float64)
        if x.shape[-1] != self.p or u.shape[-1] != self.m:
            raise ValueError("width mismatch for {}: x {}, u {}".format(
                self.tau, x.shape, u.shape))
        if w is not None:
            w = w.detach().to(dtype=torch.float64)
            if w.shape[0] != x.shape[0]:
                raise ValueError("weight count mismatch for {}".format(
                    self.tau))
        if self.c_ux.device != x.device:
            self.c_ux = self.c_ux.to(x.device)
            self.c_xx = self.c_xx.to(x.device)
            self.s_x = self.s_x.to(x.device)
            self.s_u = self.s_u.to(x.device)
        wu = u * w.unsqueeze(-1) if w is not None else u
        wx = x * w.unsqueeze(-1) if w is not None else x
        self.c_ux.add_(wu.transpose(0, 1) @ x)
        self.c_xx.add_(wx.transpose(0, 1) @ x)
        self.s_x.add_(wx.sum(dim=0))
        self.s_u.add_(wu.sum(dim=0))
        self.n += float(w.sum().item()) if w is not None else int(x.shape[0])

    def centered_moments(self):
        """Streaming-centered C_xx/n and C_ux/n (weight-aware: n is the
        accumulated weight sum, means are the weighted means)."""
        if self.n == 0:
            raise RuntimeError("quotient {} has no accumulated rows".format(self.tau))
        mu_x = self.s_x / self.n
        mu_u = self.s_u / self.n
        cxx = self.c_xx / self.n - torch.outer(mu_x, mu_x)
        cux = self.c_ux / self.n - torch.outer(mu_u, mu_x)
        return cxx, cux

    @torch.no_grad()
    def solve(self, rank_r: int, lambda_x: float = 1e-4) -> float:
        """One-shot whitened rank-r SVD; freezes R and returns eps_rank(r)."""
        cxx, cux = self.centered_moments()
        eye = torch.eye(self.p, dtype=torch.float64, device=cxx.device)
        vals, vecs = torch.linalg.eigh(cxx + lambda_x * eye)
        w = (vecs * (1.0 / vals.clamp_min(1e-12).sqrt())) @ vecs.transpose(0, 1)
        matrix = cux @ w  # (m, p)
        if self.p <= 200:
            u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
        else:
            u, s, vh = randomized_svd(matrix, rank_r)
        r = min(int(rank_r), int(s.numel()))
        self.r_matrix = vh[:r] @ w          # (r, p)
        self.sigma = s[:r].clone()
        total = float((s ** 2).sum().item())
        self.rank_tail = float((s[r:] ** 2).sum().item()) / total \
            if total > 0 else 0.0
        self.solved = True
        return self.rank_tail

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """z = R x for any leading batch dims; requires solve() first.

        Computed in float64 (R is a fixed calibration result); callers that
        feed the result into a float32 readout get torch's usual promotion.
        """
        if not self.solved:
            raise RuntimeError("quotient {} not solved".format(self.tau))
        r_t = self.r_matrix.transpose(0, 1).to(device=x.device)
        return x.double() @ r_t

    def snapshot(self):
        return {
            "tau": self.tau,
            "n": self.n,
            "p": self.p,
            "m": self.m,
            "solved": self.solved,
            "rank_tail": self.rank_tail,
            "sigma_top": None if self.sigma is None
            else [float(v) for v in self.sigma.cpu()],
        }
