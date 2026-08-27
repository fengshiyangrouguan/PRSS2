"""Ky Fan spectral score: the single component-internal loss.

For one interface type tau, with rows ``Z`` (compressed states, gradient
connected) and ``P`` (fixed joint tests psi(c,y), constant)::

    J_tau = tr[ (Sigma_ZZ + e_z I)^-1  Sigma_ZP  (Sigma_PP + e_p I)^-1  Sigma_PZ ]

computed with two Cholesky factorizations and triangular solves — no SVD, no
explicit inverses.  The training loss maximizes the per-interface score
(equivalently minimizes the positive form sum_tau alpha_tau (d_tau - J_tau)
which differs only by a theta-constant).

Ridge terms are relative to trace/dim so the same epsilon behaves across
scales, plus an absolute jitter so a constant Z/P cannot make the Cholesky
fail outright.
"""

from typing import Dict, List, Tuple

import torch


def _ridge(eps: float, sigma: torch.Tensor, jitter: float = 1e-12) -> torch.Tensor:
    rel = eps * torch.trace(sigma).clamp(min=0.0) / sigma.shape[0]
    return rel + jitter


def _cholesky_retry(mat: torch.Tensor, tries: int = 3) -> torch.Tensor:
    """Cholesky with escalating absolute jitter (a constant Z/P would make
    the relative ridge vanish; the jitter keeps the factorization alive)."""
    jitter = 1e-12
    for _ in range(tries):
        try:
            return torch.linalg.cholesky(mat)
        except RuntimeError:
            n = mat.shape[0]
            mat = mat + jitter * torch.eye(n, dtype=mat.dtype, device=mat.device)
            jitter *= 10.0
    return torch.linalg.cholesky(mat)


def kf_score_fixed(z_c: torch.Tensor, p_c: torch.Tensor,
                   szz: torch.Tensor, spp: torch.Tensor,
                   eps: float = 1e-4) -> torch.Tensor:
    """J with all whitening statistics held constant (a true fixed scale).

    ``z_c`` carries gradient (the only gradient path is the cross term
    ``S_ZP = z_c^T p_c / M``).  This core serves two purposes: (a) the
    gradcheck target, and (b) a future fixed-reference-scale variant, which
    must precompute ``szz``/``spp`` once from a frozen calibration model
    and add an explicit covariance constraint — it is NOT what ``kf_score``
    currently uses.
    """
    m = z_c.shape[0]
    szp = z_c.t() @ p_c / m
    lz = _cholesky_retry(szz + _ridge(eps, szz) * torch.eye(
        szz.shape[0], dtype=szz.dtype, device=szz.device))
    lp = _cholesky_retry(spp + _ridge(eps, spp) * torch.eye(
        spp.shape[0], dtype=spp.dtype, device=spp.device))
    w = torch.cholesky_solve(szp, lz)             # [r, m] = (S_ZZ+e)^-1 S_ZP
    s = torch.cholesky_solve(szp.t(), lp)         # [m, r] = (S_PP+e)^-1 S_PZ
    return (w * s.t()).sum()  # tr[(S_ZZ+e)^-1 S_ZP (S_PP+e)^-1 S_PZ]


def kf_score(Z: torch.Tensor, P: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """J_tau for one type.  ``Z``: [M, r] (gradient flows); ``P``: [M, m] (detached).

    This is the STANDARD normalized Ky Fan / CCA score: ``C_zz`` is the
    normalization term of the current ``q_theta`` and therefore theta-dependent
    — its gradient is a *necessary* part of the objective (shrinking Z shrinks
    the cross-covariance too; the two effects cancel exactly, giving the
    scale invariance ``<grad_Z J, Z> = 0``).  Stop-gradding the batch ``C_zz``
    would be a half-gradient (forward changes, backward pretends not), whose
    radial derivative is ``2J`` instead of 0 — it is NOT the gradient of any
    objective.  The P-side statistics carry no gradient because P is a fixed
    measurement (``psi`` runs under no_grad).

    ``kf_score_fixed`` below is the core for a genuinely fixed reference
    scale (precomputed once from a frozen calibration model + an explicit
    scale constraint), which this implementation does NOT use.
    """
    if Z.shape[0] < 2:
        raise ValueError("kf_score needs at least 2 rows")
    P = P.detach()                                 # hard API-level isolation
    z = Z - Z.mean(dim=0, keepdim=True)
    p = P - P.mean(dim=0, keepdim=True)           # P constant -> no grad anyway
    m = z.shape[0]
    szz = z.t() @ z / m                           # normalization term: full grad
    spp = p.t() @ p / m
    return kf_score_fixed(z, p, szz, spp, eps=eps)


def kf_loss(scores: Dict[str, torch.Tensor], alphas: Dict[str, float]) -> torch.Tensor:
    """L_KF = -sum_tau alpha_tau J_tau (maximize the spectral score)."""
    if not scores:
        return torch.zeros((), dtype=torch.float32)
    return -sum(float(alphas.get(tau, 1.0)) * j for tau, j in scores.items())


def score_rows_by_type(rows: List, interfaces: Dict[str, int]) -> Dict[str, list]:
    """Split CutRecord rows into per-tau (z list, p list) pairs."""
    by_tau = {}
    for r in rows:
        if r.tau not in interfaces:
            continue
        by_tau.setdefault(r.tau, []).append(r)
    return by_tau


def dedup_cut_rows(rows: List, fixed_maps):
    """One row per unique cut: z appears once, P rows of one cut are averaged.

    Stage-1 link cuts repeat the same z across 1 positive + k negative rows;
    counting those rows as independent samples gives the small-sample CCA
    saturation (rank(Z_c) is bounded by the number of UNIQUE cuts, not by the
    row count).  Averaging the P rows of a cut is the corresponding
    context-balancing weight.
    """
    by_cut: Dict[int, list] = {}
    for r in rows:
        by_cut.setdefault(int(r.cut_id), []).append(r)
    zs, ps = [], []
    for _, cut_rows in by_cut.items():
        zs.append(cut_rows[0].z)
        p_rows = [fixed_maps.pv(r.context, r.outcome) for r in cut_rows]
        ps.append(torch.stack(p_rows).mean(dim=0) if len(p_rows) > 1
                  else p_rows[0])
    return torch.stack(zs), torch.stack(ps)


class KyFanTracker:
    """Running (EMA) per-tau statistics for the Ky Fan score.

    The score is computed over UNIQUE cuts accumulated across batches — the
    batch-level stochastic approximation of the population statistics
    ``E[z z^T]`` etc.  The EMA history is detached (constant measurement
    history); the CURRENT batch's covariance contributions carry the full
    gradient, which preserves the scale invariance ``<grad_z J, z> = 0``
    within each step.  A tau only produces a score (and therefore a training
    signal) once its effective unique-cut count exceeds the threshold
    ``max(min_ratio * r_tau, min_abs)``: below that, in-batch canonical
    correlations saturate on independent noise (J -> min(r, M-1)), so the
    term is gated out instead of feeding noise gradients.

    ``update`` returns ``(scores, skipped)``; ``skipped`` lists taus that
    received data but did not meet the gate this step, so callers can alert.
    """

    def __init__(self, interfaces: Dict[str, int], *, ema_rho: float = 0.05,
                 min_ratio: float = 2.0, min_abs: int = 64, eps: float = 1e-4,
                 fixed_maps=None):
        self.interfaces = dict(interfaces)
        self.ema_rho = float(ema_rho)
        self.min_ratio = float(min_ratio)
        self.min_abs = int(min_abs)
        self.eps = float(eps)
        self.fixed_maps = fixed_maps
        self._state: Dict[str, Tuple[torch.Tensor, ...]] = {}

    def _threshold(self, tau: str) -> float:
        return max(self.min_ratio * int(self.interfaces[tau]),
                   float(self.min_abs))

    def update(self, rows: List):
        """``rows``: this batch's CutRecord rows (any tau)."""
        by_tau = score_rows_by_type(rows, self.interfaces)
        scores: Dict[str, torch.Tensor] = {}
        skipped: List[str] = []
        for tau, tau_rows in by_tau.items():
            if self.fixed_maps is None:
                raise ValueError("KyFanTracker requires fixed_maps")
            zs, ps = dedup_cut_rows(tau_rows, self.fixed_maps)
            m_u = int(zs.shape[0])
            if m_u < 2:
                skipped.append(tau)
                continue
            mu_z = zs.mean(dim=0)
            mu_p = ps.mean(dim=0)
            czz = zs.t() @ zs / m_u - torch.outer(mu_z, mu_z)   # grad via zs
            czp = zs.t() @ ps / m_u - torch.outer(mu_z, mu_p)   # grad via zs
            cpp = ps.t() @ ps / m_u - torch.outer(mu_p, mu_p)   # P constant
            st = self._state.get(tau)
            if st is None:
                n, a, b, d = float(m_u), czz, czp, cpp
            else:
                n0, a, b, d = st   # history is detached (constant history)
                rho = self.ema_rho
                # n is a CUMULATIVE unique-cut count (the gate asks "have we
                # ever seen enough unique cuts"), while the covariances are
                # EMA-smoothed.  An EMA'd n would converge to the batch size
                # and gate out any tau whose per-batch count sits below the
                # threshold forever (e.g. the root layer).
                n = n0 + float(m_u)
                a = rho * czz + (1 - rho) * a
                b = rho * czp + (1 - rho) * b
                d = rho * cpp + (1 - rho) * d
            # Store detached so the graph never spans batches.
            self._state[tau] = (n, a.detach(), b.detach(), d.detach())
            if n < self._threshold(tau):
                skipped.append(tau)
                continue
            r = int(self.interfaces[tau])
            lz = _cholesky_retry(a + _ridge(self.eps, a) * torch.eye(
                r, dtype=a.dtype, device=a.device))
            lp = _cholesky_retry(d + _ridge(self.eps, d) * torch.eye(
                d.shape[0], dtype=d.dtype, device=d.device))
            w = torch.cholesky_solve(b, lz)           # [r, m] = (A+e)^-1 B
            s = torch.cholesky_solve(b.t(), lp)       # [m, r] = (D+e)^-1 B^T
            scores[tau] = (w * s.t()).sum()
        return scores, skipped

    def effective_n(self, tau: str) -> float:
        st = self._state.get(tau)
        return float(st[0]) if st is not None else 0.0
