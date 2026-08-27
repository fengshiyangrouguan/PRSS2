"""Ky Fan spectral score: the single component-internal loss.

For one interface type tau, with rows ``Z`` (compressed states, gradient
connected) and ``P`` (fixed joint tests psi(c,y), constant)::

    J_tau = tr[ (Sigma_ZZ + e_z I)^-1  Sigma_ZP  (Sigma_PP + e_p I)^-1  Sigma_PZ ]

computed with two Cholesky factorizations and triangular solves — no SVD, no
explicit inverses.  The training loss maximizes the per-interface score
(equivalently minimizes the positive form sum_tau alpha_tau (d_tau - J_tau)
which differs only by a theta-constant).

Whitening statistics are estimated per batch; ridge terms are relative to
trace/dim so the same epsilon behaves across scales.
"""

from typing import Dict, List

import torch


def _ridge(eps: float, sigma: torch.Tensor) -> torch.Tensor:
    return eps * torch.trace(sigma).clamp(min=0.0) / sigma.shape[0]


def kf_score_fixed(z_c: torch.Tensor, p_c: torch.Tensor,
                   szz: torch.Tensor, spp: torch.Tensor,
                   eps: float = 1e-4) -> torch.Tensor:
    """J with all whitening statistics held constant.

    ``z_c`` carries gradient (the only gradient path is the cross term
    ``S_ZP = z_c^T p_c / M``); ``szz``/``spp``/``p_c`` are fixed measurement
    scales.  This is the gradcheck-able core; ``kf_score`` wraps it with
    stop-gradient batch statistics.
    """
    m = z_c.shape[0]
    szp = z_c.t() @ p_c / m
    lz = torch.linalg.cholesky(szz + _ridge(eps, szz) * torch.eye(
        szz.shape[0], dtype=szz.dtype, device=szz.device))
    lp = torch.linalg.cholesky(spp + _ridge(eps, spp) * torch.eye(
        spp.shape[0], dtype=spp.dtype, device=spp.device))
    w = torch.cholesky_solve(szp, lz)             # [r, m] = (S_ZZ+e)^-1 S_ZP
    s = torch.cholesky_solve(szp.t(), lp)         # [m, r] = (S_PP+e)^-1 S_PZ
    return (w * s.t()).sum()  # tr[(S_ZZ+e)^-1 S_ZP (S_PP+e)^-1 S_PZ]


def kf_score(Z: torch.Tensor, P: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """J_tau for one type.  ``Z``: [M, r] (gradient flows); ``P``: [M, m] (detached).

    Whitening statistics are STOP-GRADIENT measurement scales per the method
    definition (``z_w = Sigma_ZZ^{-1/2}(q - E q)`` with population statistics):
    the gradient may only flow through the centered cross path
    ``Sigma_ZP = Z_c^T P_c / M``.  Letting the batch ``Sigma_ZZ`` carry
    gradient would give theta a spurious channel to inflate J by shrinking
    its own covariance instead of learning better cross-correlations.
    """
    if Z.shape[0] < 2:
        raise ValueError("kf_score needs at least 2 rows")
    z = Z - Z.mean(dim=0, keepdim=True).detach()
    p = P - P.mean(dim=0, keepdim=True).detach()
    m = z.shape[0]
    szz = (z.t() @ z / m).detach()                # measurement scale (fixed)
    spp = (p.t() @ p / m).detach()                # P is constant anyway
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


def kf_scores_from_rows(rows: List, interfaces: Dict[str, int], fixed_maps,
                        min_cuts_per_type: int = 32, eps: float = 1e-4):
    """Stack rows per tau, compute P (no grad) and J_tau for each type.

    Returns (scores, skipped): types below ``min_cuts_per_type`` are skipped
    so a nearly-empty batch never feeds a rank-deficient covariance.
    """
    by_tau = score_rows_by_type(rows, interfaces)
    scores: Dict[str, torch.Tensor] = {}
    skipped: List[str] = []
    for tau, tau_rows in by_tau.items():
        if len(tau_rows) < int(min_cuts_per_type):
            skipped.append(tau)
            continue
        zs = torch.stack([r.z for r in tau_rows])                    # [M, r]
        ps = fixed_maps.pv_batch([r.context for r in tau_rows],
                                 [r.outcome for r in tau_rows])      # [M, m]
        scores[tau] = kf_score(zs, ps, eps=eps)
    return scores, skipped
