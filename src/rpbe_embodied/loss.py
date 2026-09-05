"""rpbe_embodied.loss — high-dimensional (4096D) RPBE statistics.

Sample-space dual balancing (plan §17-19): the cognitive state z is 4096D,
so the 4096x4096 feature covariance is forbidden.  The dual computes the
EXACT same scale-normalized full-balancing objective in the N x N sample
space (N = window rows):

    X_t = sqrt(w)(z - mu_z);  Q_t = sqrt(w)(p - mu_p)
    s_Z = mean diag(C_ZZ) = sum_i w_i ||zc_i||^2 / (D * 4096)
    X~ = X / sqrt(D s_Z);  Q~ = Q / sqrt(D s_P)
    K_Z = X~ X~^T;  K_P = Q~ Q~^T
    H = K @ cholesky_solve(I, chol(K + eps I))
    J_dual = tr(H_Z H_P)  ==  ||A^{-1/2} C B^{-1/2}||_F^2

Failure contract mirrors rpbe.loss._score_from_covs: cholesky failure ->
differentiable zero + diagnostics; strict=True raises.

Gate (plan §20): test_rpbe_embodied_loss.py::TestDualExactEquivalence must
pass before any LIBERO-Mem training.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

from .records import EmbodiedCutRow


def _weighted_center_and_scale(
    z: torch.Tensor, p: torch.Tensor, w: torch.Tensor,
    cut_ids: List[tuple],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Weighted centering + scale-normalized whitening (fp64).

    Returns (X~, Q~, s_Z, s_P, D).  D is the clustered dof correction
    D = W - W2_cut/W with W2_cut = sum_v (sum_h w_{v,h})^2.
    """
    z64 = z.double()
    p64 = p.double()
    w64 = w.double()
    W = w64.sum()
    # clustered W2 by cut
    W2_cut = torch.zeros((), dtype=torch.float64, device=z64.device)
    acc: Dict[tuple, float] = {}
    for cid, wi in zip(cut_ids, w64.tolist()):
        acc[cid] = acc.get(cid, 0.0) + wi
    for v in acc.values():
        W2_cut = W2_cut + v * v
    D = (W - W2_cut / W).item()
    if D <= 0:
        raise ValueError(f"degenerate window: D={D}")

    mu_z = (z64 * w64[:, None]).sum(0) / W
    mu_p = (p64 * w64[:, None]).sum(0) / W
    zc = (z64 - mu_z) * w64[:, None].sqrt()
    pc = (p64 - mu_p) * w64[:, None].sqrt()

    d_z = z64.shape[1]
    d_p = p64.shape[1]
    s_Z = (zc * zc).sum() / (D * d_z)
    s_P = (pc * pc).sum() / (D * d_p)
    if s_Z <= 0 or s_P <= 0:
        raise ValueError(f"degenerate scale: s_Z={s_Z} s_P={s_P}")
    Xt = zc / (D * s_Z).sqrt()
    Qt = pc / (D * s_P).sqrt()
    return Xt, Qt, s_Z.item(), s_P.item(), D


def _hat(K: torch.Tensor, eps: float, strict: bool) -> Optional[torch.Tensor]:
    """H = K @ (K + eps I)^{-1} via cholesky (no explicit inverse)."""
    try:
        L = torch.linalg.cholesky(K + eps * torch.eye(K.shape[0], dtype=K.dtype, device=K.device))
    except torch._C._LinAlgError:  # noqa: PERF203
        if strict:
            raise
        return None
    Id = torch.eye(K.shape[0], dtype=K.dtype, device=K.device)
    return K @ torch.cholesky_solve(Id, L)


def dual_full_score(
    z: torch.Tensor, p: torch.Tensor, w: torch.Tensor,
    cut_ids: List[tuple], eps: float = 1e-4, strict: bool = False,
) -> Tuple[torch.Tensor, dict]:
    """Sample-space dual score; z carries gradients (leaf inside adjoint)."""
    Xt, Qt, s_Z, s_P, D = _weighted_center_and_scale(z, p, w, cut_ids)
    K_Z = Xt @ Xt.T
    K_P = Qt @ Qt.T
    H_Z = _hat(K_Z, eps, strict)
    H_P = _hat(K_P, eps, strict)
    diag = {"s_Z": s_Z, "s_P": s_P, "D": D, "N": z.shape[0]}
    if H_Z is None or H_P is None:
        return z.sum() * 0.0, {**diag, "failed": "cholesky"}
    J = (H_Z * H_P.T).sum()   # tr(H_Z @ H_P) = sum(H_Z . H_P^T) (symmetric)
    return J, diag


def dual_latent_z_adjoint(
    z_detached: torch.Tensor, p: torch.Tensor, w: torch.Tensor,
    cut_ids: List[tuple], eps: float = 1e-4, strict: bool = False,
) -> Tuple[float, Dict[tuple, torch.Tensor], dict]:
    """Latent adjoint: J as a function of a temp leaf Z; per-cut gradients.

    Returns (j_float, g_by_cut, diag).  g_by_cut[cut_id] = sum_h dJ/dz_{v,h}
    (plan §23: two horizon rows of one cut share the merged state).
    """
    z = z_detached.clone().double().requires_grad_(True)
    J, diag = dual_full_score(z, p, w, cut_ids, eps=eps, strict=strict)
    if diag.get("failed"):
        return float(J.item()), {}, diag
    (g,) = torch.autograd.grad(J, z, retain_graph=False)
    g = g.detach().float()
    g_by_cut: Dict[tuple, torch.Tensor] = {}
    for cid, gi in zip(cut_ids, g):
        g_by_cut[cid] = g_by_cut.get(cid, torch.zeros_like(gi)) + gi
    return float(J.item()), g_by_cut, diag


def diag_score(
    z: torch.Tensor, p: torch.Tensor, w: torch.Tensor,
    cut_ids: List[tuple], eps: float = 1e-4, strict: bool = False,
) -> Tuple[torch.Tensor, dict]:
    """RPBE-Diag-4096 (plan §22): diagonal whitening only, O(4096 * m).

    dz = sum w zc^2 / D; dp likewise; C_zp = zc^T pc / D;
    J_diag = || C_zp / sqrt(dz dp^T) ||_F^2 (dz carries gradients).
    """
    z64 = z.double()
    p64 = p.double()
    w64 = w.double()
    W = w64.sum()
    W2_cut = torch.zeros((), dtype=torch.float64, device=z64.device)
    acc: Dict[tuple, float] = {}
    for cid, wi in zip(cut_ids, w64.tolist()):
        acc[cid] = acc.get(cid, 0.0) + wi
    for v in acc.values():
        W2_cut = W2_cut + v * v
    D = W - W2_cut / W
    if D <= 0:
        if strict:
            raise ValueError("degenerate D")
        return z.sum() * 0.0, {"failed": "degenerate_D"}
    mu_z = (z64 * w64[:, None]).sum(0) / W
    mu_p = (p64 * w64[:, None]).sum(0) / W
    zc = (z64 - mu_z) * w64[:, None].sqrt()
    pc = (p64 - mu_p) * w64[:, None].sqrt()
    dz = (zc * zc).sum(0) / D
    dp = (pc * pc).sum(0) / D
    C_zp = (zc.T @ pc) / D
    # A_{ij} = C_zp_{ij} / sqrt(dz_i * dp_j)
    scale = (dz[:, None] * dp[None, :]).clamp(min=1e-30).sqrt()
    J = ((C_zp / scale) ** 2).sum()
    return J, {"D": D.item()}


def diag_latent_z_adjoint(
    z_detached: torch.Tensor, p: torch.Tensor, w: torch.Tensor,
    cut_ids: List[tuple], eps: float = 1e-4, strict: bool = False,
) -> Tuple[float, Dict[tuple, torch.Tensor], dict]:
    z = z_detached.clone().double().requires_grad_(True)
    J, diag = diag_score(z, p, w, cut_ids, eps=eps, strict=strict)
    if diag.get("failed"):
        return float(J.item()), {}, diag
    (g,) = torch.autograd.grad(J, z)
    g = g.detach().float()
    g_by_cut: Dict[tuple, torch.Tensor] = {}
    for cid, gi in zip(cut_ids, g):
        g_by_cut[cid] = g_by_cut.get(cid, torch.zeros_like(gi)) + gi
    return float(J.item()), g_by_cut, diag


class EmbodiedRPBEWindow:
    """Thin-row window (plan §21): store detached rows; assemble Z,P,W at
    close; dispatch to full_dual / diag adjoint.  Gate on unique merges."""

    def __init__(self, variant: str = "full_dual", eps: float = 1e-4,
                 min_ratio: float = 2.0, min_abs: int = 128,
                 strict: bool = False):
        assert variant in ("full_dual", "diag")
        self.variant = variant
        self.eps = eps
        self.min_ratio = min_ratio
        self.min_abs = min_abs
        self.strict = strict
        self.rows: Dict[tuple, EmbodiedCutRow] = {}
        self.closed = False

    def add(self, rows: List[EmbodiedCutRow]) -> None:
        assert not self.closed
        for r in rows:
            # review ruling B5: a statistics window must never mix merge
            # states written by different parameter versions; the trainer
            # closes windows at every repr boundary, this is fail-fast
            if self.rows:
                existing_v = next(iter(self.rows.values())).param_version
                assert r.param_version == existing_v, (
                    f"window mixes param versions {existing_v} and "
                    f"{r.param_version}")
            key = r.cut_id + (r.horizon,)
            self.rows[key] = r   # row_id dedup: distinct horizons both stay

    @property
    def n_unique_cuts(self) -> int:
        return len({r.cut_id for r in self.rows.values()})

    def ready(self) -> bool:
        # review ruling B5: the gate is exactly min_abs (the min_ratio * m
        # product made --kf-min-abs misleading; that coupling is removed)
        return self.n_unique_cuts >= self.min_abs

    def discard(self) -> int:
        """Drop an underfull window (review ruling B5: windows never span
        parameter versions; the trainer discards them at repr boundaries)."""
        n = self.n_unique_cuts
        self.closed = True
        return n

    def close(self) -> Tuple[float, Dict[tuple, torch.Tensor], dict]:
        """Assemble thin rows -> adjoint -> (j_float, g_by_cut, diagnostics)."""
        assert not self.closed
        self.closed = True
        rs = list(self.rows.values())
        if not rs:
            return 0.0, {}, {"failed": "empty_window"}
        # window statistics run on CPU fp64 (rows come from GPU detach or
        # CPU futures; N x 4096 is tiny)
        z = torch.stack([r.z.detach().cpu() for r in rs])
        p = torch.stack([r.outcome.detach().cpu() for r in rs])
        w = torch.tensor([r.weight for r in rs], dtype=torch.float64)
        cut_ids = [r.cut_id for r in rs]
        fn = (dual_latent_z_adjoint if self.variant == "full_dual"
              else diag_latent_z_adjoint)
        j, g_by_cut, diag = fn(z, p, w, cut_ids, eps=self.eps, strict=self.strict)
        diag["n_rows"] = len(rs)
        diag["n_unique_cuts"] = self.n_unique_cuts
        diag["n_unique_episodes"] = len({r.cut_id[0] for r in rs})
        return j, g_by_cut, diag


def gamma_replay_loss(gamma, m_a: torch.Tensor, m_b: torch.Tensor,
                      cotangents: Dict[tuple, torch.Tensor],
                      merge_keys: List[tuple]) -> torch.Tensor:
    """Replay the Gamma merges against given cotangents (plan §23/§25).

    z_hat = Gamma(m_a.detach(), m_b.detach()) [V, dim];
    returns sum_v <sg(g_v), z_hat_v>.  Caller fixes the sign:
      task part:   + <g_task, z_hat>   (minimize)
      rpbe part:   - <g_rpbe, z_hat>   (maximize J)
    """
    z_hat = gamma(m_a.detach(), m_b.detach())           # [V, dim]
    gs = torch.stack([cotangents[k].to(z_hat.device, dtype=z_hat.dtype)
                      for k in merge_keys])
    return (gs.detach() * z_hat).sum()
