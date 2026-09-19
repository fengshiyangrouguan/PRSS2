"""Ky-Fan / CCA score core with paired OAS shrinkage -- VENDORED VERBATIM.

Source: PRSS2 repository, branch `fix/avg-lora-clock` / `develop_CCM`,
`src/rpbe/loss.py`. The functions below are copied unchanged so that
"OAS is enabled" means bit-identical behaviour to the frozen RPBE line rather
than a re-implementation that merely looks similar.

Why vendored rather than imported: Meta^n and PRSS2 are separate trees, and a
silent divergence here would change the training objective. If `loss.py`
changes upstream, this file must be re-synced deliberately.

Truncated to what the window needs: `_covs`, `_oas_alpha`, `_oas_shrink`,
`_score_from_covs`, `latent_z_adjoint`. `_matrix_diag` and the moment-adjoint
(`kf_adjoint`) path are NOT included -- the window uses the cut-level adjoint.

CRITICAL: `latent_z_adjoint(..., oas=True)` is the ONLY entry point that
enables OAS. `kf_score()` / `kf_adjoint()` in the upstream module have no OAS
parameter at all, and calling those while believing OAS is on is exactly the
silent failure this vendoring is meant to prevent.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch


def _covs(zc: torch.Tensor, pc: torch.Tensor, den: float,
          w: torch.Tensor = None):
    """Covariances from CENTERED float64 rows, explicitly symmetrized."""
    if w is not None:
        wc = w.reshape(-1, 1).to(zc.dtype)
        zc = zc * wc.sqrt()
        pc = pc * wc.sqrt()
    czz = zc.t() @ zc / den
    cpp = pc.t() @ pc / den
    czp = zc.t() @ pc / den
    sym_err = float((czz - czz.t()).detach().abs().max())
    return 0.5 * (czz + czz.t()), 0.5 * (cpp + cpp.t()), czp, sym_err


def _oas_alpha(cov: torch.Tensor, n: float) -> float:
    """OAS shrinkage intensity for one covariance (Chen et al. 2010).

        alpha = min(beta / delta, 1)
        beta  = (1 - 2/p) tr(C^2) + tr(C)^2
        delta = (n + 1 - 2/p) (tr(C^2) - tr(C)^2 / p)

    ``n`` is the effective degrees of freedom (the window's cluster-level D).
    The intensity is DETACHED (stop-gradient): it is a window statistic, never
    an optimization variable. ``delta`` at zero means the sample covariance is
    already isotropic -- full shrinkage to the identity then costs nothing, so
    the clamp is safe.
    """
    p = int(cov.shape[0])
    tr2 = torch.trace(cov) ** 2
    trc2 = (cov * cov).sum()
    beta = (1.0 - 2.0 / p) * trc2 + tr2
    delta = (n + 1.0 - 2.0 / p) * (trc2 - tr2 / p)
    denom = torch.clamp(delta, min=1e-12)
    alpha = torch.clamp(beta / denom, 0.0, 1.0)
    return float(alpha.detach())


def _oas_shrink(czz: torch.Tensor, czp: torch.Tensor,
                cpp: torch.Tensor, n: float):
    """Paired OAS covariance shrinkage (review round 8).

    Keeps the joint covariance CONSISTENT while damping the small-sample
    over-correlation: each side shrinks toward the SCALED identity
    (the standard OAS target mu*I with mu = tr(C)/d -- a plain identity breaks
    scale invariance and rewards memory-magnitude growth):

        C_ZZ' = (1 - a_Z) C_ZZ + a_Z mu_Z I
        C_PP' = (1 - a_P) C_PP + a_P mu_P I
        C_ZP' = sqrt((1 - a_Z)(1 - a_P)) C_ZP

    Returns the shrunk triple plus ``{"alpha_z", "alpha_p"}`` for diagnostics.
    The tiny ridge keeps its old role (numerical safety only).
    """
    az = _oas_alpha(czz, n)
    ap = _oas_alpha(cpp, n)
    mu_z = torch.trace(czz) / float(czz.shape[0])
    mu_p = torch.trace(cpp) / float(cpp.shape[0])
    czz_s = (1.0 - az) * czz + az * mu_z * torch.eye(
        czz.shape[0], dtype=czz.dtype, device=czz.device)
    cpp_s = (1.0 - ap) * cpp + ap * mu_p * torch.eye(
        cpp.shape[0], dtype=cpp.dtype, device=cpp.device)
    czp_s = ((1.0 - az) * (1.0 - ap)) ** 0.5 * czp
    return czz_s, czp_s, cpp_s, {"alpha_z": az, "alpha_p": ap}


def _score_from_covs(czz: torch.Tensor, czp: torch.Tensor,
                     cpp: torch.Tensor, eps: float,
                     variant: str = "full_balancing"):
    """Scale-normalized Ky Fan score with gradient; ``(J, diag)``.

    ``full_balancing``: ``J_KF = ||S_ZZ^-1/2 S_ZP S_PP^-1/2||_F^2``; the
    normalization ``A = Czz / mean(diag Czz)`` keeps the ridge scale-free and
    keeps the score exactly scale-invariant in ``Z`` and ``P``;
    ``mean(diag)`` is NOT detached so the gradient is the true gradient of the
    normalized objective.
    """
    sz = czz.diagonal().mean()
    sp = cpp.diagonal().mean()
    if float(sz.detach()) <= 0.0:
        return None, {"failed": "nonpositive_scale",
                      "scale_z": float(sz.detach()),
                      "scale_p": float(sp.detach())}
    if variant == "full_balancing":
        if float(sp.detach()) <= 0.0:
            return None, {"failed": "nonpositive_scale",
                          "scale_z": float(sz.detach()),
                          "scale_p": float(sp.detach())}
        a = czz / sz
        b = cpp / sp
        c = czp / torch.sqrt(sz * sp)
        r, q = a.shape[0], b.shape[0]
        a = 0.5 * (a + a.t()) + eps * torch.eye(
            r, dtype=a.dtype, device=a.device)
        b = 0.5 * (b + b.t()) + eps * torch.eye(
            q, dtype=b.dtype, device=b.device)
        lz, info_z = torch.linalg.cholesky_ex(a)
        lp, info_p = torch.linalg.cholesky_ex(b)
        if bool((info_z != 0).any()) or bool((info_p != 0).any()):
            return None, {"failed": "cholesky",
                          "info_z": int(info_z.max().item()),
                          "info_p": int(info_p.max().item()),
                          "scale_z": float(sz.detach()),
                          "scale_p": float(sp.detach())}
        w = torch.linalg.solve_triangular(lz, c, upper=False)
        k = torch.linalg.solve_triangular(lp, w.t(), upper=False).t()
        return k.square().sum(), {"failed": None,
                                  "scale_z": float(sz.detach()),
                                  "scale_p": float(sp.detach())}
    if variant == "diagonal":
        dz = czz.diagonal()
        dp = cpp.diagonal()
        if float((dz > 0).all().detach()) and \
                float((dp > 0).all().detach()):
            c = czp / torch.sqrt(dz[:, None] * dp[None, :])
            return c.square().sum(), {"failed": None,
                                      "scale_z": float(sz.detach()),
                                      "scale_p": float(sp.detach())}
        return None, {"failed": "nonpositive_scale",
                      "scale_z": float(sz.detach()),
                      "scale_p": float(sp.detach())}
    raise ValueError("unknown variant {}".format(variant))


def latent_z_adjoint(z_rows, p_rows, w, cut_ids, mu_z, mu_p, D,
                     eps, strict=False, oas=False):
    """Contract the moment adjoint onto CUT-LEVEL z-adjoints.

    At window close, the whole-window score F(S) is replayed on the window's z
    ROWS as small-matrix leaves; one backward yields ``g_i = grad_{z_i} L_KF``
    per row, and rows sharing a cut id are merged ``g_v = sum``.

    Returns ``(j_float, g_by_cut, score_diag)``.

    ``oas=True`` is REQUIRED by the task book (§2.7). The default is False,
    which is why the task book says to never reach this through any other
    entry point.
    """
    z = z_rows.clone().double().requires_grad_(True)
    zc = z - mu_z
    pc = p_rows.double() - mu_p
    sw = w.double().sqrt().reshape(-1, 1)
    mzz = (zc * sw).t() @ (zc * sw)
    mzp = (zc * sw).t() @ (pc * sw)
    mpp = (pc * sw).t() @ (pc * sw)
    czz = mzz / D
    czp = mzp / D
    cpp = mpp / D
    if oas:
        # Paired OAS shrinkage with stop-gradded intensities (window
        # statistics, never optimization variables).
        czz, czp, cpp, shrink_diag = _oas_shrink(czz, czp, cpp, D)
    else:
        shrink_diag = {}
    j, score_diag = _score_from_covs(czz, czp, cpp, eps)
    if score_diag["failed"] is not None:
        if strict:
            raise RuntimeError("latent_z_adjoint close failed: {}"
                               .format(score_diag))
        return None, None, score_diag
    score_diag.update(shrink_diag)
    j.backward()
    g = z.grad.detach()               # [M, r]
    g_by_cut: Dict[tuple, torch.Tensor] = {}
    for cid, gi in zip(cut_ids, g):
        if cid in g_by_cut:
            g_by_cut[cid] = g_by_cut[cid] + gi
        else:
            g_by_cut[cid] = gi
    return float(j.detach()), g_by_cut, score_diag
