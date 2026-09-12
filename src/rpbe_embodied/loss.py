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

import math
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
    """Window over detached cut rows PLUS, per episode, the full merge DAG
    (edges + raw leaf inputs) so at close the merged state of every cut is
    REBUILT by recursively replaying the FIXED merge trace with the CURRENT
    Gamma -- no cross-version merged-state is ever reused, and multi-episode
    windows keep every episode's replay records (never dropped via a shared
    registry that a task boundary would clear)."""

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
        self.n_dropped_version = 0
        self.edges: Dict[tuple, tuple] = {}      # (eid,node) -> (left_id, right_id)
        self.leaf_state: Dict[tuple, torch.Tensor] = {}  # (eid,node) -> raw input

    def add_records(self, recs) -> None:
        """Retain an episode's full merge DAG (edges + child inputs) so its
        cuts can be replayed after later task boundaries clear any registry."""
        for rec in recs:
            self.edges[(rec.episode_id, rec.node_id)] = (rec.left_id, rec.right_id)
            self.leaf_state.setdefault((rec.episode_id, rec.left_id), rec.left_state)
            self.leaf_state.setdefault((rec.episode_id, rec.right_id), rec.right_state)

    def add(self, rows: List[EmbodiedCutRow]) -> None:
        assert not self.closed
        for r in rows:
            self.rows[r.cut_id + (r.horizon,)] = r

    @property
    def n_unique_cuts(self) -> int:
        return len({r.cut_id for r in self.rows.values()})

    @property
    def n_unique_episodes(self) -> int:
        return len({r.cut_id[0] for r in self.rows.values()})

    def ready(self) -> bool:
        return self.n_unique_cuts >= self.min_abs

    def discard(self) -> int:
        n = self.n_unique_cuts
        self.closed = True
        return n

    def close(self, merge_fn=None, n_perm: int = 0):
        """Returns (j_float, g_by_cut, replay_inputs, diag).

        replay_inputs[cut_id] = (child_left, child_right) rebuilt with the
        CURRENT Gamma by recursing the fixed merge trace, so the caller can
        replay <g, Gamma(child_left, child_right)> without any shared registry.
        n_perm > 0 runs a CUT-BLOCK permutation null (both horizon rows of a
        cut move together); 0 disables it (formal training).
        """
        assert not self.closed
        self.closed = True
        rs = list(self.rows.values())
        if not rs:
            return 0.0, {}, {}, {"failed": "empty_window"}
        replay_inputs: Dict[tuple, tuple] = {}
        if merge_fn is not None and self.edges:
            memo: Dict[tuple, torch.Tensor] = {}

            def build(eid, nid):
                k = (eid, nid)
                if k in memo:
                    return memo[k]
                if k in self.edges:
                    li, ri = self.edges[k]
                    m = merge_fn(build(eid, li), build(eid, ri))
                else:
                    m = self.leaf_state[k]
                memo[k] = m
                return m

            zs = []
            for r in rs:
                eid = r.cut_id[0]
                if (eid, r.node_id) in self.edges:
                    li, ri = self.edges[(eid, r.node_id)]
                    cl = build(eid, li); cr = build(eid, ri)
                    replay_inputs.setdefault(r.cut_id, (cl, cr))
                    zs.append(merge_fn(cl, cr))
                else:
                    zs.append(r.z)
            z = torch.stack([t.detach().float().cpu() for t in zs])
        else:
            z = torch.stack([r.z.detach().cpu() for r in rs])
        p = torch.stack([r.outcome.detach().cpu() for r in rs])
        w = torch.tensor([r.weight for r in rs], dtype=torch.float64)
        cut_ids = [r.cut_id for r in rs]
        self.last_zpw = (z, p, w, cut_ids)   # kept for the mode-level audit
        fn = (dual_latent_z_adjoint if self.variant == "full_dual"
              else diag_latent_z_adjoint)
        j, g_by_cut, diag = fn(z, p, w, cut_ids, eps=self.eps, strict=self.strict)
        # null-signal check with a CUT-BLOCK permutation (both horizon rows of
        # a cut move together).  Disabled (n_perm=0) in formal training to
        # avoid the CPU Cholesky cost; calibration uses n_perm>=128.
        if n_perm > 0:
            try:
                scoring = (dual_full_score if self.variant == "full_dual"
                           else diag_score)
                groups = {}
                for i, cid in enumerate(cut_ids):
                    groups.setdefault(cid, []).append(i)
                bundles = list(groups.values())
                g = torch.Generator().manual_seed(12345)
                null = []
                for _ in range(int(n_perm)):
                    order = torch.randperm(len(bundles), generator=g).tolist()
                    idx = [i for c in order for i in bundles[c]]
                    jn, _ = scoring(z, p[idx], w, cut_ids, eps=self.eps,
                                    strict=False)
                    null.append(float(jn))
                null.sort()
                diag["J_real"] = float(j)
                diag["J_shuffled"] = float(sum(null) / len(null))
                diag["J_perm_p95"] = float(null[min(len(null) - 1,
                                                    int(0.95 * len(null)))])
                diag["J_gap"] = float(j) - float(sum(null) / len(null))
                diag["n_perm"] = int(n_perm)
            except Exception as e:  # never let the diagnostic break training
                diag["null_failed"] = str(e)
        diag["n_rows"] = len(rs)
        diag["n_unique_cuts"] = self.n_unique_cuts
        diag["n_unique_episodes"] = self.n_unique_episodes
        return j, g_by_cut, replay_inputs, diag


def _mat_sqrt(A: torch.Tensor) -> torch.Tensor:
    w, V = torch.linalg.eigh(A)
    w = w.clamp(min=0.0)
    return V @ torch.diag(w.sqrt()) @ V.t()


def dual_latent_z_adjoint_modes(
    z_detached: torch.Tensor, p: torch.Tensor, w: torch.Tensor,
    cut_ids: List[tuple], eps: float = 1e-4, n_modes: int = 8,
    strict: bool = False,
):
    """Component-wise decomposition of J into PREDICTIVE MODES.

    J = tr(H_Z H_P) is decomposed as the spectrum of the symmetric
    S = H_Z^{1/2} H_P H_Z^{1/2}: the eigenvalues J_k = rho_k^2 are the squared
    canonical correlations between the memory states z and the future
    outcomes p (the "predictive modes").  Returns
    (modes, J) with modes[k] = (J_k, g_by_cut_k) where g_by_cut_k = dJ_k/dz
    summed per cut.  Used to audit cos(g_{J_k}, g_task) per mode.
    """
    z = z_detached.clone().double().requires_grad_(True)
    Xt, Qt, s_Z, s_P, D = _weighted_center_and_scale(z, p, w, cut_ids)
    K_Z = Xt @ Xt.t()
    K_P = Qt @ Qt.t()
    H_Z = _hat(K_Z, eps, strict)
    H_P = _hat(K_P, eps, strict)
    if H_Z is None or H_P is None:
        return {}, 0.0, {"failed": "cholesky"}
    HZh = _mat_sqrt(H_Z)
    S = HZh @ H_P @ HZh
    lam, _ = torch.linalg.eigh(S)              # ascending, real
    order = torch.argsort(lam, descending=True)
    modes = {}
    nm = min(n_modes, lam.shape[0])
    for m in range(nm):
        Jk = lam[order[m]]
        try:
            (g,) = torch.autograd.grad(Jk, z, retain_graph=True)
        except Exception:
            continue
        g = g.detach().float()
        gbc: Dict[tuple, torch.Tensor] = {}
        for cid, gi in zip(cut_ids, g):
            gbc[cid] = gbc.get(cid, torch.zeros_like(gi)) + gi
        modes[m] = (float(Jk.detach()), gbc)
    return modes, float(lam.clamp(min=0).sum().detach()), {"n_modes": len(modes)}


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


# ---------------------------------------------------------------------------
# Interface-wise feasibility projection (VLA port of the TGN final algorithm)
#
# TGN's B3 ruling: RPBE is NOT a second optimisation objective.  Additive
# ``L = L_task - lambda*J`` was retired because ``J`` is first-order
# orthogonal AND second-order flat to the task loss (compaudit: cos ~ 0,
# D2 ~ 0), so ``lambda`` has no task meaning.  The migration keeps the task
# gradient as the ONLY direction and demotes RPBE to a set of local
# preservation CONSTRAINTS:
#
#     d* = argmin_d 1/2 ||d - t||_2^2   s.t.  g_i . d >= -kappa ||g_i|| ||t||
#
# with ``t = -g_task`` the task descent direction and one half-space per
# memory merge/compression INTERFACE i (the VLA analogue of a TGN tree).
# Only the Gamma/compressor gradient is replaced (p.grad = -d*); every host
# parameter keeps its untouched task gradient.  One clip, one AdamW step.
# The three helpers below are verbatim copies of scripts/train_lamp.py
# (branch ``lamp_rpbe_project``) so the VLA and TGN lines cannot drift.
# ---------------------------------------------------------------------------


def _fista_nonneg(Q: torch.Tensor, c: torch.Tensor, iters: int) -> torch.Tensor:
    """min_{mu>=0} 1/2 mu^T Q mu - c^T mu  (Q PSD) by accelerated projected GD.

    Tiny (N x N, N ~ #interfaces) dual of the tree-wise feasibility QP."""
    L = float(torch.linalg.eigvalsh(Q).max().clamp(min=1e-12))
    mu = torch.zeros_like(c)
    y = mu.clone()
    tk = 1.0
    for _ in range(iters):
        grad = Q @ y - c
        mu_new = torch.clamp(y - grad / L, min=0.0)
        tk_new = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * tk * tk))
        y = mu_new + ((tk - 1.0) / tk_new) * (mu_new - mu)
        mu, tk = mu_new, tk_new
    return mu


def kappa_at(step: int, kappa_max: float, start: int, end: int,
             kappa_to: float = 1.0) -> float:
    """kappa schedule for the per-interface guardrail.

    kappa_max (e.g. 0.05) until ``start`` (step), then LINEARLY to ``kappa_to``
    at ``end``.  NOTE the direction: in ``g_i.d >= -kappa*||g_i||*||t||`` a
    LARGER kappa is LOOSER; kappa >= 1 never binds (== pure task), kappa = 0 is
    the strictest.  start < 0 disables annealing (constant kappa_max)."""
    if start is None or start < 0:
        return float(kappa_max)
    if step < start:
        return float(kappa_max)
    if end is None or end <= start:
        return float(kappa_to)
    if step >= end:
        return float(kappa_to)
    f = (float(step) - float(start)) / float(end - start)
    return float(kappa_max) + f * (float(kappa_to) - float(kappa_max))


def per_cut_influence_grads(
    gamma, cotangents: Dict[tuple, torch.Tensor],
    input_map: Dict[tuple, Tuple[torch.Tensor, torch.Tensor]],
    keys: List[tuple], params: Optional[List[torch.Tensor]] = None,
    device: str = "cuda", max_rows: int = 0, seed: int = 0,
) -> Tuple[Optional[torch.Tensor], List[tuple]]:
    """Interface-wise RPBE influence gradients, ONE ROW PER INTERFACE.

    g_i = (dz_i/dGamma)^T a_i, where a_i = dJ/dz_i is the window adjoint for
    interface i (``cotangents[cut_id]``).  Rows are taken SEPARATELY -- never
    summed -- so the interface-interface geometry (e.g. g_1 ~ -g_2) survives
    into the feasibility QP instead of cancelling.  This is the exact VLA
    analogue of ``tree_wise_influence_grads`` in the TGN trainer.

    The window's J is still estimated JOINTLY (one ``dual_latent_z_adjoint``
    over the whole window), so each a_i is interface i's effect ON THE JOINT J.

    Returns (G [n, P] fp32 on ``device``, used_keys).  ``max_rows`` > 0
    deterministically subsamples the interfaces to bound memory: P ~ 0.574M
    for the 2-token merge operator, so n=256 already costs ~0.6 GB fp32.
    """
    if params is None:
        params = list(gamma.parameters())
    cand = [k for k in keys if k in cotangents and k in input_map]
    if max_rows and len(cand) > max_rows:
        g = torch.Generator().manual_seed(int(seed))
        order = torch.randperm(len(cand), generator=g).tolist()[:max_rows]
        cand = [cand[i] for i in order]
    if not cand or not params:
        return None, []
    md = params[0].dtype
    rows: List[torch.Tensor] = []
    used: List[tuple] = []
    with torch.autocast("cuda", enabled=False):
        m_a = torch.stack([input_map[k][0] for k in cand]).to(device, dtype=md)
        m_b = torch.stack([input_map[k][1] for k in cand]).to(device, dtype=md)
        z = gamma(m_a, m_b)                                   # [n, dim]
        C = torch.stack([cotangents[k].reshape(-1) for k in cand]).to(
            device, dtype=torch.float32)
        n = z.shape[0]
        for i in range(n):
            # share ONE forward; N separate VJPs keep the rows independent.
            # upcast z (differentiable) so the fp32 adjoint is not quantised.
            term = (C[i] * z[i].float()).sum()
            grads = torch.autograd.grad(term, params,
                                        retain_graph=(i < n - 1),
                                        allow_unused=True)
            rows.append(torch.cat([
                (gr if gr is not None else torch.zeros_like(p)
                 ).reshape(-1).float() for gr, p in zip(grads, params)]))
            used.append(cand[i])
    return torch.stack(rows), used


def treewise_feasibility_projection(
    g_task_gamma: List[torch.Tensor], gamma_params: List[torch.Tensor],
    G: Optional[torch.Tensor], kappa: float, iters: int = 400,
    min_norm: float = 1e-9,
) -> dict:
    """Interface-wise RPBE Feasibility Projection (TGN final algorithm).

    One global task direction t = -g_task; each valid interface i contributes a
    half-space g_i.d >= -kappa*||g_i||*||t||.  Solve

        d* = argmin_d 1/2||d - t||^2   s.t.  G d >= b,  b_i=-kappa||g_i||.||t||

    through the N x N dual (mu >= 0): d* = t + sum_i mu_i g_i.  Writes ONLY
    Gamma: p.grad = -d* = g_task - sum_i mu_i g_i.  If ``G`` carries no valid
    row, nothing is written -- the accumulated task gradient stays in p.grad.

    This is a PRE-OPTIMIZER constraint on the raw gradient combination; it does
    NOT claim AdamW's realized displacement lies in the half-space.  All
    constraints enter the QP (a repair for interface 1 may otherwise push
    interface 7 into violation); inactive ones simply get mu=0.
    """
    sizes = [p.numel() for p in gamma_params]
    t = torch.cat([-x.flatten().float() for x in g_task_gamma])
    nt = float(t.norm())
    diag = {"proj_n_trees": int(G.shape[0]) if G is not None else 0,
            "proj_n_valid": 0, "proj_n_active_init": 0, "proj_n_mu_pos": 0,
            "proj_norm_t": nt, "proj_corr_ratio": 0.0, "proj_cos_min": 0.0,
            "proj_min_slack_after": float("nan"),
            "proj_max_viol_before": 0.0}
    if G is None or G.numel() == 0 or nt == 0.0:
        return diag
    Gf = G.flatten(1).float()
    ng = Gf.norm(dim=1)
    valid = ng > min_norm
    if not bool(valid.any()):
        return diag
    Gv = Gf[valid]
    ngv = ng[valid]
    Gt = Gv @ t
    b = -kappa * ngv * nt
    cos_init = Gt / (ngv * nt)
    diag["proj_n_valid"] = int(Gv.shape[0])
    diag["proj_n_active_init"] = int((cos_init < -kappa).sum())
    diag["proj_cos_min"] = float(cos_init.min())
    diag["proj_max_viol_before"] = float(
        torch.clamp(-cos_init - kappa, min=0.0).max())
    Q = Gv @ Gv.t()
    c = b - Gt
    mu = _fista_nonneg(Q, c, iters)
    corr = Gv.t() @ mu                 # = sum_i mu_i g_i  (= d* - t)
    d = t + corr
    with torch.no_grad():
        for p, cp, gt in zip(gamma_params, torch.split(corr, sizes),
                             g_task_gamma):
            # grad = -d* = g_task - sum_i mu_i g_i, in the param's own dtype
            p.grad = (gt.reshape(-1).float() - cp).to(p.dtype).view_as(p)
    slack = Gv @ d - b
    diag["proj_n_mu_pos"] = int((mu > 1e-8).sum())
    diag["proj_min_slack_after"] = float(slack.min())
    diag["proj_corr_ratio"] = float((d - t).norm() / max(nt, 1e-12))
    return diag
