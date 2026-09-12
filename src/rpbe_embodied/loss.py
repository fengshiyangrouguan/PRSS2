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
# kappa is a FIXED constant on the formal method (no annealing); the solver
# core is a verbatim port of scripts/train_lamp.py (branch
# ``lamp_rpbe_project``) so the VLA and TGN lines cannot drift.
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


_ROWS_BACKEND = {"batched_chunks": 0, "fallback_chunks": 0, "last_error": None}


def rows_backend_stats() -> dict:
    """Counters for the per-interface VJP backend.

    ``fallback_chunks > 0`` means the batched ``torch.func`` path was
    unavailable and the slow per-interface loop ran; tests assert it stays 0
    so a silent regression to the slow path cannot hide.
    """
    return dict(_ROWS_BACKEND)


def reset_rows_backend_stats() -> None:
    _ROWS_BACKEND.update(batched_chunks=0, fallback_chunks=0, last_error=None)


def _rows_chunk(gamma, m_a, m_b, C, params, device):
    """Per-interface VJP rows for one chunk -> [c, P] fp32.

    Primary path: ``torch.func.grad`` + ``vmap`` over the interface batch, so
    a chunk of ``c`` interfaces costs ONE batched backward.  (Note
    ``torch.autograd.grad(..., is_grads_batched=True)`` does NOT do this: its
    grad_outputs leading dim must align with the WHOLE output, not the
    interface axis.)  If the vmap path is unavailable it falls back to a
    per-interface loop, but the fallback is COUNTED and warned about loudly --
    never silent, so a slow run can never masquerade as a fast one.
    """
    n = C.shape[0]
    try:
        from torch.func import functional_call, grad, vmap
        pdict = {name: p for name, p in gamma.named_parameters()}

        def scalar(pd, a, b, c):
            z = functional_call(gamma, pd, (a.unsqueeze(0), b.unsqueeze(0)))[0]
            return (c * z.float()).sum()

        out = vmap(grad(scalar), in_dims=(None, 0, 0, 0))(pdict, m_a, m_b, C)
        _ROWS_BACKEND["batched_chunks"] += 1
        return torch.cat([out[name].reshape(n, -1).float()
                          for name, _ in gamma.named_parameters()], dim=1)
    except Exception as e:  # noqa: BLE001 -- must be visible, never silent
        _ROWS_BACKEND["fallback_chunks"] += 1
        _ROWS_BACKEND["last_error"] = f"{type(e).__name__}: {e}"
        print(f"[rpbe rows] WARNING: batched vmap VJP unavailable "
              f"({type(e).__name__}: {e}); using the per-interface loop "
              f"(SLOW).  Reported via rows_backend_stats().", flush=True)
        rows = []
        for i in range(n):
            zi = gamma(m_a[i:i + 1], m_b[i:i + 1])[0]
            gs = torch.autograd.grad((C[i] * zi.float()).sum(), params,
                                     allow_unused=True)
            rows.append(torch.cat([
                (g if g is not None else torch.zeros_like(p)
                 ).reshape(-1).float() for g, p in zip(gs, params)]))
        return torch.stack(rows)


def _stack_to(seq, device: str, dtype) -> torch.Tensor:
    """Stack tensors that may live on MIXED devices.

    Replay inputs are not uniform: the window's fixed-trace rebuild returns a
    CUDA merged state for a merged child but the raw CPU leaf for a leaf child,
    so ``torch.stack([...]).to(...)`` fails.  Cast each element first.
    """
    return torch.stack([t.to(device, dtype=dtype) for t in seq])


def interface_influence_rows(
    gamma, cotangents: Dict[tuple, torch.Tensor],
    input_map: Dict[tuple, Tuple[torch.Tensor, torch.Tensor]],
    keys: List[tuple], params: Optional[List[torch.Tensor]] = None,
    device: str = "cuda", chunk: int = 128,
) -> Tuple[Optional[torch.Tensor], List[tuple], int]:
    """Interface-wise RPBE influence gradients, ONE ROW PER INTERFACE, fp32.

    g_i = (dz_i/dGamma)^T a_i, where a_i = dJ/dz_i is the window adjoint for
    interface i (``cotangents[cut_id]``).  Rows are taken SEPARATELY -- never
    summed -- so interface-interface geometry (e.g. g_1 ~ -g_2) survives into
    the feasibility QP instead of cancelling.  This is the exact VLA analogue
    of ``tree_wise_influence_grads`` in the TGN trainer.

    The window's J is still estimated JOINTLY (one ``dual_latent_z_adjoint``
    over the whole window), so each a_i is interface i's effect ON THE JOINT J.

    IMPORTANT: every interface is returned.  There is deliberately NO row cap
    or subsampling -- each row is an inequality that must be enforced, so
    dropping rows at random has no unbiased-constraint interpretation.
    Memory is bounded instead by (a) batching the VJPs per ``chunk`` and
    (b) keeping rows in CPU fp32 (P ~ 0.574M -> 2.3 MB/row); VRAM never holds
    more than one chunk.  Non-finite rows are DROPPED here -- an Inf row would
    poison Q and eigvalsh -- but the count is returned so the caller can FAIL
    CLOSED (abort the boundary) instead of silently enforcing one constraint
    fewer than the interfaces it claims to protect.

    Returns ``(G_cpu [n_valid, P] fp32 | None, used_keys, n_nonfinite)``.
    """
    if params is None:
        params = list(gamma.parameters())
    cand = [k for k in keys if k in cotangents and k in input_map]
    if not cand or not params:
        return None, [], 0
    md = params[0].dtype
    out: List[torch.Tensor] = []
    used: List[tuple] = []
    n_nonfinite = 0
    with torch.autocast("cuda", enabled=False):
        for c0 in range(0, len(cand), chunk):
            sl = cand[c0:c0 + chunk]
            rows = _rows_chunk(
                gamma,
                _stack_to([input_map[k][0] for k in sl], device, md),
                _stack_to([input_map[k][1] for k in sl], device, md),
                _stack_to([cotangents[k].reshape(-1) for k in sl],
                          device, torch.float32),
                params, device)
            fin = torch.isfinite(rows).all(dim=1)
            n_nonfinite += int((~fin).sum())
            # detach: the rows are constants for the QP.  torch.func.grad can
            # hand back graph-carrying tensors, and keeping that graph alive
            # across every cutting-plane round would retain it (and leak).
            if bool(fin.all()):
                out.append(rows.detach().cpu())
                used.extend(sl)
            else:
                out.append(rows[fin].detach().cpu())
                used.extend([k for k, ok in zip(sl, fin.tolist()) if ok])
    if not out:
        return None, [], n_nonfinite
    G = torch.cat(out, dim=0)
    return (G if G.numel() else None), used, n_nonfinite


def _norm_and_dot(G_cpu: torch.Tensor, v_cpu: torch.Tensor,
                  chunk: int = 256) -> Tuple[torch.Tensor, torch.Tensor]:
    """Chunked ``||g_i||`` and ``g_i . v`` for every CPU row (no GPU peak)."""
    n = G_cpu.shape[0]
    ng = torch.empty(n, dtype=torch.float32)
    gd = torch.empty(n, dtype=torch.float32)
    for i in range(0, n, chunk):
        Gc = G_cpu[i:i + chunk]
        ng[i:i + chunk] = Gc.norm(dim=1)
        gd[i:i + chunk] = Gc @ v_cpu
    return ng, gd


def _row_dot(G_cpu: torch.Tensor, v_cpu: torch.Tensor,
             chunk: int = 256) -> torch.Tensor:
    """Chunked ``g_i . v`` for every CPU row (norms already known)."""
    n = G_cpu.shape[0]
    gd = torch.empty(n, dtype=torch.float32)
    for i in range(0, n, chunk):
        gd[i:i + chunk] = G_cpu[i:i + chunk] @ v_cpu
    return gd


def _iters_ladder(base: int, cap: int) -> List[int]:
    """FISTA budget ladder, e.g. (400, 1600) -> [400, 800, 1600].

    The certificate is checked at each rung, so an easy boundary never pays for
    the larger budget; a hard one escalates instead of aborting."""
    base = max(1, int(base))
    if cap is None or int(cap) <= base:
        return [base]
    out = [base]
    while out[-1] < int(cap):
        out.append(min(out[-1] * 2, int(cap)))
    return out


def _cutting_plane(G_cpu, t, dev, kappa, tau_feas, ng, Gt, valid, scale, nt,
                   active, iters, max_rounds, max_active, add_per_round):
    """One cutting-plane sweep at a fixed FISTA budget.

    Returns (corr, active, mu, vmax, rounds, min_slack, n_mu_pos).  The active
    rows are EXACTLY row-normalised before the QP: with
    ``g_hat_i = g_i / ||g_i||`` the half-space ``g_i.d >= -kappa||g_i|| ||t||``
    is the identical constraint ``g_hat_i.d >= -kappa||t||`` (dividing by the
    positive ``||g_i||`` does not change the feasible set, hence not d*), but
    the dual Gram becomes a cosine matrix with unit diagonal instead of one
    whose conditioning is set by the spread of the row norms -- which is what
    made FISTA stall on real Gamma gradients.  Rows reaching here already
    passed ``min_norm``, so the division needs no epsilon.
    """
    corr = torch.zeros(t.numel(), dtype=torch.float32, device=dev)
    mu = None
    rnd = 0
    vmax = float("inf")
    min_slack = float("nan")
    n_mu_pos = 0
    active_set = set(active)
    for rnd in range(1, max_rounds + 1):
        if active:
            idx = torch.tensor(active, dtype=torch.long)
            nga = ng[idx].to(dev, dtype=torch.float32)
            Ha = G_cpu[idx].to(dev, dtype=torch.float32) / nga[:, None]
            b = torch.full((Ha.shape[0],), -kappa * nt, device=dev)
            mu = _fista_nonneg(Ha @ Ha.t(), b - Ha @ t, iters)
            corr = (mu[:, None] * Ha).sum(0)
            d = t + corr
        else:
            corr = torch.zeros_like(corr)
            d = t
        # RE-CHECK every interface at the new d (not just the active set)
        Gd = _row_dot(G_cpu, d.cpu())
        vv = torch.where(valid, torch.clamp(-kappa * scale - Gd, min=0.0)
                         / (scale + 1e-30), torch.zeros_like(Gt))
        vmax = float(vv.max())
        if vmax <= tau_feas:
            break
        added = 0
        for i in torch.argsort(vv, descending=True).tolist():
            if len(active) >= max_active or added >= add_per_round:
                break
            if vv[i] > tau_feas and i not in active_set:
                active.append(i); active_set.add(i); added += 1
        if added == 0:
            break
    if active:
        idx = torch.tensor(active, dtype=torch.long)
        nga = ng[idx].to(dev, dtype=torch.float32)
        Ha = G_cpu[idx].to(dev, dtype=torch.float32) / nga[:, None]
        slack = Ha @ (t + corr) - (-kappa * nt)
        min_slack = float(slack.min())
    if mu is not None:
        n_mu_pos = int((mu > 1e-8).sum())
    return corr, active, mu, vmax, rnd, min_slack, n_mu_pos


def active_set_feasibility_projection(
    g_task_gamma: List[torch.Tensor], gamma_params: List[torch.Tensor],
    G_cpu: Optional[torch.Tensor], kappa: float, iters: int = 400,
    iters_max: int = 1600, tau_feas: float = 1e-3, max_rounds: int = 8,
    max_active: int = 2048, add_per_round: int = 512, min_norm: float = 1e-9,
) -> dict:
    """Cutting-plane feasibility projection over EVERY interface.

    One global task direction t = -g_task; each interface i contributes the
    half-space g_i.d >= -kappa*||g_i||*||t||.  Unlike the single-shot QP this
    CHECKS ALL interfaces: round 0 ranks every interface by its violation at
    d = t, the worst enter the active set, then each round solves the QP on the
    active set and RE-SCANS every interface at the new d, adding new violators,
    until the dimensionless max violation

        v_i = max(0, b_i - g_i.d) / (||g_i|| ||t|| + eps)

    is <= ``tau_feas`` (or ``max_rounds``/``max_active`` is exhausted).

    Writes ONLY Gamma (p.grad = g_task - sum_i mu_i g_i) and ONLY when the
    result is feasible.  On failure nothing is written and
    diag["proj_feasible"] is False so the caller can ABORT the boundary
    instead of stepping with a violated preservation constraint.

    VRAM holds only the active rows; ``G_cpu`` stays in host memory.
    """
    sizes = [p.numel() for p in gamma_params]
    # t lives on the Gamma params' device; the full-interface scans run on the
    # CPU-resident G rows, and only the active rows ever reach `dev`.
    t = torch.cat([-x.detach().flatten().float() for x in g_task_gamma])
    dev = t.device
    t_cpu = t.detach().cpu()
    nt = float(t.norm())
    diag = {
        "proj_n_interfaces": int(G_cpu.shape[0]) if G_cpu is not None else 0,
        "proj_n_checked": 0, "proj_n_valid": 0, "proj_n_candidates": 0,
        "proj_n_active": 0, "proj_n_active_init": 0, "proj_n_mu_pos": 0,
        "proj_rounds": 0, "proj_coverage": 1.0, "proj_norm_t": nt,
        "proj_cos_min": 0.0, "proj_max_viol_before": 0.0,
        "proj_max_viol_after": 0.0, "proj_min_slack_after": float("nan"),
        "proj_corr_ratio": 0.0, "proj_kappa": float(kappa),
        "proj_feasible": True,
    }
    if G_cpu is None or G_cpu.numel() == 0 or nt == 0.0:
        return diag
    if not math.isfinite(nt):
        # a non-finite task direction can never be feasibly projected
        diag["proj_feasible"] = False
        diag["proj_abort"] = "nonfinite_task_direction"
        return diag
    ng, Gt = _norm_and_dot(G_cpu, t_cpu)
    valid = (ng > min_norm) & torch.isfinite(ng) & torch.isfinite(Gt)
    diag["proj_n_valid"] = int(valid.sum())
    diag["proj_n_checked"] = int(G_cpu.shape[0])
    if not bool(valid.any()):
        return diag
    scale = ng * nt
    cos = torch.where(valid, Gt / (scale + 1e-30), torch.ones_like(Gt))
    viol = torch.clamp(-cos - kappa, min=0.0)
    diag["proj_cos_min"] = float(cos[valid].min())
    # cos quantiles over the valid interfaces: the EVIDENCE for choosing kappa
    # (an interface binds iff cos(g_i, t) < -kappa).  A kappa inherited from
    # another host can leave the projection permanently inactive.
    qv = torch.quantile(cos[valid], torch.tensor(
        [0.01, 0.05, 0.10, 0.25, 0.50]))
    for nm, v in zip(("q01", "q05", "q10", "q25", "q50"), qv.tolist()):
        diag[f"proj_cos_{nm}"] = float(v)
    diag["proj_max_viol_before"] = float(viol.max())
    # Only interfaces that BREACH the feasibility tolerance need a constraint.
    # Anything with 0 < v_i <= tau_feas already satisfies the formal criterion,
    # so admitting it would burn active-set slots (and could push a later, real
    # violator out of the budget) for no reason.
    n_cand = int((viol > tau_feas).sum())
    diag["proj_n_candidates"] = n_cand
    active = [i for i in torch.argsort(viol, descending=True).tolist()[:max_active]
              if viol[i] > tau_feas]
    diag["proj_n_active_init"] = len(active)
    # FISTA budget ladder: an easy boundary certifies at `iters` and stops; a
    # hard one escalates (400 -> 800 -> 1600) instead of aborting.  tau is the
    # certificate tolerance and is NEVER relaxed to buy feasibility -- that
    # would fold a solver failure into the method's hyper-parameters.
    corr = torch.zeros(t.numel(), dtype=torch.float32, device=dev)
    mu, vmax, min_slack, rnd, n_mu_pos = None, float(viol.max()), float("nan"), 0, 0
    for budget in _iters_ladder(iters, iters_max):
        corr, active, mu, vmax, rnd, min_slack, n_mu_pos = _cutting_plane(
            G_cpu, t, dev, kappa, tau_feas, ng, Gt, valid, scale, nt, active,
            budget, max_rounds, max_active, add_per_round)
        diag["proj_iters_used"] = int(budget)
        if vmax <= tau_feas:
            break
    diag["proj_rounds"] = int(rnd)
    diag["proj_n_active"] = len(active)
    # coverage = share of ALL checked interfaces that carry an enforced
    # constraint.  n_candidates (violators at d = t) can be smaller than
    # n_active because a cutting-plane round may add an interface that only
    # violates at the new d.
    diag["proj_coverage"] = len(active) / max(1, int(G_cpu.shape[0]))
    diag["proj_max_viol_after"] = vmax
    diag["proj_feasible"] = bool(vmax <= tau_feas)
    diag["proj_corr_ratio"] = float(corr.norm() / max(nt, 1e-12))
    diag["proj_n_mu_pos"] = n_mu_pos
    diag["proj_min_slack_after"] = min_slack
    # internal (test-only) handles; the trainer never logs keys starting "_"
    diag["_active"] = active
    diag["_mu"] = None if mu is None else mu.detach().cpu()
    if diag["proj_feasible"]:
        with torch.no_grad():
            for p, cp, gt in zip(gamma_params, torch.split(corr, sizes),
                                 g_task_gamma):
                p.grad = (gt.reshape(-1).float() - cp).to(p.dtype).view_as(p)
    return diag
