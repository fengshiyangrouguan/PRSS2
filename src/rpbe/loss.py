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


def _cholesky_retry(mat: torch.Tensor, tries: int = 8) -> torch.Tensor:
    """Cholesky with escalating jitter, then a relative diagonal floor.

    The caller's ridge scales with trace/dim, so a covariance that is merely
    NEAR-singular — z directions whose variance collapsed over training (the
    last Cholesky pivot goes ~0) — can still fail through the escalations,
    which stay absolute.  The final floor adds ``(rel * 1e-2 + 1e-6) * I``:
    by Weyl, the smallest eigenvalue of the result is at least ``1e-6``, so
    the factorization cannot fail and a near-singular window close can never
    crash training (the crashed run loses hours; a slightly inflated score
    is the correct degradation).
    """
    n = mat.shape[0]
    rel = float(torch.trace(mat).detach().clamp(min=0.0)) / n
    jitter = 1e-12
    for _ in range(tries):
        try:
            return torch.linalg.cholesky(mat)
        except RuntimeError:
            mat = mat + (jitter + rel * 1e-8) * torch.eye(
                n, dtype=mat.dtype, device=mat.device)
            jitter *= 10.0
    floor = rel * 1e-2 + 1e-6
    return torch.linalg.cholesky(
        mat + floor * torch.eye(n, dtype=mat.dtype, device=mat.device))


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


def score_rows_by_type(rows: List, state_dims: Dict[str, int]) -> Dict[str, list]:
    """Split CutRecord rows into per-tau (z list, p list) pairs."""
    by_tau = {}
    for r in rows:
        if r.tau not in state_dims:
            continue
        by_tau.setdefault(r.tau, []).append(r)
    return by_tau


def dedup_cut_rows(rows: List, fixed_maps):
    """One row per unique cut: z appears once, P rows of one cut are averaged.

    Returns ``(keys, zs, ps)`` where ``keys`` are ``(node, time, tau)``
    triples — the cross-batch identity of a cut (``cut_id`` is only unique
    within one trace).  The link stage no longer fabricates negative rows
    (each cut carries ONE real future continuation), so the dedup is a
    defensive layer for any residual repetition.
    """
    by_cut: Dict[int, list] = {}
    for r in rows:
        by_cut.setdefault(int(r.cut_id), []).append(r)
    keys, zs, ps = [], [], []
    for _, cut_rows in by_cut.items():
        r0 = cut_rows[0]
        keys.append((int(r0.node), float(r0.time), str(r0.tau)))
        zs.append(r0.z)
        p_rows = [fixed_maps.pv(r.context, r.outcome) for r in cut_rows]
        ps.append(torch.stack(p_rows).mean(dim=0) if len(p_rows) > 1
                  else p_rows[0])
    return keys, torch.stack(zs), torch.stack(ps)


class KFMomentWindow:
    """Cross-microbatch moment accumulation for the Ky Fan score.

    Per tau, accumulate (M, sum z, sum p, sum zz^T, sum pp^T, sum zp^T) over
    temporal microbatches.  All z-carrying quantities stay graph-connected;
    the window closes when M — counted in UNIQUE cut ids seen IN THIS WINDOW —
    reaches ``max(min_ratio * d_tau, min_abs)``.  Only then J_tau is computed
    (once) from the window moments, and the caller performs ONE backward for
    the accumulated task loss plus the Ky Fan term.  The window then resets.

    This is not a second trainer or a calibration pass: it is just a Ky Fan
    minibatch large enough that the small-sample CCA saturation (fake full
    score on independent noise) cannot occur.
    """

    def __init__(self, state_dims: Dict[str, int], *, min_ratio: float = 2.0,
                 min_abs: int = 64, eps: float = 1e-4, fixed_maps=None):
        self.state_dims = dict(state_dims)
        self.min_ratio = float(min_ratio)
        self.min_abs = int(min_abs)
        self.eps = float(eps)
        self.fixed_maps = fixed_maps
        self._windows: Dict[str, dict] = {}

    def _threshold(self, tau: str) -> float:
        return max(self.min_ratio * int(self.state_dims[tau]),
                   float(self.min_abs))

    def add(self, rows: List):
        """Accumulate a batch of CutRecord rows.

        Returns ``(closed, diagnostics, gated)``:
        ``closed`` = {tau: J} for windows that closed this step (graph-
        connected J, one per tau); ``diagnostics`` = per-tau {M, J_shuffled,
        cond_zz, cond_pp} (detached); ``gated`` = taus still accumulating.
        """
        by_tau = score_rows_by_type(rows, self.state_dims)
        closed: Dict[str, torch.Tensor] = {}
        diagnostics: Dict[str, dict] = {}
        gated: List[str] = []
        for tau, tau_rows in by_tau.items():
            if self.fixed_maps is None:
                raise ValueError("KFMomentWindow requires fixed_maps")
            keys, zs, ps = dedup_cut_rows(tau_rows, self.fixed_maps)
            win = self._windows.get(tau)
            if win is None:
                win = {"m": 0, "sz": None, "sp": None, "szz": None,
                       "spp": None, "szp": None, "seen": set(),
                       "zs_list": [], "ps_list": []}
                self._windows[tau] = win
            # Cross-batch dedupe on the cut's (node, as-of time, tau)
            # identity: cut_id repeats across traces, the triple does not
            # (one cut = one node state at one moment at one interface).
            fresh = [i for i, key in enumerate(keys)
                     if key not in win["seen"]]
            if not fresh:
                gated.append(tau)
                continue
            zs = zs[fresh]
            ps = ps[fresh]
            win["seen"].update(keys[i] for i in fresh)
            zs = zs.float()                       # never AMP fp16 moments
            ps = ps.float()
            win["m"] += len(fresh)
            win["sz"] = zs.sum(0) if win["sz"] is None else win["sz"] + zs.sum(0)
            win["sp"] = ps.sum(0) if win["sp"] is None else win["sp"] + ps.sum(0)
            win["szz"] = zs.t() @ zs if win["szz"] is None else win["szz"] + zs.t() @ zs
            win["spp"] = ps.t() @ ps if win["spp"] is None else win["spp"] + ps.t() @ ps
            win["szp"] = zs.t() @ ps if win["szp"] is None else win["szp"] + zs.t() @ ps
            # Detached row copies for the shuffled diagnostic only.
            win["zs_list"].append(zs.detach())
            win["ps_list"].append(ps.detach())
            if win["m"] < self._threshold(tau):
                gated.append(tau)
        # ALL non-empty windows close together: the z graphs of different
        # taus share the same per-batch forward graph, so an asynchronous
        # close would backward through an already-consumed graph.  The
        # window length is therefore set by the slowest (root) interface —
        # which is exactly the interface the depth-balancing exists for.
        nonempty = [tau for tau, win in self._windows.items()
                    if win is not None and win["m"] > 0]
        if nonempty and all(self._windows[tau]["m"] >= self._threshold(tau)
                            for tau in nonempty):
            for tau in nonempty:
                win = self._windows[tau]
                m = float(win["m"])
                zbar = win["sz"] / m
                pbar = win["sp"] / m
                czz = win["szz"] / m - torch.outer(zbar, zbar)
                czp = win["szp"] / m - torch.outer(zbar, pbar)
                cpp = win["spp"] / m - torch.outer(pbar, pbar)
                closed[tau] = _j_from_covs(czz, czp, cpp, self.eps, zs)
                diagnostics[tau] = self._diagnostics(win, czz, cpp)
            for tau in nonempty:
                self._windows.pop(tau)
        return closed, diagnostics, gated

    def _diagnostics(self, win, czz, cpp) -> dict:
        with torch.no_grad():
            z_all = torch.cat(win["zs_list"], dim=0)   # [M, d] detached
            p_all = torch.cat(win["ps_list"], dim=0)   # [M, m] detached
            m = float(z_all.shape[0])
            # Shuffled score: permute the SAMPLE pairing of P (column
            # permutations are trace-invariant and would prove nothing).
            perm = torch.randperm(int(m), generator=torch.Generator(
                device="cpu").manual_seed(int(m * 7919) % (2 ** 31)))
            zc = z_all - z_all.mean(0, keepdim=True)
            pc = p_all[perm] - p_all[perm].mean(0, keepdim=True)
            czp_shuffled = zc.t() @ pc / m
            j_shuffled = _j_from_covs(czz.detach(), czp_shuffled,
                                      cpp.detach(), self.eps, None)
            cond_zz = float(torch.linalg.cond(czz.detach()))
            cond_pp = float(torch.linalg.cond(cpp.detach()))
        return {"M_unique": int(m), "J_shuffled": float(j_shuffled),
                "cond_zz": cond_zz, "cond_pp": cond_pp}

    def window_m(self, tau: str) -> int:
        win = self._windows.get(tau)
        return int(win["m"]) if win is not None else 0

    def reset(self):
        """Discard all open windows (used at epoch drain: the accumulated
        moments' z graphs are consumed by the task-only backward, so the
        unfinished window cannot survive into the next epoch)."""
        self._windows.clear()


def _j_from_covs(czz, czp, cpp, eps, zs_for_zero):
    """J = tr[(C_ZZ+e)^-1 C_ZP (C_PP+e)^-1 C_PZ], constant-safe.

    A constant Z makes C_ZZ zero; return a differentiable zero instead of
    letting a vanishing ridge feed Cholesky.
    """
    if torch.allclose(czz, torch.zeros_like(czz), atol=1e-12) \
            or torch.allclose(cpp, torch.zeros_like(cpp), atol=1e-12):
        # Differentiable zero: keeps the graph connected without a score.
        return czp.sum() * 0.0
    r = czz.shape[0]
    lz = _cholesky_retry(czz + _ridge(eps, czz) * torch.eye(
        r, dtype=czz.dtype, device=czz.device))
    lp = _cholesky_retry(cpp + _ridge(eps, cpp) * torch.eye(
        cpp.shape[0], dtype=cpp.dtype, device=cpp.device))
    w = torch.cholesky_solve(czp, lz)
    s = torch.cholesky_solve(czp.t(), lp)
    return (w * s.t()).sum()
