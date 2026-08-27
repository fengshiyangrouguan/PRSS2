"""Ky Fan spectral score: the single component-internal loss.

For one interface type tau, with rows ``Z`` (compressed states, gradient
connected) and ``P`` (fixed joint tests psi(c,y), constant)::

    J_tau = tr[ (Sigma_ZZ + e_z I)^-1  Sigma_ZP  (Sigma_PP + e_p I)^-1  Sigma_PZ ]

computed with two Cholesky factorizations and triangular solves — no SVD, no
explicit inverses.  The training loss maximizes the per-interface score
(equivalently minimizes the positive form sum_tau alpha_tau (d_tau - J_tau)
which differs only by a theta-constant).

Numerical contract (after the cloud crash review, 2026-08-27):
* covariances are built by DIRECT centering of the stacked rows in float64
  — never raw-moment accumulation ``E[zz^T] - E[z]E[z]^T``, which loses the
  signal to catastrophic cancellation when z carries a large DC offset;
* before Cholesky every covariance is normalized by its mean diagonal
  (``A = Czz / mean(diag Czz)``, gradient connected), so the ridge
  ``eps * I`` is scale-free: a collapsed z scale can neither blow J up
  (``J ~ c^2/(c^2+delta) -> 0``) nor break the factorization;
* Cholesky failures are NOT papered over with a floor: the window is
  skipped (differentiable zero, no gradient), its full matrix diagnostics
  are reported, and ``strict=True`` raises (debug runs).
"""

from typing import Dict, List, Tuple

import torch


def _covs(zc: torch.Tensor, pc: torch.Tensor, den: float):
    """Covariances from CENTERED float64 rows, explicitly symmetrized.

    Returns ``(czz, cpp, czp, sym_err)`` where ``sym_err`` is the
    asymmetry of ``czz`` measured BEFORE symmetrization — a moment-algebra
    bug shows up there as a large value.
    """
    czz = zc.t() @ zc / den
    cpp = pc.t() @ pc / den
    czp = zc.t() @ pc / den
    sym_err = float((czz - czz.t()).detach().abs().max())
    return 0.5 * (czz + czz.t()), 0.5 * (cpp + cpp.t()), czp, sym_err


def _score_from_covs(czz: torch.Tensor, czp: torch.Tensor,
                     cpp: torch.Tensor, eps: float):
    """Scale-normalized Ky Fan score with gradient; ``(J, diag)``.

    ``diag["failed"]`` is ``None`` on success, else a short code.  The
    normalization ``A = Czz / mean(diag Czz)`` keeps the ridge ``eps * I``
    scale-free and keeps the score exactly scale-invariant in ``Z`` and
    ``P``; ``mean(diag)`` is NOT detached so the gradient path is the true
    gradient of the normalized objective.
    """
    sz = czz.diagonal().mean()
    sp = cpp.diagonal().mean()
    if float(sz.detach()) <= 0.0 or float(sp.detach()) <= 0.0:
        return None, {"failed": "nonpositive_scale",
                      "scale_z": float(sz.detach()),
                      "scale_p": float(sp.detach())}
    a = czz / sz
    b = cpp / sp
    c = czp / torch.sqrt(sz * sp)
    r, q = a.shape[0], b.shape[0]
    a = 0.5 * (a + a.t()) + eps * torch.eye(r, dtype=a.dtype, device=a.device)
    b = 0.5 * (b + b.t()) + eps * torch.eye(q, dtype=b.dtype, device=b.device)
    lz, info_z = torch.linalg.cholesky_ex(a)
    lp, info_p = torch.linalg.cholesky_ex(b)
    if bool((info_z != 0).any()) or bool((info_p != 0).any()):
        return None, {"failed": "cholesky",
                      "info_z": int(info_z.max().item()),
                      "info_p": int(info_p.max().item()),
                      "scale_z": float(sz.detach()),
                      "scale_p": float(sp.detach())}
    w = torch.linalg.solve_triangular(lz, c, upper=False)    # Lz^-1 C
    k = torch.linalg.solve_triangular(lp, w.t(), upper=False).t()  # w Lp^-T
    return k.square().sum(), {"failed": None,
                              "scale_z": float(sz.detach()),
                              "scale_p": float(sp.detach())}


def _matrix_diag(name: str, x: torch.Tensor) -> dict:
    """Spectral diagnostics (no grad): effective rank via spectral entropy,
    min/max eigenvalue, eigenvalue condition number."""
    evals = torch.linalg.eigvalsh(x.detach().double())
    lam = evals.clamp(min=0.0)
    total = float(lam.sum())
    q = lam / max(total, 1e-12)
    r_eff = 0.0
    if total > 0.0:
        r_eff = float(torch.exp(-(q * torch.log(q.clamp(min=1e-300))).sum()))
    emin = float(evals.min())
    emax = float(evals.max())
    return {"{}_r_eff".format(name): r_eff,
            "{}_min_eig".format(name): emin,
            "{}_max_eig".format(name): emax,
            "{}_cond".format(name): emax / max(emin, 1e-30)}


def _joint_min_eig(czz, czp, cpp) -> float:
    """Smallest eigenvalue of the joint [[Czz, Czp],[Czp^T, Cpp]] matrix.

    Negative here means the three matrices are NOT consistent with any
    distribution (a moment-algebra / covariance-construction bug).
    """
    top = torch.cat([czz, czp], dim=1)
    bot = torch.cat([czp.t(), cpp], dim=1)
    joint = torch.cat([top, bot], dim=0)
    return float(torch.linalg.eigvalsh(joint.detach().double()).min())


def kf_score_fixed(z_c: torch.Tensor, p_c: torch.Tensor,
                   szz: torch.Tensor, spp: torch.Tensor,
                   eps: float = 1e-4) -> torch.Tensor:
    """J with all whitening statistics held constant (a true fixed scale).

    ``z_c`` carries gradient (the only gradient path is the cross term
    ``S_ZP = z_c^T p_c / M``).  This core serves two purposes: (a) the
    gradcheck target, and (b) a future fixed-reference-scale variant, which
    must precompute ``szz``/``spp`` once from a frozen calibration model
    and add an explicit covariance constraint — it is NOT what ``kf_score``
    currently uses.  A Cholesky failure here is a caller bug (the inputs
    are fixed statistics), so it asserts instead of degrading.
    """
    m = z_c.shape[0]
    szp = z_c.t() @ p_c / m
    r = szz.shape[0]
    rz = eps * torch.trace(szz) / r
    rp = eps * torch.trace(spp) / spp.shape[0]
    lz, iz = torch.linalg.cholesky_ex(
        szz + rz * torch.eye(r, dtype=szz.dtype, device=szz.device))
    lp, ip = torch.linalg.cholesky_ex(
        spp + rp * torch.eye(spp.shape[0], dtype=spp.dtype, device=spp.device))
    assert not bool((iz != 0).any()), "fixed C_ZZ must stay PSD"
    assert not bool((ip != 0).any()), "fixed C_PP must stay PSD"
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

    Implementation: direct centering in float64, scale-normalized Cholesky
    (see module docstring).  A degenerate window (nonpositive scale or
    non-PSD after the ridge) yields a differentiable zero instead of a crash.

    ``kf_score_fixed`` below is the core for a genuinely fixed reference
    scale (precomputed once from a frozen calibration model + an explicit
    scale constraint), which this implementation does NOT use.
    """
    if Z.shape[0] < 2:
        raise ValueError("kf_score needs at least 2 rows")
    P = P.detach()                                 # hard API-level isolation
    zc = (Z - Z.mean(dim=0, keepdim=True)).double()
    pc = (P - P.mean(dim=0, keepdim=True)).double()
    m = zc.shape[0]
    czz, cpp, czp, _ = _covs(zc, pc, float(m))
    j, diag = _score_from_covs(czz, czp, cpp, eps)
    if diag["failed"] is not None:
        return Z.sum() * 0.0                       # differentiable zero
    return j.float()


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

    Returns ``(keys, tree_ids, zs, ps)`` where ``keys`` are
    ``(node, time, tau)`` triples — the cross-batch identity of a cut
    (``cut_id`` is only unique within one trace) — and ``tree_ids`` the
    identity of the trace each cut came from (independent samples for the
    window gate are counted per TREE, not per cut: several cuts of one
    trace share the same history and are not independent).
    """
    by_cut: Dict[int, list] = {}
    for r in rows:
        by_cut.setdefault(int(r.cut_id), []).append(r)
    keys, tree_ids, zs, ps = [], [], [], []
    for _, cut_rows in by_cut.items():
        r0 = cut_rows[0]
        keys.append((int(r0.node), float(r0.time), str(r0.tau)))
        tree_ids.append(int(r0.tree_id))
        zs.append(r0.z)
        p_rows = [fixed_maps.pv(r.context, r.outcome) for r in cut_rows]
        ps.append(torch.stack(p_rows).mean(dim=0) if len(p_rows) > 1
                  else p_rows[0])
    return keys, tree_ids, torch.stack(zs), torch.stack(ps)


class KFMomentWindow:
    """Cross-microbatch accumulation for the Ky Fan score.

    Per tau, accumulate the graph-connected z rows and the (constant) p
    rows of unique cuts.  The window closes when the number of unique cut
    TREES seen in this window reaches ``max(min_ratio * d_tau, min_abs)``
    — cuts of one trace share their history, so only distinct trees count
    as independent samples.  Only then J_tau is computed (once) and the
    caller performs ONE backward for the accumulated task loss plus the
    Ky Fan term.  The window then resets.

    The close path follows the numerical contract in the module docstring:
    direct float64 centering of the stacked rows (no raw-moment algebra),
    explicit symmetrization, scale normalization (gradient connected),
    ``cholesky_ex``.  A failing close (nonpositive scale or non-PSD after
    the ridge) contributes a differentiable zero — the window takes no
    part in the backward — and its full matrix diagnostics are reported,
    so the failure is visible instead of papered over.  ``strict=True``
    raises instead (debug runs).

    This is not a second trainer or a calibration pass: it is just a Ky Fan
    minibatch large enough that the small-sample CCA saturation (fake full
    score on independent noise) cannot occur.
    """

    def __init__(self, state_dims: Dict[str, int], *, min_ratio: float = 2.0,
                 min_abs: int = 64, eps: float = 1e-4, fixed_maps=None,
                 strict: bool = False):
        self.state_dims = dict(state_dims)
        self.min_ratio = float(min_ratio)
        self.min_abs = int(min_abs)
        self.eps = float(eps)
        self.fixed_maps = fixed_maps
        self.strict = bool(strict)
        self._windows: Dict[str, dict] = {}

    def _threshold(self, tau: str) -> float:
        return max(self.min_ratio * int(self.state_dims[tau]),
                   float(self.min_abs))

    def add(self, rows: List):
        """Accumulate a batch of CutRecord rows.

        Returns ``(closed, diagnostics, gated)``:
        ``closed`` = {tau: J} for windows that closed this step (graph-
        connected J, one per tau; a failed close contributes a
        differentiable zero); ``diagnostics`` = per-tau {M_unique,
        M_unique_trees, J_shuffled, ...} (detached); ``gated`` = taus still
        accumulating.
        """
        by_tau = score_rows_by_type(rows, self.state_dims)
        closed: Dict[str, torch.Tensor] = {}
        diagnostics: Dict[str, dict] = {}
        gated: List[str] = []
        for tau, tau_rows in by_tau.items():
            if self.fixed_maps is None:
                raise ValueError("KFMomentWindow requires fixed_maps")
            keys, tree_ids, zs, ps = dedup_cut_rows(tau_rows, self.fixed_maps)
            win = self._windows.get(tau)
            if win is None:
                win = {"zs_list": [], "ps_list": [], "seen": set(),
                       "seen_trees": set(), "tree_count": 0}
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
            win["seen_trees"].update(tree_ids[i] for i in fresh)
            win["tree_count"] = len(win["seen_trees"])
            # Graph-connected rows (float32); the close stacks them once
            # and casts to float64.  P is already gradient-free.
            win["zs_list"].append(zs.float())
            win["ps_list"].append(ps.float())
            if win["tree_count"] < self._threshold(tau):
                gated.append(tau)
        # ALL non-empty windows close together: the z graphs of different
        # taus share the same per-batch forward graph, so an asynchronous
        # close would backward through an already-consumed graph.  The
        # window length is therefore set by the slowest (root) interface —
        # which is exactly the interface the depth-balancing exists for.
        nonempty = [tau for tau, win in self._windows.items()
                    if win is not None and win["tree_count"] > 0]
        if nonempty and all(self._windows[tau]["tree_count"]
                            >= self._threshold(tau) for tau in nonempty):
            for tau in nonempty:
                win = self._windows[tau]
                closed[tau], diagnostics[tau] = self._close(win)
            for tau in nonempty:
                self._windows.pop(tau)
        return closed, diagnostics, gated

    def _close(self, win: dict):
        """Close one window: stack, center in float64, score, diagnose."""
        z_all = torch.cat(win["zs_list"], dim=0).double()   # graph-connected
        p_all = torch.cat(win["ps_list"], dim=0).double()   # constant
        assert z_all.shape[0] == p_all.shape[0], \
            "Z and P must come from the same rows/mask ({} vs {})".format(
                z_all.shape[0], p_all.shape[0])
        m = float(z_all.shape[0])
        zc = z_all - z_all.mean(0, keepdim=True)
        pc = p_all - p_all.mean(0, keepdim=True)
        czz, cpp, czp, sym_err = _covs(zc, pc, m - 1.0)
        j, score_diag = _score_from_covs(czz, czp, cpp, self.eps)
        if score_diag["failed"] is not None:
            if self.strict:
                raise RuntimeError(
                    "KFMomentWindow close failed: {}".format(score_diag))
            j = z_all.sum() * 0.0                     # no gradient contribution
        diag = self._diagnostics(win, czz, czp, cpp, j, sym_err, score_diag)
        return j.float(), diag

    def _diagnostics(self, win, czz, czp, cpp, j, sym_err, score_diag) -> dict:
        with torch.no_grad():
            z_all = torch.cat([z.detach() for z in win["zs_list"]], dim=0)
            p_all = torch.cat(win["ps_list"], dim=0)
            m = float(z_all.shape[0])
            # Shuffled score: permute the SAMPLE pairing of P (column
            # permutations are trace-invariant and would prove nothing).
            perm = torch.randperm(int(m), generator=torch.Generator(
                device="cpu").manual_seed(int(m * 7919) % (2 ** 31)))
            zc = (z_all - z_all.mean(0, keepdim=True)).double()
            pc = (p_all[perm] - p_all[perm].mean(0, keepdim=True)).double()
            czzs, cpps, czps, _ = _covs(zc, pc, m - 1.0)
            j_shuffled, _ = _score_from_covs(czzs, czps, cpps, self.eps)
            d = {"M_unique": int(m),
                 "M_unique_trees": int(win["tree_count"]),
                 "J_shuffled": (float(j_shuffled) if j_shuffled is not None
                                else float("nan")),
                 "J_real_minus_shuffled":
                     (float(j.detach())
                      - (float(j_shuffled) if j_shuffled is not None
                         else float("nan"))),
                 "symmetry_error": sym_err,
                 "joint_min_eig": _joint_min_eig(czz, czp, cpp),
                 "failed": score_diag["failed"]}
            d.update(_matrix_diag("zz", czz))
            d.update(_matrix_diag("pp", cpp))
            d.update(score_diag)
            return d

    def window_m(self, tau: str) -> int:
        win = self._windows.get(tau)
        return int(win["tree_count"]) if win is not None else 0

    def reset(self):
        """Discard all open windows (used at epoch drain: the accumulated
        z rows' graphs are consumed by the task-only backward, so the
        unfinished window cannot survive into the next epoch)."""
        self._windows.clear()
