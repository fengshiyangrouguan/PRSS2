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


def _covs(zc: torch.Tensor, pc: torch.Tensor, den: float,
          w: torch.Tensor = None):
    """Covariances from CENTERED float64 rows, explicitly symmetrized.

    With ``w`` (detached float64 weights) the centered rows are weighted
    (``zc * sqrt(w)``), so the result is the weighted covariance with
    effective degrees of freedom ``den`` supplied by the caller.  Returns
    ``(czz, cpp, czp, sym_err)`` where ``sym_err`` is the asymmetry of
    ``czz`` measured BEFORE symmetrization — a moment-algebra bug shows up
    there as a large value.
    """
    if w is not None:
        wc = w.reshape(-1, 1).to(zc.dtype)
        zc = zc * wc.sqrt()
        pc = pc * wc.sqrt()
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


def kf_adjoint(wf_result: dict, eps: float, strict: bool = False):
    """A = grad of the Ky Fan score w.r.t. the window statistics.

    The Moment-Adjoint Replay bridge: the Welford result (pass 1, detached)
    becomes small-matrix leaves; the score graph is replayed on them and
    one backward yields the cotangents ``A_zz, A_zp, A_pp`` (d J / d M2_*).
    Pass 2 then replays each batch with

        surrogate_b = <A_zz, M2_zz_b> + <A_zp, M2_zp_b> + <A_pp, M2_pp_b>

    (batched moments centered with the SAME global detached means), which
    is the exact first-order gradient at the current parameters — the
    statistics span and the autograd graph span are fully decoupled.

    ``W``/``W2_cut``/``D`` carry no model gradient under FIXED weights and
    are deliberately NOT part of the adjoint (they only fix the numeric
    value of A).  Returns ``(j_float, adjoints, score_diag)``; a failed
    close returns ``(None, None, diag)`` (or raises in strict mode).
    """
    D = float(wf_result["D"])
    W = float(wf_result["W"])
    if not (W > 0.0 and D > 0.0):
        diag = {"failed": "nonpositive_weight", "W": W, "D_cut": D}
        if strict:
            raise RuntimeError("kf_adjoint: {}".format(diag))
        return None, None, diag
    czz = wf_result["M2_zz"].clone().requires_grad_(True)
    cpp = wf_result["M2_pp"].clone().requires_grad_(True)
    czp = wf_result["M2_zp"].clone().requires_grad_(True)
    # _score_from_covs signature is (czz, czp, cpp, eps).
    j, score_diag = _score_from_covs(czz / D, czp / D, cpp / D, eps)
    if score_diag["failed"] is not None:
        if strict:
            raise RuntimeError("kf_adjoint close failed: {}".format(score_diag))
        return None, None, score_diag
    j.backward()
    adjoints = {"M2_zz": czz.grad.detach(),
                "M2_pp": cpp.grad.detach(),
                "M2_zp": czp.grad.detach()}
    return float(j.detach()), adjoints, score_diag


def kf_vjp_batch(z_b: torch.Tensor, p_b: torch.Tensor, w_b: torch.Tensor,
                 mu_z: torch.Tensor, mu_p: torch.Tensor,
                 adjoints: dict) -> torch.Tensor:
    """Pass-2 surrogate for one batch: <A, S_b(theta)>.

    ``z_b`` carries gradient; ``p_b``, ``w_b``, ``mu_z``, ``mu_p`` and the
    adjoints are all detached constants.  The batched moments are centered
    with the SAME global detached means, so the per-batch terms add up to
    the exact gradient of the whole-window score.
    """
    zc = z_b.double() - mu_z
    pc = p_b.double() - mu_p
    sw = w_b.double().sqrt().reshape(-1, 1)
    mzz_b = (zc * sw).t() @ (zc * sw)
    mzp_b = (zc * sw).t() @ (pc * sw)
    mpp_b = (pc * sw).t() @ (pc * sw)
    return ((adjoints["M2_zz"] * mzz_b).sum()
            + (adjoints["M2_zp"] * mzp_b).sum()
            + (adjoints["M2_pp"] * mpp_b).sum())


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


class WeightedWelford:
    """FP64 weighted ONLINE central moments (Chan et al. merge form).

    The pass-1 accumulator of the Moment-Adjoint Replay: one no_grad pass
    merges every batch's weighted centered moments into the whole-window
    statistics ``W, W2_cut, mu_z, mu_p, M2_zz, M2_pp, M2_zp`` in float64,
    from which the global mean mu_hat and the cluster degrees of freedom
    ``D = W - W2_cut/W`` come out (``W2_cut = sum_v (sum_h w_{v,h})^2``,
    rows of one cut share z so the cut's weight is its rows' sum).

    The merge keeps the direct-centering numerical contract: batch means
    and M2 are computed on CENTERED rows, and the cross-batch merge adds
    the Chan correction term ``d d^T W W_b/(W+W_b)`` — raw-moment
    cancellation (``E[zz^T] - E[z]E[z]^T``) never appears.
    """

    def __init__(self, dim_z: int, dim_p: int):
        self.dim_z = int(dim_z)
        self.dim_p = int(dim_p)
        self.W = 0.0
        self.W2_cut = 0.0
        self.mu_z = torch.zeros(self.dim_z, dtype=torch.float64)
        self.mu_p = torch.zeros(self.dim_p, dtype=torch.float64)
        self.M2_zz = torch.zeros(self.dim_z, self.dim_z, dtype=torch.float64)
        self.M2_pp = torch.zeros(self.dim_p, self.dim_p, dtype=torch.float64)
        self.M2_zp = torch.zeros(self.dim_z, self.dim_p, dtype=torch.float64)
        self.n_batches = 0

    def add(self, z_b: torch.Tensor, p_b: torch.Tensor,
            w_b: torch.Tensor, cut_ids_b: List[tuple]):
        """Merge one batch: ``z_b`` [n, dim_z] (detached), ``p_b`` [n, dim_p]
        (constant), ``w_b`` [n] weights, ``cut_ids_b`` per row."""
        z = z_b.detach().double()
        p = p_b.detach().double()
        w = w_b.detach().double().reshape(-1)
        n = z.shape[0]
        assert n == p.shape[0] == w.shape[0] == len(cut_ids_b), \
            "row mismatch: z {} p {} w {} ids {}".format(
                n, p.shape[0], w.shape[0], len(cut_ids_b))
        # Lazy device placement: the accumulator is created CPU-side and
        # follows the first batch's device (weights arrive on z's device
        # via dedup_cut_rows).
        if self.mu_z.device != z.device:
            self.mu_z = self.mu_z.to(z.device)
            self.mu_p = self.mu_p.to(z.device)
            self.M2_zz = self.M2_zz.to(z.device)
            self.M2_pp = self.M2_pp.to(z.device)
            self.M2_zp = self.M2_zp.to(z.device)
        W_b = float(w.sum())
        if W_b <= 0.0 or n == 0:
            return
        # Batch-internal weighted mean and centered M2 (direct centering).
        wc = w[:, None]
        mz = (z * wc).sum(0, keepdim=True) / W_b
        mp = (p * wc).sum(0, keepdim=True) / W_b
        zc = z - mz
        pc = p - mp
        sw = w.sqrt()[:, None]
        M2_zz_b = (zc * sw).t() @ (zc * sw)
        M2_pp_b = (pc * sw).t() @ (pc * sw)
        M2_zp_b = (zc * sw).t() @ (pc * sw)
        # Cluster W2: rows of one cut share z; its weight is the row sum.
        wsum = {}
        for cid, wv in zip(cut_ids_b, w.tolist()):
            wsum[cid] = wsum.get(cid, 0.0) + wv
        W2_b = float(sum(v * v for v in wsum.values()))
        # Chan merge into the running statistics.
        W_new = self.W + W_b
        dz = mz - self.mu_z                      # [1, dim_z]
        dp = mp - self.mu_p
        gain = self.W * W_b / W_new
        self.M2_zz = self.M2_zz + M2_zz_b + gain * (dz.t() @ dz)
        self.M2_pp = self.M2_pp + M2_pp_b + gain * (dp.t() @ dp)
        self.M2_zp = self.M2_zp + M2_zp_b + gain * (dz.t() @ dp)
        self.mu_z = self.mu_z + (W_b / W_new) * dz[0]
        self.mu_p = self.mu_p + (W_b / W_new) * dp[0]
        self.W = float(W_new)
        self.W2_cut += W2_b
        self.n_batches += 1

    def result(self) -> dict:
        """(W, W2_cut, D, mu_z, mu_p, M2_zz, M2_pp, M2_zp) — detached FP64."""
        D = self.W - self.W2_cut / self.W if self.W > 0.0 else 0.0
        return {"W": float(self.W), "W2_cut": float(self.W2_cut),
                "D": float(D),
                "mu_z": self.mu_z.clone(),
                "mu_p": self.mu_p.clone(),
                "M2_zz": self.M2_zz.clone(),
                "M2_pp": self.M2_pp.clone(),
                "M2_zp": self.M2_zp.clone()}


def dedup_cut_rows(rows: List, fixed_maps):
    """Row-wise projection of CutRecords (NO averaging across horizons).

    One row per (cut, horizon) pair: horizons of one cut are mixed by
    WEIGHT (``w_{v,h} = w_v / |H_v|``), never by averaging the p vectors —
    averaging would create p1*p2^T cross terms and change Cov(P) and the
    Ky Fan objective.  Returns ``(row_ids, cut_ids, tree_ids, zs, ps,
    weights)``:

    * ``row_ids``  — (tree, occurrence, tau, horizon): window row dedupe
    * ``cut_ids``  — (tree, occurrence, tau): unique-cut gate counting
    * ``tree_ids`` — trace identity (window gate counts per TREE: several
      cuts of one trace share the same history and are not independent)
    """
    row_ids, cut_ids, tree_ids, zs, ps, weights = [], [], [], [], [], []
    for r in rows:
        row_ids.append(r.row_id)
        cut_ids.append(r.cut_id)
        tree_ids.append(int(r.tree_id))
        zs.append(r.z)
        ps.append(fixed_maps.pv(r.context, r.outcome))
        weights.append(float(r.weight))
    zs_t = torch.stack(zs)
    return (row_ids, cut_ids, tree_ids, zs_t, torch.stack(ps),
            torch.tensor(weights, dtype=torch.float64, device=zs_t.device))


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
                 strict: bool = False, autoclose: bool = True):
        self.state_dims = dict(state_dims)
        self.min_ratio = float(min_ratio)
        self.min_abs = int(min_abs)
        self.eps = float(eps)
        self.fixed_maps = fixed_maps
        self.strict = bool(strict)
        # autoclose=False (replay mode): ``add`` only accumulates; the
        # loop decides when to call ``close_replay``.
        self.autoclose = bool(autoclose)
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
            row_ids, cut_ids, tree_ids, zs, ps, weights = dedup_cut_rows(
                tau_rows, self.fixed_maps)
            win = self._windows.get(tau)
            if win is None:
                win = {"zs_list": [], "ps_list": [], "weights_list": [],
                       "row_seen": set(), "cut_seen": set(),
                       "cut_ids_list": [], "tree_seen": set()}
                self._windows[tau] = win
            # Cross-batch dedupe on row_id (cut + horizon).  cut_id is
            # globally unique (adapter-wide occurrence counter), so a
            # repeated row is a true duplicate; distinct horizons of one
            # cut are DIFFERENT rows and both stay.
            fresh = [i for i, rid in enumerate(row_ids)
                     if rid not in win["row_seen"]]
            if not fresh:
                gated.append(tau)
                continue
            zs = zs[fresh]
            ps = ps[fresh]
            weights = weights[fresh]
            for i in fresh:
                win["row_seen"].add(row_ids[i])
                win["cut_seen"].add(cut_ids[i])
                win["tree_seen"].add(tree_ids[i])
                win["cut_ids_list"].append(cut_ids[i])
            # Graph-connected rows (float32); the close stacks them once
            # and casts to float64.  P is already gradient-free.
            win["zs_list"].append(zs.float())
            win["ps_list"].append(ps.float())
            win["weights_list"].append(weights)
            if len(win["cut_seen"]) < self._threshold(tau):
                gated.append(tau)
        # ALL non-empty windows close together: the z graphs of different
        # taus share the same per-batch forward graph, so an asynchronous
        # close would backward through an already-consumed graph.  The
        # window length is therefore set by the slowest (root) interface —
        # which is exactly the interface the depth-balancing exists for.
        nonempty = [tau for tau, win in self._windows.items()
                    if win is not None and len(win["cut_seen"]) > 0]
        if self.autoclose and nonempty and all(
                len(self._windows[tau]["cut_seen"])
                >= self._threshold(tau) for tau in nonempty):
            for tau in nonempty:
                win = self._windows[tau]
                closed[tau], diagnostics[tau] = self._close(win)
            for tau in nonempty:
                self._windows.pop(tau)
        return closed, diagnostics, gated

    def _close(self, win: dict):
        """Close one window: weighted float64 centering, score, diagnose.

        Row weights ``w_{v,h}`` enter the moments; the effective degrees of
        freedom use the CLUSTER-level second weight moment (rows of one cut
        share the same z, so their summed weight is the cut's weight):
        ``D = W - W2_cut/W`` with ``W2_cut = sum_v (sum_h w_{v,h})^2``.
        """
        z_all = torch.cat(win["zs_list"], dim=0).double()   # graph-connected
        p_all = torch.cat(win["ps_list"], dim=0).double()   # constant
        w = torch.cat(win["weights_list"], dim=0).double()  # detached rows
        assert z_all.shape[0] == p_all.shape[0] == w.shape[0], \
            "Z/P/weights must share rows ({} vs {} vs {})".format(
                z_all.shape[0], p_all.shape[0], w.shape[0])
        wsum_by_cut: Dict[tuple, float] = {}
        for cid, wv in zip(win["cut_ids_list"], w.tolist()):
            wsum_by_cut[cid] = wsum_by_cut.get(cid, 0.0) + wv
        W = float(w.sum())
        W2_cut = float(sum(v * v for v in wsum_by_cut.values()))
        D = W - W2_cut / W
        if not (W > 0.0 and D > 0.0):
            diag = {"failed": "nonpositive_weight", "W": W, "D_cut": D,
                    "M_unique": int(len(win["cut_seen"])),
                    "M_unique_trees": int(len(win["tree_seen"]))}
            return (z_all.sum() * 0.0).float(), diag
        mu_z = (z_all * w[:, None]).sum(0, keepdim=True) / W
        mu_p = (p_all * w[:, None]).sum(0, keepdim=True) / W
        zc = z_all - mu_z
        pc = p_all - mu_p
        czz, cpp, czp, sym_err = _covs(zc, pc, D, w=w)
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
            w = torch.cat(win["weights_list"], dim=0)
            m_rows = float(z_all.shape[0])
            W = float(w.sum())
            wsum_by_cut: Dict[tuple, float] = {}
            for cid, wv in zip(win["cut_ids_list"], w.tolist()):
                wsum_by_cut[cid] = wsum_by_cut.get(cid, 0.0) + wv
            W2_cut = float(sum(v * v for v in wsum_by_cut.values()))
            D = W - W2_cut / W
            # Shuffled score: permute the SAMPLE pairing of P (column
            # permutations are trace-invariant and would prove nothing);
            # weights travel with their rows.
            perm = torch.randperm(int(m_rows), generator=torch.Generator(
                device="cpu").manual_seed(int(m_rows * 7919) % (2 ** 31)))
            wc = w.reshape(-1, 1)
            mu_z = (z_all * wc).sum(0, keepdim=True) / W
            mu_p = (p_all[perm] * wc[perm]).sum(0, keepdim=True) / W
            zc = (z_all - mu_z).double()
            pc = (p_all[perm] - mu_p).double()
            czzs, cpps, czps, _ = _covs(zc, pc, D, w=w[perm])
            j_shuffled, _ = _score_from_covs(czzs, czps, cpps, self.eps)
            d = {"M_unique": int(len(win["cut_seen"])),
                 "M_rows": int(m_rows),
                 "M_unique_trees": int(len(win["tree_seen"])),
                 "w_eff_cut": (W * W / W2_cut) if W2_cut > 0.0
                 else float("nan"),
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
        return int(len(win["cut_seen"])) if win is not None else 0

    # ---------------------------------------------------------- replay close
    def close_replay(self):
        """Moment-Adjoint close (pass-1 -> pass-2 bridge).

        Accumulates the stored (detached) rows through WeightedWelford,
        derives the small-matrix adjoints ``A = grad_S F(S)`` and returns
        ``(closed, adjoints, diagnostics)`` where ``closed[tau]`` is the
        REAL window score ``F(S_W)`` (the number to log — never the
        surrogate), and ``adjoints[tau] = (adj, wf_result)`` feeds the
        pass-2 replay.  All nonempty windows close together (same shared-
        graph argument as the direct path).
        """
        closed: Dict[str, float] = {}
        adjoints: Dict[str, tuple] = {}
        diagnostics: Dict[str, dict] = {}
        nonempty = [tau for tau, win in self._windows.items()
                    if win is not None and len(win["cut_seen"]) > 0]
        for tau in nonempty:
            win = self._windows[tau]
            z_all = torch.cat(win["zs_list"], dim=0)
            p_all = torch.cat(win["ps_list"], dim=0)
            w = torch.cat(win["weights_list"], dim=0)
            wf = WeightedWelford(int(z_all.shape[1]), int(p_all.shape[1]))
            wf.add(z_all, p_all, w, win["cut_ids_list"])
            r = wf.result()
            j, adj, score_diag = kf_adjoint(r, self.eps, self.strict)
            if j is None:
                closed[tau] = 0.0
                adjoints[tau] = None
                diagnostics[tau] = {"failed": score_diag["failed"],
                                    "W": r["W"], "D_cut": r["D"],
                                    "M_unique": int(len(win["cut_seen"])),
                                    "M_unique_trees": int(len(win["tree_seen"]))}
            else:
                czz = r["M2_zz"] / r["D"]
                cpp = r["M2_pp"] / r["D"]
                czp = r["M2_zp"] / r["D"]
                closed[tau] = float(j)
                adjoints[tau] = (adj, r)
                diagnostics[tau] = self._diagnostics(
                    win, czz, czp, cpp, torch.tensor(j, dtype=torch.float64),
                    0.0, score_diag)
        for tau in nonempty:
            self._windows.pop(tau)
        return closed, adjoints, diagnostics

    def window_ready(self) -> bool:
        """All nonempty windows have reached their unique-cut threshold."""
        nonempty = [tau for tau, win in self._windows.items()
                    if win is not None and len(win["cut_seen"]) > 0]
        return bool(nonempty) and all(
            len(self._windows[tau]["cut_seen"]) >= self._threshold(tau)
            for tau in nonempty)

    def reset(self):
        """Discard all open windows (used at epoch drain: the accumulated
        z rows' graphs are consumed by the task-only backward, so the
        unfinished window cannot survive into the next epoch)."""
        self._windows.clear()
