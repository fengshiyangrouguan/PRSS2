#!/usr/bin/env python3
"""Joint 4-class predictive-gain probe (B2 spec, 2026-09-14).

Measures how much of a source's joint local future the model's REAL propagated
state still lets an independent probe decode, at each downstream position.

Estimator (fixed, no null correction)
-------------------------------------
    J_k = (1/n) sum_i (NLL0_i - NLL1_{k,i}) / ln 2      [bits per joint query]

No permutation-mean subtraction, no clipping to zero, no null gate.  A negative
value means the fitted full probe did not beat the fitted base probe on held-out
rows -- not that the state carries negative information.

Probe family (fixed)
--------------------
Candidate pairs are ORDERED: for node s the two candidates are in presentation
order (position 0 = the presented one, position 1 = the other); likewise the
parent.  With E the frozen r-dim SVD encoder:

    phi_ab  = [E(c_s^a); E(c_p^b); E(c_s^a) * E(c_p^b)]      in R^{3r}
    score0_ab = (w_phi + W_C C) . phi_ab                       base
    score1_ab = (w_phi + W_C C + W_Z Z_k) . phi_ab              full

One softmax over the four (a, b) in {0,1}^2; the label class is
``2 * Y_s + Y_p`` where ``Y_v`` is the presentation position of the true
candidate.  This is candidate-order equivariant and can express joint / XOR
structure that two independent per-node sigmoids cannot.

Z_k is the REAL propagated state (the rows' ``keep``) -- never the keep/remove
difference Delta.  ``Delta = keep - remove`` is not the model's state: with
independent bits Y, K and Z_keep = Y xor K, Z_remove = K, Delta carries Y while
Z_keep does not, so the two are informationally unrelated.

The state block can be driven to exactly zero (W_Z = 0), so the full family
strictly contains the base family; ``fit_joint(..., use_state=False)`` is the
base.  Objective is mean joint NLL + (lam/2)*||W||_F^2, which is convex in the
parameters, so L-BFGS-B converges reliably.

Everything here is pure numpy; no torch, no model, no GPU.

Terminology (fixed)
-------------------
  * ``J_{s->k}`` -- held-out PREDICTIVE GAIN of the state at position k about
    the source's joint local future, under the restricted probe family.  It is
    NOT Shannon mutual information, and a negative value is a finite-sample
    failure of the fitted probe, not negative information.
  * ``R_{s->k} = J_{s->k} / J_{s->s}`` -- predictive-signal retention ratio.
  * ``R_{s->s} = 1`` -- only the normalisation origin; it does not mean the
    source carries one bit.
  * ``RootGain_h`` -- predictive gain at the root of the h-hop source signal.
  * ``p_tail_boot`` -- Monte-Carlo bootstrap tail probability (descriptive);
    the formal test is ``block_signflip_p``.
"""

import hashlib

import numpy as np


def _sha16(a):
    return hashlib.sha256(np.ascontiguousarray(
        np.asarray(a, np.float64)).tobytes()).hexdigest()[:16]


def build_svd_encoder(train_src, train_dst, n_nodes, rank=16, random_state=0):
    """Frozen training-graph truncated-SVD candidate encoder.

    ``M[i,j] = log(1 + #edges i->j)``, degree-normalized, then
    ``E(j) = V_r[j, :] * sqrt(S_r)``.  Build it from a TRAIN PREFIX that ends no
    later than the earliest probe fit point, and record the hashes so every arm
    provably shares the identical table.
    """
    M = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    np.add.at(M, (np.asarray(train_src, np.int64),
                  np.asarray(train_dst, np.int64)), 1.0)
    M = np.log1p(M)
    dout, din = M.sum(axis=1), M.sum(axis=0)
    do = np.where(dout > 0, dout ** -0.5, 0.0)
    di = np.where(din > 0, din ** -0.5, 0.0)
    Mbar = (do[:, None] * M) * di[None, :]
    r = int(min(rank, min(Mbar.shape)))
    _, S, Vt = np.linalg.svd(Mbar, full_matrices=False)
    E = np.ascontiguousarray(Vt[:r, :].T * np.sqrt(S[:r])[None, :])
    return E, {"svd_rank": r, "matrix_hash": _sha16(Mbar),
               "embedding_hash": _sha16(E), "train_edges": int(len(train_src))}


# --------------------------------------------------------------- label kinds
# CANONICAL: Y_v = the PRESENTATION POSITION of the true candidate (0 or 1).
# LEGACY (audit_retention_v2 rows): Y_v = 1 iff the presented candidate is the
# true future destination, i.e. Y_canonical = 1 - Y_legacy.
CANONICAL_LABEL_KIND = "true_candidate_position"
LEGACY_LABEL_KIND = "presented_is_positive"
LABEL_KINDS = (CANONICAL_LABEL_KIND, LEGACY_LABEL_KIND)


def canonical_bits(man, key):
    """Canonical label for one node: 0 iff the true candidate sits at pos 0."""
    return 0 if int(man["presented"][key]) == int(man["pos_cand"][key]) else 1


def row_label_to_canonical(y_row, label_kind):
    """Map a stored row label into the canonical convention."""
    if label_kind == CANONICAL_LABEL_KIND:
        return int(y_row)
    if label_kind == LEGACY_LABEL_KIND:
        return 1 - int(y_row)
    raise ValueError("unknown label_kind {!r}".format(label_kind))


# --------------------------------------------------------------- feature build
def ordered_pair_ids(man, sk, pk):
    """Presentation-ordered candidate ids for the (source, parent) nodes.

    ``man`` is one manifest row; ``sk``/``pk`` are the node keys (e.g.
    ``"leaf"``/``"a2"``).  Position 0 is the PRESENTED candidate and position 1
    is the other one, so the canonical label ``Y_v`` is the position of the true
    candidate.  These labels come from the manifest alone and are therefore
    independent of whatever convention the stored rows use.
    """
    out = {}
    for key in (sk, pk):
        pres = int(man["presented"][key])
        pos = int(man["pos_cand"][key])
        neg = int(man["neg_cand"][key])
        out[key] = (pres, neg if pres == pos else pos)
    return out[sk], out[pk], canonical_bits(man, sk), canonical_bits(man, pk)


def phi_tensor(E, s_pair, p_pair):
    """``(N, 4, 3r)`` candidate-pair features, row order (a,b) = (0,0),(0,1),(1,0),(1,1).

    ``s_pair``/``p_pair`` are ``(N, 2)`` int arrays of ordered candidate node
    ids; ``E`` is ``(n_nodes, r)``.  Index ``c = 2*a + b`` matches the label
    class ``2*Y_s + Y_p``.
    """
    E = np.asarray(E, dtype=np.float64)
    s_pair = np.asarray(s_pair, dtype=np.int64)
    p_pair = np.asarray(p_pair, dtype=np.int64)
    n, r = len(s_pair), E.shape[1]
    out = np.empty((n, 4, 3 * r), dtype=np.float64)
    for a in (0, 1):
        es = E[s_pair[:, a]]
        for b in (0, 1):
            ep = E[p_pair[:, b]]
            out[:, 2 * a + b, :] = np.concatenate(
                [es, ep, es * ep], axis=1)
    return out


def joint_class(ys, yp):
    """Joint 4-class label ``2*Y_s + Y_p``."""
    return 2 * np.asarray(ys, np.int64) + np.asarray(yp, np.int64)


# -------------------------------------------------------------------- fitting
def rms_scaler(F, eps=1e-12):
    """Single isotropic RMS scale learned on the fit set (scale/rotation safe)."""
    F = np.asarray(F, dtype=np.float64)
    if F.size == 0:
        return 1.0
    s = float(np.sqrt(np.mean(F * F)))
    return s if s > eps else 1.0


def _unpack(theta, d_phi, d_c, d_z):
    i = 0
    w_phi = theta[i:i + d_phi]
    i += d_phi
    W_C = theta[i:i + d_phi * d_c].reshape(d_phi, d_c)
    i += d_phi * d_c
    if d_z:
        W_Z = theta[i:i + d_phi * d_z].reshape(d_phi, d_z)
    else:
        W_Z = None
    return w_phi, W_C, W_Z


def _logits(w_phi, W_C, W_Z, Phi, Cs, Zs):
    U = w_phi[None, :] + Cs @ W_C.T
    if W_Z is not None and Zs is not None:
        U = U + Zs @ W_Z.T
    return np.einsum('nk,nck->nc', U, Phi)


def _loss_grad(theta, Phi, Cs, Zs, y, lam, d_phi, d_c, d_z):
    w_phi, W_C, W_Z = _unpack(theta, d_phi, d_c, d_z)
    logits = _logits(w_phi, W_C, W_Z, Phi, Cs, Zs)
    n = len(y)
    m = logits.max(axis=1, keepdims=True)
    e = np.exp(logits - m)
    s = e.sum(axis=1, keepdims=True)
    logp = (logits - m) - np.log(s)
    nll = -logp[np.arange(n), y]
    P = e / s
    P[np.arange(n), y] -= 1.0
    g = np.einsum('nc,nck->nk', P, Phi)
    reg = 0.5 * lam * float(w_phi @ w_phi + (W_C * W_C).sum()
                            + (0.0 if W_Z is None else float((W_Z * W_Z).sum())))
    loss = float(nll.mean()) + reg
    parts = [(g.mean(axis=0) + lam * w_phi),
             ((g.T @ Cs) / n + lam * W_C).reshape(-1)]
    if W_Z is not None:
        parts.append(((g.T @ Zs) / n + lam * W_Z).reshape(-1))
    return loss, np.concatenate(parts)


class ProbeFitError(RuntimeError):
    """The convex fit did not converge cleanly; the number is not reportable."""


def fit_joint(Phi, C, Z, y, lam, use_state=True, sC=None, sZ=None,
              max_iter=500):
    """Fit the joint probe; base when ``use_state=False``.

    ``Z`` is the real state ``(N, d_z)``; ignored when ``use_state=False``.
    Scalers are learned on THIS set unless passed in, so a caller doing an inner
    time split must pass the train-side scalers for the tune set.

    Raises ``ProbeFitError`` if the optimiser reports failure, if any parameter
    is non-finite, or if the objective is NaN -- an unconverged fit must never
    silently become a number.
    """
    from scipy.optimize import minimize
    Phi = np.asarray(Phi, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    d_phi = Phi.shape[-1]
    d_c = C.shape[1]
    d_z = int(np.asarray(Z).shape[1]) if use_state else 0
    if sC is None:
        sC = rms_scaler(C)
    if use_state and sZ is None:
        sZ = rms_scaler(Z)
    Cs = C / sC
    Zs = (np.asarray(Z, dtype=np.float64) / (sZ or 1.0)) if use_state else None
    x0 = np.zeros(d_phi + d_phi * d_c + d_phi * d_z)
    try:
        res = minimize(_loss_grad, x0, jac=True, method="L-BFGS-B",
                       args=(Phi, Cs, Zs, y, lam, d_phi, d_c, d_z),
                       options={"maxiter": int(max_iter)})
    except Exception as e:                       # noqa: BLE001
        raise ProbeFitError(
            "joint probe fit raised (lam={}, use_state={}, n={}): {}".format(
                lam, use_state, len(y), e)) from e
    if (not res.success) or (not np.all(np.isfinite(res.x))) \
            or (not np.isfinite(res.fun)):
        raise ProbeFitError(
            "joint probe fit failed (lam={}, use_state={}, n={}): status={} "
            "success={} nit={} fun={} msg={}".format(
                lam, use_state, len(y), res.status, res.success, res.nit,
                res.fun, res.message))
    w_phi, W_C, W_Z = _unpack(res.x, d_phi, d_c, d_z)
    return {"w_phi": w_phi, "W_C": W_C, "W_Z": W_Z, "sC": float(sC),
            "sZ": float(sZ) if use_state else 1.0, "use_state": bool(use_state),
            "fun": float(res.fun), "nit": int(res.nit),
            "success": bool(res.success), "status": int(res.status),
            "n_params": int(x0.size), "lam": float(lam)}


def joint_row_nll(params, Phi, C, Z, y):
    """Per-row natural-log NLL of the fitted probe on the given rows."""
    Phi = np.asarray(Phi, dtype=np.float64)
    Cs = np.asarray(C, dtype=np.float64) / params["sC"]
    Zs = (np.asarray(Z, dtype=np.float64) / params["sZ"]
          if params["use_state"] else None)
    if not params["use_state"]:
        Zs = np.zeros((len(y), 0))
    logits = _logits(params["w_phi"], params["W_C"],
                     params["W_Z"] if params["use_state"] else None,
                     Phi, Cs, Zs)
    n = len(y)
    m = logits.max(axis=1, keepdims=True)
    logp = (logits - m) - np.log(np.exp(logits - m).sum(axis=1, keepdims=True))
    return -logp[np.arange(n), np.asarray(y, np.int64)]


def params_hash(params):
    """Stable hash of the fitted numeric parameters (leakage / A-A checks)."""
    h = hashlib.sha256()
    for k in ("w_phi", "W_C", "W_Z"):
        v = params.get(k)
        if v is not None:
            h.update(np.ascontiguousarray(v, np.float64).tobytes())
    for k in ("sC", "sZ", "use_state", "lam"):
        h.update(("%s=%r;" % (k, params.get(k))).encode())
    return h.hexdigest()[:16]


def oof_nll(Phi, C, Z, y, folds, lam, use_state=True, max_iter=500):
    """Out-of-fold NLL: every fold fits on ``fit`` and scores ``tune``.

    Nothing from ``tune`` ever enters the parameters used to score it, so this
    is a genuine held-out estimate of the family's generalisation.
    """
def class_counts(y, n_class=4):
    """Joint-class histogram (classes 0..n_class-1)."""
    return np.bincount(np.asarray(y, np.int64),
                       minlength=int(n_class))[:int(n_class)].tolist()


def require_all_classes(y, n_class=4, where="", exc=ProbeFitError):
    """Raise unless the four joint classes are all present.  Returns counts.

    A probe family is only identified when every joint class is represented;
    with a missing class the fit can be nearly separable and the optimiser is
    unreliable, so this fails loudly instead of pruning rows.
    """
    c = class_counts(y, n_class)
    missing = [i for i, k in enumerate(c) if k == 0]
    if missing:
        raise exc("{}: joint classes {} missing (counts {})".format(
            where or "probe rows", missing, c))
    return c


def oof_nll(Phi, C, Z, y, folds, lam, use_state=True, max_iter=500):
    """Out-of-fold NLL for one lambda.

    Returns ``(value, statuses)``; ``value`` is ``None`` when ANY fold failed to
    fit (the failure is recorded per fold in ``statuses`` and that lambda is
    simply skipped, rather than aborting the whole selection).
    """
    per, statuses = [], []
    for fi, (fit, tune) in enumerate(folds):
        try:
            p = fit_joint(Phi[fit], C[fit], Z[fit], y[fit], lam,
                          use_state=use_state, max_iter=max_iter)
            v = float(joint_row_nll(
                p, Phi[tune], C[tune], Z[tune], y[tune]).mean())
            if not np.isfinite(v):
                raise ProbeFitError("non-finite OOF NLL")
            per.append(v)
            statuses.append({"fold": fi, "ok": True, "oof_nll": v})
        except ProbeFitError as e:
            statuses.append({"fold": fi, "ok": False, "error": str(e)})
    if len(per) != len(folds):
        return None, statuses
    return float(np.mean(per)), statuses


def select_joint(Phi, C, Z, y, folds,
                 lams=(1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0),
                 use_state=True, fallback=None, max_iter=500, tol=0.0):
    """Pick lambda on OUT-OF-FOLD NLL, then refit on all rows once.

    ``folds``: ``(fit_idx, tune_idx)`` pairs (expanding-window time split).
    Selection uses only out-of-fold predictions; the returned parameters are a
    single refit on the full row set with the chosen lambda, and the audit set
    is never touched.  Ties go to the LARGER lambda (the simpler model).

    A lambda that fails to fit in any fold is marked invalid and skipped; only
    when EVERY candidate lambda is invalid does the selection raise.  The refit
    at the chosen lambda is not tolerated -- it must converge.

    ``fallback``: optional ``(params, oof_nll)`` for a simpler family that must
    not be beaten before the richer family is used (e.g. the base reproduced
    with ``W_Z = 0``).  Its ``oof_nll`` must have been computed on the same
    folds.  On a tie the fallback wins.

    Returns ``(params, oof_nll, diagnostics)``.
    """
    trials, best_lam, best_oof = [], None, None
    for lam in sorted([float(x) for x in lams], reverse=True):
        v, st = oof_nll(Phi, C, Z, y, folds, lam, use_state=use_state,
                        max_iter=max_iter)
        trials.append({"lam": lam, "valid": v is not None, "oof_nll": v,
                       "folds": st})
        if v is None:
            continue
        if best_oof is None or v < best_oof - tol:   # ties -> larger lambda
            best_lam, best_oof = lam, v
    if best_lam is None:
        raise ProbeFitError(
            "every candidate lambda failed in at least one fold (use_state={}, "
            "n_folds={}); reasons: {}".format(
                use_state, len(folds),
                [[f.get("error") for f in t["folds"]] for t in trials]))
    diag = {"trials": trials, "chosen_lam": best_lam,
            "n_invalid": int(sum(1 for t in trials if not t["valid"]))}
    if use_state and fallback is not None and fallback[1] <= best_oof + 1e-12:
        return fallback[0], float(fallback[1]), dict(diag, used_fallback=True)
    params = fit_joint(Phi, C, Z, y, best_lam, use_state=use_state,
                       max_iter=max_iter)
    return params, float(best_oof), dict(diag, used_fallback=False)


def base_as_full(base_params, d_z):
    """Reproduce a fitted base as a full probe with the state block zeroed."""
    return {"w_phi": base_params["w_phi"], "W_C": base_params["W_C"],
            "W_Z": np.zeros((base_params["w_phi"].shape[0], int(d_z))),
            "sC": base_params["sC"], "sZ": 1.0, "use_state": True,
            "fun": base_params["fun"], "nit": base_params["nit"],
            "success": base_params["success"],
            "n_params": int(base_params["n_params"] + base_params["w_phi"].shape[0] * d_z),
            "lam": base_params["lam"]}


# ----------------------------------------------------------------- statistics
def gain_bits(nll0, nll1):
    """J = mean(NLL0 - NLL1)/ln2 for two per-row NLL vectors (bits)."""
    return float(np.mean(np.asarray(nll0, np.float64)
                         - np.asarray(nll1, np.float64)) / np.log(2.0))


# ---------------------------------------------------------- block bootstrap
def pair_key(pair_id):
    """Canonical string key for a globally-unique pair id.

    ``np.unique`` on an array built from a list of tuples flattens the tuples
    into scalars, silently collapsing the grouping; a string key is immune.
    """
    return "|".join(str(int(x)) for x in pair_id)


def time_blocks(times, n_blocks):
    """Assign rows to contiguous time blocks by quantile of ``times``."""
    t = np.asarray(times, dtype=np.float64)
    if n_blocks <= 1 or t.size == 0:
        return np.zeros(t.size, dtype=np.int64)
    qs = np.quantile(t, np.linspace(0.0, 1.0, int(n_blocks) + 1)[1:-1])
    return np.digitize(t, qs).astype(np.int64)


def bootstrap_indices(block_ids, n_boot, seed):
    """Block-resampled row-index sets, usable unchanged for every arm/metric."""
    block_ids = np.asarray(block_ids)
    uniq = np.unique(block_ids)
    by = {b: np.where(block_ids == b)[0] for b in uniq}
    rng = np.random.RandomState(int(seed))
    out = []
    for _ in range(int(n_boot)):
        pick = uniq[rng.randint(0, len(uniq), size=len(uniq))]
        out.append(np.concatenate([by[b] for b in pick]))
    return out


def paired_bootstrap(idx_sets, nll0, nll1_by_pos):
    """J replicates for every position, computed jointly on shared resamples.

    ``nll0``: ``(N,)`` base per-row NLL, ``nll1_by_pos``: ``{pos: (N,)}``.
    Returns ``{pos: (n_boot,)}`` of J in bits.  Source and downstream positions
    enter the same replicate, so ratios of them are paired.
    """
    out = {pos: np.empty(len(idx_sets), dtype=np.float64)
           for pos in nll1_by_pos}
    nll0 = np.asarray(nll0, dtype=np.float64)
    for b, idx in enumerate(idx_sets):
        m0 = float(nll0[idx].mean())
        for pos, v in nll1_by_pos.items():
            out[pos][b] = (m0 - float(np.asarray(v, np.float64)[idx].mean())) \
                / np.log(2.0)
    return out


def ratio_ci(j_src_reps, j_pos_reps, j_src_point, j_pos_point, alpha=0.05):
    """Paired R = J_pos / J_src from replicate vectors and the point estimates.

    Returns ``(R_point, lo, hi, identifiable)``.  Identifiability is decided by
    whether the percentile CI of the SOURCE gain has a lower bound above zero.
    When it does, the CI uses ALL paired replicates (never dropping the ones
    with a non-positive denominator, which would fake a narrow interval); when
    it does not, the ratio is unbounded and reported as not identifiable.  The
    point estimate is strictly ``J_point,pos / J_point,src`` -- it is never
    replaced by a bootstrap mean.
    """
    j_src = np.asarray(j_src_reps, dtype=np.float64)
    j_pos = np.asarray(j_pos_reps, dtype=np.float64)
    if j_src.size == 0 or not np.isfinite(j_src_point) \
            or not np.isfinite(j_pos_point):
        return (float("nan"), float("nan"), float("nan"), False)
    lo_src = float(np.percentile(j_src, 100 * alpha / 2))
    if not np.isfinite(lo_src) or lo_src <= 0.0:
        return (float("nan"), float("nan"), float("nan"), False)
    r = j_pos / j_src
    return (float(j_pos_point) / float(j_src_point),
            float(np.percentile(r, 100 * alpha / 2)),
            float(np.percentile(r, 100 * (1 - alpha / 2))), True)


def paired_diff_ci(a_reps, b_reps, a_point, b_point, alpha=0.05):
    """Paired difference ``a - b`` across shared replicates.

    Returns ``(diff_point, lo, hi, p_tail_boot)``.  ``p_tail_boot`` is a
    MONTE-CARLO BOOTSTRAP TAIL PROBABILITY (twice the smaller tail share of the
    replicate distribution) -- it is not an exact sign test and is reported as a
    descriptive tail share only.  The formal p-value is
    :func:`block_signflip_p`.  The point estimate uses the two point estimates,
    not the replicate means.
    """
    d = np.asarray(a_reps, dtype=np.float64) - np.asarray(b_reps, dtype=np.float64)
    if d.size == 0:
        return (float("nan"),) * 4
    p = 2.0 * min(float((d <= 0).mean()), float((d >= 0).mean()))
    p = min(1.0, max(p, 1.0 / (d.size + 1)))
    return (float(a_point) - float(b_point),
            float(np.percentile(d, 100 * alpha / 2)),
            float(np.percentile(d, 100 * (1 - alpha / 2))), float(p))


def block_signflip_p(d_rows, block_ids, n_perm=2000, seed=0,
                     alternative="greater"):
    """Paired time-block sign-flip (randomization) p-value.

    ``d_rows`` are the per-row paired contributions to the statistic (for a
    RootGain difference, ``(nll_other - nll_ours)/ln2`` per row).  Rows are
    grouped into time blocks; under the null each block's contribution is
    equally likely to have either sign, so the reference distribution is
    ``(1/n) * sum_b eps_b * s_b`` with ``eps_b`` i.i.d. +/-1.  The direction is
    PRE-REGISTERED, so the default is a one-sided test.
    """
    d = np.asarray(d_rows, dtype=np.float64)
    blocks = np.asarray(block_ids)
    uniq = np.unique(blocks)
    sums = np.array([float(d[blocks == b].sum()) for b in uniq],
                    dtype=np.float64)
    n = d.size
    if n == 0 or uniq.size == 0:
        return float("nan")
    obs = float(d.mean())
    rng = np.random.RandomState(int(seed))
    cnt = 0
    for _ in range(int(n_perm)):
        eps = rng.choice([-1.0, 1.0], size=uniq.size)
        star = float(eps @ sums) / n
        extreme = (star >= obs - 1e-15) if alternative == "greater" \
            else (abs(star) >= abs(obs) - 1e-15)
        if extreme:
            cnt += 1
    return (cnt + 1.0) / (int(n_perm) + 1.0)


def holm_adjust(pvals):
    """Holm step-down adjustment; returns an array in the input order."""
    p = np.asarray(pvals, dtype=np.float64)
    m = p.size
    if m == 0:
        return p
    order = np.argsort(p)
    adj = np.empty(m, dtype=np.float64)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, float(m - rank) * float(p[idx]))
        adj[idx] = min(1.0, running)
    return adj


def verify_encoder(E, meta, expect_rank=None, cutoff_max=None,
                   dataset_hash=None):
    """Return a list of problems with a supplied/locked SVD table (empty = ok).

    A precomputed ``E`` must be provably the table it claims to be: the stored
    embedding hash must equal the recomputed one, the rank must match the array
    width, the cutoff must not be later than the earliest probe fit time, and
    the dataset hash must match when one is available.
    """
    problems = []
    E = np.asarray(E, dtype=np.float64)
    stored = meta.get("embedding_hash")
    if stored is None:
        problems.append("encoder meta has no embedding_hash")
    elif str(stored) != _sha16(E):
        problems.append("embedding_hash mismatch: stored {} != recomputed {}"
                        .format(stored, _sha16(E)))
    width = int(E.shape[1]) if E.ndim == 2 else -1
    if int(meta.get("svd_rank", width)) != width:
        problems.append("svd_rank {} != E width {}".format(
            meta.get("svd_rank"), width))
    if expect_rank is not None and width != int(expect_rank):
        problems.append("E width {} != expected rank {}".format(
            width, int(expect_rank)))
    if cutoff_max is not None:
        if "cutoff_time" not in meta:
            problems.append("encoder meta has no cutoff_time")
        elif float(meta["cutoff_time"]) > float(cutoff_max):
            problems.append(
                "encoder cutoff {} is after the earliest calib time {}"
                .format(meta["cutoff_time"], cutoff_max))
    if dataset_hash is not None:
        if "dataset_hash" not in meta:
            problems.append(
                "encoder meta has no dataset_hash but one is required "
                "(expected {})".format(dataset_hash))
        elif str(meta["dataset_hash"]) != str(dataset_hash):
            problems.append("dataset_hash mismatch: {} != {}".format(
                meta.get("dataset_hash"), dataset_hash))
    return problems
