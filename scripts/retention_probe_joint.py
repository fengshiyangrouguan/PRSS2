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


# --------------------------------------------------------------- feature build
def ordered_pair_ids(man, sk, pk):
    """Presentation-ordered candidate ids for the (source, parent) nodes.

    ``man`` is one manifest row; ``sk``/``pk`` are the node keys (e.g.
    ``"leaf"``/``"a2"``).  Position 0 is the PRESENTED candidate and position 1
    is the other one, matching the swap bit recorded in the manifest.  The new
    label is ``Y_v = 0`` iff the presented candidate is the true future, i.e.
    the true candidate sits at position 0.
    """
    out = {}
    for key in (sk, pk):
        pres = int(man["presented"][key])
        pos = int(man["pos_cand"][key])
        neg = int(man["neg_cand"][key])
        out[key] = (pres, neg if pres == pos else pos)
    ys = 0 if int(man["presented"][sk]) == int(man["pos_cand"][sk]) else 1
    yp = 0 if int(man["presented"][pk]) == int(man["pos_cand"][pk]) else 1
    return out[sk], out[pk], int(ys), int(yp)


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


def fit_joint(Phi, C, Z, y, lam, use_state=True, sC=None, sZ=None,
              max_iter=500):
    """Fit the joint probe; base when ``use_state=False``.

    ``Z`` is the real state ``(N, d_z)``; ignored when ``use_state=False``.
    Scalers are learned on THIS set unless passed in, so a caller doing an inner
    time split must pass the train-side scalers for the tune set.
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
    res = minimize(_loss_grad, x0, jac=True, method="L-BFGS-B",
                   args=(Phi, Cs, Zs, y, lam, d_phi, d_c, d_z),
                   options={"maxiter": int(max_iter)})
    w_phi, W_C, W_Z = _unpack(res.x, d_phi, d_c, d_z)
    return {"w_phi": w_phi, "W_C": W_C, "W_Z": W_Z, "sC": float(sC),
            "sZ": float(sZ) if use_state else 1.0, "use_state": bool(use_state),
            "fun": float(res.fun), "nit": int(res.nit),
            "success": bool(res.success),
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


def select_joint(Phi, C, Z, y, folds,
                 lams=(1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0),
                 use_state=True, fallback_fn=None, max_iter=500):
    """Pick lambda by mean inner-fold validation NLL (ties -> larger lambda).

    ``folds`` is a list of ``(fit_idx, tune_idx)`` index arrays into the rows --
    an expanding-window time split with a purge gap.  ``fallback_fn`` returns an
    optional simpler probe (e.g. the fitted base reproduced with W_Z = 0) that
    the result must beat; on a tie the simpler one wins.
    """
    def val_of(p):
        return float(np.mean([
            float(joint_row_nll(p, Phi[t], C[t], Z[t], y[t]).mean())
            for _, t in folds]))

    best = None
    for lam in sorted([float(x) for x in lams], reverse=True):
        tot = 0.0
        for fit, tune in folds:
            p = fit_joint(Phi[fit], C[fit], Z[fit], y[fit], lam,
                          use_state=use_state, max_iter=max_iter)
            tot += float(joint_row_nll(
                p, Phi[tune], C[tune], Z[tune], y[tune]).mean())
        key = (tot / len(folds), lam)          # ties -> larger lambda = simpler
        if best is None or key < best[0]:
            best = (key, lam)
    params = fit_joint(Phi, C, Z, y, best[1], use_state=use_state,
                       max_iter=max_iter)
    val = val_of(params)
    if use_state and fallback_fn is not None:
        fb = fallback_fn()
        if fb is not None and val_of(fb) <= val + 1e-12:
            return fb, val_of(fb)
    return params, val


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


def ratio_ci(j_src, j_pos, alpha=0.05):
    """Paired R = J_pos / J_src from replicate vectors.

    Returns ``(R_point, lo, hi, identifiable)``.  ``identifiable`` is False when
    a non-trivial share of replicates has a source denominator at or below
    zero, in which case the ratio is not bounded and no CI is reported (the
    negative-denominator replicates are never dropped to fake a narrow one).
    """
    j_src = np.asarray(j_src, dtype=np.float64)
    j_pos = np.asarray(j_pos, dtype=np.float64)
    ok = j_src > 1e-12
    frac_ok = float(ok.mean()) if j_src.size else 0.0
    if frac_ok < 0.95:
        return (float("nan"), float("nan"), float("nan"), False)
    r = j_pos[ok] / j_src[ok]
    return (float(np.mean(j_pos) / np.mean(j_src)),
            float(np.percentile(r, 100 * alpha / 2)),
            float(np.percentile(r, 100 * (1 - alpha / 2))), True)
