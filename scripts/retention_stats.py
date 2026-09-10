"""Pure-numpy statistics for the retention audit (no torch / no model deps).

Retention of a *source* prediction component along a leaf-to-root path.

For a source depth s, let X_s be the source representation (full state at the
source position) and P the fixed future witness.  The source predictive
component is Q_s = X_s W_s with W_s ridge-fit on the calibration set mapping
X_s -> P.  At every downstream position k the source leaves a paired-removal
delta Delta_{s->k} (keep minus remove, so parent self / siblings cancel).
Retention at k is the explained variance of Q_s by [Delta_{s->k}; C]:

    R_{s->k} = 1 - ||Q_s - Qhat_s||_F^2 / (||Q_s - mean(Q_s)||_F^2 + eps)

with Qhat_s the ridge prediction of Q_s from [Delta; C] fitted on calib and
evaluated on the audit set.  Because the numerator is a squared error,
R <= 1 by construction; negative values are reported honestly (never clipped
except an explicit display clamp at +1).  Each point is estimated by a direct
regression against the SAME Q_s -- retention is never a chain product of local
factors and never a ratio of two unlike quantities.
"""

import numpy as np


def _center_scale(Fc, Fa, eps_scale=1e-9):
    """Center by calib mean / scale by calib std; apply same to audit."""
    m = Fc.mean(axis=0)
    s = Fc.std(axis=0) + eps_scale
    return m, s, (Fc - m) / s, (Fa - m) / s


def _ridge_coefs(F, Y, lam):
    d = F.shape[1]
    A = F.T @ F + lam * np.eye(d)
    return np.linalg.solve(A, F.T @ Y)


def fit_ridge_map(Fc, Yc, lam=1e-2):
    """Fit a linear (ridge) map on calibrated, standardized features.

    Returns a dict carrying everything needed to evaluate on audit rows:
    coefficients on standardized features, and calib mean/scale/intercept.
    """
    m, s, Fc_, _ = _center_scale(Fc, Fc)
    my = Yc.mean(axis=0, keepdims=True)
    Yc_ = Yc - my
    w = _ridge_coefs(Fc_, Yc_, lam)
    return {"w": w, "mf": m, "sf": s, "my": my}


def apply_ridge_map(mp, Fa):
    """Predictions of the target from audit-row features Fa."""
    Fa_ = (Fa - mp["mf"]) / mp["sf"]
    return mp["my"] + Fa_ @ mp["w"]


def explained_var(Ya, Yhat, eps=1e-6):
    """1 - SSE/SST in audit rows.  <=1 since SSE >= 0; negative is honest."""
    sse = float(np.sum((Ya - Yhat) ** 2))
    sst = float(np.sum((Ya - Ya.mean(axis=0, keepdims=True)) ** 2))
    return 1.0 - sse / (sst + eps)


def retention_R(Qc, Fc, Qa, Fa, lam=1e-2, eps=1e-6):
    """Explained variance of source component Q by features F, held out."""
    return explained_var(Qa, apply_ridge_map(fit_ridge_map(Fc, Qc, lam), Fa),
                         eps=eps)


def retention_map_R(mp, Qa, Fa, eps=1e-6):
    """Retention with a pre-fitted map (used inside the bootstrap loop)."""
    return explained_var(Qa, apply_ridge_map(mp, Fa), eps=eps)


def source_component(Xc, Pc, Xa, lam=1e-2):
    """Q_s = ridge(X_c -> P_c), evaluated on calib and audit rows.

    Returns (Qcal, Qaud) -- the source's own linear prediction of the future
    witness.  This is the single object every retention point of that line
    tries to recover downstream.
    """
    mp = fit_ridge_map(Xc, Pc, lam=lam)
    return apply_ridge_map(mp, Xc), apply_ridge_map(mp, Xa)


def strata_ids(keys, n_bins=12):
    """Matched-context strata on a 1-D numeric key (quantile bins)."""
    keys = np.asarray(keys, dtype=np.float64)
    q = np.quantile(keys, np.linspace(0.0, 1.0, n_bins + 1))
    q[0] -= 1.0
    q[-1] += 1.0
    return np.digitize(keys, q) - 1


def permute_within_strata(X, strata, rng):
    """Row-permuted copy of X inside each stratum (breaks the X<->target link
    while preserving the matched-context distribution)."""
    Xp = X.copy()
    for s in np.unique(strata):
        m = strata == s
        perm = rng.permutation(int(m.sum()))
        Xp[m] = X[m][perm]
    return Xp


def source_signal_and_null(Xc, Pc, Xa, Pa, strata, seed, lam=1e-2,
                           n_null=200, eps=1e-6):
    """Predictive energy of the source rep X_s for P, gated vs shuffle null.

    value = explained variance of audit P from X_s (map fit on calib).
    null_p95 = 95th percentile of the same energy after permuting X_s within
    matched-context strata.  A line is only usable when value > null_p95.
    """
    mp = fit_ridge_map(Xc, Pc, lam=lam)
    base = explained_var(Pa, apply_ridge_map(mp, Xa), eps=eps)
    rng = np.random.RandomState(seed)
    nulls = []
    for _ in range(n_null):
        Xa_p = permute_within_strata(Xa, strata, rng)
        nulls.append(explained_var(Pa, apply_ridge_map(mp, Xa_p), eps=eps))
    return float(base), float(np.percentile(nulls, 95))


def retention_bootstrap(mp, Qa, Fa, idx_all, n_boot, seed, eps=1e-6):
    """Cluster-bootstrap CIs.  Map is fit once on calib; audit rows resampled
    with replacement (each row = one root tree -> root-level clusters)."""
    rng = np.random.RandomState(seed)
    out = []
    for _ in range(n_boot):
        idx = rng.choice(idx_all, size=len(idx_all), replace=True)
        out.append(retention_map_R(mp, Qa[idx], Fa[idx], eps=eps))
    return np.asarray(out)


def weighted_dir_sqcorr(A, B, weights):
    """Weighted mean over columns of squared Pearson corr between A and B.

    Used to score a (recovered) source component against the fixed root-target
    directions: value in [0, 1], so it can be read as a percentage."""
    return corr2_recoverability(A, B, weights=weights)


def retention_ci(arr):
    return (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))


def centered_sq_corr_strength(A, B, lam=1e-2, eps=1e-6):
    """Audit-centered squared-canonical-correlation energy J (>=0).

    J = tr[(Cbb + eps I)^{-1} Cba (Caa + lam I)^{-1} Cab] on row-centered A,B
    = sum of squared canonical correlations.  Mean/scale invariant (add a
    constant to either audit set and J is unchanged), so cross-block drift
    cannot push it negative -- unlike plain explained variance."""
    A = A - A.mean(axis=0)
    B = B - B.mean(axis=0)
    n = len(A)
    Caa = A.T @ A / n
    Cbb = B.T @ B / n
    Cab = A.T @ B / n
    M = np.linalg.solve(Caa + lam * np.eye(A.shape[1]), Cab)
    return float(np.trace(np.linalg.solve(Cbb + eps * np.eye(B.shape[1]),
                                          Cab.T @ M)))


def _dir_pearson2(Q, Qhat):
    """Per-direction squared Pearson correlation (0 where a direction has no
    variance)."""
    Q = Q - Q.mean(axis=0)
    Qh = Qhat - Qhat.mean(axis=0)
    vq = np.sqrt((Q ** 2).mean(axis=0))
    vh = np.sqrt((Qh ** 2).mean(axis=0))
    den = vq * vh
    good = den > 1e-12
    vals = np.zeros(Q.shape[1])
    vals[good] = ((Q * Qh).mean(axis=0)[good] / den[good]) ** 2
    return vals


def corr2_recoverability(Q, Qhat, weights=None):
    """Centered, correlation-type recoverability in [0, 1].

    Mean over the k source directions of the squared Pearson correlation
    between Q_s and its recovery Qhat.  With ``weights`` (e.g. the canonical
    strengths squared), directions are combined as a strength-weighted mean
    instead of equally.
    """
    vals = _dir_pearson2(Q, Qhat)
    if weights is None:
        return float(vals.mean())
    w = np.asarray(weights, dtype=np.float64)
    sw = w.sum()
    if sw <= 0:
        return float(vals.mean())
    return float((w * vals).sum() / sw)


def corr2_map_metric(mp, Qa, Fa, weights=None):
    """corr² recoverability of Q_s from features Fa using a pre-fitted map."""
    return corr2_recoverability(Qa, apply_ridge_map(mp, Fa), weights=weights)


def corr2_bootstrap(mp, Qa, Fa, n_boot, seed, weights=None):
    """Cluster-bootstrap CI for the corr² recoverability (map fit on calib)."""
    rng = np.random.RandomState(seed)
    n = len(Qa)
    idx_all = np.arange(n)
    out = []
    for _ in range(n_boot):
        idx = rng.choice(idx_all, size=n, replace=True)
        out.append(corr2_map_metric(mp, Qa[idx], Fa[idx], weights=weights))
    return np.asarray(out)


def j_strength_bootstrap(A, B, n_boot, seed, lam=1e-2, eps=1e-6):
    """Cluster-bootstrap CI for the centered squared-canonical strength J."""
    rng = np.random.RandomState(seed)
    n = len(A)
    idx_all = np.arange(n)
    out = []
    for _ in range(n_boot):
        idx = rng.choice(idx_all, size=n, replace=True)
        out.append(centered_sq_corr_strength(A[idx], B[idx], lam=lam,
                                             eps=eps))
    return np.asarray(out)


# ------------------------------------------------------------------ source dirs
def _chol_lower_inv(A):
    """Returns T = L^{-1} with A = L L^T (lower Cholesky)."""
    L = np.linalg.cholesky(A)
    return np.linalg.inv(L)


def canonical_dirs(Xc, Pc, k, lam=1e-2, eps=1e-6, scale_eps=1e-9):
    """Top-k predictive directions between source state X_s and its local
    future P_s, estimated once on the calibration set.

    The whole P vector is dominated by unpredictable noise, so we only keep
    the directions of P that X can actually predict (whitened cross-covariance
    / canonical structure).  Returns everything needed to (a) form the fixed
    source component Q_s = ((X - mx)/sx) @ W and (b) project the held-out
    future onto the same directions Pq = ((P - mp)/sp) @ B.

    Z = Tx @ Cxp @ Tp^T,  SVD Z = U S V^T ;   W = inv(Cxx) Cxp B,
    B = Tp^T V[:, :k], where Tx/Tp are inverse Cholesky factors of Cxx/Cpp.
    """
    Xs = (Xc - Xc.mean(axis=0)) / (Xc.std(axis=0) + scale_eps)
    Ps = (Pc - Pc.mean(axis=0)) / (Pc.std(axis=0) + scale_eps)
    n = len(Xc)
    Cxx = Xs.T @ Xs / n + lam * np.eye(Xs.shape[1])
    Cpp = Ps.T @ Ps / n + eps * np.eye(Ps.shape[1])
    Cxp = Xs.T @ Ps / n
    Tx = _chol_lower_inv(Cxx)          # Cxx^{-1} = Tx^T Tx
    Tp = _chol_lower_inv(Cpp)          # Cpp^{-1} = Tp^T Tp
    Z = Tx @ Cxp @ Tp.T
    U, S, Vt = np.linalg.svd(Z)
    B = Tp.T @ Vt.T[:, :k]             # P-side directions (p x k)
    W = Tx.T @ Tx @ Cxp @ B            # d x k  (Q = X_std W)
    return {
        "mx": Xc.mean(axis=0), "sx": Xc.std(axis=0) + scale_eps,
        "mp": Pc.mean(axis=0), "sp": Pc.std(axis=0) + scale_eps,
        "W": W, "B": B, "sv": S[:k],
    }


def predict_source_component(mp, Xa):
    """Q_s = ((Xa-mx)/sx) @ W  -- the fixed source predictive component."""
    return (Xa - mp["mx"]) / mp["sx"] @ mp["W"]


def project_future(mp, Pa):
    """Pq = ((Pa-mp)/sp) @ B  -- held-out future along the source directions."""
    return (Pa - mp["mp"]) / mp["sp"] @ mp["B"]
