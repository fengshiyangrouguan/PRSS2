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


def retention_ci(arr):
    return (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))
