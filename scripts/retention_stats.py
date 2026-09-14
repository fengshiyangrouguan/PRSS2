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


# ============================================================ NLL-based MI
# Conditional future-predictive information estimated by the HELD-OUT drop in
# NLL of a multinomial (logistic) probe for the joint future label
# S_s = (Y_s, Y_pa(s)), label = 2*Y_s + Y_pa(s) in {0,1,2,3}.
#
#   I_hat_{s,k} = ( NLL_base - NLL_full ) / ln 2      [bits / sample]
#   NLL_base    = NLL( S_s | C_s, Z^-_{s->k} )
#   NLL_full    = NLL( S_s | C_s, Z^-_{s->k}, Delta_{s->k} )
#
# class_weight=None (so the estimate tracks the natural-distribution MI);
# the probe family and hyper-parameters are identical for TGN and ours.


def joint_target(y_s, y_parent):
    """2 * Y_s + Y_pa(s) in {0,1,2,3} from two binary future labels."""
    y_s = np.asarray(y_s, dtype=np.int64)
    yp = np.asarray(y_parent, dtype=np.int64)
    return 2 * y_s + yp


def _std_fit(Fc):
    m = Fc.mean(axis=0)
    s = Fc.std(axis=0) + 1e-9
    return m, s


def multinomial_nll(Fc, yc, Fa, ya, lam=1e-2, n_class=4, max_iter=3000):
    """Held-out NLL of a multinomial logistic probe (natural log).

    Fit on (Fc,yc) with L2 strength C=1/lam, evaluate NLL on (Fa,ya).
    Features are standardized on the calibration set only.
    """
    from sklearn.linear_model import LogisticRegression
    Fc = np.asarray(Fc, dtype=np.float64)
    Fa = np.asarray(Fa, dtype=np.float64)
    yc = np.asarray(yc, dtype=np.int64)
    ya = np.asarray(ya, dtype=np.int64)
    if len(np.unique(yc)) < 2 or len(ya) == 0:
        return float("nan")
    m, s = _std_fit(Fc)
    clf = LogisticRegression(C=1.0 / max(lam, 1e-9), solver="lbfgs",
                             max_iter=max_iter, class_weight=None)
    clf.fit((Fc - m) / s, yc)
    P = clf.predict_proba((Fa - m) / s)
    cls = clf.classes_.astype(np.int64)
    # map target labels -> column index (missing classes contribute nothing)
    col = {int(c): j for j, c in enumerate(cls)}
    hit = np.array([col.get(int(y), -1) for y in ya])
    ok = hit >= 0
    if ok.sum() == 0:
        return float("nan")
    p = P[np.arange(len(ya))[ok], hit[ok]]
    return float(-np.mean(np.log(np.clip(p, 1e-12, 1.0))))


def conditional_info_bits(Fc_base, Fc_full, yc, Fa_base, Fa_full, ya,
                          lam=1e-2):
    """I_hat = (NLL_base - NLL_full)/ln2  [bits/sample]; can be negative."""
    nll_b = multinomial_nll(Fc_base, yc, Fa_base, ya, lam=lam)
    nll_f = multinomial_nll(Fc_full, yc, Fa_full, ya, lam=lam)
    return (nll_b - nll_f) / np.log(2.0)


def nll_bootstrap_ci(Fc_base, Fc_full, yc, Fa_base, Fa_full, ya,
                     n_boot=200, seed=0, lam=1e-2, alpha=0.05):
    """Cluster-bootstrap CI for I_hat (rows resampled with replacement)."""
    rng = np.random.RandomState(seed)
    n = len(ya)
    out = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        out.append(conditional_info_bits(
            Fc_base, Fc_full, yc, Fa_base[idx], Fa_full[idx], ya[idx],
            lam=lam))
    out = np.asarray(out, dtype=np.float64)
    out = out[np.isfinite(out)]
    if out.size == 0:
        return (float("nan"), float("nan"))
    return (float(np.percentile(out, 100 * alpha / 2)),
            float(np.percentile(out, 100 * (1 - alpha / 2))))


# --------------------------------------------------- interaction probe (v3)
def fixed_proj(d_in, d_out, seed):
    """Frozen Gaussian projection [d_in] -> [d_out] (candidate interaction)."""
    g = np.random.RandomState(int(seed))
    return g.normal(0.0, 1.0 / np.sqrt(max(1, d_in)), size=(d_out, d_in))


def interaction_feats(ctx, S, cand_s, cand_p, P):
    """psi = [C, S, e(c_s), e(c_p), S*(P e(c_s)), S*(P e(c_p))].

    The S x candidate interaction terms are the only channel that can express
    "does this state match this candidate"; S is either Z^- (base) or Z^+ =
    Z^- + Delta (full) -- SAME dimension and SAME architecture for both, so
    the full probe gains no extra capacity from the delta.
    """
    cs = cand_s @ P.T
    cp = cand_p @ P.T
    return np.concatenate([ctx, S, cand_s, cand_p, S * cs, S * cp], axis=1)


def info_bits_interaction(ctx_c, S_rem_c, S_keep_c, cs_c, cp_c, yc,
                          ctx_a, S_rem_a, S_keep_a, cs_a, cp_a, ya,
                          P, lam=1e-2):
    """I = (NLL(base) - NLL(full))/ln2 with the interaction scorer; base uses
    Z^-, full uses Z^+ (same dim)."""
    base_c = interaction_feats(ctx_c, S_rem_c, cs_c, cp_c, P)
    full_c = interaction_feats(ctx_c, S_keep_c, cs_c, cp_c, P)
    base_a = interaction_feats(ctx_a, S_rem_a, cs_a, cp_a, P)
    full_a = interaction_feats(ctx_a, S_keep_a, cs_a, cp_a, P)
    return conditional_info_bits(base_c, full_c, yc, base_a, full_a, ya,
                                 lam=lam)


def cluster_bootstrap_ci(ctx_c, S_rem_c, S_keep_c, cs_c, cp_c, yc,
                         ctx_a, S_rem_a, S_keep_a, cs_a, cp_a, ya, groups_a,
                         P, n_boot=200, seed=0, lam=1e-2, alpha=0.05):
    """Paired cluster bootstrap over ``groups_a`` (e.g. pair_id) with BOTH the
    numerator and denominator computed inside each replicate."""
    groups = np.asarray(groups_a)
    uniq = np.unique(groups)
    by = {g: np.where(groups == g)[0] for g in uniq}
    rng = np.random.RandomState(seed)
    out = []
    for _ in range(n_boot):
        pick = uniq[rng.randint(0, len(uniq), size=len(uniq))]
        idx = np.concatenate([by[g] for g in pick])
        out.append(info_bits_interaction(
            ctx_c, S_rem_c, S_keep_c, cs_c, cp_c, yc,
            ctx_a[idx], S_rem_a[idx], S_keep_a[idx], cs_a[idx], cp_a[idx],
            ya[idx], P, lam=lam))
    out = np.asarray(out, dtype=np.float64)
    out = out[np.isfinite(out)]
    if out.size == 0:
        return (float("nan"), float("nan"))
    return (float(np.percentile(out, 100 * alpha / 2)),
            float(np.percentile(out, 100 * (1 - alpha / 2))))




# ==========================================================================
# v3 nested source-contribution probe (fair, arm-agnostic).
#
# Two problems are measured SEPARATELY:
#   (a) mechanism retention -- how much of the SOURCE's predictive
#       contribution about the joint local future (Y_s, Y_p) survives to
#       each downstream position k;
#   (b) root-task gain -- see audit_retention_v2 (native frozen task head).
#
# Fairness contract for (a):
#   * identical C (context) and identical candidate pairs Q across arms;
#   * both the POSITIVE and the NEGATIVE candidate of a node are fed in, and
#     the label says WHICH one is the true future destination (a random swap);
#   * one COMMON base b_theta(C, Q) is fit once; each position k only adds a
#     rank-limited increment g_phi_k(Delta_{s->k}, Q) on top of the FROZEN
#     base, so full always contains the g=0 base (never two unrelated probes);
#   * E(candidate) and the Delta projections are frozen (arm-shared) maps.
#
#   J_{s->k} = (NLL_base - NLL_full,k) / ln2   [bits/tree]
#   R_{s->k} = J_{s->k} / J_{s->s}
# The source position's own delta is Delta_{s->s} = U_s (the leaf interface
# state); report J_{s->s}, J_{s->k} and R_{s->k} together.
# ==========================================================================

def _std_cols(F):
    F = np.asarray(F, dtype=np.float64)
    return F.mean(axis=0), F.std(axis=0) + 1e-9


def _logit_fit(F, y, lam):
    from sklearn.linear_model import LogisticRegression
    m, s = _std_cols(F)
    clf = LogisticRegression(C=1.0 / max(float(lam), 1e-9), solver="lbfgs",
                             max_iter=2000)
    clf.fit((np.asarray(F, dtype=np.float64) - m) / s,
            np.asarray(y, dtype=np.int64))
    return m, s, clf


def _logit_score(m, s, clf, F):
    return clf.decision_function(
        (np.asarray(F, dtype=np.float64) - m) / s).astype(np.float64)


def _nll_logit(z, y):
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    return float(np.mean(np.logaddexp(0.0, z) - y * z))


def _offset_fit(base_c, Zc, yc, lam):
    """Fit increment weights w with the base logit FROZEN as an offset."""
    from scipy.optimize import minimize
    base_c = np.asarray(base_c, dtype=np.float64)
    Zc = np.asarray(Zc, dtype=np.float64)
    yc = np.asarray(yc, dtype=np.float64)
    l2 = float(lam)

    def obj(w):
        z = base_c + Zc @ w
        return float(np.mean(np.logaddexp(0.0, z) - yc * z) + l2 * float(w @ w))

    r = minimize(obj, np.zeros(Zc.shape[1]), method="L-BFGS-B")
    return r.x


def _offset_nll(base, Z, y, w):
    z = np.asarray(base, dtype=np.float64) + np.asarray(Z, np.float64) @ w
    return _nll_logit(z, np.asarray(y, dtype=np.float64))


def _proj(d_in, d_out, seed):
    g = np.random.RandomState(int(seed))
    return g.normal(0.0, 1.0 / np.sqrt(max(1, int(d_in))),
                    size=(int(d_out), int(d_in)))


def increment_feats(delta, q, Gd, Gq):
    """Rank-limited bilinear candidate match: (Gd @ delta) * (Gq @ q).

    With a linear head this realizes g = <A delta, q> for a rank<=r matrix A.
    """
    return (np.asarray(delta, np.float64) @ Gd.T) * (
        np.asarray(q, np.float64) @ Gq.T)


def _base_two(C, qs, qp, ys, yp, lam):
    F = np.concatenate([np.asarray(C, np.float64),
                        np.asarray(qs, np.float64),
                        np.asarray(qp, np.float64)], axis=1)
    ms, ss, clf_s = _logit_fit(F, ys, lam)
    mp, sp, clf_p = _logit_fit(F, yp, lam)
    return (F, (ms, ss, clf_s), (mp, sp, clf_p))


def _base_nll_and_logit(F, model_s, model_p, ys, yp):
    zs = _logit_score(*model_s, F)
    zp = _logit_score(*model_p, F)
    return _nll_logit(zs, ys) + _nll_logit(zp, yp), zs, zp


def nested_info_bits(Cc, qsc, qpc, Dc, ysc, ypc,
                     Ca, qsa, qpa, Da, ysa, ypa,
                     ranks=(4, 8, 16), lam=1e-2, seed=0):
    """Audit-set conditional info J of the source contribution at ONE position.

    ``Dc``/``Da`` are the source contribution Delta_{s->k} on calib/audit
    (for the source position itself pass U_s).  Returns J (bits/tree), the
    calib-selected rank, and the base/full audit NLLs.
    """
    Fc, model_s, model_p = _base_two(Cc, qsc, qpc, ysc, ypc, lam)
    Fa, _, _ = _base_two(Ca, qsa, qpa, ysa, ypa, lam)   # same shapes only
    nll_base_a, bas, bap = _base_nll_and_logit(Fa, model_s, model_p, ysa, ypa)
    nll_base_c, bcs, bcp = _base_nll_and_logit(Fc, model_s, model_p, ysc, ypc)
    best = None
    for r in ranks:
        Gd = _proj(np.asarray(Dc).shape[1], int(r), seed + 101)
        Gq = _proj(np.asarray(qsc).shape[1], int(r), seed + 202)
        Zc_s = increment_feats(Dc, qsc, Gd, Gq)
        Zc_p = increment_feats(Dc, qpc, Gd, Gq)
        Za_s = increment_feats(Da, qsa, Gd, Gq)
        Za_p = increment_feats(Da, qpa, Gd, Gq)
        ws = _offset_fit(bcs, Zc_s, ysc, lam)
        wp = _offset_fit(bcp, Zc_p, ypc, lam)
        calib = _offset_nll(bcs, Zc_s, ysc, ws) + _offset_nll(bcp, Zc_p, ypc, wp)
        if best is None or calib < best[0]:
            best = (calib, int(r), ws, wp, Za_s, Za_p)
    _, rank, ws, wp, Za_s, Za_p = best
    nll_full_a = _offset_nll(bas, Za_s, ysa, ws) + _offset_nll(bap, Za_p, ypa, wp)
    return {"J": float((nll_base_a - nll_full_a) / np.log(2.0)),
            "rank": rank, "nll_base": float(nll_base_a),
            "nll_full": float(nll_full_a)}


def nested_curve(Cc, qsc, qpc, ysc, ypc, Ca, qsa, qpa, ysa, ypa,
                 Dc_bypos, Da_bypos, origin_pos, ranks=(4, 8, 16),
                 lam=1e-2, seed=0):
    """J at every position + R relative to ``origin_pos`` (the source)."""
    out = {}
    j0 = None
    for pos in sorted(Da_bypos):
        res = nested_info_bits(
            Cc, qsc, qpc, Dc_bypos[pos], ysc, ypc,
            Ca, qsa, qpa, Da_bypos[pos], ysa, ypa,
            ranks=ranks, lam=lam, seed=seed)
        out[int(pos)] = res
        if int(pos) == int(origin_pos):
            j0 = res["J"]
    for pos, res in out.items():
        res["R"] = (float(res["J"]) / j0) if (j0 and abs(j0) > 1e-12) \
            else float("nan")
    return {"origin_pos": int(origin_pos), "J_source": j0, "points": out}


def nested_null_source(Cc, qsc, qpc, ysc, ypc, Ca, qsa, qpa, ysa, ypa,
                       Dc_s, Da_s, strata_a, ranks=(4, 8, 16), lam=1e-2,
                       seed=0, n_null=200):
    """Permutation null for J_{s->s}: shuffle the audit delta within strata."""
    rng = np.random.RandomState(int(seed))
    strata_a = np.asarray(strata_a)
    uniq = np.unique(strata_a)
    by = {g: np.where(strata_a == g)[0] for g in uniq}
    null = []
    for t in range(int(n_null)):
        idx = np.arange(len(strata_a))
        for g in uniq:
            rows = by[g].copy()
            rng.shuffle(rows)
            idx[by[g]] = rows
        per = nested_info_bits(
            Cc, qsc, qpc, Dc_s, ysc, ypc,
            Ca, qsa, qpa, np.asarray(Da_s)[idx],
            ysa, ypa, ranks=ranks, lam=lam, seed=seed + 7 * (t + 1))
        null.append(per["J"])
    return np.asarray(null, dtype=np.float64)


def cluster_join_ids(*key_arrays):
    """Intersection helper: rows share a key iff every array's element equals.

    Returns (keep_masks, common_keys) for the FIRST array against the rest,
    so cross-arm statistics can be computed on the shared (pair_id,line,phys)
    set only.
    """
    keys = [tuple(str(k) for k in a) for a in key_arrays]
    common = set(keys[0])
    for k in keys[1:]:
        common &= set(k)
    masks = [np.asarray([kk in common for kk in k], dtype=bool) for k in keys]
    return masks, sorted(common)
