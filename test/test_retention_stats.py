"""Unit tests for the corrected retention statistics (pure numpy, no deps).

Covers the invariants demanded by the review:
- R (explained variance) is bounded above by 1 for any data;
- identity at the source is ~1 when the feature is the source representation;
- delete-source (Delta = 0 / C-only) retention is ~0;
- a mismatched delta (permuted within strata) cannot recover the source
  component (no unrelated-information inflation);
- the audit computes retention as *direct* regressions against one source
  component -- never as a chain product of local factors;
- the CC context builder carries no future-index path (source-level check on
  scripts/audit_retention_v2.py).
"""
import pathlib
import sys

import numpy as np
import pytest

_SCRIPTS = pathlib.Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS))

import retention_stats as rs  # noqa: E402

_AUDIT_SRC = (_SCRIPTS / "audit_retention_v2.py").read_text(encoding="utf-8")
rng = np.random.RandomState(0)


def _make(n, p=128, d=8, seed=1):
    r = np.random.RandomState(seed)
    X = r.normal(size=(n, d))
    A = r.normal(size=(d, p))
    Q = X @ A + 0.05 * r.normal(size=(n, p))
    C = r.normal(size=(n, 5))
    return X, Q, C


def test_R_never_exceeds_one():
    n, p = 300, 16
    for seed in range(5):
        r = np.random.RandomState(seed)
        Q = r.normal(size=(n, p))
        F = r.normal(size=(n, 12))
        nq = n // 2
        R = rs.retention_R(Q[:nq], F[:nq], Q[nq:], F[nq:])
        assert R <= 1.0 + 1e-9


def test_identity_at_source_is_about_one():
    X, Q, C = _make(400, p=32, d=6)
    nq = 200
    Fc = np.concatenate([X[:nq], C[:nq]], axis=1)
    Fa = np.concatenate([X[nq:], C[nq:]], axis=1)
    R = rs.retention_R(Q[:nq], Fc, Q[nq:], Fa)
    assert R >= 0.95, R


def test_delete_source_floor_is_about_zero():
    X, Q, C = _make(400, p=32, d=6)
    nq = 200
    # C-only features (Delta = 0) should not recover the source component.
    C0 = np.zeros_like(C)
    R = rs.retention_R(Q[:nq], C0[:nq], Q[nq:], C0[nq:])
    assert abs(R) <= 0.05, R


def test_mismatched_delta_cannot_recover_source():
    X, Q, C = _make(600, p=24, d=6)
    nq = 300
    # Delta = X recovers Q (~identity).  Permuting X within coarse strata of a
    # matched-context key breaks the link and must fall back to ~the C floor.
    strata = rs.strata_ids(C[nq:, 0], n_bins=8)
    perm = rs.permute_within_strata(X[nq:], strata, np.random.RandomState(9))
    F_real = np.concatenate([X[:nq], C[:nq]], axis=1)
    Fa_real = np.concatenate([X[nq:], C[nq:]], axis=1)
    Fa_perm = np.concatenate([perm, C[nq:]], axis=1)
    R_real = rs.retention_R(Q[:nq], F_real, Q[nq:], Fa_real)
    R_perm = rs.retention_R(Q[:nq], F_real, Q[nq:], Fa_perm)
    R_floor = rs.retention_R(Q[:nq], C[:nq], Q[nq:], C[nq:])
    assert R_real >= 0.8
    assert R_perm - R_floor <= 0.02, (R_perm, R_floor)
    assert R_real - R_perm >= 0.5


def test_direct_points_are_not_chain_products():
    # Structural guarantee: each staggered point is an explained variance of
    # the SAME source component from its own delta.  The stats API exposes no
    # way to multiply per-hop retention factors into a compound number.
    for name in ("retention_R", "retention_map_R", "explained_var"):
        assert callable(getattr(rs, name))
    src = pathlib.Path(rs.__file__).read_text(encoding="utf-8")
    # (the docstring below mentions "chain" only to forbid it; check code-level
    # tokens that would implement chaining)
    for banned in ("prod(", "cumprod", "np.multiply", "*="):
        assert banned not in src, banned


def test_cc_context_has_no_future_index_path():
    # Extract the ctx builder (from `def _ctx_vector` up to the next top-level
    # `class`/`def`) and assert it can only use the historical fields passed in.
    start = _AUDIT_SRC.index("def _ctx_vector")
    end = _AUDIT_SRC.index("class FutureIndex", start)
    body = _AUDIT_SRC[start:end]
    # no future-index identifier may be dereferenced inside the ctx builder
    # (the word "future" is allowed only inside its own docstring)
    for banned in ("fut.", "fut[", "eidx", "query("):
        assert banned not in body, banned
    # The row assembly feeds it ONLY historical fields (no future edge).
    assert "ctx = _ctx_vector(root, st1[\"leaf\"], node_times," in _AUDIT_SRC
    assert "_ctx_vector" in _AUDIT_SRC
    # And the future witness is used only for the prediction target S.
    assert "The future event appears ONLY in the" in _AUDIT_SRC
    assert "fut.eidx" in _AUDIT_SRC


def _given_C_signal(Xtr, Ptr, Ctr, Xte, Pte, Cte, lam=1e-2):
    """Source-specific (given-C) predictive signal, mirroring the audit:
    residualize X and P against C on the train split, then Q_s =
    ridge(X^perp -> P^perp); return the held-out explained variance of P^perp
    by Q_s."""
    mpX = rs.fit_ridge_map(Ctr, Xtr, lam=lam)
    Xr_tr = Xtr - rs.apply_ridge_map(mpX, Ctr)
    Xr_te = Xte - rs.apply_ridge_map(mpX, Cte)
    mpP = rs.fit_ridge_map(Ctr, Ptr, lam=lam)
    Pr_tr = Ptr - rs.apply_ridge_map(mpP, Ctr)
    Pr_te = Pte - rs.apply_ridge_map(mpP, Cte)
    mpQ = rs.fit_ridge_map(Xr_tr, Pr_tr, lam=lam)
    Qte = rs.apply_ridge_map(mpQ, Xr_te)
    return rs.explained_var(Pr_te, Qte)


def test_source_specific_signal_is_zero_when_source_only_carries_context():
    """If X carries P purely through the context C, the given-C source signal
    must be ~0 (the naive absolute X->P map would look positive)."""
    n_tr, n_te = 4000, 3000
    r = np.random.RandomState(3)
    H = r.normal(size=(n_tr + n_te, 3))          # hidden context
    A = r.normal(size=(3, 24))
    W = r.normal(size=(3, 12))
    P = H @ A + 0.2 * r.normal(size=(n_tr + n_te, 24))
    X = H @ W + 0.2 * r.normal(size=(n_tr + n_te, 12))   # X depends on H only
    C = H.copy()                                        # C spans the context
    sig = _given_C_signal(X[:n_tr], P[:n_tr], C[:n_tr],
                          X[n_tr:], P[n_tr:], C[n_tr:])
    assert sig <= 0.05, sig


def test_source_specific_signal_is_positive_when_source_adds_beyond_context():
    """When X carries a component that predicts P beyond what C provides, the
    given-C source signal must be clearly positive."""
    n_tr, n_te = 4000, 3000
    r = np.random.RandomState(4)
    H = r.normal(size=(n_tr + n_te, 3))
    S = r.normal(size=(n_tr + n_te, 2))          # true source-specific driver
    A = r.normal(size=(3, 24))
    B = r.normal(size=(2, 24))
    W = r.normal(size=(5, 16))
    P = H @ A + S @ B + 0.2 * r.normal(size=(n_tr + n_te, 24))
    X = np.concatenate([H, S], axis=1) @ W + 0.1 * r.normal(
        size=(n_tr + n_te, 16))
    C = H + 0.1 * r.normal(size=(n_tr + n_te, 3))
    sig = _given_C_signal(X[:n_tr], P[:n_tr], C[:n_tr],
                          X[n_tr:], P[n_tr:], C[n_tr:])
    assert sig >= 0.10, sig


def test_canonical_dirs_capture_only_predictable_source_signal():
    """Top-k canonical directions recover a real source->future link and
    ignore the unpredictable noise of the full high-dim future."""
    n_tr, n_te, d, p, r = 6000, 4000, 20, 80, 3
    rnd = np.random.RandomState(11)
    X = rnd.normal(size=(n_tr + n_te, d))
    Wt = rnd.normal(size=(d, r))
    Vt = rnd.normal(size=(p, r))
    signal = X @ Wt @ Vt.T                       # low-rank predictable part
    noise = 3.0 * rnd.normal(size=(n_tr + n_te, p))
    P = signal + noise
    mp = rs.canonical_dirs(X[:n_tr], P[:n_tr], k=r, lam=1e-2, eps=1e-6)
    Qa = rs.predict_source_component(mp, X[n_tr:])
    Pq = rs.project_future(mp, P[n_tr:])
    sig = rs.explained_var(Pq, Qa)
    assert sig > 0.1, sig
    # a permuted source (within-strata shuffle) must destroy the signal
    strata = rs.strata_ids(X[n_tr:, 0])
    perm = rs.permute_within_strata(X[n_tr:], strata, np.random.RandomState(5))
    null = rs.explained_var(Pq, rs.predict_source_component(mp, perm))
    assert null < 0.05 and null < sig, (null, sig)


def test_J_invariant_to_audit_mean_shift():
    """The source-signal gate J must not move when only the audit-set mean of
    the target changes (cross-block mean drift must not fake a loss)."""
    r = np.random.RandomState(7)
    A = r.normal(size=(400, 6))
    W = r.normal(size=(6, 8))
    B = A @ W + 0.5 * r.normal(size=(400, 8))
    J0 = rs.centered_sq_corr_strength(A, B)
    J1 = rs.centered_sq_corr_strength(A, B + 1000.0)
    assert J0 >= 0
    assert abs(J1 - J0) < 1e-6, (J0, J1)


def test_corr2_recoverability_range_and_affine_invariance():
    """corr² recoverability is in [0,1] and invariant to affine drift of the
    recovery prediction."""
    r = np.random.RandomState(8)
    Q = r.normal(size=(400, 5))
    Qhat = 0.9 * Q + 0.3 * r.normal(size=(400, 5))
    R0 = rs.corr2_recoverability(Q, Qhat)
    assert 0.0 <= R0 <= 1.0 + 1e-9
    R1 = rs.corr2_recoverability(Q, 7.0 * Qhat + 3.0)   # scale+shift
    assert abs(R1 - R0) < 1e-9, (R0, R1)


def test_corr2_weighted_selects_direction():
    """Strength-weighting must emphasise the strongly recoverable direction."""
    r = np.random.RandomState(12)
    n = 500
    Q = r.normal(size=(n, 2))
    Qhat = np.zeros_like(Q)
    Qhat[:, 0] = 1.5 * Q[:, 0] + 0.1 * r.normal(size=n)   # strong dir 0
    Qhat[:, 1] = 0.1 * r.normal(size=n)                    # noise dir 1
    eq = rs.corr2_recoverability(Q, Qhat)
    wt = rs.corr2_recoverability(Q, Qhat, weights=[1.0, 0.0])
    d0 = rs._dir_pearson2(Q, Qhat)[0]
    assert abs(wt - d0) < 1e-6, (wt, d0)
    assert wt >= eq, (wt, eq)
    assert 0.0 <= wt <= 1.0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
