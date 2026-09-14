"""Acceptance tests for the B2 joint 4-class predictive-gain probe.

These are the measurement-instrument checks from the reviewed spec (2026-09-14
§9).  They use synthetic data only; no model, no real rows, no GPU.  Each check
is repeated over several synthetic datasets so a single lucky draw cannot pass
it.
"""

import numpy as np

import retention_probe_joint as rj


# ------------------------------------------------------------------ helpers
def _rand_encoder(rng, n_nodes=6, r=4):
    return rng.randn(n_nodes, r)


def _pairs(n, s_ids=(0, 1), p_ids=(2, 3)):
    s = np.tile(np.asarray(s_ids, np.int64), (n, 1))
    p = np.tile(np.asarray(p_ids, np.int64), (n, 1))
    return s, p


def _fit_eval(Phi_f, C_f, Z_f, y_f, Phi_a, C_a, Z_a, y_a, lam=1e-2,
              use_state=True):
    p = rj.fit_joint(Phi_f, C_f, Z_f, y_f, lam, use_state=use_state)
    return p, rj.joint_row_nll(p, Phi_a, C_a, Z_a, y_a)


# ------------------------------------------------------------- 1. uniform ln4
def test_zero_weight_is_uniform_ln4():
    """A zero-weight probe is exactly uniform: NLL == ln 4 per row."""
    rng = np.random.RandomState(0)
    n = 300
    Phi = rng.randn(n, 4, 12)
    C = rng.randn(n, 5)
    Z = rng.randn(n, 6)
    y = rng.randint(0, 4, size=n)
    p = {"w_phi": np.zeros(12), "W_C": np.zeros((12, 5)),
         "W_Z": np.zeros((12, 6)), "sC": 1.0, "sZ": 1.0, "use_state": True}
    nll = rj.joint_row_nll(p, Phi, C, Z, y)
    assert np.allclose(nll, np.log(4.0), atol=1e-9), nll[:5]
    # fitted probes never do worse than uniform on their own fit set
    pf = rj.fit_joint(Phi, C, Z, y, 1e-2, use_state=True)
    assert float(rj.joint_row_nll(pf, Phi, C, Z, y).mean()) <= np.log(4.0) + 1e-6


# --------------------------------------------------------- 2. joint XOR 1 bit
def test_joint_xor_one_bit_and_marginals_zero():
    """Z = one-hot(Y_s xor Y_p): joint probe ~1 bit, each marginal 0 bits."""
    for seed in (0, 1, 2):
        rng = np.random.RandomState(seed)
        n = 2000
        E = _rand_encoder(rng)
        s_pair, p_pair = _pairs(n)
        ys = rng.randint(2, size=n)
        yp = rng.randint(2, size=n)
        b = ys ^ yp
        Z = np.zeros((n, 2))
        Z[np.arange(n), b] = 1.0
        C = rng.randn(n, 5)
        y = rj.joint_class(ys, yp)
        Phi = rj.phi_tensor(E, s_pair, p_pair)

        # in-sample identifiability: a weak ridge must reach the ~1 bit the XOR
        # conditional structure allows (lambda=1e-2 is heavy enough to shrink it)
        p_base = rj.fit_joint(Phi, C, Z, y, 1e-3, use_state=False)
        p_full = rj.fit_joint(Phi, C, Z, y, 1e-3, use_state=True)
        nll0 = rj.joint_row_nll(p_base, Phi, C, Z, y).mean()
        nll1 = rj.joint_row_nll(p_full, Phi, C, Z, y).mean()
        j = (nll0 - nll1) / np.log(2.0)
        assert 0.90 < j < 1.05, (seed, j)

        # each marginal is uninformative about Z: a binary probe on Y_s gains 0
        assert _marginal_gain(Z, ys) < 0.05, (seed, _marginal_gain(Z, ys))


def _marginal_gain(Z, y):
    """Held-out binary NLL drop of Z about a single binary label y."""
    from sklearn.linear_model import LogisticRegression
    n = len(y)
    tr, te = slice(0, n // 2), slice(n // 2, n)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(Z[tr], y[tr])
    def nll(X, yy):
        p = clf.predict_proba(X)[:, 1]
        p = np.clip(p, 1e-12, 1)
        return float(-np.mean(np.where(yy == 1, np.log(p), np.log(1 - p))))
    # uniform reference vs fitted, both on the held-out half
    uni = float(np.mean(-np.log(np.full(len(y[te]), 0.5))))
    return (uni - nll(Z[te], y[te])) / np.log(2.0)


# ------------------------------------------- 3. real state carries gain
def test_state_gain_positive_and_decays_with_noise():
    for seed in (0, 1, 2):
        rng = np.random.RandomState(seed + 10)
        n = 1200
        E = _rand_encoder(rng)
        s_pair, p_pair = _pairs(n)
        Z = rng.randn(n, 6)
        hs = rng.randn(6, 2)
        hp = rng.randn(6, 2)
        ys = np.argmax(Z @ hs, axis=1)
        yp = np.argmax(Z @ hp, axis=1)
        C = rng.randn(n, 5)
        y = rj.joint_class(ys, yp)
        Phi = rj.phi_tensor(E, s_pair, p_pair)
        Zmid = 0.4 * Z + 1.0 * rng.randn(n, 6)

        def J(Zx):
            pb = rj.fit_joint(Phi, C, Zx, y, 1e-2, use_state=False)
            pf = rj.fit_joint(Phi, C, Zx, y, 1e-2, use_state=True)
            return (rj.joint_row_nll(pb, Phi, C, Zx, y).mean()
                    - rj.joint_row_nll(pf, Phi, C, Zx, y).mean()) / np.log(2.0)

        assert J(Z) > 0.4, (seed, J(Z))
        assert J(Zmid) < J(Z), (seed, J(Z), J(Zmid))


# ------------------------------------------------- 4. candidate-swap invariance
def test_candidate_swap_invariance():
    for seed in (0, 1, 2):
        rng = np.random.RandomState(seed + 20)
        n = 800
        E = _rand_encoder(rng)
        s_pair, p_pair = _pairs(n)
        Z = rng.randn(n, 6)
        C = rng.randn(n, 5)
        ys = (Z[:, 0] > 0).astype(int)
        yp = (Z[:, 1] > 0).astype(int)
        y = rj.joint_class(ys, yp)
        Phi = rj.phi_tensor(E, s_pair, p_pair)

        # swap both candidate orders and relabel with the same permutation
        p_base = rj.fit_joint(Phi, C, Z, y, 1e-2, use_state=False)
        n0 = rj.joint_row_nll(p_base, Phi, C, Z, y).mean()

        s_sw = s_pair[:, ::-1]
        p_sw = p_pair[:, ::-1]
        Phi_sw = rj.phi_tensor(E, s_sw, p_sw)
        y_sw = rj.joint_class(1 - ys, 1 - yp)
        p_sw_base = rj.fit_joint(Phi_sw, C, Z, y_sw, 1e-2, use_state=False)
        n1 = rj.joint_row_nll(p_sw_base, Phi_sw, C, Z, y_sw).mean()
        assert abs(n0 - n1) < 1e-6, (seed, n0, n1)


# ------------------------------------------------------ 5. optional ignorance
def test_optional_ignorance_reduces_to_base():
    rng = np.random.RandomState(30)
    n = 600
    E = _rand_encoder(rng)
    s_pair, p_pair = _pairs(n)
    Z = rng.randn(n, 6)
    C = rng.randn(n, 5)
    y = rng.randint(0, 4, size=n)
    Phi = rj.phi_tensor(E, s_pair, p_pair)
    base = rj.fit_joint(Phi, C, Z, y, 1e-2, use_state=False)
    full = rj.base_as_full(base, d_z=Z.shape[1])
    assert np.allclose(rj.joint_row_nll(base, Phi, C, Z, y),
                       rj.joint_row_nll(full, Phi, C, Z, y), atol=1e-9)


def test_zero_state_gives_zero_gain():
    """All-zero state == base; the estimator has no permutation term to lift it."""
    for seed in (0, 1):
        rng = np.random.RandomState(seed + 40)
        n = 1000
        E = _rand_encoder(rng)
        s_pair, p_pair = _pairs(n)
        Z = np.zeros((n, 6))
        C = rng.randn(n, 5)
        y = rng.randint(0, 4, size=n)
        Phi = rj.phi_tensor(E, s_pair, p_pair)
        pb = rj.fit_joint(Phi, C, Z, y, 1e-2, use_state=False)
        pf = rj.fit_joint(Phi, C, Z, y, 1e-2, use_state=True)
        j = (rj.joint_row_nll(pb, Phi, C, Z, y).mean()
             - rj.joint_row_nll(pf, Phi, C, Z, y).mean()) / np.log(2.0)
        assert abs(j) < 1e-6, (seed, j)


# --------------------------------------------------- 6. determinism / A-A
def test_aa_identical():
    rng = np.random.RandomState(50)
    n = 500
    E = _rand_encoder(rng)
    s_pair, p_pair = _pairs(n)
    Z = rng.randn(n, 6)
    C = rng.randn(n, 5)
    y = rng.randint(0, 4, size=n)
    Phi = rj.phi_tensor(E, s_pair, p_pair)
    a = rj.fit_joint(Phi, C, Z, y, 1e-2, use_state=True)
    b = rj.fit_joint(Phi, C, Z, y, 1e-2, use_state=True)
    assert rj.params_hash(a) == rj.params_hash(b)
    assert np.allclose(rj.joint_row_nll(a, Phi, C, Z, y),
                       rj.joint_row_nll(b, Phi, C, Z, y))


# ------------------------------------------------------ 7. no audit leakage
def test_audit_labels_do_not_change_the_fit():
    rng = np.random.RandomState(60)
    n = 800
    E = _rand_encoder(rng)
    s_pair, p_pair = _pairs(n)
    Z = rng.randn(n, 6)
    C = rng.randn(n, 5)
    y = rng.randint(0, 4, size=n)
    Phi = rj.phi_tensor(E, s_pair, p_pair)
    fit, aud = slice(0, 400), slice(400, 800)
    p1 = rj.fit_joint(Phi[fit], C[fit], Z[fit], y[fit], 1e-2, use_state=True)
    # scramble ONLY the audit labels; the fit and lambda selection never see them
    y2 = y.copy()
    y2[aud] = rng.randint(0, 4, size=len(y[aud]))
    p2 = rj.fit_joint(Phi[fit], C[fit], Z[fit], y2[fit], 1e-2, use_state=True)
    assert rj.params_hash(p1) == rj.params_hash(p2)
    # a lambda selected on an internal split of the fit block is unchanged too
    folds = [(np.arange(0, 250), np.arange(250, 400))]
    s1, _ = rj.select_joint(Phi[fit], C[fit], Z[fit], y[fit], folds,
                            use_state=True)
    s2, _ = rj.select_joint(Phi[fit], C[fit], Z[fit], y2[fit], folds,
                            use_state=True)
    assert abs(s1["lam"] - s2["lam"]) < 1e-15


# ------------------------------------------------ 8. scale / rotation invariance
def test_state_scale_and_rotation_invariance():
    for seed in (0, 1):
        rng = np.random.RandomState(seed + 70)
        n = 1000
        E = _rand_encoder(rng)
        s_pair, p_pair = _pairs(n)
        Z = rng.randn(n, 6)
        hs, hp = rng.randn(6, 2), rng.randn(6, 2)
        ys = np.argmax(Z @ hs, axis=1)
        yp = np.argmax(Z @ hp, axis=1)
        C = rng.randn(n, 5)
        y = rj.joint_class(ys, yp)
        Phi = rj.phi_tensor(E, s_pair, p_pair)

        def J(Zx):
            pb = rj.fit_joint(Phi, C, Zx, y, 1e-2, use_state=False)
            pf = rj.fit_joint(Phi, C, Zx, y, 1e-2, use_state=True)
            return (rj.joint_row_nll(pb, Phi, C, Zx, y).mean()
                    - rj.joint_row_nll(pf, Phi, C, Zx, y).mean()) / np.log(2.0)

        Q, _ = np.linalg.qr(rng.randn(6, 6))
        j0 = J(Z)
        js = J(3.7 * Z)
        jr = J(Z @ Q)
        assert abs(j0 - js) < 1e-3, (seed, j0, js)
        assert abs(j0 - jr) < 1e-3, (seed, j0, jr)


# ------------------------------------------------- 9. pair_id grouping is safe
def test_pair_key_is_not_flattened_by_numpy():
    pids = [(11, 3, 5, 7, 9), (11, 3, 5, 7, 8), (12, 4, 6, 8, 10)]
    flat = np.asarray(pids, dtype=object)
    # the buggy form: np.unique flattens the three 5-tuples into 10 scalars, so
    # the three distinct pairs would collapse into a bogus group count
    assert np.unique(flat).size == 10, np.unique(flat).size
    keys = [rj.pair_key(p) for p in pids]
    assert len(set(keys)) == 3
    assert np.unique(np.asarray(keys)).size == 3


# ------------------------------------------------- 10. bootstrap / ratio guard
def test_block_bootstrap_shares_indices_and_guards_ratio():
    rng = np.random.RandomState(80)
    n = 900
    blocks = np.repeat(np.arange(30), 30)
    nll0 = rng.randn(n)
    nll1 = {3: nll0 - 0.30, 1: nll0 - 0.05}
    idx = rj.bootstrap_indices(blocks, n_boot=200, seed=1)
    # the same index sets apply to every arm/metric by construction
    J = rj.paired_bootstrap(idx, nll0, nll1)
    assert J[3].mean() > J[1].mean()
    R, lo, hi, ok = rj.ratio_ci(J[3], J[1])
    assert ok and lo < R < hi

    # a source whose denominator straddles zero is not identifiable
    Jsrc = np.concatenate([np.full(100, -0.01), np.full(100, 0.01)])
    _, _, _, ok2 = rj.ratio_ci(Jsrc, np.abs(Jsrc) + 0.1)
    assert not ok2


def test_time_blocks_are_contiguous():
    t = np.arange(100, dtype=float)
    b = rj.time_blocks(t, 10)
    assert b.min() == 0 and b.max() == 9
    assert np.all(np.diff(b) >= 0)
