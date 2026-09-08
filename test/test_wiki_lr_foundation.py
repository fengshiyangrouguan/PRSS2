"""Foundation-review gates for Wiki-LR-Binary (spec §8 + §1.1, non-TGB).

Covers what is unit-testable without a GPU / real TGB dataset:
  * AP/AUC definitions match a scalar reference AND behave on all-ties inputs
    (AP=positive rate, AUC=0.5, always within [0,1]) — §1.1 #1/#2;
  * HistRandTrainSampler determinism, prefix dedup, hist/random disjointness,
    same-(src,t) exclusion — §1.1 #5;
  * stable negative digest determinism — §1.1 #4 foundation.
"""

import numpy as np
import pytest

from rpbe.training.histrand_sampler import HistRandTrainSampler
from rpbe.training.wiki_binary_eval import _ap, _auc, global_metrics
from rpbe.data.wiki_binary_negatives import _pick, _stable_digest


def _ref_ap(p, n):
    """Tie-group reference: equal scores processed as one block."""
    p = np.asarray(p, float); n = np.asarray(n, float)
    s = np.concatenate([p, n])
    y = np.concatenate([np.ones(len(p)), np.zeros(len(n))])
    order = np.argsort(-s, kind="stable")
    ss = s[order]; yy = y[order]
    cuts = np.flatnonzero(np.diff(ss) != 0) + 1
    starts = np.concatenate([[0], cuts]); ends = np.concatenate([cuts, [len(ss)]])
    cum_p = cum_t = 0.0
    acc = 0.0
    for a, b in zip(starts, ends):
        gp = yy[a:b].sum(); cum_p += gp; cum_t += b - a
        acc += gp * (cum_p / cum_t)
    return acc / len(p)


def _ref_auc(p, n):
    w = t = 0.0
    for a in p:
        for b in n:
            w += a > b
            t += a == b
    return (w + 0.5 * t) / (len(p) * len(n))


def test_ap_auc_match_scalar_reference():
    rng = np.random.RandomState(0)
    pos = rng.rand(150)
    neg = rng.rand(150)
    assert abs(_ap(pos, neg) - _ref_ap(pos, neg)) < 1e-12
    assert abs(_auc(pos, neg) - _ref_auc(pos, neg)) < 1e-12


def test_ap_all_ties_is_positive_rate():
    pos = np.full(100, 0.5)
    neg = np.full(100, 0.5)
    assert abs(_ap(pos, neg) - 0.5) < 1e-12
    # imbalanced all-ties -> positive rate p/(p+n)
    assert abs(_ap(np.full(30, 0.5), np.full(70, 0.5)) - 0.3) < 1e-12


def test_auc_all_ties_is_half_and_bounded():
    assert abs(_auc(np.full(100, 0.5), np.full(100, 0.5)) - 0.5) < 1e-12
    for k in (10, 100):
        p = np.full(k, 1.0); n = np.full(k, 1.0)
        assert 0.0 <= _auc(p, n) <= 1.0
        assert 0.0 <= _auc(np.full(k, 0.0), np.full(k, 0.0)) <= 1.0


def test_global_metrics_hist_subset():
    rng = np.random.RandomState(1)
    pos = rng.rand(100)
    neg = rng.rand(100)
    neg[:40] += 0.4
    hm = np.array([True] * 40 + [False] * 60)
    res = global_metrics(pos, neg, hm)
    assert res["n_hist"] == 40 and res["n_random"] == 60
    assert abs(res["ap_hist"] - _ref_ap(pos[:40], neg[:40])) < 1e-12
    assert abs(res["auc_hist"] - _ref_auc(pos[:40], neg[:40])) < 1e-12
    assert res["n_pos"] == 100


def test_histrand_deterministic_and_prefix_unique():
    s = HistRandTrainSampler(
        np.array([1, 1, 1, 1, 2]), np.array([10, 20, 30, 40, 50]),
        np.array([0.0, 1.0, 2.0, 3.0, 4.0]), model_seed=7)
    a = s.sample(0, 0, [1], [40], [3.0], [3])
    b = s.sample(0, 0, [1], [40], [3.0], [3])
    assert int(a[0]) == int(b[0])
    assert s._hist_dsts(1, 3, set()) == [10, 20, 30]
    assert s._hist_dsts(1, 1, set()) == [10]
    # dedup: same dst repeated is listed once
    s2 = HistRandTrainSampler(
        np.array([1, 1, 1]), np.array([7, 7, 8]),
        np.array([0.0, 1.0, 2.0]))
    assert s2._hist_dsts(1, 3, set()) == [7, 8]


def test_histrand_random_excludes_prefix_seen():
    # hist_p=0 forces the random branch; src 1 has seen {10,20,30} before row 3
    s = HistRandTrainSampler(
        np.array([1, 1, 1, 1, 9]),
        np.array([10, 20, 30, 40, 200]),
        np.array([0.0, 1.0, 2.0, 3.0, 0.0]), model_seed=1, hist_p=0.0)
    for _ in range(60):
        n = int(s.sample(0, 0, [1], [40], [3.0], [3])[0])
        assert n != 40
        assert n not in (10, 20, 30)   # never a seen dst on the random branch
        assert n in (200,)


def test_histrand_same_st_exclusion():
    src = np.array([1, 1, 1, 1, 9, 9, 9, 9])
    dst = np.array([10, 20, 30, 40, 100, 200, 300, 400])
    t = np.array([1.0, 1.0, 2.0, 3.0, 1.0, 1.0, 1.0, 1.0])
    s = HistRandTrainSampler(src, dst, t, model_seed=3)
    for _ in range(50):
        n = int(s.sample(0, 0, [1], [10], [1.0], [1])[0])
        assert n not in (10, 20)
        n = int(s.sample(0, 0, [1], [20], [1.0], [1])[0])
        assert n not in (10, 20)


def test_stable_digest_deterministic_inrange():
    a = _stable_digest(20260908, "val", 7, 3, 5, 1.25)
    b = _stable_digest(20260908, "val", 7, 3, 5, 1.25)
    c = _stable_digest(20260908, "val", 7, 3, 5, 1.2500000001)
    assert a == b
    assert a != c
    negs = np.arange(20)
    assert _pick(negs, a) in negs.tolist()
