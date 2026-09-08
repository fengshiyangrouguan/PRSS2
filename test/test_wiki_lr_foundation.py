"""Foundation gates for Wiki-LR-Binary (spec §8, non-TGB subset).

Covers what is unit-testable without a GPU / real TGB dataset:
  * metric definitions match a scalar reference exactly;
  * HistRandTrainSampler is deterministic, prefix-causal, excludes the current
    positive dst (and same-(src,t) positives);
  * stable negative digest / pick are deterministic and in-range.
The evaluator-vs-host and real-data gates (manifest hash, zero collision,
group yield, VRAM, speed) run on the server where the dataset lives.
"""

import numpy as np
import pytest

from rpbe.training.histrand_sampler import HistRandTrainSampler
from rpbe.training.wiki_binary_eval import _ap, _auc, global_metrics
from rpbe.data.wiki_binary_negatives import _pick, _stable_digest


def _ref_ap(p, n):
    s = np.concatenate([p, n])
    y = np.concatenate([np.ones(len(p)), np.zeros(len(n))])
    order = np.lexsort((-y, -s))
    tot = 0.0
    corr = 0.0
    ap = 0.0
    for i in order:
        tot += 1
        if y[i] == 1:
            corr += 1
            ap += corr / tot
    return ap / len(p)


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


def test_histrand_deterministic_and_prefix():
    # 5 events, node 1 seen dsts {10, 20, 30} then 40 at rows 0..3
    src = np.array([1, 1, 1, 1, 2])
    dst = np.array([10, 20, 30, 40, 50])
    t = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    s = HistRandTrainSampler(src, dst, t, model_seed=7)
    # event row 3 (dst 40): hist pool for node1 is {10,20,30} (rows<3)
    a = s.sample(0, 0, [1], [40], [3.0], [3])
    b = s.sample(0, 0, [1], [40], [3.0], [3])
    assert a[0] == b[0]
    assert int(a[0]) in (10, 20, 30)
    assert int(a[0]) != 40
    # event row 0 (dst 10): no history yet -> can only come from universe
    # (forced-hist not testable directly since hist_p=0.5; check exclusion)
    all_neg = [int(s.sample(e, 0, [1], [10], [0.0], [0])[0])
               for e in range(5) for _ in range(10)]
    assert 10 not in all_neg  # current dst always excluded


def test_histrand_same_st_exclusion():
    # node 1 has two positives at the SAME (src,t); universe is padded out by
    # other nodes so exclusion leaves plenty of legal candidates (real data)
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
    # _pick stays within the candidate array
    negs = np.arange(20)
    assert _pick(negs, a) in negs.tolist()
