"""Unit tests for the v3 nested source-contribution probe (retention_stats).

Synthetic construction mirrors the real semantics: the label is "which of the
two candidates is the true future destination", and the answer is determined by
the MATCH between the model state (Delta) and the candidate geometry (q), not
by the candidate geometry alone.  Verifies: positive source info, decay with a
degraded downstream delta, R<1, the permutation null collapsing, and the
cluster-join helper.
"""

import numpy as np
import pytest

import retention_stats as rs


def _build(n=800, seed=0, degrade=0.35, noise=1.1):
    rng = np.random.RandomState(seed)
    C = rng.randn(n, 6)
    q_s = rng.randn(n, 4)
    q_p = rng.randn(n, 4)
    D = rng.randn(n, 4)
    ys = ((D * q_s).sum(1) > 0).astype(int)
    yp = ((D * q_p).sum(1) > 0).astype(int)
    D_mid = degrade * D + noise * rng.randn(n, 4)
    return C, q_s, q_p, D, ys, yp, D_mid


def _split(x, tr=slice(0, 400), au=slice(400, 800)):
    return x[tr], x[au]


def test_source_info_positive_and_decays():
    C, qs, qp, D, ys, yp, Dmid = _build()
    Cs, Ca = _split(C)
    qss, qsa = _split(qs)
    qps, qpa = _split(qp)
    Ds, Da = _split(D)
    Dms, Dma = _split(Dmid)
    yss, ysa = _split(ys)
    yps, ypa = _split(yp)
    src = rs.nested_info_bits(Cs, qss, qps, Ds, yss, yps,
                              Ca, qsa, qpa, Da, ysa, ypa, seed=1)
    mid = rs.nested_info_bits(Cs, qss, qps, Dms, yss, yps,
                              Ca, qsa, qpa, Dma, ysa, ypa, seed=1)
    assert src["J"] > 0.1, src
    assert src["J"] > mid["J"], (src, mid)
    assert mid["J"] / src["J"] < 1.0


def test_permutation_null_collapses_and_source_passes_gate():
    C, qs, qp, D, ys, yp, _ = _build()
    Cs, Ca = _split(C)
    qss, qsa = _split(qs)
    qps, qpa = _split(qp)
    Ds, Da = _split(D)
    yss, ysa = _split(ys)
    yps, ypa = _split(yp)
    src = rs.nested_info_bits(Cs, qss, qps, Ds, yss, yps,
                              Ca, qsa, qpa, Da, ysa, ypa, seed=1)
    null = rs.nested_null_source(Cs, qss, qps, yss, yps,
                                 Ca, qsa, qpa, ysa, ypa,
                                 Ds, Da, (np.arange(len(Ca)) // 50),
                                 seed=3, n_null=25)
    assert np.percentile(null, 95) < 0.2, np.percentile(null, 95)
    assert src["J"] > np.percentile(null, 95)


def test_curve_relative_retention():
    C, qs, qp, D, ys, yp, Dmid = _build()
    Cs, Ca = _split(C)
    qss, qsa = _split(qs)
    qps, qpa = _split(qp)
    Ds, Da = _split(D)
    Dms, Dma = _split(Dmid)
    yss, ysa = _split(ys)
    yps, ypa = _split(yp)
    curve = rs.nested_curve(
        Cs, qss, qps, yss, yps, Ca, qsa, qpa, ysa, ypa,
        {3: Ds, 2: Dms, 1: Dms}, {3: Da, 2: Dma, 1: Dma},
        origin_pos=3, seed=1)
    assert curve["J_source"] > 0.1
    R = {k: v["R"] for k, v in curve["points"].items()}
    assert R[3] == pytest.approx(1.0, abs=1e-6)
    assert R[2] < 1.0 and R[1] < 1.0


def test_cluster_join_ids():
    masks, common = rs.cluster_join_ids(["a", "b", "c"], ["c", "b", "d"])
    assert masks[0].tolist() == [False, True, True]
    assert masks[1].tolist() == [True, True, False]
    assert common == ["b", "c"]
