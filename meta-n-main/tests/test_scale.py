"""Scale-normalization helper tests (P0.3, roadmap v2 N4a).

Locks in the deterministic ``Archive.score_range`` and the per-adapter
``score_scale`` descriptor that the selection / gate / stopping steps key on.
The helper is additive here — not yet wired into runtime behavior.
"""

import math

from meta_n.core.archive import Archive
from meta_n.integrations.benchmark import BenchmarkAdapter
from meta_n.integrations.openevolve import OpenEvolveBaseAdapter


class _DummyAdapter(BenchmarkAdapter):
    @property
    def name(self):
        return "dummy"

    def load_tasks(self, limit=None):
        return []

    async def evaluate(self, task, solution):
        raise NotImplementedError


def _archive(means):
    from meta_n.core.archive import Candidate
    a = Archive()
    for i, m in enumerate(means):
        a.add(Candidate(candidate_id=f"c{i}", mean_score=m, iteration=i))
    return a


def test_score_range_unit_scale():
    a = _archive([0.2, 0.5, 0.8])
    assert abs(a.score_range() - 0.6) < 1e-9


def test_score_range_continuous_scale(make_continuous_archive):
    a = make_continuous_archive()  # means 38, 40, 29
    assert abs(a.score_range() - 11.0) < 1e-9


def test_score_range_single_candidate_is_floored():
    a = _archive([0.5])
    assert a.score_range() == 1e-9  # safe denominator, no divide-by-zero


def test_score_range_is_deterministic():
    a = _archive([0.8, 0.2, 0.5, 0.5])
    assert a.score_range() == a.score_range()


def test_score_range_ignores_non_finite():
    a = _archive([0.2, 0.8, float("nan"), float("inf")])
    assert abs(a.score_range() - 0.6) < 1e-9


def test_freeze_score_range_stops_drift():
    a = _archive([0.2, 0.8])
    frozen = a.freeze_score_range()
    assert abs(frozen - 0.6) < 1e-9
    from meta_n.core.archive import Candidate
    a.add(Candidate(candidate_id="big", mean_score=100.0, iteration=9))
    assert a.score_range() == frozen  # frozen value wins despite the new outlier


def test_default_score_scale_is_unit():
    assert _DummyAdapter().score_scale() == {
        "kind": "unit", "lo": 0.0, "hi": 1.0, "failure_sentinel": None,
    }


def test_openevolve_score_scale_is_continuous_with_sentinel():
    # score_scale is a stateless descriptor; call on the class.
    s = OpenEvolveBaseAdapter.score_scale(None)
    assert s["kind"] == "continuous"
    assert s["failure_sentinel"] == -1e9
