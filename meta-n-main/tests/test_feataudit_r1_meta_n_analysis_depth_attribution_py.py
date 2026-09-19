"""ROUND-1 feature-audit regressions for meta_n/analysis/depth_attribution.py.

R1-B_depth-3 — the within-task-depth verdict must not gate on a seed-inflated MAX.
Genuine within-task depth requires a task re-solved by >= 2 STACKED DEEP layers
(persistence); the seed's universal depth-1 fresh baseline is EXCLUDED from the
genuine/verdict determination. The raw seed-INCLUSIVE ``multiplicity`` /
``max_multiplicity`` / ``multiplicity_hist`` fields are kept unchanged (backward
compat), while a new DEEP (seed-excluded) ``deep_multiplicity`` /
``max_deep_multiplicity`` drives ``genuine_depth_tasks`` and the FAIL clause.

LLM-free / offline: builds tiny in-memory archives and calls the read-only
depth-attribution helpers directly.
"""

from __future__ import annotations

import pytest

from meta_n.analysis import depth_attribution as DA
from meta_n.analysis.depth_attribution import (
    deepest_leaf_id,
    run_depth_probe,
    split_candidate_mean,
    within_task_depth_profile,
    within_task_verdict,
)
from meta_n.core.archive import Archive, Candidate
from meta_n.core.meta_layer import InjectedCode, Trace


def _ic(d, *, pre=None, lib=None, tsm=None) -> InjectedCode:
    return InjectedCode(
        pre_process=pre, code_library=lib or {}, task_solution_map=tsm or {},
        source_depth=d,
    )


# --------------------------------------------------------------------------- #
# R1-B_depth-2 — split buckets d > candidate.depth as INHERITED, not fresh
# --------------------------------------------------------------------------- #
def test_split_deeper_inherited_frozen_is_not_fresh():
    """A trace at d > candidate.depth was NOT produced by this candidate's own solve
    (a deeper per-task-best inherited frozen under plain --consolidate);
    split_candidate_mean buckets it INHERITED, not fresh. Its score is still counted
    in the inherited + overall means (defensive, not dropped)."""
    cand = Candidate(
        candidate_id="c", parent_id="p", depth=3,
        per_task_scores={"same": 0.8, "deeper": 0.6, "shallower": 0.4},
        traces=[
            Trace(task_id="same", depth=3, score=0.8, success=True),       # == fresh
            Trace(task_id="deeper", depth=5, score=0.6, success=True),      # >  inherited
            Trace(task_id="shallower", depth=1, score=0.4, success=True),   # <  inherited
        ],
    )
    s = split_candidate_mean(cand)
    assert s.fresh_tasks == ["same"]
    assert sorted(s.inherited_tasks) == ["deeper", "shallower"]
    # the deeper score is NOT dropped: it is counted in the inherited + overall means.
    assert s.inherited_mean == pytest.approx((0.6 + 0.4) / 2)
    assert s.overall_mean == pytest.approx((0.8 + 0.6 + 0.4) / 3)


# --------------------------------------------------------------------------- #
# seed + one deep focus  ->  deep multiplicity 1  ->  NOT genuine  ->  FAIL
# --------------------------------------------------------------------------- #
def _seed_plus_single_focus_chain() -> Archive:
    """gen0 seed fresh-solves {t1,t2} at depth 1; a single deep layer re-solves
    ONLY t2 at depth 2 (generic pre_process). t2 is fresh at depths {1,2} — the
    seed baseline plus ONE deep focus (deep multiplicity 1), so NOT genuine."""
    arc = Archive()
    arc.add(Candidate(
        candidate_id="s", parent_id=None, depth=1, mean_score=0.4,
        per_task_scores={"t1": 0.4, "t2": 0.4},
        traces=[Trace(task_id="t1", depth=1, score=0.4),
                Trace(task_id="t2", depth=1, score=0.4)],
    ))
    arc.add(Candidate(
        candidate_id="d2", parent_id="s", depth=2, mean_score=0.5,
        injected_codes=[_ic(2, pre='n = task.metadata.get("size", 0)')],
        per_task_scores={"t1": 0.4, "t2": 0.6},
        traces=[Trace(task_id="t1", depth=1, score=0.4),
                Trace(task_id="t2", depth=2, score=0.6)],
    ))
    arc.add(Candidate(
        candidate_id="d3", parent_id="d2", depth=3, mean_score=0.5,
        injected_codes=[_ic(2, pre='n = task.metadata.get("size", 0)'),
                        _ic(3, pre='m = task.metadata.get("size", 0)')],
        per_task_scores={"t1": 0.4, "t2": 0.6},
        # d3 re-solves t2 AGAIN at depth 3: t2 now fresh at {1,2,3} -> deep {2,3}.
        traces=[Trace(task_id="t1", depth=1, score=0.4),
                Trace(task_id="t2", depth=3, score=0.6)],
    ))
    return arc


def test_seed_baseline_excluded_from_genuine_determination():
    """A task fresh at {1,2,3} has raw multiplicity 3 but DEEP multiplicity 2
    (depths 2,3) — the seed depth-1 baseline is dropped from the deep count."""
    arc = _seed_plus_single_focus_chain()
    p = within_task_depth_profile(arc, "d3")
    # raw seed-INCLUSIVE fields unchanged: t2 fresh at depths {1,2,3}.
    assert p.multiplicity == {"t1": 1, "t2": 3}
    assert p.max_multiplicity == 3
    # DEEP (seed-excluded): t1 has 0 deep re-solves, t2 has 2 (depths 2 and 3).
    assert p.deep_multiplicity == {"t1": 0, "t2": 2}
    assert p.max_deep_multiplicity == 2
    # >= 2 deep re-solves -> genuine within-task depth on t2.
    assert p.genuine_depth_tasks == ["t2"]


def test_single_deep_focus_is_not_genuine_and_fails():
    """Seed baseline + exactly one deep focus per task -> deep multiplicity 1 ->
    NOT genuine -> the verdict FAILs on the DEEP clause (not the router clause)."""
    arc = Archive()
    arc.add(Candidate(
        candidate_id="s", parent_id=None, depth=1, mean_score=0.4,
        per_task_scores={"t1": 0.4, "t2": 0.4, "t3": 0.4},
        traces=[Trace(task_id=t, depth=1, score=0.4) for t in ("t1", "t2", "t3")],
    ))
    arc.add(Candidate(
        candidate_id="d2", parent_id="s", depth=2, mean_score=0.5,
        injected_codes=[_ic(2, pre='n = task.metadata.get("size", 0)')],
        per_task_scores={"t1": 0.4, "t2": 0.6, "t3": 0.4},
        traces=[Trace(task_id="t1", depth=1, score=0.4),
                Trace(task_id="t2", depth=2, score=0.6),
                Trace(task_id="t3", depth=1, score=0.4)],
    ))
    arc.add(Candidate(
        candidate_id="d3", parent_id="d2", depth=3, mean_score=0.6,
        injected_codes=[_ic(2, pre='n = task.metadata.get("size", 0)'),
                        _ic(3, pre='m = task.metadata.get("size", 0)')],
        per_task_scores={"t1": 0.4, "t2": 0.6, "t3": 0.7},
        traces=[Trace(task_id="t1", depth=1, score=0.4),
                Trace(task_id="t2", depth=2, score=0.6),
                Trace(task_id="t3", depth=3, score=0.7)],
    ))
    p = within_task_depth_profile(arc, "d3")
    assert p.router_fraction == 0.0                  # generic pre_process only
    assert p.max_multiplicity == 2                   # seed + one focus (raw)
    assert p.max_deep_multiplicity == 1
    assert p.genuine_depth_tasks == []
    v = within_task_verdict(p)
    assert v["verdict"] == "FAIL"
    assert "STACKED DEEP" in v["reason"]
    assert "disjoint" in v["reason"] and "router breadth" in v["reason"]


# --------------------------------------------------------------------------- #
# edge case — task never solved by the seed but fresh at deep depths 2,3
# --------------------------------------------------------------------------- #
def test_task_unsolved_by_seed_but_deep_multiple_is_genuine():
    """A task the seed never fresh-solved, then fresh at DEEP depths 2 and 3, has
    deep multiplicity 2 -> correctly genuine (no seed baseline to exclude)."""
    arc = Archive()
    arc.add(Candidate(
        candidate_id="s", parent_id=None, depth=1, mean_score=0.3,
        per_task_scores={"anchor": 0.3},
        traces=[Trace(task_id="anchor", depth=1, score=0.3)],
    ))
    arc.add(Candidate(
        candidate_id="d2", parent_id="s", depth=2, mean_score=0.5,
        injected_codes=[_ic(2, pre='n = task.metadata.get("size", 0)')],
        per_task_scores={"anchor": 0.3, "late": 0.5},
        traces=[Trace(task_id="anchor", depth=1, score=0.3),
                Trace(task_id="late", depth=2, score=0.5)],
    ))
    arc.add(Candidate(
        candidate_id="d3", parent_id="d2", depth=3, mean_score=0.7,
        injected_codes=[_ic(2, pre='n = task.metadata.get("size", 0)'),
                        _ic(3, pre='m = task.metadata.get("size", 0)')],
        per_task_scores={"anchor": 0.3, "late": 0.7},
        traces=[Trace(task_id="anchor", depth=1, score=0.3),
                Trace(task_id="late", depth=3, score=0.7)],
    ))
    p = within_task_depth_profile(arc, "d3")
    assert p.deep_multiplicity == {"anchor": 0, "late": 2}
    assert p.max_deep_multiplicity == 2
    assert p.genuine_depth_tasks == ["late"]
    assert within_task_verdict(p)["verdict"] == "PASS"


# --------------------------------------------------------------------------- #
# edge case — empty / seed-only chain -> deep all 0
# --------------------------------------------------------------------------- #
def test_seed_only_chain_has_zero_deep_multiplicity():
    arc = Archive()
    arc.add(Candidate(
        candidate_id="s", parent_id=None, depth=1, mean_score=0.5,
        per_task_scores={"t": 0.5},
        traces=[Trace(task_id="t", depth=1, score=0.5)],
    ))
    p = within_task_depth_profile(arc, "s")
    assert p.deep_multiplicity == {"t": 0}
    assert p.max_deep_multiplicity == 0
    assert p.genuine_depth_tasks == []
    # max_chain_depth 1 < 3 -> INCONCLUSIVE regardless of the deep count.
    assert within_task_verdict(p)["verdict"] == "INCONCLUSIVE"


# --------------------------------------------------------------------------- #
# genuine persistence — same task fresh at multiple DEEP depths -> PASS
# --------------------------------------------------------------------------- #
def _genuine_deep_chain() -> Archive:
    arc = Archive()
    for cid, parent, depth, sc in (
        ("s", None, 1, 0.3), ("d2", "s", 2, 0.5), ("d3", "d2", 3, 0.7),
    ):
        codes = [_ic(k, pre='n = task.metadata.get("size", 0)') for k in range(2, depth + 1)]
        arc.add(Candidate(
            candidate_id=cid, parent_id=parent, depth=depth, mean_score=sc,
            injected_codes=codes,
            per_task_scores={"t1": sc},
            traces=[Trace(task_id="t1", depth=depth, score=sc)],
        ))
    return arc


def test_genuine_deep_persistence_passes_and_payload_carries_deep():
    arc = _genuine_deep_chain()
    p = within_task_depth_profile(arc, deepest_leaf_id(arc))
    # t1 fresh at {1,2,3}: raw max 3 (compat), deep max 2 (depths 2,3).
    assert p.max_multiplicity == 3
    assert p.max_deep_multiplicity == 2
    assert p.genuine_depth_tasks == ["t1"]
    v = within_task_verdict(p)
    assert v["verdict"] == "PASS"
    # both signals surfaced on the verdict payload; raw kept for compat.
    assert v["max_multiplicity"] == 3
    assert v["max_deep_multiplicity"] == 2


def test_run_depth_probe_serialises_deep_multiplicity(monkeypatch):
    monkeypatch.setattr(DA, "load_archive", lambda _d: _genuine_deep_chain())
    v = run_depth_probe("ignored/path")
    assert v["verdict"] == "PASS"
    # per-task raw (compat) and DEEP (seed-excluded) counts both serialised.
    assert v["multiplicity"] == {"t1": 3}
    assert v["deep_multiplicity"] == {"t1": 2}
    assert v["max_deep_multiplicity"] == 2
