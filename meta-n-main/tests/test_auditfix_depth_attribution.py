"""Regression tests for audit fixes in meta_n/analysis/depth_attribution.py.

Finding 42 — dead-code helper ``_fresh_tasks_at`` defined but never called.
The de-duplication its docstring promised never existed (its only plausible
consumer, ``within_task_depth_profile``, re-implements the split inline). The
fix removes the dead helper.

These tests are LLM-free / offline: they build a tiny in-memory Archive and
exercise the read-only depth-attribution helpers directly.
"""

from __future__ import annotations

from meta_n.analysis import depth_attribution as da
from meta_n.core.archive import Archive, Candidate
from meta_n.core.meta_layer import Trace


def _candidate(cid, depth, parent, task_depths, scores):
    """Build a Candidate with one trace per task at the given depth."""
    traces = [Trace(task_id=tid, depth=d, score=scores[tid]) for tid, d in task_depths.items()]
    return Candidate(
        candidate_id=cid,
        parent_id=parent,
        depth=depth,
        traces=traces,
        per_task_scores=dict(scores),
    )


def _archive():
    arc = Archive()
    # seed at depth 1 fresh-solves t1; child at depth 2 fresh-solves t2 (t1 inherited).
    seed = _candidate("c1", 1, None, {"t1": 1}, {"t1": 1.0})
    child = _candidate("c2", 2, "c1", {"t1": 1, "t2": 2}, {"t1": 1.0, "t2": 0.5})
    arc.add(seed)
    arc.add(child)
    return arc


def test_fresh_tasks_at_dead_helper_removed():
    """Finding 42: the unused private helper must no longer be present."""
    assert not hasattr(da, "_fresh_tasks_at"), (
        "_fresh_tasks_at was dead code (never called) and should be removed"
    )


def test_within_task_depth_profile_still_correct_after_removal():
    """Removing the dead helper must not change the depth-attribution output."""
    arc = _archive()
    profile = da.within_task_depth_profile(arc, "c2")
    # c1 fresh-solves t1 at depth 1; c2 fresh-solves t2 at depth 2.
    assert profile.fresh_depths_by_task["t1"] == [1]
    assert profile.fresh_depths_by_task["t2"] == [2]
    # Each task solved once along the chain → multiplicity 1 (no within-task recursion).
    assert profile.multiplicity == {"t1": 1, "t2": 2 - 1}
    assert profile.max_multiplicity == 1
    # fresh_split is still populated from split_candidate_mean (the real de-dup path).
    assert [s.candidate_id for s in profile.fresh_split] == ["c1", "c2"]


def test_split_candidate_mean_fresh_inherited():
    """split_candidate_mean still cleanly separates fresh vs inherited tasks."""
    arc = _archive()
    split = da.split_candidate_mean(arc.get("c2"))
    assert set(split.fresh_tasks) == {"t2"}
    assert set(split.inherited_tasks) == {"t1"}
