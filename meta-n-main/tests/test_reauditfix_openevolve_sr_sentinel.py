"""Reaudit #9 regression test — the SymbolicRegression failure path must NOT
route the ``-1e9`` ``score_scale`` sentinel into meta-n's ``score`` channel.

Finding (regression from OE-3, commit ca178f12): ``SymbolicRegressionAdapter``
routed a hard failure/crash to ``score_scale()['failure_sentinel']`` (``-1e9``)
in ``_failure_score()``. That value becomes ``trace.score`` and is then AVERAGED
into every reported metric — ``candidate.mean_score`` / ``best_mean_score``,
``per_task_best_scores`` / ``oracle_mean_score``, ``test_mean_score`` /
``chain_test_mean_score``, ``convergence_history`` and the STOP-rule delta. The
``score_scale`` contract explicitly states this sentinel "must NOT be averaged
as a real value". So one SR task crashed by the archive-best / by every
candidate dragged the headline mean to ~ ``-1e9 / N`` (~ ``-1e8``).

The fix clamps the failure score in meta-n's ``score`` channel to the reporting
floor (``0.0`` — identical to ``_coerce_eval_score`` and to the AlphaEvolve /
AlgoTune failure floor), so it never enters an averaged metric, WHILE the
``score_scale()`` advertisement of the evaluator's raw ``-1e9`` sentinel is
unchanged.

Every assertion below is offline (no LLM / Docker / real evaluator subprocess).
Each ``*_is_not_sentinel_poison`` assertion FAILS on the pre-fix code (``-1e9``)
and PASSES after the fix (``0.0``).
"""

from __future__ import annotations

import asyncio
import math

import meta_n.integrations.openevolve as oe
from meta_n.core.archive import Archive, Candidate
from meta_n.core.meta_layer import TaskDescription, Trace
from meta_n.integrations.openevolve import (
    AlphaEvolveMathAdapter,
    SymbolicRegressionAdapter,
)

# A failure score at or below this magnitude is the ``-1e9`` family of
# sentinels that ``_coerce_eval_score`` is designed to clamp; it must never
# reach an averaged/reported metric.
_SENTINEL_FLOOR = -1e8


def _orchestrator_mean(traces: list[Trace]) -> float:
    """Replicate evolutionary_orchestrator.py candidate.mean_score (only finite
    trace scores are averaged)."""
    finite = [t.score for t in traces if math.isfinite(t.score)]
    return sum(finite) / len(finite) if finite else 0.0


# ---------------------------------------------------------------------------
# 1. Adapter-level: the SR failure score is the bounded reporting floor.
# ---------------------------------------------------------------------------

def test_sr_failure_score_is_not_sentinel_poison():
    """FAILS pre-fix (returns -1e9), PASSES post-fix (returns 0.0)."""
    fail = SymbolicRegressionAdapter()._failure_score()
    assert math.isfinite(fail)
    assert fail > _SENTINEL_FLOOR, (
        f"SR _failure_score()={fail!r} is a -1e9-family sentinel; it becomes "
        "trace.score and poisons every averaged/reported mean."
    )
    assert fail == 0.0


def test_alphaevolve_failure_score_byte_identical():
    """Non-negative OpenEvolve scales keep the historical 0.0 floor (guards that
    the fix does not touch the byte-identical AlphaEvolve/AlgoTune path)."""
    assert AlphaEvolveMathAdapter()._failure_score() == 0.0


def test_score_scale_advertisement_unchanged():
    """The advertised sentinel documents the EVALUATOR's raw -1e9 output and is
    intentionally left unchanged — only meta-n's averaged ``score`` channel is
    clamped. (Passes both pre- and post-fix; guards against over-reaching.)"""
    scale = SymbolicRegressionAdapter().score_scale()
    assert scale["failure_sentinel"] == -1e9
    assert scale["kind"] == "continuous"


# ---------------------------------------------------------------------------
# 2. evaluate() failure branches return the bounded floor, not the sentinel.
# ---------------------------------------------------------------------------

def _sr_task() -> TaskDescription:
    return TaskDescription(
        task_id="synth_poly2d",
        description="",
        metadata={
            "problem_dir": "/nonexistent/problem",
            "score_key": "combined_score",
            "timeout": 5,
        },
    )


def _run_evaluate(monkey_return):
    """Drive SymbolicRegressionAdapter.evaluate() with the subprocess runner
    stubbed to ``monkey_return`` (a ('ok'|'error', payload) tuple). No real
    evaluator, LLM, or Docker is touched."""
    adapter = SymbolicRegressionAdapter()
    original = oe._run_openevolve_with_timeout
    oe._run_openevolve_with_timeout = lambda *a, **k: monkey_return
    try:
        return asyncio.run(adapter.evaluate(_sr_task(), "def run_search():\n    return None\n"))
    finally:
        oe._run_openevolve_with_timeout = original


def test_evaluate_crash_branch_is_not_sentinel_poison():
    """A subprocess crash (timeout / import error / no result). FAILS pre-fix
    (score=raw_score=-1e9), PASSES post-fix (0.0)."""
    res = _run_evaluate(("error", "boom: ImportError"))
    assert res.success is False
    assert math.isfinite(res.score) and res.score > _SENTINEL_FLOOR, res.score
    # Sibling channel: raw_score must match score (both clamped), consistent
    # with the ok-sentinel branch — the finding flagged raw_score=-1e9 here.
    assert math.isfinite(res.raw_score) and res.raw_score > _SENTINEL_FLOOR, res.raw_score
    assert res.score == 0.0 and res.raw_score == 0.0


def test_evaluate_ok_sentinel_branch_is_not_sentinel_poison():
    """The evaluator ran and emitted its -1e9 sentinel (clamped to 0.0 +
    flagged by ``raw_failure_sentinel``). FAILS pre-fix (score=-1e9), PASSES
    post-fix (0.0). ``success`` is False (OE-2: crash-flagged run)."""
    res = _run_evaluate((
        "ok",
        {"score": 0.0, "details": {"raw_failure_sentinel": -1e9, "error": "bad fit"}},
    ))
    assert res.success is False  # OE-2: sentinel failure is not a success
    assert math.isfinite(res.score) and res.score > _SENTINEL_FLOOR, res.score
    assert res.score == 0.0


def test_evaluate_valid_negative_fit_preserved():
    """A genuinely valid but poor fit (combined_score < 0, NOT a sentinel) keeps
    its real negative score and counts as a success (OE-2). Passes pre & post —
    guards that clamping the sentinel did not also clamp real negative fits."""
    res = _run_evaluate(("ok", {"score": -0.7, "details": {"combined_score": -0.7}}))
    assert res.success is True
    assert res.score == -0.7 and res.raw_score == -0.7


# ---------------------------------------------------------------------------
# 3. End-to-end via the real Archive: an all-crash SR task does NOT drag the
#    reported means to a huge-negative number, while ranking still prefers the
#    non-failing (better) candidate.
# ---------------------------------------------------------------------------

def _candidate(cid: str, task_scores: dict[str, tuple[float, bool]]) -> Candidate:
    traces = [
        Trace(task_id=tid, score=score, success=success)
        for tid, (score, success) in task_scores.items()
    ]
    cand = Candidate(candidate_id=cid, traces=traces)
    cand.mean_score = _orchestrator_mean(traces)
    cand.per_task_scores = {
        t.task_id: t.score for t in traces if math.isfinite(t.score)
    }
    return cand


def test_archive_all_crash_task_does_not_poison_reported_means_and_ranking():
    """Two candidates over {t_easy, t_hard}. BOTH crash t_hard (the finding's
    "unsolved by every candidate" trigger). GOOD scores higher on t_easy.

    FAILS pre-fix: with fail=-1e9 both candidate means and the oracle mean are
    ~ -5e8 (huge-negative). PASSES post-fix: fail=0.0 keeps every reported mean
    bounded, and the archive still ranks the non-failing/better candidate best.
    """
    adapter = SymbolicRegressionAdapter()
    fail = adapter._failure_score()

    good = _candidate("GOOD", {"t_easy": (5.0, True), "t_hard": (fail, False)})
    bad = _candidate("BAD", {"t_easy": (1.0, True), "t_hard": (fail, False)})

    archive = Archive()
    # Insert the weaker candidate first to prove ranking is by score, not order.
    archive.add(bad)
    archive.add(good)

    # (a) Reported means are bounded — the headline metric is not corrupted.
    assert archive.best_mean_score > -1.0, archive.best_mean_score
    ptb = archive.per_task_best_scores()
    oracle = sum(ptb.get(t, 0.0) for t in ("t_easy", "t_hard")) / 2
    assert oracle > -1.0, (oracle, ptb)
    # per-task best for the all-crash task is the bounded floor, not -1e9.
    assert ptb["t_hard"] > _SENTINEL_FLOOR, ptb

    # (b) Ranking still prefers the non-failing / better candidate.
    assert archive.best_candidate is not None
    assert archive.best_candidate.candidate_id == "GOOD", (
        archive.best_candidate.candidate_id,
        good.mean_score,
        bad.mean_score,
    )
