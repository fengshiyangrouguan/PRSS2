"""Shared synthetic fixtures for the meta-n test suite.

Top-level conftest so these factories are visible to every ``tests/*.py``
module. They centralize the Candidate / Trace / Archive / InjectedCode builders
that the roadmap-v2 implementation steps need (selection-core, quality gate,
Ω_merge / oracle, scale-invariance, persistence round-trips).

Additive by design: the module-local ``make_trace`` / ``make_candidate`` /
``make_tasks`` helpers in ``test_evolutionary_orchestrator.py`` are called
directly (not as fixtures) and keep working unchanged. New tests inject these
factories by name.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from meta_n.core.archive import Archive, Candidate
from meta_n.core.meta_layer import InjectedCode, TaskDescription, Trace


# --- plain factory implementations (exposed as fixtures at the bottom) -------

def _make_trace(task_id, success=True, score=1.0, depth=1, **kw):
    """A Trace. ``score`` may be [0,1], continuous (e.g. 38.0) or the -1e9 sentinel."""
    return Trace(
        task_id=task_id,
        depth=depth,
        script=kw.pop("script", f"def solve(**kwargs): pass  # {task_id}"),
        success=success,
        score=score,
        error_summary=kw.pop("error_summary", "" if success else "error"),
        **kw,
    )


def _make_candidate(cid, mean_score=0.5, tasks=None, per_task_scores=None,
                    parent_id=None, depth=1, iteration=0, num_children=0,
                    injected_codes=None, traces=None):
    tasks = tasks or ["task_a", "task_b"]
    if per_task_scores is None:
        per_task_scores = {t: mean_score for t in tasks}
    if traces is None:
        traces = [_make_trace(t, success=True, score=per_task_scores[t]) for t in tasks]
    return Candidate(
        candidate_id=cid,
        parent_id=parent_id,
        iteration=iteration,
        depth=depth,
        injected_codes=injected_codes or [],
        traces=traces,
        pass_at_1=1.0,
        mean_score=mean_score,
        per_task_scores=per_task_scores,
        num_children=num_children,
    )


def _make_tasks(names=None):
    names = names or ["task_a", "task_b"]
    return [TaskDescription(task_id=n, description=f"Solve {n}") for n in names]


def _make_injected(pre_process="x = 1", code_library=None, code_library_bash=None,
                   rationale="", source_depth=2):
    return InjectedCode(
        pre_process=pre_process,
        code_library=code_library or {},
        code_library_bash=code_library_bash or {},
        rationale=rationale,
        source_depth=source_depth,
    )


def _make_disjoint_archive(n_tasks=3, base=0.4, win=0.9):
    """Archive where each task's best score comes from a DISTINCT lineage.

    Candidate ``c_task_i`` wins ``task_i`` (score=``win``) and is mediocre
    (``base``) elsewhere, so ``per_task_best_scores()`` picks a different
    candidate per task. This is the Ω_merge / oracle ground-truth fixture:
    the oracle mean (== ``win``) strictly exceeds every single candidate mean.
    """
    archive = Archive()
    tasks = [f"task_{i}" for i in range(n_tasks)]
    for i, winner in enumerate(tasks):
        scores = {t: (win if t == winner else base) for t in tasks}
        mean = sum(scores.values()) / len(scores)
        traces = [_make_trace(t, score=scores[t]) for t in tasks]
        archive.add(_make_candidate(
            f"c_{winner}", mean_score=mean, tasks=tasks,
            per_task_scores=scores, traces=traces, iteration=i,
        ))
    return archive


def _make_continuous_archive(means=(38.0, 40.0, 29.0)):
    """Archive on a continuous, non-[0,1] score scale (e.g. AlgoTune speedups)."""
    archive = Archive()
    for i, m in enumerate(means):
        archive.add(_make_candidate(f"cont_{i}", mean_score=m, iteration=i))
    return archive


def _make_sentinel_archive(means=(38.0, 40.0, 29.0), sentinel=-1e9):
    """Continuous archive containing one large-negative failure sentinel."""
    archive = Archive()
    for i, m in enumerate(list(means) + [sentinel]):
        archive.add(_make_candidate(
            f"sent_{i}", mean_score=m, tasks=["task_a"],
            per_task_scores={"task_a": m},
            traces=[_make_trace("task_a", score=m, success=(m > 0))],
            iteration=i,
        ))
    return archive


def _make_mock_omega():
    """An Ω stand-in whose ``generate(**kwargs)`` mirrors ``omega.generate`` and
    swallows unknown kwargs, so future Ω params (env notes, failure-class tags,
    ceiling maps) never re-break callers the way ``no_code_library`` just did."""
    from unittest.mock import AsyncMock

    omega = AsyncMock()

    async def _generate(traces=None, context_stack=None, tasks=None, depth=2,
                        temperature=None, inspiration_traces=None,
                        previous_scores=None, archive_best_scores=None,
                        solver_language="python", no_code_library=False, **kwargs):
        return InjectedCode(pre_process="x = 1", source_depth=depth), 50

    omega.generate = _generate
    return omega


def _write_candidate_dir(archive_dir, candidate: Candidate) -> Path:
    """Write ``candidate`` to the ``archive/<id>/`` layout that
    ``Archive.rebuild_from_disk`` reads: ``summary.json`` + ``traces/<task>.json``
    + ``injected_code_d{N}.json``. Field set mirrors what rebuild consumes."""
    cdir = Path(archive_dir) / candidate.candidate_id
    (cdir / "traces").mkdir(parents=True, exist_ok=True)
    summary = {
        "candidate_id": candidate.candidate_id,
        "parent_id": candidate.parent_id,
        "iteration": candidate.iteration,
        "depth": candidate.depth,
        "mean_score": candidate.mean_score,
        "pass_at_1": candidate.pass_at_1,
        "num_children": candidate.num_children,
        "temperature_used": candidate.temperature_used,
        "total_tokens": candidate.total_tokens,
        "per_task_scores": candidate.per_task_scores,
        "created_at": candidate.created_at,
    }
    (cdir / "summary.json").write_text(json.dumps(summary, indent=2))
    for tr in candidate.traces:
        (cdir / "traces" / f"{tr.task_id}.json").write_text(
            json.dumps(tr.model_dump(), indent=2, default=str)
        )
    for j, ic in enumerate(candidate.injected_codes):
        depth = j + 2  # injected_codes[0] reconstructs the depth-2 layer
        (cdir / f"injected_code_d{depth}.json").write_text(
            json.dumps(ic.model_dump(), indent=2, default=str)
        )
    return cdir


# --- fixtures (return the factory callable) ----------------------------------

@pytest.fixture
def make_trace():
    return _make_trace


@pytest.fixture
def make_candidate():
    return _make_candidate


@pytest.fixture
def make_tasks():
    return _make_tasks


@pytest.fixture
def make_injected():
    return _make_injected


@pytest.fixture
def make_disjoint_archive():
    return _make_disjoint_archive


@pytest.fixture
def make_continuous_archive():
    return _make_continuous_archive


@pytest.fixture
def make_sentinel_archive():
    return _make_sentinel_archive


@pytest.fixture
def make_mock_omega():
    return _make_mock_omega


@pytest.fixture
def write_candidate_dir():
    return _write_candidate_dir
