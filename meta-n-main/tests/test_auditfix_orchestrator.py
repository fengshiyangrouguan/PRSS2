"""Regression tests for the confirmed audit fixes in
``meta_n/core/evolutionary_orchestrator.py``.

Each test FAILS on the original (un-fixed) code and PASSES after the fix.
Everything here is LLM-free / offline (no LM Studio, Docker, or network).

Findings covered (ids from .audit/audit_findings.json):
  1  — inner-LLM token accounting lost across --resume
  2  — gate median-of-R sort not guarded against non-finite scores
  3  — ``Optional`` annotations used but never imported (NameError under
       type-hint introspection)
 17  — oracle_summary.json oracle_mean used a feasible-subset denominator
 18  — regression-guard base-floor resample hardcoded depth=1 (mis-route crash)
 29  — gate-phase outer (script-generation) tokens silently discarded
 30  — within-task saturation release hardcoded a 1.0 score ceiling (scale-blind)
 46  — full-eval gather had no per-task exception isolation
 47  — per-candidate telemetry rollup re-parsed the whole ledger every save
"""

import json
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from meta_n.core.archive import Candidate
from meta_n.core.evolutionary_orchestrator import (
    EvolutionaryConfig,
    EvolutionaryOrchestrator,
    EvolutionaryResult,
)
from meta_n.core.llm_client import stable_crn_seed
from meta_n.core.meta_layer import InjectedCode, TaskDescription, Trace


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _make_orch(*, llm_client=None, **overrides) -> EvolutionaryOrchestrator:
    defaults = dict(
        max_depth=3,
        parallel=1,
        patience=1,
        gate_tasks=0,
        beam_width=1,
        beam_candidates=1,
    )
    defaults.update(overrides)
    config = EvolutionaryConfig(**defaults)
    orch = EvolutionaryOrchestrator(
        llm_client=llm_client or MagicMock(),
        executor=MagicMock(),
        omega=MagicMock(),
        config=config,
        solver_language="bash",
    )
    return orch


def _trace(
    task_id,
    success=True,
    score=1.0,
    inner_tokens=0,
    inner_calls=0,
    inner_prompt_tokens=0,
    inner_completion_tokens=0,
) -> Trace:
    return Trace(
        task_id=task_id,
        depth=2,
        script="s",
        success=success,
        score=score,
        error_summary="" if success else "err",
        inner_tokens=inner_tokens,
        inner_calls=inner_calls,
        inner_prompt_tokens=inner_prompt_tokens,
        inner_completion_tokens=inner_completion_tokens,
    )


def _tasks(names) -> list[TaskDescription]:
    return [TaskDescription(task_id=n, description=f"Solve {n}") for n in names]


# --------------------------------------------------------------------------- #
# Finding 3 — ``Optional`` used in annotations but never imported
# --------------------------------------------------------------------------- #

def test_finding3_annotations_resolve_without_nameerror():
    """get_type_hints must resolve the orchestrator's annotations.

    Original: ``seed_code_library: Optional[...]`` and
    ``_within_task_focus(...) -> Optional[str]`` reference an unimported
    ``Optional`` (inert only because of ``from __future__ import annotations``),
    so resolving the hints raises ``NameError: name 'Optional' is not defined``.
    """
    from typing import get_type_hints

    # The method's return + parameter annotations must resolve.
    hints = get_type_hints(EvolutionaryOrchestrator._within_task_focus)
    assert hints["return"] == (str | None)

    # The dataclass field annotation must resolve too.
    cfg_hints = get_type_hints(EvolutionaryConfig)
    assert cfg_hints["seed_code_library"] == (dict[str, str] | None)


# --------------------------------------------------------------------------- #
# Finding 2 — gate median-of-R sort must guard non-finite scores
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_finding2_gate_median_guards_nonfinite_score():
    """With three repeats scored [0.1, 0.9, NaN] (in that physical order), the
    raw-score sort (original) leaves NaN last and picks 0.9 as the "median",
    while the non-finite-safe sort (fixed, NaN -> -inf) picks 0.1.
    """
    orch = _make_orch(gate_tasks=1, gate_repeats=3, gate_margin=None)
    tasks = _tasks(["t0"])
    samples_in_order = [
        _trace("t0", success=True, score=0.1),
        _trace("t0", success=True, score=0.9),
        _trace("t0", success=True, score=float("nan")),
    ]
    orch._gate_solve_one = AsyncMock(side_effect=samples_in_order)
    orch.rng = MagicMock()
    orch.rng.sample = lambda pop, k: list(pop)[:k]

    child = Candidate(candidate_id="child", parent_id="p", iteration=1, depth=2)
    parent = Candidate(candidate_id="p", depth=1, per_task_scores={"t0": 0.5})

    passed, n, traces = await orch._gate_check(child, parent, MagicMock(), tasks)

    # The median-scoring trace kept for reuse must be the score-0.1 sample.
    assert traces["t0"].score == pytest.approx(0.1)


# --------------------------------------------------------------------------- #
# Finding 18 — base-floor resample must use the seed's actual depth
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_finding18_base_floor_resample_routes_by_seed_depth():
    """A seeded depth-2 MetaLayer chain solves via ``execute()`` (no ``seed=``
    kwarg). The base-floor resample must inherit the seed's depth so the
    dispatch routes to ``execute()``; the original hardcoded depth=1 routed to
    the native ``solve(task, seed=)`` path, raising TypeError and aborting.
    """
    orch = _make_orch(regression_guard=True, regression_guard_repeats=2, parallel=1)
    tasks = _tasks(["t0"])

    class FakeMetaSolver:
        def __init__(self):
            self.execute = AsyncMock(
                return_value=(_trace("t0", success=True, score=0.5), 0)
            )

        async def solve(self, task, additional_context=""):  # NO seed kwarg
            return ("script", "reasoning", 0)

    seed_solver = FakeMetaSolver()
    seed = Candidate(
        candidate_id="gen0_seed",
        iteration=0,
        depth=2,  # seeded chain depth
        traces=[_trace("t0", success=True, score=0.5)],
        per_task_scores={"t0": 0.5},
    )
    result = EvolutionaryResult()

    await orch._establish_base_floor(seed, seed_solver, tasks, result)

    # The single resample (R=2 -> r in range(1,2)) routed through execute().
    assert seed_solver.execute.await_count == 1
    assert orch.archive._base_floor.get("t0") is not None


# --------------------------------------------------------------------------- #
# R1-D — regression-guard resamples must draw DISTINCT CRN seeds (not collide  #
# on index 0) via the per-resample ``crn_repeat_offset``.                     #
# --------------------------------------------------------------------------- #

class _RecordingSeedSolver:
    """Depth-1 native solver that records every CRN seed threaded into solve()."""

    def __init__(self):
        self.seeds: list[tuple[str, int | None]] = []

    async def solve(self, task, seed=None, additional_context=""):
        self.seeds.append((task.task_id, seed))
        return ("script", "reasoning", 0)


def _azure_llm():
    return SimpleNamespace(config=SimpleNamespace(model="gpt-x", backend="azure"))


@pytest.mark.asyncio
async def test_regression_guard_resamples_get_distinct_crn_seeds():
    """Each base-floor resample must offset its CRN repeat index so distinct
    resamples draw distinct paired-eval seeds instead of all colliding on 0.

    PRE-FIX both resamples record ``stable_crn_seed(...,0)``; POST-FIX resample
    r records ``stable_crn_seed(..., r)`` (eval_repeats==1 => offset == r).
    """
    orch = _make_orch(
        regression_guard=True,
        regression_guard_repeats=3,
        paired_eval=True,
        parallel=1,
        llm_client=_azure_llm(),
    )
    orch.executor.execute = AsyncMock(
        return_value=_trace("t0", success=True, score=0.5)
    )
    seed_solver = _RecordingSeedSolver()
    seed = Candidate(
        candidate_id="gen0_seed",
        iteration=0,
        depth=1,  # native depth-1 solve() path
        traces=[_trace("t0", success=True, score=0.5)],
        per_task_scores={"t0": 0.5},
    )
    await orch._establish_base_floor(
        seed, seed_solver, tasks=_tasks(["t0"]), result=EvolutionaryResult()
    )

    recorded = [s for (tid, s) in seed_solver.seeds if tid == "t0"]
    # R=3 -> two resamples (r=1, r=2), each one solve of t0.
    assert len(recorded) == 2
    s0 = stable_crn_seed(orch.config.seed, "t0", 0)
    assert recorded[0] == stable_crn_seed(orch.config.seed, "t0", 1)
    assert recorded[1] == stable_crn_seed(orch.config.seed, "t0", 2)
    # Critically, NEITHER resample collides with draw 0 (the seed's own eval).
    assert all(s != s0 for s in recorded)


@pytest.mark.asyncio
async def test_regression_guard_resamples_seeds_none_when_paired_eval_off():
    """OFF-path byte-identity guard: paired_eval=False => no seed on the wire for
    any resample (offset is inert since _crn_seed short-circuits to None)."""
    orch = _make_orch(
        regression_guard=True,
        regression_guard_repeats=3,
        paired_eval=False,
        parallel=1,
        llm_client=_azure_llm(),
    )
    orch.executor.execute = AsyncMock(
        return_value=_trace("t0", success=True, score=0.5)
    )
    seed_solver = _RecordingSeedSolver()
    seed = Candidate(
        candidate_id="gen0_seed",
        iteration=0,
        depth=1,
        traces=[_trace("t0", success=True, score=0.5)],
        per_task_scores={"t0": 0.5},
    )
    await orch._establish_base_floor(
        seed, seed_solver, tasks=_tasks(["t0"]), result=EvolutionaryResult()
    )

    recorded = [s for (tid, s) in seed_solver.seeds if tid == "t0"]
    assert len(recorded) == 2
    assert all(s is None for s in recorded)


# --------------------------------------------------------------------------- #
# Finding 30 — saturation release ceiling must come from the score scale
# --------------------------------------------------------------------------- #

def test_finding30_saturation_release_is_scale_aware():
    """On a continuous / unbounded scale (hi is None) a score above 1.0 must NOT
    be treated as saturated (the chain keeps deepening the focus task). On a
    unit scale (hi=1.0) a 1.0 score is still released to round-robin (parity).
    """
    orch = _make_orch(within_task_recursion=True, consolidate=True)
    cons_targets = ["other"]

    # (a) continuous scale, focus score 3.0 (e.g. AlgoTune speedup): no release.
    cont_adapter = MagicMock()
    cont_adapter.score_scale.return_value = {
        "kind": "continuous", "lo": None, "hi": None,
    }
    orch.adapter = cont_adapter
    parent_cont = Candidate(candidate_id="pc", depth=2, per_task_scores={"focus": 3.0})
    with patch(
        "meta_n.analysis.depth_attribution.split_candidate_mean",
        return_value=types.SimpleNamespace(fresh_tasks=["focus"]),
    ):
        focus_cont = orch._within_task_focus(parent_cont, cons_targets, k=0)
    assert focus_cont == "focus"  # original released to "other"

    # (b) unit scale, focus score 1.0: release to round-robin (unchanged).
    unit_adapter = MagicMock()
    unit_adapter.score_scale.return_value = {
        "kind": "unit", "lo": 0.0, "hi": 1.0,
    }
    orch.adapter = unit_adapter
    parent_unit = Candidate(candidate_id="pu", depth=2, per_task_scores={"focus": 1.0})
    with patch(
        "meta_n.analysis.depth_attribution.split_candidate_mean",
        return_value=types.SimpleNamespace(fresh_tasks=["focus"]),
    ):
        focus_unit = orch._within_task_focus(parent_unit, cons_targets, k=0)
    assert focus_unit == "other"


# --------------------------------------------------------------------------- #
# Finding 46 — full-eval gather isolates a single task's solver fault
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_finding46_full_eval_isolates_solver_exception():
    """One task's solve raising must not abort the whole evaluation; it becomes
    a failed Trace (score 0.0, success=False) while the others still score.
    """
    orch = _make_orch(parallel=1)
    orch.adapter = None
    tasks = _tasks(["t0", "t1"])
    candidate = Candidate(candidate_id="c", depth=2, iteration=1)

    async def execute(task):
        if task.task_id == "t1":
            raise RuntimeError("boom on t1")
        return (_trace("t0", success=True, score=0.7), 5)

    solver = MagicMock()
    solver.execute = execute

    candidate = await orch._evaluate_candidate(candidate, solver, tasks)

    by_task = {t.task_id: t for t in candidate.traces}
    assert set(by_task) == {"t0", "t1"}
    assert by_task["t0"].score == pytest.approx(0.7)
    assert by_task["t1"].success is False
    assert by_task["t1"].score == 0.0
    assert "raised" in by_task["t1"].error_summary.lower()
    # both scores are finite, so both land in per_task_scores
    assert candidate.per_task_scores["t1"] == 0.0


# --------------------------------------------------------------------------- #
# Finding 47 — telemetry rollup caches the parsed ledger (no O(N^2) re-parse)
# --------------------------------------------------------------------------- #

def test_finding47_agent_run_rows_are_cached(tmp_path, monkeypatch):
    """Two reads of an unchanged ledger must parse it once; a grown ledger
    invalidates the cache and re-parses.
    """
    import meta_n.core.evolutionary_orchestrator as mod

    orch = _make_orch(output_dir=str(tmp_path))
    tdir = tmp_path / "telemetry"
    tdir.mkdir()
    ledger = tdir / "agent_runs.jsonl"
    ledger.write_text(
        json.dumps({"run_id": "r1", "candidate_id": "c", "total_tokens": 10}) + "\n"
    )

    calls = {"n": 0}
    real_loads = json.loads

    def counting_loads(s, *a, **k):
        calls["n"] += 1
        return real_loads(s, *a, **k)

    # Patch the stdlib module itself: parsing moved to run_persistence /
    # telemetry.iter_jsonl_objects, but they all share this module object.
    monkeypatch.setattr(json, "loads", counting_loads)

    rows1 = orch._read_agent_run_rows()
    assert len(rows1) == 1
    after_first = calls["n"]
    assert after_first >= 1

    # Second read of the unchanged file: cache hit, no re-parse.
    rows2 = orch._read_agent_run_rows()
    assert rows2 == rows1
    assert calls["n"] == after_first

    # Grow the ledger -> (size, mtime) changes -> cache invalidates -> re-parse.
    ledger.write_text(
        json.dumps({"run_id": "r1", "candidate_id": "c", "total_tokens": 10}) + "\n"
        + json.dumps({"run_id": "r2", "candidate_id": "c", "total_tokens": 20}) + "\n"
    )
    rows3 = orch._read_agent_run_rows()
    assert len(rows3) == 2
    assert calls["n"] > after_first


# --------------------------------------------------------------------------- #
# Finding 17 — oracle_summary.json uses the full-task-set denominator
# --------------------------------------------------------------------------- #

def test_finding17_oracle_summary_full_set_denominator(tmp_path):
    """oracle_summary.json must average the oracle over the FULL task set
    (missing tasks counted as 0.0), agreeing with summary.json — not over only
    the feasible subset.
    """
    orch = _make_orch(output_dir=str(tmp_path))
    orch.adapter = None
    orch._tasks = _tasks(["t0", "t1"])  # full set is two tasks

    cand = Candidate(
        candidate_id="c",
        depth=1,
        traces=[_trace("t0", success=True, score=0.8)],
        per_task_scores={"t0": 0.8},
        mean_score=0.8,
    )
    orch.archive.add(cand)  # only t0 has a finite-scored candidate

    result = EvolutionaryResult()
    orch.save_results(result, output_dir=str(tmp_path))

    data = json.loads((tmp_path / "oracle_summary.json").read_text())
    # full-set: (0.8 + 0.0) / 2 == 0.4 ; subset (original) would be 0.8 / 1.
    assert data["oracle_mean_score"] == pytest.approx(0.4)


# --------------------------------------------------------------------------- #
# Finding 29 — gate-phase outer tokens reach result.total_tokens
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_finding29_gate_outer_tokens_counted(tmp_path):
    """A bred child's gate solve is a real outer LLM call; its tokens (the
    cumulative_usage delta across the gate) must be added to result.total_tokens.
    """
    orch = _make_orch(
        output_dir=str(tmp_path),
        gate_tasks=1,
        gate_margin=0.0,
        max_iterations=1,
        patience=1,
    )
    orch.adapter = None
    cu = {"total": 0, "prompt": 0, "completion": 0, "calls": 0}
    orch.llm_client.cumulative_usage = cu

    tasks = _tasks(["t0"])

    # Seed: native depth-1 solve, 0 tokens, strong baseline score so the child's
    # gate (low score) FAILS -> exercises the gate-fail accounting path.
    orch.solver.solve = AsyncMock(return_value=("s", "r", 0))
    orch.executor.execute = AsyncMock(
        return_value=_trace("t0", success=True, score=0.8)
    )
    # Omega: one non-empty injection (depth-2 child), 50 outer tokens.
    orch.omega.generate = AsyncMock(
        return_value=(InjectedCode(pre_process="echo hi"), 50)
    )

    async def fake_child_execute(task):
        cu["total"] += 100  # the gate solve spends 100 outer tokens
        return (_trace("t0", success=True, score=0.2), 100)

    fake_solver = MagicMock()
    fake_solver.execute = fake_child_execute
    # The child (depth 2) routes through the candidate-solver builder; the seed
    # uses the native self.solver and is unaffected.
    orch._build_solver_from_candidate = MagicMock(return_value=fake_solver)

    result = await orch.run(tasks)

    # seed(0) + gate delta(100) + omega(50) == 150 ; original counted only 50.
    assert result.total_tokens == 150


# --------------------------------------------------------------------------- #
# Finding 1 — inner-LLM token accounting persists + restores across --resume
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_finding1_inner_tokens_persist_and_restore_on_resume(tmp_path):
    """The seed's inner-LLM tokens must be written to checkpoint.json and
    restored on resume; otherwise a resumed run's inner_* totals cover only the
    post-resume slice.
    """
    out = str(tmp_path)

    def build():
        orch = _make_orch(
            output_dir=out,
            max_iterations=1,
            patience=1,
            gate_tasks=0,
        )
        orch.solver.solve = AsyncMock(return_value=("s", "r", 0))
        orch.executor.execute = AsyncMock(
            return_value=_trace(
                "t0",
                success=True,
                score=0.5,
                inner_tokens=123,
                inner_calls=4,
                inner_prompt_tokens=100,
                inner_completion_tokens=23,
            )
        )
        orch.omega.generate = AsyncMock(return_value=(InjectedCode(), 50))  # empty
        return orch

    tasks = _tasks(["t0"])

    res1 = await build().run(tasks)
    assert res1.inner_tokens == 123

    # The checkpoint persisted the inner accounting.
    ckpt = json.loads((tmp_path / "checkpoint.json").read_text())
    assert ckpt["inner_tokens"] == 123
    assert ckpt["inner_prompt_tokens"] == 100
    assert ckpt["inner_completion_tokens"] == 23
    assert ckpt["inner_calls"] == 4

    # A resumed run restores them instead of restarting at zero.
    res2 = await build().run(tasks, resume=True)
    assert res2.inner_tokens == 123
    assert res2.inner_prompt_tokens == 100
    assert res2.inner_completion_tokens == 23
    assert res2.inner_calls == 4
