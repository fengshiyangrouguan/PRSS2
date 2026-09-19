"""Per-candidate repeated-eval denoising tests (eval_repeats → median-of-R)."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from meta_n.core.archive import Candidate
from meta_n.core.evolutionary_orchestrator import (
    EvolutionaryConfig,
    EvolutionaryOrchestrator,
)
from meta_n.core.meta_layer import TaskDescription, Trace


def _orch(**cfg):
    d = dict(max_depth=3, parallel=1, gate_tasks=0)
    d.update(cfg)
    return EvolutionaryOrchestrator(
        llm_client=MagicMock(), executor=MagicMock(), omega=MagicMock(),
        config=EvolutionaryConfig(**d), solver_language="bash",
    )


def _seed():
    return Candidate(candidate_id="seed", depth=1)


@pytest.mark.asyncio
async def test_eval_repeats_takes_median_per_task():
    orch = _orch(eval_repeats=3)
    solver = MagicMock()
    solver.solve = AsyncMock(return_value=("script", "r", 10))
    scores = iter([0.2, 0.9, 0.85])  # noisy draws; median is 0.85

    async def execute(script, task, timeout=30):
        return Trace(task_id=task.task_id, success=True, score=next(scores), script=script)

    orch.executor.execute = execute
    cand = _seed()
    await orch._evaluate_candidate(cand, solver, [TaskDescription(task_id="t1", description="x")])
    assert cand.per_task_scores["t1"] == 0.85    # median of [0.2, 0.85, 0.9], NOT the lucky 0.9
    assert solver.solve.await_count == 3         # solved R=3 times


@pytest.mark.asyncio
async def test_eval_repeats_1_is_single_solve():
    orch = _orch(eval_repeats=1)  # default
    solver = MagicMock()
    solver.solve = AsyncMock(return_value=("script", "r", 10))
    orch.executor.execute = AsyncMock(return_value=Trace(
        task_id="t1", success=True, score=0.5, script="s"))
    cand = _seed()
    await orch._evaluate_candidate(cand, solver, [TaskDescription(task_id="t1", description="x")])
    assert cand.per_task_scores["t1"] == 0.5
    assert solver.solve.await_count == 1         # no extra cost at the default


@pytest.mark.asyncio
async def test_eval_repeats_ignores_nan_in_median():
    orch = _orch(eval_repeats=3)
    solver = MagicMock()
    solver.solve = AsyncMock(return_value=("script", "r", 10))
    scores = iter([float("nan"), 0.6, 0.7])  # NaN sorts to bottom → median 0.6

    async def execute(script, task, timeout=30):
        s = next(scores)
        return Trace(task_id=task.task_id, success=True, score=s, script=script)

    orch.executor.execute = execute
    cand = _seed()
    await orch._evaluate_candidate(cand, solver, [TaskDescription(task_id="t1", description="x")])
    assert cand.per_task_scores["t1"] == 0.6     # median of [nan→-inf, 0.6, 0.7]


@pytest.mark.asyncio
async def test_eval_repeats_sums_inner_tokens_over_all_R():
    """H7: under eval_repeats>1 each task runs R full solves, each spending its
    OWN inner-LLM tokens — so candidate.inner_* must sum ALL R samples, not just
    the single median trace (which would understate inner_total ~R-fold)."""
    orch = _orch(eval_repeats=3)
    solver = MagicMock()
    solver.solve = AsyncMock(return_value=("script", "r", 0))
    inners = iter([10, 20, 30])   # three noisy draws; median trace is the 20

    async def execute(script, task, timeout=30):
        tok = next(inners)
        return Trace(
            task_id=task.task_id, success=True, score=0.5, script=script,
            inner_tokens=tok, inner_prompt_tokens=tok, inner_completion_tokens=0,
            inner_calls=1,
        )

    orch.executor.execute = execute
    cand = _seed()
    await orch._evaluate_candidate(cand, solver, [TaskDescription(task_id="t1", description="x")])
    assert cand.inner_tokens == 60          # 10 + 20 + 30 (ALL R), not the median 20
    assert cand.inner_prompt_tokens == 60
    assert cand.inner_calls == 3            # one llm() call per sample × 3 samples


@pytest.mark.asyncio
async def test_eval_repeats_1_inner_tokens_unchanged():
    """H7 default: at eval_repeats=1 the all-R sum equals the single trace's
    inner fields — byte-identical to the prior single-eval accounting."""
    orch = _orch(eval_repeats=1)
    solver = MagicMock()
    solver.solve = AsyncMock(return_value=("script", "r", 0))
    orch.executor.execute = AsyncMock(return_value=Trace(
        task_id="t1", success=True, score=0.5, script="s",
        inner_tokens=42, inner_calls=1,
    ))
    cand = _seed()
    await orch._evaluate_candidate(cand, solver, [TaskDescription(task_id="t1", description="x")])
    assert cand.inner_tokens == 42
    assert cand.inner_calls == 1
