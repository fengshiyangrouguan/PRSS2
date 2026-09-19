"""Quality-gate tests (Step 2): score-threshold gate (1.1) + gate-trace reuse (1.6)."""

from unittest.mock import MagicMock

import pytest

from meta_n.core.archive import Candidate
from meta_n.core.evolutionary_orchestrator import (
    EvolutionaryConfig,
    EvolutionaryOrchestrator,
)
from meta_n.core.meta_layer import TaskDescription, Trace


def _make_orchestrator(**overrides):
    defaults = dict(max_depth=3, parallel=1, patience=1, gate_tasks=1,
                    beam_width=1, beam_candidates=1)
    defaults.update(overrides)
    return EvolutionaryOrchestrator(
        llm_client=MagicMock(), executor=MagicMock(), omega=MagicMock(),
        config=EvolutionaryConfig(**defaults), solver_language="bash",
    )


def _trace(task_id, success=True, score=1.0, inner_tokens=0):
    return Trace(task_id=task_id, depth=2, success=success, score=score,
                 script="s", inner_tokens=inner_tokens)


def _solver(execute_fn):
    s = MagicMock()
    s.execute = execute_fn
    return s


def _const_solver(traces_by_task):
    async def execute(task):
        return traces_by_task[task.task_id], 0
    return _solver(execute)


def _child(depth=2):
    return Candidate(candidate_id="child", parent_id="gen0_seed", iteration=1, depth=depth)


def _parent(per_task):
    return Candidate(candidate_id="gen0_seed", depth=1, per_task_scores=per_task)


def _tasks(n=1):
    return [TaskDescription(task_id=f"t{i}", description=f"t{i}") for i in range(n)]


# --------------------------------------------------------------------------- #
# 1.1 — score-threshold gate
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_gate_rejects_low_score_vs_parent():
    orch = _make_orchestrator(gate_margin=0.0)
    solver = _const_solver({"t0": _trace("t0", success=True, score=0.3)})
    passed, _, traces = await orch._gate_check(_child(), _parent({"t0": 0.8}), solver, _tasks(1))
    assert passed is False            # 0.3 < 0.8 — the old liveness gate would PASS this
    assert "t0" in traces             # ...but the trace is still captured for reuse


@pytest.mark.asyncio
async def test_gate_passes_at_or_above_parent_minus_margin():
    orch = _make_orchestrator(gate_margin=0.05)
    passed_hi, _, _ = await orch._gate_check(
        _child(), _parent({"t0": 0.8}), _const_solver({"t0": _trace("t0", score=0.79)}), _tasks(1))
    passed_lo, _, _ = await orch._gate_check(
        _child(), _parent({"t0": 0.8}), _const_solver({"t0": _trace("t0", score=0.70)}), _tasks(1))
    assert passed_hi is True          # 0.79 >= 0.80 - 0.05
    assert passed_lo is False         # 0.70 <  0.75


@pytest.mark.asyncio
async def test_gate_retains_crash_rejection():
    orch = _make_orchestrator(gate_margin=0.0)

    async def boom(task):
        raise RuntimeError("broken injection")

    passed, _, traces = await orch._gate_check(_child(), _parent({"t0": 0.5}), _solver(boom), _tasks(1))
    assert passed is False
    assert traces == {}               # a crash records no trace and rejects


@pytest.mark.asyncio
async def test_gate_fail_open_when_no_parent_baseline():
    orch = _make_orchestrator(gate_margin=0.0)
    solver = _const_solver({"t0": _trace("t0", success=True, score=0.0)})
    passed, _, _ = await orch._gate_check(_child(), _parent({}), solver, _tasks(1))
    assert passed is True             # missing baseline → liveness (never reject everything)


@pytest.mark.asyncio
async def test_gate_none_parent_is_liveness():
    orch = _make_orchestrator(gate_margin=0.0)
    solver = _const_solver({"t0": _trace("t0", success=True, score=0.01)})
    passed, _, _ = await orch._gate_check(_child(), None, solver, _tasks(1))
    assert passed is True


@pytest.mark.asyncio
@pytest.mark.parametrize("baseline,score,expect", [
    (1.0, 1.0, True), (1.0, 0.0, False),    # binary {0,1}
    (12.0, 12.0, True), (12.0, 5.0, False),  # continuous
])
async def test_gate_scale_independent(baseline, score, expect):
    orch = _make_orchestrator(gate_margin=0.0)
    solver = _const_solver({"t0": _trace("t0", success=True, score=score)})
    passed, _, _ = await orch._gate_check(_child(), _parent({"t0": baseline}), solver, _tasks(1))
    assert passed is expect           # no [0,1] assumption


@pytest.mark.asyncio
async def test_gate_disabled_when_margin_none():
    orch = _make_orchestrator(gate_margin=None)  # thresholding off (ablation)
    solver = _const_solver({"t0": _trace("t0", success=True, score=0.01)})
    passed, _, _ = await orch._gate_check(_child(), _parent({"t0": 0.8}), solver, _tasks(1))
    assert passed is True


@pytest.mark.asyncio
async def test_gate_median_over_repeats():
    orch = _make_orchestrator(gate_margin=0.0, gate_repeats=3)
    scores = iter([0.2, 0.9, 0.85])
    calls = 0

    async def execute(task):
        nonlocal calls
        calls += 1
        return _trace("t0", success=True, score=next(scores)), 0

    passed, _, traces = await orch._gate_check(_child(), _parent({"t0": 0.8}), _solver(execute), _tasks(1))
    assert passed is True             # median(0.2,0.9,0.85)=0.85 >= 0.8
    assert calls == 3                 # solved R times
    assert traces["t0"].score == 0.85  # the median trace is kept


# --------------------------------------------------------------------------- #
# 1.6 — gate-trace reuse in eval
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_gate_traces_reused_in_eval_no_double_solve():
    orch = _make_orchestrator(gate_margin=None)  # liveness, 2 tasks, gate samples 1
    calls = {"t0": 0, "t1": 0}

    async def execute(task):
        calls[task.task_id] += 1
        return _trace(task.task_id, success=True, score=0.6), 0

    child, solver, tasks = _child(), _solver(execute), _tasks(2)
    passed, gate_n, gate_traces = await orch._gate_check(child, _parent({}), solver, tasks)
    assert passed is True and gate_n == 1
    await orch._evaluate_candidate(child, solver, tasks, precomputed=gate_traces)
    # each task executed EXACTLY once across gate+eval (the gated one reused) — 2 not 3
    assert calls["t0"] == 1 and calls["t1"] == 1


@pytest.mark.asyncio
async def test_reuse_inner_tokens_counted_once():
    orch = _make_orchestrator(gate_margin=None)

    async def execute(task):
        return _trace(task.task_id, success=True, score=0.6, inner_tokens=500), 0

    child, solver, tasks = _child(), _solver(execute), _tasks(1)
    _, _, gate_traces = await orch._gate_check(child, _parent({}), solver, tasks)
    await orch._evaluate_candidate(child, solver, tasks, precomputed=gate_traces)
    assert child.inner_tokens == 500   # reused trace's inner tokens counted once


@pytest.mark.asyncio
async def test_reuse_composes_with_threshold():
    orch = _make_orchestrator(gate_margin=0.0)

    async def execute(task):
        return _trace(task.task_id, success=True, score=0.9), 0

    child, solver, tasks = _child(), _solver(execute), _tasks(1)
    passed, _, gate_traces = await orch._gate_check(child, _parent({"t0": 0.8}), solver, tasks)
    assert passed is True              # 0.9 >= 0.8
    await orch._evaluate_candidate(child, solver, tasks, precomputed=gate_traces)
    assert child.per_task_scores["t0"] == 0.9  # the 0.9 gate trace became the eval trace
