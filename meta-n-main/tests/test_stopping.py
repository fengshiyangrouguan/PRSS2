"""Stopping tests (Step 6): right-size patience (5.4), magnitude-aware epsilon
(1.5a), oracle-aware stop (N7)."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from meta_n.core.evolutionary_orchestrator import (
    EvolutionaryConfig,
    EvolutionaryOrchestrator,
)
from meta_n.core.meta_layer import InjectedCode, TaskDescription, Trace

_improved = EvolutionaryOrchestrator._iteration_improved


def _orch(**cfg):
    defaults = dict(max_depth=3, parallel=1, beam_width=1, beam_candidates=1, gate_tasks=0)
    defaults.update(cfg)
    return EvolutionaryOrchestrator(
        llm_client=MagicMock(), executor=MagicMock(), omega=MagicMock(),
        config=EvolutionaryConfig(**defaults), solver_language="bash",
    )


# --------------------------------------------------------------------------- #
# 5.4 — right-size patience
# --------------------------------------------------------------------------- #

def test_effective_patience_clamped_when_ge_max_iter():
    assert _orch(patience=10, max_iterations=3)._effective_patience == 2   # max_iter - 1


def test_effective_patience_unchanged_when_below_max_iter():
    assert _orch(patience=3, max_iterations=50)._effective_patience == 3


def test_effective_patience_floor_is_one():
    assert _orch(patience=10, max_iterations=1)._effective_patience == 1   # max(1, 0)


def test_no_early_stop_disables_patience():
    assert _orch(patience=2, max_iterations=5, no_early_stop=True)._effective_patience == 6


# --------------------------------------------------------------------------- #
# 1.5a + N7 — improvement logic
# --------------------------------------------------------------------------- #

def test_best_rises_above_tol_improves():
    assert _improved(0.55, 0.5, 0.5, 0.5, tol=0.02) is True


def test_best_rises_below_tol_is_not_improvement():
    # [0,1] best up by 0.005 (< 0.02), oracle flat → NOT improving (patience ticks)
    assert _improved(0.505, 0.5, 0.5, 0.5, tol=0.02) is False


def test_oracle_rises_while_best_flat_improves():
    # N7: best FLAT, oracle up by > tol → still improving (the merge payoff)
    assert _improved(0.5, 0.5, 0.63, 0.5, tol=0.02) is True


def test_both_flat_is_not_improvement():
    assert _improved(0.5, 0.5, 0.5, 0.5, tol=0.02) is False


def test_scale_invariant_on_continuous():
    tol = 0.02 * 38  # range 38 → tol ≈ 0.76
    assert _improved(38.5, 38.0, 38.5, 38.0, tol) is False   # delta 0.5 < 0.76 → plateau
    assert _improved(39.0, 38.0, 39.0, 38.0, tol) is True    # delta 1.0 > 0.76 → improvement


# --------------------------------------------------------------------------- #
# checkpoint persists the new state (resume determinism)
# --------------------------------------------------------------------------- #

def test_checkpoint_persists_oracle_and_frozen_range(tmp_path):
    orch = _orch(patience=3, max_iterations=10)
    orch.archive.freeze_score_range(0.6)
    orch._save_checkpoint(tmp_path, 1, 0, 0.5, 100, [0.4, 0.5],
                          prev_oracle=0.7, oracle_history=[0.5, 0.7])
    ck = json.loads((tmp_path / "checkpoint.json").read_text())
    assert ck["prev_oracle"] == 0.7
    assert ck["oracle_history"] == [0.5, 0.7]
    assert ck["frozen_score_range"] == 0.6


# --------------------------------------------------------------------------- #
# integration: the stop CAN fire, and the oracle trajectory is tracked
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_empty_omega_stops_via_patience():
    orch = _orch(patience=2, max_iterations=50)
    orch.solver.solve = AsyncMock(return_value=("echo hi", "r", 10))
    orch.executor.execute = AsyncMock(return_value=Trace(
        task_id="t1", success=True, score=0.5, script="echo hi"))
    orch.omega.generate = AsyncMock(return_value=(InjectedCode(), 50))  # empty → no improvement
    result = await orch.run([TaskDescription(task_id="t1", description="x")])
    assert result.total_iterations <= 3            # stopped at patience=2, not max_iterations=50
    assert len(result.oracle_history) >= 1         # the oracle trajectory was tracked


# --------------------------------------------------------------------------- #
# N9a — run_status tag (skip budget-starved stubs)
# --------------------------------------------------------------------------- #

def _seed_run_mocks(orch):
    orch.solver.solve = AsyncMock(return_value=("echo hi", "r", 10))
    orch.executor.execute = AsyncMock(return_value=Trace(
        task_id="t1", success=True, score=0.5, script="echo hi"))
    orch.omega.generate = AsyncMock(return_value=(InjectedCode(), 50))


@pytest.mark.asyncio
async def test_run_status_completed_when_iterated():
    orch = _orch(patience=2, max_iterations=50)
    _seed_run_mocks(orch)
    result = await orch.run([TaskDescription(task_id="t1", description="x")])
    assert result.run_status == "completed"        # iterated (>=1) → completed
    assert result.to_dict()["run_status"] == "completed"


@pytest.mark.asyncio
async def test_run_status_completed_when_max_iterations_zero():
    orch = _orch(max_iterations=0)                  # seed only, loop never runs
    _seed_run_mocks(orch)
    result = await orch.run([TaskDescription(task_id="t1", description="x")])
    assert result.total_iterations == 0
    # R5: the deliberate --max-iterations 0 control (documented best-of-N
    # baseline) completes exactly as configured — aborted_pre_iteration is
    # reserved for budget-starved stubs (see test_r5_orchestrator.py).
    assert result.run_status == "completed"
