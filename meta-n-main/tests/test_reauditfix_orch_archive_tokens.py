"""Re-audit regression tests for the orchestrator + archive token/estimator fixes.

Each test FAILS on the pre-fix code and PASSES after the fix. Everything here is
LLM-free / offline (no LM Studio, Docker, or network); no model- or Ω-generated
code is executed on the host.

Findings covered (ids from .audit/reaudit_findings.json):
  1  — score_ceiling fix was INERT: the orchestrator never plumbed the bound
       adapter's ``score_scale()['hi']`` into the four Archive constructions
       (incl. ``rebuild_from_disk``), so ``_fresh_headroom_signal`` kept a
       hardcoded 1.0 ceiling on continuous scales.
  6  — gate-FAIL inner-LLM tokens were silently dropped from ``result.inner_*``
       (only the OUTER gate channel was reconciled by audit #29).
 10  — ``omega_tokens`` double-counted into ``result.total_tokens`` on the
       consolidate empty-injection focus-resample fall-through.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from meta_n.core.archive import Archive, Candidate
from meta_n.core.evolutionary_orchestrator import (
    EvolutionaryConfig,
    EvolutionaryOrchestrator,
)
from meta_n.core.meta_layer import InjectedCode, TaskDescription, Trace


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _orch(*, hi=1.0, executor=None, **cfg) -> EvolutionaryOrchestrator:
    """A minimal orchestrator with MagicMock deps (no LLM / Docker).

    ``hi`` sets the bound adapter's ``score_scale()['hi']`` (the benchmark score
    ceiling): 1.0 is a unit/binary scale, ``None`` a continuous/unbounded scale.
    """
    if executor is None:
        executor = MagicMock()
        executor.adapter.score_scale.return_value = {"lo": 0.0, "hi": hi}
    defaults = dict(
        max_depth=4, parallel=1, patience=1, gate_tasks=0,
        beam_width=1, beam_candidates=1,
    )
    defaults.update(cfg)
    return EvolutionaryOrchestrator(
        llm_client=MagicMock(), executor=executor, omega=MagicMock(),
        config=EvolutionaryConfig(**defaults), solver_language="bash",
    )


def _trace(task_id, *, success=True, score=1.0, depth=2, inner_tokens=0,
           inner_calls=0, inner_prompt_tokens=0, inner_completion_tokens=0):
    return Trace(
        task_id=task_id, depth=depth, script="s", success=success, score=score,
        error_summary="" if success else "err",
        inner_tokens=inner_tokens, inner_calls=inner_calls,
        inner_prompt_tokens=inner_prompt_tokens,
        inner_completion_tokens=inner_completion_tokens,
    )


def _tasks(names):
    return [TaskDescription(task_id=n, description=f"Solve {n}") for n in names]


def _best_holding_candidate(task_id, score):
    """A depth-2 candidate whose one FRESH task holds the per-task-best at
    ``score`` (its trace depth == candidate depth)."""
    return Candidate(
        candidate_id="c", parent_id="gen0_seed", iteration=1, depth=2,
        traces=[_trace(task_id, score=score, depth=2)],
        per_task_scores={task_id: score}, mean_score=score,
    )


# --------------------------------------------------------------------------- #
# Finding 1 — score_ceiling plumbed from the adapter's score_scale()['hi']
# --------------------------------------------------------------------------- #

def test_finding1_continuous_scale_drops_ceiling_in_archive():
    """On a continuous scale (hi=None) the orchestrator must plumb ``None`` into
    the Archive so ``_fresh_headroom_signal`` DROPS the absolute-ceiling clause.
    Pre-fix the constructor never passed ``score_ceiling`` → it stayed the
    hardcoded default 1.0.
    """
    orch = _orch(hi=None, within_task_recursion=True, consolidate=True)
    assert orch.archive._score_ceiling is None  # pre-fix: 1.0
    # depth bonus must be live for the ceiling to matter at all.
    assert orch.archive.within_task_depth_bonus > 0


def test_finding1_unit_scale_keeps_ceiling_1_0():
    """A unit / binary scale (hi=1.0) must keep the ceiling at 1.0 — byte-
    identical to the historical hardcoded value (the fix must not over-correct
    every scale to None)."""
    orch = _orch(hi=1.0, within_task_recursion=True, consolidate=True)
    assert orch.archive._score_ceiling == 1.0


def test_finding1_headroom_signal_scale_aware_behavioral():
    """The wiring must change actual behavior: a fresh best-holding task scoring
    0.5 has NO headroom on a continuous scale (ceiling dropped → 0.0) but DOES on
    a unit scale (0.5 < 1.0 ceiling → 1.0). Pre-fix BOTH returned 1.0 because the
    continuous archive still carried the 1.0 default.
    """
    cont = _orch(hi=None, within_task_recursion=True, consolidate=True)
    unit = _orch(hi=1.0, within_task_recursion=True, consolidate=True)
    cand_c = _best_holding_candidate("t0", 0.5)
    cand_u = _best_holding_candidate("t0", 0.5)
    cont.archive.add(cand_c)
    unit.archive.add(cand_u)
    # continuous: ceiling dropped → saturated (best-holder, no lag) → 0.0
    assert cont.archive._fresh_headroom_signal(cand_c) == 0.0  # pre-fix: 1.0
    # unit: 0.5 < 1.0 ceiling → headroom (unchanged legacy behavior)
    assert unit.archive._fresh_headroom_signal(cand_u) == 1.0


def test_finding1_rebuild_from_disk_accepts_and_restores_ceiling(tmp_path):
    """``Archive.rebuild_from_disk`` must accept ``score_ceiling`` so the resume
    path restores it. Pre-fix the method did not declare the parameter →
    ``TypeError`` (the resume-fallback plumbing was impossible).
    """
    archive = Archive.rebuild_from_disk(
        tmp_path / "nonexistent_archive", score_ceiling=None,
    )
    assert archive._score_ceiling is None  # pre-fix: TypeError (unexpected kwarg)


# --------------------------------------------------------------------------- #
# Finding 6 — gate-FAIL inner-LLM tokens reach result.inner_*
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_finding6_gate_fail_inner_tokens_counted(tmp_path):
    """A gate-REJECTED child's gate solve is a real INNER-LLM spend on solve()-
    heavy benchmarks. On gate-PASS those inner tokens flow via the reused
    precomputed traces; on gate-FAIL the candidate is never evaluated, so pre-fix
    they vanished from ``result.inner_*`` (only the OUTER channel was reconciled).
    """
    orch = _orch(
        output_dir=str(tmp_path), gate_tasks=1, gate_margin=0.0,
        max_iterations=1, patience=1,
    )
    orch.adapter = None
    tasks = _tasks(["t0"])

    # Seed: native depth-1 solve, 0 tokens, strong baseline so the child gate
    # (low score) FAILS — exercising the gate-fail accounting path.
    orch.solver.solve = AsyncMock(return_value=("s", "r", 0))
    orch.executor.execute = AsyncMock(
        return_value=_trace("t0", success=True, score=0.8, inner_tokens=0)
    )
    # Ω: one non-empty injection (depth-2 child), 50 outer tokens.
    orch.omega.generate = AsyncMock(
        return_value=(InjectedCode(pre_process="echo hi"), 50)
    )

    async def fake_child_execute(task):
        # The child's gate solve fails the gate (0.2 < 0.8) but spent real inner
        # tokens on its per-instance llm() calls.
        return (
            _trace(
                "t0", success=True, score=0.2,
                inner_tokens=77, inner_calls=5,
                inner_prompt_tokens=60, inner_completion_tokens=17,
            ),
            0,
        )

    fake_solver = MagicMock()
    fake_solver.execute = fake_child_execute
    orch._build_solver_from_candidate = MagicMock(return_value=fake_solver)

    result = await orch.run(tasks)

    # Seed inner == 0; only the gate-rejected child's gate solve spent inner
    # tokens. Pre-fix result.inner_tokens stayed 0.
    assert result.inner_tokens == 77
    assert result.inner_prompt_tokens == 60
    assert result.inner_completion_tokens == 17
    assert result.inner_calls == 5


# --------------------------------------------------------------------------- #
# Finding 10 — omega_tokens counted ONCE on the empty-injection focus resample
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_finding10_empty_injection_focus_resample_counts_omega_once(tmp_path):
    """In consolidate mode an EMPTY Ω injection falls through to a plain resample
    of the focus task; the child carries ``total_tokens=omega_tokens`` and is
    booked once via ``result.total_tokens += child.total_tokens``. Pre-fix
    ``omega_tokens`` was ALSO pre-added unconditionally, double-counting it on
    every empty-injection focus attempt.
    """
    OMEGA = 13
    orch = _orch(
        output_dir=str(tmp_path), consolidate=True, beam_candidates=1,
        patience=4, max_depth=20,
    )
    orch.adapter = None
    # Deterministic outer-token reconciliation no-op (mock client has no real
    # cumulative_usage): keep result.total_tokens as accumulated.
    orch.llm_client.cumulative_usage = {}
    tasks = _tasks(["t1", "t2", "t3"])

    # Every solve spends 0 OUTER tokens, so the ONLY outer-token source is Ω.
    orch.solver.solve = AsyncMock(return_value=("script", "r", 0))

    async def _execute(script, task, timeout=30):
        return Trace(task_id=task.task_id, success=True, score=0.5, script=script)

    orch.executor.execute = _execute
    # Ω returns EMPTY every time with OMEGA outer tokens → the focus fall-through.
    orch.omega.generate = AsyncMock(return_value=(InjectedCode(), OMEGA))

    result = await orch.run(tasks)

    n_omega = orch.omega.generate.await_count
    assert n_omega >= 1  # at least one empty-injection focus attempt occurred
    # Each empty focus attempt books OMEGA exactly once (via child.total_tokens);
    # the seed spent 0 outer tokens. Pre-fix this was 2 * OMEGA * n_omega.
    assert result.total_tokens == OMEGA * n_omega
