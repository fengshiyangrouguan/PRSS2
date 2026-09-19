"""G9 targeted per-task consolidation (--consolidate).

Proves the mechanism that fixes the "improve A, break B" thrash: each candidate
improves ONE target task while INHERITING every other task's per-task-best frozen
score (no re-solve), so gains are monotonic and collateral-free, and the
deployable Ω_merge oracle is always materialized. Also the W4 offline validation
(no GPU / no LLM): the oracle never regresses even when the target solve is noisy.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from meta_n.core.archive import Candidate
from meta_n.core.evolutionary_orchestrator import (
    EvolutionaryConfig,
    EvolutionaryOrchestrator,
    EvolutionaryResult,
)
from meta_n.core.meta_layer import InjectedCode, TaskDescription, Trace


def _orch(**cfg) -> EvolutionaryOrchestrator:
    d = dict(max_depth=20, parallel=1, patience=8, gate_tasks=0,
             beam_width=1, beam_candidates=1)
    d.update(cfg)
    return EvolutionaryOrchestrator(
        llm_client=MagicMock(), executor=MagicMock(), omega=MagicMock(),
        config=EvolutionaryConfig(**d), solver_language="bash",
    )


def _trace(tid: str, score: float, inner_tokens: int = 0) -> Trace:
    return Trace(task_id=tid, depth=1, script=f"def solve(**k): pass  # {tid}",
                 success=True, score=score, inner_tokens=inner_tokens)


def _cand(cid: str, scores: dict[str, float], depth: int = 1,
          parent_id: str | None = None) -> Candidate:
    traces = [_trace(t, s) for t, s in scores.items()]
    return Candidate(
        candidate_id=cid, parent_id=parent_id, iteration=0, depth=depth,
        injected_codes=[], traces=traces, pass_at_1=1.0,
        mean_score=sum(scores.values()) / len(scores), per_task_scores=dict(scores),
    )


def _tasks(names):
    return [TaskDescription(task_id=n, description=f"Solve {n}") for n in names]


# --- unit: target selection -------------------------------------------------

def test_consolidation_targets_round_robin_covers_all_tasks():
    orch = _orch()
    tasks = _tasks(["t1", "t2", "t3"])
    picked = [orch._consolidation_targets(tasks, 1, it)[0] for it in range(3)]
    assert set(picked) == {"t1", "t2", "t3"}        # every task gets a turn
    # n>1 per generation returns distinct targets
    two = orch._consolidation_targets(tasks, 2, 0)
    assert len(two) == 2 and len(set(two)) == 2


# --- unit: inherit-frozen precomputed --------------------------------------

def test_consolidation_precomputed_excludes_only_the_target():
    orch = _orch()
    tasks = _tasks(["t1", "t2", "t3"])
    orch.archive.add(_cand("gen0_seed", {"t1": 0.2, "t2": 0.5, "t3": 0.9}))
    # stamp inner cost on the stored bests to prove the copy zeroes it (W5 HIGH).
    for tr in orch.archive.per_task_best_traces().values():
        tr.inner_tokens = 50
    pre = orch._consolidation_precomputed("t2", tasks, child_depth=2)
    assert set(pre) == {"t1", "t3"}                  # target t2 is NOT frozen
    assert pre["t1"].score == 0.2 and pre["t3"].score == 0.9
    assert all(pre[t].script for t in pre)
    # W5 fixes: returned traces are deep copies (not the archive's objects) with
    # inner cost zeroed; the archive's own records are untouched.
    stored = orch.archive.per_task_best_traces()
    assert pre["t1"] is not stored["t1"]
    assert pre["t1"].inner_tokens == 0
    assert stored["t1"].inner_tokens == 50


# --- unit: forced merge fires even on a sub-threshold gap -------------------

def test_force_materializes_oracle_when_default_gate_would_skip():
    orch = _orch(consolidate=True)
    tasks = _tasks(["t1", "t2", "t3"])
    # per-task bests scattered across candidates; oracle just 0.02 over best.
    orch.archive.add(_cand("gen0_seed", {"t1": 0.50, "t2": 0.50, "t3": 0.50}))
    orch.archive.add(_cand("c1", {"t1": 0.56, "t2": 0.50, "t3": 0.50}, depth=2))
    orch.archive.add(_cand("c2", {"t1": 0.50, "t2": 0.56, "t3": 0.50}, depth=2))
    # A low candidate widens score_range (max-min of means) WITHOUT changing the
    # oracle (0.54) or best mean (0.52), so the 0.02 gap is genuinely under the
    # 0.05*range threshold.
    orch.archive.add(_cand("c_low", {"t1": 0.0, "t2": 0.0, "t3": 0.0}, depth=2))
    oracle = sum(orch.archive.per_task_best_scores().values()) / 3   # 0.54
    result = EvolutionaryResult(best_mean_score=orch.archive.best_mean_score)
    result.oracle_mean_score = oracle
    result.total_iterations = 2

    # Construct a genuinely sub-threshold gap relative to the archive's own
    # score_range, so the DEFAULT gate would skip it but force overrides.
    gap = oracle - orch.archive.best_mean_score
    threshold = 0.05 * orch.archive.score_range()
    assert gap <= threshold, f"gap {gap} not sub-threshold ({threshold})"
    assert orch._build_merged_candidate(tasks, result, force=False) is None
    merged = orch._build_merged_candidate(tasks, result, force=True)
    assert merged is not None
    assert merged.mean_score == pytest.approx(oracle)
    assert len(merged.injected_codes[0].task_solution_map) == 3   # routes all tasks


# --- integration (W4): monotonic + collateral-free, no GPU/LLM --------------

@pytest.mark.asyncio
async def test_consolidate_run_is_monotonic_and_collateral_free():
    """A noisy target solve (sometimes WORSE than the inherited best) must never
    drop any task's per-task-best — the collateral-free guarantee — and only the
    target is ever re-solved."""
    orch = _orch(consolidate=True, beam_candidates=1, patience=6, max_depth=20)
    tasks = _tasks(["t1", "t2", "t3"])

    seed = {"t1": 0.20, "t2": 0.50, "t3": 0.90}
    # Per-task solve sequence: index 0 is the seed eval; later indices are the
    # consolidation re-solves — deliberately noisy (some BELOW the seed) so the
    # no-regression guarantee is actually exercised.
    seqs = {
        "t1": [0.20, 0.10, 0.65, 0.05],
        "t2": [0.50, 0.55, 0.30, 0.60],
        "t3": [0.90, 0.40, 0.95, 0.20],
    }
    calls = {t: 0 for t in seqs}
    solved_tasks: list[str] = []

    async def execute(script, task, timeout=30):
        i = calls[task.task_id]
        calls[task.task_id] += 1
        solved_tasks.append(task.task_id)
        s = seqs[task.task_id][min(i, len(seqs[task.task_id]) - 1)]
        # 100 inner tokens per FRESH solve — used to prove inherited tasks add 0.
        return Trace(task_id=task.task_id, success=True, score=s, script=script,
                     inner_tokens=100)

    orch.solver.solve = AsyncMock(return_value=("script", "r", 5))
    orch.executor.execute = execute
    orch.omega.generate = AsyncMock(return_value=(InjectedCode(pre_process="x=1"), 5))

    result = await orch.run(tasks)

    ptb = orch.archive.per_task_best_scores()
    # 1. No task's per-task-best ever drops below the seed (collateral-free).
    for t, s in seed.items():
        assert ptb[t] >= s - 1e-9, f"{t} regressed below seed: {ptb[t]} < {s}"
    # 2. The oracle history is non-decreasing (monotonic consolidation).
    hist = result.oracle_history or []
    assert all(b >= a - 1e-9 for a, b in zip(hist, hist[1:])), hist
    # 3. Each consolidation candidate solved EXACTLY its one target task — the
    #    seed solved all 3; every later candidate added exactly one solve. This
    #    is the proof that non-target tasks are never re-rolled.
    consol_cands = [
        c for c in orch.archive.candidates
        if c.candidate_id not in ("gen0_seed", result.merge_candidate_id)
    ]
    assert len(consol_cands) >= 1
    assert len(solved_tasks) == 3 + len(consol_cands)
    # 4. W5 HIGH: each consolidation candidate charges inner-LLM cost for its ONE
    #    fresh focus solve only (100), NOT the 2 inherited tasks (would be 300
    #    before the fix). Proves inherited traces add zero inner cost.
    for c in consol_cands:
        assert c.inner_tokens == 100, (c.candidate_id, c.inner_tokens)
    # 5. The deployable oracle was materialized (forced merge).
    assert result.merge_candidate_id is not None


# --- W3: per-task protection floor (6.2) ------------------------------------

@pytest.mark.asyncio
async def test_protect_floor_vetoes_regression_even_when_another_clears():
    """The E1 bug: the gate admitted a candidate that tanked capac_WH (0.198 <
    baseline 0.614) because another task cleared first. With the floor on, a hard
    per-task regression vetoes the candidate; with it off, the legacy
    pass-on-first-clear still admits it."""
    names = ["t1", "t2", "t3", "t4"]
    tasks = _tasks(names)
    by_id = {t.task_id: t for t in tasks}
    parent = _cand("parent", {n: 0.60 for n in names})
    child = _cand("child", {n: 0.60 for n in names}, depth=2, parent_id="parent")

    async def gate_solve(solver, task, candidate, repeat_index=0):
        # t1 improves (clears); t2 collapses far below the floor.
        return _trace(task.task_id, 0.90 if task.task_id == "t1" else 0.20)

    async def _run(protect):
        orch = _orch(gate_tasks=4, gate_margin=0.0, protect_floor=protect)
        orch._gate_solve_one = gate_solve
        orch.rng = MagicMock()
        orch.rng.sample = lambda pop, k: [by_id["t1"], by_id["t2"]]
        passed, _, _ = await orch._gate_check(child, parent, MagicMock(), tasks)
        return passed

    assert await _run(0.05) is False    # 6.2: t2 regression vetoes the candidate
    assert await _run(None) is True     # legacy: t1 clears first → admitted (the bug)


@pytest.mark.asyncio
async def test_consolidate_empty_injection_resamples_target_not_skip():
    """An empty Ω injection in consolidate mode must still re-solve the FOCUS task
    (a fresh draw), not skip the candidate — otherwise an unlucky frozen score is
    locked in (the E2-G9 aircraft_landing bug: its one focus generation returned
    empty, was skipped, and the task stayed at the seed's unlucky 0.318)."""
    orch = _orch(consolidate=True, beam_candidates=1, patience=4, max_depth=20)
    tasks = _tasks(["t1", "t2", "t3"])
    solved: list[str] = []

    async def execute(script, task, timeout=30):
        solved.append(task.task_id)
        return Trace(task_id=task.task_id, success=True, score=0.5, script=script)

    orch.solver.solve = AsyncMock(return_value=("script", "r", 5))
    orch.executor.execute = execute
    # Ω returns EMPTY every time — the case that used to skip every candidate.
    orch.omega.generate = AsyncMock(return_value=(InjectedCode(), 5))

    result = await orch.run(tasks)

    # Consolidation candidates were still ADDED (resampled), not all skipped.
    consol = [
        c for c in orch.archive.candidates
        if c.candidate_id not in ("gen0_seed", result.merge_candidate_id)
    ]
    assert len(consol) >= 1, "empty Ω in consolidate mode skipped every candidate (the bug)"
    # And the target was actually re-solved beyond the seed's 3 solves.
    assert len(solved) > 3
