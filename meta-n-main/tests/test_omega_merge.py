"""Ω_merge tests (Step 4): per-task output channel (4.2) + deployable oracle (4.1)."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from meta_n.core.archive import Archive, Candidate
from meta_n.core.evolutionary_orchestrator import (
    EvolutionaryConfig,
    EvolutionaryOrchestrator,
    EvolutionaryResult,
)
from meta_n.core.meta_layer import InjectedCode, MetaLayer, TaskDescription, Trace


# --------------------------------------------------------------------------- #
# 4.2 — per-task output channel
# --------------------------------------------------------------------------- #

def test_task_solution_map_makes_injected_code_nonempty():
    assert InjectedCode().is_empty is True
    assert InjectedCode(task_solution_map={"t1": "echo X"}).is_empty is False


def test_task_solution_map_roundtrips():
    ic = InjectedCode(task_solution_map={"t1": "echo X", "t2": "echo Y"})
    ic2 = InjectedCode.model_validate(ic.model_dump())
    assert ic2.task_solution_map == {"t1": "echo X", "t2": "echo Y"}


def test_legacy_injected_code_without_map_roundtrips():
    ic = InjectedCode.model_validate({"pre_process": "x=1", "source_depth": 2})
    assert ic.task_solution_map == {}  # additive default → legacy JSON loads


@pytest.mark.asyncio
async def test_metalayer_short_circuits_frozen_output():
    inner = MagicMock()

    async def execute(script, task, timeout=30):
        return Trace(task_id=task.task_id, success=True, score=0.9, script=script)

    executor = MagicMock()
    executor.execute = execute
    ic = InjectedCode(task_solution_map={"task_a": "FROZEN_SCRIPT"})
    layer = MetaLayer(depth=2, injected_code=ic, inner_solver=inner, executor=executor)

    trace, tokens = await layer.execute(TaskDescription(task_id="task_a", description="x"))
    assert tokens == 0
    assert trace.script == "FROZEN_SCRIPT"   # ran the frozen winner
    assert trace.score == 0.9
    inner.solve.assert_not_called()           # inner solver NEVER invoked — no re-solve


@pytest.mark.asyncio
async def test_metalayer_solve_short_circuits_frozen_output():
    """solve() must route to the frozen winner SYMMETRICALLY with execute(), so
    the merged oracle deploys correctly on the solve() path (the classification
    held-out chain-test re-solve) and not only on execute() (CO-Bench). Fixes the
    W5 known limitation."""
    inner = MagicMock()
    inner.solve = AsyncMock(return_value=("RESOLVED_SCRIPT", "r", 99))
    layer = MetaLayer(
        depth=2,
        injected_code=InjectedCode(task_solution_map={"task_a": "FROZEN_SCRIPT"}),
        inner_solver=inner, executor=MagicMock(),
    )

    # Mapped task → returns the frozen winner script, zero tokens, no re-solve.
    script, _, tokens = await layer.solve(TaskDescription(task_id="task_a", description="x"))
    assert script == "FROZEN_SCRIPT"
    assert tokens == 0
    inner.solve.assert_not_called()

    # Unmapped task → falls through to the inner solver (re-solve), unchanged.
    script2, _, tokens2 = await layer.solve(TaskDescription(task_id="task_b", description="y"))
    assert script2 == "RESOLVED_SCRIPT"
    assert tokens2 == 99
    inner.solve.assert_called_once()


# --------------------------------------------------------------------------- #
# 4.1 — deployable oracle by assembly
# --------------------------------------------------------------------------- #

def _orch(adapter=None):
    orch = EvolutionaryOrchestrator(
        llm_client=MagicMock(), executor=MagicMock(), omega=MagicMock(),
        config=EvolutionaryConfig(max_depth=3, parallel=1, gate_tasks=0),
        solver_language="bash",
    )
    orch.adapter = adapter
    return orch


def _tasks(n):
    return [TaskDescription(task_id=f"task_{i}", description="x") for i in range(n)]


def _result_with_oracle(archive, tasks):
    r = EvolutionaryResult()
    ptb = archive.per_task_best_scores()
    r.oracle_mean_score = sum(ptb.get(t.task_id, 0.0) for t in tasks) / len(tasks)
    return r


def test_merge_assembles_oracle(make_disjoint_archive):
    orch = _orch(adapter=None)             # adapter None → kind 'unit' (not binary)
    orch.archive = make_disjoint_archive(3)
    tasks = _tasks(3)
    result = _result_with_oracle(orch.archive, tasks)
    merged = orch._build_merged_candidate(tasks, result)
    assert merged is not None
    assert merged.candidate_id == "merge_oracle"
    assert merged.mean_score == pytest.approx(0.9)            # the oracle mean
    assert merged.per_task_scores == orch.archive.per_task_best_scores()
    assert merged.parent_id is None
    assert merged.injected_codes[0].task_solution_map         # routes every task
    assert orch.archive.best_candidate.candidate_id == "merge_oracle"  # now the best


def test_merge_excluded_from_breedable_pool(make_disjoint_archive):
    orch = _orch(adapter=None)
    orch.archive = make_disjoint_archive(3)
    tasks = _tasks(3)
    orch._build_merged_candidate(tasks, _result_with_oracle(orch.archive, tasks))
    pool_ids = {c.candidate_id for c in orch.archive.breedable_pool(max_depth=3)}
    assert "merge_oracle" not in pool_ids   # synthetic oracle is never a parent


def test_merge_fires_on_binary_disjoint_wins():
    # Y4-P_benchmarks-4: a binary scale does NOT imply oracle == best — two
    # chains with disjoint {0,1} wins give oracle 1.0 vs best 0.5, so the
    # merge must assemble the deployable oracle there too (no score-scale
    # ``kind`` gate; the oracle==best degeneracy is ``len(sources) <= 1``).
    class _Bin:
        def score_scale(self):
            return {"kind": "binary"}

    orch = _orch(adapter=_Bin())
    arch = Archive()
    arch.add(Candidate(candidate_id="c0", mean_score=0.5, per_task_scores={"t0": 1.0, "t1": 0.0},
                       traces=[Trace(task_id="t0", success=True, score=1.0, script="s"),
                               Trace(task_id="t1", success=False, score=0.0, script="s")]))
    arch.add(Candidate(candidate_id="c1", mean_score=0.5, per_task_scores={"t0": 0.0, "t1": 1.0},
                       traces=[Trace(task_id="t0", success=False, score=0.0, script="s"),
                               Trace(task_id="t1", success=True, score=1.0, script="s")]))
    orch.archive = arch
    tasks = [TaskDescription(task_id="t0", description="x"), TaskDescription(task_id="t1", description="x")]
    result = EvolutionaryResult()
    result.oracle_mean_score = 1.0  # oracle (1.0) > best (0.5): disjoint wins
    merged = orch._build_merged_candidate(tasks, result)
    assert merged is not None
    assert merged.candidate_id == "merge_oracle"
    assert merged.mean_score == pytest.approx(1.0)


def test_merge_no_op_on_single_task(make_candidate):
    orch = _orch(adapter=None)
    orch.archive = Archive()
    orch.archive.add(make_candidate("c0", mean_score=0.5, tasks=["only"]))
    result = EvolutionaryResult()
    result.oracle_mean_score = 0.5
    assert orch._build_merged_candidate([TaskDescription(task_id="only", description="x")], result) is None


def test_merge_roundtrips_through_rebuild(make_disjoint_archive, tmp_path, write_candidate_dir):
    orch = _orch(adapter=None)
    orch.archive = make_disjoint_archive(3)
    tasks = _tasks(3)
    orch._build_merged_candidate(tasks, _result_with_oracle(orch.archive, tasks))
    archive_dir = tmp_path / "archive"
    for c in orch.archive.candidates:
        write_candidate_dir(archive_dir, c)
    rebuilt = Archive.rebuild_from_disk(archive_dir)
    m = rebuilt.get("merge_oracle")
    assert m.parent_id is None
    assert m.injected_codes[0].task_solution_map   # routing map survives resume


def test_merge_result_to_dict_conditional():
    # merge_candidate_id only surfaces in summary.json when a merge fired
    r = EvolutionaryResult(archive_size=1, best_candidate_id="c0")
    assert "merge_candidate_id" not in r.to_dict()
    r.merge_candidate_id = "merge_oracle"
    assert r.to_dict()["merge_candidate_id"] == "merge_oracle"
