"""Refine-wave R5 regression tests (orchestrator / archive / persistence).

Findings covered:
  * verified_code non-adoption bars (``bar_from_best``) survive --resume:
    ``Archive.add`` records fired bars, ``save_checkpoint`` persists them
    (``barred_from_best``, absent when no bar ever fired), and
    ``try_resume``/``rebuild_from_disk`` re-apply them.
  * a deliberate ``--max-iterations 0`` control run stamps the same
    ``run_status`` a normal max-iterations exit gets ("completed"), while a
    budget halt before the first generation stays "aborted_pre_iteration".
  * ``total_iterations`` equals completed generations on the budget-halt and
    all-at-max-depth break paths (previously completed + 1).
  * ``save_checkpoint`` persists ``cached`` / ``cost_usd`` inside
    ``outer_cumulative_usage`` so a resume round-trips them.

All tests are LLM-free / offline (no Docker, no network).
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from meta_n.core.archive import Archive, Candidate
from meta_n.core.evolutionary_orchestrator import (
    EvolutionaryConfig,
    EvolutionaryOrchestrator,
)
from meta_n.core.meta_layer import InjectedCode, TaskDescription, Trace


def _make_orch(**overrides) -> EvolutionaryOrchestrator:
    defaults = dict(
        max_depth=3, parallel=1, patience=2, gate_tasks=0,
        beam_width=1, beam_candidates=1,
    )
    defaults.update(overrides)
    return EvolutionaryOrchestrator(
        llm_client=MagicMock(), executor=MagicMock(), omega=MagicMock(),
        config=EvolutionaryConfig(**defaults), solver_language="bash",
    )


def _seed_run_mocks(orch, score: float = 0.5) -> None:
    orch.solver.solve = AsyncMock(return_value=("echo hi", "r", 10))
    orch.executor.execute = AsyncMock(return_value=Trace(
        task_id="t1", success=True, score=score, script="echo hi"))
    orch.omega.generate = AsyncMock(return_value=(InjectedCode(), 50))


def _tasks(names) -> list[TaskDescription]:
    return [TaskDescription(task_id=n, description=f"Solve {n}") for n in names]


def _candidate(cid, scores: dict[str, float], **kwargs) -> Candidate:
    traces = [
        Trace(task_id=tid, success=True, score=s, script=f"echo {cid}-{tid}")
        for tid, s in scores.items()
    ]
    mean = sum(scores.values()) / len(scores) if scores else 0.0
    defaults = dict(
        candidate_id=cid, iteration=0, depth=1, traces=traces,
        mean_score=mean, pass_at_1=1.0, per_task_scores=dict(scores),
    )
    defaults.update(kwargs)
    return Candidate(**defaults)


# --------------------------------------------------------------------------- #
# verified_code bars survive --resume
# --------------------------------------------------------------------------- #


def test_add_records_fired_bars_in_snapshot():
    arch = Archive()
    arch.add(_candidate("gen0_seed", {"t1": 0.5, "t2": 0.4}))
    arch.add(_candidate("c1", {"t1": 0.9, "t2": 0.7}, iteration=1),
             bar_from_best={"t1"})
    assert arch.per_task_best_scores() == {"t1": 0.5, "t2": 0.7}
    assert arch.barred_from_best_snapshot() == {"c1": ["t1"]}
    # Unbarred adds leave the snapshot untouched (empty on default runs).
    arch.add(_candidate("c2", {"t2": 0.8}, iteration=2))
    assert arch.barred_from_best_snapshot() == {"c1": ["t1"]}


def test_rebuild_recovers_bar_from_candidate_summary(tmp_path):
    # The bar is co-located with the candidate (barred_from_best_task_ids in
    # summary.json), so rebuild recovers it WITHOUT the checkpoint snapshot —
    # this closes the crash-window where a candidate is on disk but the atomic
    # prior checkpoint predates its bar.
    orch = _make_orch(output_dir=str(tmp_path))
    seed = _candidate("gen0_seed", {"t1": 0.5})
    child = _candidate("gen1_b0_k0", {"t1": 0.9}, iteration=1)
    orch.archive.add(seed)
    orch.archive.add(child, bar_from_best={"t1"})
    orch._save_candidate_incremental(seed, tmp_path)
    orch._save_candidate_incremental(child, tmp_path)

    archive_dir = tmp_path / "archive"
    # No checkpoint bar supplied (simulates the stale/crash checkpoint): the
    # bar is still recovered from the candidate's own summary.json.
    recovered = Archive.rebuild_from_disk(archive_dir)
    assert recovered.per_task_best_scores()["t1"] == 0.5
    assert recovered.per_task_best_sources()["t1"] == "gen0_seed"
    assert recovered.barred_from_best_snapshot() == {"gen1_b0_k0": ["t1"]}

    # Checkpoint-supplied bar still works and unions (no double-bar / no drift).
    both = Archive.rebuild_from_disk(
        archive_dir, barred_from_best={"gen1_b0_k0": ["t1"]}
    )
    assert both.per_task_best_scores()["t1"] == 0.5
    assert both.barred_from_best_snapshot() == {"gen1_b0_k0": ["t1"]}


def test_unbarred_candidate_summary_omits_bar_key(tmp_path):
    # verified_code-OFF / bar-free candidates keep a byte-identical summary.json
    # (the additive key is absent, not empty).
    orch = _make_orch(output_dir=str(tmp_path))
    seed = _candidate("gen0_seed", {"t1": 0.5})
    orch.archive.add(seed)
    orch._save_candidate_incremental(seed, tmp_path)
    summary = json.loads(
        (tmp_path / "archive" / "gen0_seed" / "summary.json").read_text()
    )
    assert "barred_from_best_task_ids" not in summary


def test_bar_survives_checkpoint_resume_roundtrip(tmp_path):
    orch1 = _make_orch(output_dir=str(tmp_path))
    seed = _candidate("gen0_seed", {"t1": 0.5})
    child = _candidate("gen1_b0_k0", {"t1": 0.9}, iteration=1)
    orch1.archive.add(seed)
    orch1.archive.add(child, bar_from_best={"t1"})
    orch1._save_candidate_incremental(seed, tmp_path)
    orch1._save_candidate_incremental(child, tmp_path)
    orch1._save_checkpoint(tmp_path, 1, 0, 0.5, 10, [0.5])
    ck = json.loads((tmp_path / "checkpoint.json").read_text())
    assert ck["barred_from_best"] == {"gen1_b0_k0": ["t1"]}

    orch2 = _make_orch(output_dir=str(tmp_path))
    checkpoint = orch2._try_resume(tmp_path)
    assert checkpoint is not None
    assert orch2.archive.per_task_best_scores()["t1"] == 0.5
    assert orch2.archive.per_task_best_sources()["t1"] == "gen0_seed"
    # The resumed process's next checkpoint re-persists the bars (chained
    # pause/resume keeps the penalty durable).
    orch2._save_checkpoint(tmp_path, 2, 0, 0.5, 20, [0.5, 0.5])
    ck2 = json.loads((tmp_path / "checkpoint.json").read_text())
    assert ck2["barred_from_best"] == {"gen1_b0_k0": ["t1"]}


def test_checkpoint_omits_barred_key_when_no_bars(tmp_path):
    orch = _make_orch(output_dir=str(tmp_path))
    orch.archive.add(_candidate("gen0_seed", {"t1": 0.5}))
    orch._save_checkpoint(tmp_path, 1, 0, 0.5, 10, [0.5])
    ck = json.loads((tmp_path / "checkpoint.json").read_text())
    assert "barred_from_best" not in ck


def test_legacy_checkpoint_without_barred_key_resumes_fine(tmp_path):
    orch1 = _make_orch(output_dir=str(tmp_path))
    seed = _candidate("gen0_seed", {"t1": 0.5})
    child = _candidate("gen1_b0_k0", {"t1": 0.9}, iteration=1)
    orch1.archive.add(seed)
    orch1.archive.add(child)  # legacy run: no bar ever fired
    orch1._save_candidate_incremental(seed, tmp_path)
    orch1._save_candidate_incremental(child, tmp_path)
    orch1._save_checkpoint(tmp_path, 1, 0, 0.9, 10, [0.5, 0.9])
    assert "barred_from_best" not in json.loads(
        (tmp_path / "checkpoint.json").read_text()
    )

    orch2 = _make_orch(output_dir=str(tmp_path))
    checkpoint = orch2._try_resume(tmp_path)
    assert checkpoint is not None
    assert orch2.archive.per_task_best_scores()["t1"] == 0.9  # legacy behavior


# --------------------------------------------------------------------------- #
# run_status: deliberate --max-iterations 0 control vs budget-starved stub
# --------------------------------------------------------------------------- #


async def test_max_iterations_zero_control_is_completed(tmp_path):
    orch = _make_orch(output_dir=str(tmp_path), max_iterations=0)
    _seed_run_mocks(orch)
    result = await orch.run(_tasks(["t1"]))
    assert result.total_iterations == 0
    # The run terminated exactly as configured — same status as a normal
    # max-iterations exit, never the budget-starved-stub label.
    assert result.run_status == "completed"
    assert result.to_dict()["run_status"] == "completed"


async def test_budget_halt_pre_iteration_still_aborted(tmp_path):
    orch = _make_orch(output_dir=str(tmp_path), max_iterations=50, patience=2)
    _seed_run_mocks(orch)
    orch._cost_guard = SimpleNamespace(headroom_exhausted=lambda: True)
    result = await orch.run(_tasks(["t1"]))
    assert result.run_status == "aborted_pre_iteration"
    assert result.to_dict()["run_status"] == "aborted_pre_iteration"


# --------------------------------------------------------------------------- #
# total_iterations == completed generations on the two break paths
# --------------------------------------------------------------------------- #


async def test_budget_halt_pre_iteration_total_iterations_is_zero(tmp_path):
    orch = _make_orch(output_dir=str(tmp_path), max_iterations=50, patience=2)
    _seed_run_mocks(orch)
    orch._cost_guard = SimpleNamespace(headroom_exhausted=lambda: True)
    result = await orch.run(_tasks(["t1"]))
    assert result.total_iterations == 0
    # F033 contract: len(convergence_history) == completed_generations + 1.
    assert len(result.convergence_history) == result.total_iterations + 1


async def test_budget_halt_mid_total_iterations_counts_completed_only(tmp_path):
    orch = _make_orch(output_dir=str(tmp_path), max_iterations=50, patience=3)
    _seed_run_mocks(orch)
    answers = iter([False])  # iteration 1 proceeds; iteration 2 top halts
    orch._cost_guard = SimpleNamespace(
        headroom_exhausted=lambda: next(answers, True)
    )
    result = await orch.run(_tasks(["t1"]))
    assert result.run_status == "aborted_mid"
    assert result.total_iterations == 1
    assert len(result.convergence_history) == result.total_iterations + 1


async def test_max_depth_break_total_iterations_is_zero(tmp_path):
    # max_depth=1: the seed (depth 1) is not extendable, so the pool exhausts
    # at the top of iteration 1 — zero completed generations.
    orch = _make_orch(
        output_dir=str(tmp_path), max_depth=1, max_iterations=50, patience=3,
    )
    _seed_run_mocks(orch)
    result = await orch.run(_tasks(["t1"]))
    assert result.total_iterations == 0
    assert result.run_status == "completed"  # deliberate config, not a stub
    assert len(result.convergence_history) == result.total_iterations + 1


async def test_normal_exits_total_iterations_unchanged(tmp_path):
    orch = _make_orch(output_dir=str(tmp_path / "a"), max_iterations=1, patience=1)
    _seed_run_mocks(orch)
    result = await orch.run(_tasks(["t1"]))
    assert result.total_iterations == 1  # max-iterations exit
    assert result.run_status == "completed"

    orch2 = _make_orch(output_dir=str(tmp_path / "b"), max_iterations=5, patience=1)
    _seed_run_mocks(orch2)
    result2 = await orch2.run(_tasks(["t1"]))
    assert result2.total_iterations == len(result2.convergence_history) - 1
    assert result2.run_status == "completed"  # patience exit


# --------------------------------------------------------------------------- #
# outer_cumulative_usage round-trips cached / cost_usd across resume
# --------------------------------------------------------------------------- #


async def test_checkpoint_roundtrips_cached_and_cost_usd(tmp_path):
    usage = {"prompt": 1000, "completion": 500, "total": 1500, "calls": 3,
             "cached": 800, "cost_usd": 1.23}
    orch1 = _make_orch(output_dir=str(tmp_path), max_iterations=1, patience=1)
    _seed_run_mocks(orch1)
    orch1.llm_client.cumulative_usage = dict(usage)
    await orch1.run(_tasks(["t1"]))
    ck = json.loads((tmp_path / "checkpoint.json").read_text())
    assert ck["outer_cumulative_usage"] == usage

    orch2 = _make_orch(
        output_dir=str(tmp_path), max_iterations=1, patience=1,
        no_early_stop=True,
    )
    _seed_run_mocks(orch2)
    orch2.llm_client.cumulative_usage = {}
    await orch2.run(_tasks(["t1"]), resume=True)
    assert orch2.llm_client.cumulative_usage == usage


# --------------------------------------------------------------------------- #
# read_agent_run_rows never raises on malformed telemetry envelopes
# --------------------------------------------------------------------------- #


def test_read_agent_run_rows_skips_malformed_envelopes(tmp_path):
    orch = _make_orch(output_dir=str(tmp_path))
    tdir = tmp_path / "telemetry"
    tdir.mkdir()
    lines = [
        json.dumps({"extra": {"record": {
            "run_id": "r1", "candidate_id": "c1", "success": True,
        }}}),
        json.dumps([1, 2]),                          # non-object top level
        json.dumps({"extra": "x", "run_id": "r2"}),  # non-dict extra → top-level
        json.dumps({"extra": {"record": [1]}}),      # non-dict record → skipped
        json.dumps({"extra": {"record": None}}),     # null record → skipped
    ]
    (tdir / "agent_runs.jsonl").write_text("\n".join(lines) + "\n")

    rows = orch._read_agent_run_rows()
    assert {r.get("run_id") for r in rows} == {"r1", "r2"}

    # The seam that crashed mid-run: the candidate save's telemetry rollup
    # now survives the malformed ledger.
    orch._save_candidate_incremental(_candidate("c1", {"t1": 0.5}), tmp_path)
    summary = json.loads(
        (tmp_path / "archive" / "c1" / "summary.json").read_text()
    )
    assert summary["agent_telemetry"]["runs"] == 1
