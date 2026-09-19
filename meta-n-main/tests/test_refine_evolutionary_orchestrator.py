"""Regression tests for the orchestrator refine wave (clusters C10a/C10b, spec B4).

Findings covered (C10a fixes):
  F028 — dead ``completed_candidates`` checkpoint key removed (legacy
         checkpoints carrying it stay loadable)
  F029 — cost-guard budget halt stamps ``run_status`` honestly
         (``aborted_mid`` / ``aborted_pre_iteration``, never ``completed``)
  F030 — ``_gate_solve_one`` delegates to ``_eval_solve_once`` (gate traces
         get the same stamping as eval traces)
  F033 — budget halt exit classifier logs a budget message, not "max depth"
  F036 — Ω_merge candidate is persisted by run() and never aliases the
         archive's per-task-best traces
  F037 — ``oracle_mean_score`` uses the full task-set denominator (the
         semantic main.py's console line must display)
  F052 — run() restores the ``meta_n`` logger level and rotates LLM-IO
         JSONLs on a FAILED resume
Findings covered (C10b consolidations):
  F040 — ``_archive_kwargs`` is the single source for every Archive
         construction / rebuild
  F044 — ``_score_scale_hi`` is the single "finite numeric hi" extraction
  F045/F195 — persisted trace extensions follow ``detect_script_language``
         (helper + frozen-tuple contract test live in meta_layer / its tests)
  F046 — ``_persist_progress`` keeps summary.json + checkpoint.json paired
  F051 — eval-repeats inner-token accounting goes through the shared
         ``_sum_inner_traces`` staticmethod
  F192 — the two agent-telemetry rollups share ``_aggregate_agent_rows``
         with frozen persisted key order
  F200 — ``_format_test_eval_status`` pins the [test]/[chain-test] cells
Wave-1 handoffs:
  F019 Part A (writer) — per-candidate inner-LLM accounting in summary.json
  F042 step 2 — ``_uses_external_spine`` delegates to spine_routing

All tests are LLM-free / offline (no Docker, no network).
"""

import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from meta_n.core.archive import Archive, Candidate
from meta_n.core.evolutionary_orchestrator import (
    _WITHIN_TASK_DEPTH_BETA,
    EvolutionaryConfig,
    EvolutionaryOrchestrator,
    EvolutionaryResult,
    _score_scale_hi,
)
from meta_n.core.meta_layer import InjectedCode, TaskDescription, Trace

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _make_orch(**overrides) -> EvolutionaryOrchestrator:
    defaults = dict(
        max_depth=3,
        parallel=1,
        patience=2,
        gate_tasks=0,
        beam_width=1,
        beam_candidates=1,
    )
    defaults.update(overrides)
    return EvolutionaryOrchestrator(
        llm_client=MagicMock(),
        executor=MagicMock(),
        omega=MagicMock(),
        config=EvolutionaryConfig(**defaults),
        solver_language="bash",
    )


def _seed_run_mocks(orch, score: float = 0.5) -> None:
    orch.solver.solve = AsyncMock(return_value=("echo hi", "r", 10))
    orch.executor.execute = AsyncMock(return_value=Trace(
        task_id="t1", success=True, score=score, script="echo hi"))
    orch.omega.generate = AsyncMock(return_value=(InjectedCode(), 50))


def _tasks(names) -> list[TaskDescription]:
    return [TaskDescription(task_id=n, description=f"Solve {n}") for n in names]


def _candidate(cid, scores: dict[str, float], scripts: dict[str, str] | None = None,
               **kwargs) -> Candidate:
    scripts = scripts or {}
    traces = [
        Trace(task_id=tid, success=True, score=s,
              script=scripts.get(tid, f"echo {cid}-{tid}"))
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
# F028 — completed_candidates checkpoint snapshot removed
# --------------------------------------------------------------------------- #


def test_checkpoint_omits_completed_candidates_snapshot(tmp_path):
    orch = _make_orch()
    orch._save_checkpoint(tmp_path, 1, 0, 0.5, 10, [0.5], result=EvolutionaryResult())
    ck = json.loads((tmp_path / "checkpoint.json").read_text())
    assert "completed_candidates" not in ck
    # The load-bearing resume keys are still present.
    for key in ("iteration", "rng_state", "outer_cumulative_usage", "inner_tokens"):
        assert key in ck


async def test_resume_tolerates_legacy_checkpoint_with_snapshot_key(tmp_path):
    """A checkpoint that still carries the legacy key must resume cleanly
    (resume derives completed work from the disk archive, never the key)."""
    out = str(tmp_path)

    def build(**kw):
        # The resumed run sets no_early_stop: the first run converged with
        # patience_counter=1, so this lets the loop re-enter (the test is
        # about checkpoint tolerance, not the converged-resume no-op).
        orch = _make_orch(output_dir=out, max_iterations=1, patience=1, **kw)
        _seed_run_mocks(orch)
        return orch

    tasks = _tasks(["t1"])
    res1 = await build().run(tasks)
    assert res1.run_status == "completed"

    # Re-inject the legacy snapshot key into the production checkpoint.
    ck_path = tmp_path / "checkpoint.json"
    ck = json.loads(ck_path.read_text())
    assert "completed_candidates" not in ck  # F028: writer no longer emits it
    ck["completed_candidates"] = ["gen0_seed"]
    ck_path.write_text(json.dumps(ck, indent=2))

    res2 = await build(no_early_stop=True).run(tasks, resume=True)
    assert res2.run_status == "completed"
    # The resume path was actually taken (state restored from checkpoint,
    # not a fresh-start fallback): the restored token total carries over.
    assert res2.total_tokens >= res1.total_tokens


# --------------------------------------------------------------------------- #
# F029 — run_status on cost-guard budget halt
# --------------------------------------------------------------------------- #


async def test_budget_halt_before_first_generation_is_aborted_pre_iteration(tmp_path):
    orch = _make_orch(output_dir=str(tmp_path), max_iterations=50, patience=2)
    _seed_run_mocks(orch)
    orch._cost_guard = SimpleNamespace(headroom_exhausted=lambda: True)
    result = await orch.run(_tasks(["t1"]))
    # Halt fired at the top of iteration 1: zero completed generations.
    assert result.run_status == "aborted_pre_iteration"
    assert result.to_dict()["run_status"] == "aborted_pre_iteration"


async def test_budget_halt_after_one_generation_is_aborted_mid(tmp_path):
    orch = _make_orch(output_dir=str(tmp_path), max_iterations=50, patience=3)
    _seed_run_mocks(orch)
    answers = iter([False])  # iteration 1 proceeds; iteration 2 top halts
    orch._cost_guard = SimpleNamespace(
        headroom_exhausted=lambda: next(answers, True)
    )
    result = await orch.run(_tasks(["t1"]))
    assert result.run_status == "aborted_mid"
    assert result.to_dict()["run_status"] == "aborted_mid"


async def test_run_status_completed_without_cost_guard(tmp_path):
    """Legacy path (no cost guard): stamping is unchanged."""
    orch = _make_orch(output_dir=str(tmp_path), max_iterations=1, patience=1)
    _seed_run_mocks(orch)
    assert orch._cost_guard is None
    result = await orch.run(_tasks(["t1"]))
    assert result.run_status == "completed"


# --------------------------------------------------------------------------- #
# F033 — budget halt logs a budget termination message, not "max depth"
# --------------------------------------------------------------------------- #


async def test_budget_halt_logs_budget_termination_not_max_depth(tmp_path, caplog):
    orch = _make_orch(output_dir=str(tmp_path), max_iterations=50, patience=2)
    _seed_run_mocks(orch)
    orch._cost_guard = SimpleNamespace(headroom_exhausted=lambda: True)
    with caplog.at_level(logging.INFO, logger="meta_n"):
        await orch.run(_tasks(["t1"]))
    assert "Terminated: daily budget headroom exhausted" in caplog.text
    assert "all candidates at max_depth" not in caplog.text


# --------------------------------------------------------------------------- #
# F030 — _gate_solve_one delegates to _eval_solve_once
# --------------------------------------------------------------------------- #


async def test_gate_native_depth1_branch_stamps_via_eval_solve_once():
    """The native depth-1 gate path now returns a fully stamped trace
    (depth / reasoning / duration + adoption fields), identical to eval."""
    orch = _make_orch()
    orch.solver.solve = AsyncMock(return_value=("echo hi", "why", 10))
    orch.executor.execute = AsyncMock(return_value=Trace(
        task_id="t1", success=True, score=0.5, script="echo hi"))
    cand = Candidate(candidate_id="c", iteration=0, depth=1)
    trace = await orch._gate_solve_one(
        orch.solver, TaskDescription(task_id="t1", description="d"), cand
    )
    assert trace.depth == 1
    assert trace.reasoning == "why"
    assert trace.duration_s > 0


# --------------------------------------------------------------------------- #
# F036 — merge candidate persistence + no aliasing
# --------------------------------------------------------------------------- #


async def test_merge_candidate_persisted_by_run(tmp_path):
    """run() must persist the Ω_merge candidate through the PRODUCTION save
    path so a --resume rebuild does not silently drop the reported best."""
    orch = _make_orch(output_dir=str(tmp_path), max_iterations=0)
    _seed_run_mocks(orch)

    def fake_merge(tasks, result, force=False):
        merged = _candidate(
            "merge_oracle", {"t1": 0.9}, scripts={"t1": "echo merged"},
            depth=2, iteration=result.total_iterations,
            injected_codes=[InjectedCode(
                task_solution_map={"t1": "echo merged"},
                rationale="SYNTHESIZED:Ω_merge", source_depth=2,
            )],
        )
        orch.archive.add(merged)
        return merged

    orch._build_merged_candidate = fake_merge
    result = await orch.run(_tasks(["t1"]))
    assert result.merge_candidate_id == "merge_oracle"

    mdir = tmp_path / "archive" / "merge_oracle"
    assert (mdir / "summary.json").exists()
    assert (mdir / "injected_code_d2.json").exists()
    # The production files round-trip through a resume rebuild.
    rebuilt = Archive.rebuild_from_disk(tmp_path / "archive")
    merged = rebuilt.get("merge_oracle")
    assert merged.depth == 2
    assert merged.injected_codes[0].task_solution_map == {"t1": "echo merged"}


def test_merge_traces_do_not_alias_archive_best():
    orch = _make_orch()
    orch.archive.add(_candidate("c1", {"t1": 0.9, "t2": 0.1}))
    orch.archive.add(_candidate("c2", {"t1": 0.1, "t2": 0.9}))
    tasks = _tasks(["t1", "t2"])
    result = EvolutionaryResult(total_iterations=1, oracle_mean_score=0.9)

    merged = orch._build_merged_candidate(tasks, result, force=True)
    assert merged is not None

    best_t1 = orch.archive.per_task_best_traces()["t1"]
    assert merged.traces[0] is not best_t1  # deep copy, not the archive ref
    merged.traces[0].failure_class = "mutated"
    assert best_t1.failure_class == ""  # archive entry untouched


# --------------------------------------------------------------------------- #
# F037 — oracle mean uses the full task-set denominator
# --------------------------------------------------------------------------- #


async def test_oracle_mean_uses_full_task_denominator(tmp_path):
    """A task with no finite-scored candidate counts as 0.0 in
    ``oracle_mean_score`` — it must NOT shrink the denominator (this is the
    number main.py's "Per-task best mean (oracle)" console line displays)."""
    orch = _make_orch(output_dir=str(tmp_path), max_iterations=0)
    orch.solver.solve = AsyncMock(return_value=("echo hi", "r", 10))

    async def _exec(script, task):
        if task.task_id == "t1":
            return Trace(task_id="t1", success=True, score=0.5, script="echo hi")
        # Non-finite score → filtered from per_task_best (archive contract).
        return Trace(task_id="t2", success=False, score=float("nan"),
                     script="echo hi")

    orch.executor.execute = AsyncMock(side_effect=_exec)
    orch.omega.generate = AsyncMock(return_value=(InjectedCode(), 50))

    result = await orch.run(_tasks(["t1", "t2"]))
    assert result.per_task_best_scores == {"t1": 0.5}
    subset_mean = (
        sum(result.per_task_best_scores.values()) / len(result.per_task_best_scores)
    )
    assert result.oracle_mean_score == pytest.approx(0.25)  # 0.5 / 2 tasks
    assert result.oracle_mean_score != subset_mean


# --------------------------------------------------------------------------- #
# F052 — logger level restore + LLM-IO rotation on failed resume
# --------------------------------------------------------------------------- #


async def test_run_restores_meta_n_logger_level(tmp_path):
    lg = logging.getLogger("meta_n")
    orig = lg.level
    try:
        lg.setLevel(logging.WARNING)
        orch = _make_orch(output_dir=str(tmp_path), max_iterations=0)
        _seed_run_mocks(orch)
        await orch.run(_tasks(["t1"]))
        assert lg.level == logging.WARNING  # not a leaked DEBUG
    finally:
        lg.setLevel(orig)


async def test_failed_resume_rotates_llm_io_logs(tmp_path):
    """resume=True with no checkpoint ⇒ fresh run ⇒ the old JSONLs must be
    rotated out (not appended to) exactly like a non-resume run."""
    llm_io = tmp_path / "llm_io"
    llm_io.mkdir(parents=True)
    (llm_io / "outer.jsonl").write_text('{"old": 1}\n')

    orch = _make_orch(output_dir=str(tmp_path), max_iterations=0)
    _seed_run_mocks(orch)
    await orch.run(_tasks(["t1"]), resume=True)  # no checkpoint.json on disk

    bak = llm_io / "outer.jsonl.bak.0"
    assert bak.exists()
    assert bak.read_text() == '{"old": 1}\n'
    fresh = llm_io / "outer.jsonl"
    assert (not fresh.exists()) or fresh.stat().st_size == 0


async def test_successful_resume_keeps_llm_io_logs(tmp_path):
    """A SUCCESSFUL resume must keep appending to the same JSONL (no rotation)."""
    out = str(tmp_path)

    def build():
        orch = _make_orch(output_dir=out, max_iterations=1, patience=1)
        _seed_run_mocks(orch)
        return orch

    tasks = _tasks(["t1"])
    await build().run(tasks)
    (tmp_path / "llm_io" / "outer.jsonl").write_text('{"prior": 1}\n')

    await build().run(tasks, resume=True)
    assert not (tmp_path / "llm_io" / "outer.jsonl.bak.0").exists()
    assert (tmp_path / "llm_io" / "outer.jsonl").read_text() == '{"prior": 1}\n'


# --------------------------------------------------------------------------- #
# F019 Part A (wave-1 handoff) — inner-LLM accounting in candidate summary
# --------------------------------------------------------------------------- #


def test_save_candidate_incremental_persists_inner_llm_accounting(tmp_path):
    orch = _make_orch()
    cand = _candidate(
        "c1", {"t1": 0.5}, iteration=1,
        inner_tokens=120, inner_prompt_tokens=80,
        inner_completion_tokens=40, inner_calls=3,
    )
    orch._save_candidate_incremental(cand, tmp_path)
    summary = json.loads((tmp_path / "archive" / "c1" / "summary.json").read_text())
    assert summary["inner_tokens"] == 120
    assert summary["inner_prompt_tokens"] == 80
    assert summary["inner_completion_tokens"] == 40
    assert summary["inner_calls"] == 3


def test_save_candidate_incremental_omits_inner_keys_when_zero(tmp_path):
    """Byte-identity gate: zero inner usage ⇒ the keys are ABSENT, not 0."""
    orch = _make_orch()
    orch._save_candidate_incremental(_candidate("c0", {"t1": 0.5}), tmp_path)
    summary = json.loads((tmp_path / "archive" / "c0" / "summary.json").read_text())
    for key in ("inner_tokens", "inner_prompt_tokens",
                "inner_completion_tokens", "inner_calls"):
        assert key not in summary


# --------------------------------------------------------------------------- #
# F042 step 2 (wave-1 handoff) — _uses_external_spine delegates
# --------------------------------------------------------------------------- #


def test_uses_external_spine_delegates_to_shared_predicate(monkeypatch):
    orch = _make_orch()
    calls = {}

    def probe(base_solver, adapter):
        calls["args"] = (base_solver, adapter)
        return True

    monkeypatch.setattr(
        "meta_n.core.evolutionary_orchestrator.uses_external_spine", probe
    )
    assert orch._uses_external_spine() is True
    assert calls["args"] == (orch.config.base_solver, orch.adapter)


# --------------------------------------------------------------------------- #
# F040 — _archive_kwargs is the single Archive kwarg source
# --------------------------------------------------------------------------- #


def test_archive_kwargs_single_source():
    executor = SimpleNamespace(adapter=SimpleNamespace(
        score_scale=lambda: {"lo": 0.0, "hi": 1.0, "kind": "unit"}
    ))
    orch = EvolutionaryOrchestrator(
        llm_client=MagicMock(),
        executor=executor,
        omega=MagicMock(),
        config=EvolutionaryConfig(within_task_recursion=True, consolidate=True),
    )
    kw = orch._archive_kwargs()
    assert kw["within_task_depth_bonus"] == _WITHIN_TASK_DEPTH_BETA
    assert kw["score_ceiling"] == 1.0
    assert kw["novelty_alpha"] == orch.config.novelty_alpha
    assert kw["regression_guard"] is orch.config.regression_guard
    # __init__ constructed the archive from the exact same bundle.
    assert orch.archive.novelty_alpha == kw["novelty_alpha"]
    assert orch.archive.within_task_depth_bonus == kw["within_task_depth_bonus"]
    assert orch.archive._score_ceiling == kw["score_ceiling"]


def test_archive_kwargs_default_flags_off():
    """Default config ⇒ depth bonus 0.0 and (mock adapter) no ceiling."""
    orch = _make_orch()
    kw = orch._archive_kwargs()
    assert kw["within_task_depth_bonus"] == 0.0
    assert kw["score_ceiling"] is None  # MagicMock adapter → no usable hi
    assert kw["regression_guard"] is False


# --------------------------------------------------------------------------- #
# F044 — _score_scale_hi extraction
# --------------------------------------------------------------------------- #


def _raising_adapter():
    def _boom():
        raise RuntimeError("no scale")
    return SimpleNamespace(score_scale=_boom)


@pytest.mark.parametrize(
    "adapter, expected",
    [
        (SimpleNamespace(score_scale=lambda: {"hi": 1.0}), 1.0),
        (SimpleNamespace(score_scale=lambda: {"hi": 3}), 3.0),  # int → float
        (SimpleNamespace(score_scale=lambda: {"hi": True}), None),  # bool is not a ceiling
        (SimpleNamespace(score_scale=lambda: {"hi": float("inf")}), None),
        (SimpleNamespace(score_scale=lambda: {"hi": float("nan")}), None),
        (SimpleNamespace(score_scale=lambda: {}), None),
        (SimpleNamespace(score_scale=lambda: {"hi": None}), None),
        (_raising_adapter(), None),
        (None, None),
    ],
)
def test_score_scale_hi_extraction(adapter, expected):
    got = _score_scale_hi(adapter)
    assert got == expected
    if expected is not None:
        assert isinstance(got, float)


# --------------------------------------------------------------------------- #
# F045 / F195 — persisted extensions follow detect_script_language
# (the helper itself lives in meta_layer.py; its parametrized contract test
#  is in tests/test_refine_meta_layer.py)
# --------------------------------------------------------------------------- #


def test_candidate_trace_extension_follows_detected_language(tmp_path):
    orch = _make_orch()
    cand = _candidate(
        "cx", {"py_task": 0.5, "sh_task": 0.5},
        scripts={"py_task": "import os\nprint(1)", "sh_task": "echo hi"},
    )
    orch._save_candidate_incremental(cand, tmp_path)
    traces_dir = tmp_path / "archive" / "cx" / "traces"
    assert (traces_dir / "py_task.py").exists()
    assert (traces_dir / "sh_task.sh").exists()


# --------------------------------------------------------------------------- #
# F046 — _persist_progress keeps summary.json + checkpoint.json paired
# --------------------------------------------------------------------------- #


async def test_persist_progress_writes_both_files_per_call(tmp_path):
    orch = _make_orch(output_dir=str(tmp_path), max_iterations=1, patience=1)
    _seed_run_mocks(orch)
    result = await orch.run(_tasks(["t1"]))
    ck = json.loads((tmp_path / "checkpoint.json").read_text())
    summary = json.loads((tmp_path / "summary.json").read_text())
    # The closure persisted both files at the same moment with the same state.
    assert ck["iteration"] == result.total_iterations
    assert summary["iteration"] == ck["iteration"]
    assert summary["total_tokens"] == ck["total_tokens"] == result.total_tokens


# --------------------------------------------------------------------------- #
# F051 — eval-repeats inner-token sum uses the shared staticmethod
# --------------------------------------------------------------------------- #


async def test_eval_repeats_inner_token_sum_uses_shared_helper():
    orch = _make_orch(eval_repeats=3)
    orch.solver.solve = AsyncMock(return_value=("echo hi", "r", 10))
    orch.executor.execute = AsyncMock(return_value=Trace(
        task_id="t1", success=True, score=0.5, script="echo hi",
        inner_tokens=10, inner_prompt_tokens=6,
        inner_completion_tokens=4, inner_calls=2,
    ))
    cand = Candidate(candidate_id="c", iteration=0, depth=1)
    cand = await orch._evaluate_candidate(
        cand, orch.solver, _tasks(["t1"])
    )
    # All 3 samples' inner spend is accounted, not just the median trace's.
    assert cand.inner_tokens == 30
    assert cand.inner_prompt_tokens == 18
    assert cand.inner_completion_tokens == 12
    assert cand.inner_calls == 6


# --------------------------------------------------------------------------- #
# F192 — telemetry rollups share _aggregate_agent_rows, frozen key order
# --------------------------------------------------------------------------- #

_ROLLUP_ROWS = [
    {"run_id": "r1", "candidate_id": "c1", "agent": "openhands", "success": True,
     "token_basis": "inner", "total_tokens": 100, "cost_usd": 0.5,
     "terminated_by": "agent_done", "score": 1.0, "cost_basis": "native_usd"},
    {"run_id": "r2", "candidate_id": "c1", "agent": "builtin", "success": False,
     "token_basis": "outer", "total_tokens": 40, "cost_usd": 0.25,
     "terminated_by": "timeout", "score": 0.0,
     "cost_basis": "priced_from_tokens"},
    {"run_id": "r3", "candidate_id": "c2", "agent": "openhands", "success": True,
     "token_basis": "inner", "total_tokens": 60, "cost_usd": None,
     "terminated_by": None, "score": 0.5},
]


def test_rollups_key_order_frozen():
    """Both rollups are persisted into summary.json — insertion order is part
    of the byte contract and must survive the shared-core extraction."""
    orch = _make_orch()
    orch._read_agent_run_rows = lambda: list(_ROLLUP_ROWS)

    cand = orch._candidate_agent_telemetry_rollup("c1")
    assert list(cand.keys()) == [
        "runs", "agents", "successes", "mean_score", "inner_total_tokens",
        "outer_total_tokens", "cost_usd", "cost_basis", "terminated_by",
    ]
    assert cand["runs"] == 2
    assert cand["agents"] == ["builtin", "openhands"]
    assert cand["successes"] == 1
    assert cand["mean_score"] == pytest.approx(0.5)
    assert cand["inner_total_tokens"] == 100
    assert cand["outer_total_tokens"] == 40
    assert cand["cost_usd"] == pytest.approx(0.75)
    assert cand["cost_basis"] == ["native_usd", "priced_from_tokens"]
    assert cand["terminated_by"] == {"agent_done": 1, "timeout": 1}

    run = orch._run_level_agent_telemetry_rollup()
    assert list(run.keys()) == [
        "runs", "runs_by_agent", "successes", "inner_total_tokens",
        "outer_total_tokens", "cost_usd", "terminated_by",
    ]
    assert run["runs"] == 3
    assert run["runs_by_agent"] == {"openhands": 2, "builtin": 1}
    assert run["successes"] == 2
    assert run["inner_total_tokens"] == 160
    assert run["outer_total_tokens"] == 40
    assert run["cost_usd"] == pytest.approx(0.75)
    assert run["terminated_by"] == {"agent_done": 1, "timeout": 1, "unknown": 1}


def test_rollups_return_none_without_rows():
    orch = _make_orch()
    orch._read_agent_run_rows = lambda: []
    assert orch._candidate_agent_telemetry_rollup("c1") is None
    assert orch._run_level_agent_telemetry_rollup() is None


# --------------------------------------------------------------------------- #
# F200 — [test]/[chain-test] status cell format pinned
# --------------------------------------------------------------------------- #


def test_test_eval_status_formatting():
    ok = SimpleNamespace(success=True, score=0.5)
    bad = SimpleNamespace(success=False, score=0.25)
    fmt = EvolutionaryOrchestrator._format_test_eval_status
    assert fmt(ok) == "[green]✓ 0.500[/green]"
    assert fmt(bad) == "[red]✗ 0.250[/red]"
