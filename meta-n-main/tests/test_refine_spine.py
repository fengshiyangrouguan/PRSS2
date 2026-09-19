"""Regression tests for the C5_spine audit-refinement wave.

One test (or group) per implemented finding:

* F080 — ``normalize_model``/``_MODEL_PREFIX`` (never-called) removed.
* F083 — ``_unsafe_helper_name`` simplification is byte-identical (truth table).
* F087 — ``task_from_workspace`` consolidated into ``backend.py``.
* F089 — OH ``_normalize_status`` single-sourced from ``terminated._normalize``.
* F090 — T2/builtin term overrides folded into ``_T2_FAILURE_TO_ENUM``.
* F091 — ``_TEARDOWN_WAIT_S`` single-sourced from ``_bridge.TEARDOWN_WAIT_S``.
* F092/F093 — CostGuard shared headroom read + single ledger append.
* F094 — one-pass ``_load_run_id_index`` preserves both loaders' semantics.
* F098 — injection-plan build faults degrade to a recorded row (never escape).
* F099 — degraded paths (timeout / in-lease fault) de-reap the lease logs.
* F101 — ``backends`` package: lazy PEP 562 exports for all five backends.
* F103 — no wall-clock (``time.time()``) duration deltas in the TB base/builtin.
* F109 — dead ``BudgetExceededError`` re-export removed from ``budget``.
* F201 — ``_run_coordinates`` shared by ``start_record`` + ``restamp_reused_gate``.

Install-free / offline: imports only ``meta_n`` + stdlib. No SDK, no Docker,
no LLM.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

import meta_n.core.external_agents._bridge as _bridge
import meta_n.core.external_agents.backends._external_tb as ext_tb_mod
import meta_n.core.external_agents.backends.builtin as builtin_mod
import meta_n.core.external_agents.backends.builtin_tb as builtin_tb_mod
import meta_n.core.external_agents.backends.openhands as oh_mod
import meta_n.core.external_agents.backends.terminus2 as t2_mod
import meta_n.core.external_agents.budget as budget_mod
import meta_n.core.external_agents.solver as solver_mod
import meta_n.core.external_agents.telemetry as telemetry_mod
import meta_n.core.external_agents.terminated as terminated_mod
from meta_n.core.external_agents.backend import (
    AgentBackend,
    AgentRunContext,
    AgentRunResult,
    task_from_workspace,
)
from meta_n.core.external_agents.budget import CostGuard
from meta_n.core.external_agents.concurrency import DockerRunGuard
from meta_n.core.external_agents.env import AgentEnvProvider, Scorer
from meta_n.core.external_agents.injection import _unsafe_helper_name
from meta_n.core.external_agents.solver import ExternalAgentSolver
from meta_n.core.external_agents.telemetry import AgentTelemetry
from meta_n.core.external_agents.terminated import TerminatedBy, from_t2_failure
from meta_n.integrations.benchmark import EvalResult
from meta_n.utils.cost_tracker import compute_cost_usd


# ---------------------------------------------------------------------------
# Shared stubs / helpers
# ---------------------------------------------------------------------------


class _TD:
    task_id = "t-refine"
    description = "do the thing"
    metadata: dict = {}


def _read_rows(output_dir: Path) -> list[dict]:
    """Un-nest the LLMIOLogger ``extra.record`` envelope from agent_runs.jsonl."""
    path = Path(output_dir) / "telemetry" / "agent_runs.jsonl"
    if not path.exists():
        return []
    out: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        rec = (obj.get("extra", {}) or {}).get("record")
        out.append(rec if rec is not None else obj)
    return out


class _LogWritingBackend(AgentBackend):
    """Inner-basis backend that writes a log file into ctx.logging_dir."""

    name = "terminus2"
    outer_token_mode = False

    def __init__(self, *, sleep_s: float = 0.0, raise_cancel: bool = False):
        self._sleep_s = sleep_s
        self._raise_cancel = raise_cancel

    async def run(self, ctx: AgentRunContext, tel, rec) -> AgentRunResult:
        logs = Path(ctx.logging_dir)
        logs.mkdir(parents=True, exist_ok=True)
        (logs / "instruction.txt").write_text("staged before spawn", encoding="utf-8")
        if self._raise_cancel:
            raise asyncio.CancelledError()
        if self._sleep_s:
            await asyncio.sleep(self._sleep_s)
        return AgentRunResult(
            transcript="ran",
            agent_tokens=100,
            agent_prompt_tokens=60,
            agent_completion_tokens=40,
            agent_calls=1,
            cost_basis="priced_from_tokens",
            terminated_by=TerminatedBy.COMPLETED,
            attribution_available=False,
        )


class _EnvProvider(AgentEnvProvider):
    @asynccontextmanager
    async def provision(self, task, lease):
        root = lease.workdir / "workspace"
        root.mkdir(parents=True, exist_ok=True)

        class _Env:
            workspace_handle = str(root)

        yield _Env()

    async def stage_files(self, env, files):
        return None

    async def extract_solution(self, env, run) -> str:
        return "print('hi')"


class _FaultingEnvProvider(_EnvProvider):
    """Raises AFTER the agent ran, while the lease is still held."""

    async def extract_solution(self, env, run) -> str:
        raise RuntimeError("post-run fault")


class _OkScorer(Scorer):
    async def score(self, task, env, solution, run) -> EvalResult:
        return EvalResult(success=True, score=1.0, raw_score=1.0, feedback="ok")


def _build_solver(
    tmp_path,
    *,
    backend,
    telemetry,
    env_provider=None,
    scorer=None,
    time_limit_s=None,
):
    run_guard = DockerRunGuard(max_docker=1, scratch_root=str(tmp_path / "scratch"))
    return ExternalAgentSolver(
        backend=backend,
        env_provider=env_provider or _EnvProvider(),
        scorer=scorer or _OkScorer(),
        injected_codes=[],
        depth=1,
        run_guard=run_guard,
        telemetry=telemetry,
        cost_guard=None,
        adapter=object(),
        time_limit_s=time_limit_s,
        run_ctx={
            "output_dir": str(telemetry.output_dir),
            "candidate_id": "cand-refine",
        },
    )


# ---------------------------------------------------------------------------
# F080 — normalize_model removed (never called; priced ids stay UN-normalized)
# ---------------------------------------------------------------------------


def test_normalize_model_removed():
    assert not hasattr(telemetry_mod, "normalize_model")
    assert not hasattr(telemetry_mod, "_MODEL_PREFIX")


# ---------------------------------------------------------------------------
# F083 — _unsafe_helper_name: pre-change truth table (behavior byte-identical)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,unsafe",
    [
        ("", True),
        ("/abs", True),
        ("\\abs", True),
        ("a/b", True),
        ("a\\b", True),
        ("..", True),
        ("a..b", True),
        ("../x", True),
        ("x/..", True),
        ("ok_name", False),
        ("name2", False),
        ("_x", False),
    ],
)
def test_unsafe_helper_name_table(name, unsafe):
    assert _unsafe_helper_name(name) is unsafe


# ---------------------------------------------------------------------------
# F087 — task_from_workspace consolidated into backend.py
# ---------------------------------------------------------------------------


def test_task_from_workspace_shapes():
    task = SimpleNamespace(task_id="t", description="d")
    # Bare task handle → returned as-is by both requirement sets.
    assert task_from_workspace(task) is task
    assert (
        task_from_workspace(task, nested_requires=("description", "task_id")) is task
    )
    # Env whose ``.task`` has only task_id: recovered with the default requires
    # (CO-Bench builtin), rejected by the TB requires (caller synthesizes).
    env_partial = SimpleNamespace(task=SimpleNamespace(task_id="t"))
    assert task_from_workspace(env_partial) is env_partial.task
    assert (
        task_from_workspace(env_partial, nested_requires=("description", "task_id"))
        is None
    )
    # Env bundling a full task under ``.task_description`` → both recover it.
    env_full = SimpleNamespace(task_description=task)
    assert task_from_workspace(env_full) is task
    assert (
        task_from_workspace(env_full, nested_requires=("description", "task_id"))
        is task
    )
    # No task anywhere → None.
    assert task_from_workspace(None) is None
    assert task_from_workspace(SimpleNamespace()) is None


def test_backend_local_task_helpers_removed():
    assert not hasattr(builtin_mod, "_task_from_workspace")
    assert not hasattr(builtin_tb_mod, "_task_from_workspace")


# ---------------------------------------------------------------------------
# F089 — OH status normalization single-sourced from terminated._normalize
# ---------------------------------------------------------------------------


def test_normalize_status_is_terminated_normalize():
    assert oh_mod._normalize_status is terminated_mod._normalize
    assert oh_mod._normalize_status("STUCK ") == "stuck"
    assert oh_mod._normalize_status("") is None
    assert oh_mod._normalize_status(None) is None


# ---------------------------------------------------------------------------
# F090 — runner/bridge tags live in the shared _T2_FAILURE_TO_ENUM table
# ---------------------------------------------------------------------------


def test_t2_failure_table_covers_bridge_tags():
    assert from_t2_failure("env_error") is TerminatedBy.ENV_ERROR
    assert from_t2_failure("output_length_exceeded") is TerminatedBy.MAX_TURNS
    assert from_t2_failure("agent_installation_failed") is TerminatedBy.ENV_ERROR
    # _normalize strips + lowercases, matching the old override's behavior.
    assert from_t2_failure("ENV_ERROR ") is TerminatedBy.ENV_ERROR
    assert not hasattr(t2_mod, "_T2_TERM_OVERRIDES")


def test_backend_resolvers_defer_to_shared_table():
    # Neither resolver reads ``self`` — call unbound to avoid construction.
    t2_resolve = t2_mod.Terminus2Backend._resolve_terminated
    btb_resolve = builtin_tb_mod.BuiltinTBBackend._resolve_terminated
    assert t2_resolve(None, "env_error") is TerminatedBy.ENV_ERROR
    assert t2_resolve(None, "output_length_exceeded") is TerminatedBy.MAX_TURNS
    assert t2_resolve(None, "agent_installation_failed") is TerminatedBy.ENV_ERROR
    assert btb_resolve(None, "env_error") is TerminatedBy.ENV_ERROR
    # Clean/unknown behavior unchanged.
    assert t2_resolve(None, None) is TerminatedBy.COMPLETED
    assert t2_resolve(None, "no_such_tag") is TerminatedBy.UNKNOWN


# ---------------------------------------------------------------------------
# F091 — teardown reap bound single-sourced in _bridge
# ---------------------------------------------------------------------------


def test_teardown_wait_single_source():
    assert oh_mod._TEARDOWN_WAIT_S is _bridge.TEARDOWN_WAIT_S
    assert ext_tb_mod._TEARDOWN_WAIT_S is _bridge.TEARDOWN_WAIT_S
    assert _bridge.TEARDOWN_WAIT_S == 70.0


# ---------------------------------------------------------------------------
# F092 / F093 — CostGuard: shared headroom read + single ledger append
# ---------------------------------------------------------------------------


class _StubTracker:
    daily_cap_usd = 10.0
    reservation_usd = 1.0

    def __init__(self, today: object = 0.0):
        self._today = today
        self.recorded: list[tuple] = []

    def today_total_usd(self):
        if isinstance(self._today, BaseException):
            raise self._today
        return self._today

    def record_usd(self, model, cost_usd, **kwargs):
        self.recorded.append((model, cost_usd, kwargs))


def test_nonfinite_ledger_fails_closed_on_both_gates():
    guard = CostGuard(_StubTracker(today=float("nan")), model="gpt-5.2")
    assert guard.headroom_exhausted() is True  # fail closed
    assert guard.precheck(0.5) is True  # deny


def test_tracker_read_error_fails_open_on_both_gates():
    guard = CostGuard(_StubTracker(today=RuntimeError("io")), model="gpt-5.2")
    assert guard.headroom_exhausted() is False
    assert guard.precheck(0.5) is False


def test_exact_threshold_boundary_denies():
    # today == cap - reservation → headroom == 0.0 → exhausted AND denied
    # (pins the ``<= 0.0`` semantics through the shared helper).
    guard = CostGuard(_StubTracker(today=9.0), model="gpt-5.2")
    assert guard.headroom_exhausted() is True
    assert guard.precheck(0.5) is True


def test_record_single_ledger_append_shape():
    tracker = _StubTracker()
    guard = CostGuard(tracker, model="gpt-5.2")

    native = AgentRunResult(
        cost_usd=1.25,
        cost_basis="native_usd",
        agent_prompt_tokens=60,
        agent_completion_tokens=40,
        agent_cached_tokens=5,
    )
    guard.record(
        native, task=SimpleNamespace(task_id="t1"), solver=SimpleNamespace(depth=2)
    )
    priced = AgentRunResult(
        cost_basis="priced_from_tokens",
        agent_prompt_tokens=1000,
        agent_completion_tokens=500,
        agent_cached_tokens=0,
    )
    guard.record(
        priced, task=SimpleNamespace(task_id="t2"), solver=SimpleNamespace(depth=1)
    )

    assert len(tracker.recorded) == 2
    model, cost, kw = tracker.recorded[0]
    assert model == "gpt-5.2" and cost == 1.25
    assert kw["prompt_tokens"] == 60
    assert kw["completion_tokens"] == 40
    assert kw["cached_tokens"] == 5
    assert set(kw["extra"]) == {"source", "basis", "task_id", "depth"}
    assert kw["extra"]["basis"] == "native_usd"
    assert kw["extra"]["task_id"] == "t1" and kw["extra"]["depth"] == 2

    model2, cost2, kw2 = tracker.recorded[1]
    assert model2 == "gpt-5.2"
    assert cost2 == compute_cost_usd("gpt-5.2", 1000, 500, 0)
    assert kw2["extra"]["basis"] == "priced_from_tokens"


def test_record_native_nan_clamped_to_zero():
    tracker = _StubTracker()
    guard = CostGuard(tracker, model="gpt-5.2")
    guard.record(AgentRunResult(cost_usd=float("nan"), cost_basis="native_usd"))
    assert tracker.recorded[0][1] == 0.0


# ---------------------------------------------------------------------------
# F109 — dead BudgetExceededError re-export removed from budget.py
# ---------------------------------------------------------------------------


def test_budget_error_reexport_removed():
    assert budget_mod.__all__ == ["CostGuard"]
    assert not hasattr(budget_mod, "BudgetExceededError")


# ---------------------------------------------------------------------------
# F094 — one-pass run_id index preserves both loaders' exact semantics
# ---------------------------------------------------------------------------


def _envelope(rec: dict) -> str:
    return json.dumps({"extra": {"record": rec}})


def test_run_id_index_single_pass_semantics(tmp_path):
    out = tmp_path / "run"
    tel_dir = out / "telemetry"
    tel_dir.mkdir(parents=True)
    lines = [
        # (1) enveloped clean row.
        _envelope({"run_id": "aaa", "terminated_by": "completed"}),
        # (2) envelope record LACKS run_id; top-level has one → in seen (the
        # cross-envelope fallback), NOT in degraded (same-record read only).
        json.dumps(
            {"run_id": "bbb", "extra": {"record": {"terminated_by": "timeout"}}}
        ),
        # (3) degraded then clean for the same run_id → last row wins.
        _envelope({"run_id": "ccc", "terminated_by": "timeout"}),
        _envelope({"run_id": "ccc", "terminated_by": "completed"}),
        # (4) degraded and stays degraded.
        _envelope({"run_id": "ddd", "terminated_by": "env_error"}),
        # (5) garbage line → skipped.
        "{not json",
    ]
    (tel_dir / "agent_runs.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    tel = AgentTelemetry(out)
    assert tel._seen_run_ids == {"aaa", "bbb", "ccc", "ddd"}
    assert tel._degraded_run_ids == {"ddd"}


# ---------------------------------------------------------------------------
# F098 — injection-plan build faults degrade to a recorded row
# ---------------------------------------------------------------------------


def _fault_build(exc: BaseException):
    def _boom(task, additional_context=""):
        raise exc

    return _boom


def test_execute_survives_injection_build_fault(tmp_path):
    out = tmp_path / "run"
    tel = AgentTelemetry(output_dir=str(out))
    solver = _build_solver(tmp_path, backend=_LogWritingBackend(), telemetry=tel)
    solver._injection.build = _fault_build(ValueError("bad helper source"))

    trace, tokens = asyncio.run(solver.execute(_TD()))
    assert trace.success is False and trace.score == 0.0 and tokens == 0

    rows = _read_rows(out)
    assert len(rows) == 1
    assert rows[0]["terminated_by"] == "agent_error"
    assert rows[0]["failure_mode"] == "agent_error"


def test_solve_survives_injection_build_fault(tmp_path):
    out = tmp_path / "run"
    tel = AgentTelemetry(output_dir=str(out))
    solver = _build_solver(tmp_path, backend=_LogWritingBackend(), telemetry=tel)
    solver._injection.build = _fault_build(ValueError("bad helper source"))

    script, reasoning, tokens = asyncio.run(solver.solve(_TD(), "ctx"))
    assert (script, reasoning, tokens) == ("", "", 0)
    assert len(_read_rows(out)) == 1


def test_injection_build_memoryerror_is_degraded(tmp_path):
    # The verified real vector class: ast.parse can MemoryError on pathological
    # Ω-generated helper source, escaping every ``except SyntaxError``.
    out = tmp_path / "run"
    tel = AgentTelemetry(output_dir=str(out))
    solver = _build_solver(tmp_path, backend=_LogWritingBackend(), telemetry=tel)
    solver._injection.build = _fault_build(MemoryError())

    trace, tokens = asyncio.run(solver.execute(_TD()))
    assert trace.success is False and tokens == 0
    rows = _read_rows(out)
    assert len(rows) == 1
    assert rows[0]["terminated_by"] == "agent_error"


def test_injection_build_cancelled_propagates(tmp_path):
    out = tmp_path / "run"
    tel = AgentTelemetry(output_dir=str(out))
    solver = _build_solver(tmp_path, backend=_LogWritingBackend(), telemetry=tel)
    solver._injection.build = _fault_build(asyncio.CancelledError())

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(solver.execute(_TD()))
    assert _read_rows(out) == []  # no row on cooperative cancellation


# ---------------------------------------------------------------------------
# F099 — degraded paths de-reap the lease agent_logs before the rmtree
# ---------------------------------------------------------------------------


def test_outer_timeout_dereaps_lease_logs(tmp_path, monkeypatch):
    out = tmp_path / "run"
    tel = AgentTelemetry(output_dir=str(out))
    # Make the outer wall fire almost immediately (backend sleeps past it).
    monkeypatch.setattr(solver_mod, "hard_timeout", lambda s: 0.05)
    monkeypatch.setattr(solver_mod, "_OUTER_GRACE_S", 0.0)
    solver = _build_solver(
        tmp_path,
        backend=_LogWritingBackend(sleep_s=30.0),
        telemetry=tel,
        time_limit_s=0.01,
    )

    trace, tokens = asyncio.run(solver.execute(_TD()))
    assert trace.success is False and tokens == 0

    rows = _read_rows(out)
    assert len(rows) == 1
    row = rows[0]
    assert row["terminated_by"] == "timeout"
    expected_ptr = f"archive/cand-refine/agent_logs/{row['run_id']}"
    assert row["transcript_ptr"] == expected_ptr
    # The pointer RESOLVES after the run (lease scratch already rmtree'd).
    assert (out / expected_ptr).is_dir()
    assert (out / expected_ptr / "instruction.txt").exists()


def test_in_lease_fault_dereaps_lease_logs(tmp_path):
    out = tmp_path / "run"
    tel = AgentTelemetry(output_dir=str(out))
    solver = _build_solver(
        tmp_path,
        backend=_LogWritingBackend(),
        telemetry=tel,
        env_provider=_FaultingEnvProvider(),
    )

    trace, tokens = asyncio.run(solver.execute(_TD()))
    assert trace.success is False and tokens == 0

    rows = _read_rows(out)
    assert len(rows) == 1
    row = rows[0]
    assert row["terminated_by"] == "agent_error"
    expected_ptr = f"archive/cand-refine/agent_logs/{row['run_id']}"
    assert row["transcript_ptr"] == expected_ptr
    assert (out / expected_ptr / "instruction.txt").exists()


def test_cancel_does_not_dereap(tmp_path):
    out = tmp_path / "run"
    tel = AgentTelemetry(output_dir=str(out))
    solver = _build_solver(
        tmp_path, backend=_LogWritingBackend(raise_cancel=True), telemetry=tel
    )

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(solver.execute(_TD()))
    assert _read_rows(out) == []  # no row on cancellation (unchanged)
    assert not (out / "archive" / "cand-refine" / "agent_logs").exists()


# ---------------------------------------------------------------------------
# F101 — backends package: lazy PEP 562 exports for all five backends
# ---------------------------------------------------------------------------


def test_backends_package_import_is_lazy():
    # Fresh interpreter: importing the package must NOT import any backend
    # module (import discipline), yet the advertised names must resolve.
    code = (
        "import sys\n"
        "import meta_n.core.external_agents.backends as pkg\n"
        "assert 'meta_n.core.external_agents.backends.openhands' not in sys.modules\n"
        "assert 'meta_n.core.external_agents.backends.terminus2' not in sys.modules\n"
        "from meta_n.core.external_agents.backends import OpenHandsBackend\n"
        "import meta_n.core.external_agents.backends.openhands as oh\n"
        "assert OpenHandsBackend is oh.OpenHandsBackend\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_backends_package_exports_all_five():
    import meta_n.core.external_agents.backends as pkg
    import meta_n.core.external_agents.backends.openhands_tb as oh_tb_mod
    from meta_n.core.external_agents.backends import (
        BuiltinBackend,
        BuiltinTBBackend,
        OpenHandsBackend,
        OpenHandsTBBackend,
        Terminus2Backend,
    )

    assert BuiltinBackend is builtin_mod.BuiltinBackend
    assert BuiltinTBBackend is builtin_tb_mod.BuiltinTBBackend
    assert OpenHandsBackend is oh_mod.OpenHandsBackend
    assert OpenHandsTBBackend is oh_tb_mod.OpenHandsTBBackend
    assert Terminus2Backend is t2_mod.Terminus2Backend

    with pytest.raises(AttributeError):
        pkg.NotABackend  # noqa: B018 - attribute access is the assertion
    assert set(pkg.__all__) <= set(dir(pkg))


# ---------------------------------------------------------------------------
# F103 — monotonic duration deltas (no wall-clock time.time() in the sources)
# ---------------------------------------------------------------------------


def test_no_wall_clock_deltas():
    assert "time.time()" not in Path(ext_tb_mod.__file__).read_text(encoding="utf-8")
    assert "time.time()" not in Path(builtin_mod.__file__).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# F177 / F178 — dead imports removed
# ---------------------------------------------------------------------------


def test_openhands_module_has_no_sys_import():
    assert not hasattr(oh_mod, "sys")


# ---------------------------------------------------------------------------
# F201 — shared _run_coordinates (start_record + restamp stay compatible)
# ---------------------------------------------------------------------------


def test_run_coordinates_fallbacks(tmp_path):
    tel = AgentTelemetry(tmp_path / "run", generation=7)
    assert tel._run_coordinates(object(), object()) == ("", 1, 7, "")


def test_restamp_matches_start_record_coordinates(tmp_path):
    # Drift guard: a gate row started via start_record must be found by
    # restamp_reused_gate after the phase stamp flips off — both read the same
    # coordinates through the shared helper.
    tel = AgentTelemetry(tmp_path / "run")
    task = SimpleNamespace(task_id="t-coord")
    solver = SimpleNamespace(
        depth=1,
        generation=3,
        candidate_id="cand",
        execution_phase="gate",
        backend=None,
        max_turns=0,
        repeat_index=0,
    )
    rec = tel.start_record(task, solver)
    assert rec.phase == "gate"
    tel._append_run(rec)

    solver.execution_phase = "eval"
    assert tel.restamp_reused_gate(task, solver) is True
    rows = _read_rows(tmp_path / "run")
    assert [r["phase"] for r in rows] == ["gate", "eval"]
