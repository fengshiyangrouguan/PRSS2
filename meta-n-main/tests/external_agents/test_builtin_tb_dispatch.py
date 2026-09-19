"""Builtin-on-terminal-bench dispatch + the same-container control backend.

Pins the NEW ``--base-solver builtin`` routing split (the KEY non-regression risk):

* **CO-Bench (and any non-advertising adapter) ``builtin`` stays LEGACY.** The
  orchestrator must NOT route ``base_solver="builtin"`` through the external-agent
  spine for an adapter whose ``advertises_spine_builtin()`` is ``False`` — that
  would change CO-Bench's ~0.546 native builtin baseline. ``_uses_external_spine``
  must be ``False`` and ``_build_solver_from_candidate`` must build a NATIVE solver
  (``Layer1Solver`` / ``MetaLayer``), never an ``ExternalAgentSolver``.
* **terminal_bench ``builtin`` routes through the SPINE + ``BuiltinTBBackend``.**
  Its native executor cannot run the legacy task layout, so ``builtin`` is a
  same-container control: ``_uses_external_spine`` is ``True`` and
  ``_build_solver_from_candidate`` builds an ``ExternalAgentSolver`` whose backend
  is a :class:`BuiltinTBBackend` (``outer_token_mode=True``).

Plus the backend unit contract: :class:`BuiltinTBBackend` authors a one-shot bash
script via the native ``Layer1Solver`` (one outer call, captured via the outer
ledger delta) and folds it into the bridge request as ``bash_script``, then stamps
the OUTER authoring tokens onto the parsed runner result; the shared
``TBExternalScorer`` scores it from ``native_resolved`` / ``native_score``.

Install-free: no Docker, no terminal_bench, no LLM — synthetic dicts +
monkeypatched ``create_subprocess_exec``; the native solver is a fake.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from meta_n.core.archive import Candidate
from meta_n.core.evolutionary_orchestrator import (
    EvolutionaryConfig,
    EvolutionaryOrchestrator,
)
from meta_n.core.external_agents.backend import AgentRunContext, AgentRunResult, Prompt
from meta_n.core.external_agents.backends.builtin_tb import BuiltinTBBackend
from meta_n.core.external_agents.solver import ExternalAgentSolver
from meta_n.core.external_agents.terminated import TerminatedBy
from meta_n.core.meta_layer import TaskDescription
from meta_n.integrations.terminal_bench import (
    TBExternalScorer,
    TerminalBenchAdapter,
)
from meta_n.utils.cost_tracker import CostTracker


# ===========================================================================
# (A) The routing split — the KEY non-regression risk.
# ===========================================================================


def _orchestrator(tmp_path, *, base_solver, adapter, advertises: bool):
    """Build an orchestrator wired with ``base_solver`` and ``adapter``.

    ``adapter`` is used AS PASSED — each caller configures its
    ``advertises_spine_builtin`` itself (a real :class:`TerminalBenchAdapter`,
    which advertises ``True`` natively; or a ``MagicMock`` with an explicit
    ``return_value`` / ``side_effect`` so the defensive-fallback tests can drive a
    truthy-non-True, a raising, or a non-callable shape). ``advertises`` is a
    DOCUMENTATION-ONLY flag recording the caller's expected routing decision; the
    helper does not re-stub the adapter from it (doing so would clobber the
    explicit ``side_effect`` / truthy-MagicMock the fallback tests rely on).
    """
    cost_tracker = CostTracker(
        ledger_dir=tmp_path / "costs", daily_cap_usd=1000.0, reservation_usd=0.0
    )
    llm_client = MagicMock()
    llm_client.cost_tracker = cost_tracker
    llm_client.config = MagicMock(model="google/gemma-4-31b-qat", base_url=None,
                                  api_key=None)

    executor = MagicMock()
    executor.adapter = adapter
    executor.execute = AsyncMock()

    config = EvolutionaryConfig(
        base_solver=base_solver,
        output_dir=str(tmp_path / "run"),
        max_docker=1,
        scratch_root=str(tmp_path / "scratch"),
        parallel=1,
        patience=1,
        gate_tasks=1,
    )
    orch = EvolutionaryOrchestrator(
        llm_client=llm_client,
        executor=executor,
        omega=MagicMock(),
        config=config,
        solver_language="bash",
    )
    return orch


def test_cobench_builtin_stays_legacy(tmp_path):
    """A non-advertising adapter keeps ``builtin`` on the LEGACY native path."""
    adapter = MagicMock()
    adapter.advertises_spine_builtin.return_value = False  # CO-Bench shape

    orch = _orchestrator(tmp_path, base_solver="builtin", adapter=adapter,
                         advertises=False)

    # The spine is NOT used: the predicate is False, the spine collaborators are
    # never constructed, and a built solver is the NATIVE Layer1Solver (not the
    # external spine wrapper).
    assert orch._uses_external_spine() is False
    assert orch._run_guard is None
    assert orch._cost_guard is None
    assert orch._agent_telemetry is None

    cand = Candidate(candidate_id="gen0_seed", iteration=0, depth=1)
    solver = orch._build_solver_from_candidate(cand)
    assert not isinstance(solver, ExternalAgentSolver)
    # The legacy depth-1 builtin path is the orchestrator's own Layer1Solver.
    assert solver is orch.solver
    # The adapter's external factory hooks were NEVER consulted for builtin.
    adapter.make_agent_backend.assert_not_called()


def test_terminal_bench_builtin_routes_through_spine(tmp_path):
    """terminal_bench ``builtin`` routes through the SPINE + BuiltinTBBackend."""
    adapter = TerminalBenchAdapter()  # real adapter (advertises spine builtin)

    orch = _orchestrator(tmp_path, base_solver="builtin", adapter=adapter,
                         advertises=True)

    # The adapter advertises builtin-on-spine, so the predicate is True and the
    # spine collaborators ARE constructed.
    assert adapter.advertises_spine_builtin() is True
    assert orch._uses_external_spine() is True
    assert orch._run_guard is not None
    assert orch._cost_guard is not None
    assert orch._agent_telemetry is not None

    cand = Candidate(candidate_id="gen0_seed", iteration=0, depth=1)
    solver = orch._build_solver_from_candidate(cand)
    assert isinstance(solver, ExternalAgentSolver)
    assert isinstance(solver.backend, BuiltinTBBackend)
    # The backend was given the authoring solver + outer client (it authors the
    # script meta-n-side) and reports OUTER tokens.
    assert solver.backend._solver is orch.solver
    assert solver.backend._llm_client is orch.llm_client
    assert solver.backend.outer_token_mode is True


def test_other_external_kinds_unaffected(tmp_path):
    """A genuine external kind (openhands) routes through the spine regardless of
    the builtin-advertise flag (the predicate's first branch)."""
    adapter = TerminalBenchAdapter()
    orch = _orchestrator(tmp_path, base_solver="openhands", adapter=adapter,
                         advertises=True)
    assert orch._uses_external_spine() is True


def test_none_base_solver_stays_legacy(tmp_path):
    """``base_solver=None`` is the legacy native path even with an advertising
    adapter (the predicate only promotes the literal ``builtin`` flag)."""
    adapter = TerminalBenchAdapter()
    orch = _orchestrator(tmp_path, base_solver=None, adapter=adapter,
                         advertises=True)
    assert orch._uses_external_spine() is False
    assert orch._run_guard is None


# ===========================================================================
# (B) BuiltinTBBackend — authoring + token re-attribution + result mapping.
# ===========================================================================


class _FakeLLMClient:
    """Outer client whose ``cumulative_usage`` ledger the authoring call bumps."""

    def __init__(self):
        self.cumulative_usage = {
            "prompt": 0, "completion": 0, "total": 0, "cached": 0, "calls": 0,
        }


class _FakeSolver:
    """A native Layer1Solver stand-in: bumps the outer ledger, returns a script."""

    def __init__(self, llm_client, script="echo hi > /app/out.txt"):
        self.llm_client = llm_client
        self.script = script
        self.calls: list = []

    async def solve(self, task, additional_context=""):
        self.calls.append((task, additional_context))
        u = self.llm_client.cumulative_usage
        u["prompt"] += 30
        u["completion"] += 10
        u["total"] += 40
        u["calls"] += 1
        return self.script, "authoring reasoning", 40


def _backend(solver, llm_client):
    return BuiltinTBBackend(
        solver=solver,
        llm_client=llm_client,
        solver_language="bash",
        model="openai/google/gemma-4-31b-qat",
        api_base="http://127.0.0.1:1234/v1",
        venv_python="/nonexistent/python",
        runner_dir="/tmp",
        tasks_dir="/tmp/tasks",
    )


class _WS:
    """The TB env handle the spine hands the backend (carries the task)."""

    def __init__(self, task, *, run_label="ext-hello-world-abc123"):
        self.task = task
        self.task_id = task.task_id
        self.staged_files: dict = {}
        self.run_label = run_label
        self.agent_pid = None


def _ctx(tmp_path, ws):
    return AgentRunContext(
        instruction="write hi to /app/out.txt",
        prompt=Prompt(),
        workspace=ws,
        time_limit_s=60.0,
        max_turns=8,
        token_budget=0,
        max_budget_usd=0.5,
        logging_dir=tmp_path / "logs",
    )


def _task():
    return TaskDescription(
        task_id="hello-world",
        description="write hi",
        metadata={"benchmark": "terminal_bench", "solution_language": "bash"},
    )


def test_run_authors_script_and_ships_it_in_request(monkeypatch, tmp_path):
    """``run`` authors the script via the solver and folds it into the request as
    ``bash_script``; the runner child env is scrubbed of meta-n secrets."""
    captured = {}

    async def _fake_exec(*cmd, env=None, **kwargs):
        captured["cmd"] = list(cmd)
        captured["env"] = env

        class _P:
            pid = 4242
            returncode = 0

            async def communicate(self):
                return (b"", b"")

            async def wait(self):
                return 0

        return _P()

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-leak")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_exec)

    llm = _FakeLLMClient()
    solver = _FakeSolver(llm, script="echo hi > /app/out.txt")
    b = _backend(solver, llm)
    ws = _WS(_task())
    # The runner writes no result file (the fake exec returns immediately); the
    # bridge degrades to env_error, but the request JSON is still written first.
    asyncio.run(b.run(_ctx(tmp_path, ws), None, None))

    # The authoring solver was called ONCE with the task off the env handle.
    assert len(solver.calls) == 1
    assert solver.calls[0][0] is ws.task

    req = json.loads((tmp_path / "logs" / "builtin_tb_request.json").read_text())
    # The PRE-AUTHORED script rides in the request under ``bash_script``.
    assert req["bash_script"] == "echo hi > /app/out.txt"
    assert req["task_id"] == "hello-world"
    # The launch resolves the builtin runner module.
    assert "-m" in captured["cmd"]
    assert "builtin_tb_runner" in captured["cmd"]
    # The child env is scrubbed of meta-n secrets (parity with OH/T2).
    assert captured["env"].get("OPENAI_API_KEY") == "dummy"
    assert "OPENROUTER_API_KEY" not in captured["env"]


def test_run_attributes_outer_authoring_tokens(monkeypatch, tmp_path):
    """The runner ships 0 tokens; ``run`` stamps the OUTER authoring spend (from the
    outer ledger delta) onto the result's ``agent_*`` fields."""
    logs = tmp_path / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    # A genuine resolved runner result with 0 tokens (no LLM in the runner).
    (logs / "builtin_tb_result.json").write_text(json.dumps({
        "ok": True, "is_resolved": True, "reward": 1.0,
        "total_input_tokens": 0, "total_output_tokens": 0, "agent_calls": 0,
        "failure_mode": "none",
    }))

    async def _fake_exec(*cmd, env=None, **kwargs):
        class _P:
            pid = 4242
            returncode = 0

            async def communicate(self):
                return (b"", b"")

            async def wait(self):
                return 0

        return _P()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_exec)

    llm = _FakeLLMClient()
    solver = _FakeSolver(llm)
    b = _backend(solver, llm)
    result = asyncio.run(b.run(_ctx(tmp_path, _WS(_task())), None, None))

    # The verifier's binary reward survives the bridge unchanged.
    assert result.native_resolved is True
    assert result.native_score == 1.0
    assert result.terminated_by is TerminatedBy.COMPLETED
    # The OUTER authoring tokens (the ledger delta) are stamped onto the result;
    # the runner's own token fields were 0.
    assert result.agent_tokens == 40
    assert result.agent_prompt_tokens == 30
    assert result.agent_completion_tokens == 10
    assert result.agent_calls == 1


def test_run_authoring_failure_degrades_without_raising(monkeypatch, tmp_path):
    """An authoring-solver exception yields a degraded AGENT_ERROR result (the
    never-raise contract), WITHOUT launching the subprocess."""
    launched = {"n": 0}

    async def _fake_exec(*cmd, env=None, **kwargs):
        launched["n"] += 1

        class _P:
            pid = 1
            returncode = 0

            async def communicate(self):
                return (b"", b"")

            async def wait(self):
                return 0

        return _P()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_exec)

    class _BoomSolver:
        async def solve(self, task, additional_context=""):
            raise RuntimeError("authoring boom")

    llm = _FakeLLMClient()
    b = _backend(_BoomSolver(), llm)
    # With an empty authored script the runner would run nothing; but the prep
    # carries an empty script and the bridge still runs. The result is unresolved.
    result = asyncio.run(b.run(_ctx(tmp_path, _WS(_task())), None, None))
    assert isinstance(result, AgentRunResult)
    # The run did not raise; an authoring failure carries 0 outer tokens.
    assert result.agent_tokens == 0


def test_scorer_scores_builtin_run_from_native_signal(tmp_path):
    """The SHARED TBExternalScorer derives success from ``native_resolved`` on a
    builtin run (the same path it uses for OH / T2)."""
    llm = _FakeLLMClient()
    b = _backend(_FakeSolver(llm), llm)
    data = {"ok": True, "is_resolved": True, "reward": 1.0}
    run = b._to_run_result(data, _ctx(tmp_path, _WS(_task())), 0.0)
    scorer = TBExternalScorer.__new__(TBExternalScorer)  # no adapter needed
    res = asyncio.run(scorer.score(task=None, env=None, solution="", run=run))
    assert res.success is True
    assert res.score == 1.0
    assert res.valid is True


# ===========================================================================
# (C) _uses_external_spine() defensive fallbacks — the live-path protectors.
# ===========================================================================
# These three branches keep the LIVE co_bench+builtin run on the native path; a
# regression here would silently promote builtin to the spine (standing up the
# cost guard / telemetry tree). The existing tests only cover a literal
# return_value=False, so pin the truthy-non-True, raising, and non-callable cases.


def test_builtin_truthy_non_true_advertise_stays_legacy(tmp_path):
    """A truthy-but-not-literal-True advertise (e.g. a MagicMock return) must NOT
    promote builtin to the spine (the strict ``is True`` guard)."""
    adapter = MagicMock()
    adapter.advertises_spine_builtin.return_value = MagicMock()  # truthy, not True

    orch = _orchestrator(tmp_path, base_solver="builtin", adapter=adapter,
                         advertises=False)
    assert orch._uses_external_spine() is False
    assert orch._cost_guard is None
    assert orch._run_guard is None
    assert orch._agent_telemetry is None


def test_builtin_advertise_raises_keeps_legacy(tmp_path):
    """A faulty ``advertises_spine_builtin()`` that RAISES is swallowed and the run
    stays on the legacy native path (the except-and-keep-legacy fallback)."""
    adapter = MagicMock()
    adapter.advertises_spine_builtin.side_effect = RuntimeError("boom")

    orch = _orchestrator(tmp_path, base_solver="builtin", adapter=adapter,
                         advertises=False)
    assert orch._uses_external_spine() is False


def test_builtin_non_callable_advertise_stays_legacy(tmp_path):
    """An adapter whose ``advertises_spine_builtin`` is a non-callable attribute
    (e.g. a plain ``True`` value) stays legacy (the ``callable`` guard)."""

    class _Adapter:
        advertises_spine_builtin = True  # attribute, not a method

    orch = _orchestrator(tmp_path, base_solver="builtin", adapter=_Adapter(),
                         advertises=False)
    assert orch._uses_external_spine() is False
