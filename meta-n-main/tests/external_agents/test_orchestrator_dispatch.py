"""Orchestrator -> spine dispatch contract (seed / evaluate / gate).

The other tests in this package construct an :class:`ExternalAgentSolver`
directly and call ``spine.execute()``. That proves the spine INTERNALS but NOT
that the :class:`EvolutionaryOrchestrator` actually routes an external
``base_solver`` through the spine rather than the legacy native path. This module
closes that gap: it drives the REAL orchestrator methods
(``_init_external_agent_spine`` / ``_build_solver_from_candidate`` /
``_build_external_solver`` / ``_evaluate_candidate`` / ``_gate_check`` and the
gen0 seed path inside ``run()``) against a STUB adapter whose
``make_agent_backend`` / ``make_env_provider`` / ``make_scorer`` return install-free
fakes, and asserts the spine path — not the legacy ``solve() + executor`` path —
was exercised.

No Docker, no OpenHands SDK, no LLM: the fake backend's ``run`` returns a canned
:class:`AgentRunResult`; the fake env provider yields a host scratch dir; the fake
scorer returns a fixed :class:`EvalResult`.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest

from meta_n.core.archive import Candidate
from meta_n.core.evolutionary_orchestrator import (
    EvolutionaryConfig,
    EvolutionaryOrchestrator,
)
from meta_n.core.external_agents.backend import AgentBackend, AgentRunResult
from meta_n.core.external_agents.env import AgentEnvProvider, Scorer
from meta_n.core.external_agents.solver import ExternalAgentSolver
from meta_n.core.external_agents.terminated import TerminatedBy
from meta_n.core.meta_layer import InjectedCode, TaskDescription
from meta_n.integrations.benchmark import BenchmarkAdapter, EvalResult
from meta_n.utils.cost_tracker import CostTracker


# --- install-free fakes the orchestrator must wire through the spine --------


class _FakeBackend(AgentBackend):
    """An agent backend (NOT outer_token_mode) whose ``run`` is observable.

    Behaving like OpenHands / Terminus 2 (``outer_token_mode=False``) is what
    forces the orchestrator down the external dispatch branch — the same branch
    ``base_solver in {'openhands','terminus2'}`` takes. ``run`` records that it
    was awaited so the test can prove ``execute()`` (not the legacy
    ``solve()+executor``) drove the task.
    """

    name = "openhands"  # a real external kind so the orchestrator routes here
    outer_token_mode = False

    def __init__(self) -> None:
        self.run_calls = 0
        self.last_ctx = None

    async def run(self, ctx, tel, rec) -> AgentRunResult:
        self.run_calls += 1
        self.last_ctx = ctx
        return AgentRunResult(
            transcript="solved",
            reasoning_summary="fake-agent reasoning",
            agent_tokens=120,
            agent_prompt_tokens=80,
            agent_completion_tokens=40,
            agent_calls=1,
            cost_usd=0.0,
            cost_basis="priced_from_tokens",
            wall_s=0.01,
            steps=1,
            terminated_by=TerminatedBy.COMPLETED,
        )


class _FakeEnvProvider(AgentEnvProvider):
    """Yields a host scratch dir; extract_solution returns a fixed solution."""

    @asynccontextmanager
    async def provision(self, task, lease):
        root = lease.workdir / "workspace"
        root.mkdir(parents=True, exist_ok=True)

        class _Env:
            workspace_handle = task

        yield _Env()

    async def stage_files(self, env, files):
        return None

    async def extract_solution(self, env, run) -> str:
        return "def solve(**kwargs):\n    return {}\n"


class _FakeScorer(Scorer):
    """Grades every solution as a perfect pass (deterministic)."""

    async def score(self, task, env, solution, run) -> EvalResult:
        return EvalResult(
            success=True, score=1.0, raw_score=1.0, feedback="fake pass"
        )


class _StubAdapter(BenchmarkAdapter):
    """A minimal adapter exposing the three external-agent factory hooks.

    Records each factory call so the test can prove the orchestrator pulled the
    backend / env provider / scorer from the adapter (the external dispatch
    path), not the native solver.
    """

    def __init__(self) -> None:
        self.backend = _FakeBackend()
        self.env_provider = _FakeEnvProvider()
        self.scorer = _FakeScorer()
        self.make_backend_calls: list[str] = []

    # BenchmarkAdapter requires these abstract members; the spine never calls
    # them on the external path, so trivial stubs suffice.
    @property
    def name(self) -> str:  # pragma: no cover - identity only
        return "stub"

    def load_tasks(self, limit=None):  # pragma: no cover - unused on this path
        return []

    async def evaluate(self, task, solution):  # pragma: no cover - unused
        return EvalResult(success=True, score=1.0)

    def make_agent_backend(self, kind: str, **kw):
        self.make_backend_calls.append(kind)
        assert kind == "openhands"
        return self.backend

    def make_env_provider(self, kind: str):
        return self.env_provider

    def make_scorer(self, kind: str):
        return self.scorer


def _make_orchestrator(tmp_path, **overrides) -> EvolutionaryOrchestrator:
    """Build an orchestrator with base_solver='openhands' + the stub adapter.

    The LLM client is a MagicMock carrying a REAL ``cost_tracker`` (so
    ``_init_external_agent_spine`` does not refuse for a missing ledger), and the
    executor is a MagicMock whose ``.adapter`` is the stub (the orchestrator reads
    the adapter off ``executor.adapter``).
    """
    cost_tracker = CostTracker(
        ledger_dir=tmp_path / "costs", daily_cap_usd=1000.0, reservation_usd=0.0
    )
    llm_client = MagicMock()
    llm_client.cost_tracker = cost_tracker
    llm_client.config = MagicMock(model="gpt-5.2")

    adapter = _StubAdapter()
    executor = MagicMock()
    executor.adapter = adapter
    # If the legacy native path were ever taken, executor.execute would be the
    # one driving the task. We leave it as an AsyncMock so an accidental legacy
    # dispatch is observable (asserted-not-called below).
    executor.execute = AsyncMock(
        return_value=MagicMock(success=False, score=0.0, error_summary="LEGACY")
    )

    omega = MagicMock()

    defaults = dict(
        base_solver="openhands",
        output_dir=str(tmp_path / "run"),
        max_docker=1,
        scratch_root=str(tmp_path / "scratch"),
        parallel=1,
        patience=1,
        gate_tasks=1,
        beam_width=1,
        beam_candidates=1,
        max_depth=2,
        agentic_max_budget_usd=0.01,
        agentic_time_limit_s=60,
    )
    defaults.update(overrides)
    config = EvolutionaryConfig(**defaults)
    orch = EvolutionaryOrchestrator(
        llm_client=llm_client,
        executor=executor,
        omega=omega,
        config=config,
        solver_language="python",
    )
    return orch, adapter, executor


# --- (a) spine init ---------------------------------------------------------


def test_external_base_solver_inits_spine(tmp_path):
    """base_solver='openhands' must construct the shared spine collaborators."""
    orch, adapter, executor = _make_orchestrator(tmp_path)
    assert orch._run_guard is not None
    assert orch._cost_guard is not None
    assert orch._agent_telemetry is not None
    # The adapter is read off the executor for the factory hooks.
    assert orch.adapter is adapter


# --- (b) seed builds an ExternalAgentSolver ---------------------------------


def test_gen0_seed_builds_external_agent_solver(tmp_path):
    """The gen0 seed candidate must build an ExternalAgentSolver wrapping the
    adapter's backend — NOT a native Layer1Solver / MetaLayer.

    This is the object the seed path at evolutionary_orchestrator.py:343-347
    constructs via ``_build_solver_from_candidate``.
    """
    orch, adapter, executor = _make_orchestrator(tmp_path)
    seed = Candidate(candidate_id="gen0_seed", iteration=0, depth=1)
    solver = orch._build_solver_from_candidate(seed)

    assert isinstance(solver, ExternalAgentSolver)
    # Built from the adapter's factory hooks (the external dispatch path).
    assert adapter.make_backend_calls == ["openhands"]
    assert solver.backend is adapter.backend
    assert solver.env_provider is adapter.env_provider
    assert solver.scorer is adapter.scorer
    # The shared spine collaborators are passed by reference (one ledger /
    # semaphore / telemetry tree spans the whole run, plan §6.7).
    assert solver.run_guard is orch._run_guard
    assert solver.cost_guard is orch._cost_guard
    assert solver.telemetry is orch._agent_telemetry
    # gen0 vanilla baseline: no injected behavior.
    assert solver.injected_codes == []


# --- (c) _evaluate_candidate routes via solver.execute() -> backend.run -----


@pytest.mark.asyncio
async def test_evaluate_candidate_routes_through_spine(tmp_path):
    """_evaluate_candidate must drive the external solver via execute() (which
    runs the backend), and must NOT fall through to the legacy executor.execute.
    """
    orch, adapter, executor = _make_orchestrator(tmp_path)
    seed = Candidate(candidate_id="gen0_seed", iteration=0, depth=1)
    solver = orch._build_solver_from_candidate(seed)

    tasks = [TaskDescription(task_id="t1", description="solve t1")]
    await orch._evaluate_candidate(seed, solver, tasks)

    # The spine ran the agent backend (proves execute() was the dispatch path).
    assert adapter.backend.run_calls == 1
    # The legacy native executor.execute() was NEVER touched (no silent
    # fallthrough to the script-and-executor path).
    executor.execute.assert_not_awaited()
    # The candidate carries the scorer's perfect score (the spine's Trace).
    assert seed.mean_score == 1.0
    assert seed.traces[0].success is True
    # A telemetry row was written to the shared tree.
    rows = _read_rows(orch._agent_telemetry.output_dir)
    assert len(rows) == 1
    assert rows[0]["agent"] == "openhands"
    assert rows[0]["token_basis"] == "inner"  # external backend -> inner spend
    assert rows[0]["terminated_by"] == "completed"
    assert rows[0]["success"] is True


# --- (d) _gate_check routes via solver.execute() ----------------------------


@pytest.mark.asyncio
async def test_gate_check_routes_through_spine(tmp_path):
    """_gate_check must use the external solver's execute() at the gate, not the
    legacy ``solve() + executor.execute`` path.

    The gate samples ``base * min(depth-1, 2)`` tasks, so it is exercised on
    depth>=2 candidates (the children the gate exists to filter); a depth-2
    candidate samples ``gate_tasks`` tasks and routes each through execute().
    """
    orch, adapter, executor = _make_orchestrator(tmp_path)
    ic = InjectedCode(pre_process="additional_context = 'hint'", source_depth=2)
    child = Candidate(
        candidate_id="child1", iteration=1, depth=2, injected_codes=[ic]
    )
    solver = orch._build_solver_from_candidate(child)

    tasks = [TaskDescription(task_id=f"t{i}", description=f"solve t{i}") for i in range(3)]
    passed, gate_n, _ = await orch._gate_check(child, None, solver, tasks)

    assert passed is True
    assert gate_n >= 1
    # The agent backend ran (execute()); the legacy executor was never used.
    assert adapter.backend.run_calls >= 1
    executor.execute.assert_not_awaited()


# --- end-to-end: the seed path inside run() drives the spine ----------------


@pytest.mark.asyncio
async def test_run_seed_path_drives_spine_end_to_end(tmp_path):
    """A gen0-only ``run()`` (empty Omega) must dispatch the seed through the
    spine: the orchestrator builds an ExternalAgentSolver at :343-347 and
    evaluates it via execute(), never via the native executor.
    """
    orch, adapter, executor = _make_orchestrator(tmp_path, gate_tasks=0)
    # Empty Omega -> only the seed lands in the archive (no children/gate).
    orch.omega.generate = AsyncMock(return_value=(InjectedCode(), 0))

    tasks = [TaskDescription(task_id="t1", description="solve t1")]
    result = await orch.run(tasks)

    assert result.archive_size == 1  # seed only
    assert result.best_mean_score == 1.0
    # The spine path was exercised for the seed: the agent backend ran and the
    # legacy native executor never did.
    assert adapter.backend.run_calls >= 1
    executor.execute.assert_not_awaited()


# --- helper -----------------------------------------------------------------


def _read_rows(output_dir):
    import json
    from pathlib import Path

    path = Path(output_dir) / "telemetry" / "agent_runs.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        rec = (obj.get("extra", {}) or {}).get("record")
        rows.append(rec if rec is not None else obj)
    return rows
