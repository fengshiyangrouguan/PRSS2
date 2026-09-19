"""Shared fixtures for the install-free external-agents unit suite (plan §12.1).

Every fixture here imports only ``meta_n`` + the standard library. Nothing in
this package imports ``openhands``, ``terminal_bench`` or ``docker``, and nothing
starts a container — this is the Phase-0 CI gate (plan §12.1).

Install-free caveat: ``test_schema_basis.py`` is the one exception — it exercises
``meta_n.analysis.telemetry.fair_comparison`` (the ``analysis`` extra / pandas)
and ``pytest.importorskip("pandas")``-skips itself when that extra is absent, so
the minimal Phase-0 gate stays green without it.

The fixtures provide:

* lightweight builders for ``InjectedCode`` / ``TaskDescription`` so each test
  can compose a candidate chain without the orchestrator;
* a ``FakeBackend`` (the fault-injection harness of plan §12.1's
  ``test_never_raise``) plus a small set of fault-configurable variants;
* in-memory ``AgentEnvProvider`` / ``Scorer`` stubs that provision a host scratch
  dir, stage files, echo a fixed solution, and score deterministically;
* a real ``CostTracker`` rooted in a tmp dir and a real ``AgentTelemetry`` so the
  budget / telemetry tests exercise the genuine ledger + JSONL write path; and
* a JSONL reader that un-nests the ``LLMIOLogger`` ``extra.record`` envelope.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from meta_n.core.external_agents import (
    AgentEnvProvider,
    AgentRunResult,
    AgentTelemetry,
    DockerRunGuard,
    Scorer,
)
from meta_n.core.external_agents.backend import AgentBackend
from meta_n.core.external_agents.budget import CostGuard
from meta_n.core.external_agents.terminated import TerminatedBy
from meta_n.core.meta_layer import InjectedCode, TaskDescription
from meta_n.integrations.benchmark import EvalResult

# The canonical BudgetExceededError home is the cost ledger (budget.py no
# longer re-exports it); conftest keeps re-exporting it for the test suite.
from meta_n.utils.cost_tracker import BudgetExceededError, CostTracker


# ---------------------------------------------------------------------------
# Pure-data builders
# ---------------------------------------------------------------------------


def make_task(task_id: str = "task-0", description: str = "do the thing", **metadata):
    """Build a ``TaskDescription`` with optional metadata."""
    return TaskDescription(
        task_id=task_id, description=description, metadata=dict(metadata)
    )


def make_injected(
    *,
    pre_process: str | None = None,
    code_library: dict[str, str] | None = None,
    code_library_bash: dict[str, str] | None = None,
    source_depth: int = 0,
) -> InjectedCode:
    """Build a single ``InjectedCode`` block (one meta-layer's emission)."""
    return InjectedCode(
        pre_process=pre_process,
        code_library=dict(code_library or {}),
        code_library_bash=dict(code_library_bash or {}),
        source_depth=source_depth,
    )


@pytest.fixture
def task() -> TaskDescription:
    return make_task()


@pytest.fixture
def py_helper_source() -> str:
    """A trivially valid python helper named ``greet`` (with a docstring)."""
    return (
        "def greet(name):\n"
        '    """Return a greeting for name.\n\n'
        "    Second line of the docstring.\n"
        '    """\n'
        "    return f'hi {name}'\n"
    )


@pytest.fixture
def bash_helper_source() -> str:
    """A trivially valid bash helper named ``count_lines``."""
    return 'count_lines() {\n  wc -l "$1"\n}\n'


# ---------------------------------------------------------------------------
# Fault-injection backend (plan §12.1 — test_never_raise)
# ---------------------------------------------------------------------------


class FakeBackend(AgentBackend):
    """Configurable backend used by the fault-injection tests.

    ``raise_in`` selects whether ``run`` / ``collect_metrics`` raises and which
    exception class is raised. When nothing is configured to raise, ``run``
    returns a normal :class:`AgentRunResult` (the ``result`` arg, or a benign
    default) so the success path can be exercised too.
    """

    name = "fake"
    outer_token_mode = False

    def __init__(
        self,
        *,
        raise_in: str | None = None,
        exc: BaseException | None = None,
        result: AgentRunResult | None = None,
    ) -> None:
        self.raise_in = raise_in
        self.exc = exc
        self._result = result
        self.run_called = 0
        self.collect_called = 0

    async def run(self, ctx, tel, rec) -> AgentRunResult:
        self.run_called += 1
        if self.raise_in == "run":
            raise self.exc if self.exc is not None else RuntimeError("boom")
        if self._result is not None:
            return self._result
        return AgentRunResult(
            transcript="ok",
            reasoning_summary="did it",
            terminated_by=TerminatedBy.COMPLETED,
            attribution_available=False,
        )

    async def collect_metrics(self, env, run) -> AgentRunResult:
        self.collect_called += 1
        if self.raise_in == "collect_metrics":
            raise self.exc if self.exc is not None else RuntimeError("boom")
        return run


# ---------------------------------------------------------------------------
# In-memory env provider + scorer
# ---------------------------------------------------------------------------


class _FakeEnv:
    """Opaque env handle exposing a workspace dir + a mutable solution slot."""

    def __init__(self, workspace_root: Path, solution: str):
        self.workspace_root = workspace_root
        self.workspace_handle = str(workspace_root)
        self.staged: dict[str, str] = {}
        self._solution = solution


class FakeEnvProvider(AgentEnvProvider):
    """Provisions a host scratch dir, records staged files, echoes a solution.

    Touches no Docker/SDK — it just makes a subdir under the lease workdir. A
    ``provision_error`` flips it into a provider that raises during provision so
    the spine's never-raise wrap can be exercised.
    """

    def __init__(self, *, solution: str = "print('hi')", provision_error: bool = False):
        self.solution = solution
        self.provision_error = provision_error
        self.last_env: _FakeEnv | None = None

    @asynccontextmanager
    async def provision(self, task, lease):
        if self.provision_error:
            raise RuntimeError("provision blew up")
        root = lease.workdir / "workspace"
        root.mkdir(parents=True, exist_ok=True)
        env = _FakeEnv(root, self.solution)
        self.last_env = env
        try:
            yield env
        finally:
            pass

    async def stage_files(self, env, files):
        env.staged = dict(files)
        root: Path = env.workspace_root
        for rel, content in files.items():
            target = root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)

    async def extract_solution(self, env, run) -> str:
        return env._solution


class FakeScorer(Scorer):
    """Scores deterministically; mirrors the run's inner-token accounting.

    Note: the configured values are stored under private names so they do not
    shadow the bound :meth:`score` method on the instance.
    """

    def __init__(self, *, score: float = 1.0, success: bool = True, raw_score: float | None = None):
        self._score = score
        self._success = success
        self._raw_score = score if raw_score is None else raw_score

    async def score(self, task, env, solution, run) -> EvalResult:
        return EvalResult(
            success=self._success,
            score=self._score,
            raw_score=self._raw_score,
            feedback="scored by FakeScorer",
            inner_tokens=int(getattr(run, "agent_tokens", 0) or 0),
            inner_prompt_tokens=int(getattr(run, "agent_prompt_tokens", 0) or 0),
            inner_completion_tokens=int(getattr(run, "agent_completion_tokens", 0) or 0),
            inner_calls=int(getattr(run, "agent_calls", 0) or 0),
        )


# ---------------------------------------------------------------------------
# Real collaborators rooted in tmp dirs
# ---------------------------------------------------------------------------


@pytest.fixture
def cost_tracker(tmp_path) -> CostTracker:
    """A real file-backed ``CostTracker`` in a tmp ledger dir."""
    return CostTracker(
        ledger_dir=tmp_path / "costs", daily_cap_usd=100.0, reservation_usd=1.0
    )


@pytest.fixture
def cost_guard(cost_tracker) -> CostGuard:
    """A ``CostGuard`` priced against a model present in PRICING."""
    return CostGuard(cost_tracker, model="gpt-5.2")


@pytest.fixture
def telemetry(tmp_path) -> AgentTelemetry:
    """A real ``AgentTelemetry`` writing under ``tmp_path/run``."""
    return AgentTelemetry(tmp_path / "run", generation=0)


@pytest.fixture
def run_guard(tmp_path) -> DockerRunGuard:
    """A ``DockerRunGuard`` with a single inner slot (host scratch only).

    Pass ``scratch_root=tmp_path`` so the guard's leases mkdtemp UNDER the test's
    tmp dir, never into the system temp root — a guard built with no
    ``scratch_root`` lands ``ext_agent_*`` dirs in ``tempfile.gettempdir()``,
    polluting the temp root and tripping the reaper's orphaned-scratch scan.
    """
    return DockerRunGuard(max_docker=1, scratch_root=str(tmp_path))


# ---------------------------------------------------------------------------
# JSONL reader (un-nests the LLMIOLogger envelope)
# ---------------------------------------------------------------------------


def read_run_records(output_dir: Path) -> list[dict]:
    """Return the un-nested ``AgentRunRecord`` dicts from ``agent_runs.jsonl``."""
    return _read_records(Path(output_dir) / "telemetry" / "agent_runs.jsonl")


def _read_records(path: Path) -> list[dict]:
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


@pytest.fixture
def records_of():
    """Return the ``read_run_records`` helper (callable on an output dir)."""
    return read_run_records


# Re-export the never-raise exception type so tests can import it from conftest.
__all__ = [
    "BudgetExceededError",
    "FakeBackend",
    "FakeEnvProvider",
    "FakeScorer",
    "make_injected",
    "make_task",
    "read_run_records",
]
