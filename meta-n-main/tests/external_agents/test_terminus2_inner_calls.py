"""Terminus 2 inner LLM-call-count instrumentation (real ``inner_calls``).

The T2 runner now counts the agent's inner LLM calls (one per COMPLETED
``_BudgetedLiteLLM.call``) and emits it as ``agent_calls`` in the result JSON.
These install-free tests assert the count threads:

    result JSON ``agent_calls``  ->  Terminus2Backend._to_run_result
                                 ->  AgentRunResult.agent_calls
                                 ->  AgentRunRecord.inner_calls (telemetry)

so T2 reports a real inner-call count (not the prior 0/"unmeasured").
"""

from __future__ import annotations

from pathlib import Path

from meta_n.core.external_agents.backend import AgentRunContext, Prompt
from meta_n.core.external_agents.backends.terminus2 import Terminus2Backend


def _backend() -> Terminus2Backend:
    return Terminus2Backend(
        model="openai/google/gemma-4-31b-qat",
        api_base="http://127.0.0.1:1234/v1",
        venv_python="/nonexistent/python",
        runner_dir="/tmp",
        tasks_dir="/tmp/tasks",
    )


class _WS:
    def __init__(self, *, task_id="t", staged_files=None, run_label=""):
        self.task_id = task_id
        self.staged_files = staged_files if staged_files is not None else {}
        self.run_label = run_label


def _ctx(tmp_path: Path) -> AgentRunContext:
    return AgentRunContext(
        instruction="do it",
        prompt=Prompt(),
        workspace=_WS(),
        time_limit_s=60.0,
        max_turns=8,
        token_budget=0,
        max_budget_usd=0.5,
        logging_dir=tmp_path,
    )


def test_agent_calls_threads_to_result(tmp_path):
    """A runner ``agent_calls`` count flows into AgentRunResult.agent_calls."""
    b = _backend()
    data = {
        "ok": True,
        "is_resolved": True,
        "reward": 1.0,
        "total_input_tokens": 100,
        "total_output_tokens": 50,
        "agent_calls": 4,
        "failure_mode": "none",
    }
    r = b._to_run_result(data, _ctx(tmp_path), 0.0)
    assert r.agent_calls == 4


def test_agent_calls_recovered_on_abort(tmp_path):
    """On an abort the runner recovers the partial call count off the budgeted LLM
    and ships it as ``agent_calls`` even with a failure tag — it still threads."""
    b = _backend()
    data = {
        "ok": False,
        "is_resolved": False,
        "reward": 0.0,
        "total_input_tokens": 3000,
        "total_output_tokens": 1200,
        "agent_calls": 3,
        "failure_mode": "token_budget",
    }
    r = b._to_run_result(data, _ctx(tmp_path), 0.0)
    assert r.agent_calls == 3


def test_missing_agent_calls_defaults_zero(tmp_path):
    """A legacy result JSON without ``agent_calls`` degrades to 0 (never raises)."""
    b = _backend()
    data = {"ok": True, "is_resolved": True, "reward": 1.0, "failure_mode": "none"}
    r = b._to_run_result(data, _ctx(tmp_path), 0.0)
    assert r.agent_calls == 0


class _FakeTask:
    task_id = "t"


class _FakeSolver:
    """Minimal stand-in exposing the attributes ``start_record`` reads."""

    def __init__(self, backend):
        self.backend = backend
        self.depth = 1
        self.generation = 0
        self.candidate_id = "c0"
        self.max_turns = 8


def test_inner_calls_reaches_telemetry_record(tmp_path):
    """End-to-end: AgentRunResult.agent_calls -> AgentRunRecord.inner_calls."""
    from meta_n.core.external_agents.telemetry import AgentTelemetry

    b = _backend()
    data = {
        "ok": True,
        "is_resolved": True,
        "reward": 1.0,
        "total_input_tokens": 100,
        "total_output_tokens": 50,
        "agent_calls": 5,
        "failure_mode": "none",
    }
    run = b._to_run_result(data, _ctx(tmp_path), 0.0)

    tel = AgentTelemetry(output_dir=str(tmp_path))
    rec = tel.start_record(_FakeTask(), _FakeSolver(b))
    assert rec.agent == "terminus2"
    assert rec.token_basis == "inner"
    tel.finish_record(rec, run, evalr=None, lease=None)
    assert rec.inner_calls == 5
