"""F4: ``_to_run_result`` attribution_available reflects capture-POSSIBLE.

Before F4 ``attribution_available`` was ``bool(command_history)`` — so a real
agent run that simply issued no captured command (or whose stream was lost)
collapsed to ``attribution_available=False`` (UNMEASURABLE / None), silently
indistinguishable from a backend that cannot capture at all.

After F4 ``_to_run_result`` defaults ``attribution_available`` to True for the
``_external_tb`` backends (every one drives a real agent whose commands are
captured in-process), so an empty ``command_history`` is a MEASURED-none ([]),
not an UNMEASURABLE (None). A runner may still override with an explicit
``attribution_available=False`` key for a genuinely-unmeasurable path.

Install-free: imports only ``meta_n`` + stdlib (no SDK / Docker).
"""

from __future__ import annotations

from pathlib import Path

from meta_n.core.external_agents.backend import AgentRunContext, Prompt
from meta_n.core.external_agents.backends.terminus2 import Terminus2Backend
from meta_n.core.external_agents.telemetry import attribute_utilities


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


def test_empty_command_history_is_measurable_not_none(tmp_path):
    # A clean run that captured an EMPTY command stream is still MEASURABLE.
    b = _backend()
    data = {
        "ok": True,
        "is_resolved": True,
        "reward": 1.0,
        "command_history": [],
        "failure_mode": "none",
    }
    r = b._to_run_result(data, _ctx(tmp_path), 0.0)
    assert r.attribution_available is True
    # -> attribution is [] (measured-none), NOT None (unmeasurable).
    called, _ = attribute_utilities(
        r.command_history, ["greet"], r.attribution_available
    )
    assert called == []


def test_populated_command_history_is_measurable(tmp_path):
    b = _backend()
    data = {
        "ok": True,
        "is_resolved": True,
        "reward": 1.0,
        "command_history": ["python3 ./helpers/_lib_greet.py x"],
        "failure_mode": "none",
    }
    r = b._to_run_result(data, _ctx(tmp_path), 0.0)
    assert r.attribution_available is True
    called, _ = attribute_utilities(
        r.command_history, ["greet"], r.attribution_available
    )
    assert called == ["greet"]


def test_runner_can_override_attribution_unmeasurable(tmp_path):
    # An explicit attribution_available=False in the result dict (a genuinely
    # unmeasurable / degraded path) is honored over the capture-possible default.
    b = _backend()
    data = {
        "ok": True,
        "is_resolved": True,
        "reward": 1.0,
        "command_history": [],
        "attribution_available": False,
        "failure_mode": "none",
    }
    r = b._to_run_result(data, _ctx(tmp_path), 0.0)
    assert r.attribution_available is False
    called, _ = attribute_utilities(
        r.command_history, ["greet"], r.attribution_available
    )
    assert called is None


# --- H13: the runner traceback tail folds into stderr_tail when panes empty ---


def test_error_dict_folds_into_stderr_tail(tmp_path):
    # A synthesized-degraded result dict with EMPTY panes but an `error` traceback
    # tail surfaces that error in AgentRunResult.stderr_tail (otherwise the degraded
    # row carried no inspectable stderr at all).
    b = _backend()
    data = {
        "ok": False,
        "is_resolved": False,
        "reward": 0.0,
        "failure_mode": "parse_error",
        "post_agent_pane": "",
        "post_test_pane": "",
        "error": "Traceback (most recent call last):\n  ... boom",
    }
    r = b._to_run_result(data, _ctx(tmp_path), 0.0)
    assert "boom" in r.stderr_tail
    # stdout_tail stays the (empty) agent pane — the error only folds into stderr.
    assert r.stdout_tail == ""


def test_post_test_pane_preferred_over_error_tail(tmp_path):
    # The error fold is the both-panes-empty fallback ONLY: a present test pane wins.
    b = _backend()
    data = {
        "ok": False,
        "is_resolved": False,
        "reward": 0.0,
        "failure_mode": "none",
        "post_agent_pane": "agent out",
        "post_test_pane": "test pane text",
        "error": "should not appear",
    }
    r = b._to_run_result(data, _ctx(tmp_path), 0.0)
    assert r.stderr_tail == "test pane text"
    assert r.stdout_tail == "agent out"


def test_agent_pane_used_for_stderr_when_only_test_pane_empty(tmp_path):
    # If the test pane is empty but the agent pane is present, the agent pane is the
    # stderr fallback (the error tail is only used when BOTH panes are empty).
    b = _backend()
    data = {
        "ok": False,
        "is_resolved": False,
        "reward": 0.0,
        "failure_mode": "none",
        "post_agent_pane": "agent diagnostic",
        "post_test_pane": "",
        "error": "should not appear",
    }
    r = b._to_run_result(data, _ctx(tmp_path), 0.0)
    assert r.stderr_tail == "agent diagnostic"
    assert r.stdout_tail == "agent diagnostic"
