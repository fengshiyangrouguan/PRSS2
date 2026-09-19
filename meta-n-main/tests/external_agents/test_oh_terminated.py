"""OpenHands ``terminated_by`` nuance — finished/low-score, max-iterations, stuck.

Regression for the OH ``terminated_by`` mapping nuance: a *finished* OH run with
no exception (``ConversationExecutionStatus.FINISHED``, ``failure_mode=None``)
that merely scored low must telemeter ``terminated_by=COMPLETED`` (success comes
from the scorer, not the status), NOT ``agent_error``. The SDK overloads
``execution_status = ERROR`` for the max-iterations ceiling, so the runner
rewrites that to ``status="max_iterations"`` → ``MAX_TURNS``. ``AGENT_ERROR`` is
reserved for genuine failures (exception / ERROR / STUCK), and whenever it (or any
other failure terminal state) is produced the run must carry a non-None
``failure_mode`` so the pair is never contradictory.

These tests drive the REAL :meth:`OpenHandsBackend.run` over the subprocess
bridge, but point the runner at a tiny stand-in script that writes a controlled
``oh_result.json`` — so the full spawn → parse → reconcile path is exercised with
NO ``openhands`` install (Phase-0 install-free gate, plan §12.1).
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import pytest

from meta_n.core.external_agents.backend import AgentRunContext, Prompt
from meta_n.core.external_agents.backends.openhands import (
    OpenHandsBackend,
    _FAILURE_TERMINAL_STATES,
    _reconcile_termination,
)
from meta_n.core.external_agents.terminated import TerminatedBy


# ---------------------------------------------------------------------------
# A fake runner that writes a caller-controlled result JSON (no openhands).
# ---------------------------------------------------------------------------
def _write_fake_runner(tmp_path: Path, payload: dict) -> Path:
    """Write a stand-in runner script that emits ``payload`` to ``--result-file``.

    The real ``oh_runner.py`` needs the OpenHands SDK; this stand-in mimics only
    its argv contract (``--result-file``) and the result-JSON schema, so the
    backend's spawn/parse/reconcile path runs install-free.
    """
    runner = tmp_path / "fake_runner.py"
    runner.write_text(
        textwrap.dedent(
            f"""
            import argparse, json, sys
            from pathlib import Path

            PAYLOAD = json.loads({json.dumps(json.dumps(payload))})

            p = argparse.ArgumentParser()
            # Accept (and ignore) every flag the backend forwards.
            for flag in (
                "--workspace", "--instruction-file", "--system-suffix-file",
                "--result-file", "--model", "--base-url", "--api-key",
                "--max-output-tokens", "--max-iterations", "--max-budget-usd",
                "--token-budget", "--solution-file",
            ):
                p.add_argument(flag)
            a = p.parse_args()
            Path(a.result_file).write_text(json.dumps(PAYLOAD))
            sys.exit(0)
            """
        )
    )
    return runner


def _make_ctx(tmp_path: Path) -> AgentRunContext:
    ws = tmp_path / "workspace"
    ws.mkdir(parents=True, exist_ok=True)
    logdir = tmp_path / "log"
    return AgentRunContext(
        instruction="solve the task",
        prompt=Prompt(system_suffix="", prefix=""),
        workspace=str(ws),
        time_limit_s=30.0,
        max_turns=8,
        token_budget=10_000,
        max_budget_usd=0.0,
        logging_dir=logdir,
    )


async def _run_with_payload(tmp_path: Path, payload: dict):
    """Drive the real backend.run() over a fake runner emitting ``payload``."""
    runner = _write_fake_runner(tmp_path, payload)
    backend = OpenHandsBackend(
        model="google/gemma-4-31b-qat",
        venv_python=sys.executable,  # host python exists; fake runner needs no SDK
        runner_script=str(runner),
        local_default=True,
    )
    ctx = _make_ctx(tmp_path)
    return await backend.run(ctx, tel=None, rec=None)


# ---------------------------------------------------------------------------
# The core nuance: finished-but-low-score -> COMPLETED, not agent_error.
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_finished_low_score_maps_to_completed(tmp_path):
    """A FINISHED run with no exception (even a low-scoring one) is COMPLETED.

    Success is decided by the scorer (``native_score`` / ``native_resolved``), not
    by ``terminated_by``; a finished run is therefore COMPLETED regardless of how
    well it scored, and never ``agent_error``.
    """
    payload = {
        "status": "finished",  # ConversationExecutionStatus.FINISHED, lowercased
        "last_message": "I wrote a (poor) solution.",
        "command_history": [{"command": "echo hi", "output": "hi"}],
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "accumulated_cost_usd": 0.0,
        "error": None,
    }
    res = await _run_with_payload(tmp_path, payload)
    assert res.terminated_by is TerminatedBy.COMPLETED
    assert res.failure_mode is None  # consistent: no failure tag on a clean finish


@pytest.mark.asyncio
async def test_max_iterations_maps_to_max_turns(tmp_path):
    """The iteration cap (runner rewrites ERROR+MaxIterationsReached) -> MAX_TURNS.

    The SDK sets ``execution_status = ERROR`` when it hits
    ``max_iteration_per_run`` without finishing; the runner disambiguates this to
    ``status="max_iterations"``. That is a non-failure stop, so MAX_TURNS with no
    ``failure_mode``.
    """
    payload = {
        "status": "max_iterations",
        "last_message": "still working...",
        "command_history": [{"command": "ls", "output": ""}],
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "accumulated_cost_usd": 0.0,
        "error_code": "MaxIterationsReached",
        "error_detail": "Agent reached maximum iterations limit (8).",
        "error": None,
    }
    res = await _run_with_payload(tmp_path, payload)
    assert res.terminated_by is TerminatedBy.MAX_TURNS
    assert res.failure_mode is None


@pytest.mark.asyncio
async def test_stuck_maps_to_agent_error_with_failure_mode(tmp_path):
    """A genuine STUCK status -> AGENT_ERROR, and carries a non-None failure_mode.

    STUCK is a real failure (the agent looped / made no progress). The runner
    leaves ``error`` None because ``conv.run()`` did not raise, so without
    reconciliation this would be the forbidden ``failure_mode=None`` + AGENT_ERROR
    pair. Reconciliation synthesizes a token from the status.
    """
    payload = {
        "status": "stuck",
        "last_message": "",
        "command_history": [],
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "accumulated_cost_usd": 0.0,
        "error": None,
    }
    res = await _run_with_payload(tmp_path, payload)
    assert res.terminated_by is TerminatedBy.AGENT_ERROR
    assert res.failure_mode == "stuck"  # consistent: failure carries a tag


@pytest.mark.asyncio
async def test_genuine_error_maps_to_agent_error(tmp_path):
    """A real exception (runner ``error`` set, status=error) stays AGENT_ERROR."""
    payload = {
        "status": "error",
        "last_message": "",
        "command_history": [],
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "accumulated_cost_usd": 0.0,
        "error": "RuntimeError: agent blew up mid-step",
    }
    res = await _run_with_payload(tmp_path, payload)
    assert res.terminated_by is TerminatedBy.AGENT_ERROR
    assert res.failure_mode is not None  # classified from the error string


# ---------------------------------------------------------------------------
# Unit-level reconciliation invariants (no subprocess).
# ---------------------------------------------------------------------------
def test_reconcile_failure_state_never_pairs_with_none_failure_mode():
    """Every failure terminal state forces a non-None failure_mode."""
    for state in _FAILURE_TERMINAL_STATES:
        term, fm = _reconcile_termination(state, None, "stuck")
        assert term is state
        assert fm, f"{state} must carry a non-None failure_mode, got {fm!r}"


def test_reconcile_non_failure_state_clears_failure_flavored_mode():
    """A non-failure terminal state drops a stale failure-flavored failure_mode."""
    # COMPLETED paired with a failure-flavored 'agent_error' token is contradictory.
    term, fm = _reconcile_termination(TerminatedBy.COMPLETED, "agent_error", "finished")
    assert term is TerminatedBy.COMPLETED
    assert fm is None


def test_reconcile_completed_with_none_stays_none():
    term, fm = _reconcile_termination(TerminatedBy.COMPLETED, None, "finished")
    assert term is TerminatedBy.COMPLETED
    assert fm is None


def test_reconcile_max_turns_keeps_no_failure_mode():
    term, fm = _reconcile_termination(TerminatedBy.MAX_TURNS, None, "max_iterations")
    assert term is TerminatedBy.MAX_TURNS
    assert fm is None
