"""Terminus 2 score-derivation, token round-trip, and bridge-safety guards.

Install-free: drives ``Terminus2Backend._to_run_result`` / ``_staged_files`` /
``_run_label`` and ``TBTerminus2Scorer.score`` on synthetic runner-result dicts;
imports only ``meta_n`` + stdlib (never ``terminal_bench``).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from meta_n.core.external_agents.backend import AgentRunContext, Prompt
from meta_n.core.external_agents.backends.terminus2 import (
    Terminus2Backend,
    _sanitize_compose_label,
)
from meta_n.core.external_agents.terminated import TerminatedBy


def _backend() -> Terminus2Backend:
    return Terminus2Backend(
        model="openai/google/gemma-4-31b-qat",
        api_base="http://127.0.0.1:1234/v1",
        venv_python="/nonexistent/python",
        runner_dir="/tmp",
        tasks_dir="/tmp/tasks",
    )


class _WS:
    """Minimal workspace handle exposing the provider-contract attributes."""

    def __init__(self, *, task_id="t", staged_files=None, run_label=""):
        self.task_id = task_id
        self.staged_files = staged_files if staged_files is not None else {}
        self.run_label = run_label


def _ctx(tmp_path, ws=None, max_turns=8, token_budget=0):
    return AgentRunContext(
        instruction="do it",
        prompt=Prompt(),
        workspace=ws if ws is not None else _WS(),
        time_limit_s=60.0,
        max_turns=max_turns,
        token_budget=token_budget,
        max_budget_usd=2.0,
        logging_dir=tmp_path,
    )


# --- _to_run_result: terminated_by / score derivation ----------------------


def test_resolved_run_is_completed(tmp_path):
    b = _backend()
    data = {"ok": True, "is_resolved": True, "reward": 1.0,
            "total_input_tokens": 10, "total_output_tokens": 5,
            "failure_mode": "none"}
    r = b._to_run_result(data, _ctx(tmp_path), 0.0)
    assert r.terminated_by is TerminatedBy.COMPLETED
    assert r.native_resolved is True
    assert r.native_score == 1.0
    assert r.failure_mode is None  # nulled on COMPLETED
    assert r.agent_tokens == 15


def test_clean_failing_test_is_unknown_not_agent_error(tmp_path):
    """ran-but-wrong (ok, not resolved, failure_mode='unset') -> UNKNOWN, score 0,
    NOT a synthesized AGENT_ERROR (which over-counts agent errors)."""
    b = _backend()
    data = {"ok": True, "is_resolved": False, "reward": 0.0,
            "failure_mode": "unset"}
    r = b._to_run_result(data, _ctx(tmp_path), 0.0)
    assert r.terminated_by is TerminatedBy.UNKNOWN
    assert r.native_resolved is False
    assert r.native_score == 0.0
    assert r.failure_mode is None


def test_resolved_with_timeout_tag_nulls_failure_mode(tmp_path):
    """A resolved run whose tests passed after an AGENT_TIMEOUT must be COMPLETED
    with NO coexisting contradictory failure_mode."""
    b = _backend()
    data = {"ok": True, "is_resolved": True, "reward": 1.0,
            "failure_mode": "agent_timeout"}
    r = b._to_run_result(data, _ctx(tmp_path), 0.0)
    assert r.terminated_by is TerminatedBy.COMPLETED
    assert r.failure_mode is None


def test_real_failure_tag_surfaces(tmp_path):
    b = _backend()
    data = {"ok": True, "is_resolved": False, "reward": 0.0,
            "failure_mode": "context_length_exceeded"}
    r = b._to_run_result(data, _ctx(tmp_path), 0.0)
    assert r.terminated_by is TerminatedBy.CONTEXT_LEN
    assert r.failure_mode == "context_length_exceeded"


def test_env_error_run_zero_tokens(tmp_path):
    b = _backend()
    data = {"ok": False, "is_resolved": False, "reward": 0.0,
            "failure_mode": "env_error"}
    r = b._to_run_result(data, _ctx(tmp_path), 0.0)
    assert r.terminated_by is TerminatedBy.ENV_ERROR
    assert r.agent_tokens == 0
    assert r.native_resolved is None  # not ok -> no authoritative verdict


def test_token_budget_failure_maps_through(tmp_path):
    b = _backend()
    data = {"ok": False, "is_resolved": False, "reward": 0.0,
            "failure_mode": "token_budget",
            "total_input_tokens": 100, "total_output_tokens": 50}
    r = b._to_run_result(data, _ctx(tmp_path), 0.0)
    assert r.terminated_by is TerminatedBy.TOKEN_BUDGET
    # partial tokens survive the abort (round-trip)
    assert r.agent_tokens == 150


# --- _staged_files traversal guard -----------------------------------------


def test_staged_files_drops_absolute_and_traversal(tmp_path):
    b = _backend()
    ws = _WS(staged_files={
        "helpers/ok.py": "x",
        "/etc/passwd": "evil",
        "../../../etc/evil": "evil",
        "a/../b.py": "evil",  # contains '..'
    })
    out = b._staged_files(_ctx(tmp_path, ws))
    # only the safe relative key survives
    assert out == {"helpers/ok.py": "x"}


# --- _run_label per-run uniqueness -----------------------------------------


def test_run_label_uses_lease_session(tmp_path):
    b = _backend()
    ws = _WS(run_label="ext-myTask-AbC123")
    label = b._run_label(_ctx(tmp_path, ws))
    assert label == "ext-mytask-abc123"  # sanitized lowercase


def test_run_label_falls_back_to_uuid(tmp_path):
    b = _backend()
    ws = _WS(run_label="")
    l1 = b._run_label(_ctx(tmp_path, ws))
    l2 = b._run_label(_ctx(tmp_path, ws))
    assert l1.startswith("t2-") and l2.startswith("t2-")
    assert l1 != l2  # uuid-unique per call


def test_sanitize_compose_label():
    assert _sanitize_compose_label("ext-Foo_Bar-99") == "ext-foo_bar-99"
    assert _sanitize_compose_label("../evil/!!") == "evil"
    assert _sanitize_compose_label("") == "t2-bridge"


# --- _resolve_max_episodes honors ctx.max_turns ----------------------------


def test_max_episodes_honors_ctx_max_turns(tmp_path):
    b = _backend()
    assert b._resolve_max_episodes(_ctx(tmp_path, max_turns=3)) == 3
    assert b._resolve_max_episodes(_ctx(tmp_path, max_turns=99)) == 8  # clamp
    assert b._resolve_max_episodes(_ctx(tmp_path, max_turns=0)) == 8  # fallback


# --- TBTerminus2Scorer derives success from the verifier signal -------------


def _score(run):
    from meta_n.integrations.terminal_bench import TBTerminus2Scorer

    scorer = TBTerminus2Scorer.__new__(TBTerminus2Scorer)  # no adapter needed
    return asyncio.run(scorer.score(None, None, "", run))


def test_scorer_uses_native_resolved(tmp_path):
    from meta_n.core.external_agents.backend import AgentRunResult

    # native_resolved True even though terminated_by is (impossibly) not COMPLETED
    # -> success follows the verifier, not the status enum.
    run = AgentRunResult(
        terminated_by=TerminatedBy.UNKNOWN,
        native_resolved=True,
        native_score=1.0,
        agent_tokens=20,
    )
    res = _score(run)
    assert res.success is True
    assert res.score == 1.0
    assert res.feasible is True  # never mirrors success


def test_scorer_unresolved_is_feasible_true(tmp_path):
    from meta_n.core.external_agents.backend import AgentRunResult

    run = AgentRunResult(
        terminated_by=TerminatedBy.UNKNOWN,
        native_resolved=False,
        native_score=0.0,
    )
    res = _score(run)
    assert res.success is False
    assert res.score == 0.0
    assert res.feasible is True  # well-formed-but-failed is NOT infeasible
    assert res.valid is True


def test_scorer_env_error_is_invalid(tmp_path):
    from meta_n.core.external_agents.backend import AgentRunResult

    run = AgentRunResult(
        terminated_by=TerminatedBy.ENV_ERROR,
        native_resolved=None,
        native_score=0.0,
        failure_mode="env_error",
    )
    res = _score(run)
    assert res.success is False
    assert res.valid is False  # no verifier result obtained


def test_scorer_fallback_to_status_when_no_native(tmp_path):
    """A backend with no verifier signal falls back to terminated_by."""
    from meta_n.core.external_agents.backend import AgentRunResult

    run = AgentRunResult(terminated_by=TerminatedBy.COMPLETED)  # native_* None
    res = _score(run)
    assert res.success is True
    assert res.score == 1.0
