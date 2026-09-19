"""OpenHands inner cumulative-token KILL SWITCH (the OH↔T2 fairness fix).

Two layers, both exercised install-free (no ``openhands`` in meta-n's env):

1. The runner-side kill-switch primitives in ``scripts/oh_runner.py``
   (``_install_token_budget_killswitch`` + ``_is_token_budget_error``). The
   runner's module top level is stdlib-only (the SDK is imported lazily inside
   ``_run``), so it imports here and the pure-Python budget logic is unit-tested
   against a tiny fake ``LLM`` standing in for the OpenHands ``LLM``/``Telemetry``.

2. The backend end-to-end: ``OpenHandsBackend`` must forward ``ctx.token_budget``
   as ``--token-budget`` and map a runner ``status="token_budget"`` to
   :attr:`TerminatedBy.TOKEN_BUDGET` with a clean ``failure_mode=None`` (a SOFT
   envelope stop, not an agent error). Driven over a fake runner — no SDK.
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import pytest

# scripts/ holds the runner; add it so we can import the install-free primitives.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import oh_runner  # noqa: E402  (after sys.path insert)

from meta_n.core.external_agents.backend import (  # noqa: E402
    AgentRunContext,
    Prompt,
)
from meta_n.core.external_agents.backends.openhands import (  # noqa: E402
    OpenHandsBackend,
)
from meta_n.core.external_agents.terminated import (  # noqa: E402
    TerminatedBy,
    from_oh_status,
)


# ---------------------------------------------------------------------------
# Fake LLM / Telemetry mirroring the openhands surface the kill switch touches.
# ---------------------------------------------------------------------------
class _FakeUsage:
    def __init__(self, prompt_tokens=0, completion_tokens=0):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _FakeMetrics:
    def __init__(self):
        self.accumulated_token_usage = _FakeUsage()


class _FakeTelemetry:
    """Stand-in for ``LLM._telemetry``; records on_request invocations."""

    def __init__(self):
        self.calls = 0

    def on_request(self, telemetry_ctx=None):  # signature parity with the SDK
        self.calls += 1


class _FakeLLM:
    """Minimal stand-in: the kill switch only reads ``metrics`` and wraps
    ``_telemetry.on_request`` (which it sets via ``object.__setattr__``)."""

    def __init__(self):
        self.metrics = _FakeMetrics()
        self._telemetry = _FakeTelemetry()


# ---------------------------------------------------------------------------
# 1) Runner-side primitives — pure-Python, no SDK.
# ---------------------------------------------------------------------------
def test_killswitch_disabled_when_budget_zero():
    llm = _FakeLLM()
    # Underlying function of the original bound method (bound methods are not
    # identity-stable across attribute access, so compare ``__func__``).
    original_func = llm._telemetry.on_request.__func__
    oh_runner._install_token_budget_killswitch(llm, 0)
    # No-op: the original unwrapped on_request is left in place (un-budgeted run),
    # so even a wildly-over-budget metrics state never raises.
    assert llm._telemetry.on_request.__func__ is original_func
    llm.metrics.accumulated_token_usage = _FakeUsage(10**9, 10**9)
    llm._telemetry.on_request(telemetry_ctx=None)  # must NOT raise (no cap)


def test_killswitch_passes_under_budget():
    llm = _FakeLLM()
    oh_runner._install_token_budget_killswitch(llm, 100)
    llm.metrics.accumulated_token_usage = _FakeUsage(40, 30)  # 70 <= 100
    # Under budget: the wrapped on_request runs (and delegates to the original).
    llm._telemetry.on_request(telemetry_ctx=None)  # must NOT raise


def test_killswitch_fires_over_budget():
    llm = _FakeLLM()
    oh_runner._install_token_budget_killswitch(llm, 100)
    llm.metrics.accumulated_token_usage = _FakeUsage(80, 40)  # 120 > 100
    with pytest.raises(oh_runner.TokenBudgetExceeded):
        llm._telemetry.on_request(telemetry_ctx=None)


def test_killswitch_overshoot_is_one_call():
    """The check fires BEFORE a call once already over budget, so the call that
    first crosses the budget still completes (documented residual overshoot)."""
    llm = _FakeLLM()
    oh_runner._install_token_budget_killswitch(llm, 100)
    # Exactly at budget (100) is NOT over (strict ``>``): the call proceeds.
    llm.metrics.accumulated_token_usage = _FakeUsage(60, 40)  # 100, not > 100
    llm._telemetry.on_request(telemetry_ctx=None)  # proceeds (the crossing call)
    # Now strictly over: the NEXT call is pre-empted.
    llm.metrics.accumulated_token_usage = _FakeUsage(60, 41)  # 101 > 100
    with pytest.raises(oh_runner.TokenBudgetExceeded):
        llm._telemetry.on_request(telemetry_ctx=None)


def test_is_token_budget_error_direct_and_wrapped():
    direct = oh_runner.TokenBudgetExceeded("budget 100 exceeded")
    assert oh_runner._is_token_budget_error(direct) is True

    # Wrapped (as ``conv.run()`` does: ``raise ConversationRunError(...) from e``).
    class _Wrapper(Exception):
        pass

    try:
        try:
            raise direct
        except oh_runner.TokenBudgetExceeded as e:
            raise _Wrapper("conversation run error") from e
    except _Wrapper as w:
        assert oh_runner._is_token_budget_error(w) is True

    # An unrelated error is NOT a token-budget stop.
    assert oh_runner._is_token_budget_error(RuntimeError("boom")) is False


def test_accumulated_inner_tokens_is_defensive():
    # A structurally surprising metrics object never raises out of the read.
    class _Broken:
        metrics = object()  # no accumulated_token_usage attribute

    assert oh_runner._accumulated_inner_tokens(_Broken()) == 0


def test_token_budget_status_maps_to_token_budget_enum():
    assert from_oh_status("token_budget") is TerminatedBy.TOKEN_BUDGET


# ---------------------------------------------------------------------------
# 2) Backend end-to-end over a fake runner (no openhands install).
# ---------------------------------------------------------------------------
def _write_argv_capturing_runner(tmp_path: Path, payload: dict) -> tuple[Path, Path]:
    """Fake runner that records its argv (so we can assert --token-budget) AND
    emits a caller-controlled result JSON."""
    argv_dump = tmp_path / "argv.json"
    runner = tmp_path / "fake_runner.py"
    runner.write_text(
        textwrap.dedent(
            f"""
            import argparse, json, sys
            from pathlib import Path

            PAYLOAD = json.loads({json.dumps(json.dumps(payload))})
            ARGV_DUMP = {json.dumps(str(argv_dump))}

            p = argparse.ArgumentParser()
            for flag in (
                "--workspace", "--instruction-file", "--system-suffix-file",
                "--result-file", "--model", "--base-url", "--api-key",
                "--max-output-tokens", "--max-iterations", "--max-budget-usd",
                "--token-budget", "--solution-file",
            ):
                p.add_argument(flag)
            a = p.parse_args()
            Path(ARGV_DUMP).write_text(json.dumps(vars(a)))
            Path(a.result_file).write_text(json.dumps(PAYLOAD))
            sys.exit(0)
            """
        )
    )
    return runner, argv_dump


def _make_ctx(tmp_path: Path, token_budget: int) -> AgentRunContext:
    ws = tmp_path / "workspace"
    ws.mkdir(parents=True, exist_ok=True)
    return AgentRunContext(
        instruction="solve the task",
        prompt=Prompt(system_suffix="", prefix=""),
        workspace=str(ws),
        time_limit_s=30.0,
        max_turns=8,
        token_budget=token_budget,
        max_budget_usd=0.0,
        logging_dir=tmp_path / "log",
    )


@pytest.mark.asyncio
async def test_backend_forwards_token_budget_flag(tmp_path):
    payload = {
        "status": "finished",
        "last_message": "done",
        "command_history": [],
        "prompt_tokens": 1,
        "completion_tokens": 1,
        "accumulated_cost_usd": 0.0,
        "error": None,
    }
    runner, argv_dump = _write_argv_capturing_runner(tmp_path, payload)
    backend = OpenHandsBackend(
        model="google/gemma-4-31b-qat",
        venv_python=sys.executable,
        runner_script=str(runner),
        local_default=True,
    )
    await backend.run(_make_ctx(tmp_path, token_budget=20_000), tel=None, rec=None)
    seen = json.loads(argv_dump.read_text())
    assert seen["token_budget"] == "20000"


@pytest.mark.asyncio
async def test_backend_maps_token_budget_status_to_enum(tmp_path):
    """A runner ``status="token_budget"`` is a SOFT inner-token stop: it maps to
    TOKEN_BUDGET with failure_mode=None (not an agent error)."""
    payload = {
        "status": "token_budget",
        "last_message": "ran out of inner token budget",
        "command_history": [{"command": "echo hi", "output": "hi"}],
        "prompt_tokens": 18000,
        "completion_tokens": 4000,
        "accumulated_cost_usd": 0.0,
        "error_code": "TokenBudgetExceeded",
        "error": None,
    }
    runner, _ = _write_argv_capturing_runner(tmp_path, payload)
    backend = OpenHandsBackend(
        model="google/gemma-4-31b-qat",
        venv_python=sys.executable,
        runner_script=str(runner),
        local_default=True,
    )
    res = await backend.run(_make_ctx(tmp_path, token_budget=20_000), tel=None, rec=None)
    assert res.terminated_by is TerminatedBy.TOKEN_BUDGET
    assert res.failure_mode is None  # soft stop: no failure tag
    assert res.agent_tokens == 22000  # consumed inner tokens still reported
