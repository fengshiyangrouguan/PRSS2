"""Regression test for SPINE finding F8 (builtin backend degraded-path tokens).

``BuiltinBackend.run`` snapshots the outer ``LLMClient.cumulative_usage`` ledger
*before* invoking the native solver, but pre-fix it only computed the usage
delta and stamped the ``agent_*`` token fields on the SUCCESS path. When
``_invoke_native`` raises AFTER the wrapped solver already spent outer LLM tokens
(e.g. a depth-1 single-shot whose ``executor is None``, or a solver that raises
after its authoring ``complete()`` call), the ``except BaseException`` handler
returned an ``AGENT_ERROR`` result with the default ``0`` agent tokens — real
outer spend recorded as $0 / 0-token.

The fix computes ``self._usage_delta(before)`` inside the except handler too and
stamps the recovered ``agent_tokens`` / prompt / completion / cached / calls onto
the degraded result (mirroring ``BuiltinTBBackend._apply``).

Offline: no Docker, no LLM, no network — a fake solver mutates a fake ledger then
the depth-1 ``executor is None`` branch raises.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from meta_n.core.external_agents.backend import AgentRunContext
from meta_n.core.external_agents.backends.builtin import BuiltinBackend
from meta_n.core.external_agents.terminated import TerminatedBy


class _FakeLLMClient:
    """Outer client exposing a mutable ``cumulative_usage`` ledger."""

    def __init__(self) -> None:
        self.cumulative_usage = {
            "prompt": 0,
            "completion": 0,
            "total": 0,
            "cached": 0,
            "calls": 0,
        }


class _FakeTask:
    """Minimal handle recoverable by ``_task_from_workspace`` (task_id + description)."""

    task_id = "spinefix-task"
    description = "do the thing"


class _SpendThenScriptSolver:
    """Depth-1 solver: ``solve`` spends outer tokens, then returns a script.

    Paired with ``executor=None`` so ``_invoke_native`` raises the RuntimeError
    AFTER the outer-usage delta has already accrued — exactly the F8 scenario.
    """

    def __init__(self, client: _FakeLLMClient) -> None:
        self._client = client

    async def solve(self, task):  # noqa: ANN001 - fake
        u = self._client.cumulative_usage
        u["prompt"] += 100
        u["completion"] += 40
        u["total"] += 140
        u["cached"] += 5
        u["calls"] += 1
        return "echo hi", "some reasoning", 140


class _NoSpendRaisingSolver:
    """Solver whose ``solve`` raises WITHOUT spending — degraded tokens stay 0."""

    async def solve(self, task):  # noqa: ANN001 - fake
        raise ValueError("boom before any spend")


def _make_ctx() -> AgentRunContext:
    return AgentRunContext(
        instruction="do the thing",
        prompt=None,  # dataclass does not validate; unused on this path
        workspace=_FakeTask(),
        time_limit_s=None,
        max_turns=1,
        token_budget=0,
        max_budget_usd=0.0,
        logging_dir=Path("/tmp/spinefix-builtin"),
    )


def test_degraded_path_recovers_outer_spend() -> None:
    """A raise AFTER outer spend yields a degraded result carrying the tokens."""
    client = _FakeLLMClient()
    backend = BuiltinBackend(
        solver=_SpendThenScriptSolver(client),
        llm_client=client,
        use_agentic=False,
        executor=None,  # depth-1 single-shot with no executor -> RuntimeError
        depth=1,
    )

    result = asyncio.run(backend.run(_make_ctx(), tel=None, rec=None))

    # Degraded (the native invoke raised) ...
    assert result.terminated_by == TerminatedBy.AGENT_ERROR
    assert result.failure_mode and result.failure_mode.startswith("agent_error:")
    assert result.cost_basis == "priced_from_tokens"
    # ... but the real outer spend is recovered (pre-fix these were all 0).
    assert result.agent_tokens == 140
    assert result.agent_prompt_tokens == 100
    assert result.agent_completion_tokens == 40
    assert result.agent_cached_tokens == 5
    assert result.agent_calls == 1


def test_degraded_path_no_spend_stays_zero() -> None:
    """A raise with no intervening outer spend records zero tokens (no phantom count)."""
    client = _FakeLLMClient()
    backend = BuiltinBackend(
        solver=_NoSpendRaisingSolver(),
        llm_client=client,
        use_agentic=False,
        executor=None,
        depth=1,
    )

    result = asyncio.run(backend.run(_make_ctx(), tel=None, rec=None))

    assert result.terminated_by == TerminatedBy.AGENT_ERROR
    assert result.agent_tokens == 0
    assert result.agent_prompt_tokens == 0
    assert result.agent_completion_tokens == 0
    assert result.agent_cached_tokens == 0
    assert result.agent_calls == 0
