"""Regression tests for ``mirror_agent_tokens`` (spec B9, F199).

The agent-token quad — ``(agent_tokens, agent_prompt_tokens,
agent_completion_tokens, agent_calls)`` read defensively off an
``AgentRunResult`` — used to be hand-rolled in both ``TBTerminus2Scorer.score``
(terminal_bench.py) and ``co_bench._agent_usage_kwargs``. The canonical helper
now lives next to the ``Scorer`` ABC in ``meta_n/core/external_agents/env.py``;
these tests pin its defensive semantics and the two consumers' parity.
"""

from __future__ import annotations

from types import SimpleNamespace

from meta_n.core.external_agents.env import mirror_agent_tokens
from meta_n.integrations.co_bench import _agent_usage_kwargs


class TestMirrorAgentTokens:
    def test_defaults(self):
        """A bare object with none of the attributes yields all zeros."""
        assert mirror_agent_tokens(object()) == (0, 0, 0, 0)

    def test_none_coerced(self):
        """Explicit None attribute values coerce to 0 (the ``or 0`` guard)."""
        run = SimpleNamespace(
            agent_tokens=None,
            agent_prompt_tokens=None,
            agent_completion_tokens=None,
            agent_calls=None,
        )
        assert mirror_agent_tokens(run) == (0, 0, 0, 0)

    def test_values_pass_through(self):
        run = SimpleNamespace(
            agent_tokens=110,
            agent_prompt_tokens=70,
            agent_completion_tokens=40,
            agent_calls=5,
        )
        assert mirror_agent_tokens(run) == (110, 70, 40, 5)

    def test_partial_attributes(self):
        """Missing attributes default independently of the present ones."""
        run = SimpleNamespace(agent_tokens=9)
        assert mirror_agent_tokens(run) == (9, 0, 0, 0)

    def test_non_int_values_coerced_to_int(self):
        run = SimpleNamespace(
            agent_tokens=10.0,
            agent_prompt_tokens=6.0,
            agent_completion_tokens=4.0,
            agent_calls=True,
        )
        assert mirror_agent_tokens(run) == (10, 6, 4, 1)


class TestConsumerParity:
    def test_co_bench_agent_usage_kwargs_matches_helper(self):
        """co_bench's dict-shaped wrapper is a pure re-keying of the quad."""
        run = SimpleNamespace(
            agent_tokens=110,
            agent_prompt_tokens=70,
            agent_completion_tokens=40,
            agent_calls=5,
        )
        tokens, prompt, completion, calls = mirror_agent_tokens(run)
        assert _agent_usage_kwargs(run) == {
            "inner_tokens": tokens,
            "inner_prompt_tokens": prompt,
            "inner_completion_tokens": completion,
            "inner_calls": calls,
        }

    def test_co_bench_agent_usage_kwargs_defensive_defaults(self):
        assert _agent_usage_kwargs(object()) == {
            "inner_tokens": 0,
            "inner_prompt_tokens": 0,
            "inner_completion_tokens": 0,
            "inner_calls": 0,
        }

    def test_tb_terminus2_scorer_mirrors_quad(self):
        """TBTerminus2Scorer.score copies the agent quad into EvalResult.inner_*."""
        import asyncio

        from meta_n.core.external_agents.backend import AgentRunResult
        from meta_n.core.external_agents.terminated import TerminatedBy
        from meta_n.integrations.terminal_bench import TBTerminus2Scorer

        run = AgentRunResult(
            terminated_by=TerminatedBy.COMPLETED,
            native_resolved=True,
            native_score=1.0,
            agent_tokens=110,
            agent_prompt_tokens=70,
            agent_completion_tokens=40,
            agent_calls=5,
        )
        scorer = TBTerminus2Scorer.__new__(TBTerminus2Scorer)  # no adapter needed
        res = asyncio.run(scorer.score(None, None, "", run))
        assert (
            res.inner_tokens,
            res.inner_prompt_tokens,
            res.inner_completion_tokens,
            res.inner_calls,
        ) == mirror_agent_tokens(run)
        assert res.inner_tokens == 110
