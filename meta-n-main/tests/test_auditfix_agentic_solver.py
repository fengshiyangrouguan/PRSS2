"""Regression tests for audit findings #49 and #50 (AgenticSolver).

Both assert the PERSISTED ``best_trace.terminated_by`` (the Trace object that
``_agentic_loop`` / ``execute`` returns), not just ``_AgenticResult.terminated_by``.

#49: a post-retry LLM-call failure (exception escaping ``llm_client.complete``)
     must be labeled ``env_error`` on the persisted trace, NOT mislabeled as the
     token-estimate-derived ``max_turns`` / ``token_budget`` cap reason.
#50: the SUCCESS early-return paths (``confirmed`` two-stage completion and
     ``perfect_score``) must stamp ``best_trace.terminated_by`` symmetrically with
     the fallback path — previously they only set it on ``_AgenticResult`` and the
     persisted trace kept the empty-string default.

All tests are fully offline: the LLM client and executor are local fakes.
"""

import asyncio

from meta_n.core.agentic_solver import AgenticSolver
from meta_n.core.meta_layer import TaskDescription, Trace


class _FakeLLMClient:
    """Scripted async LLM client. ``responses`` is a list of (text, tokens).

    If ``raise_always`` is set, every ``complete`` call raises — emulating a
    fully-retry-exhausted infrastructure failure (the real client only lets an
    exception escape after exhausting its internal retries).
    """

    def __init__(self, responses=None, raise_always=False):
        self._responses = list(responses or [])
        self._raise_always = raise_always
        self.calls = 0

    async def complete(self, messages, temperature=0.7):
        self.calls += 1
        if self._raise_always:
            raise RuntimeError("simulated post-retry API failure")
        if self._responses:
            return self._responses.pop(0)
        return ("<status>working</status>", 1)


class _FakeExecutor:
    """Async executor returning a Trace with a fixed score."""

    def __init__(self, score):
        self._score = score
        self.calls = 0

    async def execute(self, script, task):
        self.calls += 1
        return Trace(task_id=task.task_id, script=script, score=self._score,
                     success=self._score >= 1.0)


def _task():
    return TaskDescription(task_id="t1", description="do a thing")


def _run_loop(solver, task):
    """Drive ``_agentic_loop`` and return the persisted best_trace."""
    result = asyncio.run(solver._agentic_loop(task, ""))
    return result.best_trace, result


# ---------------------------------------------------------------------------
# Finding #49 — swallowed LLM-call exception must yield terminated_by='env_error'
# ---------------------------------------------------------------------------

def test_finding_49_llm_failure_labeled_env_error_not_cap():
    solver = AgenticSolver(
        llm_client=_FakeLLMClient(raise_always=True),
        executor=_FakeExecutor(score=0.0),
        max_turns=3,
        token_budget=10_000_000,  # huge: token-estimate branch would pick max_turns
    )
    best_trace, result = _run_loop(solver, _task())
    # On the ORIGINAL code this is "max_turns" (derived purely from the token
    # estimate). With the fix it is the distinct infra-failure reason.
    assert best_trace.terminated_by == "env_error"
    assert result.terminated_by == "env_error"


# ---------------------------------------------------------------------------
# Finding #50 — 'confirmed' must reach the persisted trace
# ---------------------------------------------------------------------------

def test_finding_50_confirmed_stamped_on_persisted_trace():
    # Turn 1: code + first 'complete' -> executes (score 0.5), asks confirmation.
    # Turn 2: bare 'complete' -> two-stage confirmation fires -> 'confirmed'.
    responses = [
        ('<code lang="python">print("hi")</code>\n<status>complete</status>', 5),
        ("<status>complete</status>", 5),
    ]
    solver = AgenticSolver(
        llm_client=_FakeLLMClient(responses=responses),
        executor=_FakeExecutor(score=0.5),  # < 1.0 so perfect_score does not pre-empt
        max_turns=5,
        token_budget=10_000_000,
    )
    best_trace, result = _run_loop(solver, _task())
    assert result.terminated_by == "confirmed"
    # On the ORIGINAL code the persisted trace keeps the "" default here.
    assert best_trace.terminated_by == "confirmed"


# ---------------------------------------------------------------------------
# Finding #50 — 'perfect_score' must reach the persisted trace
# ---------------------------------------------------------------------------

def test_finding_50_perfect_score_stamped_on_persisted_trace():
    # Turn 1: code with status working -> normal exec path -> score 1.0 -> exit.
    responses = [
        ('<code lang="python">print("hi")</code>\n<status>working</status>', 5),
    ]
    solver = AgenticSolver(
        llm_client=_FakeLLMClient(responses=responses),
        executor=_FakeExecutor(score=1.0),
        max_turns=5,
        token_budget=10_000_000,
    )
    best_trace, result = _run_loop(solver, _task())
    assert result.terminated_by == "perfect_score"
    # On the ORIGINAL code the persisted trace keeps the "" default here.
    assert best_trace.terminated_by == "perfect_score"
