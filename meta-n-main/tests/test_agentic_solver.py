"""Tests for the Terminus 2-inspired agentic solver."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from meta_n.core.agentic_solver import AgenticSolver, _AgenticResult, _ParsedResponse
from meta_n.core.meta_layer import InjectedCode, TaskDescription, Trace


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_task(task_id: str = "test_task", description: str = "Solve X") -> TaskDescription:
    return TaskDescription(task_id=task_id, description=description)


def _make_solver(
    *,
    llm_responses: list[str] | None = None,
    executor_scores: list[float] | None = None,
    injected_codes: list[InjectedCode] | None = None,
    max_turns: int = 5,
    solver_language: str = "python",
) -> AgenticSolver:
    """Create an AgenticSolver with mock LLM and executor."""
    llm_responses = llm_responses or []
    executor_scores = executor_scores or []

    llm_client = AsyncMock()
    call_count = {"llm": 0, "exec": 0}

    async def mock_complete(messages, temperature=None, max_tokens=None):
        idx = min(call_count["llm"], len(llm_responses) - 1)
        call_count["llm"] += 1
        return llm_responses[idx], 100  # 100 tokens per call

    llm_client.complete = mock_complete

    executor = AsyncMock()

    async def mock_execute(script, task, timeout=30):
        idx = min(call_count["exec"], len(executor_scores) - 1)
        score = executor_scores[idx]
        call_count["exec"] += 1
        return Trace(
            task_id=task.task_id,
            script=script,
            score=score,
            success=score >= 0.5,
            exit_code=0,
            stdout=f"score={score:.4f}",
            stderr="",
            eval_feedback=f"Score: {score:.4f}",
        )

    executor.execute = mock_execute

    return AgenticSolver(
        llm_client=llm_client,
        executor=executor,
        injected_codes=injected_codes or [],
        solver_language=solver_language,
        max_turns=max_turns,
        token_budget=100_000,
    )


# ---------------------------------------------------------------------------
# _parse_response tests
# ---------------------------------------------------------------------------


class TestParseResponse:
    def setup_method(self):
        self.solver = AgenticSolver.__new__(AgenticSolver)

    def test_xml_tags(self):
        raw = "<analysis>analyze</analysis><plan>do it</plan><code>x=1</code><status>working</status>"
        p = self.solver._parse_response(raw)
        assert p.analysis == "analyze"
        assert p.plan == "do it"
        assert p.code == "x=1"
        assert not p.task_complete
        assert not p.parse_errors

    def test_complete_status(self):
        p = self.solver._parse_response("<code>x=1</code><status>complete</status>")
        assert p.task_complete
        assert p.code == "x=1"

    def test_working_status(self):
        p = self.solver._parse_response("<code>x=1</code><status>working</status>")
        assert not p.task_complete

    def test_missing_status_defaults_to_working(self):
        p = self.solver._parse_response("<code>x=1</code>")
        assert not p.task_complete

    def test_code_with_lang_attribute(self):
        p = self.solver._parse_response('<code lang="python">def solve(): pass</code>')
        assert "def solve" in p.code

    def test_greedy_code_match(self):
        """</code> inside generated code should not truncate extraction."""
        raw = '<code>print("</code>")\nreal_end</code>'
        p = self.solver._parse_response(raw)
        assert "real_end" in p.code

    def test_fenced_code_fallback(self):
        raw = "Here is my solution:\n```python\ndef solve(): return 42\n```"
        p = self.solver._parse_response(raw)
        assert "def solve" in p.code

    def test_raw_text_fallback(self):
        p = self.solver._parse_response("x = 1\nprint(x)")
        assert p.code == "x = 1\nprint(x)"
        assert len(p.parse_errors) == 1

    def test_confirmation_turn_no_code(self):
        """On confirmation turn, LLM may respond with just status."""
        p = self.solver._parse_response("<status>complete</status>")
        assert p.task_complete
        assert p.code == ""
        assert not p.parse_errors

    def test_status_with_analysis_no_code(self):
        raw = "<analysis>Looks good</analysis><status>complete</status>"
        p = self.solver._parse_response(raw)
        assert p.task_complete
        assert p.analysis == "Looks good"
        assert p.code == ""
        assert not p.parse_errors

    def test_raw_response_preserved(self):
        raw = "<code>x=1</code>"
        p = self.solver._parse_response(raw)
        assert p.raw_response == raw


# ---------------------------------------------------------------------------
# _run_all_pre_process tests
# ---------------------------------------------------------------------------


class TestRunAllPreProcess:
    def test_empty_injected_codes(self):
        solver = _make_solver()
        task = _make_task()
        result = solver._run_all_pre_process(task)
        assert result == ""

    def test_single_pre_process(self):
        ic = InjectedCode(
            pre_process='additional_context = "hello from depth 2"',
            source_depth=2,
        )
        solver = _make_solver(injected_codes=[ic])
        task = _make_task()
        result = solver._run_all_pre_process(task)
        assert result == "hello from depth 2"

    def test_reverse_order_with_outer_context(self):
        """Deepest (last in list) runs first; its output becomes outer_context for shallower."""
        ic_depth2 = InjectedCode(
            pre_process='additional_context = f"d2 sees: {outer_context}"',
            source_depth=2,
        )
        ic_depth3 = InjectedCode(
            pre_process='additional_context = "from_d3"',
            source_depth=3,
        )
        solver = _make_solver(injected_codes=[ic_depth2, ic_depth3])
        task = _make_task()
        result = solver._run_all_pre_process(task)
        # depth-3 runs first → "from_d3"
        # depth-2 runs with outer_context="from_d3" → "d2 sees: from_d3"
        assert "from_d3" in result
        assert "d2 sees: from_d3" in result

    def test_pre_process_error_skipped(self):
        ic = InjectedCode(
            pre_process="raise ValueError('boom')",
            source_depth=2,
        )
        solver = _make_solver(injected_codes=[ic])
        task = _make_task()
        result = solver._run_all_pre_process(task)
        assert result == ""


# ---------------------------------------------------------------------------
# _summarize_context tests
# ---------------------------------------------------------------------------


class TestSummarizeContext:
    async def test_too_short_returns_unchanged(self):
        solver = _make_solver(llm_responses=["summary"])
        messages = [
            {"role": "user", "content": "system"},
            {"role": "assistant", "content": "resp1"},
            {"role": "user", "content": "obs1"},
        ]
        result, tokens = await (
            solver._summarize_context(messages)
        )
        assert result == messages
        assert tokens == 0

    async def test_summarizes_when_long_enough(self):
        solver = _make_solver(llm_responses=["SUMMARY TEXT"])
        messages = [
            {"role": "user", "content": "system"},
            {"role": "assistant", "content": "r1"},
            {"role": "user", "content": "o1"},
            {"role": "assistant", "content": "r2"},
            {"role": "user", "content": "o2"},
            {"role": "assistant", "content": "r3"},
            {"role": "user", "content": "o3"},
        ]
        result, tokens = await (
            solver._summarize_context(messages)
        )
        # First message + summary + last 4 messages = 6
        assert len(result) == 6
        assert result[0] == messages[0]  # system preserved
        assert "SUMMARY TEXT" in result[1]["content"]
        assert result[-1] == messages[-1]  # last message preserved
        assert tokens == 100  # mock returns 100


# ---------------------------------------------------------------------------
# _estimate_tokens tests
# ---------------------------------------------------------------------------


class TestEstimateTokens:
    def test_basic(self):
        solver = _make_solver()
        messages = [{"role": "user", "content": "a" * 400}]
        assert solver._estimate_tokens(messages) == 100

    def test_multiple_messages(self):
        solver = _make_solver()
        messages = [
            {"role": "user", "content": "a" * 400},
            {"role": "assistant", "content": "b" * 800},
        ]
        assert solver._estimate_tokens(messages) == 300


# ---------------------------------------------------------------------------
# Agentic loop integration tests
# ---------------------------------------------------------------------------


class TestAgenticLoop:
    async def test_single_turn_success(self):
        """Agent produces working code on turn 1, signals complete, confirms."""
        solver = _make_solver(
            llm_responses=[
                "<analysis>ok</analysis><plan>solve</plan><code>x=1</code><status>complete</status>",
                "<status>complete</status>",  # confirmation
            ],
            executor_scores=[1.0],
            max_turns=5,
        )
        task = _make_task()
        trace, tokens = await (
            solver.execute(task)
        )
        assert trace.score == 1.0
        assert tokens > 0

    async def test_iterative_improvement(self):
        """Agent improves over multiple turns."""
        solver = _make_solver(
            llm_responses=[
                "<code>attempt1</code><status>working</status>",
                "<code>attempt2</code><status>working</status>",
                "<code>attempt3</code><status>complete</status>",
                "<status>complete</status>",  # confirmation
            ],
            executor_scores=[0.3, 0.6, 0.9],
            max_turns=5,
        )
        task = _make_task()
        trace, tokens = await (
            solver.execute(task)
        )
        assert trace.score == 0.9  # best-of-N

    async def test_best_of_n_tracking(self):
        """Even if later turns regress, best trace is returned."""
        solver = _make_solver(
            llm_responses=[
                "<code>good</code><status>working</status>",
                "<code>bad</code><status>working</status>",
            ],
            executor_scores=[0.8, 0.3],
            max_turns=2,
        )
        task = _make_task()
        trace, _ = await (
            solver.execute(task)
        )
        assert trace.score == 0.8  # kept the better one

    async def test_max_turns_termination(self):
        """Loop stops at max_turns."""
        solver = _make_solver(
            llm_responses=[
                "<code>x=1</code><status>working</status>",
                "<code>x=2</code><status>working</status>",
            ],
            executor_scores=[0.3, 0.4],
            max_turns=2,
        )
        task = _make_task()
        trace, _ = await (
            solver.execute(task)
        )
        assert trace.score == 0.4

    async def test_perfect_score_early_exit(self):
        """Loop exits immediately on score >= 1.0."""
        solver = _make_solver(
            llm_responses=[
                "<code>perfect</code><status>working</status>",
                "<code>never_reached</code><status>working</status>",
            ],
            executor_scores=[1.0, 0.5],
            max_turns=5,
        )
        task = _make_task()
        trace, _ = await (
            solver.execute(task)
        )
        assert trace.score == 1.0

    async def test_parse_error_feedback(self):
        """Parse errors are fed back to the LLM."""
        solver = _make_solver(
            llm_responses=[
                "malformed response with no tags at all and no code",
                "<code>fixed</code><status>working</status>",
            ],
            executor_scores=[0.5],
            max_turns=3,
        )
        task = _make_task()
        trace, _ = await (
            solver.execute(task)
        )
        assert trace.score == 0.5

    async def test_no_code_fallback_trace(self):
        """When no executable code is ever produced, fallback trace is returned."""
        solver = _make_solver(
            llm_responses=[
                "<analysis>thinking...</analysis><status>working</status>",
            ],
            executor_scores=[],
            max_turns=1,
        )
        task = _make_task()
        trace, _ = await (
            solver.execute(task)
        )
        assert trace.score == 0.0
        assert "no executable code" in trace.error_summary

    async def test_two_stage_completion_rejects_on_working(self):
        """If agent says complete then working, pending_completion resets."""
        solver = _make_solver(
            llm_responses=[
                "<code>v1</code><status>complete</status>",
                "<code>v2</code><status>working</status>",  # changed mind
                "<code>v3</code><status>working</status>",
            ],
            executor_scores=[0.5, 0.7, 0.8],
            max_turns=3,
        )
        task = _make_task()
        trace, _ = await (
            solver.execute(task)
        )
        assert trace.score == 0.8  # didn't confirm, kept going

    async def test_last_turn_completion_executes_code(self):
        """When agent signals complete on the last turn, code still executes."""
        solver = _make_solver(
            llm_responses=[
                "<code>v1</code><status>working</status>",
                "<code>v2</code><status>complete</status>",  # last turn, first complete
            ],
            executor_scores=[0.3, 0.9],
            max_turns=2,
        )
        task = _make_task()
        trace, _ = await (
            solver.execute(task)
        )
        assert trace.score == 0.9  # code was executed despite being last turn

    async def test_solve_returns_raw_script(self):
        """solve() strips library prefix from trace.script."""
        solver = _make_solver(
            llm_responses=[
                "<code>real_code</code><status>complete</status>",
                "<status>complete</status>",
            ],
            executor_scores=[0.8],
            max_turns=5,
        )
        task = _make_task()
        script, reasoning, tokens = await (
            solver.solve(task)
        )
        # No library injected (empty injected_codes), so script == raw code
        assert "real_code" in script

    async def test_confirmed_without_code_returns_fallback_trace(self):
        """Phantom-completion guard (6.3): confirming complete without ever
        executing code is REJECTED (not returned as 'confirmed'); it falls
        through to the honest max_turns no-executable-code fallback."""
        solver = _make_solver(
            llm_responses=["<status>complete</status>"] * 5,  # always complete, never code
            executor_scores=[],
            max_turns=5,
        )
        task = _make_task()
        trace, _ = await solver.execute(task)
        assert trace.score == 0.0
        assert trace.task_id == "test_task"
        # the honest fallback, NOT the old "confirmed without producing code"
        assert "no executable code" in trace.error_summary

    async def test_depth_set_on_trace(self):
        """execute() sets trace.depth correctly."""
        solver = _make_solver(
            llm_responses=[
                "<code>x=1</code><status>complete</status>",
                "<status>complete</status>",
            ],
            executor_scores=[0.5],
            max_turns=5,
        )
        solver.depth = 3
        task = _make_task()
        trace, _ = await (
            solver.execute(task)
        )
        assert trace.depth == 3

    async def test_nan_score_does_not_poison_best_trace(self):
        """If executor returns a NaN-scored trace first, a later finite-scored
        trace must still be picked as best.

        Regression: bare `>` comparison treated NaN as best (because
        `best_trace is None or NaN > anything` short-circuits on the None
        branch), and then no finite score could displace it because
        `finite > NaN` is False.
        """
        solver = _make_solver(
            llm_responses=[
                "<code>v1</code><status>working</status>",
                "<code>v2</code><status>working</status>",
                "<code>v3</code><status>working</status>",
            ],
            executor_scores=[float("nan"), 0.4, 0.7],
            max_turns=3,
        )
        task = _make_task()
        trace, _ = await (
            solver.execute(task)
        )
        assert trace.score == 0.7  # not NaN

    async def test_all_nan_scores_falls_back(self):
        """If every executor result is NaN, best_trace stays None and the
        loop falls back to the no-executable-code trace path."""
        solver = _make_solver(
            llm_responses=[
                "<code>v1</code><status>working</status>",
                "<code>v2</code><status>working</status>",
            ],
            executor_scores=[float("nan"), float("nan")],
            max_turns=2,
        )
        task = _make_task()
        trace, _ = await (
            solver.execute(task)
        )
        # Score is the default 0.0 from the fallback Trace constructor;
        # crucially, it is not NaN and the result is well-formed.
        import math
        assert math.isfinite(trace.score)
        assert trace.score == 0.0

    async def test_nan_score_during_completion_does_not_poison(self):
        """First-complete branch with NaN trace should not become best."""
        solver = _make_solver(
            llm_responses=[
                # Turn 1: complete with code (NaN score) — should NOT become best.
                "<code>v_nan</code><status>complete</status>",
                # Turn 2: confirmation request — agent revokes complete.
                "<code>v_finite</code><status>working</status>",
                # Turn 3: nothing — max_turns=3 ends here.
            ],
            executor_scores=[float("nan"), 0.6],
            max_turns=3,
        )
        task = _make_task()
        trace, _ = await (
            solver.execute(task)
        )
        assert trace.score == 0.6
