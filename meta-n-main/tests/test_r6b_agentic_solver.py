"""§6b agentic-solver remedies — F063 / F064 / F065 / F079.

F063: ``token_budget`` is a chars//4 CONTEXT bound; cumulative REAL spend
      (outer + inner tokens) is capped by the new default-OFF ``spend_budget``.
F064: phantom guard splits never-executed (byte-identical scold + S0.3 count)
      from executed-but-all-non-finite (accurate message, no S0.3 count).
F065: a prompt-invited ``<status>working</status>`` decline of a completion
      check is nudged for code, not scolded as INVALID / counted as a parse
      failure; the non-confirmation bare-working scold is byte-identical.
F079: cap hit with an unconfirmed completion pending gets an honest suffixed
      label ("max_turns_unconfirmed_complete" / "token_budget_unconfirmed_
      complete") that still substring-matches its base class in classify_error.

Harness: scripted-LLM/AsyncMock pattern from tests/test_stage0_parse_failures.py.
"""

from __future__ import annotations

import math
from unittest.mock import AsyncMock

from meta_n.core.agentic_prompts import COMPLETION_CONFIRMATION
from meta_n.core.agentic_solver import AgenticSolver
from meta_n.core.meta_layer import TaskDescription, Trace, classify_error


def _make_solver(
    llm_responses,
    executor_scores=(),
    *,
    max_turns=4,
    token_budget=100_000,
    spend_budget=None,
    tokens_per_call=100,
    inner_tokens=0,
    captured=None,
):
    """AgenticSolver with a scripted LLM and executor.

    ``captured`` (dict) receives the live message list under "messages" —
    the loop mutates one list object, so after the run it holds every
    appended user/assistant message.
    """
    counter = {"llm": 0, "exec": 0}

    async def mock_complete(messages, temperature=None, max_tokens=None, **kw):
        if captured is not None:
            captured["messages"] = messages
        idx = min(counter["llm"], len(llm_responses) - 1)
        counter["llm"] += 1
        return llm_responses[idx], tokens_per_call

    async def mock_execute(script, task, timeout=30):
        idx = min(counter["exec"], len(executor_scores) - 1)
        score = executor_scores[idx]
        counter["exec"] += 1
        return Trace(
            task_id=task.task_id,
            script=script,
            score=score,
            success=bool(score >= 0.999),
            exit_code=0,
            stdout=f"score={score}",
            eval_feedback="ok",
            inner_tokens=inner_tokens,
        )

    llm_client = AsyncMock()
    llm_client.complete = mock_complete
    executor = AsyncMock()
    executor.execute = mock_execute
    return AgenticSolver(
        llm_client=llm_client,
        executor=executor,
        injected_codes=[],
        max_turns=max_turns,
        token_budget=token_budget,
        spend_budget=spend_budget,
    )


def _task() -> TaskDescription:
    return TaskDescription(task_id="t", description="d")


def _user_texts(captured) -> list[str]:
    return [m["content"] for m in captured["messages"] if m["role"] == "user"]


_CODE_WORKING = '<code lang="python">\nx = 1\n</code>\n<status>working</status>'
_CODE_COMPLETE = '<code lang="python">\nx = 1\n</code>\n<status>complete</status>'
_BARE_WORKING = "<status>working</status>"
_BARE_COMPLETE = "<status>complete</status>"


# ---------------------------------------------------------------------------
# F063 — spend_budget (real-spend cap, default OFF)
# ---------------------------------------------------------------------------


class TestSpendBudget:
    async def test_spend_budget_none_is_byte_identical(self):
        """Default (None) keeps HEAD behavior: 4 turns, 3600 tokens, max_turns."""
        solver = _make_solver(
            llm_responses=[_CODE_WORKING] * 4,
            executor_scores=[0.1, 0.2, 0.3, 0.4],
            max_turns=4,
            tokens_per_call=900,
        )
        assert solver.spend_budget is None
        result = await solver._agentic_loop(_task(), "")
        assert result.turns_used == 4
        assert result.total_tokens == 3600
        assert result.terminated_by == "max_turns"

    async def test_spend_budget_stops_loop_and_labels(self):
        """Real spend >= spend_budget stops BEFORE the next LLM call."""
        solver = _make_solver(
            llm_responses=[_CODE_WORKING] * 4,
            executor_scores=[0.1, 0.2, 0.3, 0.4],
            max_turns=4,
            tokens_per_call=900,
            spend_budget=1000,
        )
        result = await solver._agentic_loop(_task(), "")
        # turn 1: spend 0 < 1000 -> call (900); turn 2: 900 < 1000 -> call
        # (1800); turn 3: 1800 >= 1000 -> stop without a third call.
        assert result.turns_used == 2
        assert result.total_tokens == 1800
        assert result.terminated_by == "spend_budget"
        assert result.best_trace.terminated_by == "spend_budget"

    async def test_spend_budget_counts_inner_tokens(self):
        """Inner llm()/llm_batch() usage trips the cap when outer alone would not."""
        solver = _make_solver(
            llm_responses=[_CODE_WORKING] * 4,
            executor_scores=[0.1, 0.2, 0.3, 0.4],
            max_turns=4,
            tokens_per_call=100,  # outer alone: 400 max, never trips 1000
            inner_tokens=800,
            spend_budget=1000,
        )
        result = await solver._agentic_loop(_task(), "")
        # turn 3 precheck: outer 200 + inner 1600 >= 1000 -> stop.
        assert result.turns_used == 2
        assert result.terminated_by == "spend_budget"
        assert result.best_trace.terminated_by == "spend_budget"


# ---------------------------------------------------------------------------
# F064 — phantom guard: never-executed vs executed-but-all-non-finite
# ---------------------------------------------------------------------------


class TestPhantomVsNanCompletion:
    async def test_true_phantom_message_and_count_byte_identical(self):
        """Never-executed double-complete keeps the HEAD scold + S0.3 count."""
        captured = {}
        solver = _make_solver(
            llm_responses=[_BARE_COMPLETE] * 3,
            max_turns=3,
            captured=captured,
        )
        result = await solver._agentic_loop(_task(), "")
        assert result.parse_failure_turns == 1
        assert result.terminated_by != "confirmed"
        assert any(
            "no executable code has run yet" in t for t in _user_texts(captured)
        )

    async def test_nan_completion_gets_accurate_message_not_phantom(self):
        """Executed-but-all-NaN completion: accurate message, no S0.3 count."""
        captured = {}
        solver = _make_solver(
            llm_responses=[_CODE_COMPLETE, _BARE_COMPLETE, _CODE_WORKING],
            executor_scores=[float("nan"), 0.5],
            max_turns=3,
            captured=captured,
        )
        result = await solver._agentic_loop(_task(), "")
        texts = _user_texts(captured)
        assert any(
            "none of your executed runs produced a valid (finite) score" in t
            for t in texts
        )
        assert not any("no executable code has run yet" in t for t in texts)
        assert result.parse_failure_turns == 0
        # The loop continued and a later finite turn became best.
        assert result.best_trace.score == 0.5

    async def test_all_nan_fallback_error_summary(self):
        """Loop-end fallback text discriminates all-NaN from never-executed."""
        # Executed every turn, all NaN -> numeric-failure summary.
        solver = _make_solver(
            llm_responses=[_CODE_WORKING] * 2,
            executor_scores=[float("nan"), float("nan")],
            max_turns=2,
        )
        result = await solver._agentic_loop(_task(), "")
        assert "not finite" in result.best_trace.error_summary
        assert math.isfinite(result.best_trace.score)
        # Never executed -> byte-identical HEAD summary.
        solver = _make_solver(
            llm_responses=["<analysis>thinking</analysis><status>working</status>"],
            max_turns=1,
        )
        result = await solver._agentic_loop(_task(), "")
        assert (
            result.best_trace.error_summary
            == "AgenticSolver produced no executable code"
        )


# ---------------------------------------------------------------------------
# F065 — completion-check decline without code is not a parse failure
# ---------------------------------------------------------------------------


class TestCompletionDecline:
    def test_confirmation_prompt_asks_for_code_with_decline(self):
        assert "include the complete fixed code" in COMPLETION_CONFIRMATION
        assert "and fix the issues." not in COMPLETION_CONFIRMATION

    async def test_decline_after_confirmation_not_a_parse_failure(self):
        captured = {}
        solver = _make_solver(
            llm_responses=[_CODE_COMPLETE, _BARE_WORKING, _CODE_WORKING],
            executor_scores=[0.3, 0.6],
            max_turns=3,
            captured=captured,
        )
        result = await solver._agentic_loop(_task(), "")
        texts = _user_texts(captured)
        assert not any("INVALID: no <code>" in t for t in texts)
        assert any(
            "You declined completion but did not include the fixed code" in t
            for t in texts
        )
        assert result.parse_failure_turns == 0
        assert result.best_trace.score == 0.6

    async def test_decline_with_analysis_and_status_also_carved_out(self):
        captured = {}
        decline = "<analysis>needs work</analysis><status>working</status>"
        solver = _make_solver(
            llm_responses=[_CODE_COMPLETE, decline, _CODE_WORKING],
            executor_scores=[0.3, 0.6],
            max_turns=3,
            captured=captured,
        )
        result = await solver._agentic_loop(_task(), "")
        texts = _user_texts(captured)
        assert not any("INVALID: no <code>" in t for t in texts)
        assert any("You declined completion" in t for t in texts)
        assert result.parse_failure_turns == 0
        assert result.best_trace.score == 0.6

    async def test_bare_working_without_confirmation_still_scolded(self):
        """No prior completion check -> byte-identical INVALID scold + count."""
        captured = {}
        solver = _make_solver(
            llm_responses=[_BARE_WORKING, _CODE_COMPLETE, _BARE_COMPLETE],
            executor_scores=[0.7],
            max_turns=4,
            captured=captured,
        )
        result = await solver._agentic_loop(_task(), "")
        assert any(
            t.startswith("INVALID: no <code> block found. Wasted turn.")
            for t in _user_texts(captured)
        )
        assert result.parse_failure_turns == 1
        assert result.terminated_by == "confirmed"


# ---------------------------------------------------------------------------
# F079 — cap-with-pending-completion fidelity labels
# ---------------------------------------------------------------------------


class TestUnconfirmedCompleteLabels:
    async def test_last_turn_first_complete_labeled_unconfirmed(self):
        solver = _make_solver(
            llm_responses=[_CODE_WORKING, _CODE_COMPLETE],
            executor_scores=[0.3, 0.9],
            max_turns=2,
        )
        result = await solver._agentic_loop(_task(), "")
        assert result.best_trace.score == 0.9  # code still executed at the cap
        assert result.terminated_by == "max_turns_unconfirmed_complete"
        assert result.best_trace.terminated_by == "max_turns_unconfirmed_complete"

    async def test_budget_starved_awaiting_confirmation_labeled(self):
        # Turn 1's assistant response is large enough that the 95% context
        # precheck fires on turn 2, while the confirmation answer is pending.
        big_code = (
            '<code lang="python">\n# ' + "x" * 12_000 + "\n</code>\n"
            "<status>complete</status>"
        )
        solver = _make_solver(
            llm_responses=[big_code],
            executor_scores=[0.4],
            max_turns=3,
            token_budget=2000,
        )
        result = await solver._agentic_loop(_task(), "")
        assert result.turns_used == 1
        assert result.terminated_by == "token_budget_unconfirmed_complete"
        assert (
            result.best_trace.terminated_by == "token_budget_unconfirmed_complete"
        )

    async def test_spend_starved_awaiting_confirmation_labeled(self):
        # Mirror of test_budget_starved_awaiting_confirmation_labeled for the
        # spend cap: turn 1 executes code and declares complete (confirmation
        # pending); the spend guard fires on turn 2 -> suffixed label that
        # preserves the unconfirmed-completion signal.
        solver = _make_solver(
            llm_responses=[_CODE_COMPLETE],
            executor_scores=[0.4],
            max_turns=3,
            tokens_per_call=900,
            spend_budget=500,
        )
        result = await solver._agentic_loop(_task(), "")
        assert result.turns_used == 1
        assert result.terminated_by == "spend_budget_unconfirmed_complete"
        assert (
            result.best_trace.terminated_by
            == "spend_budget_unconfirmed_complete"
        )

    async def test_both_caps_crossed_spend_guard_keeps_suffix(self):
        # Spend AND token caps both crossed while a confirmation is pending:
        # the spend guard runs FIRST, so it must itself carry the suffixed
        # label rather than pre-empting the pending signal with plain
        # "spend_budget".
        big_complete = (
            '<code lang="python">\n# ' + "x" * 12_000 + "\n</code>\n"
            "<status>complete</status>"
        )
        solver = _make_solver(
            llm_responses=[big_complete],
            executor_scores=[0.4],
            max_turns=3,
            token_budget=2000,  # context estimate > 95% on turn 2
            tokens_per_call=900,
            spend_budget=500,  # real spend also over its cap on turn 2
        )
        result = await solver._agentic_loop(_task(), "")
        assert result.turns_used == 1
        assert result.terminated_by == "spend_budget_unconfirmed_complete"
        assert (
            result.best_trace.terminated_by
            == "spend_budget_unconfirmed_complete"
        )

    async def test_plain_caps_keep_legacy_labels(self):
        # No completion signal at the turn cap -> plain "max_turns".
        solver = _make_solver(
            llm_responses=[_CODE_WORKING] * 2,
            executor_scores=[0.3, 0.4],
            max_turns=2,
        )
        result = await solver._agentic_loop(_task(), "")
        assert result.terminated_by == "max_turns"
        assert result.best_trace.terminated_by == "max_turns"
        # No completion signal at the context cap -> plain "token_budget".
        big_working = (
            '<code lang="python">\n# ' + "x" * 12_000 + "\n</code>\n"
            "<status>working</status>"
        )
        solver = _make_solver(
            llm_responses=[big_working],
            executor_scores=[0.3],
            max_turns=3,
            token_budget=2000,
        )
        result = await solver._agentic_loop(_task(), "")
        assert result.terminated_by == "token_budget"
        assert result.best_trace.terminated_by == "token_budget"

    def test_classify_error_on_new_labels(self):
        # Substring contract: the suffixed labels CONTAIN their base label.
        t = Trace(task_id="t", terminated_by="max_turns_unconfirmed_complete")
        assert classify_error(t) == "Turn starvation"
        # token_budget variant classifies exactly like plain token_budget on
        # the same error text (no structured branch for either -> text scan).
        t1 = Trace(
            task_id="t",
            terminated_by="token_budget_unconfirmed_complete",
            error_summary="stopped early",
        )
        t2 = Trace(
            task_id="t",
            terminated_by="token_budget",
            error_summary="stopped early",
        )
        assert classify_error(t1) == classify_error(t2)
        # spend_budget variant: same substring contract, same text-scan class.
        s1 = Trace(
            task_id="t",
            terminated_by="spend_budget_unconfirmed_complete",
            error_summary="stopped early",
        )
        s2 = Trace(
            task_id="t",
            terminated_by="spend_budget",
            error_summary="stopped early",
        )
        assert "spend_budget" in s1.terminated_by
        assert classify_error(s1) == classify_error(s2)


# ---------------------------------------------------------------------------
# F060 — agentic solve-time temperature (plumbing stage; the config/CLI wiring
# is pinned in tests/test_r6b_main.py + test_r6b_evolutionary_orchestrator.py)
# ---------------------------------------------------------------------------


class TestAgenticTemperature:
    async def _solve_turn_temperatures(self, temperature=None) -> list:
        """Run a scripted 2-turn loop and capture the ``temperature`` kwarg the
        SOLVE turns pass to ``complete()`` (no summarize call fires here)."""
        temps: list = []

        async def mock_complete(messages, temperature=None, max_tokens=None, **kw):
            temps.append(temperature)
            return _CODE_WORKING, 100

        async def mock_execute(script, task, timeout=30):
            return Trace(
                task_id=task.task_id, script=script, score=0.5, success=False,
                exit_code=0, stdout="score=0.5", eval_feedback="ok",
            )

        llm_client = AsyncMock()
        llm_client.complete = mock_complete
        executor = AsyncMock()
        executor.execute = mock_execute
        kw = {} if temperature is None else {"temperature": temperature}
        solver = AgenticSolver(
            llm_client=llm_client, executor=executor, injected_codes=[],
            max_turns=2, **kw,
        )
        await solver._agentic_loop(_task(), "")
        return temps

    async def test_temperature_default_is_the_historical_pin(self):
        # Byte-identity anchor: no kwarg == the 0.7 constructor pin on every
        # solve turn (mirrors the F060 repro output).
        assert await self._solve_turn_temperatures() == [0.7, 0.7]

    async def test_temperature_override_reaches_every_solve_turn(self):
        assert await self._solve_turn_temperatures(0.2) == [0.2, 0.2]
