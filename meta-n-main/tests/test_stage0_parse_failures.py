"""S0.3 unit gate — `parse_failure_turns` counter (+ S0.2 agentic command_count).

A scripted 2-parse-fail-then-success agentic loop must report
``parse_failure_turns == 2`` and ``command_count == 1`` on ``best_trace``; a
clean loop must stay ``0``.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

from meta_n.core.agentic_solver import AgenticSolver
from meta_n.core.meta_layer import InjectedCode, TaskDescription, Trace


def _make_solver(llm_responses, executor_scores, *, injected_codes=None,
                 max_turns=8, code_library_is_live=True):
    counter = {"llm": 0, "exec": 0}

    async def mock_complete(messages, temperature=None, max_tokens=None, **kw):
        idx = min(counter["llm"], len(llm_responses) - 1)
        counter["llm"] += 1
        return llm_responses[idx], 100

    async def mock_execute(script, task, timeout=30):
        idx = min(counter["exec"], len(executor_scores) - 1)
        score = executor_scores[idx]
        counter["exec"] += 1
        return Trace(task_id=task.task_id, script=script, score=score,
                     success=score >= 0.999, exit_code=0,
                     stdout=f"score={score}", eval_feedback="ok")

    llm_client = AsyncMock()
    llm_client.complete = mock_complete
    executor = AsyncMock()
    executor.execute = mock_execute
    return AgenticSolver(
        llm_client=llm_client,
        executor=executor,
        injected_codes=injected_codes or [],
        max_turns=max_turns,
        code_library_is_live=code_library_is_live,
    )


_CODE_COMPLETE = '<code lang="python">\nprint(1)\n</code>\n<status>complete</status>'
_NO_CODE = "<status>working</status>"          # status set -> no Level-3 code fallback
_CONFIRM = "<status>complete</status>"


async def test_two_parse_fails_then_success():
    """no-code, no-code, code+complete, confirm -> 2 parse fails, 1 execute."""
    solver = _make_solver(
        llm_responses=[_NO_CODE, _NO_CODE, _CODE_COMPLETE, _CONFIRM],
        executor_scores=[0.7],
    )
    trace, _ = await solver.execute(TaskDescription(task_id="t", description="d"))
    assert trace.parse_failure_turns == 2
    assert trace.command_count == 1


async def test_clean_run_zero_parse_failures():
    """A first-turn code+complete then confirm -> no parse failures."""
    solver = _make_solver(
        llm_responses=[_CODE_COMPLETE, _CONFIRM],
        executor_scores=[0.8],
    )
    trace, _ = await solver.execute(TaskDescription(task_id="t", description="d"))
    assert trace.parse_failure_turns == 0
    assert trace.command_count == 1


async def test_agentic_demoted_helpers_keep_none():
    """No live helpers -> utilities_called stays None on the agentic path."""
    solver = _make_solver(
        llm_responses=[_CODE_COMPLETE, _CONFIRM],
        executor_scores=[0.8],
        injected_codes=[InjectedCode(code_library={"h": "def h(): ..."})],
        code_library_is_live=False,   # demoted -> merged_py zeroed in __init__
    )
    trace, _ = await solver.execute(TaskDescription(task_id="t", description="d"))
    assert trace.utilities_called is None
    assert trace.command_count == 1


async def test_agentic_live_helper_measured():
    """A live helper the solver calls is measured on the agentic path."""
    code = '<code lang="python">\nresult = myhelper(3)\n</code>\n<status>complete</status>'
    solver = _make_solver(
        llm_responses=[code, _CONFIRM],
        executor_scores=[0.8],
        injected_codes=[InjectedCode(code_library={"myhelper": "def myhelper(x):\n    return x"})],
        code_library_is_live=True,
    )
    trace, _ = await solver.execute(TaskDescription(task_id="t", description="d"))
    assert trace.utilities_available == ["myhelper"]
    assert trace.utilities_called == ["myhelper"]
    assert trace.command_count == 1
