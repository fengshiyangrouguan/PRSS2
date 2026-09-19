"""Regression tests for the C4_solvers_llm refinement wave.

Covers: F056/F187 (llm_helpers factory consolidation), F057 (shared
extract_fenced_block), F058 (bash fence tag-order fix), F070/F191
(agentic execute/track consolidation), F071 (dead summarize branch),
F074 (LocalExecutor host-exec warning), F168 (phantom_rejected removal),
plus the F014 agentic-side marker-strip delegation.
"""

from __future__ import annotations

import inspect
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from meta_n.core.agentic_solver import AgenticSolver
from meta_n.core.base_executor import LocalExecutor
from meta_n.core.llm_client import LLMClient
from meta_n.core.llm_helpers import (
    LLMUsageTracker,
    make_llm_func,
    make_llm_func_from_config,
)
from meta_n.core.meta_layer import (
    TaskDescription,
    Trace,
    _strip_library_prefix_for_scan,
)
from meta_n.core.solver import Layer1Solver, extract_fenced_block


# ---------------------------------------------------------------------------
# Shared fixtures/helpers
# ---------------------------------------------------------------------------


def _make_task(task_id: str = "refine_task") -> TaskDescription:
    return TaskDescription(task_id=task_id, description="Solve X")


def _make_agentic_solver(
    *,
    llm_responses: list[str],
    executor_traces: list[Trace] | None = None,
    executor_raises: bool = False,
    max_turns: int = 5,
) -> AgenticSolver:
    """AgenticSolver with a scripted LLM and a scripted executor."""
    llm_client = AsyncMock()
    counts = {"llm": 0, "exec": 0}

    async def mock_complete(messages, temperature=None, max_tokens=None):
        idx = min(counts["llm"], len(llm_responses) - 1)
        counts["llm"] += 1
        return llm_responses[idx], 100

    llm_client.complete = mock_complete

    executor = AsyncMock()

    async def mock_execute(script, task, timeout=30):
        if executor_raises:
            raise RuntimeError("executor boom")
        idx = min(counts["exec"], len(executor_traces) - 1)
        counts["exec"] += 1
        trace = executor_traces[idx]
        trace.script = script
        return trace

    executor.execute = mock_execute

    return AgenticSolver(
        llm_client=llm_client,
        executor=executor,
        injected_codes=[],
        solver_language="python",
        max_turns=max_turns,
        token_budget=100_000,
    )


# ---------------------------------------------------------------------------
# F058 — bash fence tag order: labeled ``sh`` must beat the "" wildcard
# ---------------------------------------------------------------------------


class TestBashFenceTagOrder:
    def setup_method(self):
        self.solver = Layer1Solver(MagicMock(), language="bash")

    def test_two_sh_blocks_returns_first_sh_block(self):
        """Two ```sh blocks: must return the FIRST block, not the prose
        between them (the "" wildcard used to capture the interstice)."""
        response = (
            "Here is my solution:\n"
            "```sh\necho first\n```\n"
            "Some prose explaining the second option.\n"
            "```sh\necho second\n```\n"
        )
        assert self.solver._extract_bash_script(response) == "echo first"

    def test_plain_then_sh_returns_sh_block(self):
        """A labeled ```sh block wins over a preceding plain ``` block."""
        response = (
            "```\nsome plain text\n```\n"
            "```sh\necho labeled\n```\n"
        )
        assert self.solver._extract_bash_script(response) == "echo labeled"

    def test_single_sh_block_unchanged(self):
        response = "text\n```sh\necho only\n```\nmore text"
        assert self.solver._extract_bash_script(response) == "echo only"

    def test_bash_still_wins_over_sh(self):
        response = "```sh\necho sh\n```\n```bash\necho bash\n```"
        assert self.solver._extract_bash_script(response) == "echo bash"


# ---------------------------------------------------------------------------
# F057 — shared extract_fenced_block helper
# ---------------------------------------------------------------------------


class TestExtractFencedBlock:
    def test_tag_priority_order(self):
        response = "```py\npy_code\n```\n```python\npython_code\n```"
        assert extract_fenced_block(response, ("python", "py")) == "python_code"
        assert extract_fenced_block(response, ("py", "python")) == "py_code"

    def test_dotall_multiline(self):
        response = "```json\n{\n  \"a\": 1\n}\n```"
        assert extract_fenced_block(response, ("json", "")) == '{\n  "a": 1\n}'

    def test_empty_block_returns_empty_string(self):
        response = "```bash\n```"
        assert extract_fenced_block(response, ("bash",)) == ""

    def test_no_match_returns_none(self):
        assert extract_fenced_block("no fences here", ("bash", "sh", "")) is None

    def test_equivalence_with_layer1_extractors(self):
        """The three Layer1Solver extractors must agree with the helper +
        their own documented fallbacks on a shared corpus."""
        solver = Layer1Solver(MagicMock())
        corpus = [
            "```bash\necho hi\n```",
            "```sh\necho sh\n```",
            "```\nplain\n```",
            "no fence at all",
            "```python\nx = 1\n```",
            "```py\ny = 2\n```",
            "```json\n{\"case_1\": \"a\"}\n```",
            "prefix {\"case_1\": \"a\", \"case_2\": \"b\"} suffix",
            "```bash\n```",
        ]
        for response in corpus:
            bash = extract_fenced_block(response, ("bash", "sh", ""))
            expected_bash = bash if bash is not None else response.strip()
            assert solver._extract_bash_script(response) == expected_bash

            py = extract_fenced_block(response, ("python", "py", ""))
            expected_py = py if py is not None else response.strip()
            assert solver._extract_python_code(response) == expected_py

            js = extract_fenced_block(response, ("json", ""))
            if js is not None:
                assert solver._extract_json(response) == js

    def test_json_raw_object_fallback_preserved(self):
        solver = Layer1Solver(MagicMock())
        response = 'Sure: {"case_1": "flu", "case_2": "cold"} done'
        assert solver._extract_json(response) == '{"case_1": "flu", "case_2": "cold"}'

    def test_agentic_level2_fenced_fallback(self):
        solver = _make_agentic_solver(llm_responses=["x"], executor_traces=[Trace(task_id="t")])
        parsed = solver._parse_response("Some text\n```python\nx = 42\n```")
        assert parsed.code == "x = 42"

    def test_agentic_level2_empty_block_suppresses_level3(self):
        """An empty fenced block is still assigned ("") so Level-3
        whole-response fallback interplay is unchanged."""
        solver = _make_agentic_solver(llm_responses=["x"], executor_traces=[Trace(task_id="t")])
        parsed = solver._parse_response("<analysis>thinking</analysis>\n```python\n```")
        assert parsed.code == ""
        assert parsed.analysis == "thinking"


# ---------------------------------------------------------------------------
# F056/F187 — llm_helpers factory consolidation
# ---------------------------------------------------------------------------


_CONFIG = {
    "base_url": "http://test", "api_key": "key", "model": "m",
    "temperature": 0.7, "max_tokens": 1024, "max_retries": 1,
}


def _stub_client(response: str = "resp", pt: int = 10, ct: int = 5, tt: int = 15):
    client = MagicMock()
    client.config.model = "m"
    client.complete_with_breakdown = AsyncMock(return_value=(response, pt, ct, tt))
    return client


class TestLlmFuncFactoriesLocked:
    def test_identical_docstrings_and_signatures(self):
        llm_a, batch_a = make_llm_func(_stub_client())
        llm_b, batch_b = make_llm_func_from_config(dict(_CONFIG))

        assert llm_a.__doc__ == llm_b.__doc__
        assert batch_a.__doc__ == batch_b.__doc__
        assert inspect.signature(llm_a) == inspect.signature(llm_b)
        assert inspect.signature(batch_a) == inspect.signature(batch_b)

        # Pin the runtime API advertised to evolved solve() code
        # (LANG_INSTRUCTIONS_PYTHON: llm(prompt, *, temperature=0.3, max_tokens=512)).
        sig = inspect.signature(llm_a)
        assert list(sig.parameters) == ["prompt", "temperature", "max_tokens"]
        assert sig.parameters["temperature"].kind is inspect.Parameter.KEYWORD_ONLY
        assert sig.parameters["temperature"].default == 0.3
        assert sig.parameters["max_tokens"].kind is inspect.Parameter.KEYWORD_ONLY
        assert sig.parameters["max_tokens"].default == 512

        bsig = inspect.signature(batch_a)
        assert list(bsig.parameters) == [
            "prompts", "temperature", "max_tokens", "max_concurrent",
        ]
        assert bsig.parameters["max_concurrent"].default == 10

    def test_same_tracker_deltas_and_iolog_record_both_paths(self, tmp_path):
        # Path A: eager main-process factory.
        with patch("meta_n.core.llm_helpers.LLMIOLogger") as logger_cls_a:
            logger_a = MagicMock()
            logger_cls_a.return_value = logger_a
            tracker_a = LLMUsageTracker()
            llm_a, _ = make_llm_func(
                _stub_client(), tracker_a, log_path=tmp_path / "a.jsonl"
            )
            out_a = llm_a("hello")
            record_a = logger_a.log.call_args.kwargs

        # Path B: lazy subprocess factory (same model + usage numbers).
        with patch("meta_n.core.llm_helpers.LLMIOLogger") as logger_cls_b, \
             patch("meta_n.core.llm_client.AsyncOpenAI"), \
             patch.object(LLMClient, "complete_with_breakdown",
                          new_callable=AsyncMock, return_value=("resp", 10, 5, 15)):
            logger_b = MagicMock()
            logger_cls_b.return_value = logger_b
            tracker_b = LLMUsageTracker()
            llm_b, _ = make_llm_func_from_config(
                dict(_CONFIG), tracker_b, log_path=tmp_path / "b.jsonl"
            )
            out_b = llm_b("hello")
            record_b = logger_b.log.call_args.kwargs

        assert out_a == out_b == "resp"
        assert (
            tracker_a.total_tokens, tracker_a.prompt_tokens,
            tracker_a.completion_tokens, tracker_a.call_count,
        ) == (
            tracker_b.total_tokens, tracker_b.prompt_tokens,
            tracker_b.completion_tokens, tracker_b.call_count,
        ) == (15, 10, 5, 1)
        # Byte-identical inner.jsonl record fields across the two factories.
        assert record_a == record_b
        assert set(record_a) == {
            "messages", "response", "model",
            "prompt_tokens", "completion_tokens", "total_tokens", "extra",
        }
        # extra.pid stamps the writing process so _usage_from_log_since(pid=...)
        # can exclude sibling subprocesses sharing one log file.
        import os

        assert record_a["extra"] == {"pid": os.getpid()}


# ---------------------------------------------------------------------------
# F070/F191 — single execute/track path for completion + normal branches
# ---------------------------------------------------------------------------


class TestExecuteAndTrackConsolidation:
    async def test_completion_and_normal_paths_share_tracking(self):
        """Turn 1 (normal path) and turn 2 (first-'complete' path) each
        execute once; command_count and inner_* totals must sum across both."""
        t1 = Trace(task_id="refine_task", score=0.5, inner_tokens=10,
                   inner_prompt_tokens=6, inner_completion_tokens=4, inner_calls=1)
        t2 = Trace(task_id="refine_task", score=0.8, inner_tokens=20,
                   inner_prompt_tokens=12, inner_completion_tokens=8, inner_calls=2)
        solver = _make_agentic_solver(
            llm_responses=[
                '<code lang="python">print(1)</code>\n<status>working</status>',
                '<code lang="python">print(2)</code>\n<status>complete</status>',
                "<status>complete</status>",
            ],
            executor_traces=[t1, t2],
            max_turns=5,
        )
        result = await solver._agentic_loop(_make_task(), "")

        assert result.terminated_by == "confirmed"
        assert result.command_count == 2
        assert result.best_trace.score == 0.8
        assert result.best_trace.inner_tokens == 30
        assert result.best_trace.inner_prompt_tokens == 18
        assert result.best_trace.inner_completion_tokens == 12
        assert result.best_trace.inner_calls == 3

    async def test_executor_error_counts_command_on_both_paths(self):
        """Executor exceptions still count as commands (S0.2) and yield a
        fallback error trace, on the normal AND completion paths."""
        solver = _make_agentic_solver(
            llm_responses=[
                '<code lang="python">a</code>\n<status>working</status>',
                '<code lang="python">b</code>\n<status>complete</status>',
                "<status>complete</status>",
            ],
            executor_raises=True,
            max_turns=5,
        )
        result = await solver._agentic_loop(_make_task(), "")
        assert result.command_count == 2
        # Executor-error trace has score 0.0 (finite) → becomes best_trace.
        assert "Executor error" in result.best_trace.error_summary


# ---------------------------------------------------------------------------
# F071 — _summarize_context boundary (dead-branch deletion)
# ---------------------------------------------------------------------------


class TestSummarizeBoundary:
    async def test_exactly_six_messages_summarizes(self):
        """len == 6 passes the guard: exactly one LLM call, result is
        [first, summary, *last-4] (len 6)."""
        solver = _make_agentic_solver(llm_responses=["x"], executor_traces=[Trace(task_id="t")])
        solver.llm_client.complete = AsyncMock(return_value=("the summary", 42))
        messages = [
            {"role": "user", "content": f"msg{i}"} for i in range(6)
        ]
        compressed, tokens = await solver._summarize_context(messages)

        solver.llm_client.complete.assert_awaited_once()
        assert tokens == 42
        assert len(compressed) == 6
        assert compressed[0] == messages[0]
        assert compressed[1]["content"].startswith("## Prior work summary")
        assert "the summary" in compressed[1]["content"]
        assert compressed[2:] == messages[-4:]


# ---------------------------------------------------------------------------
# F168 — phantom_rejected counter removed; rejection behavior intact
# ---------------------------------------------------------------------------


class TestPhantomRejectionBehavior:
    async def test_phantom_completion_counts_parse_failure_and_continues(self):
        solver = _make_agentic_solver(
            llm_responses=["<status>complete</status>"] * 3,
            executor_traces=[],
            max_turns=3,
        )
        result = await solver._agentic_loop(_make_task(), "")
        # Turn 2's confirmation with no executed code → phantom rejection.
        assert result.parse_failure_turns == 1
        assert result.turns_used == 3  # loop continued after the rejection
        assert result.terminated_by != "confirmed"
        assert "no executable code" in result.best_trace.error_summary

    def test_dead_counter_gone(self):
        source = inspect.getsource(AgenticSolver._agentic_loop)
        assert "phantom_rejected" not in source


# ---------------------------------------------------------------------------
# F074 — LocalExecutor warns once about host execution
# ---------------------------------------------------------------------------


class TestLocalExecutorHostWarning:
    def test_warns_exactly_once(self, caplog, monkeypatch):
        # monkeypatch restores the class flag afterwards — mutating it bare
        # would re-arm/consume the once-per-process warning for later tests.
        monkeypatch.setattr(LocalExecutor, "_warned_host_exec", False)
        with caplog.at_level(logging.WARNING, logger="meta_n.core.base_executor"):
            LocalExecutor()
            LocalExecutor()
        warnings = [
            r for r in caplog.records
            if r.name == "meta_n.core.base_executor" and "HOST" in r.getMessage()
        ]
        assert len(warnings) == 1
        assert LocalExecutor._warned_host_exec is True


# ---------------------------------------------------------------------------
# F014 (agentic side) — solve() delegates to the canonical marker-strip helper
# ---------------------------------------------------------------------------


class TestSolveStripDelegation:
    async def test_solve_strips_library_prefix_via_canonical_helper(self):
        marked_script = (
            "helper_code()\n# --- end injected code library ---\nreal_code()"
        )
        trace = Trace(task_id="refine_task", score=0.9, script=marked_script)
        solver = _make_agentic_solver(
            llm_responses=["x"], executor_traces=[trace]
        )
        solver.execute = AsyncMock(return_value=(trace, 7))

        script, _reasoning, tokens = await solver.solve(_make_task())
        assert script == _strip_library_prefix_for_scan(marked_script)
        assert script == "real_code()"
        assert tokens == 7
