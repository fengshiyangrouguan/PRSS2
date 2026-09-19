"""Tests for signal quality improvements — error extraction, tail-truncation, metadata categories."""

import pytest

from meta_n.core.meta_layer import InjectedCode, Trace, _head_tail, _tail
from meta_n.core.omega import OmegaEngine
from meta_n.integrations.terminal_bench import TerminalBenchExecutor


# ---------------------------------------------------------------------------
# Truncation helpers
# ---------------------------------------------------------------------------


class TestTail:
    def test_short_text_unchanged(self):
        assert _tail("hello", 100) == "hello"

    def test_exact_limit(self):
        assert _tail("abcde", 5) == "abcde"

    def test_truncated_with_ellipsis(self):
        result = _tail("a" * 100, 10)
        assert result.startswith("...")
        assert len(result) == 10

    def test_tail_content_preserved(self):
        text = "noise " * 50 + "REAL_ERROR"
        result = _tail(text, 20)
        assert "REAL_ERROR" in result

    def test_empty(self):
        assert _tail("", 100) == ""


class TestHeadTail:
    def test_short_text_unchanged(self):
        assert _head_tail("hello", 3, 3) == "hello"

    def test_exact_limit(self):
        assert _head_tail("abcde", 3, 2) == "abcde"

    def test_truncated_with_marker(self):
        text = "HEAD" + "x" * 100 + "TAIL"
        result = _head_tail(text, 4, 4)
        assert result.startswith("HEAD")
        assert result.endswith("TAIL")
        assert "[truncated]" in result

    def test_empty(self):
        assert _head_tail("", 10, 10) == ""


# ---------------------------------------------------------------------------
# _extract_error_summary
# ---------------------------------------------------------------------------


class TestExtractErrorSummary:
    def test_traceback_at_tail(self):
        stderr = (
            "debconf: delaying package configuration\n" * 10
            + "Traceback (most recent call last):\n"
            "  File '/app/main.py', line 5\n"
            "ImportError: cannot import name 'BicScore'"
        )
        result = TerminalBenchExecutor._extract_error_summary(stderr)
        assert "Traceback" in result
        assert "ImportError" in result
        assert "debconf" not in result

    def test_error_keyword(self):
        stderr = "debconf noise\nError: command not found\n"
        result = TerminalBenchExecutor._extract_error_summary(stderr)
        assert "Error: command not found" in result

    def test_failed_keyword(self):
        stderr = "apt noise\nFAILED tests/test_foo.py::test_bar\n"
        result = TerminalBenchExecutor._extract_error_summary(stderr)
        assert "FAILED" in result

    def test_command_not_found(self):
        stderr = "debconf noise\n/tmp/solve.sh: line 5: sqlite3: command not found\n"
        result = TerminalBenchExecutor._extract_error_summary(stderr)
        assert "command not found" in result

    def test_no_marker_returns_tail(self):
        stderr = "A" * 500
        result = TerminalBenchExecutor._extract_error_summary(stderr, max_len=100)
        assert len(result) <= 100
        assert result == "A" * 100  # tail of all-A string

    def test_empty_stderr(self):
        assert TerminalBenchExecutor._extract_error_summary("") == ""

    def test_max_len_respected(self):
        stderr = "Traceback (most recent call last):\n" + "x" * 500
        result = TerminalBenchExecutor._extract_error_summary(stderr, max_len=50)
        assert len(result) <= 50


# ---------------------------------------------------------------------------
# _extract_eval_feedback
# ---------------------------------------------------------------------------


class TestExtractEvalFeedback:
    def test_pytest_failures_section(self):
        raw = (
            "apt-get noise\n" * 50
            + "============================== FAILURES ==============================\n"
            "________________ test_json_data ________________\n"
            "assert 0 > 6\n"
            "======================== 1 failed ========================\n"
        )
        result = TerminalBenchExecutor._extract_eval_feedback(raw)
        assert "FAILURES" in result
        assert "assert 0 > 6" in result
        assert "apt-get noise" not in result

    def test_pytest_summary_line(self):
        raw = "pip install output\n" * 100 + "3 failed, 2 passed in 5.2s\n"
        result = TerminalBenchExecutor._extract_eval_feedback(raw)
        assert "3 failed" in result

    def test_traceback_in_verifier(self):
        raw = "uv install stuff\n" * 50 + "Traceback (most recent call last):\n  File 'test.py'\nAssertionError\n"
        result = TerminalBenchExecutor._extract_eval_feedback(raw)
        assert "Traceback" in result

    def test_no_markers_returns_tail(self):
        raw = "noise\n" * 1000
        result = TerminalBenchExecutor._extract_eval_feedback(raw, max_len=200)
        assert len(result) <= 200

    def test_short_output_returned_as_is(self):
        raw = "PASSED test_foo\n"
        result = TerminalBenchExecutor._extract_eval_feedback(raw)
        assert result == "PASSED test_foo"

    def test_empty_output(self):
        assert TerminalBenchExecutor._extract_eval_feedback("") == ""

    def test_failed_keyword(self):
        raw = "downloading packages...\n" * 50 + "FAILED tests/test_output.py\n"
        result = TerminalBenchExecutor._extract_eval_feedback(raw)
        assert "FAILED" in result
        assert "downloading" not in result


# ---------------------------------------------------------------------------
# Omega trace formatting — tail truncation
# ---------------------------------------------------------------------------


class TestOmegaTraceFormatting:
    def test_stderr_tail_in_formatted_traces(self):
        engine = OmegaEngine.__new__(OmegaEngine)
        traces = [
            Trace(
                task_id="test_task",
                success=False,
                stderr="debconf noise\n" * 100 + "ImportError: foo",
            )
        ]
        result = engine._format_raw_traces(traces)
        assert "ImportError: foo" in result
        # debconf noise may or may not appear depending on 800-char limit
        # but the important thing is the error IS there

    def test_eval_feedback_tail_in_formatted_traces(self):
        engine = OmegaEngine.__new__(OmegaEngine)
        traces = [
            Trace(
                task_id="test_task",
                success=False,
                eval_feedback="pip install noise\n" * 100 + "FAILED test_data",
            )
        ]
        result = engine._format_raw_traces(traces)
        assert "FAILED test_data" in result

    def test_stdout_head_tail(self):
        engine = OmegaEngine.__new__(OmegaEngine)
        traces = [
            Trace(
                task_id="test_task",
                success=True,
                score=1.0,
                stdout="HEAD_CONTENT" + "x" * 2000 + "TAIL_CONTENT",
            )
        ]
        result = engine._format_raw_traces(traces)
        assert "HEAD_CONTENT" in result
        assert "TAIL_CONTENT" in result
        assert "[truncated]" in result


# ---------------------------------------------------------------------------
# Task categorization with metadata
# ---------------------------------------------------------------------------


class TestMetadataCategorization:
    def test_metadata_category_preferred(self):
        assert OmegaEngine._categorize_task(
            "unknown-task", metadata={"category": "scientific-computing"}
        ) == "scientific-computing"

    def test_metadata_empty_falls_back(self):
        assert OmegaEngine._categorize_task(
            "job_shop_scheduling", metadata={"category": ""}
        ) == "scheduling"

    def test_metadata_none_falls_back(self):
        assert OmegaEngine._categorize_task(
            "job_shop_scheduling", metadata=None
        ) == "scheduling"

    def test_no_metadata_no_keyword(self):
        assert OmegaEngine._categorize_task("unknown-task") == "other"

    def test_cobench_keywords_still_work(self):
        assert OmegaEngine._categorize_task("tsp_route_opt") == "routing/path"
        assert OmegaEngine._categorize_task("knapsack_10") == "knapsack"


# ---------------------------------------------------------------------------
# Validate against real pilot data (if available)
# ---------------------------------------------------------------------------


class TestRealPilotData:
    """Validate the new extraction against actual TB2 pilot traces."""

    @pytest.fixture
    def pilot_trace(self):
        """Load a real trace if pilot data exists."""
        import json
        from pathlib import Path

        path = Path("experiments/tb2_pilot_04/archive/gen0_seed/traces/bn-fit-modify.json")
        if not path.exists():
            pytest.skip("Pilot data not available")
        return json.load(open(path))

    def test_error_summary_captures_real_error(self, pilot_trace):
        stderr = pilot_trace.get("stderr", "")
        if not stderr:
            pytest.skip("No stderr in trace")
        result = TerminalBenchExecutor._extract_error_summary(stderr)
        # bn-fit-modify has "error: externally-managed-environment" (PEP 668)
        # or "ImportError" / "Traceback" — any error marker should be captured
        assert any(kw in result.lower() for kw in ["error", "traceback", "command not found", "failed"]), \
            f"Expected error marker in: {result[:150]}"
