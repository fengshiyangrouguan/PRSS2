"""Byte-identity goldens + regression tests for the shared AdapterExecutor.

The three adapter executors (COBenchExecutor, TextClassificationExecutor,
OpenEvolveExecutor) wrap ``adapter.evaluate()`` into a ``Trace`` whose
``stdout`` / ``stderr`` / ``error_summary`` / ``eval_feedback`` fields feed Ω
prompt rendering and summary.json. The golden tests below pin EVERY Trace
field (except the wall-clock ``duration_s``) to literal expected values so any
consolidation of the executors is provably byte-identical.

Golden literals were captured from the pre-consolidation executor bodies:
  * CO-Bench / text-classification: ``.4f`` score format, feedback[:500]
    stderr, feedback[:200] error_summary, inner_* quad propagated.
  * OpenEvolve: ``.6f`` score format, structured JSON error-summary synthesis
    with feedback[:200] fallback.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from meta_n.core.meta_layer import TaskDescription
from meta_n.integrations.benchmark import AdapterExecutor, EvalResult
from meta_n.integrations.co_bench import COBenchExecutor
from meta_n.integrations.openevolve import OpenEvolveExecutor
from meta_n.integrations.text_classification import TextClassificationExecutor


def _executor_for(cls, result: EvalResult):
    adapter = MagicMock()
    adapter.evaluate = AsyncMock(return_value=result)
    return cls(adapter)


def _task() -> TaskDescription:
    return TaskDescription(
        task_id="golden_task",
        description="golden",
        metadata={"task_name": "golden"},
    )


async def _trace_fields(cls, result: EvalResult) -> dict:
    """Run the executor and return every Trace field except duration_s."""
    executor = _executor_for(cls, result)
    trace = await executor.execute("def solve(): ...", _task())
    fields = trace.model_dump()
    fields.pop("duration_s")
    return fields


def _expected(
    *,
    stdout: str,
    stderr: str,
    exit_code: int,
    success: bool,
    score: float,
    error_summary: str,
    eval_feedback: str,
) -> dict:
    """Full expected Trace dump (minus duration_s) with default-valued fields."""
    return {
        "task_id": "golden_task",
        "depth": 0,
        "script": "def solve(): ...",
        "stdout": stdout,
        "stderr": stderr,
        "exit_code": exit_code,
        "success": success,
        "score": score,
        "reasoning": "",
        "error_summary": error_summary,
        "eval_feedback": eval_feedback,
        "failure_class": "",
        "terminated_by": "",
        "inner_tokens": 0,
        "inner_prompt_tokens": 0,
        "inner_completion_tokens": 0,
        "inner_calls": 0,
        "utilities_available": [],
        "utilities_called": None,
        "utilities_call_counts": {},
        "command_count": 0,
        "parse_failure_turns": 0,
    }


_SUCCESS = EvalResult(success=True, score=0.85, raw_score=0.85, feedback="OK")
_FAIL_FEEDBACK = "x" * 300
_FAILURE = EvalResult(success=False, score=0.0, feedback=_FAIL_FEEDBACK)
_OE_JSON_FEEDBACK = json.dumps({
    "correctness_score": 0,
    "baseline_comparison": {"num_valid_solutions": 3, "num_total_trials": 10},
})
_FAILURE_OE_JSON = EvalResult(success=False, score=0.0, feedback=_OE_JSON_FEEDBACK)


class TestExecutorGoldenByteIdentity:
    """Literal-field goldens for all three executors (F124/F185 proof)."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("cls", [COBenchExecutor, TextClassificationExecutor])
    async def test_dot4_success(self, cls):
        assert await _trace_fields(cls, _SUCCESS) == _expected(
            stdout="score=0.8500\nraw_score=0.8500",
            stderr="",
            exit_code=0,
            success=True,
            score=0.85,
            error_summary="",
            eval_feedback="OK",
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("cls", [COBenchExecutor, TextClassificationExecutor])
    async def test_dot4_failure_truncation(self, cls):
        assert await _trace_fields(cls, _FAILURE) == _expected(
            stdout="score=0.0000\nraw_score=0.0000",
            stderr="x" * 300,  # feedback[:500] — under the cap, kept whole
            exit_code=1,
            success=False,
            score=0.0,
            error_summary="x" * 200,  # feedback[:200]
            eval_feedback="x" * 300,
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("cls", [COBenchExecutor, TextClassificationExecutor])
    async def test_dot4_failure_json_feedback_is_raw(self, cls):
        """The .4f executors do NOT synthesize from JSON — raw truncation only."""
        assert await _trace_fields(cls, _FAILURE_OE_JSON) == _expected(
            stdout="score=0.0000\nraw_score=0.0000",
            stderr=_OE_JSON_FEEDBACK,
            exit_code=1,
            success=False,
            score=0.0,
            error_summary=_OE_JSON_FEEDBACK[:200],
            eval_feedback=_OE_JSON_FEEDBACK,
        )

    @pytest.mark.asyncio
    async def test_oe_success(self):
        assert await _trace_fields(OpenEvolveExecutor, _SUCCESS) == _expected(
            stdout="score=0.850000\nraw_score=0.850000",
            stderr="",
            exit_code=0,
            success=True,
            score=0.85,
            error_summary="",
            eval_feedback="OK",
        )

    @pytest.mark.asyncio
    async def test_oe_failure_non_json_fallback(self):
        assert await _trace_fields(OpenEvolveExecutor, _FAILURE) == _expected(
            stdout="score=0.000000\nraw_score=0.000000",
            stderr="x" * 300,
            exit_code=1,
            success=False,
            score=0.0,
            error_summary="x" * 200,
            eval_feedback="x" * 300,
        )

    @pytest.mark.asyncio
    async def test_oe_failure_json_synthesis(self):
        assert await _trace_fields(OpenEvolveExecutor, _FAILURE_OE_JSON) == _expected(
            stdout="score=0.000000\nraw_score=0.000000",
            stderr=_OE_JSON_FEEDBACK,
            exit_code=1,
            success=False,
            score=0.0,
            error_summary="correctness=0 (3/10 valid)",
            eval_feedback=_OE_JSON_FEEDBACK,
        )


class TestAdapterExecutorConsolidation:
    """Structural pins for the shared-base consolidation (F124/F185)."""

    @pytest.mark.parametrize(
        "cls", [COBenchExecutor, TextClassificationExecutor, OpenEvolveExecutor]
    )
    def test_executors_subclass_the_shared_base(self, cls):
        # All three adapter executors are thin AdapterExecutor subclasses:
        # no execute/__init__ overrides — behavior differences live only in
        # the _score_fmt / _error_summary hooks (goldens above pin the bytes).
        assert issubclass(cls, AdapterExecutor)
        assert cls.execute is AdapterExecutor.execute
        assert cls.__init__ is AdapterExecutor.__init__

    @pytest.mark.parametrize("cls", [COBenchExecutor, TextClassificationExecutor])
    def test_dot4_executors_use_pure_base_defaults(self, cls):
        # CO-Bench / text-classification take the base hooks unchanged.
        assert cls._score_fmt == ".4f"
        assert cls._error_summary is AdapterExecutor._error_summary

    def test_default_error_summary_is_feedback_truncation(self):
        base = AdapterExecutor(MagicMock())
        ok = EvalResult(success=True, score=1.0, feedback="fine")
        bad = EvalResult(success=False, score=0.0, feedback="y" * 300)
        assert base._error_summary(ok) == ""
        assert base._error_summary(bad) == "y" * 200

    @pytest.mark.asyncio
    async def test_oe_executor_propagates_inner_usage(self):
        """F185 drift fix: OpenEvolveExecutor no longer drops the inner_* quad.

        OE/ARC adapters never populate inner_* today (so goldens above are
        byte-identical), but an evolved program that calls llm() must not lose
        its inner-token accounting if a future adapter populates the fields.
        """
        result = EvalResult(
            success=True, score=0.5, raw_score=0.5, feedback="OK",
            inner_tokens=70, inner_prompt_tokens=40,
            inner_completion_tokens=30, inner_calls=3,
        )
        executor = _executor_for(OpenEvolveExecutor, result)
        trace = await executor.execute("def run(): ...", _task())
        assert trace.inner_tokens == 70
        assert trace.inner_prompt_tokens == 40
        assert trace.inner_completion_tokens == 30
        assert trace.inner_calls == 3
