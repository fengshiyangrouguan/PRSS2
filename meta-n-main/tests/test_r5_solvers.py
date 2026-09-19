"""R5 refine tests.

Covers:
  * llm_batch partial failure — completed siblings' usage is recorded
    (tracker + inner JSONL) and still-pending siblings are cancelled before
    the exception propagates (meta_n/core/llm_helpers.py).
  * F075 --classify-balanced-json-fallback live-site wiring — the flag now
    gates a recovery pass (fenced -> flat regex -> balanced scan) over a
    solve() that returned a string instead of a dict, on every classify
    parse path (sequential subprocess, in-process, parallel chunks). Flag
    OFF stays byte-identical to the flag-less behaviour.
"""

import asyncio
import json
import time
from types import SimpleNamespace

import pytest

from meta_n.core.llm_helpers import LLMUsageTracker, make_llm_func
from meta_n.core.solver import extract_case_json
from meta_n.integrations.text_classification import (
    TextClassificationAdapter,
    _coerce_predictions,
    _run_chunk_inprocess,
    _run_solve_with_timeout,
)

# ---------------------------------------------------------------------------
# llm_batch partial failure
# ---------------------------------------------------------------------------


class _PartialFailureClient:
    """Fake LLMClient: 'fast' completes immediately, 'fail' raises at +0.05s,
    'slow' would complete at +0.5s unless cancelled."""

    def __init__(self):
        self.slow_completed = False
        self.config = SimpleNamespace(model="fake-model")

    async def complete_with_breakdown(
        self, messages, temperature=0.3, max_tokens=512, _suppress_io_log=False,
    ):
        prompt = messages[0]["content"]
        if prompt == "fast":
            return "fast-resp", 1, 2, 3
        if prompt == "fail":
            await asyncio.sleep(0.05)
            raise RuntimeError("boom")
        await asyncio.sleep(0.5)
        self.slow_completed = True
        return "slow-resp", 10, 20, 30


class TestBatchPartialFailure:
    def test_completed_sibling_recorded_and_pending_cancelled(self, tmp_path):
        client = _PartialFailureClient()
        tracker = LLMUsageTracker()
        log_path = tmp_path / "inner.jsonl"
        _, llm_batch = make_llm_func(client, tracker, log_path=log_path)

        with pytest.raises(RuntimeError, match="boom"):
            llm_batch(["fast", "fail", "slow"])

        assert tracker.call_count == 1
        assert tracker.total_tokens == 3
        assert tracker.prompt_tokens == 1
        assert tracker.completion_tokens == 2

        records = [
            json.loads(line) for line in log_path.read_text().splitlines()
        ]
        assert len(records) == 1
        assert records[0]["messages"][0]["content"] == "fast"
        assert records[0]["response"] == "fast-resp"
        assert records[0]["total_tokens"] == 3

        # The in-flight 'slow' sibling was cancelled before llm_batch raised
        # (and never completes afterwards on the daemon loop).
        assert client.slow_completed is False
        time.sleep(0.7)
        assert client.slow_completed is False
        assert tracker.call_count == 1

    def test_all_success_unchanged(self, tmp_path):
        client = _PartialFailureClient()
        tracker = LLMUsageTracker()
        log_path = tmp_path / "inner.jsonl"
        _, llm_batch = make_llm_func(client, tracker, log_path=log_path)

        results = llm_batch(["fast", "fast"])

        assert results == ["fast-resp", "fast-resp"]
        assert tracker.call_count == 2
        assert tracker.total_tokens == 6
        assert len(log_path.read_text().splitlines()) == 2


# ---------------------------------------------------------------------------
# F075 — shared extraction chain
# ---------------------------------------------------------------------------

_NESTED_RESPONSE = (
    "Here are the predictions:\n"
    '{"case_0": "psoriasis {plaque}", "case_1": "common cold"}\n'
    "Done."
)

_STRING_SOLUTION = (
    f"RESPONSE = {_NESTED_RESPONSE!r}\n"
    "def solve(cases, labels, few_shot):\n"
    "    return RESPONSE\n"
)

_EXAMPLES = [
    {"text": "itchy plaques on elbows", "label": "psoriasis {plaque}"},
    {"text": "sneezing and cough", "label": "common cold"},
]

_LABELS = ["psoriasis {plaque}", "common cold"]

_RECOVERED = {"case_0": "psoriasis {plaque}", "case_1": "common cold"}


class TestExtractCaseJson:
    def test_fenced_wins(self):
        assert extract_case_json('x\n```json\n{"case_0": "a"}\n```') == (
            '{"case_0": "a"}'
        )

    def test_flat_regex(self):
        assert extract_case_json('pre {"case_0": "a"} post') == (
            '{"case_0": "a"}'
        )

    def test_nested_requires_balanced_fallback(self):
        assert extract_case_json(_NESTED_RESPONSE) is None
        assert extract_case_json(_NESTED_RESPONSE, balanced_fallback=True) == (
            '{"case_0": "psoriasis {plaque}", "case_1": "common cold"}'
        )

    def test_no_json_returns_none(self):
        assert extract_case_json("no json here", balanced_fallback=True) is None


class TestCoercePredictions:
    def test_flag_off_passthrough(self):
        assert _coerce_predictions(_NESTED_RESPONSE, False) is _NESTED_RESPONSE

    def test_non_str_passthrough(self):
        d = {"case_0": "x"}
        assert _coerce_predictions(d, True) is d
        assert _coerce_predictions(42, True) == 42

    def test_unrecoverable_string_passthrough(self):
        s = "no json here"
        assert _coerce_predictions(s, True) is s

    def test_invalid_recovered_json_passthrough(self):
        s = "```json\n{not valid json}\n```"
        assert _coerce_predictions(s, True) is s

    def test_balanced_recovery(self):
        assert _coerce_predictions(_NESTED_RESPONSE, True) == _RECOVERED

    def test_fenced_recovery(self):
        s = 'text\n```json\n{"case_0": "flu"}\n```'
        assert _coerce_predictions(s, True) == {"case_0": "flu"}


# ---------------------------------------------------------------------------
# F075 — end-to-end on the live classify parse paths
# ---------------------------------------------------------------------------


def _adapter(balanced: bool, eval_workers: int = 1) -> TextClassificationAdapter:
    adapter = TextClassificationAdapter(
        "symptom2disease",
        subprocess_isolate=False,
        balanced_json_fallback=balanced,
        eval_workers=eval_workers,
    )
    adapter._data = {
        "train": [{"text": "t", "label": "common cold"}],
        "val": _EXAMPLES,
        "test": [],
        "labels": _LABELS,
        "metric": "accuracy",
        "language": "en",
    }
    return adapter


class TestClassifyLiveSite:
    def test_adapter_default_flag_off(self):
        assert TextClassificationAdapter(
            "symptom2disease"
        )._balanced_json_fallback is False

    def test_inprocess_flag_off_string_return_errors(self):
        result = _adapter(False)._run_solve_code(
            _STRING_SOLUTION, _EXAMPLES, "accuracy"
        )
        assert result.success is False
        assert result.score == 0.0
        assert "solve() returned str, expected dict" in result.feedback

    def test_inprocess_flag_on_recovers_predictions(self):
        result = _adapter(True)._run_solve_code(
            _STRING_SOLUTION, _EXAMPLES, "accuracy"
        )
        assert result.success is True
        assert result.score == 1.0

    def test_chunk_inprocess_flag_off_raises(self):
        with pytest.raises(TypeError, match="expected dict"):
            _run_chunk_inprocess(
                _STRING_SOLUTION, {"case_0": "x"}, _LABELS, [], None
            )

    def test_chunk_inprocess_flag_on_recovers(self):
        preds, usage = _run_chunk_inprocess(
            _STRING_SOLUTION, {"case_0": "x"}, _LABELS, [], None,
            balanced_json_fallback=True,
        )
        assert preds == _RECOVERED
        assert usage["calls"] == 0

    def test_parallel_flag_off_errors(self):
        adapter = _adapter(False, eval_workers=2)
        examples = _EXAMPLES + [{"text": "z", "label": "common cold"}]
        result = adapter._run_solve_code(_STRING_SOLUTION, examples, "accuracy")
        assert result.success is False
        assert "solve() returned str, expected dict" in result.feedback

    def test_parallel_flag_on_recovers(self):
        adapter = _adapter(True, eval_workers=2)
        examples = _EXAMPLES + [{"text": "z", "label": "common cold"}]
        result = adapter._run_solve_code(_STRING_SOLUTION, examples, "accuracy")
        assert result.success is True
        assert result.score == pytest.approx(2 / 3)

    @pytest.mark.parametrize(
        "flag,expected_status", [(False, "error"), (True, "ok")]
    )
    def test_subprocess_path_honors_flag(self, flag, expected_status):
        status, payload, usage = _run_solve_with_timeout(
            _STRING_SOLUTION,
            {"case_0": "itchy plaques", "case_1": "sneezing"},
            _LABELS,
            [],
            None,
            60,
            balanced_json_fallback=flag,
        )
        assert status == expected_status
        if flag:
            assert payload == _RECOVERED
        else:
            assert "solve() returned str, expected dict" in payload
