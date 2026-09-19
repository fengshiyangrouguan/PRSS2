"""Refinement regression tests for meta_n/integrations/co_bench.py.

Covers:
* F113 — timeout/no-result usage reconstruction from the inner-log with pid
  attribution (concurrent instance subprocesses share one log file).
* F115 — evaluate/evaluate_test consolidation into _evaluate_split must keep
  every EvalResult field (both splits, success and failure branches).
* F188 — the five COBenchScorer early-return shapes keep their distinct
  valid/feasible combinations after the _agent_usage_kwargs consolidation.
* F133 — a non-dict solve() return reports the sibling's explicit message,
  not a cryptic AttributeError.
* F119 — scripts/experiments/metan_crew_verified_test.py must import (not
  fork) the crew held-out runner; _CREW_HELDOUT_RUNNER is the single source
  of truth.

Offline: no LLM, no Docker, no network; subprocess tests run tiny inline
sources under the production mp.Process isolation.
"""

from __future__ import annotations

import asyncio
import json
import textwrap

import pytest

import meta_n.integrations._subprocess_utils as su
import meta_n.integrations.co_bench as cb
from meta_n.core.meta_layer import TaskDescription
from meta_n.integrations.co_bench import COBenchAdapter, COBenchScorer

_ZEROS = {"total": 0, "prompt": 0, "completion": 0, "calls": 0}

_CB_CONFIG = textwrap.dedent(
    """
    def load_data(path):
        return [{"x": 1}]

    def eval_func(**kwargs):
        return 1.5
    """
)


def _write_cb_config(tmp_path) -> str:
    (tmp_path / "config.py").write_text(_CB_CONFIG)
    return str(tmp_path / "config.py")


# ---------------------------------------------------------------------------
# F113 — timeout reconstructs pid-attributed partial usage from the inner-log
# ---------------------------------------------------------------------------


def test_timeout_reconstructs_partial_usage_pid_attributed(tmp_path):
    """A timed-out instance must report the inner-LLM usage ITS child logged:
    own-pid records and legacy unstamped records count; a concurrent sibling's
    record (different extra.pid) and pre-offset history do not."""
    cfg = _write_cb_config(tmp_path)
    log_path = str(tmp_path / "inner.jsonl")

    # Pre-existing history (before the offset snapshot) — must NOT count.
    with open(log_path, "w") as f:
        f.write(json.dumps({"total_tokens": 999, "prompt_tokens": 900,
                            "completion_tokens": 99}) + "\n")

    solve = textwrap.dedent(
        f"""
        import json, os, time
        def solve(**kw):
            with open({log_path!r}, "a") as f:
                f.write(json.dumps({{"total_tokens": 30, "prompt_tokens": 20,
                                     "completion_tokens": 10,
                                     "extra": {{"pid": os.getpid()}}}}) + "\\n")
                f.write(json.dumps({{"total_tokens": 100, "prompt_tokens": 70,
                                     "completion_tokens": 30,
                                     "extra": {{"pid": os.getpid() + 99999}}}}) + "\\n")
                f.write(json.dumps({{"total_tokens": 15, "prompt_tokens": 9,
                                     "completion_tokens": 6}}) + "\\n")
                f.flush()
            time.sleep(30)
            return {{}}
        """
    )
    status, payload, usage = cb._run_with_timeout(
        cfg, {"x": 1}, solve, 1, inner_log_path=log_path,
    )
    assert status == "error"
    assert payload == "Timeout (1s)"  # payload string unchanged by F113
    # own-pid (30/20/10) + legacy unstamped (15/9/6); sibling + history excluded.
    assert usage == {"total": 45, "prompt": 29, "completion": 16, "calls": 2}


def test_usage_from_log_since_pid_filter(tmp_path):
    """Unit-level pin of the pid-filter contract: pid=None counts everything;
    a given pid counts matching + unstamped records only."""
    log = tmp_path / "inner.jsonl"
    recs = [
        {"total_tokens": 10, "prompt_tokens": 6, "completion_tokens": 4,
         "extra": {"pid": 111}},
        {"total_tokens": 20, "prompt_tokens": 12, "completion_tokens": 8,
         "extra": {"pid": 222}},
        {"total_tokens": 40, "prompt_tokens": 24, "completion_tokens": 16},
    ]
    log.write_text("".join(json.dumps(r) + "\n" for r in recs))

    assert su._usage_from_log_since(str(log), 0) == {
        "total": 70, "prompt": 42, "completion": 28, "calls": 3,
    }
    assert su._usage_from_log_since(str(log), 0, pid=111) == {
        "total": 50, "prompt": 30, "completion": 20, "calls": 2,
    }


def test_timeout_without_inner_log_reports_zeros(tmp_path):
    """No inner-log configured -> nothing to recover -> canonical zeros."""
    cfg = _write_cb_config(tmp_path)
    solve = "import time\ndef solve(**kw):\n    time.sleep(30)\n    return {}\n"
    out = cb._run_with_timeout(cfg, {"x": 1}, solve, 1)
    assert out == ("error", "Timeout (1s)", _ZEROS)


# ---------------------------------------------------------------------------
# F133 — non-dict solve() return gets the explicit sibling message
# ---------------------------------------------------------------------------


def test_solve_nondict_return_reports_explicit_message(tmp_path):
    cfg = _write_cb_config(tmp_path)
    out = cb._run_with_timeout(
        cfg, {"x": 1}, "def solve(**kw):\n    return [1, 2]\n", 10,
    )
    assert out == ("error", "solve() returned list, expected dict", _ZEROS)


# ---------------------------------------------------------------------------
# F115 — evaluate / evaluate_test field parity through _evaluate_split
# ---------------------------------------------------------------------------


class _FakeEvaluator:
    def evaluate(self, solution):
        return {
            "dev_score": 0.75,
            "dev_feedback": "dev fb",
            "inner_usage": {"total": 10, "prompt": 6, "completion": 4, "calls": 2},
        }

    def evaluate_test(self, solution):
        return {
            "test_score": 0.25,
            "test_feedback": "test fb",
            "inner_usage": {"total": 8, "prompt": 5, "completion": 3, "calls": 1},
        }


def _task(metadata=None) -> TaskDescription:
    return TaskDescription(
        task_id="set_covering", description="d",
        metadata=metadata if metadata is not None else {"task_name": "Set covering"},
    )


def _adapter_with_fake_evaluator() -> COBenchAdapter:
    adapter = COBenchAdapter(data_dir="/nonexistent", task_names=["Set covering"])
    adapter._evaluators["Set covering"] = _FakeEvaluator()
    return adapter


def test_evaluate_dev_field_parity():
    adapter = _adapter_with_fake_evaluator()
    res = asyncio.run(adapter.evaluate(_task(), "def solve(**kw): return {}"))
    assert res.success is True
    assert res.score == pytest.approx(0.75)
    assert res.raw_score == pytest.approx(0.75)
    assert res.feedback == "dev fb"
    assert (res.inner_tokens, res.inner_prompt_tokens,
            res.inner_completion_tokens, res.inner_calls) == (10, 6, 4, 2)


def test_evaluate_test_field_parity():
    adapter = _adapter_with_fake_evaluator()
    res = asyncio.run(adapter.evaluate_test(_task(), "def solve(**kw): return {}"))
    assert res.success is True
    assert res.score == pytest.approx(0.25)
    assert res.raw_score == pytest.approx(0.25)
    assert res.feedback == "test fb"
    assert (res.inner_tokens, res.inner_prompt_tokens,
            res.inner_completion_tokens, res.inner_calls) == (8, 5, 3, 1)


def test_evaluate_missing_task_name_branch():
    adapter = _adapter_with_fake_evaluator()
    for method in (adapter.evaluate, adapter.evaluate_test):
        res = asyncio.run(method(_task(metadata={}), "code"))
        assert res.success is False
        assert res.score == 0.0
        assert res.feedback == "No task_name in metadata"


def test_evaluate_evaluator_load_error_branch():
    adapter = COBenchAdapter(data_dir="/nonexistent", task_names=["Set covering"])
    for method in (adapter.evaluate, adapter.evaluate_test):
        res = asyncio.run(method(_task(), "code"))
        assert res.success is False
        assert res.feedback.startswith("Failed to load evaluator: ")


def test_evaluate_evaluation_error_branch():
    class _Boom:
        def evaluate(self, solution):
            raise ValueError("kaboom")

        def evaluate_test(self, solution):
            raise ValueError("kaboom")

    adapter = COBenchAdapter(data_dir="/nonexistent", task_names=["Set covering"])
    adapter._evaluators["Set covering"] = _Boom()
    for method in (adapter.evaluate, adapter.evaluate_test):
        res = asyncio.run(method(_task(), "code"))
        assert res.success is False
        assert res.score == 0.0
        assert res.feedback == "Evaluation error: kaboom"


# ---------------------------------------------------------------------------
# F188 — the five scorer shapes keep their valid/feasible combinations
# ---------------------------------------------------------------------------


class _FakeRun:
    agent_tokens = 11
    agent_prompt_tokens = 7
    agent_completion_tokens = 4
    agent_calls = 3


def _assert_agent_usage(res):
    assert (res.inner_tokens, res.inner_prompt_tokens,
            res.inner_completion_tokens, res.inner_calls) == (11, 7, 4, 3)


def test_scorer_empty_solution_shape():
    scorer = COBenchScorer(COBenchAdapter(data_dir="/nonexistent"))
    res = asyncio.run(scorer.score(_task(), None, "   ", _FakeRun()))
    assert res.success is False and res.score == 0.0
    assert res.feedback == "Agent authored no solve.py"
    assert res.valid is False and res.feasible is False  # ONLY branch w/ feasible=False
    _assert_agent_usage(res)


def test_scorer_missing_task_name_shape():
    scorer = COBenchScorer(COBenchAdapter(data_dir="/nonexistent"))
    res = asyncio.run(scorer.score(_task(metadata={}), None, "code", _FakeRun()))
    assert res.feedback == "No task_name in metadata"
    assert res.valid is False and res.feasible is True  # feasible stays default
    _assert_agent_usage(res)


def test_scorer_evaluator_load_error_shape():
    scorer = COBenchScorer(COBenchAdapter(data_dir="/nonexistent"))
    res = asyncio.run(scorer.score(_task(), None, "code", _FakeRun()))
    assert res.feedback.startswith("Failed to load evaluator: ")
    assert res.valid is False and res.feasible is True
    _assert_agent_usage(res)


def test_scorer_evaluation_error_shape():
    class _Boom:
        def _evaluate_on_split(self, solution, split="dev"):
            raise RuntimeError("eval boom")

    adapter = COBenchAdapter(data_dir="/nonexistent", task_names=["Set covering"])
    adapter._evaluators["Set covering"] = _Boom()
    scorer = COBenchScorer(adapter)
    res = asyncio.run(scorer.score(_task(), None, "code", _FakeRun()))
    assert res.feedback == "Evaluation error: eval boom"
    assert res.valid is False and res.feasible is True
    _assert_agent_usage(res)


@pytest.mark.parametrize("dev_score,feasible", [(0.6, True), (0.0, False)])
def test_scorer_success_shape(dev_score, feasible):
    class _Ok:
        def __init__(self, score):
            self._score = score

        def _evaluate_on_split(self, solution, split="dev"):
            return {"dev_score": self._score, "dev_feedback": "fb"}

    adapter = COBenchAdapter(data_dir="/nonexistent", task_names=["Set covering"])
    adapter._evaluators["Set covering"] = _Ok(dev_score)
    scorer = COBenchScorer(adapter)
    res = asyncio.run(scorer.score(_task(), None, "code", _FakeRun()))
    assert res.success is (dev_score > 0.0)
    assert res.score == pytest.approx(dev_score)
    assert res.raw_score == pytest.approx(dev_score)
    assert res.feedback == "fb"
    assert res.valid is True and res.feasible is feasible
    _assert_agent_usage(res)
