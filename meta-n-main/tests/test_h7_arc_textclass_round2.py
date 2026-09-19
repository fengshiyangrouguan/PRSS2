"""Round-2 silent-void fixes for the H7-arc-textclass group.

Covers three CONFIRMED round-2 bugs from
``docs/metan_silent_void_audit_round2.md``:

  * R2-ARC-1 (arc_agi) — ``split_type() == "proxy"`` has no selection-path
    consumer (deferred); the orchestrator now emits a one-time run-start warning
    that split-aware overfit protection is disabled. Non-proxy adapters
    (CO-Bench ``dev_equals_test``, classification) stay silent → default path
    byte-identical.
  * TC-1 (text_classification) — the dead ``needs_re_solve`` /
    ``adapter.get_test_task`` branches in ``_run_test_evaluation`` are removed;
    the single correct ``evaluate_test(task, trace.script)`` path remains, and
    is NOT bypassed even if an adapter happens to define ``get_test_task``.
  * TC-2 (text_classification) — ``_score_accuracy.raw_score`` is now the
    fraction (0..1), matching ``_score_f1`` (previously a 0..N count).

LLM-free / Docker-free; default Omega/CO-Bench/[0,1] path untouched.
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from meta_n.core.evolutionary_orchestrator import (
    EvolutionaryConfig,
    EvolutionaryOrchestrator,
)
from meta_n.core.meta_layer import Trace
from meta_n.integrations.arc_agi import ARCAGI2Adapter
from meta_n.integrations.benchmark import EvalResult
from meta_n.integrations.text_classification import TextClassificationAdapter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_orchestrator():
    return EvolutionaryOrchestrator(
        llm_client=MagicMock(),
        executor=MagicMock(),
        omega=MagicMock(),
        config=EvolutionaryConfig(max_depth=1, parallel=1, patience=1),
        solver_language="bash",
    )


class _Task:
    def __init__(self, task_id: str) -> None:
        self.task_id = task_id


# ===========================================================================
# R2-ARC-1 — proxy-split warning at run start (deferred selection consumer)
# ===========================================================================

def test_arc_split_type_is_proxy():
    # The leaky-proxy advertisement is intact (unchanged behavior).
    assert ARCAGI2Adapter().split_type() == "proxy"


def test_warn_fires_for_proxy_adapter(caplog):
    orch = _make_orchestrator()
    adapter = MagicMock()
    adapter.split_type.return_value = "proxy"
    adapter.name = "arc_agi_2"
    with caplog.at_level(logging.WARNING):
        fired = orch._warn_if_proxy_split_unconsumed(adapter)
    assert fired is True
    assert any("proxy" in r.message for r in caplog.records)


def test_warn_silent_for_dev_equals_test_adapter(caplog):
    """CO-Bench-style adapter (default [0,1] path) must stay silent."""
    orch = _make_orchestrator()
    adapter = MagicMock()
    adapter.split_type.return_value = "dev_equals_test"
    with caplog.at_level(logging.WARNING):
        fired = orch._warn_if_proxy_split_unconsumed(adapter)
    assert fired is False
    assert caplog.records == []


def test_warn_silent_for_none_or_no_split_type(caplog):
    orch = _make_orchestrator()
    with caplog.at_level(logging.WARNING):
        assert orch._warn_if_proxy_split_unconsumed(None) is False
        assert orch._warn_if_proxy_split_unconsumed(object()) is False
    assert caplog.records == []


# ===========================================================================
# TC-1 — dead needs_re_solve / get_test_task branches removed
# ===========================================================================

@pytest.mark.asyncio
async def test_run_test_evaluation_uses_evaluate_test_with_trace_script():
    orch = _make_orchestrator()
    task = _Task("t1")
    trace = Trace(task_id="t1", script="SOLUTION_CODE", success=True, score=1.0)

    adapter = MagicMock()
    # No get_test_task attribute → the old gate would have been False anyway,
    # but we assert the correct call regardless of its presence below.
    del adapter.get_test_task
    adapter.evaluate_test = AsyncMock(
        return_value=EvalResult(success=True, score=0.7, feedback="")
    )
    orch.executor = MagicMock()
    orch.executor.adapter = adapter
    orch.archive = MagicMock()
    orch.archive.per_task_best_traces.return_value = {"t1": trace}
    orch.archive._best_per_task = {"t1": (1.0, "cand-1")}

    scores = await orch._run_test_evaluation([task])

    assert scores == {"t1": 0.7}
    adapter.evaluate_test.assert_awaited_once_with(task, "SOLUTION_CODE")


@pytest.mark.asyncio
async def test_run_test_evaluation_ignores_get_test_task_when_present():
    """Even if an adapter defines get_test_task, the removed dead branch must
    NOT fire — evaluate_test(task, trace.script) is the only path."""
    orch = _make_orchestrator()
    task = _Task("t1")
    trace = Trace(task_id="t1", script="SOLUTION_CODE", success=True, score=1.0)

    adapter = MagicMock()
    adapter.get_test_task = MagicMock()  # would have triggered the dead branch
    adapter.evaluate_test = AsyncMock(
        return_value=EvalResult(success=True, score=0.5, feedback="")
    )
    orch.executor = MagicMock()
    orch.executor.adapter = adapter
    orch.archive = MagicMock()
    orch.archive.per_task_best_traces.return_value = {"t1": trace}
    orch.archive._best_per_task = {"t1": (1.0, "cand-1")}

    scores = await orch._run_test_evaluation([task])

    assert scores == {"t1": 0.5}
    adapter.evaluate_test.assert_awaited_once_with(task, "SOLUTION_CODE")
    adapter.get_test_task.assert_not_called()


# ===========================================================================
# TC-2 — accuracy raw_score is the fraction (0..1), matching f1
# ===========================================================================

def test_score_accuracy_raw_score_is_fraction():
    a = TextClassificationAdapter(dataset_name="symptom2disease")
    preds = {"case_0": "flu", "case_1": "cold", "case_2": "flu"}
    examples = [
        {"label": "flu"},   # correct
        {"label": "flu"},   # wrong
        {"label": "flu"},   # correct
    ]
    res = a._score_accuracy(preds, examples)
    # 2/3 correct → score and raw_score both the fraction (not the count 2.0).
    assert res.score == pytest.approx(2 / 3)
    assert res.raw_score == pytest.approx(2 / 3)
    assert res.raw_score == res.score


def test_score_accuracy_raw_score_unit_matches_f1():
    """raw_score units now agree across both scorers (both 0..1 fractions)."""
    a = TextClassificationAdapter(dataset_name="lawbench_charge")
    preds = {"case_0": "x", "case_1": "y"}
    acc_examples = [{"label": "x"}, {"label": "x"}]      # accuracy: 1/2
    f1_examples = [{"label": "x"}, {"label": "y"}]       # f1: 1.0 avg
    acc = a._score_accuracy(preds, acc_examples)
    f1 = a._score_f1(preds, f1_examples)
    assert 0.0 <= acc.raw_score <= 1.0
    assert 0.0 <= f1.raw_score <= 1.0
