"""Refinement regression tests for meta_n/utils/context_manager.py (audit cluster C2).

Covers:
- F166: truncate_context_stack keeps the token total incrementally — same
  layers dropped as the naive re-summing loop, with O(n) estimator calls.
- F140: the rng constructor kwarg is a real injection seam; default falls
  back to the module-level random.
"""

import random

from meta_n.core.meta_layer import InjectedCode
from meta_n.utils.context_manager import ContextBudget, ContextManager


def _stack() -> list[InjectedCode]:
    """6 layers of varied sizes; per-layer estimates 2080/1580/1080/830/580/330
    tokens (total 6480)."""
    sizes = [8000, 6000, 4000, 3000, 2000, 1000]
    return [
        InjectedCode(pre_process="x" * size, source_depth=depth)
        for depth, size in enumerate(sizes, start=2)
    ]


def _budget_2000() -> ContextBudget:
    # context_stack_budget = int((7715 - 2000) * 0.35) = 2000 → forces 3 drops.
    return ContextBudget(max_tokens=7715, prompt_overhead=2000)


def _naive_truncate(cm: ContextManager, stack: list[InjectedCode]) -> list[InjectedCode]:
    """Reference implementation: re-sums the tail every iteration (the
    pre-F166 loop). The optimized loop must match it exactly."""
    budget = cm.budget.context_stack_budget
    if not stack:
        return []
    if sum(cm._estimate_injected_code_tokens(c) for c in stack) <= budget:
        return list(stack)
    result = list(stack)
    while (
        len(result) > 1
        and sum(cm._estimate_injected_code_tokens(c) for c in result) > budget
    ):
        result.pop(0)
    return result


def test_truncation_result_matches_naive_recompute():
    cm = ContextManager(_budget_2000())
    stack = _stack()
    got = cm.truncate_context_stack(stack)
    want = _naive_truncate(cm, stack)
    assert [c.source_depth for c in got] == [c.source_depth for c in want]
    # Sanity: the budget really forces 3 drops and keeps the most recent 3.
    assert [c.source_depth for c in got] == [5, 6, 7]


def test_truncation_keep_at_least_one_matches_naive():
    """Budget smaller than any single layer: both impls keep exactly the most
    recent layer."""
    # context_stack_budget = int((2300 - 2000) * 0.35) = 105 tokens.
    cm = ContextManager(ContextBudget(max_tokens=2300, prompt_overhead=2000))
    stack = _stack()
    got = cm.truncate_context_stack(stack)
    want = _naive_truncate(cm, stack)
    assert [c.source_depth for c in got] == [c.source_depth for c in want] == [7]


def test_truncation_estimator_call_count_is_linear(monkeypatch):
    calls = {"n": 0}
    orig = ContextManager._estimate_injected_code_tokens

    def counting(self, code):
        calls["n"] += 1
        return orig(self, code)

    monkeypatch.setattr(ContextManager, "_estimate_injected_code_tokens", counting)
    cm = ContextManager(_budget_2000())
    stack = _stack()
    cm.truncate_context_stack(stack)
    # One initial full sum + one re-estimate per dropped victim: <= 2n.
    assert calls["n"] <= 2 * len(stack)


def test_truncation_warns_on_drop(caplog):
    """The drop warning still fires (dropped list still built) after F166."""
    import logging

    cm = ContextManager(_budget_2000())
    with caplog.at_level(logging.WARNING, logger="meta_n.utils.context_manager"):
        cm.truncate_context_stack(_stack())
    assert any("truncation dropped" in r.message for r in caplog.records)


def test_rng_kwarg_is_injection_seam():
    """F140: default rng is the module-level random; the kwarg overrides it."""
    assert ContextManager().rng is random
    seeded = random.Random(42)
    assert ContextManager(rng=seeded).rng is seeded
