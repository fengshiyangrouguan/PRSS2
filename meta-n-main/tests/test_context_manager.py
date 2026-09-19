"""Tests for context window management."""

import pytest

from meta_n.core.meta_layer import InjectedCode, Trace
from meta_n.utils.context_manager import ContextBudget, ContextManager


# --- Helpers ---

def make_trace(task_id: str, success: bool, script_len: int = 50) -> Trace:
    """Create a trace with a script of given length."""
    return Trace(
        task_id=task_id,
        success=success,
        script="x" * script_len,
        stdout="out" if success else "",
        stderr="" if success else "error " * 10,
        error_summary="" if success else "failed",
    )


def make_injected_code(depth: int, code_len: int = 100) -> InjectedCode:
    """Create an InjectedCode with content of given length."""
    return InjectedCode(
        pre_process="a" * code_len,
        rationale="reason " * 5,
        source_depth=depth,
    )


# --- ContextBudget tests ---

class TestContextBudget:
    def test_defaults(self):
        budget = ContextBudget()
        assert budget.max_tokens == 100_000
        assert budget.available_tokens == 98_000
        assert budget.traces_budget == int(98_000 * 0.65)
        assert budget.context_stack_budget == int(98_000 * 0.35)

    def test_custom_budget(self):
        budget = ContextBudget(max_tokens=10_000, prompt_overhead=500)
        assert budget.available_tokens == 9_500
        assert budget.traces_budget == int(9_500 * 0.65)

    def test_ratios_sum(self):
        budget = ContextBudget()
        # Traces + context stack should use all available tokens
        assert budget.traces_ratio + budget.context_stack_ratio == 1.0


# --- Trace sampling tests ---

class TestTraceSampling:
    def test_basic_ratio(self):
        cm = ContextManager()
        failures = [make_trace(f"f{i}", False) for i in range(30)]
        successes = [make_trace(f"s{i}", True) for i in range(10)]
        traces = failures + successes

        sampled = cm.sample_traces(traces, max_total=20, failure_ratio=0.75)

        sampled_f = [t for t in sampled if not t.success]
        sampled_s = [t for t in sampled if t.success]
        assert len(sampled_f) == 15  # 75% of 20
        assert len(sampled_s) == 5   # 25% of 20

    def test_fewer_traces_than_max(self):
        cm = ContextManager()
        traces = [make_trace("f1", False), make_trace("s1", True)]
        sampled = cm.sample_traces(traces, max_total=20)
        assert len(sampled) == 2

    def test_all_failures(self):
        cm = ContextManager()
        traces = [make_trace(f"f{i}", False) for i in range(5)]
        sampled = cm.sample_traces(traces, max_total=20)
        assert len(sampled) == 5
        assert all(not t.success for t in sampled)

    def test_all_successes(self):
        cm = ContextManager()
        traces = [make_trace(f"s{i}", True) for i in range(5)]
        sampled = cm.sample_traces(traces, max_total=20)
        assert len(sampled) == 5
        assert all(t.success for t in sampled)

    def test_empty_traces(self):
        cm = ContextManager()
        sampled = cm.sample_traces([], max_total=20)
        assert sampled == []

    def test_token_budget_truncation(self):
        """Large traces get truncated to fit budget."""
        # Tiny budget: only room for a few traces
        budget = ContextBudget(max_tokens=500, prompt_overhead=100)
        cm = ContextManager(budget)

        # Each trace ~50 chars script → ~12 tokens + 50 overhead ≈ 62 tokens
        # Budget: 400 * 0.65 = 260 tokens → room for ~4 traces
        traces = [make_trace(f"f{i}", False, script_len=50) for i in range(20)]
        sampled = cm.sample_traces(traces, max_total=20)
        assert len(sampled) < 20
        assert len(sampled) > 0


# --- Context stack truncation tests ---

class TestContextStackTruncation:
    def test_fits_within_budget(self):
        cm = ContextManager()
        stack = [make_injected_code(d, code_len=50) for d in range(1, 4)]
        truncated = cm.truncate_context_stack(stack)
        assert len(truncated) == 3  # all fit

    def test_bottom_up_truncation(self):
        """Oldest layers (lowest depth) are removed first."""
        budget = ContextBudget(max_tokens=1_000, prompt_overhead=100)
        cm = ContextManager(budget)

        # Each code: ~300 chars each part * 3 + overhead = ~300+ tokens
        # Budget: 900 * 0.35 = 315 tokens → room for ~1 code
        stack = [make_injected_code(d, code_len=300) for d in range(1, 5)]
        truncated = cm.truncate_context_stack(stack)

        # Should keep the most recent (highest depth)
        assert len(truncated) >= 1
        assert truncated[-1].source_depth == 4  # most recent kept
        if len(truncated) > 1:
            # Remaining should be the most recent ones
            depths = [c.source_depth for c in truncated]
            assert depths == sorted(depths)  # ascending order preserved

    def test_always_keeps_at_least_one(self):
        """Even with tiny budget, keep the most recent layer."""
        budget = ContextBudget(max_tokens=100, prompt_overhead=50)
        cm = ContextManager(budget)

        stack = [make_injected_code(d, code_len=500) for d in range(1, 4)]
        truncated = cm.truncate_context_stack(stack)

        assert len(truncated) == 1
        assert truncated[0].source_depth == 3  # most recent

    def test_empty_stack(self):
        cm = ContextManager()
        truncated = cm.truncate_context_stack([])
        assert truncated == []

    def test_preserves_order(self):
        cm = ContextManager()
        stack = [make_injected_code(d, code_len=20) for d in [2, 3, 5]]
        truncated = cm.truncate_context_stack(stack)
        depths = [c.source_depth for c in truncated]
        assert depths == [2, 3, 5]


# --- Token estimation tests ---

class TestTokenEstimation:
    def test_estimate_tokens(self):
        assert ContextManager.estimate_tokens("") == 0
        assert ContextManager.estimate_tokens("abcd") == 1
        assert ContextManager.estimate_tokens("a" * 400) == 100

    def test_trace_token_estimate(self):
        cm = ContextManager()
        trace = make_trace("t1", False, script_len=200)
        tokens = cm._estimate_trace_tokens(trace)
        assert tokens > 0
        # Should be roughly: (200 + stderr + overhead) / 4 + 50
        assert tokens > 50

    def test_injected_code_token_estimate(self):
        cm = ContextManager()
        code = make_injected_code(1, code_len=200)
        tokens = cm._estimate_injected_code_tokens(code)
        assert tokens > 0
        # 3 parts * 200 chars + rationale + overhead
        assert tokens > 100


# --- C2.1: warn-on-drop during context-stack truncation ---

class TestTruncationDropWarning:
    """C2.1: dropped oldest layers stay LIVE in the solver but vanish from the
    Omega prompt; truncate_context_stack must emit ONE warning naming their
    source_depths. When no truncation fires the path stays silent (byte-identical).
    """

    def test_no_truncation_is_silent(self, caplog):
        # Small stack well within budget -> no drop, no warning.
        cm = ContextManager()
        stack = [make_injected_code(d, code_len=20) for d in [1, 2, 3]]
        with caplog.at_level("WARNING"):
            out = cm.truncate_context_stack(stack)
        assert [c.source_depth for c in out] == [1, 2, 3]
        assert not [r for r in caplog.records if r.levelname == "WARNING"]

    def test_drop_emits_single_warning_with_depths(self, caplog):
        # Force truncation with a tiny budget and large layers.
        cm = ContextManager(budget=ContextBudget(max_tokens=10_000))
        stack = [make_injected_code(d, code_len=4_000) for d in [1, 2, 3, 4]]
        with caplog.at_level("WARNING"):
            out = cm.truncate_context_stack(stack)
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        # Exactly ONE warning, and it names the dropped (oldest) source_depths.
        assert len(warnings) == 1
        kept_depths = {c.source_depth for c in out}
        dropped_depths = {1, 2, 3, 4} - kept_depths
        assert dropped_depths  # something was actually dropped
        msg = warnings[0].getMessage()
        for d in dropped_depths:
            assert str(d) in msg
