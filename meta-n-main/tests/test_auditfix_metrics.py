"""Regression tests for audit fixes in meta_n/analysis/metrics.py.

Findings:
  * 12 — _classify_trace_errors must preserve the structured ``terminated_by``
    signal when rebuilding a Trace, so classify_error's structured-precedence
    branches (e.g. the no-text-fallback ``env_error`` -> "Environment fault")
    remain reachable.
  * 43 — compute_convergence tpi loop's statically-unreachable
    ``if i > 0 else cumulative[i]`` dead guard is removed.

All tests are offline / LLM-free.

NOTE: this worktree's HEAD predates the ``Trace.terminated_by`` field and the
structured-precedence logic in ``_classify_error`` (both are pure-text here).
The finding-12 passthrough fix is therefore forward-compatible: it matches the
audit's prescribed correction and becomes load-bearing once ``Trace`` gains the
field. The finding-12 behavioural assertion is guarded to skip on this older
code version (it cannot fail-then-pass where the field does not exist).
"""

import inspect

import pytest

from meta_n.analysis.metrics import ExperimentAnalyzer
from meta_n.core.meta_layer import Trace


# ---------------------------------------------------------------------------
# Finding 12
# ---------------------------------------------------------------------------

def test_classify_trace_errors_passes_terminated_by_through():
    """The Trace rebuild in _classify_trace_errors must forward terminated_by
    so classify_error's structured precedence stays reachable. On this older
    worktree HEAD the field does not yet exist, so we skip; on any version that
    defines it, a max_turns stop whose text says 'timeout' must classify as
    'Turn starvation' (structured signal wins over keyword sniffing)."""
    if "terminated_by" not in Trace.model_fields:
        pytest.skip("worktree HEAD predates Trace.terminated_by (pure-text classify)")
    traces = [
        {
            "task_id": "t1",
            "success": False,
            "terminated_by": "max_turns",
            "error_summary": "agent stopped",
            "stderr": "operation timeout reached",
        }
    ]
    result = ExperimentAnalyzer._classify_trace_errors(traces)
    assert result["distribution"].get("Turn starvation") == 1, result
    assert "Timeout" not in result["distribution"], result


def test_classify_trace_errors_forwards_terminated_by_in_source():
    """Static guard that survives across code versions: the Trace rebuild must
    name terminated_by among the forwarded fields. This fails on the original
    (which only forwarded task_id/error_summary/stderr) and passes after the
    audit fix, independent of whether the field is wired up yet."""
    src = inspect.getsource(ExperimentAnalyzer._classify_trace_errors)
    assert "terminated_by" in src, (
        "Trace rebuild must forward terminated_by so structured precedence "
        "is not silently dropped"
    )


# ---------------------------------------------------------------------------
# Finding 43
# ---------------------------------------------------------------------------

def test_compute_convergence_has_no_dead_else_guard():
    """The tpi loop runs over range(1, len(history)) so i is always >= 1; the
    ``if i > 0 else cumulative[i]`` guard is statically unreachable dead code
    and must be removed."""
    src = inspect.getsource(ExperimentAnalyzer.compute_convergence)
    assert "if i > 0 else cumulative[i]" not in src, (
        "dead always-true guard should be removed from step_tokens"
    )
