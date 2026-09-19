"""§6b regression tests for meta_n/utils/context_manager.py (F156).

F156 — Ω trace-sampling asymmetries:
- OFF (default, ``symmetric_sampling=False``): characterization pins of the
  historical behavior — failure slots are hard-capped with NO backfill
  (successes DO backfill), and budget eviction pops successes first (a tight
  budget can yield a 100%-failure sample). These pins close the
  "neither documented nor test-pinned" half of the finding.
- ON (``symmetric_sampling=True``): failures backfill leftover slots and
  budget eviction preserves ``failure_ratio`` across classes.
- Plumbing: ``OmegaEngine(symmetric_trace_sampling=...)`` reaches the
  ContextManager; default is OFF.
"""

import random

from meta_n.core.llm_client import LLMClient, LLMConfig
from meta_n.core.meta_layer import Trace
from meta_n.core.omega import OmegaEngine
from meta_n.utils.context_manager import ContextBudget, ContextManager


def make_trace(task_id: str, success: bool, script_len: int = 50) -> Trace:
    return Trace(
        task_id=task_id,
        success=success,
        script="x" * script_len,
        stdout="out" if success else "",
        stderr="" if success else "error " * 10,
        error_summary="" if success else "failed",
    )


def _skewed_failures() -> list[Trace]:
    """100 failures + 2 successes (successes scarce)."""
    return [make_trace(f"f{i}", False) for i in range(100)] + [
        make_trace(f"s{i}", True) for i in range(2)
    ]


def _skewed_successes() -> list[Trace]:
    """2 failures + 100 successes (failures scarce)."""
    return [make_trace(f"f{i}", False) for i in range(2)] + [
        make_trace(f"s{i}", True) for i in range(100)
    ]


def _mixed() -> list[Trace]:
    """30 failures + 10 successes."""
    return [make_trace(f"f{i}", False) for i in range(30)] + [
        make_trace(f"s{i}", True) for i in range(10)
    ]


def _tight_budget() -> ContextBudget:
    """traces_budget = int(1100 * 0.65) = 715 tokens — forces eviction of a
    15F+5S sample (~79 tokens per failure, ~63 per success)."""
    return ContextBudget(max_tokens=1200, prompt_overhead=100)


def _counts(sample: list[Trace]) -> tuple[int, int]:
    return (
        sum(1 for t in sample if not t.success),
        sum(1 for t in sample if t.success),
    )


# ---------------------------------------------------------------------------
# OFF path — characterization pins of the historical default
# ---------------------------------------------------------------------------


class TestOffPathCharacterization:
    def test_off_no_backfill_when_successes_scarce(self):
        """Failure slots are hard-capped at int(20*0.75)=15 with NO backfill:
        100F+2S yields only 17 traces — 3 slots silently wasted."""
        cm = ContextManager(rng=random.Random(0))
        sampled = cm.sample_traces(_skewed_failures(), max_total=20)
        assert len(sampled) == 17
        assert _counts(sampled) == (15, 2)

    def test_off_successes_do_backfill_when_failures_scarce(self):
        """The asymmetry: successes DO backfill (max_total - n_failures grows
        when failures are scarce) — 2F+100S fills all 20 slots."""
        cm = ContextManager(rng=random.Random(0))
        sampled = cm.sample_traces(_skewed_successes(), max_total=20)
        assert len(sampled) == 20
        assert _counts(sampled) == (2, 18)

    def test_off_budget_eviction_drops_successes_first(self):
        """Budget eviction pops from the back of failures-then-successes, so
        the 15F+5S sample degrades to 100% failures under a tight budget."""
        cm = ContextManager(_tight_budget(), rng=random.Random(0))
        sampled = cm.sample_traces(_mixed(), max_total=20)
        n_f, n_s = _counts(sampled)
        assert n_s == 0
        assert n_f == 9  # all survivors are failures — far past the 3:1 bias

    def test_off_sampled_ids_byte_identical(self):
        """Seeded rng, flag OFF: the sampled task_id sequence equals the
        literal captured at implementation time — guards the rng-consumption
        identity of the OFF path (the F156 guards must not add rng calls)."""
        cm = ContextManager(rng=random.Random(0))
        sampled = cm.sample_traces(_skewed_failures(), max_total=20)
        assert [t.task_id for t in sampled] == [
            "f49", "f97", "f53", "f5", "f33", "f65", "f62", "f51", "f38",
            "f61", "f45", "f74", "f27", "f64", "f17", "s0", "s1",
        ]


# ---------------------------------------------------------------------------
# ON path — symmetric_sampling=True
# ---------------------------------------------------------------------------


class TestSymmetricSampling:
    def test_on_backfills_failures(self):
        """Leftover slots (successes scarce) are backfilled with failures:
        100F+2S fills all 20 slots (18F, 2S) and never exceeds max_total."""
        cm = ContextManager(rng=random.Random(0), symmetric_sampling=True)
        sampled = cm.sample_traces(_skewed_failures(), max_total=20)
        assert len(sampled) == 20
        assert _counts(sampled) == (18, 2)

    def test_on_noop_when_no_leftover(self):
        """No leftover and no eviction ⇒ ON output equals OFF output
        element-for-element (same rng consumption, same truncation no-op)."""
        off = ContextManager(rng=random.Random(7)).sample_traces(
            _mixed(), max_total=20
        )
        on = ContextManager(
            rng=random.Random(7), symmetric_sampling=True
        ).sample_traces(_mixed(), max_total=20)
        assert [t.task_id for t in on] == [t.task_id for t in off]

    def test_on_proportional_eviction_keeps_both_classes(self):
        """Ratio-preserving eviction: the tight-budget repro keeps BOTH
        classes and the failure share stays within one trace of the ratio."""
        cm = ContextManager(
            _tight_budget(), rng=random.Random(0), symmetric_sampling=True
        )
        sampled = cm.sample_traces(_mixed(), max_total=20, failure_ratio=0.75)
        n_f, n_s = _counts(sampled)
        assert n_f > 0 and n_s > 0
        assert abs(n_f / len(sampled) - 0.75) <= 1 / len(sampled)

    def test_on_eviction_deterministic(self):
        """Same inputs, same seed → identical output (eviction uses no rng)."""
        runs = []
        for _ in range(2):
            cm = ContextManager(
                _tight_budget(), rng=random.Random(3), symmetric_sampling=True
            )
            runs.append(
                [t.task_id for t in cm.sample_traces(_mixed(), max_total=20)]
            )
        assert runs[0] == runs[1]

    def test_on_all_failures_matches_off(self):
        """Single-class input: eviction falls through to the only non-empty
        class, so ON behaves exactly like today."""
        traces = [make_trace(f"f{i}", False) for i in range(30)]
        off = ContextManager(_tight_budget(), rng=random.Random(1)).sample_traces(
            traces, max_total=20
        )
        on = ContextManager(
            _tight_budget(), rng=random.Random(1), symmetric_sampling=True
        ).sample_traces(traces, max_total=20)
        assert [t.task_id for t in on] == [t.task_id for t in off]
        assert all(not t.success for t in on)

    def test_on_all_successes_matches_off(self):
        traces = [make_trace(f"s{i}", True) for i in range(30)]
        off = ContextManager(_tight_budget(), rng=random.Random(1)).sample_traces(
            traces, max_total=20
        )
        on = ContextManager(
            _tight_budget(), rng=random.Random(1), symmetric_sampling=True
        ).sample_traces(traces, max_total=20)
        assert [t.task_id for t in on] == [t.task_id for t in off]
        assert all(t.success for t in on)


# ---------------------------------------------------------------------------
# Plumbing — OmegaEngine kwarg reaches the ContextManager
# ---------------------------------------------------------------------------


def test_omega_engine_plumbs_symmetric_sampling():
    client = LLMClient(LLMConfig(api_key="test"))
    on = OmegaEngine(client, symmetric_trace_sampling=True)
    assert on.context_manager.symmetric_sampling is True
    default = OmegaEngine(client)
    assert default.context_manager.symmetric_sampling is False
