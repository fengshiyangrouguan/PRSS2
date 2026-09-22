"""Context window management for Omega prompts.

Handles:
- Failure-biased trace sampling (3:1 default)
- Bottom-up context truncation (oldest layers dropped first)
- Token budget enforcement
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass

from meta_n.core.meta_layer import InjectedCode, Trace

logger = logging.getLogger(__name__)


@dataclass
class ContextBudget:
    """Token budget allocation for Omega prompt sections."""

    max_tokens: int = 100_000
    # Reserve space for the prompt template and LLM response
    prompt_overhead: int = 2_000
    # Allocation ratios for remaining budget
    traces_ratio: float = 0.65
    context_stack_ratio: float = 0.35

    def __post_init__(self):
        if self.prompt_overhead >= self.max_tokens:
            raise ValueError(
                f"prompt_overhead ({self.prompt_overhead}) must be less than "
                f"max_tokens ({self.max_tokens})"
            )

    @property
    def available_tokens(self) -> int:
        return self.max_tokens - self.prompt_overhead

    @property
    def traces_budget(self) -> int:
        return int(self.available_tokens * self.traces_ratio)

    @property
    def context_stack_budget(self) -> int:
        return int(self.available_tokens * self.context_stack_ratio)


class ContextManager:
    """Manages context window for Omega prompts."""

    def __init__(
        self,
        budget: ContextBudget | None = None,
        rng: random.Random | None = None,
        symmetric_sampling: bool = False,
    ):
        self.budget = budget or ContextBudget()
        # F156: when True, sample_traces backfills leftover slots with extra
        # FAILURES (successes already backfill implicitly) and budget eviction
        # preserves the failure_ratio across classes instead of popping
        # successes first. Default False pins the historical behavior — every
        # existing construction site is unchanged and byte-identical.
        self.symmetric_sampling = symmetric_sampling
        # Sampling RNG. Callers construct ContextManager without rng and the
        # EvolutionaryOrchestrator threads its seeded `random.Random` in
        # AFTERWARD by assigning `.rng` (orchestrator __init__ plus the two
        # checkpoint-fallback paths) — `Random.setstate` mutates that same
        # object, so the shared reference stays valid across --resume. The
        # `rng` kwarg is an injection seam for tests/direct callers. Falling
        # back to the module-level `random` keeps callers that don't care
        # about determinism working unchanged.
        self.rng = rng if rng is not None else random

    def sample_traces(
        self,
        traces: list[Trace],
        max_total: int = 4,
        failure_ratio: float = 0.75,
    ) -> list[Trace]:
        """
        Sample traces with failure-biased ratio.

        Args:
            traces: All available traces
            max_total: Maximum number of traces to return. **4, not upstream's
                20** -- this is the MATCHED-CONTEXT-BUDGET rule (frozen
                2026-09-21): the predictive arm is bounded by Gamma's
                ``n_slots=4`` attention slots, so the official arm is capped to
                the same k. Without it the arms run under different resource
                ceilings and a difference cannot be attributed to the selection
                METHOD. The baseline is reported as ``Meta^n (matched context
                budget, k=4)``; upstream's 20 is still reachable by passing it
                explicitly and belongs in the appendix.
            failure_ratio: Target ratio of failures in sample (default 0.75 = 3:1)

        Returns:
            Sampled list of traces fitting within token budget

        Default (``symmetric_sampling=False``) contract — two documented
        asymmetries, pinned by tests/test_r6b_context_manager.py:

        - Failure slots are HARD-CAPPED at ``int(max_total * failure_ratio)``
          with NO backfill: when successes are scarce the sample may be
          SMALLER than ``max_total`` (leftover slots go unused). Successes DO
          backfill implicitly (``max_total - n_failures`` grows when failures
          are scarce).
        - Budget eviction pops from the back of failures-then-successes, i.e.
          successes are evicted FIRST — a tight budget can yield a
          100%-failure sample, well past the 3:1 target ratio.

        ``symmetric_sampling=True`` (F156) opts into failure backfill of
        leftover slots plus ratio-preserving budget eviction.
        """
        failures = [t for t in traces if not t.success]
        successes = [t for t in traces if t.success]

        n_failures = min(len(failures), int(max_total * failure_ratio))
        n_successes = min(len(successes), max_total - n_failures)

        if self.symmetric_sampling:
            leftover = max_total - n_failures - n_successes
            if leftover > 0:
                n_failures = min(len(failures), n_failures + leftover)

        sampled_failures = (
            self.rng.sample(failures, n_failures) if len(failures) > n_failures else failures
        )
        sampled_successes = (
            self.rng.sample(successes, n_successes)
            if len(successes) > n_successes
            else successes
        )

        if self.symmetric_sampling:
            return self._truncate_traces_to_budget_proportional(
                sampled_failures, sampled_successes, failure_ratio
            )

        sampled = sampled_failures + sampled_successes

        # Truncate to fit token budget
        return self._truncate_traces_to_budget(sampled)

    def truncate_context_stack(
        self, context_stack: list[InjectedCode]
    ) -> list[InjectedCode]:
        """
        Truncate context stack bottom-up (oldest/lowest layers first)
        to fit within the context stack token budget.

        The most recent layers (highest depth) are kept since they contain
        the latest improvements. Older layers are dropped first.
        """
        if not context_stack:
            return []

        budget = self.budget.context_stack_budget
        total = sum(self._estimate_injected_code_tokens(c) for c in context_stack)

        if total <= budget:
            return list(context_stack)

        # Drop from the front (oldest/lowest depth) until we fit
        # Keep at least the most recent layer
        result = list(context_stack)
        dropped: list[InjectedCode] = []
        # `total` above already sums the full stack; keep it current
        # incrementally (each estimate is a pure function of the code)
        # instead of re-summing the tail every iteration — same layers
        # dropped, O(n) estimator calls instead of O(n^2).
        while len(result) > 1 and total > budget:
            victim = result.pop(0)
            dropped.append(victim)
            total -= self._estimate_injected_code_tokens(victim)

        # C2.1: dropped layers stay LIVE in the solver (MetaLayer merges the full
        # stack) but vanish from the Omega prompt view -> re-synthesis /
        # name-collision risk with no operator-visible signal. Emit ONE WARNING
        # naming the dropped layers' source_depths. Only fires when truncation
        # actually drops a layer, so the byte-identical (no-truncation) path is
        # untouched.
        if dropped:
            dropped_depths = [
                getattr(c, "source_depth", None) for c in dropped
            ]
            logger.warning(
                "Omega context truncation dropped %d oldest injected-code "
                "layer(s) (source_depths=%s) to fit the context-stack budget "
                "(~%d tokens); these layers remain LIVE in the solver but are "
                "INVISIBLE to Omega -> possible re-synthesis / helper name "
                "collisions.",
                len(dropped), dropped_depths, budget,
            )

        return result

    def _truncate_traces_to_budget(self, traces: list[Trace]) -> list[Trace]:
        """Remove traces from back until they fit the traces token budget."""
        budget = self.budget.traces_budget
        result = list(traces)

        while result and sum(self._estimate_trace_tokens(t) for t in result) > budget:
            result.pop()

        return result

    def _truncate_traces_to_budget_proportional(
        self,
        failures: list[Trace],
        successes: list[Trace],
        failure_ratio: float,
    ) -> list[Trace]:
        """Ratio-preserving budget eviction (F156, ``symmetric_sampling=True``).

        Evicts from the back of whichever class is OVER its target share
        until the summed estimates fit ``self.budget.traces_budget``, so a
        tight budget degrades BOTH classes toward ``failure_ratio`` instead
        of silently evicting all successes first. Presentation order stays
        failures-then-successes. Deterministic (no rng); incremental
        running total keeps estimator calls O(n).
        """
        f, s = list(failures), list(successes)
        total = sum(self._estimate_trace_tokens(t) for t in f + s)
        budget = self.budget.traces_budget
        while (f or s) and total > budget:
            n = len(f) + len(s)
            if s and (not f or len(f) / n <= failure_ratio):
                total -= self._estimate_trace_tokens(s.pop())
            else:
                total -= self._estimate_trace_tokens(f.pop())
        return f + s

    def _estimate_trace_tokens(self, trace: Trace) -> int:
        """Estimate token count for a single trace when formatted.

        Limits must stay consistent with omega._format_raw_traces() truncation.
        """
        stdout_chars = min(len(trace.stdout), 500)    # 200 head + 300 tail
        stderr_chars = min(len(trace.stderr), 800)    # 800 tail
        eval_chars = min(len(trace.eval_feedback), 800)  # 800 tail
        text = (
            trace.script
            + trace.stdout[:stdout_chars]
            + trace.stderr[-stderr_chars:]
            + trace.error_summary
            + trace.eval_feedback[-eval_chars:]
        )
        return self.estimate_tokens(text) + 50  # overhead for formatting

    def _estimate_injected_code_tokens(self, code: InjectedCode) -> int:
        """Estimate token count for a single InjectedCode when formatted."""
        parts = [
            code.pre_process or "",
            code.rationale or "",
        ]
        parts.extend(code.code_library.values())
        parts.extend(code.code_library_bash.values())
        text = "\n".join(parts)
        return self.estimate_tokens(text) + 80  # overhead for formatting

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Approximate token count. Chars/4 is a reasonable heuristic."""
        return len(text) // 4
