"""Omega engine — generates improvement code from execution traces."""

from __future__ import annotations

import ast
import logging
import math
import re
import statistics
from collections import Counter
from typing import Optional

from meta_n.core.llm_client import LLMClient
from meta_n.core.meta_layer import (
    InjectedCode,
    TaskDescription,
    Trace,
    _head_tail,
    _strip_library_prefix_for_scan,
    _tail,
    classify_error,
    detect_script_language,
    scan_helper_calls,
)
from meta_n.core.prompts import OMEGA_PROMPT, OMEGA_PROMPT_DISCOVER, OMEGA_PROMPT_META
from meta_n.utils.context_manager import ContextBudget, ContextManager

logger = logging.getLogger(__name__)


# Task categorization keywords (tuned to CO-Bench task IDs)
TASK_CATEGORIES = {
    "scheduling": ["schedul", "job_shop", "flow_shop", "open_shop", "landing", "crew", "due_date"],
    "packing/cutting": ["bin_pack", "cutting", "guillotine", "container", "loading", "assortment", "unequal_rectangles", "unequal_circles"],
    "assignment/location": ["assign", "warehouse", "p_median", "partition", "covering", "structur"],
    "routing/path": ["rout", "tsp", "travel", "path", "steiner"],
    "knapsack": ["knapsack"],
    "graph": ["graph", "colour", "coloring", "independent"],
    "classification": ["classify", "symptom", "disease", "diagnosis", "lawbench", "charge", "罪名"],
    "math/geometry": ["kissing", "circle_packing", "heilbronn", "hexagon_packing", "packing_rect"],
    "math/optimization": ["matmul", "max_min_dist", "minimizing"],
    "math/analysis": ["autocorr", "erdos", "sums_diffs", "uncertainty"],
    "symbolic_regression": ["sr_", "symbolic_reg", "synth_"],
    "algorithmic_speedup": [
        "affine_transform", "convolve2d", "eigenvector", "fft_",
        "lu_factor", "polynomial_real", "psd_cone",
    ],
}


class OmegaEngine:
    """Calls LLM with the universal Omega prompt and parses InjectedCode."""

    def __init__(
        self,
        llm_client: LLMClient,
        context_budget: ContextBudget | None = None,
        symmetric_trace_sampling: bool = False,
    ):
        self.llm_client = llm_client
        # F156: symmetric_trace_sampling opts the trace sampler into failure
        # backfill + ratio-preserving budget eviction; default False keeps the
        # historical failure-cap / evict-successes-first sampling byte-identical.
        self.context_manager = ContextManager(
            context_budget, symmetric_sampling=symmetric_trace_sampling
        )

    async def generate(
        self,
        traces: list[Trace],
        context_stack: list[InjectedCode],
        tasks: list[TaskDescription],
        depth: int,
        temperature: float | None = None,
        inspiration_traces: list[Trace] | None = None,
        previous_scores: dict[str, float] | None = None,
        archive_best_scores: dict[str, float] | None = None,
        current_scores: dict[str, float] | None = None,
        focus_task: str | None = None,
        solver_language: str = "python",
        no_code_library: bool = False,
        code_library_is_live: bool = True,
        prompt_variant: str | None = None,
        downstream_injections: list[InjectedCode] | None = None,
        within_task_recursion: bool = False,
    ) -> tuple[InjectedCode, int]:
        """
        Generate improvement code via Omega.

        Args:
            traces: Execution traces from the current depth
            context_stack: Previously injected code from lower layers
            tasks: Task descriptions
            depth: Current depth being generated for
            temperature: LLM temperature override (default: 0.7)
            inspiration_traces: Best solutions from other candidates for cross-pollination
            previous_scores: Per-task scores from the previous layer/baseline,
                used for score comparison at all depths
            archive_best_scores: Per-task best scores across the archive,
                shown alongside regressions so Omega knows the target
            solver_language: Language of the solver ("python" or "bash")
            prompt_variant: when 'discover', selects OMEGA_PROMPT_DISCOVER
                (measure-don't-recall mode) regardless of depth; None (default)
                keeps the depth-based OMEGA_PROMPT/OMEGA_PROMPT_META selection
                unchanged.
            downstream_injections: Stage-3 downward re-propagation only — the
                ABOVE-layer injections (depths > the one being regenerated) shown
                as DOWNSTREAM FEEDBACK so Ω error-corrects the intermediate layer
                while staying compatible with the layers built on top of it. The
                full-chain failure ``traces`` are contaminated by these layers, so
                this scopes the call to error-correction, not a fresh redesign.
                ``None`` / empty (every non-re-propagation caller) → the rendered
                prompt is BYTE-IDENTICAL (no new section), preserving the
                omega-prompt golden.

        Returns:
            Tuple of (InjectedCode, tokens_used)
        """
        sampled = self.context_manager.sample_traces(traces)
        truncated_stack = self.context_manager.truncate_context_stack(context_stack)

        logger.debug(
            "Omega generate: depth=%d, traces=%d→%d sampled, stack=%d→%d layers, "
            "inspiration=%d, temp=%.1f",
            depth, len(traces), len(sampled),
            len(context_stack), len(truncated_stack),
            len(inspiration_traces) if inspiration_traces else 0,
            temperature if temperature is not None else 0.7,
        )

        # G4: scan the FULL (pre-sampling) traces for whether the solver
        # actually called each injected helper — a 0-call helper is dead weight
        # Ω should prune rather than re-inject.
        # C2.2: scope the helper-usage scan to the SAME ``truncated_stack`` that
        # ``_build_prompt`` renders. Otherwise, when truncation drops the oldest
        # layers, Ω would be told to "remove" (e.g. "DEAD") helpers it can no
        # longer see in its rendered injected-code block. When no truncation
        # fires ``truncated_stack`` is an element-wise copy of ``context_stack``,
        # so the rendered prompt stays BYTE-IDENTICAL.
        helper_usage = self._helper_usage_section(
            traces, truncated_stack, code_library_is_live=code_library_is_live,
        )

        prompt = self._build_prompt(
            sampled, truncated_stack, tasks, depth, inspiration_traces,
            previous_scores, archive_best_scores, solver_language,
            no_code_library=no_code_library,
            current_scores=current_scores,
            helper_usage=helper_usage,
            focus_task=focus_task,
            prompt_variant=prompt_variant,
            downstream_injections=downstream_injections,
            within_task_recursion=within_task_recursion,
        )

        response, tokens = await self.llm_client.complete(
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature if temperature is not None else 0.7,
        )

        logger.debug("Omega LLM response: tokens=%d, response_len=%d", tokens, len(response))

        injected = self._parse_response(response, depth)
        # Pair the parsed result with the original prompt so callers can
        # persist the full request alongside the response. Without this,
        # ``omega_response_d{N}.txt`` is dangling — you can't reproduce why
        # the model said what it said.
        injected.raw_omega_prompt = prompt

        # Log what was extracted
        extracted = []
        if injected.pre_process:
            extracted.append(f"pre_process({len(injected.pre_process)} chars)")
        if injected.code_library:
            extracted.append(f"code_library({list(injected.code_library.keys())})")
        if injected.code_library_bash:
            extracted.append(f"code_library_bash({list(injected.code_library_bash.keys())})")
        if injected.rationale:
            extracted.append(f"rationale({len(injected.rationale)} chars)")
        if injected.is_empty:
            logger.warning(
                "Omega returned EMPTY injection at depth=%d (tokens=%d). "
                "Response preview: %.200s...",
                depth, tokens, response[:200],
            )
        else:
            logger.debug("Omega extracted: %s", ", ".join(extracted))

        return injected, tokens

    async def refine(
        self,
        prev_injection: InjectedCode,
        child_traces: list[Trace],
        context_stack: list[InjectedCode],
        tasks: list[TaskDescription],
        depth: int,
        temperature: float | None = None,
        previous_scores: dict[str, float] | None = None,
        archive_best_scores: dict[str, float] | None = None,
        current_scores: dict[str, float] | None = None,
        mean_before: float | None = None,
        solver_language: str = "python",
        no_code_library: bool = False,
        code_library_is_live: bool = True,
    ) -> tuple[InjectedCode, int]:
        """Stage 2 WITHIN-LAYER REFINE — one extra Ω call to FIX a buggy injection.

        ``generate`` writes a *new* layer; ``refine`` REPAIRS the layer just
        written. It is the within-layer leg of the error-driven self-repair
        pattern: the same-model generator is knowledge-bound, so this is scoped to
        ERROR-CORRECTION (keep the approach, fix the code), NOT novel generation.

        Reuses :meth:`_build_prompt` byte-identically — ``prev_injection`` is
        appended to ``context_stack`` so it renders as the DEEPEST "Previously
        Injected Code" layer (the thing being fixed), and ``child_traces`` (the
        gate-failure traces, carrying ``eval_feedback`` / ``error_summary``) are
        the trace corpus. The only delta vs ``generate`` is a FIX-THE-BUG
        directive PREPENDED to the rendered prompt — no signature change to
        ``_build_prompt``, so the Stage-3 omega-prompt golden is untouched.

        Args:
            prev_injection: the buggy injection that failed the gate (or
                regressed); shown to Ω as "the layer YOU produced — fix it".
            child_traces: the failing traces motivating the fix.
            context_stack: the layers BELOW ``prev_injection`` (i.e. the parent's
                ``injected_codes``); ``prev_injection`` is appended internally.
            depth: the depth of the layer being repaired (== the buggy child's
                depth); the refined injection is parsed at this depth.
            mean_before: the buggy injection's mean score (for the directive); a
                gate-failure has only a partial signal, so it may be approximate.

        Returns:
            ``(refined_injection, tokens_used)`` — the corrected injection, paired
            with the rendered refine prompt in ``raw_omega_prompt`` so the repair
            transcript persists alongside the response.
        """
        sampled = self.context_manager.sample_traces(child_traces)
        # The buggy injection is the DEEPEST layer in the rendered stack — Ω sees
        # exactly the code it must repair, with the layers below it as context.
        refine_stack = list(context_stack) + [prev_injection]
        truncated_stack = self.context_manager.truncate_context_stack(refine_stack)
        # C2.2: keep helper-usage consistent with the rendered ``truncated_stack``
        # (symmetric with ``generate``). Byte-identical when no truncation fires.
        helper_usage = self._helper_usage_section(
            child_traces, truncated_stack, code_library_is_live=code_library_is_live,
        )
        base_prompt = self._build_prompt(
            sampled, truncated_stack, tasks, depth,
            previous_scores=previous_scores,
            archive_best_scores=archive_best_scores,
            solver_language=solver_language,
            no_code_library=no_code_library,
            current_scores=current_scores,
            helper_usage=helper_usage,
        )
        prompt = self._refine_directive(prev_injection, mean_before) + "\n\n" + base_prompt

        response, tokens = await self.llm_client.complete(
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature if temperature is not None else 0.7,
        )
        logger.info(
            "Omega refine: depth=%d, mean_before=%s, tokens=%d, response_len=%d",
            depth,
            f"{mean_before:.3f}" if mean_before is not None else "gate-fail",
            tokens, len(response),
        )
        injected = self._parse_response(response, depth)
        injected.raw_omega_prompt = prompt
        return injected, tokens

    @staticmethod
    def _refine_directive(
        prev_injection: InjectedCode, mean_before: float | None,
    ) -> str:
        """The FIX-THE-BUG directive prepended to a refine prompt.

        Frames the call as error-correction on Ω's OWN prior injection: keep the
        approach + helper names, make the smallest edit that makes it run.
        """
        score_str = (
            f"scored only a mean of {mean_before:.3f}"
            if mean_before is not None
            else "FAILED THE QUALITY GATE"
        )
        return (
            "## REFINE — FIX YOUR OWN BUGGY INJECTION (error-correction, not a rewrite)\n"
            "You ALREADY produced the meta-layer injection shown as the DEEPEST "
            f"layer (depth {prev_injection.source_depth}) in '## Previously "
            f"Injected Code' below. It {score_str}: the APPROACH is right but the "
            "CODE HAS A BUG.\n"
            "Emit a CORRECTED version of THAT SAME injection. FIX THE BUG — do NOT "
            "change the approach, do NOT switch algorithms, do NOT start over. Keep "
            "the same helper names and the same overall strategy; make the SMALLEST "
            "edit that makes it run correctly (guard the failing op, fix the "
            "indexing/format/numeric error, handle the edge case the traces show). "
            "Output the full corrected injection in the same format as before.\n"
        )

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        traces: list[Trace],
        context_stack: list[InjectedCode],
        tasks: list[TaskDescription],
        depth: int,
        inspiration_traces: list[Trace] | None = None,
        previous_scores: dict[str, float] | None = None,
        archive_best_scores: dict[str, float] | None = None,
        solver_language: str = "python",
        no_code_library: bool = False,
        current_scores: dict[str, float] | None = None,
        helper_usage: str = "",
        focus_task: str | None = None,
        prompt_variant: str | None = None,
        downstream_injections: list[InjectedCode] | None = None,
        within_task_recursion: bool = False,
    ) -> str:
        """Build the Omega prompt from traces and context.

        ``downstream_injections`` (Stage-3 re-propagation) renders an
        above-layer DOWNSTREAM FEEDBACK section appended to the injected-code
        block (mirroring how ``helper_usage`` is appended — no new template
        slot, so the render surface is unchanged). When ``None`` / empty (every
        non-re-propagation caller) NOTHING is appended and the prompt is
        BYTE-IDENTICAL — the omega-prompt golden depends on this.
        """
        from meta_n.core.prompts import (
            SOLVER_LIB_OUTPUT_FORMAT_BASH,
            SOLVER_LIB_OUTPUT_FORMAT_PYTHON,
            SOLVER_LIB_SECTION_BASH,
            SOLVER_LIB_SECTION_OPENEVOLVE,
            SOLVER_LIB_SECTION_PYTHON,
        )

        # Choose solver_lib section and output format based on language.
        # E2 ablation: when no_code_library is set, both placeholders become
        # empty strings, removing the solver_lib schema and examples from the
        # rendered prompt entirely.
        if no_code_library:
            solver_lib_section = ""
            solver_lib_output_format = ""
        elif solver_language == "bash":
            solver_lib_section = SOLVER_LIB_SECTION_BASH
            solver_lib_output_format = SOLVER_LIB_OUTPUT_FORMAT_BASH
        elif solver_language == "openevolve":
            solver_lib_section = SOLVER_LIB_SECTION_OPENEVOLVE
            solver_lib_output_format = SOLVER_LIB_OUTPUT_FORMAT_PYTHON
        else:
            solver_lib_section = SOLVER_LIB_SECTION_PYTHON
            solver_lib_output_format = SOLVER_LIB_OUTPUT_FORMAT_PYTHON

        # Build task_id → metadata lookup for category-aware summaries
        task_metadata = {t.task_id: t.metadata for t in tasks}

        # Collect any env-constraint blocks the adapters populated into
        # task metadata under the ``omega_env_notes`` key. Ω uses these to
        # avoid synthesizing env-hostile helpers (venv creation, ripgrep
        # usage, etc) that we've empirically seen across both gpt-5.2 and
        # Qwen3-Coder runs. Adapters own the content; omega.py only does
        # a generic dedup-and-join so other benchmarks need no core changes.
        # Env notes are a system-level concern (apply to all tasks in the
        # run), so they render as their own ``{env_notes_section}`` slot
        # between ``## System Context`` and the per-task traces — not
        # nested inside the traces themselves.
        env_notes_unique: list[str] = []
        seen_notes: set[str] = set()
        for m in task_metadata.values():
            note = m.get("omega_env_notes")
            if isinstance(note, str) and note and note not in seen_notes:
                env_notes_unique.append(note)
                seen_notes.add(note)
        env_notes_section = (
            "\n" + "\n\n".join(env_notes_unique) + "\n" if env_notes_unique else ""
        )

        # Choose trace presentation and template. ADDITIVE: prompt_variant
        # "discover" forces the discovery template + raw-trace presentation,
        # overriding the depth-based default. When prompt_variant is None
        # (the default) this branch is skipped and the depth-based selection
        # below is byte-identical to the prior behavior.
        if prompt_variant == "discover":
            traces_text = self._format_raw_traces(traces)
            if previous_scores is not None:
                traces_text += self._format_score_comparison(
                    traces, previous_scores, archive_best_scores,
                )
            template = OMEGA_PROMPT_DISCOVER
            prompt_name = "OMEGA_PROMPT_DISCOVER"
        elif depth >= 3 and previous_scores is not None:
            traces_text = self._summarize_traces(
                traces, previous_scores, context_stack,
                archive_best_scores=archive_best_scores,
                task_metadata=task_metadata,
                focus_task=focus_task,
            )
            template = OMEGA_PROMPT_META
            prompt_name = "OMEGA_PROMPT_META"
        else:
            traces_text = self._format_raw_traces(traces)
            if previous_scores is not None:
                traces_text += self._format_score_comparison(
                    traces, previous_scores, archive_best_scores,
                )
            template = OMEGA_PROMPT
            prompt_name = "OMEGA_PROMPT"
        logger.info("Omega prompt: depth=%d, template=%s, lang=%s", depth, prompt_name, solver_language)

        # Append inspiration traces
        if inspiration_traces:
            traces_text += self._format_inspiration(inspiration_traces)

        # Format context stack
        context_stack_text = self._format_context_stack(context_stack)
        # G4: append the dead-helper / call-rate callout to the injected-code
        # section (no new template slot — keeps the render surface unchanged).
        if helper_usage:
            context_stack_text += "\n" + helper_usage
        # Stage-3 re-propagation: append the above-layer DOWNSTREAM FEEDBACK
        # section (same no-new-slot append pattern as helper_usage). Empty/absent
        # ⇒ "" ⇒ nothing appended ⇒ byte-identical prompt.
        downstream_text = self._format_downstream_feedback(downstream_injections)
        if downstream_text:
            context_stack_text += "\n" + downstream_text

        # Compute stats
        num_failures = sum(1 for t in traces if not t.success)
        num_successes = sum(1 for t in traces if t.success)
        pass_at_1 = num_successes / len(traces) if traces else 0.0

        # G1 (continuous-score objective headline) + G2 (per-task headroom
        # table). current_scores (authoritative parent per-task scores) is
        # preferred; the helper falls back to deriving per-task scores from the
        # full traces so direct callers / tests still render a sensible line.
        score_summary, headroom_section = self._format_objective_and_headroom(
            traces, current_scores, archive_best_scores, pass_at_1,
        )
        # G5: append a plateau/novelty directive to the objective headline when
        # the previous layer barely moved the score (depth ≥3 only).
        score_summary += self._novelty_directive(
            current_scores, previous_scores, depth,
        )
        # G9: in consolidate mode, Ω improves ONLY the focus task — every other
        # task is frozen at its best and will not be re-solved, so all effort
        # should go to the one target.
        #
        # Forensic improvement #3 — WITHIN-TASK RECURSION (default OFF ⇒ the
        # EXISTING freeze block is rendered VERBATIM, so the omega-prompt golden is
        # byte-identical). When ON, the FOCUS-freeze directive is replaced with a
        # DEEPEN directive: the layer below already worked this SAME task, so this
        # deeper layer must reason about that solution and improve it FURTHER (a
        # genuine within-task re-work that compounds ``trace.depth`` on one task)
        # rather than gate a disjoint task at depth. ``focus_task`` is still the
        # one task to improve in both branches; only the framing changes.
        if focus_task:
            if within_task_recursion:
                score_summary += (
                    f"\n## DEEPEN — keep improving ONLY task '{focus_task}'\n"
                    f"The layer BELOW already worked '{focus_task}'; its current "
                    f"solution is the starting point, NOT a blank slate. Reason "
                    f"about what that layer produced and improve '{focus_task}' "
                    f"FURTHER — do NOT discard or restart it. Every OTHER task is "
                    f"FROZEN and will NOT be re-solved; spend NO effort on them.\n"
                )
            else:
                score_summary += (
                    f"\n## FOCUS — improve ONLY task '{focus_task}'\n"
                    f"Every OTHER task is FROZEN at its best-known solution and will "
                    f"NOT be re-solved; spend NO effort on them. Make the single "
                    f"highest-impact change for '{focus_task}' alone.\n"
                )

        return template.format(
            depth=depth,
            num_tasks=len(tasks),
            pass_at_1=pass_at_1,
            score_summary=score_summary,
            headroom_section=headroom_section,
            num_failures=num_failures,
            num_successes=num_successes,
            traces_text=traces_text,
            env_notes_section=env_notes_section,
            context_stack_text=context_stack_text,
            solver_lib_section=solver_lib_section,
            solver_lib_output_format=solver_lib_output_format,
        )

    def _format_objective_and_headroom(
        self,
        traces: list[Trace],
        current_scores: dict[str, float] | None,
        archive_best_scores: dict[str, float] | None,
        pass_at_1: float,
    ) -> tuple[str, str]:
        """G1 + G2: the continuous-score objective headline and the per-task
        headroom-to-best opportunity table.

        Returns ``(score_summary, headroom_section)`` — both already
        newline-terminated (or empty). ``score_summary`` reframes the headline
        away from binary pass@1 toward the MEAN CONTINUOUS SCORE (the real
        selection objective on continuous-eval benchmarks); pass@1 stays as a
        secondary line. ``headroom_section`` ranks tasks by how much room they
        have to the best-known score, so Ω spends effort where the gap is —
        instead of inferring it from raw traces.

        ``current_scores`` (authoritative parent per-task scores) is preferred;
        when absent it is derived from the full ``traces`` (best score per task)
        so direct callers and tests still get a sensible objective line. All
        means ignore non-finite scores.
        """
        cur: dict[str, float] = {}
        if current_scores:
            cur = {
                k: float(v)
                for k, v in current_scores.items()
                if isinstance(v, (int, float)) and math.isfinite(v)
            }
        if not cur:
            for t in traces:
                s = t.score if math.isfinite(t.score) else 0.0
                if t.task_id not in cur or s > cur[t.task_id]:
                    cur[t.task_id] = s

        # F020: unit-scale detection, shared by the G1 headline and the G2
        # fallback below. Any observed value outside [0,1] rules out the
        # default unit benchmark scale (e.g. symbolic_regression, algotune);
        # pure function of cur + archive_best_scores.
        observed = list(cur.values()) + [
            float(v)
            for v in (archive_best_scores or {}).values()
            if isinstance(v, (int, float)) and math.isfinite(v)
        ]
        unit_scale = all(0.0 <= v <= 1.0 for v in observed)

        # --- G1: objective headline ---
        if cur:
            mean = sum(cur.values()) / len(cur)
            if unit_scale:
                lo = sum(1 for v in cur.values() if v < 0.3)
                mid = sum(1 for v in cur.values() if 0.3 <= v < 0.7)
                hi = sum(1 for v in cur.values() if v >= 0.7)
                spread_line = (
                    f"Score spread ({len(cur)} tasks): "
                    f"[0.0-0.3]x{lo}  [0.3-0.7]x{mid}  [0.7-1.0]x{hi}\n"
                )
            else:
                # Continuous / unbounded scale: fixed [0,1] buckets are
                # meaningless (F020) — communicate the spread scale-free.
                spread_line = (
                    f"Score spread ({len(cur)} tasks): "
                    f"min={min(cur.values()):.3f}  "
                    f"median={statistics.median(cur.values()):.3f}  "
                    f"max={max(cur.values()):.3f}\n"
                )
            score_summary = (
                f"Current mean score: {mean:.3f}  "
                f"← PRIMARY OBJECTIVE: raise the MEAN CONTINUOUS SCORE "
                f"(not just the pass count)\n"
                f"{spread_line}"
                f"pass@1 (binary success rate): {pass_at_1:.1%}\n"
            )
        else:
            score_summary = f"Current pass@1: {pass_at_1:.1%}\n"

        # --- G2: headroom-to-best opportunity table ---
        headroom_section = ""
        if cur and archive_best_scores:
            rows = []
            for tid, now in cur.items():
                best = archive_best_scores.get(tid, now)
                rows.append((best - now, tid, now, best))
            total_gap = sum(max(g, 0.0) for g, *_ in rows)
            # F020 reviewed this absolute 0.02 gate and left it as-is: on
            # non-unit scales its failure direction is benign (the informative
            # headroom table renders MORE often, not less).
            if total_gap > 0.02:
                rows.sort(key=lambda r: -r[0])
                lines = [
                    "## Opportunity — per-task headroom to best-known "
                    "(largest gap first)",
                    "Target the largest-gap tasks. Tasks with a small/zero gap "
                    "are near their ceiling — do NOT regress them.",
                    "",
                    "| task | now | best-known | gap |",
                    "|------|-----|-----------|-----|",
                ]
                for gap, tid, now, best in rows[:10]:
                    lines.append(f"| {tid} | {now:.3f} | {best:.3f} | {gap:+.3f} |")
                headroom_section = "\n".join(lines) + "\n"
            else:
                # Everything is at best-known (e.g. the depth-2 seed, the only
                # candidate). Fall back to a lowest-scoring-tasks table so it
                # still points at where the room is.
                #
                # The "absolute room toward a perfect 1.0" column used to
                # hardcode a 1.0 ceiling (``1.0 - now``). That is meaningless on
                # a continuous / negative-capable score_scale (e.g.
                # symbolic_regression, hi=None): a now=-5 task would report room
                # 6.0 toward a "perfect 1.0" that does not exist. We don't
                # receive ``score_scale`` here, but an unbounded/non-unit scale
                # is detectable from the observed scores — any value outside
                # [0,1] rules out the default unit benchmark scale (the shared
                # ``unit_scale`` hoisted above). On the default [0,1] scale
                # this renders BYTE-IDENTICALLY to before.
                if unit_scale:
                    rows2 = sorted(
                        ((1.0 - now, tid, now) for tid, now in cur.items()),
                        key=lambda r: -r[0],
                    )
                    lines = [
                        "## Opportunity — lowest-scoring tasks (most absolute room)",
                        "All tasks are at the best-known score; these have the most "
                        "room toward a perfect 1.0:",
                        "",
                        "| task | now | room->1.0 |",
                        "|------|-----|----------|",
                    ]
                    for room, tid, now in rows2[:10]:
                        lines.append(f"| {tid} | {now:.3f} | {room:.3f} |")
                    headroom_section = "\n".join(lines) + "\n"
                else:
                    # Continuous / unbounded scale: no fixed 1.0 ceiling exists,
                    # so rank by raw score ascending and drop the room->ceiling
                    # column (sorting by score asc is order-identical to the old
                    # sort by ``1.0 - now`` desc, so the table's job — surface the
                    # lowest-scoring tasks — is preserved).
                    rows2 = sorted(cur.items(), key=lambda r: r[1])
                    lines = [
                        "## Opportunity — lowest-scoring tasks (most room to improve)",
                        "All tasks are at the best-known score; these score lowest "
                        "and have the most room to improve:",
                        "",
                        "| task | now |",
                        "|------|-----|",
                    ]
                    for tid, now in rows2[:10]:
                        lines.append(f"| {tid} | {now:.3f} |")
                    headroom_section = "\n".join(lines) + "\n"
        return score_summary, headroom_section

    def _helper_usage_section(
        self, traces: list[Trace], context_stack: list[InjectedCode],
        code_library_is_live: bool = True,
    ) -> str:
        """G4: did the solver actually CALL each injected helper?

        Forensic log analysis found injected helpers are frequently DEFINED but
        never CALLED (e.g. OH/T2 ``utilities_called: []``; Ω re-emitting an
        ``mcmf_dijkstra`` no solver invokes). A helper the solver never calls is
        pure prompt overhead. This scans every (library-prefix-stripped) solver
        script for a word-boundary reference to each helper name and reports the
        call rate, flagging 0-call helpers as DEAD so Ω prunes them or fixes the
        CALL via pre_process instead of silently re-injecting them.

        Counts over the FULL traces (passed from ``generate`` before sampling),
        so a helper used outside the failure-biased sample is not mis-flagged.
        Returns "" when no helpers are in the active stack.

        P8(a): when ``code_library_is_live`` is False the helpers are DEMOTED —
        never prepended to the solver (CO-Bench/SWE) — so a ~0 call-rate is
        trivially true and the "DEAD — remove it" feedback would MISLEAD Ω into
        abandoning helper synthesis for a framework-level decision it cannot
        see. Skip the section entirely in that case.
        """
        if not code_library_is_live:
            return ""
        helpers: list[tuple[str, object]] = []
        seen: set[str] = set()
        for code in context_stack:
            names = list(getattr(code, "code_library", {}) or {}) + list(
                getattr(code, "code_library_bash", {}) or {}
            )
            for name in names:
                if name and name not in seen:
                    seen.add(name)
                    helpers.append((name, getattr(code, "source_depth", None)))
        if not helpers:
            return ""

        scripts = [self._strip_library_prefix(t.script or "") for t in traces]
        n = len(scripts) or 1
        # S0.2: the call-detection regex (def-shadow exclusion + helpers/-anchored
        # FILE form) now lives in ``meta_layer.scan_helper_calls`` (single source
        # of truth shared with the fresh-trace adoption populator). Pre-compute the
        # per-script set of CALLED helper names once; the per-name aggregate below
        # is byte-identical to the previous inline
        # ``sum(1 for s in scripts if (pat.search(s) and not def_pat.search(s))
        # or file_pat.search(s))`` (membership is independent across names).
        helper_names = [name for name, _ in helpers]
        per_script_called = [
            set(scan_helper_calls(s, helper_names)[0]) for s in scripts
        ]
        lines = [
            "## Helper Usage — did the solver actually CALL your injected helpers?",
            "",
        ]
        any_dead = False
        for name, d in helpers:
            called = sum(1 for cs in per_script_called if name in cs)
            depth_str = f" (depth {d})" if d is not None else ""
            if called == 0:
                any_dead = True
                lines.append(
                    f"- `{name}`{depth_str}: called in 0/{n} solver scripts — "
                    "DEAD. Remove it, or make pre_process explicitly tell the "
                    "solver WHEN and HOW to call it."
                )
            else:
                lines.append(
                    f"- `{name}`{depth_str}: called in {called}/{n} solver scripts."
                )
        if any_dead:
            lines += [
                "",
                "A helper the solver never calls is pure prompt overhead. Prefer "
                "fixing the CALL (clearer pre_process) over adding more helpers.",
            ]
        return "\n".join(lines) + "\n"

    @staticmethod
    def _scale_epsilon(*score_maps: dict[str, float] | None, base: float = 0.01) -> float:
        """Scale-aware near-zero threshold (F020).

        Default unit scale (every observed finite score in [0,1]) -> returns
        ``base`` EXACTLY (byte-identical default path — same detection rule as
        the G2 headroom fallback). Non-unit scale -> ``base * observed_range``
        (range floored at 1e-9, mirroring archive.score_range()), i.e. "base
        fraction of the observed score spread" instead of an absolute 0.01
        that is meaningless on continuous/unbounded scales.
        """
        vals = [
            float(v)
            for m in score_maps
            if m
            for v in m.values()
            if isinstance(v, (int, float)) and math.isfinite(v)
        ]
        if not vals or all(0.0 <= v <= 1.0 for v in vals):
            return base
        return base * max(max(vals) - min(vals), 1e-9)

    def _novelty_directive(
        self,
        current_scores: dict[str, float] | None,
        previous_scores: dict[str, float] | None,
        depth: int,
    ) -> str:
        """G5: when the previous layer barely moved the score, push Ω toward a
        structurally different intervention — or an honest no-op.

        Forensic log analysis: Ω reliably plateaus by ~depth 3 on EVERY
        benchmark, and the plateau is idea-exhaustion (headroom remains), not a
        ceiling — Ω re-emits / over-promises a refinement of the last approach
        (trim6 gen5). The signal is the last layer's own delta (parent vs the
        layer before it). Only fires at depth ≥3 (don't discourage the first Ω
        layer) and only when that delta is a near-no-op or regression: ≤ +0.01
        on the unit scale; ≤ 1% of the observed score spread on non-unit
        scales (F020 — scale detected from these two score maps only; this
        signature receives no archive-best map).
        """
        if depth < 3 or not current_scores or not previous_scores:
            return ""
        shared = [
            t for t in current_scores
            if t in previous_scores
            and math.isfinite(current_scores[t])
            and math.isfinite(previous_scores[t])
        ]
        if not shared:
            return ""
        delta = (
            sum(current_scores[t] for t in shared)
            - sum(previous_scores[t] for t in shared)
        ) / len(shared)
        eps = self._scale_epsilon(current_scores, previous_scores)
        if delta > eps:
            return ""
        return (
            f"\n⚠ PLATEAU RISK: the previous layer changed the mean score by only "
            f"{delta:+.3f}. Do NOT merely refine the previous approach — the "
            f"injected-code history below shows what has already been tried. "
            f"Either (a) propose a STRUCTURALLY DIFFERENT intervention (a "
            f"different algorithm class, a different failure mode, a different "
            f"task subset), or (b) if you have no genuinely new idea, output an "
            f"EMPTY response (no pre_process, no solver_lib): a no-op beats "
            f"re-emitting code that will not help.\n"
        )

    # Single source of truth for the injected-library marker strip is
    # meta_layer._strip_library_prefix_for_scan; staticmethod alias (same
    # pattern as _classify_error below) so all call sites resolve unchanged.
    _strip_library_prefix = staticmethod(_strip_library_prefix_for_scan)

    def _format_raw_traces(self, traces: list[Trace]) -> str:
        """Format traces as raw per-task details (used at depth 2)."""
        parts = []
        for t in traces:
            status = "SUCCESS" if t.success else "FAILURE"
            # Render any signed score: negative scores are valid on unbounded
            # scales (e.g. symbolic_regression). Zero stays hidden (the
            # no-signal sentinel throughout this file) and NaN stays hidden
            # (both comparisons are False) — [0,1] paths are byte-identical.
            score_str = f" score={t.score:.3f}" if (t.score > 0 or t.score < 0) else ""
            # G3: surface the structured failure CLASS inline at depth 2 (where
            # Ω otherwise gets only raw stderr and must infer the category). The
            # same taxonomy already drives the depth-3 histogram and the channel
            # guidance — exposing it per-trace lets Ω act on it one depth sooner.
            class_str = f" {self._classify_error(t)}" if not t.success else ""
            part = f"--- Task: {t.task_id} [{status}{score_str}{class_str}] ---\n"
            solver_script = self._strip_library_prefix(t.script)
            lang = detect_script_language(solver_script)
            part += f"Script:\n```{lang}\n{solver_script}\n```\n"
            if t.stdout:
                part += f"Stdout:\n{_head_tail(t.stdout, 200, 300)}\n"
            if t.stderr:
                part += f"Stderr:\n{_tail(t.stderr, 800)}\n"
            if t.error_summary:
                part += f"Error: {t.error_summary}\n"
            if t.eval_feedback:
                part += f"Eval feedback:\n{_tail(t.eval_feedback, 800)}\n"
            parts.append(part)
        return "\n".join(parts) if parts else "(no traces yet)"

    def _format_score_comparison(
        self,
        traces: list[Trace],
        previous_scores: dict[str, float],
        archive_best_scores: dict[str, float] | None = None,
    ) -> str:
        """Format a lightweight score comparison section (used at depth 2).

        Appended to raw traces to give Omega quantitative regression signal.
        """
        # Build current per-task scores (highest depth per task if mixed)
        current: dict[str, float] = {}
        for t in traces:
            if t.task_id not in current or t.score > current[t.task_id]:
                current[t.task_id] = t.score

        lines = ["\n## Baseline Comparison", ""]
        lines += self._format_delta_lines(current, previous_scores, archive_best_scores)
        return "\n".join(lines)

    def _format_delta_lines(
        self,
        current: dict[str, float],
        previous_scores: dict[str, float],
        archive_best_scores: dict[str, float] | None,
        *,
        max_rows: int | None = None,
        always_show_unchanged: bool = False,
        verbose_net_effect: bool = False,
    ) -> list[str]:
        """Single source of truth for the improved/regressed/unchanged delta
        rows shared by the depth-2 Baseline Comparison (defaults) and the
        depth-3 Section C (``max_rows=5``, ``always_show_unchanged=True``,
        ``verbose_net_effect=True``). Byte-identical to both former inline
        copies for unit-scale ([0,1]) inputs — which covers all four pinned
        tests/test_refine_omega.py::TestDeltaSectionByteIdentity cases; on
        non-unit scales the improved/regressed threshold is scale-aware
        (``_scale_epsilon``, F020) instead of an absolute ±0.01.
        """
        # F020: scale detection includes archive_best_scores (scale is a
        # benchmark property, not a property of the two dicts — mirrors the G2
        # ``observed`` construction). Unit-scale inputs get exactly 0.01.
        eps = self._scale_epsilon(current, previous_scores, archive_best_scores)
        improved, regressed, unchanged = [], [], []
        for tid, curr_score in current.items():
            prev = previous_scores.get(tid, 0.0)
            delta = curr_score - prev
            if delta > eps:
                improved.append((tid, prev, curr_score, delta))
            elif delta < -eps:
                regressed.append((tid, prev, curr_score, delta))
            else:
                unchanged.append(tid)

        lines: list[str] = []
        if improved:
            improved.sort(key=lambda x: -x[3])
            lines.append(f"Tasks that IMPROVED ({len(improved)}):")
            for tid, prev, curr, delta in (
                improved if max_rows is None else improved[:max_rows]
            ):
                lines.append(f"  {tid}: {prev:.3f} → {curr:.3f} (+{delta:.3f})")
            if max_rows is not None and len(improved) > max_rows:
                lines.append(f"  ... and {len(improved) - max_rows} more")
        if regressed:
            regressed.sort(key=lambda x: x[3])
            lines.append(f"Tasks that REGRESSED ({len(regressed)}):")
            for tid, prev, curr, delta in (
                regressed if max_rows is None else regressed[:max_rows]
            ):
                suffix = ""
                if archive_best_scores and tid in archive_best_scores:
                    suffix = f" [archive best: {archive_best_scores[tid]:.3f}]"
                lines.append(f"  {tid}: {prev:.3f} → {curr:.3f} ({delta:.3f}){suffix}")
            if max_rows is not None and len(regressed) > max_rows:
                lines.append(f"  ... and {len(regressed) - max_rows} more")
        if unchanged or always_show_unchanged:
            lines.append(f"Tasks unchanged: {len(unchanged)}")

        # Net effect over tasks present in both layers.
        shared = [tid for tid in current if tid in previous_scores]
        if shared:
            curr_mean = sum(current[tid] for tid in shared) / len(shared)
            prev_mean = sum(previous_scores[tid] for tid in shared) / len(shared)
            net = curr_mean - prev_mean
            # Normalize at display precision: any net effect that rounds to
            # zero at 3 decimals must render as "+0.000", never "-0.000".
            # Checking `net == 0.0` is not enough — pre-3.12 sum() (no Neumaier
            # compensation) leaves a tiny signed residue on exactly-cancelling
            # deltas, so the sign of the rendered zero was Python-version
            # dependent. round() collapses the residue to ±0.0 and `+ 0.0`
            # canonicalizes IEEE negative zero to positive.
            net = round(net, 3) + 0.0
            if verbose_net_effect:
                lines.append(
                    f"Net effect: {net:+.3f} mean score change (over {len(shared)} shared tasks)"
                )
            else:
                lines.append(f"Net effect: {net:+.3f} mean score change")
        elif verbose_net_effect:
            lines.append("Net effect: N/A (no shared tasks between layers)")

        return lines

    def _format_inspiration(self, inspiration_traces: list[Trace]) -> str:
        """Format inspiration traces from other candidates."""
        parts = []
        for t in inspiration_traces:
            # Same signed-score rule as _format_raw_traces: nonzero renders,
            # zero/NaN stay hidden (archive-best can be negative on SR scales).
            score_str = f" score={t.score:.3f}" if (t.score > 0 or t.score < 0) else ""
            part = f"--- Task: {t.task_id} [BEST{score_str}] (from another candidate) ---\n"
            solver_script = self._strip_library_prefix(t.script)
            lang = detect_script_language(solver_script)
            part += f"Script:\n```{lang}\n{solver_script}\n```\n"
            parts.append(part)
        return (
            "\n\n## Inspiration from Archive"
            " (best solutions from other candidates for tasks you failed)\n"
            + "\n".join(parts)
        )

    def _format_context_stack(self, context_stack: list[InjectedCode]) -> str:
        """Format previously injected code layers."""
        parts = []
        for code in context_stack:
            part = f"--- Depth {code.source_depth} ---\n"
            if code.pre_process:
                part += f"pre_process:\n```python\n{code.pre_process}\n```\n"
            if code.code_library:
                for name, src in code.code_library.items():
                    part += f"solver_lib:{name}:\n```python\n{src}\n```\n"
            if code.code_library_bash:
                for name, src in code.code_library_bash.items():
                    part += f"solver_lib_bash:{name}:\n```bash\n{src}\n```\n"
            if code.rationale:
                part += f"Rationale: {code.rationale}\n"
            parts.append(part)
        return "\n".join(parts) if parts else "(none — you are the first meta-layer)"

    def _format_downstream_feedback(
        self, downstream_injections: list[InjectedCode] | None,
    ) -> str:
        """Stage-3 re-propagation: render the ABOVE-layer injections as downstream
        feedback (error-correction framing).

        Shows the deeper meta-layer(s) that were generated AFTER and ON TOP OF the
        layer Ω is now revising, so Ω knows (a) the failure traces are FULL-CHAIN
        — contaminated by these layers' behavior — and (b) it must fix the current
        depth WITHOUT breaking what depends on it: error-correction, not a fresh
        redesign. Returns "" when there is no above-layer feedback (the default,
        non-re-propagation path) so the prompt stays byte-identical.
        """
        if not downstream_injections:
            return ""
        lines = [
            "## Downstream layers built ON TOP of this one "
            "(re-propagation — error-correction only)",
            "",
            "The meta-layer(s) below were generated AFTER and depend ON the layer "
            "you are now revising. The failure traces above are FULL-CHAIN (they "
            "ran with these deeper layers active), so they are partly contaminated "
            "by behavior these layers add. FIX the bug at THIS depth so the whole "
            "chain works — keep this layer's approach COMPATIBLE with the layers "
            "below; do NOT redesign it in a way that breaks them. This is "
            "error-correction, not a new design.",
            "",
            self._format_context_stack(downstream_injections),
        ]
        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------
    # Depth 3+ structured summary
    # ------------------------------------------------------------------

    def _summarize_traces(
        self,
        traces: list[Trace],
        previous_scores: dict[str, float],
        context_stack: list[InjectedCode],
        max_representative: int = 3,
        archive_best_scores: dict[str, float] | None = None,
        task_metadata: dict[str, dict] | None = None,
        focus_task: str | None = None,
    ) -> str:
        """Build a structured summary for depth 3+ prompts.

        Sections: task categories, failure patterns, previous layer effectiveness,
        and a few representative raw traces.
        """
        # Get current per-task traces (highest depth per task if mixed).
        # Named current_traces (values are Traces, not floats) to avoid
        # colliding with the dict[str, float] current_scores used elsewhere.
        current_traces: dict[str, Trace] = {}
        for t in traces:
            if t.task_id not in current_traces or t.depth > current_traces[t.task_id].depth:
                current_traces[t.task_id] = t

        sections = []

        # --- Section A: Task Category Performance ---
        categories: dict[str, list[Trace]] = {}
        for t in current_traces.values():
            meta = task_metadata.get(t.task_id) if task_metadata else None
            cat = self._categorize_task(t.task_id, metadata=meta)
            categories.setdefault(cat, []).append(t)

        lines = ["## Task Category Performance", ""]
        for cat in sorted(categories.keys()):
            cat_traces = categories[cat]
            count = len(cat_traces)
            passed = sum(1 for t in cat_traces if t.success)
            mean = sum(t.score for t in cat_traces) / count if count else 0
            failures = [t for t in cat_traces if not t.success]
            if failures:
                error_counts = Counter(self._classify_error(t) for t in failures)
                key_issue = error_counts.most_common(1)[0][0]
            else:
                key_issue = "-"
            lines.append(
                f"- **{cat}** ({count} tasks): {passed}/{count} pass, "
                f"mean={mean:.2f}, key issue: {key_issue}"
            )
        sections.append("\n".join(lines))

        # --- Section B: Aggregated Failure Patterns ---
        all_failures = [t for t in current_traces.values() if not t.success]
        if all_failures:
            error_counts = Counter(self._classify_error(t) for t in all_failures)
            total_fail = len(all_failures)
            lines = ["## Failure Pattern Distribution", ""]
            for error_type, count in error_counts.most_common():
                pct = count / total_fail * 100
                example = next(
                    t for t in all_failures if self._classify_error(t) == error_type
                )
                lines.append(
                    f"- **{error_type}**: {count} tasks ({pct:.0f}%) "
                    f'— e.g., "{example.error_summary[:80]}"'
                )
            sections.append("\n".join(lines))

        # --- Section C: Previous Layer Effectiveness ---
        if previous_scores:
            lines = ["## Previous Layer Effectiveness", ""]

            # Previous layer's rationale
            if context_stack and context_stack[-1].rationale:
                rationale_preview = context_stack[-1].rationale[:200]
                lines.append(f'Previous layer rationale: "{rationale_preview}..."')
                lines.append("")

            lines += self._format_delta_lines(
                {tid: tr.score for tid, tr in current_traces.items()},
                previous_scores,
                archive_best_scores,
                max_rows=5,
                always_show_unchanged=True,
                verbose_net_effect=True,
            )
            sections.append("\n".join(lines))

        # --- Section D: Representative Traces ---
        representative = self._select_representative_traces(
            current_traces, previous_scores, max_representative,
            focus_task=focus_task,
        )
        if representative:
            lines = [
                f"## Representative Traces ({len(representative)} of {len(current_traces)})",
                "",
            ]
            lines.append(self._format_raw_traces(representative))
            sections.append("\n".join(lines))

        return "\n\n".join(sections)

    def _select_representative_traces(
        self,
        current_traces: dict[str, Trace],
        previous_scores: dict[str, float] | None,
        max_count: int = 3,
        focus_task: str | None = None,
    ) -> list[Trace]:
        """Select diverse representative traces: worst failure, biggest regression, best success.

        P2 — focus-trace pin: in consolidate/focus mode (``focus_task`` set), Ω
        improves ONLY that one task, yet the diverse failure/regression/success
        picks below routinely OMIT the focus task's own trace (it may be a
        passing, non-regressed task). Ω was then left to diagnose the focus task
        from its NAME alone — the observed cause of a domain hallucination
        (a 2D cutting/packing task mistaken for MNL-revenue "assortment",
        0.575 → 0.0). So FORCE-INCLUDE the focus task's most-recent trace
        (its script source, docstring, and eval_feedback) as item 0, BEFORE the
        existing diverse picks. ``focus_task=None`` → current behavior unchanged.
        """
        selected: list[Trace] = []

        # 0. Focus-task pin (consolidate mode): the focus task's most-recent
        # trace is ``current_traces[focus_task]`` (the dict already holds the
        # highest-depth trace per task). Pin it first so it survives the
        # ``max_count`` truncation below.
        if focus_task:
            focus_trace = current_traces.get(focus_task)
            if focus_trace is not None:
                selected.append(focus_trace)

        # 1. Failure with most common error type
        failures = [t for t in current_traces.values() if not t.success]
        if failures:
            error_counts = Counter(self._classify_error(t) for t in failures)
            most_common_type = error_counts.most_common(1)[0][0]
            rep = next(t for t in failures if self._classify_error(t) == most_common_type)
            if rep not in selected:
                selected.append(rep)

        # 2. Biggest regression (if previous_scores available)
        if previous_scores:
            worst_delta = 0.0
            worst_trace = None
            for tid, trace in current_traces.items():
                delta = trace.score - previous_scores.get(tid, 0.0)
                if delta < worst_delta:
                    worst_delta = delta
                    worst_trace = trace
            if worst_trace and worst_trace not in selected:
                selected.append(worst_trace)

        # 3. Best success for contrast
        successes = [t for t in current_traces.values() if t.success]
        if successes:
            best = max(successes, key=lambda t: t.score)
            if best not in selected:
                selected.append(best)

        return selected[:max_count]

    # ------------------------------------------------------------------
    # Task and error classification
    # ------------------------------------------------------------------

    @staticmethod
    def _categorize_task(task_id: str, metadata: dict | None = None) -> str:
        """Classify a task into a category.

        Uses metadata["category"] if available (e.g. from task.toml),
        falls back to keyword matching on task_id (tuned for CO-Bench).
        """
        if metadata and metadata.get("category"):
            return metadata["category"]
        tid = task_id.lower()
        for category, keywords in TASK_CATEGORIES.items():
            if any(kw in tid for kw in keywords):
                return category
        return "other"

    # S0.1: the failure-class taxonomy now lives at module scope in
    # ``meta_layer.classify_error`` (single source of truth shared with the
    # Trace populator). The ``staticmethod`` alias preserves every existing
    # caller — ``self._classify_error(t)`` / ``OmegaEngine._classify_error(t)``
    # — byte-identically (omega.py:558/703/716/722/832/834,
    # evolutionary_orchestrator.py:1556, metrics.py:577).
    _classify_error = staticmethod(classify_error)

    # Public name for the task categorizer so external consumers (analysis/
    # metrics) need no underscore reach-in; the underscore name stays for
    # in-class callers and existing tests.
    categorize_task = _categorize_task

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_response(self, response: str, depth: int) -> InjectedCode:
        """Parse LLM response into InjectedCode."""
        pre_process = self._extract_block(response, "pre_process")
        code_library = self._extract_solver_libs(response)
        code_library_bash = self._extract_solver_libs_bash(response)
        rationale = self._extract_block(response, "rationale") or ""

        return InjectedCode(
            pre_process=pre_process,
            code_library=code_library,
            code_library_bash=code_library_bash,
            rationale=rationale,
            source_depth=depth,
            raw_omega_response=response,
        )

    def _extract_block(self, response: str, block_name: str) -> Optional[str]:
        """Extract a named fenced code block from the response."""
        pattern = rf"```{re.escape(block_name)}\s*\n(.*?)```"
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    def _extract_solver_libs(self, response: str) -> dict[str, str]:
        """Extract all solver_lib (Python) blocks, keyed by the DISCOVERED
        top-level function name rather than the fenced label.

        (b) N8: a label/funcname mismatch — ```solver_lib:foo``` whose body is
        ``def bar():`` — was silently dropped by the smoke test (which exec's the
        body and looks up the funcname), killing 17-37% of instruct-model
        libraries at load and crashing the spine ``_reexport`` (where the key
        doubles as the filename and re-export name). Re-keying by the real
        funcname fixes both. Falls back to the label if the body has no parseable
        top-level function.
        """
        libs = {}
        pattern = r"```solver_lib:(\w+)\s*\n(.*?)```"
        for match in re.finditer(pattern, response, re.DOTALL):
            label, body = match.group(1), match.group(2).strip()
            key = label
            try:
                for node in ast.parse(body).body:
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        key = node.name
                        break
            except SyntaxError:
                pass
            libs[key] = body
        return libs

    def _extract_solver_libs_bash(self, response: str) -> dict[str, str]:
        """Extract all solver_lib_bash blocks from the response."""
        libs = {}
        pattern = r"```solver_lib_bash:(\w+)\s*\n(.*?)```"
        for match in re.finditer(pattern, response, re.DOTALL):
            libs[match.group(1)] = match.group(2).strip()
        return libs
