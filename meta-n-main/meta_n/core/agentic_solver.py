"""Terminus 2-inspired agentic solver with iterative refinement loop.

Replaces Layer1Solver + MetaLayer when use_agentic=True. Handles
pre_process injection, library management, code generation, trial
execution, and multi-turn refinement in a single coherent loop.

Key patterns borrowed from Terminus 2:
- Agentic loop: observe → analyze → plan → act → repeat
- Two-stage completion confirmation
- Parse error feedback (errors become next observation)
- Context summarization when token budget is low. ``token_budget`` bounds the
  chars//4 CONTEXT estimate of the message list per call, NOT cumulative
  spend; an N-turn run can spend up to ~N x this bound (each turn re-sends
  the whole context). Cumulative real spend is capped separately by
  ``spend_budget`` (off by default).
- Best-of-N tracking (return highest-scoring trace)
"""

from __future__ import annotations

import logging
import math
import re
import time
from dataclasses import dataclass, field

from meta_n.core.base_executor import BaseExecutor
from meta_n.core.llm_client import LLMClient
from meta_n.core.meta_layer import (
    InjectedCode,
    TaskDescription,
    Trace,
    _head_tail,
    _strip_library_prefix_for_scan,
    classify_error,
    format_bash_library_descriptions,
    format_python_library_descriptions,
    merge_code_libraries,
    populate_adoption_fields,
    prepend_bash_library,
    prepend_python_library,
    run_pre_process,
)
from meta_n.core.agentic_prompts import (
    AGENTIC_PREAMBLE,
    AGENTIC_SYSTEM_PROMPT,
    COMPLETION_CONFIRMATION,
    OBSERVATION_TEMPLATE,
    PARSE_ERROR_FEEDBACK,
    SUMMARIZE_PROMPT,
    error_hint,
    get_language_instructions,
)
from meta_n.core.solver import extract_fenced_block

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal dataclasses
# ---------------------------------------------------------------------------


@dataclass
class _ParsedResponse:
    """Structured output from a single agentic turn."""

    analysis: str = ""
    plan: str = ""
    code: str = ""
    task_complete: bool = False
    # Whether a <status> tag was present at all — distinguishes a well-formed
    # decline of a completion check from a formatless (no-tag) response.
    has_status: bool = False
    raw_response: str = ""
    parse_errors: list[str] = field(default_factory=list)


@dataclass
class _AgenticResult:
    """Result of the full agentic loop."""

    best_trace: Trace
    reasoning_log: str  # concatenated analysis+plan from all turns
    total_tokens: int
    turns_used: int
    # "confirmed" | "perfect_score" | "max_turns" | "token_budget" | "env_error"
    # | "spend_budget" (real-spend cap; reachable only when spend_budget is set)
    # | "max_turns_unconfirmed_complete" | "token_budget_unconfirmed_complete"
    # | "spend_budget_unconfirmed_complete"
    #   (cap hit while an unconfirmed completion signal was pending). The
    #   suffixed labels deliberately CONTAIN their base substring so
    #   adoption.classify_error's substring match keeps the base class
    #   ("max_turns..." -> Turn starvation; "token_budget..." /
    #   "spend_budget..." fall through to the text scan exactly like their
    #   plain base labels).
    terminated_by: str
    # S0.3: turns wasted on a non-actionable parse failure (phantom-completion
    # rejection / parse error / no-code). S0.2: executor.execute invocations.
    parse_failure_turns: int = 0
    command_count: int = 0


# ---------------------------------------------------------------------------
# AgenticSolver
# ---------------------------------------------------------------------------


class AgenticSolver:
    """Terminus 2-inspired agentic solver with iterative refinement.

    Replaces Layer1Solver + MetaLayer when use_agentic=True in
    EvolutionaryConfig. Receives the full list of injected_codes from
    a candidate and handles pre_process, library, code generation,
    trial execution, and iterative refinement in a single loop.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        executor: BaseExecutor,
        injected_codes: list[InjectedCode] | None = None,
        *,
        solver_language: str = "python",
        max_turns: int = 5,
        temperature: float = 0.7,
        token_budget: int = 100_000,
        spend_budget: int | None = None,
        depth: int = 1,
        code_library_is_live: bool = True,
        demoted_code_library: dict | None = None,
        agentic_error_hints: bool = False,
        agentic_preamble: bool = False,
        no_outer_context: bool = False,
    ):
        self.llm_client = llm_client
        self.executor = executor
        self.injected_codes = injected_codes or []
        self.solver_language = solver_language
        self.max_turns = max_turns
        self.temperature = temperature
        # token_budget bounds the chars//4 CONTEXT estimate of the message
        # list per call (see _estimate_tokens) — it is NOT a spend cap.
        # spend_budget (tokens; None = off) caps cumulative REAL spend:
        # outer LLM usage plus inner llm()/llm_batch() usage summed across
        # turns; when exhausted the loop stops with
        # terminated_by="spend_budget".
        self.token_budget = token_budget
        self.spend_budget = spend_budget
        self.depth = depth
        # Stage 1 ORTHOGONAL base-agent floor-raisers — both default OFF so the
        # rendered system prompt (R2) and observation (R1) are byte-identical to
        # HEAD. Builtin-AgenticSolver ONLY (the external OH/T2 path builds its
        # prompts via InjectionMapper and never reaches these attrs).
        self.agentic_error_hints = agentic_error_hints
        self.agentic_preamble = agentic_preamble
        # E3 ablation: when True, pre_process runs with thread_outer_context=False
        # so a shallower block never sees a deeper block's emission as
        # outer_context. Default False => thread_outer_context=True =>
        # byte-identical to HEAD.
        self.no_outer_context = no_outer_context

        # Merge libraries from all injected codes
        self.merged_py, self.merged_bash = merge_code_libraries(self.injected_codes)
        # 4.3: where the solver regenerates code (measured call-rate ~0), demote
        # the dead Python-helper prepend. Bash helpers and live families keep.
        # ``demoted_code_library`` default None = legacy blanket zero (byte-identical
        # for every direct/external caller); when the orchestrator supplies the
        # already-demoted dict (seed/verified flags on), source_depth==0 seeds ride
        # through and their advertisement is restored via merged_py.
        if not code_library_is_live:
            self.merged_py = {} if demoted_code_library is None else demoted_code_library

    # ---- Public interface ------------------------------------------------

    async def execute(self, task: TaskDescription) -> tuple[Trace, int]:
        """Full pipeline: pre_process → agentic loop → best trace.

        Returns:
            Tuple of (best_trace, total_tokens_used).
        """
        start = time.time()

        # 1. Run all pre_process (reverse order: deepest first)
        context = self._run_all_pre_process(task)

        # 2. Add library descriptions to context
        lib_desc = self._format_library_descriptions()
        if lib_desc:
            context = f"{context}\n{lib_desc}" if context else lib_desc

        # 3. Agentic loop
        result = await self._agentic_loop(task, context)

        # 4. Finalize trace metadata
        result.best_trace.depth = self.depth
        result.best_trace.reasoning = result.reasoning_log
        result.best_trace.duration_s = time.time() - start
        # S0.3: stamp the parse-failure count; S0.2: stamp command_count + scan
        # helper adoption (gated on live helpers — self.merged_py is zeroed
        # upstream on the demoted path, so utilities_called stays None there).
        result.best_trace.parse_failure_turns = result.parse_failure_turns
        populate_adoption_fields(
            result.best_trace,
            command_count=result.command_count,
            merged_code_library=self.merged_py,
            merged_code_library_bash=self.merged_bash,
            executor=self.executor,
            solver_language=self.solver_language,
        )

        logger.info(
            "AgenticSolver d=%d task=%s: score=%.3f, turns=%d/%d, terminated=%s, tokens=%d",
            self.depth,
            task.task_id,
            result.best_trace.score,
            result.turns_used,
            self.max_turns,
            result.terminated_by,
            result.total_tokens,
        )
        return result.best_trace, result.total_tokens

    async def solve(
        self, task: TaskDescription, additional_context: str = ""
    ) -> tuple[str, str, int]:
        """SolverProtocol-compatible interface.

        Used by _run_chain_test_evaluation() when needs_re_solve=True (bash tasks).
        The primary interface is execute() — solve() is only for backward
        compatibility with code paths that expect (script, reasoning, tokens).
        """
        trace, tokens = await self.execute(task)
        # Strip the injected-library prefix so callers get the raw script
        # (single source of truth: meta_layer._strip_library_prefix_for_scan).
        raw_script = _strip_library_prefix_for_scan(trace.script)
        return raw_script, trace.reasoning, tokens

    # ---- Core loop -------------------------------------------------------

    async def _agentic_loop(
        self, task: TaskDescription, initial_context: str
    ) -> _AgenticResult:
        """Main loop (modeled on Terminus 2's _run_agent_loop).

        generate → execute → observe → repeat

        Key behaviors borrowed from Terminus 2:
        - Best-of-N tracking (return highest-scoring trace, not last)
        - Two-stage completion (require 2 consecutive 'complete' signals)
        - Parse error feedback (errors become next observation)
        - Context summarization at 85% token budget
        - Hard stop at 95% token budget
        - Early exit on score >= 1.0
        """
        system_msg = self._build_system_message(task, initial_context)
        messages: list[dict[str, str]] = [{"role": "user", "content": system_msg}]

        best_trace: Trace | None = None
        reasoning_parts: list[str] = []
        total_tokens = 0
        # Inner-LLM tokens accumulated across *every* turn's executor.execute().
        # Each turn evaluates a candidate script, which (for benchmarks like
        # text classification) makes per-case inner llm() calls inside the
        # chunk subprocess. The Trace returned per turn carries those
        # counts, but ``best_trace`` only retains one turn's worth — so
        # without summing here, every other turn's inner spend would be
        # in the ledger but not in summary.json's token_usage.inner_*
        # fields. The cumulative totals are stamped onto best_trace before
        # we return so the orchestrator's per-task accounting credits the
        # full cost.
        inner_tokens_total = 0
        inner_prompt_total = 0
        inner_completion_total = 0
        inner_calls_total = 0
        turns_completed = 0
        pending_completion = False  # Terminus 2 two-stage pattern
        parse_failure_turns = 0  # S0.3: turns lost to non-actionable parse failures
        command_count = 0  # S0.2: executor.execute invocations across all turns
        # 6.3 (audit #49): distinct stop reason when the LLM call raises post-retry.
        # When set, this overrides the token-estimate-derived token_budget/max_turns
        # label so an infra failure is not mislabeled as turn starvation.
        stop_reason: str | None = None

        async def _execute_and_track(code: str, turn: int, ctx: str) -> Trace:
            """Prepend library → execute once → update the loop accumulators.

            Single execution path for both the first-'complete' branch and the
            normal branch: counts the invocation (S0.2), sums inner-LLM usage,
            and applies the finite-score-only best_trace update. ``ctx`` is
            " on completion" or "" — log-string variant only.
            """
            nonlocal command_count, inner_tokens_total, inner_prompt_total, \
                inner_completion_total, inner_calls_total, best_trace
            exec_script = self._prepend_library(code)
            try:
                trace = await self.executor.execute(exec_script, task)
            except Exception as e:
                logger.warning(
                    "Turn %d task=%s: executor failed%s: %s",
                    turn, task.task_id, ctx, e,
                )
                trace = Trace(
                    task_id=task.task_id,
                    script=exec_script,
                    error_summary=f"Executor error: {e}",
                )
            command_count += 1  # S0.2: executor.execute invocation
            inner_tokens_total += int(getattr(trace, "inner_tokens", 0) or 0)
            inner_prompt_total += int(getattr(trace, "inner_prompt_tokens", 0) or 0)
            inner_completion_total += int(getattr(trace, "inner_completion_tokens", 0) or 0)
            inner_calls_total += int(getattr(trace, "inner_calls", 0) or 0)
            # Only finite scores can update best_trace. NaN > anything is
            # False; a NaN trace reaching this branch first would set
            # best_trace = NaN and then no finite trace could displace it.
            if math.isfinite(trace.score) and (
                best_trace is None or trace.score > best_trace.score
            ):
                best_trace = trace
            elif not math.isfinite(trace.score):
                logger.warning(
                    "Turn %d task=%s: trace.score=%r non-finite%s; "
                    "not updating best_trace",
                    turn, task.task_id, trace.score, ctx,
                )
            return trace

        def _stamp_inner(t: Trace) -> None:
            """Stamp the cumulative inner-LLM usage totals onto the trace
            being returned (every exit path must call this)."""
            t.inner_tokens = inner_tokens_total
            t.inner_prompt_tokens = inner_prompt_total
            t.inner_completion_tokens = inner_completion_total
            t.inner_calls = inner_calls_total

        for turn in range(1, self.max_turns + 1):
            # -- Real-spend budget check (no-op unless spend_budget is set) --
            # Unlike the token_budget gates below (context ESTIMATE only),
            # this compares actual accumulated usage: outer LLM tokens plus
            # inner llm()/llm_batch() tokens from executed scripts.
            if self.spend_budget is not None and (
                total_tokens + inner_tokens_total
            ) >= self.spend_budget:
                logger.info(
                    "Spend budget exhausted (%d >= %d tokens), stopping",
                    total_tokens + inner_tokens_total,
                    self.spend_budget,
                )
                # Cap hit while a confirmation answer was pending → the same
                # honest suffixed label as the token/turn gates; contains the
                # "spend_budget" base substring (see _AgenticResult.terminated_by).
                stop_reason = (
                    "spend_budget_unconfirmed_complete"
                    if pending_completion
                    else "spend_budget"
                )
                break
            # -- Token budget check (before LLM call) --
            estimated = self._estimate_tokens(messages)
            if estimated > self.token_budget * 0.95:
                logger.info(
                    "Token budget exhausted (%.0f%%), stopping",
                    estimated / self.token_budget * 100,
                )
                if pending_completion:
                    # Budget died while a confirmation answer was pending →
                    # honest cap label that preserves the unconfirmed
                    # completion signal. Contains the "token_budget" base
                    # substring (see _AgenticResult.terminated_by).
                    stop_reason = "token_budget_unconfirmed_complete"
                break
            if estimated > self.token_budget * 0.85:
                logger.info(
                    "Token budget at %.0f%%, summarizing context",
                    estimated / self.token_budget * 100,
                )
                try:
                    messages, summ_tokens = await self._summarize_context(
                        messages
                    )
                    total_tokens += summ_tokens
                except Exception as e:
                    logger.warning(
                        "Turn %d task=%s: summarization failed: %s — continuing without",
                        turn, task.task_id, e,
                    )

            # -- LLM call --
            try:
                response, tokens = await self.llm_client.complete(
                    messages=messages,
                    temperature=self.temperature,
                )
            except Exception as e:
                logger.warning(
                    "Turn %d task=%s: LLM call failed: %s — stopping loop",
                    turn, task.task_id, e,
                )
                stop_reason = "env_error"  # audit #49: post-retry infra failure, not a cap
                break
            total_tokens += tokens
            turns_completed = turn
            parsed = self._parse_response(response)
            # Whether this response ANSWERS a completion check — captured here
            # because pending_completion is reset further down before the
            # no-code branch runs.
            was_confirmation = pending_completion
            messages.append({"role": "assistant", "content": response})

            # -- Per-turn diagnostic (DEBUG: lands in run.log file handler) --
            logger.debug(
                "Turn %d task=%s: tokens=%d resp_len=%d code_len=%d "
                "analysis_len=%d status=%s parse_errors=%s",
                turn, task.task_id, tokens, len(response or ""),
                len(parsed.code or ""), len(parsed.analysis or ""),
                "complete" if parsed.task_complete else "working",
                parsed.parse_errors or [],
            )
            # If response has no code, log full raw to make the failure mode
            # investigable from run.log alone.
            if not parsed.code.strip():
                preview = (response or "").replace("\n", " | ")
                logger.debug(
                    "Turn %d task=%s NO-CODE raw_FULL=%s",
                    turn, task.task_id, preview,
                )

            # -- Accumulate reasoning --
            if parsed.analysis or parsed.plan:
                reasoning_parts.append(
                    f"[Turn {turn}] {parsed.analysis}\n{parsed.plan}"
                )

            # -- Two-stage completion check FIRST (before parse error check).
            #    On the confirmation turn the LLM may respond with just
            #    <status>complete</status> and no code — that's expected.
            #    If we checked parse errors first, this would be treated as
            #    a malformed response and the confirmation would never trigger.
            if parsed.task_complete:
                if pending_completion:
                    # Phantom-completion guard (6.3): the agent confirmed
                    # completion but never executed any code (e.g. Qwen
                    # narrate-not-act). REJECT — reset and push for real code
                    # instead of returning a score-0 fallback as 'confirmed'. The
                    # loop continues; max_turns then yields the honest
                    # no-executable-code fallback (terminated_by != 'confirmed').
                    if best_trace is None:
                        if command_count == 0:
                            # True phantom: nothing was ever executed.
                            logger.warning(
                                "Turn %d task=%s: phantom completion rejected "
                                "(confirmed without executing any code); pushing for code",
                                turn, task.task_id,
                            )
                            pending_completion = False
                            parse_failure_turns += 1  # S0.3: phantom-completion rejection
                            messages.append({
                                "role": "user",
                                "content": (
                                    "You marked the task complete, but no executable code "
                                    "has run yet. Provide the actual solution code now (in a "
                                    "code block), then confirm completion."
                                ),
                            })
                            continue
                        # Executed but every run's score was non-finite: the
                        # agent DID act, so this is a numeric failure, not a
                        # parse failure — accurate feedback, no S0.3 increment.
                        logger.warning(
                            "Turn %d task=%s: completion rejected: %d execution(s) "
                            "ran but none produced a finite score",
                            turn, task.task_id, command_count,
                        )
                        pending_completion = False
                        messages.append({
                            "role": "user",
                            "content": (
                                "You marked the task complete, but none of your executed "
                                "runs produced a valid (finite) score — the task is NOT "
                                "complete. Fix the underlying numeric issue (e.g. NaN / "
                                "overflow / division by zero) and provide the corrected "
                                "solution code now (in a code block)."
                            ),
                        })
                        continue
                    # A real trace was executed → confirm.
                    _stamp_inner(best_trace)
                    best_trace.terminated_by = "confirmed"  # audit #50: persist on the trace
                    return _AgenticResult(
                        best_trace=best_trace,
                        reasoning_log="\n\n".join(reasoning_parts),
                        total_tokens=total_tokens,
                        turns_used=turn,
                        terminated_by="confirmed",
                        parse_failure_turns=parse_failure_turns,
                        command_count=command_count,
                    )
                else:
                    # First 'complete' → execute code if present
                    if parsed.code.strip():
                        await _execute_and_track(
                            parsed.code, turn, " on completion"
                        )
                    if turn == self.max_turns:
                        # No room for the confirmation handshake → honest cap
                        # label that preserves the unconfirmed completion
                        # signal (additive value; the "max_turns" substring
                        # keeps classify_error → "Turn starvation"). Also stops
                        # the loop-end estimate check from relabeling this
                        # break as token_budget.
                        stop_reason = "max_turns_unconfirmed_complete"
                        break
                    # Ask for confirmation
                    pending_completion = True
                    feedback = (
                        self._format_eval_feedback(best_trace)
                        if best_trace
                        else ""
                    )
                    # best_trace.score is guaranteed finite by the guard above,
                    # but defend in depth in case Trace defaults change.
                    if best_trace and math.isfinite(best_trace.score):
                        score = best_trace.score
                    else:
                        score = 0.0
                    messages.append(
                        {
                            "role": "user",
                            "content": COMPLETION_CONFIRMATION.format(
                                score=score,
                                eval_feedback_section=feedback,
                            ),
                        }
                    )
                    continue
            else:
                pending_completion = False

            # -- Handle parse errors (feed back to LLM) --
            lang_name = "bash" if self.solver_language == "bash" else "python"
            if parsed.parse_errors and not parsed.code.strip():
                error_msg = "; ".join(parsed.parse_errors)
                parse_failure_turns += 1  # S0.3: parse error, no actionable code
                logger.warning(
                    "Turn %d: parse error, no code: %s", turn, error_msg
                )
                messages.append(
                    {
                        "role": "user",
                        "content": PARSE_ERROR_FEEDBACK.format(
                            error=error_msg, language=lang_name,
                        ),
                    }
                )
                continue

            # -- If no code, ask again --
            if not parsed.code.strip():
                if was_confirmation and parsed.has_status and not parsed.task_complete:
                    # Prompt-invited decline of the completion check: the
                    # confirmation prompt itself offers "<status>working</status>"
                    # as the decline shape, so this is well-formed, not a parse
                    # failure — nudge for the code the confirmation asked for.
                    logger.info(
                        "Turn %d task=%s: completion declined without code; nudging",
                        turn, task.task_id,
                    )
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "You declined completion but did not include the "
                                "fixed code. Provide the complete corrected code "
                                "now in a <code> block, then "
                                "<status>working</status>."
                            ),
                        }
                    )
                    continue
                parse_failure_turns += 1  # S0.3: response had no <code> block
                preview = (parsed.raw_response or "")[:600].replace("\n", " | ")
                logger.warning(
                    "Turn %d task=%s: LLM response has analysis/plan but no <code> block. RAW[0:600]=%s",
                    turn, task.task_id, preview,
                )
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"INVALID: no <code> block found. Wasted turn.\n\n"
                            f"Begin your next message with `<code lang=\"{lang_name}\">` "
                            f"as the first characters. No preamble. No description. "
                            f"Just the code and a status tag:\n\n"
                            f'<code lang="{lang_name}">\n'
                            f"# Your complete solution — actual executable code, not a description\n"
                            f"</code>\n\n"
                            f"<status>working</status>"
                        ),
                    }
                )
                continue

            # -- Prepend library, execute, track best --
            trace = await _execute_and_track(parsed.code, turn, "")

            best_score_repr = (
                f"{best_trace.score:.4f}" if best_trace else "n/a"
            )
            logger.debug(
                "Turn %d: score=%.4f (best=%s)",
                turn,
                trace.score,
                best_score_repr,
            )

            # -- Perfect score early exit --
            # NaN >= 1.0 is False (correct); explicit isfinite makes intent clear.
            if math.isfinite(trace.score) and trace.score >= 1.0:
                if best_trace is not None:
                    _stamp_inner(best_trace)
                    best_trace.terminated_by = "perfect_score"  # audit #50: persist on the trace
                return _AgenticResult(
                    best_trace=best_trace,
                    reasoning_log="\n\n".join(reasoning_parts),
                    total_tokens=total_tokens,
                    turns_used=turn,
                    terminated_by="perfect_score",
                    parse_failure_turns=parse_failure_turns,
                    command_count=command_count,
                )

            # -- Normal continuation: show execution results --
            observation = self._build_observation(trace, turn)
            messages.append({"role": "user", "content": observation})

        # -- Loop ended (max_turns or token_budget) --
        if best_trace is None:
            # Fallback: no successful execution. Capture last attempted code.
            last_code = ""
            for m in reversed(messages):
                if m["role"] == "assistant":
                    p = self._parse_response(m["content"])
                    if p.code.strip():
                        last_code = p.code
                        break
            # command_count discriminates never-executed from executed-but-all-
            # non-finite; "not finite (nan)" keys classify_error to "Numeric
            # instability" when no structured cap label applies.
            best_trace = Trace(
                task_id=task.task_id,
                script=last_code,
                error_summary=(
                    "AgenticSolver produced no executable code"
                    if command_count == 0
                    else "AgenticSolver executed code but every run's score "
                    "was not finite (nan)"
                ),
            )

        # audit #49: an explicit stop_reason (e.g. post-retry LLM failure) takes
        # precedence over the token-estimate-derived cap label, so an infra fault
        # is not mislabeled as token_budget/max_turns.
        terminated = stop_reason or (
            "token_budget"
            if self._estimate_tokens(messages) > self.token_budget * 0.95
            else "max_turns"
        )
        if best_trace is not None:
            _stamp_inner(best_trace)
            best_trace.terminated_by = terminated  # 6.3: structured failure signal
        return _AgenticResult(
            best_trace=best_trace,
            reasoning_log="\n\n".join(reasoning_parts),
            total_tokens=total_tokens,
            turns_used=turns_completed,
            terminated_by=terminated,
            parse_failure_turns=parse_failure_turns,
            command_count=command_count,
        )

    # ---- Injection handling ----------------------------------------------

    def _run_all_pre_process(self, task: TaskDescription) -> str:
        """Run all pre_process blocks in reverse order (deepest first).

        This matches the MetaLayer chain behavior where the outermost layer
        (deepest depth) runs pre_process first and its output becomes
        outer_context for inner layers.

        Delegates to the shared :func:`~meta_n.core.meta_layer.run_pre_process`
        (the single implementation behind ``MetaLayer`` and ``InjectionMapper``)
        with ``outer_context=""``. That is semantically identical to the previous
        inline loop: seeding ``outer_context=""`` makes the shared function's
        ``next_outer`` accumulation track its returned ``combined`` exactly, so
        each block sees the running deepest-first accumulation as ``outer_context``
        and the returned context is that same accumulation.
        """
        _, combined = run_pre_process(
            self.injected_codes,
            task,
            additional_context="",
            outer_context="",
            thread_outer_context=not self.no_outer_context,
        )
        return combined

    def _prepend_library(self, script: str) -> str:
        """Prepend merged code library to script."""
        if self.solver_language == "bash":
            return prepend_bash_library(
                script, self.merged_py, self.merged_bash, self.executor
            )
        return prepend_python_library(script, self.merged_py, self.executor)

    def _format_library_descriptions(self) -> str:
        """Format library function signatures for the solver prompt."""
        if self.solver_language == "bash":
            return format_bash_library_descriptions(
                self.merged_py, self.merged_bash, self.executor
            )
        return format_python_library_descriptions(self.merged_py, self.executor)

    # ---- Response parsing ------------------------------------------------

    def _parse_response(self, raw: str) -> _ParsedResponse:
        """Parse structured response with fallback chain.

        Uses greedy matching for <code> to handle </code> appearing as a
        literal inside generated code (e.g. print("</code>")). The greedy
        regex captures everything up to the LAST </code> in the response.
        Short tags (<analysis>, <plan>, <status>) use non-greedy since their
        content is plain text without nested tag-like strings.
        """
        parsed = _ParsedResponse(raw_response=raw)

        # Level 1: XML tags (non-greedy for short text fields)
        for tag in ("analysis", "plan"):
            m = re.search(rf"<{tag}>(.*?)</{tag}>", raw, re.DOTALL)
            if m:
                setattr(parsed, tag, m.group(1).strip())

        # Status: non-greedy, short value
        status_m = re.search(r"<status>(.*?)</status>", raw, re.DOTALL)

        # Code: GREEDY match to handle </code> literals inside generated code
        code_m = re.search(
            r'<code(?:\s+lang="[^"]*")?>(.*)</code>', raw, re.DOTALL
        )
        if code_m:
            parsed.code = code_m.group(1).strip()

        # Level 1.5 fallback: <code> opener present but never closed (common
        # gemma failure — model writes the code block and runs out of tokens
        # or just forgets the closing tag). Take everything from the opener
        # to the next <status> tag or end of response.
        if not parsed.code:
            opener = re.search(r'<code(?:\s+lang="[^"]*")?>', raw)
            if opener:
                tail = raw[opener.end():]
                # Stop at <status> if present (it's a sibling tag, not nested)
                status_pos = tail.find("<status>")
                if status_pos >= 0:
                    tail = tail[:status_pos]
                # Strip leading/trailing whitespace and any stray fence markers
                candidate = tail.strip()
                # Remove leading/trailing ```python ... ``` if model wrapped
                # the code in fences inside the unclosed <code> block.
                candidate = re.sub(r"^```(?:python|bash|py)?\s*\n", "", candidate)
                candidate = re.sub(r"\n```\s*$", "", candidate)
                if candidate:
                    parsed.code = candidate
                    parsed.parse_errors.append(
                        "Recovered code from unclosed <code> tag"
                    )

        # Level 2 fallback: fenced code block. An empty matched block is still
        # ASSIGNED ("" stays falsy) so the Level-3 interplay is unchanged.
        if not parsed.code:
            fenced = extract_fenced_block(raw, ("python", "bash", "py", ""))
            if fenced is not None:
                parsed.code = fenced

        # Level 3 fallback: no tags at all → treat as code
        # Don't trigger if we at least parsed a status tag (e.g. confirmation turn)
        if not parsed.code and not parsed.analysis and not status_m:
            parsed.code = raw.strip()
            parsed.parse_errors.append(
                "No tags or code blocks found; treating entire response as code"
            )

        # Parse status
        parsed.has_status = status_m is not None
        parsed.task_complete = (
            status_m.group(1).strip().lower() == "complete"
            if status_m
            else False
        )

        return parsed

    # ---- Observation & context -------------------------------------------

    def _build_system_message(
        self, task: TaskDescription, context: str
    ) -> str:
        """Build the system/initial message for the agentic loop."""
        lang_instructions = get_language_instructions(self.solver_language, task.metadata)
        ctx_section = (
            f"## Additional Context\n{context}" if context else ""
        )
        lang_name = (
            "bash" if self.solver_language == "bash" else "python"
        )
        template = AGENTIC_SYSTEM_PROMPT
        # R2 (--agentic-preamble, default OFF): inject a short behavioral
        # preamble after the role line and before "## Task". Injected at render
        # time so the module constant stays verbatim; when OFF this branch is
        # skipped and the rendered prompt is byte-identical to HEAD. "## Task\n"
        # occurs exactly once in the template (the role line precedes it), so the
        # single-shot replace lands the block in the intended slot.
        if self.agentic_preamble:
            template = template.replace(
                "## Task\n", AGENTIC_PREAMBLE + "## Task\n", 1
            )
        return template.format(
            task_description=task.description,
            language_instructions=lang_instructions,
            context_section=ctx_section,
            language=lang_name,
        )

    def _build_observation(self, trace: Trace, turn: int) -> str:
        """Format execution results as an observation for the next turn."""
        feedback = self._format_eval_feedback(trace)
        template = OBSERVATION_TEMPLATE
        error_hints = ""
        # R1 (--agentic-error-hints, default OFF): for a FAILED trace of an
        # actionable class, prepend a "### Likely cause: <cls>\n<hint>" block
        # before the eval-feedback section. classify_error is reused inline
        # (failure_class is empty in-loop). When OFF — or for a non-actionable
        # class (Unknown/Runtime/Turn starvation/Environment fault → hint "") —
        # error_hints stays "" and the constant is used verbatim, so the
        # rendered observation is byte-identical to HEAD (no stray blank line).
        if self.agentic_error_hints and not trace.success:
            cls = classify_error(trace)
            hint = error_hint(cls)
            if hint:
                error_hints = f"### Likely cause: {cls}\n{hint}\n\n"
                template = template.replace(
                    "{eval_feedback_section}",
                    "{error_hints}{eval_feedback_section}",
                    1,
                )
        return template.format(
            turn=turn,
            max_turns=self.max_turns,
            score=trace.score,
            exit_code=trace.exit_code,
            duration_s=trace.duration_s,
            stdout=(
                _head_tail(trace.stdout, 500, 1500)
                if trace.stdout
                else "(empty)"
            ),
            stderr=(
                _head_tail(trace.stderr, 300, 700)
                if trace.stderr
                else "(empty)"
            ),
            error_hints=error_hints,
            eval_feedback_section=feedback,
        )

    def _format_eval_feedback(self, trace: Trace) -> str:
        """Format evaluation feedback, if available."""
        if trace.eval_feedback:
            return f"### Evaluation Details\n{_head_tail(trace.eval_feedback, 500, 1000)}"
        return ""

    async def _summarize_context(
        self, messages: list[dict[str, str]]
    ) -> tuple[list[dict[str, str]], int]:
        """Compress conversation history. Keep first message + last 2 turn-pairs.

        Returns (compressed_messages, tokens_used_for_summarization).

        Guard: needs at least 6 messages (system + 2 full turn-pairs + something
        in between to summarize). With fewer messages, summarization would either
        produce an empty summary or duplicate the system message.

        Re-fires on every turn while the context estimate stays above 85% of
        ``token_budget`` — deliberate and bounded by ``max_turns``: each
        re-fire folds the newest aged-out turn-pair into the (<=2000-token)
        summary, delaying the 95% hard stop. The irreducible tail is
        ``messages[0]`` plus the last 4 messages.

        This 2000-token call is the only outer LLM call below 8192 max_tokens,
        so under the F068 relative escalation cap the degenerate empty+length
        retry re-issues it at min(config cap, 4x) = 8000, not 32768.
        """
        if len(messages) < 6:
            return messages, 0
        first = messages[0]
        keep_n = min(4, len(messages) - 2)
        keep = messages[-keep_n:]
        # Non-empty by construction: len >= 6 ⇒ keep_n == 4 ⇒ slice has
        # len - 5 >= 1 elements.
        to_summarize = messages[1:-keep_n]
        conversation_text = "\n\n".join(
            f"[{m['role']}]\n{m['content']}" for m in to_summarize
        )
        summary, summ_tokens = await self.llm_client.complete(
            messages=[
                {
                    "role": "user",
                    "content": SUMMARIZE_PROMPT.format(
                        conversation=conversation_text
                    ),
                }
            ],
            temperature=0.3,
            max_tokens=2000,
        )
        return [
            first,
            {
                "role": "user",
                "content": f"## Prior work summary\n{summary}",
            },
            *keep,
        ], summ_tokens

    def _estimate_tokens(self, messages: list[dict[str, str]]) -> int:
        """Estimate total tokens in message list (chars/4 heuristic).

        This CONTEXT estimate is the ONLY input to the 85% (summarize) and
        95% (hard stop) ``token_budget`` gates — real API usage never feeds
        them. Cumulative real spend is capped separately by ``spend_budget``.
        """
        return sum(len(m.get("content", "")) for m in messages) // 4
