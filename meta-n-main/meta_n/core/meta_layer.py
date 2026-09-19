"""Core data models and MetaLayer execution logic for the Meta^n framework."""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from typing import TYPE_CHECKING, Optional, Protocol

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Re-exports (F257 split): meta_layer.py stays the import hub. Every name
# below was defined here before the split into the sibling modules
# adoption.py / code_library.py and is imported from meta_n.core.meta_layer
# across the repo — the import surface must not change.
# ---------------------------------------------------------------------------
from meta_n.core.adoption import (  # noqa: F401  (re-exported)
    _build_deploy_wrapper,
    _strip_library_prefix_for_scan,
    classify_error,
    populate_adoption_fields,
    scan_helper_calls,
)
from meta_n.core.code_library import (  # noqa: F401  (re-exported)
    SandboxMarker,
    _format_wired_skeleton_python,
    build_python_lib_file,
    format_bash_library_descriptions,
    format_python_library_descriptions,
    prepend_bash_library,
    prepend_python_library,
    validate_library_function,
    wrap_python_as_file,
)

if TYPE_CHECKING:
    from meta_n.core.base_executor import BaseExecutor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Truncation helpers (used by omega.py and meta_layer.py)
# ---------------------------------------------------------------------------

def _tail(text: str, limit: int) -> str:
    """Return the last *limit* chars of *text*, prefixed with '...' if truncated."""
    if len(text) <= limit:
        return text
    if limit <= 3:
        return text[-limit:]
    return "..." + text[-(limit - 3):]


def _head_tail(text: str, head: int, tail: int) -> str:
    """Return first *head* + last *tail* chars with a truncation marker in between."""
    if len(text) <= head + tail:
        return text
    return text[:head] + "\n...[truncated]...\n" + text[-tail:]


def detect_script_language(script: str) -> str:
    """``"python"`` iff the script starts with a def/import/from line, else ``"bash"``.

    SHARED CONTRACT: drives both the Ω prompt markdown-fence language
    (omega.py trace/inspiration formatters — golden-gated rendering) and the
    persisted ``.py`` / ``.sh`` extension for trace scripts
    (evolutionary_orchestrator.py). Any change here is prompt-affecting — do
    not extend the startswith tuple casually.
    """
    return (
        "python"
        if (script or "").lstrip().startswith(("def ", "import ", "from "))
        else "bash"
    )


class TaskDescription(BaseModel):
    """A single task to be solved."""

    task_id: str
    description: str
    verification_script: Optional[str] = None
    metadata: dict = Field(default_factory=dict)


class Trace(BaseModel):
    """Execution record from running a task.

    Not immutable: finalize sites stamp ``depth``/``reasoning``/``duration_s``,
    the adoption fields (:func:`populate_adoption_fields`), ``terminated_by``
    and ``failure_class`` in place after construction — do not assume a shared
    Trace is frozen.
    """

    task_id: str
    depth: int = 0
    script: str = ""
    stdout: str = ""
    stderr: str = ""
    exit_code: int = -1
    success: bool = False
    score: float = 0.0  # numeric score; scale is benchmark-defined (see BenchmarkAdapter.score_scale)
    reasoning: str = ""
    duration_s: float = 0.0
    error_summary: str = ""
    eval_feedback: str = ""  # detailed evaluator output (per-case mismatches, per-instance scores)
    failure_class: str = ""  # 6.3: unified failure class (turn-starvation / numeric / format / ...); "" if success or unclassified
    terminated_by: str = ""  # how an agentic/spine solve ended (confirmed / max_turns / max_turns_unconfirmed_complete / token_budget / token_budget_unconfirmed_complete / spend_budget / spend_budget_unconfirmed_complete / env_error / ...); "" for native single-shot
    # Inner-LLM accounting: tokens/calls consumed BY the executed script
    # itself (e.g., a classify solve() that calls llm() per case). The
    # outer-LLM tokens used to *generate* the script are tracked separately
    # via the (script, reasoning, tokens) tuple returned by SolverProtocol.
    # Prompt/completion split mirrors the OpenAI usage object so cost
    # analysis can separate input from output.
    inner_tokens: int = 0  # = inner_prompt_tokens + inner_completion_tokens
    inner_prompt_tokens: int = 0
    inner_completion_tokens: int = 0
    inner_calls: int = 0
    # --- Helper-adoption attribution (S0.2; mirrors AgentRunRecord 153-167) ---
    # Non-behavioral instrumentation. Populated ONLY at the fresh-trace finalize
    # sites and ONLY when injected helpers are LIVE (``utilities_available``
    # non-empty). The None-vs-[] distinction is load-bearing:
    #   * ``utilities_called is None``  → unmeasurable / no live helpers staged
    #     (e.g. the CO-Bench demoted path) — NOT a measured zero.
    #   * ``utilities_called == []``    → measured, none of the staged helpers
    #     were called (only when ``utilities_available`` is non-empty).
    # ``command_count`` (number of executor.execute invocations behind this
    # trace) is the measured-zero-vs-lost discriminator.
    utilities_available: list[str] = Field(default_factory=list)
    utilities_called: Optional[list[str]] = None
    utilities_call_counts: dict[str, int] = Field(default_factory=dict)
    command_count: int = 0
    # --- Agentic parse-failure accounting (S0.3) ---
    # Number of agentic turns wasted on a non-actionable parse failure
    # (phantom-completion rejection / parse error / no-code). 0 for native
    # single-shot solves; feeds the gated ``parse_failure_rate`` rollup.
    parse_failure_turns: int = 0

    @property
    def code_hash(self) -> str:
        return hashlib.sha256(self.script.encode()).hexdigest()[:16]


class InjectedCode(BaseModel):
    """Output of the Omega engine — code to inject into a layer."""

    pre_process: Optional[str] = None
    code_library: dict[str, str] = Field(default_factory=dict)       # Python functions
    code_library_bash: dict[str, str] = Field(default_factory=dict)  # Bash functions
    rationale: str = ""
    source_depth: int = 0
    # 4.2: per-task output channel. Maps task_id -> the FROZEN winning solution
    # (inline script) so a synthesized Ω_merge candidate routes each task to its
    # per-task-best winner without re-solving. Empty for ordinary Ω output;
    # additive, so legacy InjectedCode JSON (no key) round-trips unchanged.
    task_solution_map: dict[str, str] = Field(default_factory=dict)
    # T3.2: the subset of this layer's ``code_library`` helper names that ACTUALLY
    # ran in the --network none held-out sandbox (``ran_in_sandbox==True``) — i.e.
    # genuinely verified, not merely stub-kept. Stamped by the VERIFY-THEN-INJECT
    # gate so per-layer sandbox-verification survives into the merged chain and
    # across checkpoint/resume; the inline-re-derivation penalty bars per-task-best
    # only on these names. Sorted for deterministic JSON/checkpoint round-trip.
    # Additive with an empty default ⇒ legacy InjectedCode JSON round-trips and
    # ``is_empty`` is unaffected (same pattern as source_depth/task_solution_map).
    sandbox_verified_names: list[str] = Field(default_factory=list)
    raw_omega_prompt: str = ""    # full LLM prompt — paired with response below
    raw_omega_response: str = ""  # full LLM response for post-hoc analysis

    @property
    def is_empty(self) -> bool:
        return (self.pre_process is None
                and not self.code_library
                and not self.code_library_bash
                and not self.task_solution_map)


def merge_code_libraries(
    injected_codes: list[InjectedCode],
    *,
    executor=None,
) -> tuple[dict[str, str], dict[str, str]]:
    """Merge code libraries across layers. Returns (python_libs, bash_libs).

    Later layers override earlier by name -- **but only when the later helper
    passes the EXISTING ``validate_library_function``.** A helper that fails it
    does not participate in the override at all: the most recent previous VALID
    definition is kept, and when there is none the name is simply absent (and
    therefore never prepended).

    WHY (Meta^n helper-merging bug fix, 2026-09-17): plain ``dict.update()`` let
    a later, INVALID same-named helper silently destroy an earlier working one.
    The name stayed in the merged library but was rejected at prepend time, so
    nothing was injected and every caller raised NameError. Measured on CO-Bench
    / deepseek-flash: a depth-3 helper containing ``import sys`` (blocked by the
    sandbox) replaced a valid depth-2 helper of the same name -> 160/160
    instances raised NameError, score 0.000, gate FAILED, no lineage -- while
    the discarded depth-2 helper had been working.

    No new validity standard is introduced: the predicate is exactly
    ``validate_library_function``, the same one ``prepend_python_library`` uses.
    This is an execution-behaviour fix in Meta^n, NOT part of the RPBE method,
    and it applies uniformly to Full / Official / Predictive.
    """
    merged_py: dict[str, str] = {}
    merged_bash: dict[str, str] = {}
    for ic in injected_codes:
        for name, body in ic.code_library.items():
            if validate_library_function(name, body, executor=executor):
                merged_py[name] = body
                continue
            # Rejected: never let it shadow a working earlier definition.
            if name in merged_py:
                logger.warning(
                    "Library '%s' (source_depth=%s) FAILED validation and was "
                    "NOT merged; keeping the previous valid definition so its "
                    "callers do not hit a silent NameError",
                    name, ic.source_depth)
            else:
                logger.warning(
                    "Library '%s' (source_depth=%s) FAILED validation and was "
                    "NOT merged; no previous valid definition exists, so the "
                    "name will not be prepended",
                    name, ic.source_depth)
        # Bash libraries are not validated here (validate_library_function
        # returns True for is_bash), so the historical semantics are unchanged.
        merged_bash.update(ic.code_library_bash)
    return merged_py, merged_bash


# #48: wall-clock bound (seconds) for a single model-emitted pre_process exec.
# pre_process is prompt-steering text manipulation that should complete in
# milliseconds, so this generous default never affects a healthy block; it only
# caps a pathological one (``while True`` / hanging call / O(n^2) compute) so it
# cannot stall the whole concurrent run.
_PRE_PROCESS_TIMEOUT_DEFAULT = 30.0


def run_pre_process(
    injected_codes: list[InjectedCode],
    task: TaskDescription,
    additional_context: str = "",
    outer_context: str = "",
    *,
    pre_process_timeout: float = _PRE_PROCESS_TIMEOUT_DEFAULT,
    thread_outer_context: bool = True,
) -> tuple[bool, str]:
    """Run all ``pre_process`` blocks across layers and return ``(ran, context)``.

    This is the single, importable implementation of the pre_process pipeline,
    shared by :class:`MetaLayer`, :class:`~meta_n.core.agentic_solver.AgenticSolver`,
    and the external-agents ``InjectionMapper``. Behavior matches the original
    private methods exactly:

    - Each block is validated with :func:`meta_n.utils.safety.validate_code`;
      blocks that fail validation, raise, or return a non-``str``
      ``additional_context`` are skipped (logged at WARNING).
    - Blocks execute **deepest-first** (i.e. ``reversed(injected_codes)``),
      mirroring the MetaLayer chain where the outermost (deepest) layer runs
      first and its emission becomes ``outer_context`` for inner layers.
    - The exec namespace is seeded with ``task`` (a defensive deep copy — a
      block can never mutate the live task), ``additional_context`` (the
      per-block input, default ``""``), and ``outer_context`` (the *outer_context*
      argument for the first block, then the running accumulation for the rest).
    - The returned ``context`` is only the **accumulation of block emissions**;
      it does NOT include the *outer_context* argument itself. This preserves the
      original ``MetaLayer._run_pre_process`` contract (which returned just the
      block's own emission, with ``outer_context`` only seeded into the namespace)
      as well as ``AgenticSolver._run_all_pre_process`` (which seeded
      ``outer_context=""`` and returned the full accumulation).

    Args:
        injected_codes: The injected-code layers (any order; processed
            deepest-first). Layers without a ``pre_process`` are ignored.
        task: The task being solved; exposed as ``task`` in the namespace.
        additional_context: Seed value for the namespace ``additional_context``
            input on every block (the original methods always seeded ``""``).
        outer_context: Guidance from higher meta-layers; seeded into the first
            block's namespace ``outer_context`` (then superseded by the running
            accumulation), but NOT prepended to the returned context.
        thread_outer_context: When ``True`` (default), each block's emission is
            appended into the ``outer_context`` seen by subsequent (inner) blocks,
            so inner layers are conditioned on the running deepest-first
            accumulation. When ``False``, ``outer_context`` stays pinned at its seed
            value for every block (the intra-call inter-layer channel is ablated).
            The returned ``context`` (block-emission accumulation) is INDEPENDENT of
            this flag, so the solver prompt is byte-identical either way; only the
            per-block ``outer_context`` INPUT differs.

    Returns:
        ``(ran, context)`` — ``ran`` is ``True`` if at least one block produced
        non-empty context, and ``context`` is the accumulated emission.
    """
    from meta_n.utils.safety import validate_code

    # Defensive copy: blocks exec against a COPY of ``task`` so an abandoned
    # (timed-out) or contract-violating block can never mutate the live
    # TaskDescription the solver/executor keep using. Created ONCE so
    # cross-block visibility semantics are unchanged; compliant blocks only
    # READ ``task`` (the Ω prompt declares it read-only) and are unaffected.
    try:
        task_view = task.model_copy(deep=True)
    except Exception:  # non-deepcopyable metadata — degrade to shared reference
        logger.warning("pre_process: task deep-copy failed; sharing live task")
        task_view = task

    combined = ""
    next_outer = outer_context
    ran = False
    for ic in reversed(injected_codes):
        if not ic.pre_process:
            continue
        is_valid, error = validate_code(ic.pre_process)
        if not is_valid:
            # A failed pre_process disables ONLY this steering block. The layer's
            # code_library is a separate field and is merged independently
            # (see merge_code_libraries), so say so explicitly: the previous
            # wording ("skipping") was read as "the whole layer is gone".
            logger.warning(
                "pre_process (d=%d) failed validation: %s — skipping this "
                "pre_process ONLY; this layer's code_library (%d helper(s)) is "
                "unaffected and still merged",
                ic.source_depth, error, len(ic.code_library),
            )
            continue
        namespace = {
            "task": task_view,
            "additional_context": additional_context,
            "outer_context": next_outer,
        }
        # #48: bound the model-emitted exec with a wall-clock timeout. This exec
        # runs SYNCHRONOUSLY on the asyncio event loop (callers invoke
        # ``_run_pre_process`` without ``await``), and ``validate_code`` is a
        # static blocklist that cannot detect a runtime ``while True`` / sleep /
        # hanging call. Run it in a daemon worker thread and abandon the block if
        # it overruns the budget — mirroring the existing except-and-continue —
        # so one pathological pre_process cannot stall every concurrent task.
        _exc: dict[str, BaseException] = {}

        def _exec_block(_code: str = ic.pre_process, _ns: dict = namespace) -> None:
            try:
                exec(_code, _ns)  # noqa: S102
            except Exception as _e:  # noqa: BLE001
                _exc["e"] = _e

        _worker = threading.Thread(target=_exec_block, daemon=True)
        _worker.start()
        _worker.join(pre_process_timeout)
        if _worker.is_alive():
            logger.warning(
                "pre_process (d=%d) exceeded %.1fs wall-clock — skipping",
                ic.source_depth, pre_process_timeout,
            )
            continue
        if "e" in _exc:
            logger.warning(
                "pre_process (d=%d) raised %s — skipping", ic.source_depth, _exc["e"],
            )
            continue
        own = namespace.get("additional_context", "")
        if not isinstance(own, str):
            logger.warning(
                "pre_process (d=%d) set additional_context to %s, expected str — skipping",
                ic.source_depth, type(own).__name__,
            )
            continue
        if own:
            combined = f"{combined}\n{own}" if combined else own
            if thread_outer_context:
                next_outer = f"{next_outer}\n{own}" if next_outer else own
            ran = True
    return ran, combined


class LayerResult(BaseModel):
    """Result of running all tasks at a given depth."""

    depth: int
    traces: list[Trace] = Field(default_factory=list)
    pass_at_1: float = 0.0
    mean_score: float = 0.0  # mean of trace scores (= pass@1 for binary tasks)
    injected_code: Optional[InjectedCode] = None
    token_usage: int = 0


class SolverProtocol(Protocol):
    """Protocol for anything that can solve a task (Layer1Solver or MetaLayer)."""

    async def solve(
        self, task: TaskDescription, additional_context: str = ""
    ) -> tuple[str, str, int]: ...


#: Sentinel distinguishing an OMITTED ``merged_code_library`` kwarg (→ self-wire the
#: layer's own ``InjectedCode`` library, so an ad-hoc single-layer MetaLayer advertises
#: it instead of silently dropping it — T1.3) from an EXPLICIT ``None``, which the
#: orchestrator passes for every NON-outermost layer to mean "stage nothing here".
#: Explicit ``None``/``{}``/dict therefore behave EXACTLY as on HEAD (``value or {}``);
#: only a truly omitted kwarg self-wires.
_LIBRARY_UNSET: dict[str, str] = object()  # type: ignore[assignment]


class MetaLayer:
    """
    A meta-layer that wraps an inner solver and applies injected
    pre_process and code_library improvements.
    """

    def __init__(
        self,
        depth: int,
        injected_code: InjectedCode,
        inner_solver: SolverProtocol,
        executor: BaseExecutor,
        *,
        merged_code_library: dict[str, str] | None = _LIBRARY_UNSET,
        merged_code_library_bash: dict[str, str] | None = _LIBRARY_UNSET,
        max_retries: int = 0,
        retry_threshold: float = 0.5,
        solver_language: str = "python",
        no_outer_context: bool = False,
        foster_adoption: bool = False,
        deploy_verified_code: bool = False,
    ):
        self.depth = depth
        self.injected_code = injected_code
        self.inner_solver = inner_solver
        self.executor = executor
        # T1.3 (dual-field trap): every advertise/stage/deploy method reads
        # ``self.merged_code_library``; ``injected_code.code_library`` is read by
        # no instance method. Self-wire ONLY when the kwarg is OMITTED (sentinel),
        # so an ad-hoc single-layer MetaLayer advertises its OWN per-layer library
        # instead of silently dropping it. An EXPLICIT value — including the
        # ``None`` the orchestrator passes for non-outermost layers (meaning
        # "stage nothing here") and ``{}`` — reproduces HEAD's ``value or {}``
        # EXACTLY, so the orchestrator path (and the goldens) are byte-identical.
        self.merged_code_library = (
            dict(injected_code.code_library)
            if merged_code_library is _LIBRARY_UNSET
            else (merged_code_library or {})
        )
        self.merged_code_library_bash = (
            dict(injected_code.code_library_bash)
            if merged_code_library_bash is _LIBRARY_UNSET
            else (merged_code_library_bash or {})
        )
        self.max_retries = max_retries
        self.retry_threshold = retry_threshold
        self.solver_language = solver_language
        # E3 ablation: disables inter-layer conditioning by forcing
        # outer_context="" in pre_process; see solve() below.
        self.no_outer_context = no_outer_context
        # Mechanism-0 adoption affordance: when True, the helper-advertising
        # prose REQUIRES the solver to call injected helpers (+ wired skeleton)
        # instead of re-deriving them inline. Default False ⇒ byte-identical.
        self.foster_adoption = foster_adoption
        # P1a DEPLOY FALLBACK: when True AND a verified library is staged, an
        # authored solve() that does NOT call any staged helper (empty / inline
        # re-derivation) is REPLACED by a deterministic wrapper that calls the
        # helper. Default False ⇒ the short-circuit block is skipped entirely
        # (no scan, no string build) so the authored ``script`` object is the
        # identical object HEAD produces. See ``_maybe_deploy_verified_helper``.
        self.deploy_verified_code = deploy_verified_code

    def _maybe_deploy_verified_helper(self, script: str) -> str:
        """P1a deploy fallback: deterministically deploy a verified helper when
        the authored ``script`` did NOT adopt it.

        Returns ``script`` UNCHANGED unless ``deploy_verified_code`` is ON AND
        ``solver_language`` is ``"python"`` AND a non-empty merged Python
        library is staged — the wrapper below is Python ``def solve(**kw)``
        source, so a bash (or openevolve) layer must never receive it in place
        of its authored script. When ON: if the authored solve() already CALLS
        a staged helper by name (``scan_helper_calls`` on the library-stripped
        script finds ≥1), the model's own adoption is preferred and the script
        is returned untouched. Otherwise (empty solve OR inline re-derivation),
        the body is replaced by a deterministic wrapper
        ``def solve(**kw): return <helper>(**<matching kwargs>)`` that
        introspects the helper signature so a stray instance key never raises.

        Conservative on any failure: if the helper signature cannot be parsed,
        the authored script is returned unchanged (never deploy a broken
        wrapper). With multiple verified helpers, the pick prefers an
        ENTRY-POINT helper — one whose name is NOT called by any other
        stageable helper's source (per the intra-library call graph) — over a
        dependency utility, sorted-first within that entry-point set; when no
        entry point exists (e.g. a mutually-calling cycle) it falls back to the
        sorted-first stageable helper. Both the scan and the pick see only
        stageable helpers — a call to a never-prepended helper is a guaranteed
        NameError, not adoption.
        """
        if (
            not self.deploy_verified_code
            or self.solver_language != "python"
            or not self.merged_code_library
        ):
            return script
        # R2-CS-3: both the adoption scan and the deploy pick operate on the
        # STAGEABLE set — the names prepend_python_library will actually prepend
        # (validate_library_function, same executor, NOT for_advertising). A call
        # to a validation-failing helper is a guaranteed NameError (never
        # prepended, and scan_helper_calls' def-shadow rule proves the model did
        # not define it), so it must not count as adoption.
        stageable = [
            name for name in sorted(self.merged_code_library)
            if validate_library_function(
                name, self.merged_code_library[name], executor=self.executor
            )
        ]
        if not stageable:
            return script  # nothing was actually prepended — never deploy
        called, _ = scan_helper_calls(_strip_library_prefix_for_scan(script), stageable)
        if called:
            # Model adopted ≥1 staged verified helper — prefer its own authoring.
            return script
        # Non-adoption: deploy an entry-point stageable helper deterministically.
        # A stageable helper called by ANOTHER stageable helper's source is a
        # dependency (call-graph proxy for the orchestrator's true entry-point
        # set); the remaining top-level helpers are the entry-point candidates.
        called_by_sibling: set[str] = set()
        for owner in stageable:
            deps, _ = scan_helper_calls(self.merged_code_library[owner], stageable)
            called_by_sibling.update(d for d in deps if d != owner)
        entry_points = [n for n in stageable if n not in called_by_sibling]
        helper_name = entry_points[0] if entry_points else stageable[0]
        wrapper = _build_deploy_wrapper(helper_name, self.merged_code_library[helper_name])
        if wrapper is None:
            return script  # unparseable signature ⇒ never ship a broken wrapper
        logger.info(
            "MetaLayer d=%d deploy-verified-code: authored solve() did not call "
            "any verified helper; deploying deterministic wrapper for '%s'",
            self.depth, helper_name,
        )
        return wrapper

    async def _prepare_script(
        self, task: TaskDescription, additional_context: str = "", *,
        outermost: bool,
    ) -> tuple[str, str, int, str, bool]:
        """Prepare the executable script for one solve — the pipeline front half
        shared by :meth:`execute` (``outermost=True``) and :meth:`solve`
        (``outermost=False``): frozen Ω_merge routing → pre_process → library
        advertising → inner solve → deploy fallback → library prepend.

        Returns ``(script, reasoning, tokens, context_used, was_frozen)``.
        A frozen Ω_merge winner is returned VERBATIM (no deploy/prepend, 0
        tokens); ``context_used`` is the exact context the inner solver saw
        (execute() feeds it to ``_build_debug_context``).

        CONTRACT: the two context-assembly branches are transcribed verbatim
        from the historical execute()/solve() bodies — including their
        DIFFERENT lib-desc joins (execute: no leading newline on empty context;
        solve: leading ``"\\n"``) and execute()'s NO outer_context threading —
        so both prompt paths stay byte-identical (pinned by
        tests/test_refine_meta_layer.py::TestPrepareContextPins).
        """
        # --- 4.2: frozen-output routing (Ω_merge) ---
        # If this layer routes the task to a per-task-best winner, hand back the
        # FROZEN winner script directly and skip the inner solver — no LLM
        # re-solve, no prepend. Deterministic-eval families reproduce the
        # winner's score exactly; stochastic-solver output caching is a
        # follow-up.
        frozen = self.injected_code.task_solution_map.get(task.task_id)
        if frozen:
            return frozen, "frozen Ω_merge winner (no re-solve)", 0, additional_context, True

        if outermost:
            # execute() context assembly: pre_process WITHOUT outer_context
            # threading (top of the layer stack — there is no outer emission),
            # ``no_outer_context`` not consulted.
            context = ""
            if self.injected_code.pre_process:
                context = self._run_pre_process(task)
                if context:
                    logger.debug(
                        "MetaLayer d=%d pre_process for %s: added %d chars of context",
                        self.depth, task.task_id, len(context),
                    )
                else:
                    logger.debug(
                        "MetaLayer d=%d pre_process for %s: returned empty context",
                        self.depth, task.task_id,
                    )
            lib_desc = self._format_library_descriptions()
            if lib_desc:
                context = f"{context}\n{lib_desc}" if context else lib_desc
                logger.debug(
                    "MetaLayer d=%d library: %d py + %d bash functions available for %s",
                    self.depth, len(self.merged_code_library),
                    len(self.merged_code_library_bash), task.task_id,
                )
        else:
            # solve() context assembly: the incoming additional_context from the
            # outer layer threads into pre_process as outer_context (inter-layer
            # communication; E3 ablation ``no_outer_context`` forces it to "" —
            # the accumulated additional_context still flows to the inner solver
            # so the solver retains the strategy stack).
            context = additional_context
            if self.injected_code.pre_process:
                outer = "" if self.no_outer_context else additional_context
                own_context = self._run_pre_process(task, outer_context=outer)
                context = f"{additional_context}\n{own_context}" if additional_context else own_context
            # T3.1 library channel: without this, solve() drops advertise/stage/
            # deploy entirely (W5). Empty library ⇒ lib_desc == "" ⇒ context
            # unchanged ⇒ byte-identical (depth-1 default).
            lib_desc = self._format_library_descriptions()
            context = f"{context}\n{lib_desc}" if lib_desc else context

        # --- Inner solve + P1a deploy fallback + library prepend ---
        # (single-sourced tail; deploy is a no-op + identical object when the
        # flag is OFF, prepend of an empty library returns the input unchanged)
        script, reasoning, tokens = await self.inner_solver.solve(task, context)
        script = self._prepend_library(self._maybe_deploy_verified_helper(script))
        return script, reasoning, tokens, context, False

    async def execute(self, task: TaskDescription) -> tuple[Trace, int]:
        """
        Execute the full layer pipeline.

        Returns:
            Tuple of (Trace, tokens_used)
        """
        start = time.time()

        exec_script, reasoning, total_tokens, additional_context, was_frozen = (
            await self._prepare_script(task, outermost=True)
        )

        # --- 4.2: a frozen Ω_merge winner executes directly — no retry loop,
        # no adoption fields, 0 tokens. ---
        if was_frozen:
            trace = await self.executor.execute(exec_script, task)
            trace.depth = self.depth
            trace.reasoning = reasoning
            trace.duration_s = time.time() - start
            return trace, 0

        # --- Execute ---
        trace = await self.executor.execute(exec_script, task)
        trace.depth = self.depth
        trace.reasoning = reasoning
        trace.duration_s = time.time() - start
        exec_count = 1  # S0.2: count executor.execute invocations behind the trace

        # --- Self-debug retry loop ---
        if self.max_retries > 0 and trace.score < self.retry_threshold:
            initial_score = trace.score
            best_trace = trace
            retries_used = 0
            for retry in range(self.max_retries):
                retries_used += 1
                logger.info(
                    "MetaLayer d=%d self-debug retry %d/%d for %s "
                    "(score=%.3f < %.3f)",
                    self.depth, retry + 1, self.max_retries,
                    task.task_id, best_trace.score, self.retry_threshold,
                )
                debug_context = self._build_debug_context(
                    additional_context, best_trace, retry + 1,
                )
                retry_script, retry_reasoning, retry_tokens = (
                    await self.inner_solver.solve(task, debug_context)
                )
                total_tokens += retry_tokens

                retry_script = self._maybe_deploy_verified_helper(retry_script)
                retry_exec = self._prepend_library(retry_script)
                retry_trace = await self.executor.execute(retry_exec, task)
                exec_count += 1  # S0.2: each retry is another executor.execute
                retry_trace.depth = self.depth
                retry_trace.reasoning = retry_reasoning
                retry_trace.duration_s = time.time() - start

                if retry_trace.score > best_trace.score:
                    best_trace = retry_trace
                    logger.info(
                        "MetaLayer d=%d retry %d improved %s: %.3f → %.3f",
                        self.depth, retry + 1, task.task_id,
                        initial_score, best_trace.score,
                    )
                else:
                    logger.info(
                        "MetaLayer d=%d retry %d no improvement for %s: "
                        "retry=%.3f, best=%.3f",
                        self.depth, retry + 1, task.task_id,
                        retry_trace.score, best_trace.score,
                    )
                if best_trace.score >= self.retry_threshold:
                    break

            trace = best_trace
            logger.info(
                "MetaLayer d=%d self-debug summary for %s: "
                "%d retries, %.3f → %.3f (%s)",
                self.depth, task.task_id, retries_used,
                initial_score, trace.score,
                "improved" if trace.score > initial_score else "no change",
            )

        # --- S0.2: helper-adoption attribution (fresh-trace finalize site) ---
        # Populated AFTER the retry loop so a retried task reports the FINAL
        # winning trace's helper usage and command_count = 1 + retries (not the
        # overwritten pre-retry trace). Gated on live helpers inside the
        # populator: on the CO-Bench demoted path (merged Python library zeroed
        # upstream) ``utilities_called`` stays None. Strictly separate from the
        # behavioral foster_adoption affordance.
        populate_adoption_fields(
            trace,
            command_count=exec_count,
            merged_code_library=self.merged_code_library,
            merged_code_library_bash=self.merged_code_library_bash,
            executor=self.executor,
            solver_language=self.solver_language,
        )

        logger.debug(
            "MetaLayer d=%d execute %s: score=%.3f, success=%s, time=%.1fs",
            self.depth, task.task_id, trace.score, trace.success, trace.duration_s,
        )
        return trace, total_tokens

    async def solve(
        self, task: TaskDescription, additional_context: str = ""
    ) -> tuple[str, str, int]:
        """
        Solve interface — allows MetaLayer to be nested as inner_solver of another MetaLayer.
        Runs the full pipeline but returns (script, reasoning, tokens) instead of executing.

        The incoming additional_context from the outer layer is passed as outer_context
        to this layer's pre_process, enabling inter-layer communication.

        Delegates to :meth:`_prepare_script` (``outermost=False``) — the frozen
        Ω_merge routing, library advertise/stage, and deploy fallback are
        single-sourced there, symmetric with execute() (T3.1 / 4.2 / W5).
        """
        script, reasoning, tokens, _context, _was_frozen = await self._prepare_script(
            task, additional_context, outermost=False,
        )
        return script, reasoning, tokens

    def _run_pre_process(
        self, task: TaskDescription, outer_context: str = ""
    ) -> str:
        """Run pre_process code with safety validation. Returns additional context string.

        Thin wrapper over the module-level :func:`run_pre_process`, applied to
        this layer's single ``injected_code``. The returned string is the
        block's own emission (``outer_context`` is seeded into the namespace but
        not prepended to the result), preserving the original contract.

        Args:
            task: The task being solved.
            outer_context: Guidance from higher meta-layers (inter-layer communication).
                Pre_process code can read this to adapt its tactics to the higher
                layer's strategic decisions, enabling genuine k^n composition.
        """
        _, context = run_pre_process(
            [self.injected_code], task, outer_context=outer_context,
        )
        return context

    def _format_library_descriptions(self) -> str:
        if self.solver_language == "bash":
            return format_bash_library_descriptions(
                self.merged_code_library, self.merged_code_library_bash, self.executor,
                foster_adoption=self.foster_adoption,
            )
        return format_python_library_descriptions(
            self.merged_code_library, self.executor,
            foster_adoption=self.foster_adoption,
        )

    def _prepend_library(self, script: str) -> str:
        if self.solver_language == "bash":
            return prepend_bash_library(
                script, self.merged_code_library, self.merged_code_library_bash, self.executor,
            )
        return prepend_python_library(script, self.merged_code_library, self.executor)

    def _build_debug_context(
        self, base_context: str, failed_trace: Trace, attempt: int
    ) -> str:
        """Build additional_context for a self-debug retry attempt."""
        debug_section = (
            f"\n## Self-Debug Round {attempt}\n"
            f"Your previous solution scored {failed_trace.score:.3f} and "
            f"{'failed' if not failed_trace.success else 'underperformed'}.\n"
        )
        if failed_trace.stderr:
            debug_section += (
                f"\n### Error Output\n```\n{_tail(failed_trace.stderr, 1500)}\n```\n"
            )
        if failed_trace.error_summary:
            debug_section += (
                f"\n### Error Summary\n{failed_trace.error_summary}\n"
            )
        if failed_trace.stdout:
            debug_section += (
                f"\n### Standard Output\n```\n{_head_tail(failed_trace.stdout, 200, 300)}\n```\n"
            )
        if failed_trace.eval_feedback:
            debug_section += (
                f"\n### Evaluation Details\n{_tail(failed_trace.eval_feedback, 1500)}\n"
            )

        # Strip library prefix from script so solver sees only its own code
        prev_script = _strip_library_prefix_for_scan(failed_trace.script)

        lang = "bash" if self.solver_language == "bash" else "python"
        debug_section += (
            f"\n### Previous Code\n```{lang}\n{prev_script[:3000]}\n```\n"
            "\nPlease fix the issues and generate an improved solution. "
            "Focus on the error messages and ensure correctness."
        )
        return f"{base_context}\n{debug_section}" if base_context else debug_section
