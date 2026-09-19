"""Benchmark abstraction — common interface for all benchmarks.

Each benchmark adapter must implement BenchmarkAdapter, which converts
benchmark-specific data into Meta^n's TaskDescription format and provides
evaluation logic.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

from meta_n.core.meta_layer import TaskDescription, Trace


@dataclass
class EvalResult:
    """Result of evaluating a solution against a benchmark task.

    Supports both binary pass/fail (terminal tasks) and numeric scoring
    (optimization tasks).
    """

    success: bool
    score: float = 0.0  # normalized 0-1 where possible
    feedback: str = ""  # human-readable evaluation feedback
    raw_score: float = 0.0  # benchmark-native score (may not be normalized)
    # Inner-LLM accounting: tokens/calls consumed by the EVALUATED program
    # itself (e.g., a classify solve() that calls llm() per case). Outer-LLM
    # tokens (the orchestrator's own LLM usage) are tracked elsewhere by
    # whoever drove the eval. All zero for benchmarks whose programs never
    # call llm() (e.g., CO-Bench). The prompt/completion split mirrors the
    # OpenAI usage object so cost analysis can separate input from output.
    inner_tokens: int = 0  # = inner_prompt_tokens + inner_completion_tokens
    inner_prompt_tokens: int = 0
    inner_completion_tokens: int = 0
    inner_calls: int = 0
    # External-agent scoring flags (additive; defaults preserve legacy behavior).
    # `valid` marks whether the produced solution was well-formed / parseable;
    # `feasible` marks whether it satisfies the task's hard constraints (e.g. a
    # feasible CO-Bench schedule). Both default True so existing adapters that
    # never set them are byte-for-byte unaffected.
    valid: bool = True
    feasible: bool = True


class BenchmarkAdapter(ABC):
    """Common interface for benchmark integrations.

    Solution language is communicated per-task via
    ``task.metadata["solution_language"]``, stamped by each adapter's
    ``load_tasks`` and consumed by ``Layer1Solver.solve`` for prompt
    selection — deliberately the SINGLE channel. There is no adapter-level
    ``solution_language`` property (a former one drifted from the metadata
    channel and was removed); never wire a language property to main.py's
    ``solver_language``, which is a prompt-variant key (e.g. ``"openevolve"``
    for the OpenEvolve family), not a language.

    ``evaluate_test`` is an OPTIONAL adapter protocol method (same signature as
    :meth:`evaluate`) that the orchestrator DETECTS via
    ``hasattr(adapter, "evaluate_test")`` to run the held-out / reporting pass
    against a separate test split. It is implemented by the adapters that own a
    real test split (co_bench, text_classification, arc_agi, the OpenEvolve
    family) and absent on those that do not (terminal_bench, swe_bench). It is
    deliberately NOT declared here as an abstract or base method: doing so would
    flip the ``hasattr`` gate to always-True and silently enable the test pass
    for adapters that never implemented it. Adapters with a test split add it;
    the ABC stays silent.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Benchmark name (e.g., 'co_bench', 'terminal_bench')."""
        ...

    @abstractmethod
    def load_tasks(self, limit: int | None = None) -> list[TaskDescription]:
        """
        Load tasks from the benchmark.

        Args:
            limit: Optional cap on number of tasks (for piloting)

        Returns:
            List of TaskDescription objects ready for the orchestrator
        """
        ...

    @abstractmethod
    async def evaluate(self, task: TaskDescription, solution: str) -> EvalResult:
        """
        Evaluate a solution against a benchmark task.

        Args:
            task: The task being solved
            solution: The generated solution (code string)

        Returns:
            EvalResult with score and feedback
        """
        ...

    def score_scale(self) -> dict:
        """Descriptor of this benchmark's score scale (roadmap v2 N4).

        Lets scale-invariant selection, gating, and stopping normalize correctly
        across benchmarks. Keys:

        - ``kind``: ``"unit"`` ([0,1]), ``"binary"`` ({0,1} per task), or
          ``"continuous"`` (open-ended, e.g. AlgoTune speedups, SR fitness).
        - ``lo`` / ``hi``: score bounds (either may be ``None`` for continuous).
        - ``failure_sentinel``: a *finite* score the evaluator emits for a hard
          failure that must NOT be averaged as a real value (e.g. -1e9), or
          ``None``.

        Default is the unit scale, so every existing [0,1] / binary adapter is
        unchanged; only adapters with a genuinely different scale override this.
        """
        return {"kind": "unit", "lo": 0.0, "hi": 1.0, "failure_sentinel": None}

    def split_type(self) -> str:
        """How this benchmark's dev split relates to deployment (roadmap v2 N6).

        One of:
        - ``"dev_equals_test"``: deterministic evaluator, no held-out split
          (dev == test by construction).
        - ``"held_out"``: a genuine held-out test split exists; dev→test
          generalization is meaningful and overfit-protection applies.
          Describes SPLIT STRUCTURE only — in-loop machinery that re-solves
          the test split (``get_test_task``) must gate on that capability
          being present, not on this label (CO-Bench is held_out without it).
        - ``"proxy"``: dev is a *proxy* for the real objective (e.g. train-demo
          accuracy vs a hidden test), so a high dev score may not transfer —
          naive dev-ceiling protection is HARMFUL and must use a held-out-stable
          signal.
        - ``"none"``: no test adapter / not applicable.

        Default ``"none"``; gates the overfit cluster (3.3 / 6.2 / 6.4).
        """
        return "none"

    def code_library_is_live(self) -> bool:
        """Whether Ω's Python ``code_library`` helpers are actually CALLED by the
        solver on this benchmark (roadmap v2 4.3).

        Default ``True`` (current behavior, byte-identical). Benchmarks where the
        solver REGENERATES code inline rather than importing helpers — measured
        call-rate ~0 (CO-Bench, SWE-bench) — override to ``False`` so the dead
        Python-helper prepend is demoted (it adds tokens and can inject a buggy
        unused helper). Bash helpers and live-helper families (ARC / AlphaEvolve
        / SR, where matched helpers convert hard-0 crashes to high scores) keep
        the default. Set offline from the §3.5 call-rate telemetry, NOT a live
        auto-gate (determinism + no gen0 cold-start).
        """
        return True

    def advertises_spine_builtin(self) -> bool:
        """Whether ``base_solver='builtin'`` should route through the agent spine.

        Default ``False`` — the universal contract is that ``base_solver='builtin'``
        stays on the LEGACY native ``Layer1Solver`` / ``AgenticSolver`` /
        ``MetaLayer``-chain dispatch (so e.g. CO-Bench's ~0.546 builtin baseline is
        never regressed). The terminal_bench adapter overrides this to ``True``
        because its native ``TerminalBenchExecutor`` CANNOT run the legacy
        ``original-tasks`` layout: there, the only way to give ``builtin`` a REAL
        same-container control (so builtin / OpenHands / Terminus 2 all run the
        IDENTICAL tasks scored by the IDENTICAL verifier) is to route it through the
        spine + ``BuiltinTBBackend`` + the SAME terminal-bench harness the OH / T2
        runners use.

        The orchestrator consults THIS predicate (not ``make_agent_backend`` — which
        would require constructing a backend just to test support) to decide whether
        a ``builtin`` run goes through the spine or the legacy path. An adapter that
        returns ``True`` here MUST also make ``make_agent_backend('builtin')`` /
        ``make_env_provider('builtin')`` / ``make_scorer('builtin')`` non-``None``.

        Returns:
            ``True`` only if this adapter wants ``builtin`` on the spine path.
        """
        return False

    # ------------------------------------------------------------------ #
    # External-agent factory hooks (additive; see external_agents §2.6).  #
    # Adapters that support a self-contained external agent (OpenHands /  #
    # Terminus-2 / builtin) override these to supply the per-benchmark    #
    # backend, environment provider, and scorer that ExternalAgentSolver  #
    # wires together. All default to None so legacy adapters keep their   #
    # existing script-and-executor evaluation path untouched.            #
    # ------------------------------------------------------------------ #

    def make_agent_backend(self, kind: str, **kw):  # -> AgentBackend | None
        """Return an AgentBackend for the given solver `kind`, or None.

        Args:
            kind: External solver kind, e.g. ``"builtin"``, ``"openhands"``,
                or ``"terminus2"``.
            **kw: Backend-specific construction options (model, api_base, ...).

        Returns:
            An ``AgentBackend`` instance, or ``None`` if this adapter does not
            support the requested external solver kind.
        """
        return None

    def make_env_provider(self, kind: str):  # -> AgentEnvProvider | None
        """Return an AgentEnvProvider for the given solver `kind`, or None.

        Args:
            kind: External solver kind (e.g. ``"openhands"``, ``"terminus2"``).

        Returns:
            An ``AgentEnvProvider`` that provisions, stages, and extracts
            solutions for this benchmark, or ``None`` if unsupported.
        """
        return None

    def make_scorer(self, kind: str):  # -> Scorer | None
        """Return a Scorer for the given solver `kind`, or None.

        Args:
            kind: External solver kind (e.g. ``"openhands"``, ``"terminus2"``).

        Returns:
            A ``Scorer`` that grades extracted solutions into an
            ``EvalResult``, or ``None`` if unsupported.
        """
        return None

    def make_heldout_verifier(self):  # -> HeldoutVerifier | None
        """Forensic improvement #2 — optional held-out verifier for the
        ``--verified-code`` VERIFY-THEN-INJECT gate.

        Default ``None`` ⇒ the orchestrator falls back to
        :class:`meta_n.core.verified_code.StubHeldoutVerifier` (KEEP every helper
        but flag it UNVERIFIED — the DROP/verify gate is then INERT). The ONLY
        adapter that overrides this today is ``co_bench`` (crew-scheduling), which
        returns a :class:`meta_n.core.verified_code.SandboxedHeldoutVerifier` that
        sandbox-executes each Ω helper against a held-out check (``--network none``)
        and drops the ones that fail. Every other family (text_classification,
        openevolve, arc_agi, terminal_bench, swe_bench) inherits this ``None`` and
        thus the keep-all stub — ``--verified-code`` does NOT drop anything there
        (a real value-oracle bed would have to be wired first). Consulted ONLY when
        ``--verified-code`` is ON, so a ``None`` default is byte-identical for every
        legacy run.

        Returns:
            A ``HeldoutVerifier`` (duck-typed ``verify(name, source, task_id,
            context_sources) -> VerifyResult``), or ``None`` if this adapter has no
            held-out harness.
        """
        return None


class AdapterExecutor:
    """Executor that wraps ``adapter.evaluate()`` into a :class:`Trace`.

    Shared base for the benchmark families whose evaluation is a pure
    (task, script) → :class:`EvalResult` call (CO-Bench, text classification,
    the OpenEvolve trio, ARC-AGI-2). Subclasses customize two hooks:

    * ``_score_fmt`` — the format spec for the score/raw_score stdout line.
    * :meth:`_error_summary` — the failure summary placed in
      ``Trace.error_summary``.

    ``Trace.stdout`` / ``stderr`` / ``error_summary`` / ``eval_feedback`` feed
    Ω prompt rendering and summary.json, so hook overrides must preserve their
    family's exact bytes (pinned by tests/test_refine_benchmark.py goldens).

    The ``timeout`` parameter is accepted only for interface uniformity with
    ``BaseExecutor.execute``; per-benchmark timeouts (task.toml
    ``timeout_sec``, adapter-level ``self.timeout``) are enforced internally
    by each adapter and are authoritative.
    """

    _score_fmt: str = ".4f"

    def __init__(self, adapter: "BenchmarkAdapter"):
        self.adapter = adapter

    async def execute(
        self, script: str, task: TaskDescription, timeout: int = 30
    ) -> Trace:
        """Evaluate a solution via the adapter and return a Trace with scoring info."""
        start = time.time()

        result = await self.adapter.evaluate(task, script)

        return Trace(
            task_id=task.task_id,
            script=script,
            stdout=(
                f"score={result.score:{self._score_fmt}}"
                f"\nraw_score={result.raw_score:{self._score_fmt}}"
            ),
            stderr="" if result.success else result.feedback[:500],
            exit_code=0 if result.success else 1,
            success=result.success,
            score=result.score,
            error_summary=self._error_summary(result),
            eval_feedback=result.feedback,
            duration_s=time.time() - start,
            # Most evolved programs never call llm(), but propagate the
            # inner-LLM quad defensively so any adapter that populates it
            # keeps its cost accounting.
            inner_tokens=int(getattr(result, "inner_tokens", 0) or 0),
            inner_prompt_tokens=int(getattr(result, "inner_prompt_tokens", 0) or 0),
            inner_completion_tokens=int(getattr(result, "inner_completion_tokens", 0) or 0),
            inner_calls=int(getattr(result, "inner_calls", 0) or 0),
        )

    def _error_summary(self, result: EvalResult) -> str:
        """Failure summary for ``Trace.error_summary``; ``""`` on success."""
        return result.feedback[:200] if not result.success else ""
