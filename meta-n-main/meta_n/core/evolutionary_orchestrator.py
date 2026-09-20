"""Archive-based evolutionary orchestrator for Meta^n.

Replaces the linear MetaRecurseOrchestrator with an evolutionary search
over solver chains. Maintains a monotonically growing archive of candidates
(never prunes) and uses weighted parent selection with exploration bonus.

Inspired by OpenEvolve (quality-diversity) and DGM (stepping stones).
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from rich.console import Console

from meta_n.core.archive import Archive, Candidate
from meta_n.core.base_executor import BaseExecutor
from meta_n.core.llm_client import (
    LLMClient,
    _supports_request_seed,
    stable_crn_seed,
)
from meta_n.core.meta_layer import (
    InjectedCode,
    MetaLayer,
    TaskDescription,
    Trace,
    detect_script_language,
    merge_code_libraries,
    populate_adoption_fields,
)
from meta_n.core.omega import OmegaEngine
from meta_n.core.run_persistence import RunPersistence
from meta_n.core.self_repair import (
    SelfRepairEvent,
    classify_repair,
    write_self_repair_sidecars,
)
from meta_n.core.solver import Layer1Solver
from meta_n.core.spine_routing import uses_external_spine
from meta_n.core.verified_code import StubHeldoutVerifier

logger = logging.getLogger(__name__)
console = Console()

# Forensic improvement #3 — weight of the within-task depth*headroom selection
# term, live ONLY when ``within_task_recursion`` is ON (else the archive's
# ``within_task_depth_bonus`` stays 0.0 and the term is never added). Kept modest
# (mirrors ``novelty_alpha``'s 0.3) so the deep-chain bias does not crush breadth.
_WITHIN_TASK_DEPTH_BETA = 0.3


def _score_scale_hi(adapter) -> "float | None":
    """``adapter.score_scale()['hi']`` as a finite float, else ``None``.

    A bool / NaN / inf / missing ``hi``, a raising ``score_scale()`` (e.g. a
    unit-test mock), or ``adapter is None`` all yield ``None`` — callers treat
    that as "no usable ceiling". Shared by ``_archive_score_ceiling`` (the
    archive headroom ceiling) and ``_within_task_focus`` (the saturation
    release), so the two ceilings can never drift apart.
    """
    if adapter is None:
        return None
    try:
        hi = adapter.score_scale().get("hi")
    except Exception:
        return None
    if isinstance(hi, (int, float)) and not isinstance(hi, bool) and math.isfinite(hi):
        return float(hi)
    return None


def _config_drift(snap: dict, run_config: dict) -> dict:
    """Key-wise diff of a checkpoint's ``run_config_snapshot`` against the
    resumed process's ``run_config`` → ``{key: {"checkpoint": old,
    "resumed": new}}``, sorted by key; ``{}`` when drift-free.

    The volatile ``timestamp`` is excluded from the resumed side (the
    checkpoint side is stamped without it — see
    ``RunPersistence.save_checkpoint``). Keys missing on one side compare as
    ``None``, so an added/removed key (e.g. a new benchmark-features YAML
    entry) is reported as drift rather than skipped.
    """
    current = {k: v for k, v in run_config.items() if k != "timestamp"}
    return {
        k: {"checkpoint": snap.get(k), "resumed": current.get(k)}
        for k in sorted(set(snap) | set(current))
        if snap.get(k) != current.get(k)
    }


# Canonical home is meta_n.utils.atomic_io; re-exported here because the
# orchestrator is where every persisted-artifact writer lives.
from meta_n.utils.atomic_io import atomic_json_dump  # noqa: E402


@dataclass
class OrchestratorConfig:
    """Base run configuration shared by every orchestrator entry point."""

    epsilon: float = 0.02
    max_depth: int = 10
    output_dir: str = "./experiments"
    parallel: int = 1  # max concurrent tasks (1 = sequential)
    max_retries: int = 0  # self-debug retries per failing task (0 = disabled)
    retry_threshold: float = 0.5  # retry if score < this


@dataclass
class EvolutionaryConfig(OrchestratorConfig):
    """Configuration for the evolutionary orchestrator."""

    beam_width: int = 1  # B: parents selected per iteration
    beam_candidates: int = 1  # K: children per parent
    max_iterations: int = 50  # hard cap on number of iterations
    patience: int = 5  # stop after P non-improving iterations
    # 1.5a: magnitude-aware improvement margin. An iteration counts as improving
    # only if best (or oracle, N7) rises by more than epsilon * score_range — so
    # on continuous scales the bar scales up instead of any tiny delta resetting
    # patience. On [0,1] (range≈1) this reproduces "must beat by epsilon".
    epsilon: float = 0.02
    # 5.4: when True, never stop early on patience (run to max_iterations) — for
    # runs that intentionally set patience >= max_iterations.
    no_early_stop: bool = False
    gate_tasks: int = 3  # tasks for gate check (0 = skip)
    # 1.1: the per-task gate threshold is RELATIVE to the parent — a gate task
    # clears iff trace.success AND score >= parent_score - gate_margin (scale-safe
    # across [0,1] / continuous / binary). None disables thresholding (the old
    # liveness gate, for clean ablation). FAIL-OPEN when the parent lacks a
    # baseline for the task (gen0 children).
    gate_margin: float | None = 0.0
    # 6.2: per-task PROTECTION FLOOR. When set, the gate VETOES a candidate if
    # ANY tested task drops more than this far below its parent baseline — even
    # if another task clears. Fixes the E1 bug where the gate detected
    # capac_WH 0.198 < baseline 0.614 but admitted the candidate anyway because a
    # different task cleared first. None (default) ⇒ the legacy short-circuit
    # (pass on first clear) is byte-unchanged. (Moot in --consolidate mode, which
    # skips the gate — non-target tasks are inherited and cannot regress.)
    protect_floor: float | None = None
    # 1.6 / stochastic: median-over-R gate solves per task (1 = no extra cost).
    gate_repeats: int = 1
    # 1.2/1.3: median-over-R FULL-eval solves per task, to denoise each
    # candidate's per-task score so selection / best-promotion / merge act on
    # signal, not a single noisy draw (the binding constraint on stochastic-solver
    # benchmarks like CO-Bench). 1 = no change (the default).
    eval_repeats: int = 1
    # F035 (refine §6b): when ON and eval_repeats > 1, a GATE-precomputed trace
    # is treated as sample 0 and topped up with R-1 fresh solves (median over
    # the union), so gate-sampled tasks are denoised at the same R as every
    # other task instead of keeping a single (or median-of-gate_repeats) draw.
    # Under gate_repeats > 1 sample 0 is the gate's median-of-gate_R — an
    # accepted approximation. Consolidation inherit-frozen traces are EXEMPT
    # (deliberately never re-solved). Costs (R-1) extra full solves per
    # gate-sampled task. Fault fallback: a top-up solve that raises stops the
    # top-ups for that task and the score is the median over the samples
    # collected so far (at minimum the gate sample), with outer/inner tokens
    # summed over what actually ran — a fault never discards the measured
    # gate draw. Default OFF ⇒ the reuse short-circuit is byte-identical.
    eval_repeats_gate_topup: bool = False
    # CRN / paired eval. When True, every candidate's (task, repeat_index) full
    # solve runs under stable_crn_seed(config.seed, task_id, repeat_index) — a
    # seed IDENTICAL across all candidates in a run, so on a seed-honouring
    # backend the child-vs-parent comparison cancels shared LLM-sampler noise
    # (the EFFECTIVE comparison is paired even though stored scores stay
    # absolute). Composes with eval_repeats (different repeat_index ⇒ different
    # seed). NO-OP unless the backend honours a per-request seed (Azure/OpenAI
    # yes; LM Studio / OpenRouter-gemma no — capability-gated in the LLM client).
    # Default OFF ⇒ byte-identical create() payload (no seed on the wire).
    paired_eval: bool = False
    # G9: targeted per-task consolidation. When True, each candidate improves
    # exactly ONE target task (the highest-headroom one) while INHERITING every
    # other task's per-task-best frozen score (no re-solve), so improvement is
    # monotonic and collateral-free — fixing the "improve A, break B" thrash and
    # realizing the per-task oracle the linear chain otherwise discards (E1:
    # lead 0.704 vs oracle 0.751). Forces the deployable Ω_merge ungated so the
    # reported best converges to the oracle. Default OFF ⇒ existing behavior.
    consolidate: bool = False
    temperatures: list[float] = field(default_factory=lambda: [0.5, 0.7, 0.9])
    novelty_alpha: float = 0.3  # exploration bonus weight in parent selection
    use_inspiration: bool = True
    # Refine §6b F006 — rotate the reserved-elite window by generation so
    # per-task winners beyond the first n-1 are not permanently starved by
    # task-id order. Default OFF = byte-identical parent reservation.
    elite_rotation: bool = False
    # Ablation flags
    no_code_library: bool = False  # E2: strip solver_lib from Omega prompt + clear parsed code_library
    no_outer_context: bool = False  # E3: force outer_context="" in MetaLayer.solve()
    foster_adoption: bool = False  # mechanism-0 adoption affordance (default OFF = byte-identical)
    # Adoption probe: force the adapter's code_library_is_live() to be treated as
    # True (un-demote), so Ω's Python helpers are staged + callable even on
    # families that normally demote them (CO-Bench/SWE, where the solver
    # regenerates code inline). Lets the foster_adoption HELP/HURT test reach
    # those families. Default OFF ⇒ byte-identical (CO-Bench still demotes).
    force_code_library_live: bool = False
    # Forensic improvement #2 — VERIFIED code_library + FORCED ADOPTION. When ON:
    # (A) VERIFY-THEN-INJECT — after Ω produces a code_library helper, sandbox-
    # execute it (--network none) against a held-out check and KEEP it in the
    # injection ONLY if it passes; else DROP it (dead/wrong helpers never reach a
    # candidate). The verifier comes from the adapter's make_heldout_verifier()
    # (a SandboxedHeldoutVerifier on families with a value-oracle), else a
    # capability-preserving StubHeldoutVerifier (KEEP + flag UNVERIFIED). (B)
    # FORCED ADOPTION — the MetaLayer foster affordance is forced ON (the solver
    # MUST call the verified helper), and a non-adoption penalty bars a trace that
    # re-derived the helper inline from becoming per-task-best (composes with the
    # #1 regression guard). Default OFF ⇒ no sandbox runs, no helper dropped, no
    # foster text added, no field perturbs serialization = byte-identical.
    verified_code: bool = False
    # DEPLOY FALLBACK (composes with verified_code). When ON, the outermost
    # MetaLayer post-processes the authored solve(): if no verified helper is
    # CALLED (empty solve or inline re-derivation), the body is replaced by a
    # deterministic wrapper that calls the helper. Honest semantics: this
    # DEPLOYS the verified helper (score == helper score on the focus task); it
    # is the deploy fallback, NOT a measurement of natural adoption. Default OFF
    # ⇒ no post-process, the authored script is byte-identical. INERT unless a
    # live (non-zeroed) merged Python library is staged (force_code_library_live
    # on CO-Bench).
    deploy_verified_code: bool = False
    # Forensic improvement #1 — REGRESSION GUARD + median-of-R FOCUS-DENOISE.
    # When ON: (A) the deployable per-task-best may never drop below the BASE
    # (seed-only, no-Ω) resample best for that task — per_task_best := max(Ω, base
    # floor), so a catastrophic Ω regression (e.g. crew_scheduling Ω 0.100 vs base
    # 0.658) is never shipped; (B) the consolidate FOCUS pick selects on a
    # median-of-R DENOISED base score (true highest-headroom) instead of a single
    # noisy seed draw, so a degenerate one-draw 0.0 no longer mis-picks a healthy
    # task as the catastrophic focus. ``regression_guard_repeats`` (R) is consulted
    # ONLY when the bool is ON, so its presence is byte-neutral when OFF. Default
    # OFF ⇒ no seed resampling, no floor, pure round-robin focus = byte-identical.
    regression_guard: bool = False
    regression_guard_repeats: int = 3
    # F034 (refine §6b): MODIFIER on regression_guard+consolidate. When ON, the
    # consolidate FOCUS ranking re-ranks a task that has GENUINELY improved
    # beyond its best-of-R base floor at its CURRENT per-task best, so a task
    # improved to the ceiling rotates out of focus instead of being targeted
    # forever on its gen0-frozen base score. Tasks with no improvement beyond
    # the floor keep the median-of-R denoised base ranking (the gen0 pick is
    # unchanged even when ON). Default OFF ⇒ the frozen-base ranking (and the
    # round-robin path) is byte-identical.
    focus_current_headroom: bool = False
    # P1c SEED CODE LIBRARY — a {name: source} mapping (already parsed +
    # safety-validated at load time) injected into the gen0 candidate's
    # code_library BEFORE Ω runs, so a known-good helper can be tested WITHOUT Ω
    # authoring it. Default None ⇒ gen0 keeps injected_codes=[] and routes the
    # bare Layer1Solver (byte-identical baseline). When set, gen0 becomes a
    # depth-2 candidate with the seeded library and routes through the candidate
    # solver builder — a deliberate change to the baseline, gated OFF by default.
    # T3.3 (audit) — on a DEMOTING family (CO-Bench/SWE, where
    # code_library_is_live() is False) the merged Python library is ZEROED at
    # staging, so historically the seeded helper was silently dropped (never
    # prepended or callable) unless --force-code-library-live was also set. The
    # seed (source_depth==0) is now EXEMPTED from that zeroing (_demote_python_library)
    # so a seeded known-good helper survives and is staged even without the force
    # flag; --force-code-library-live additionally keeps the REST of the live-helper
    # channel (Ω-authored / verified helpers at depth>=2) live. For the original
    # P1b semantics use {--seed-code-library + --force-code-library-live}.
    seed_code_library: dict[str, str] | None = None
    seed: int = 42
    # Agentic solver config (Terminus 2-inspired iterative loop)
    use_agentic: bool = False  # toggle between single-shot and agentic solver
    agentic_max_turns: int = 5  # max turns per task in agentic loop
    # Per-task context-size bound (chars//4 estimate of the message list per
    # call) for the agentic loop — NOT a spend cap; see agentic_spend_budget.
    agentic_token_budget: int = 100_000
    # F063 (refine §6b): cumulative REAL-spend token cap (outer LLM + inner
    # llm()/llm_batch()) for the builtin AgenticSolver; None = off.
    # agentic_token_budget is a chars//4 CONTEXT bound, not spend.
    agentic_spend_budget: int | None = None
    # F060 (refine §6b): solve-time sampling temperature for the builtin
    # AgenticSolver; default 0.7 == the historical constructor pin, so unset ==
    # byte-identical. Deliberately independent of ``temperatures`` (the
    # Ω-generation cycle) and ``LLMConfig.temperature``.
    agentic_temperature: float = 0.7
    # Stage 1 ORTHOGONAL base-agent floor-raisers (builtin AgenticSolver only).
    # Both default OFF ⇒ the rendered observation (R1) and system prompt (R2)
    # are byte-identical to HEAD. When measured, apply UNIFORMLY across baseline
    # + Ω arms and re-baseline — these are base-agent engineering, NOT Ω/thesis
    # evidence. NOT threaded into the external OH/T2 spine (InjectionMapper path).
    agentic_error_hints: bool = False  # R1: error-hint taxonomy in observation
    agentic_preamble: bool = False  # R2: behavioral preamble in system prompt
    # Stage 2 WITHIN-LAYER REFINE (the shelved "self-refinement idea"). When a
    # freshly-bred child's injection FAILS the quality gate, make ONE extra Ω
    # call to FIX THE BUG in that same injection (error-correction — keep the
    # approach, fix the code) and keep the refined result as a NEW candidate iff
    # it now clears the gate (monotonic — the rejected attempt is never mutated).
    # First emitter of the SelfRepairEvent provenance record. Default OFF ⇒ the
    # gate-fail branch is byte-identical to HEAD (the refine block is skipped and
    # NO extra Ω call / candidate / sidecar is produced). NEVER fires on
    # gen0 / depth-1 / empty injections (those never reach the gate-fail branch)
    # or in --consolidate mode (every child carries a focus task, so the gate —
    # the hook's only entry — is skipped; pair only with a gated profile).
    within_layer_refine: bool = False
    # Stage 3 DOWNWARD RE-PROPAGATION (the one genuinely thesis-relevant feature —
    # makes "a layer reasons about a layer" literally true). When a depth>=3 child
    # PLATEAUS / regresses vs its parent, regenerate an INTERMEDIATE layer's
    # injection (d_t in [2, depth-1]) from the full-chain failure traces + the
    # above-layer injections as DOWNSTREAM FEEDBACK, REPLACE that layer in a NEW
    # monotonic candidate (parent_id = the revised child; the old child is
    # retained), re-evaluate, and log + classify a `downstream` SelfRepairEvent.
    # FEAL-bounded: the knowledge-bound same-model generator makes this
    # error-correction, NOT novel generation (the classifier records which).
    # Default OFF ⇒ the post-archive.add path is byte-identical to HEAD (no extra
    # Ω call / candidate / sidecar). NEVER touches gen0 / depth-1 (d_t >= 2).
    repropagation: bool = False
    # Forensic improvement #3 — WITHIN-TASK RECURSION. A MODIFIER scoped to
    # ``consolidate`` mode (focus_task only exists in consolidate mode). Stage 3
    # ``repropagation`` is INTENTIONALLY scoped to NON-consolidate breeding (the
    # ``and not focus_task`` guard on the reprop hook): repropagation re-evaluates
    # the FULL chain via ``_evaluate_candidate``, which is incompatible with
    # consolidate's inherit-frozen per-task map, so with ``--consolidate`` ON every
    # candidate has a non-None focus_task and repropagation is SUPPRESSED — the two
    # flags do NOT co-execute in a consolidate run. When ON it flips three knobs so
    # depth becomes genuine within-task RECURSION instead of router BREADTH: (a) the Ω FOCUS
    # directive becomes a DEEPEN directive (omega.py) so a deeper layer RE-WORKS
    # the SAME task its parent worked rather than gating a disjoint task; (b) the
    # focus pick INHERITS the parent's fresh task (``_within_task_focus``) so the
    # same task persists down the chain (``trace.depth`` compounds, multiplicity
    # >= 2); (c) the archive gains a depth*headroom selection term
    # (``within_task_depth_bonus``) that favors extending a deep high-headroom
    # chain — gated on ``consolidate`` too (this is a modifier scoped to
    # consolidate mode), so all three knobs engage together or not at all. The
    # depth-probe (analysis/depth_attribution.py) is the verifier: ON
    # lowers router_fraction / flips the within-task verdict to PASS vs the
    # consolidate router-stacking. Default OFF ⇒ FOCUS-freeze on, round-robin
    # focus, no depth term = byte-identical to HEAD. UNMEASURED build-only (no
    # viable live bed; the synthetic depth-probe flip is the only validation).
    # NEVER touches gen0 / depth-1 (focus_task is None there).
    within_task_recursion: bool = False
    # External-agent base solver (plan §2.3, §8.1). When set to "openhands" or
    # "terminus2" the per-candidate solver is an ExternalAgentSolver driving a
    # self-contained external agent (built via the adapter's factory hooks +
    # the shared run/cost guards). ``None`` (default) and ``"builtin"`` keep the
    # legacy native Layer1Solver / AgenticSolver / MetaLayer-chain path
    # byte-for-byte unchanged.
    #
    # SAME-CONTAINER BUILTIN CONTROL (terminal_bench) — SHIPPED. Routing
    # ``base_solver="builtin"`` on terminal_bench through the spine gives the
    # real same-container A/B control for OH vs T2, and it IS wired: the
    # terminal_bench adapter advertises it (``advertises_spine_builtin() ->
    # True``), so ``_uses_external_spine()`` routes builtin through the spine +
    # ``BuiltinTBBackend`` (a subprocess bridge to ``scripts/builtin_tb_runner.py``,
    # which authors the script via the native ``Layer1Solver.solve`` WITHOUT
    # executing locally, then provisions the TB Harness container, runs the
    # script in its TmuxSession, runs the verifier, and emits the shared
    # t2-style result JSON) + the SAME TB env provider / scorer as OH/T2. This
    # is the documented exception that reports native OUTER tokens
    # (``token_basis="outer"``). Every NON-advertising adapter (CO-Bench,
    # text-classification, …) inherits ``advertises_spine_builtin() -> False``,
    # so ``builtin`` there stays on the LEGACY native ``Layer1Solver`` /
    # ``AgenticSolver`` / ``MetaLayer``-chain path, byte-for-byte unchanged (so
    # e.g. CO-Bench's builtin baseline is never regressed). OH-vs-T2-vs-builtin
    # on terminal_bench is fully wired and validated.
    base_solver: str | None = None
    # Per-run soft wall-clock envelope (seconds) for an external-agent run; also
    # the hard ``asyncio.wait_for`` timeout inside ExternalAgentSolver (§6.5a).
    agentic_time_limit_s: int = 1200
    # Per-run hard USD budget forwarded to the external agent and used by the
    # per-candidate budget pre-check (§4.7, §6.6). Lowered 2.0 -> 0.5: with the
    # matched token/turn/wall envelope a single local-model run consumes far less
    # than $2, and a tighter ceiling keeps a priced-backbone A/B from over-spending
    # per task. CLI ``--agent-max-budget`` mirrors this default.
    agentic_max_budget_usd: float = 0.5
    # Inner Docker concurrency cap for external-agent runs (§6.4). ``None`` →
    # ``min(parallel, 4)``; always clamped to ``<= parallel``.
    max_docker: int | None = None
    # Optional host scratch root for per-run bind-mount dirs (§6.3). ``None`` →
    # the system temp dir.
    scratch_root: str | None = None


@dataclass
class EvolutionaryResult:
    """Final result of the evolutionary orchestrator."""

    archive_size: int = 0
    total_iterations: int = 0
    total_tokens: int = 0  # outer-LLM (Omega + solver code generation)
    # Outer-LLM input/output split, captured at end-of-run from
    # LLMClient.cumulative_usage (non-zero only when an LLMClient is wired).
    outer_prompt_tokens: int = 0
    outer_completion_tokens: int = 0
    outer_calls: int = 0
    inner_tokens: int = 0  # inner-LLM (script's own llm() calls)
    inner_prompt_tokens: int = 0
    inner_completion_tokens: int = 0
    inner_calls: int = 0
    best_mean_score: float = 0.0
    best_candidate_id: str = ""
    per_task_best_scores: dict[str, float] = field(default_factory=dict)
    oracle_mean_score: float = 0.0
    test_scores: dict[str, float] = field(default_factory=dict)
    test_mean_score: float = 0.0
    chain_test_scores: dict[str, float] = field(default_factory=dict)
    chain_test_mean_score: float = 0.0
    convergence_history: list[float] = field(default_factory=list)
    # N7: per-iteration oracle (per-task-best mean) trajectory — sibling of
    # convergence_history (which stays best-mean). Written as oracle_convergence.json;
    # NOT added to to_dict (summary.json headline keys stay frozen).
    # F033 (§6b) invariant, on EVERY exit path (max_iterations / patience /
    # budget / max-depth):
    #   len(convergence_history) == len(oracle_history) + 1
    #                            == completed_generations + 1
    # (the +1 is the gen0 seed entry, which appends convergence only). The
    # budget/max-depth breaks deliberately append NOTHING (see run()).
    oracle_history: list[float] = field(default_factory=list)
    archive_data: dict = field(default_factory=dict)
    # Optional run-level external-agent telemetry rollup (plan §7.8). ``None``
    # for legacy / builtin runs; populated from the archive's per-candidate
    # ``agent_telemetry`` blocks when an external base_solver was used.
    agent_telemetry_rollup: dict | None = None
    # 4.1: id of the synthesized Ω_merge deployable-oracle candidate, when one
    # was built (property-gated). Additive — surfaced in to_dict only when set,
    # so legacy summary.json stays byte-for-byte the same.
    merge_candidate_id: str | None = None
    # N9a: "completed" | "aborted_pre_iteration" | "aborted_mid" — lets a
    # cross-seed aggregator skip budget-starved stub runs. Recomputed at
    # end-of-run, so a resumed-then-completed run flips back to "completed".
    run_status: str = "completed"

    def absorb_candidate_tokens(self, cand: Candidate) -> None:
        self.total_tokens += cand.total_tokens
        self.inner_tokens += cand.inner_tokens
        self.inner_prompt_tokens += cand.inner_prompt_tokens
        self.inner_completion_tokens += cand.inner_completion_tokens
        self.inner_calls += cand.inner_calls

    def to_dict(self) -> dict:
        out = {
            "archive_size": self.archive_size,
            "total_iterations": self.total_iterations,
            "total_tokens": self.total_tokens,
            "token_usage": {
                "outer_total": self.total_tokens,
                "outer_prompt": self.outer_prompt_tokens,
                "outer_completion": self.outer_completion_tokens,
                "outer_calls": self.outer_calls,
                "inner_total": self.inner_tokens,
                "inner_prompt": self.inner_prompt_tokens,
                "inner_completion": self.inner_completion_tokens,
                "inner_calls": self.inner_calls,
            },
            "best_mean_score": self.best_mean_score,
            "best_candidate_id": self.best_candidate_id,
            "oracle_mean_score": self.oracle_mean_score,
            "test_mean_score": self.test_mean_score,
            "chain_test_mean_score": self.chain_test_mean_score,
            "per_task_best_scores": self.per_task_best_scores,
            "convergence_history": self.convergence_history,
            "run_status": self.run_status,
        }
        # Additive: only surface the rollup when an external agent produced one,
        # so legacy summary.json stays byte-for-byte the same (plan §7.8).
        if self.agent_telemetry_rollup is not None:
            out["agent_telemetry_rollup"] = self.agent_telemetry_rollup
        if self.merge_candidate_id is not None:
            out["merge_candidate_id"] = self.merge_candidate_id
        return out


class EvolutionaryOrchestrator:
    """Archive-based evolutionary search over solver chains."""

    def __init__(
        self,
        llm_client: LLMClient,
        executor: BaseExecutor,
        omega: OmegaEngine,
        config: EvolutionaryConfig,
        solver_language: str = "bash",
        balanced_json_fallback: bool = False,
    ):
        self.llm_client = llm_client
        self.executor = executor
        self.omega = omega
        self.config = config
        # F075 (§6b): classify-path JSON extraction knob, forwarded to the
        # Layer1Solver (fenced -> flat regex -> [ON only] balanced scan -> raw).
        # Base-config provenance (classify_balanced_json_fallback), NOT an
        # EvolutionaryConfig field — both orchestrator paths consume it.
        self.solver = Layer1Solver(
            llm_client, language=solver_language,
            balanced_json_fallback=balanced_json_fallback,
        )
        self.solver_language = solver_language
        self.archive = Archive(**self._archive_kwargs())
        # Forensic improvement #1 — median-of-R DENOISED base scores per task,
        # populated at gen0 ONLY when regression_guard is ON and consumed by the
        # consolidate FOCUS pick (_consolidation_targets). Empty otherwise, so the
        # round-robin focus path is byte-identical when the guard is OFF.
        self._base_focus_scores: dict[str, float] = {}
        self.rng = random.Random(config.seed)
        # SRI formal instrumentation. DISABLED by default (a plain attribute, not
        # a config field, so a non-formal run is byte-identical): every hook is a
        # no-op until the formal runner assigns a real SRIHooks object.
        from meta_n.sri.hooks import SRIHooks
        self.sri = SRIHooks.disabled()
        # Resume-provenance stashes, read getattr-style by RunPersistence:
        # run() sets _run_config_snapshot (checkpoint stamping); the resume
        # branch fills _resume_config_drift when the current config diverges
        # from the checkpoint's snapshot (recorded in config.json).
        self._run_config_snapshot: dict | None = None
        self._resume_config_drift: dict = {}
        # F253: persistence collaborator (checkpoint / summary / rollup writers).
        # Back-reference seam — see meta_n/core/run_persistence.py. The public
        # method surface on the orchestrator is preserved via one-line delegates.
        self._persistence = RunPersistence(self)
        # 5.4: right-size patience so the stop rule can actually fire. If a user
        # sets patience >= max_iterations the loop would never stop early; clamp
        # to max_iterations-1 (warn; keep config.patience for serialization).
        # no_early_stop opts out (run to max_iterations regardless).
        if config.no_early_stop:
            self._effective_patience = config.max_iterations + 1
        else:
            self._effective_patience = min(config.patience, max(1, config.max_iterations - 1))
            if self._effective_patience < config.patience:
                logger.warning(
                    "patience=%d >= max_iterations=%d: clamping effective patience to %d "
                    "so the stop rule can fire (set no_early_stop to disable stopping)",
                    config.patience, config.max_iterations, self._effective_patience,
                )

        # The benchmark adapter wrapped by the executor; the same adapter
        # supplies the external-agent factory hooks (make_agent_backend /
        # make_env_provider / make_scorer) and the inner provider resolution
        # (plan §2.6). ``None`` for executors that wrap no adapter (unit tests).
        self.adapter = getattr(self.executor, "adapter", None)

        # External-agent spine collaborators (plan §6.7). These are inert for
        # the legacy path: the run guard's semaphore is never leased, the cost
        # guard is never consulted, and the telemetry tree is created but stays
        # empty unless an ExternalAgentSolver writes to it. They are constructed
        # only when an external base_solver is requested so the legacy path
        # never pays the import / disk cost (and never requires a cost tracker).
        self._run_guard = None
        self._cost_guard = None
        self._agent_telemetry = None
        # Spine init: external kinds always; ``builtin`` only when the bound adapter
        # advertises it (terminal_bench). See ``_uses_external_spine``. The adapter
        # is read off the executor above, so it is available here.
        if self._uses_external_spine():
            self._init_external_agent_spine()
        # Thread our seeded RNG into Omega's trace sampler so `--seed=N`
        # produces identical Omega prompts across runs (and survives
        # --resume via the checkpointed RNG state — `random.Random.setstate`
        # mutates this same object, so the ContextManager's reference stays
        # in sync). Without this, ContextManager fell back to the
        # module-level `random` which is unseeded.
        self.omega.context_manager.rng = self.rng

        if not config.temperatures:
            raise ValueError("temperatures list must not be empty")
        # F063 (§6b): a zero/negative spend cap would stop every agentic loop
        # before its first turn — reject at construction, mirroring the
        # temperatures check above. None (default) = the cap is off.
        if config.agentic_spend_budget is not None and config.agentic_spend_budget <= 0:
            raise ValueError("agentic_spend_budget must be positive when set")

        # T2.1 (audit) — --paired-eval/CRN is a STRUCTURAL no-op on any backend
        # that does not honour a per-request seed (only azure does today). The
        # legacy linear path warns; the archive path was silent. Resolve the
        # effective state ONCE here, warn loudly when the flag was requested but is
        # inert, and stamp ``paired_eval_effective`` into summary.json (see
        # save_results). Default OFF ⇒ no warning, no key ⇒ byte-identical.
        self._paired_eval_effective = False
        if config.paired_eval:
            _llm_cfg = getattr(self.llm_client, "config", None)
            _model = getattr(_llm_cfg, "model", "") or ""
            _backend = getattr(_llm_cfg, "backend", "") or ""
            self._paired_eval_effective = _supports_request_seed(_model, _backend)
            if not self._paired_eval_effective:
                logger.warning(
                    "--paired-eval/CRN is a NO-OP on backend=%r model=%r: this "
                    "backend does not honour a per-request seed (only azure does), "
                    "so no seed reaches the wire and treatment == control. "
                    "Additionally, paired_eval is NOT honored for depth>1 / agentic "
                    "/ external-spine candidates (execute() does not thread the "
                    "seed). summary.json will record paired_eval_effective=false.",
                    _backend, _model,
                )
            else:
                # L2.1 (audit) — even on a seed-honouring backend the CRN seed is
                # threaded ONLY into the outer script-generation call. The inner
                # per-instance llm()/llm_batch() closures (core/llm_helpers.py) take
                # no seed kwarg, so on native depth-1 solve()-heavy benchmarks
                # (Symptom2Disease / LawBench / ARC / CO-Bench-with-llm) the
                # DOMINANT inner channel stays unseeded. ``paired_eval_effective``
                # therefore over-claims: the denoising it reports is PARTIAL (outer
                # channel only), not full per-instance CRN. Disclose so the reported
                # number is not read as fully-paired.
                logger.warning(
                    "--paired-eval/CRN is only PARTIALLY effective on backend=%r "
                    "model=%r: the seed reaches the outer script-generation call but "
                    "NOT the inner per-instance llm()/llm_batch() channel (no seed "
                    "kwarg in core/llm_helpers.py). On native depth-1 solve()-heavy "
                    "benchmarks that inner channel dominates, so reported denoising "
                    "is partial (outer-channel only), not full per-instance CRN. "
                    "Furthermore, paired_eval is NOT honored for depth>1 candidates "
                    "(execute() threads no seed); since every bred child is "
                    "child_depth = parent.depth + 1 >= 2, the child-vs-parent "
                    "SELECTION comparison receives ZERO CRN denoising — only the "
                    "unpaired depth-1 gen0 seed's outer channel is seeded.",
                    _backend, _model,
                )

    def _warn_if_proxy_split_unconsumed(self, adapter) -> bool:
        """R2-ARC-1: warn once when an adapter advertises a ``"proxy"`` split.

        ``split_type() == "proxy"`` (ARC-AGI-2) means the dev/train-demo score
        is a LEAKY proxy for the hidden test grid, so archive selection should
        demote train-overfit candidates with a held-out-stable signal. That
        consumer is not yet wired (deferred feature) — per-task-best selection
        still ranks on the raw dev score. Surface the disabled overfit
        protection once at run start so the gap is visible rather than silent.
        (The reported test number stays honest: it comes from ``evaluate_test``,
        not selection.)

        Returns True when the warning was emitted (for unit-testing). For any
        non-proxy adapter (CO-Bench ``held_out``, classification, etc.)
        this is a no-op, keeping the default path byte-identical.
        """
        try:
            if adapter is not None and hasattr(adapter, "split_type") and \
                    adapter.split_type() == "proxy":
                logger.warning(
                    "Adapter split_type()=='proxy' (%s): dev/train score is a "
                    "leaky proxy for the hidden test grid, but split-aware "
                    "overfit protection is NOT wired into archive selection "
                    "(deferred). Per-task-best selection ranks on the raw dev "
                    "score; a train-overfit candidate can be selected.",
                    getattr(adapter, "name", type(adapter).__name__),
                )
                return True
        except Exception:
            pass
        return False

    async def run(
        self,
        tasks: list[TaskDescription],
        resume: bool = False,
        run_config: dict | None = None,
    ) -> EvolutionaryResult:
        """Main evolutionary loop."""
        if not tasks:
            raise ValueError(
                "No tasks to evaluate (0 loaded). Check --benchmark / --bench-tasks "
                "/ --bench-data-dir — e.g. terminal_bench loads from the task dir "
                "given by --bench-data-dir, not the (empty) harbor cache."
            )
        result = EvolutionaryResult()
        run_start = time.time()
        # Stash the full task set so incremental savers can compute the
        # oracle mean honestly (averaging over all evaluated tasks, not
        # just the ones that produced finite scores).
        self._tasks = tasks
        # Stash the run_config for RunPersistence.save_checkpoint, which stamps
        # it (minus the volatile "timestamp") into checkpoint.json as
        # ``run_config_snapshot`` so a later --resume can detect config drift —
        # e.g. a switched/edited --benchmark-config YAML silently flipping the
        # metacognition stack mid-experiment. None (unit tests / direct
        # callers) writes no key.
        self._run_config_snapshot = run_config

        # --- Ensure output directory exists for incremental saves ---
        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        def _persist_progress() -> None:
            """Write the paired running summary + checkpoint.

            Reads the loop locals (iteration / patience_counter / prev_best /
            prev_oracle) at CALL time — a closure over names, not values — so
            every persist site records the state as of that moment. Keeping the
            pair in one place guarantees summary.json and checkpoint.json are
            never updated one without the other.
            """
            self._save_running_summary(out_dir, iteration, result, run_start)
            self._save_checkpoint(
                out_dir, iteration, patience_counter, prev_best,
                result.total_tokens, result.convergence_history,
                prev_oracle=prev_oracle, oracle_history=result.oracle_history,
                result=result,
            )

        # F6: dump run_config to config.json at run START (in addition to the
        # richer end-of-run dump in save_results) so a crashed / budget-killed
        # / early-resumed run still leaves its full provenance on disk. The
        # START write merge-preserves any END-only fields a prior (richer) write
        # already recorded — it never clobbers them — and is byte-identical
        # content to the END dump when run_config is unchanged. No-op when
        # run_config is None (e.g. tests / direct callers that don't pass one).
        self._write_run_config(out_dir, run_config, stage="start")

        # --- LLM I/O logging tree ---
        # All raw LLM I/O lands under output_dir/llm_io/. Two JSONL files:
        #   outer.jsonl — every outer-LLM call (Omega, Layer1Solver,
        #                 AgenticSolver) made via self.llm_client.
        #   inner.jsonl — every inner-LLM call inside the evolved solve()
        #                 across all candidates (one shared file; subprocesses
        #                 use flock to serialise appends).
        # Without this seam there's no way to audit *what* the orchestrator
        # said to the LLM at each iteration — only what came back. We attach
        # the outer logger to the LLMClient here so every call site picks
        # it up automatically; the inner path is wired into the adapter
        # constructor below by tasks.py.
        from meta_n.utils.llm_io_logger import LLMIOLogger
        llm_io_dir = out_dir / "llm_io"
        llm_io_dir.mkdir(parents=True, exist_ok=True)
        outer_log_path = llm_io_dir / "outer.jsonl"
        inner_log_path = llm_io_dir / "inner.jsonl"
        # On a non-resume (or failed-resume, see below) run, rotate any
        # existing JSONLs out of the way so we don't silently concatenate two
        # unrelated runs' I/O. We rename rather than delete — preserving the
        # prior data with a ``.bak.{N}`` suffix matches how ``run.log`` is
        # rotated by operators and lets users diff old vs new traces. On a
        # successful resume, we leave the files alone; ``LLMIOLogger`` opens
        # with O_APPEND, so the resumed run cleanly continues the existing
        # JSONL. Rotating AFTER the LLMIOLogger below is constructed is safe:
        # the logger stores only the path and re-opens per append.
        def _rotate_io_logs() -> None:
            for p in (outer_log_path, inner_log_path):
                if p.exists() and p.stat().st_size > 0:
                    n = 0
                    while True:
                        bak = p.with_suffix(p.suffix + f".bak.{n}")
                        if not bak.exists():
                            p.rename(bak)
                            logger.info(
                                "Rotated existing LLM I/O log %s → %s",
                                p.name, bak.name,
                            )
                            break
                        n += 1

        if not resume:
            _rotate_io_logs()
        # Stash on self so the adapter setup below (and main.py wiring for
        # benchmarks that build their adapter outside the orchestrator) can
        # read it. None when no output_dir was configured (unit tests).
        self.llm_io_dir = llm_io_dir
        self.inner_log_path = str(inner_log_path)
        if self.llm_client is not None:
            self.llm_client.io_logger = LLMIOLogger(
                outer_log_path, source="outer_llm",
            )
        # Forward the inner path onto whatever adapter the executor wraps,
        # if it accepts it (TextClassificationAdapter, COBenchAdapter).
        # Adapter classes that don't have the attribute simply ignore it.
        adapter = getattr(getattr(self, "executor", None), "adapter", None)
        if adapter is not None and hasattr(adapter, "_inner_log_path"):
            adapter._inner_log_path = self.inner_log_path

        # R2-ARC-1: surface that split-aware overfit protection is disabled
        # for proxy-split adapters (ARC-AGI-2). One-time, at run start.
        self._warn_if_proxy_split_unconsumed(adapter)

        # --- Setup file logger for this run ---
        file_handler = logging.FileHandler(out_dir / "run.log", mode="a" if resume else "w")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            datefmt="%H:%M:%S",
        ))
        root_logger = logging.getLogger("meta_n")
        root_logger.addHandler(file_handler)
        # DEBUG only for the duration of the run: the finally block restores
        # the caller's level so in-process wrapper scripts (and repeated run()
        # calls in one process) don't inherit a leaked DEBUG level.
        prev_level = root_logger.level
        root_logger.setLevel(logging.DEBUG)
        logger.info(
            "Run started: tasks=%d, B=%d, K=%d, max_iter=%d, patience=%d, "
            "max_depth=%d, gate=%d, alpha=%.2f, seed=%d",
            len(tasks), self.config.beam_width, self.config.beam_candidates,
            self.config.max_iterations, self.config.patience,
            self.config.max_depth, self.config.gate_tasks,
            self.config.novelty_alpha, self.config.seed,
        )

        try:
            # --- Attempt resume if requested ---
            checkpoint = self._try_resume(out_dir) if resume else None
            if resume and checkpoint is None:
                # Failed resume ⇒ this is a fresh run; rotate the old JSONLs
                # so we don't concatenate two runs' I/O into one file.
                _rotate_io_logs()
            completed_set: set[str] = set()

            if checkpoint is not None:
                # SRI formal: a crashed process can leave a slot row open. Close
                # it now (as a visible generation_error) so resume cannot leave a
                # nominal slot permanently unfinished.
                _dangling = self.sri.finalize_dangling_on_resume()
                if _dangling:
                    logger.warning(
                        "SRI: closed %d dangling slot(s) left open by a previous "
                        "process", _dangling)
                # Subtract 1 because the while loop does `iteration += 1` at the top.
                # If the checkpoint saved iteration=2 mid-iteration, we need to
                # re-enter iteration 2 (with completed_set skipping done candidates).
                # If all candidates for that iteration were already done, the
                # re-entry is a no-op (all skipped) and patience/convergence
                # are restored from checkpoint, not re-computed.
                # Min 0: checkpoint iteration=0 means seed is done → loop starts at 1.
                iteration = max(checkpoint["iteration"] - 1, 0)
                patience_counter = checkpoint["patience_counter"]
                prev_best = checkpoint["prev_best"]
                result.total_tokens = checkpoint["total_tokens"]
                result.convergence_history = checkpoint["convergence_history"]
                # N7 / 1.5a resume: read with defaults so OLD checkpoints (no
                # oracle / frozen-range keys) don't KeyError.
                prev_oracle = checkpoint.get("prev_oracle", prev_best)
                result.oracle_history = checkpoint.get("oracle_history", [])
                _fsr = checkpoint.get("frozen_score_range")
                if _fsr is not None:
                    self.archive.frozen_score_range = _fsr
                # Forensic improvement #1 — re-apply the base floor + denoised
                # focus scores across a pause/resume boundary so the regression
                # guard does not silently weaken (rebuild_from_disk reconstructs
                # only the single stored seed trace; the floor lives in the
                # checkpoint). OLD checkpoints lack the key ⇒ .get default ⇒ no
                # floor (safe, no-op).
                if self.config.regression_guard:
                    _bf = checkpoint.get("base_floor")
                    if _bf:
                        floor = {tid: e["score"] for tid, e in _bf.items()}
                        floor_traces = {
                            tid: (Trace.model_validate(e["trace"]) if e.get("trace") else None)
                            for tid, e in _bf.items()
                        }
                        self.archive.set_base_floor(floor, floor_traces)
                    self._base_focus_scores = checkpoint.get("base_focus_scores", {}) or {}
                # Seed the LLMClient's running counter with whatever was on
                # disk so the end-of-run summary's prompt/completion split
                # covers the FULL run (pre-resume + post-resume), matching
                # the already-restored ``total_tokens``. Older checkpoints
                # without this key seed to zero — same behaviour as before.
                if (
                    self.llm_client is not None
                    and "outer_cumulative_usage" in checkpoint
                ):
                    ckpt_usage = checkpoint["outer_cumulative_usage"] or {}
                    self.llm_client.cumulative_usage = {
                        "prompt": int(ckpt_usage.get("prompt", 0) or 0),
                        "completion": int(ckpt_usage.get("completion", 0) or 0),
                        "total": int(ckpt_usage.get("total", 0) or 0),
                        "calls": int(ckpt_usage.get("calls", 0) or 0),
                        "cached": int(ckpt_usage.get("cached", 0) or 0),
                        "cost_usd": float(ckpt_usage.get("cost_usd", 0.0) or 0.0),
                    }
                # Audit #1: restore the INNER-LLM token accounting too. The outer
                # channel above (total_tokens + cumulative_usage) is restored, but
                # result.inner_* start at 0 and only accumulate post-resume
                # candidates, so a resumed run silently undercounts
                # token_usage.inner_total / inner_prompt / inner_completion /
                # inner_calls by the pre-resume slice — the same asymmetry the
                # outer block above was added to prevent. Mirror that restore; OLD
                # checkpoints lack these keys ⇒ .get default 0 ⇒ byte-identical.
                result.inner_tokens = int(checkpoint.get("inner_tokens", 0) or 0)
                result.inner_prompt_tokens = int(
                    checkpoint.get("inner_prompt_tokens", 0) or 0
                )
                result.inner_completion_tokens = int(
                    checkpoint.get("inner_completion_tokens", 0) or 0
                )
                result.inner_calls = int(checkpoint.get("inner_calls", 0) or 0)
                # Source of truth for "what's been done" is the disk archive (just
                # rebuilt at line ~870), not the checkpoint. Trusting a checkpoint
                # snapshot introduced a gap: a crash between archive.add and
                # _save_checkpoint left the disk with a candidate whose id was
                # missing from the snapshot, causing re-evaluation + duplicate
                # add. The old ``completed_candidates`` checkpoint key was removed
                # — the disk archive is the sole source of truth (legacy
                # checkpoints carrying the key remain loadable; it is ignored).
                completed_set = {c.candidate_id for c in self.archive.candidates}
                logger.info(
                    "Resuming at iteration %d with %d candidates already on disk",
                    checkpoint["iteration"], len(completed_set),
                )
                if self.config.use_agentic != checkpoint.get("use_agentic", False):
                    logger.warning(
                        "Resuming with different solver mode (use_agentic=%s vs checkpoint=%s)",
                        self.config.use_agentic, checkpoint.get("use_agentic", False),
                    )
                    console.print(
                        f"[yellow]Warning: solver mode changed "
                        f"(agentic={self.config.use_agentic} vs checkpoint={checkpoint.get('use_agentic', False)})[/yellow]"
                    )
                # Config-drift detection: the checkpoint stamps the original
                # process's run_config (``run_config_snapshot``, timestamp
                # excluded). A resume re-derives its config from the CURRENT
                # CLI + benchmark-features YAML, so an edited/switched YAML
                # (or a pre-YAML checkpoint resumed under the bundled
                # defaults) silently changes the search mid-experiment. Warn
                # per drifted key and stash both sides for
                # RunPersistence.write_run_config to record as
                # ``resume_config_drift`` in config.json. Warn-only: the
                # resumed values win (they are what the continued
                # generations actually run under).
                snap = checkpoint.get("run_config_snapshot")
                if snap and run_config:
                    drift = _config_drift(snap, run_config)
                    if drift:
                        self._resume_config_drift = drift
                        logger.warning(
                            "Resume config drift on %d key(s): %s",
                            len(drift), ", ".join(drift),
                        )
                        console.print(
                            f"[yellow]Warning: run config drifted since the "
                            f"checkpoint on {len(drift)} key(s): "
                            f"{', '.join(drift)} — resumed values win; both "
                            f"sides recorded in config.json under "
                            f"resume_config_drift[/yellow]"
                        )
                        # Re-issue the START dump NOW so the drift record is
                        # durable before any iteration work: the END dump never
                        # runs on a crash / budget hard-kill, and the first
                        # _persist_progress refreshes the checkpoint snapshot
                        # to the drifted config, erasing the evidence needed to
                        # re-detect it. Drift-free resumes never reach here, so
                        # their config.json is untouched (byte-compat).
                        self._write_run_config(out_dir, run_config, stage="start")
            else:
                # --- Seed: run baseline ---
                seed_label = "agentic baseline" if self.config.use_agentic else "Layer1Solver baseline"
                console.print(f"\n[bold blue]═══ Iteration 0: Seed ({seed_label}) ═══[/bold blue]")
                seed_start = time.time()
                # P1c: seed a known-good helper into gen0's code_library BEFORE Ω
                # runs (default None ⇒ injected_codes=[] ⇒ depth=1 ⇒ byte-identical
                # to the bare Layer1Solver baseline). When set, gen0 becomes a
                # depth-2 candidate carrying the seeded library at source_depth 0.
                seed_injected: list[InjectedCode] = []
                if self.config.seed_code_library:
                    seed_injected = [
                        InjectedCode(
                            code_library=dict(self.config.seed_code_library),
                            source_depth=0,
                        )
                    ]
                seed = Candidate(
                    candidate_id="gen0_seed",
                    iteration=0,
                    depth=1 + len(seed_injected),
                    injected_codes=seed_injected,
                )
                # Route the seed (gen0 baseline) through the candidate solver builder
                # whenever a non-default solver is in play. With an external
                # base_solver the seed must be the vanilla external agent
                # (injected_codes=[]) — without this the gen0 baseline would silently
                # run the wrong solver (plan §8.1 seed path). The agentic path keeps
                # its existing routing; the plain Layer1Solver path is unchanged. A
                # seeded code library (P1c) also routes through the builder so the
                # seeded helper is staged via MetaLayer.
                if (
                    self._uses_external_spine()
                    or self.config.use_agentic
                    or self.config.seed_code_library
                ):
                    seed_solver = self._build_solver_from_candidate(seed)
                else:
                    seed_solver = self.solver
                seed = await self._evaluate_candidate(seed, seed_solver, tasks)
                self.archive.add(seed)
                result.absorb_candidate_tokens(seed)
                # Forensic improvement #1 — establish the BASE (seed-only) floor
                # and the median-of-R denoised focus scores. Flag-gated: when
                # regression_guard is OFF this block is SKIPPED entirely, so the
                # seed runs exactly once and no resampling/floor exists (HEAD path).
                if self.config.regression_guard:
                    await self._establish_base_floor(seed, seed_solver, tasks, result)
                result.convergence_history.append(self.archive.best_mean_score)
                self._print_candidate(seed, "seed")
                console.print(f"  Seed tokens: {seed.total_tokens:,} | Time: {time.time()-seed_start:.1f}s")
                logger.info(
                    "Seed evaluated: mean_score=%.3f, pass@1=%.3f, tokens=%d, time=%.1fs",
                    seed.mean_score, seed.pass_at_1, seed.total_tokens, time.time()-seed_start,
                )
                self._save_candidate_incremental(seed, out_dir)
                prev_best = self.archive.best_mean_score
                prev_oracle = prev_best  # N7: oracle == best at the seed
                patience_counter = 0
                iteration = 0
                _persist_progress()

            # --- Evolutionary loop ---
            ptb_mean = self.archive.best_mean_score  # default in case loop never runs
            # N9a: set when the cost-guard headroom break fires, so the exit
            # classifier and run_status can label a budget halt honestly
            # ("aborted_mid" / "aborted_pre_iteration", not "completed").
            # Never set on the legacy path (self._cost_guard is None).
            budget_halted = False
            # Set when the all-at-max-depth break fires; with budget_halted it
            # marks the two mid-loop breaks whose top-of-loop increment counted
            # an iteration that bred nothing (see completed_iterations below).
            max_depth_halted = False

            while iteration < self.config.max_iterations and patience_counter < self._effective_patience:
                iteration += 1
                iter_start = time.time()

                # Generation-boundary daily-cap halt (plan §4.7.4): for external
                # agents the only enforcing gates are the per-task admission
                # precheck and this check. If today's spend has consumed the
                # day's headroom, stop dispatching further candidates rather than
                # admitting a whole new generation of (already-uncapped) runs.
                if self._cost_guard is not None and self._cost_guard.headroom_exhausted():
                    console.print(
                        "  [yellow]Daily budget headroom exhausted — halting "
                        "further candidate dispatch (resume tomorrow or raise "
                        "--daily-budget-usd)[/yellow]"
                    )
                    logger.warning(
                        "Daily budget headroom exhausted at iteration %d; halting "
                        "candidate dispatch (external-agent daily-cap backstop).",
                        iteration,
                    )
                    budget_halted = True
                    # F033 (§6b): no history append — the halt fires before any
                    # breeding this iteration, so an entry here would duplicate
                    # the previous generation's value, desync convergence.json
                    # from oracle_convergence.json (sibling contract) and from
                    # the checkpoint (which never contained it).
                    break

                console.print(
                    f"\n[bold blue]═══ Iteration {iteration} "
                    f"(archive={len(self.archive)}, best={self.archive.best_mean_score:.3f}) ═══[/bold blue]"
                )
                logger.info("=== Iteration %d start (archive=%d, best=%.3f) ===",
                            iteration, len(self.archive), self.archive.best_mean_score)

                # Pre-filter to the BREEDABLE pool: extendable (depth < max_depth)
                # AND non-regressing (6.1). archive.add stays unconditional (history
                # + oracle harvest preserved), but we do not breed from the weak-
                # chain cloud. Never empty when any candidate is extendable.
                extendable_pool = self.archive.breedable_pool(self.config.max_depth)
                if not extendable_pool:
                    console.print("  [yellow]All candidates at max depth — stopping[/yellow]")
                    logger.info("All candidates at max_depth=%d — stopping", self.config.max_depth)
                    max_depth_halted = True
                    # F033 (§6b): no history append — the halt fires before any
                    # breeding this iteration, so an entry here would duplicate
                    # the previous generation's value, desync convergence.json
                    # from oracle_convergence.json (sibling contract) and from
                    # the checkpoint (which never contained it).
                    break

                # F006 (§6b): per-call rotation offset keeps the Archive
                # stateless — ``iteration`` is the 1-based loop variable; 0
                # (flag OFF, default) is the byte-identical HEAD reservation.
                parents = self.archive.select_parents(
                    self.config.beam_width, rng=self.rng, pool=extendable_pool,
                    elite_rotation=(iteration if self.config.elite_rotation else 0),
                )

                # Log parent selection diagnostics with the scale-invariant UCB
                # exploration term (rank-normalized fitness is computed inside
                # Archive._selection_weights; the additive bonus is retired).
                pool_n = len(extendable_pool)
                for p in parents:
                    ucb = Archive.ucb_exploration_bonus(
                        self.config.novelty_alpha, pool_n, p.num_children
                    )
                    logger.info(
                        "  Parent selected: %s (score=%.3f, depth=%d, children=%d, ucb=%.4f)",
                        p.candidate_id, p.mean_score, p.depth, p.num_children, ucb,
                    )

                console.print(
                    f"  Parents: {[p.candidate_id for p in parents]} "
                    f"(scores: {[f'{p.mean_score:.3f}' for p in parents]}, "
                    f"depths: {[p.depth for p in parents]})"
                )

                any_added = False
                any_new_work = False  # tracks whether any candidate was actually evaluated (not skipped)
                iter_tokens = 0

                # G9: in consolidate mode, assign each candidate one target task
                # to improve (round-robin by iteration); every other task is
                # inherited at its per-task-best (no re-solve), so improvement is
                # monotonic and collateral-free. Empty list ⇒ normal breeding.
                cons_targets = (
                    self._consolidation_targets(
                        tasks, self.config.beam_candidates, iteration
                    )
                    if self.config.consolidate else []
                )

                for parent_idx, parent in enumerate(parents):

                    # Generate K children
                    for k in range(self.config.beam_candidates):
                        # Skip if already evaluated (resume case) — before Omega to save tokens
                        child_id = f"gen{iteration}_b{parent_idx}_k{k}"
                        if child_id in completed_set:
                            logger.info("  Skipping already-evaluated candidate: %s", child_id)
                            console.print(f"  [dim]Skipping {child_id} (already evaluated)[/dim]")
                            # SRI formal: this nominal slot already has a terminal
                            # row from the previous process. Reference it; a
                            # re-entered slot must not become a second
                            # experimental observation (§5).
                            self.sri.resume_note(iteration=iteration,
                                                 parent_slot=parent_idx,
                                                 child_slot=k)
                            continue

                        any_new_work = True
                        child_start = time.time()
                        temperature = self._select_temperature(k, iteration)

                        # Get inspiration from archive
                        inspiration: list[Trace] = []
                        if self.config.use_inspiration:
                            inspiration = self.archive.get_inspiration_traces(parent, tasks)
                            if inspiration:
                                insp_info = [
                                    f"{t.task_id}(score={t.score:.3f})"
                                    for t in inspiration
                                ]
                                console.print(f"    Inspiration: {len(inspiration)} traces from archive: {insp_info}")
                                logger.info(
                                    "  Inspiration for parent=%s: %d traces — %s",
                                    parent.candidate_id, len(inspiration),
                                    ", ".join(insp_info),
                                )
                            else:
                                logger.debug("  No inspiration available for parent=%s", parent.candidate_id)

                        # Call Omega
                        child_depth = parent.depth + 1
                        console.print(
                            f"  Omega: parent={parent.candidate_id}, k={k}, "
                            f"temp={temperature:.1f}, depth={child_depth}...",
                            end=" ",
                        )

                        # Provide baseline scores for Omega's regression analysis.
                        # Omega receives parent.traces → previous_scores must be from
                        # the generation BEFORE the parent, so the comparison shows
                        # whether the parent's layer helped.
                        previous_scores = None
                        archive_best_scores = self.archive.per_task_best_scores() or None
                        if child_depth == 2:
                            # At depth 2: parent is seed → compare against seed baseline
                            seed = self.archive.find("gen0_seed")
                            if seed and seed.per_task_scores:
                                previous_scores = seed.per_task_scores
                                logger.info(
                                    "  Depth 2 baseline: seed=%s (mean=%.3f)",
                                    seed.candidate_id, seed.mean_score,
                                )
                        elif child_depth >= 3 and parent.parent_id:
                            # At depth 3+: compare against grandparent (one gen before parent)
                            grandparent = self.archive.find(parent.parent_id)
                            if grandparent and grandparent.per_task_scores:
                                previous_scores = grandparent.per_task_scores
                                logger.info(
                                    "  Depth 3+ baseline: grandparent=%s (mean=%.3f)",
                                    grandparent.candidate_id, grandparent.mean_score,
                                )

                        # G9: the one task this consolidation candidate improves
                        # (None ⇒ normal breeding). Drives both the Ω FOCUS hint
                        # and the inherit-frozen precomputed map built below.
                        #
                        # Forensic improvement #3 — when within_task_recursion is
                        # ON, _within_task_focus INHERITS the parent's single fresh
                        # task so the SAME task persists down the chain (genuine
                        # within-task deepening); OFF ⇒ the exact round-robin pick
                        # below (byte-identical).
                        focus_task = self._within_task_focus(
                            parent, cons_targets, k
                        )
                        # SRI formal: the slot row is created BEFORE dispatch, so
                        # a crash mid-slot is visible as an open row rather than a
                        # silently smaller denominator (§5). `gate_effective` is
                        # recorded as EFFECTIVE, not as the configured count.
                        _sri_slot = self.sri.open(
                            iteration=iteration, parent_slot=parent_idx,
                            child_slot=k, parent_id=parent.candidate_id,
                            parent_structural_depth=parent.depth,
                            gate_effective=bool(self.config.gate_tasks > 0
                                                and not focus_task),
                            proposed_child_id=child_id,
                            # the child's STRUCTURAL depth, so the canonical
                            # transition audit can place the edge. Omitting it
                            # would make every edge unplaceable and the audit
                            # would find zero edges.
                            proposed_child_depth=child_depth,
                            temperature=temperature, focus_task=focus_task)
                        omega_start = time.time()
                        injected, omega_tokens = await self.omega.generate(
                            traces=parent.traces,
                            context_stack=parent.injected_codes,
                            tasks=tasks,
                            depth=child_depth,
                            temperature=temperature,
                            inspiration_traces=inspiration if inspiration else None,
                            previous_scores=previous_scores,
                            archive_best_scores=archive_best_scores,
                            focus_task=focus_task,
                            # G1/G2: the parent's OWN per-task scores are the
                            # authoritative "current" performance Ω is improving
                            # (the traces are failure-biased once sampled), so
                            # the objective headline + headroom table are exact.
                            current_scores=parent.per_task_scores or None,
                            solver_language=self.solver_language,
                            no_code_library=self.config.no_code_library,
                            # P8(a): when helpers are demoted (not prepended to
                            # the solver, e.g. CO-Bench/SWE), suppress the
                            # misleading "DEAD helper" callout in the Ω prompt.
                            code_library_is_live=self._code_library_is_live(),
                            # Forensic improvement #3 — DEEPEN (re-work the same
                            # task) vs FOCUS-freeze (gate a disjoint task). Default
                            # False ⇒ the freeze directive renders verbatim.
                            within_task_recursion=self.config.within_task_recursion,
                        )
                        omega_time = time.time() - omega_start
                        # E2 ablation: defensively clear any code_library that slipped through
                        # (e.g., if the LLM hallucinates the schema from training despite the
                        # stripped prompt). Pre_process is the only surviving channel.
                        if self.config.no_code_library:
                            injected.code_library = {}
                            injected.code_library_bash = {}

                        # Forensic improvement #2 — VERIFY-THEN-INJECT gate
                        # (default OFF ⇒ returns `injected` unchanged, no sandbox).
                        # When ON, sandbox-execute each Ω code_library helper
                        # (--network none) against a held-out check and keep ONLY
                        # the helpers that pass; drop dead/wrong ones. Replaces the
                        # local omega object BEFORE the candidate is built — the
                        # dropped-helper variant is never archived (monotonic).
                        injected, sandbox_verified_names = (
                            await self._verify_and_filter_code_library(
                                injected, parent.injected_codes, tasks
                            )
                        )

                        if injected.is_empty:
                            if not focus_task:
                                # Reaudit #10: only the SKIP-and-continue path books
                                # omega_tokens here (no child is built). On the
                                # consolidate FOCUS fall-through below, omega_tokens
                                # is instead carried on ``child.total_tokens`` (built
                                # at the resample, L1042) and booked ONCE via
                                # ``result.total_tokens += child.total_tokens`` after
                                # evaluation — mirroring the non-empty focus sibling.
                                # Pre-adding it here too would double-count it on that
                                # fall-through path.
                                result.total_tokens += omega_tokens
                                iter_tokens += omega_tokens
                                console.print(f"[yellow]empty injection (tokens={omega_tokens:,}, {omega_time:.1f}s)[/yellow]")
                                logger.warning(
                                    "Empty injection: parent=%s, k=%d, depth=%d, tokens=%d, time=%.1fs",
                                    parent.candidate_id, k, child_depth, omega_tokens, omega_time,
                                )
                                parent.num_children += 1  # count failed attempts to deprioritize in selection
                                # SRI formal: an empty Omega injection is a SPENT
                                # slot (§8) -- it is labelled, never silently
                                # turned into an extra proposal.
                                self.sri.close(
                                    _sri_slot, "empty_injection",
                                    failure_class="empty_omega",
                                    failure_message="Omega returned an empty "
                                                    "injection; slot consumed",
                                    omega_wall_seconds=omega_time,
                                    outer_calls=1, prompt_tokens=omega_tokens)
                                continue
                            # G9: an empty Ω injection in consolidate mode is NOT a
                            # skip — fall through to re-solve the FOCUS task fresh (a
                            # plain resample). The child carries only the parent's
                            # solver, so the target gets a new draw while the rest stay
                            # inherited — giving an unlucky frozen score the same
                            # best-of-many recovery the resampling control gets. (E2-G9:
                            # aircraft was locked at the seed's unlucky 0.318 because its
                            # one focus generation returned empty and was skipped, which
                            # alone lost the run to the best-of-3 control.)
                            console.print(
                                f"[yellow]empty Ω → plain resample of focus={focus_task} "
                                f"(tokens={omega_tokens:,})[/yellow]"
                            )
                            logger.info(
                                "Empty injection in consolidate mode → resampling "
                                "focus=%s (parent=%s, k=%d)",
                                focus_task, parent.candidate_id, k,
                            )

                        # Log what Omega produced
                        console.print(f"done (tokens={omega_tokens:,}, {omega_time:.1f}s)")
                        if injected.rationale:
                            console.print(f"    Rationale: {injected.rationale[:150]}...")
                        if injected.pre_process:
                            lines = injected.pre_process.strip().splitlines()
                            preview = lines[0][:80] if lines else ""
                            console.print(f"    pre_process: {preview}... ({len(injected.pre_process)} chars)")
                        if injected.code_library:
                            console.print(f"    code_library: {list(injected.code_library.keys())}")
                        if injected.code_library_bash:
                            console.print(f"    code_library_bash: {list(injected.code_library_bash.keys())}")

                        logger.info(
                            "  Omega produced: parent=%s, k=%d, depth=%d, tokens=%d, time=%.1fs, "
                            "pre=%s, code_lib=%s, code_lib_bash=%s, rationale_len=%d",
                            parent.candidate_id, k, child_depth, omega_tokens, omega_time,
                            f"{len(injected.pre_process)} chars" if injected.pre_process else "none",
                            list(injected.code_library.keys()) if injected.code_library else "none",
                            list(injected.code_library_bash.keys()) if injected.code_library_bash else "none",
                            len(injected.rationale),
                        )

                        # Build child candidate (unique ID includes parent index to avoid collisions
                        # when the same parent is selected multiple times in one iteration)
                        child = Candidate(
                            candidate_id=child_id,
                            parent_id=parent.candidate_id,
                            iteration=iteration,
                            depth=child_depth,
                            injected_codes=parent.injected_codes + [injected],
                            temperature_used=temperature,
                            total_tokens=omega_tokens,
                        )

                        # INHERITANCE INVARIANT (user directive 2026-09-18). A bred
                        # child EXTENDS the parent's meta-layer chain and never
                        # replaces it. The recursive-correction mechanism -- a
                        # deeper Omega reading the failure trace of a layer BELOW
                        # and repairing it -- depends entirely on the lower layers
                        # surviving into the child, so a silent replacement here
                        # would turn "recursion" into "restart" without any visible
                        # symptom. Verified against upstream: the child is built
                        # exactly as parent.injected_codes + [new_injection].
                        _parent_chain = [
                            (ic.pre_process, ic.code_library,
                             ic.code_library_bash, ic.rationale)
                            for ic in parent.injected_codes
                        ]
                        _child_chain = [
                            (ic.pre_process, ic.code_library,
                             ic.code_library_bash, ic.rationale)
                            for ic in child.injected_codes[:-1]
                        ]
                        assert _child_chain == _parent_chain, (
                            "inheritance violated: child %s does not extend "
                            "parent %s's meta-layer chain (parent has %d layers, "
                            "child prefix has %d)"
                            % (child.candidate_id, parent.candidate_id,
                               len(parent.injected_codes), len(_child_chain))
                        )
                        assert len(child.injected_codes) == len(
                            parent.injected_codes) + 1, (
                            "inheritance violated: child %s depth-in-chain %d != "
                            "parent %s depth-in-chain %d + 1"
                            % (child.candidate_id, len(child.injected_codes),
                               parent.candidate_id, len(parent.injected_codes))
                        )

                        # Build solver chain
                        child_solver = self._build_solver_from_candidate(child)

                        # Gate check. G9: skipped in consolidate mode — only the
                        # focus task is solved and inherited tasks cannot regress,
                        # so the gate is moot, and gate re-solves would re-roll the
                        # frozen tasks.
                        gate_traces: dict[str, Trace] = {}
                        if self.config.gate_tasks > 0 and not focus_task:
                            gate_start = time.time()
                            # Audit #29: the gate solves are REAL outer LLM
                            # script-generation / agent calls whose tokens were
                            # silently discarded (on gate-pass the traces are
                            # reused at 0 outer tokens; on gate-fail only
                            # omega_tokens was counted), so result.total_tokens
                            # drifted below the authoritative cumulative_usage. We
                            # measure the gate's OUTER spend as the delta of the
                            # shared LLMClient's cumulative total across the gate
                            # call (it already records every native/MetaLayer
                            # script-generation call) and add it both ways. The
                            # gate loop is sequential here, so no other outer call
                            # interleaves. Falls back to 0 when no client is wired.
                            _cu_before_gate = self._outer_cumulative_total()
                            passed, gate_n, gate_traces = await self._gate_check(child, parent, child_solver, tasks)
                            gate_time = time.time() - gate_start
                            gate_tokens = max(
                                0, self._outer_cumulative_total() - _cu_before_gate
                            )
                            result.total_tokens += gate_tokens
                            iter_tokens += gate_tokens
                            if not passed:
                                console.print(
                                    f"    [red]Gate FAILED (depth={child_depth}, "
                                    f"tested {gate_n} tasks, {gate_time:.1f}s)[/red]"
                                )
                                logger.info(
                                    "  Gate FAILED: %s, depth=%d, tested=%d tasks, time=%.1fs",
                                    child_id, child_depth, gate_n, gate_time,
                                )
                                # SRI formal (§5/§8): a gate-rejected child is a
                                # real terminal slot and MUST enter the canonical
                                # audit whenever it is executable. Its material is
                                # written OUTSIDE the archive so persistence can
                                # never make it a parent or a Final-Score
                                # candidate.
                                _mp = self.sri.capture_material(
                                    slot_id=_sri_slot.slot_id,
                                    candidate_id=child.candidate_id,
                                    structural_depth=child_depth,
                                    task_scripts={
                                        str(getattr(t, "task_id", "")):
                                            str(getattr(t, "script", "") or "")
                                        for t in (child.traces or [])})
                                self.sri.close(
                                    _sri_slot, "gate_rejected",
                                    gate_status="rejected",
                                    archive_admitted=False,
                                    archive_rejection_reason="gate",
                                    material_path=_mp,
                                    omega_wall_seconds=omega_time,
                                    gate_wall_seconds=gate_time,
                                    outer_calls=1)
                                result.total_tokens += omega_tokens
                                iter_tokens += omega_tokens
                                # Reaudit #6: the gate solves are real INNER-LLM
                                # spend on solve()-heavy benchmarks (the evolved
                                # solve() calls llm() per case). On gate-PASS these
                                # gate traces are reused via precomputed=gate_traces
                                # and their inner tokens flow into result.inner_*
                                # through _evaluate_candidate; on gate-FAIL the
                                # candidate is never evaluated, so without this the
                                # gate solves' inner tokens vanish from summary.json
                                # token_usage.inner_total. Mirror the audit-#29 outer
                                # reconciliation above on the INNER channel; counted
                                # once (the child is discarded). Empty gate_traces
                                # (external-agent / native-only benchmarks) → no-op.
                                _gate_inner = self._sum_inner_traces(
                                    gate_traces.values()
                                )
                                result.inner_tokens += _gate_inner["inner_tokens"]
                                result.inner_prompt_tokens += _gate_inner[
                                    "inner_prompt_tokens"
                                ]
                                result.inner_completion_tokens += _gate_inner[
                                    "inner_completion_tokens"
                                ]
                                result.inner_calls += _gate_inner["inner_calls"]
                                # Stage 2 WITHIN-LAYER REFINE (flag-gated, default
                                # OFF ⇒ byte-identical): one extra Ω call to FIX the
                                # buggy injection (error-correction), kept as a NEW
                                # candidate iff it now clears the gate (monotonic —
                                # the rejected `child` is never mutated). NEVER fires
                                # on empty injections / focus resamples (those never
                                # reach here) or depth<2.
                                if (
                                    self.config.within_layer_refine
                                    and not injected.is_empty
                                    and not focus_task
                                    and child_depth >= 2
                                ):
                                    refined, refine_spent = (
                                        await self._attempt_within_layer_refine(
                                            parent=parent,
                                            buggy_child=child,
                                            buggy_injection=injected,
                                            gate_traces=gate_traces,
                                            tasks=tasks,
                                            iteration=iteration,
                                            child_depth=child_depth,
                                            temperature=temperature,
                                            previous_scores=previous_scores,
                                            archive_best_scores=archive_best_scores,
                                            result=result,
                                            out_dir=out_dir,
                                        )
                                    )
                                    iter_tokens += refine_spent
                                    if refined is not None:
                                        any_added = True
                                        _persist_progress()
                                parent.num_children += 1  # deprioritize in future selection
                                continue
                            else:
                                console.print(
                                    f"    [green]Gate passed ({gate_n} tasks, {gate_time:.1f}s)[/green]"
                                )
                                logger.info(
                                    "  Gate passed: %s, depth=%d, tested=%d tasks, time=%.1fs",
                                    child_id, child_depth, gate_n, gate_time,
                                )

                        # Full evaluation. G9: in consolidate mode, inherit every
                        # NON-target task at its per-task-best frozen trace (no
                        # re-solve, 0 tokens) so this candidate can only change its
                        # one focus task — monotonic, collateral-free.
                        precomputed = gate_traces
                        if focus_task:
                            precomputed = self._consolidation_precomputed(
                                focus_task, tasks, child_depth
                            )
                        eval_start = time.time()
                        console.print(
                            f"    Evaluating {child_id} (depth={child_depth})"
                            + (f" [G9 focus={focus_task}]" if focus_task else "")
                            + "..."
                        )
                        # F035 (§6b): precomputed_frozen marks the consolidation
                        # inherit-frozen map (focus_task set), which must NEVER
                        # be re-solved — only gate traces are top-up eligible.
                        child = await self._evaluate_candidate(
                            child, child_solver, tasks, precomputed=precomputed,
                            precomputed_frozen=bool(focus_task),
                        )
                        eval_time = time.time() - eval_start
                        result.absorb_candidate_tokens(child)
                        iter_tokens += child.total_tokens
                        # Forensic improvement #2 — non-adoption penalty (default
                        # OFF ⇒ the exact original call). When verified_code is ON,
                        # a verified helper that the solver advertised but RE-DERIVED
                        # inline (never called) bars that task's trace from becoming
                        # per-task-best (composes with #1's regression guard) — the
                        # solver cannot bank a win it got by ignoring the verified
                        # helper. The trace is still archived (monotonic, mean intact).
                        if self.config.verified_code:
                            barred = self._nonadopting_verified_tasks(
                                child, injected, sandbox_verified_names
                            )
                            self.archive.add(child, bar_from_best=barred)
                        else:
                            self.archive.add(child)
                        parent.num_children += 1  # only count when child actually added
                        any_added = True
                        # SRI formal: the child ran and was admitted. `fresh` vs
                        # `inherited` task ids come from Meta^n's own depth rule
                        # (a trace is fresh iff its depth equals the candidate's).
                        _fresh = sorted({
                            str(getattr(t, "task_id", ""))
                            for t in (child.traces or [])
                            if int(getattr(t, "depth", 0) or 0) == child.depth})
                        _inh = sorted({
                            str(getattr(t, "task_id", ""))
                            for t in (child.traces or [])
                            if int(getattr(t, "depth", 0) or 0) != child.depth})
                        self.sri.close(
                            _sri_slot, "evaluated_admitted",
                            archive_admitted=True,
                            fresh_task_ids=_fresh, inherited_task_ids=_inh,
                            omega_wall_seconds=omega_time,
                            eval_wall_seconds=eval_time,
                            outer_calls=1, evaluator_calls=len(child.traces or []),
                            prompt_tokens=omega_tokens,
                            completion_tokens=child.total_tokens)
                        self._print_candidate(child, child_id)
                        console.print(
                            f"    Eval tokens: {child.total_tokens:,} | "
                            f"Eval time: {eval_time:.1f}s | "
                            f"Total child time: {time.time()-child_start:.1f}s"
                        )
                        logger.info(
                            "  Candidate added: %s (depth=%d, mean=%.3f, pass@1=%.3f, "
                            "eval_tokens=%d, eval_time=%.1fs, total_child_time=%.1fs) → "
                            "archive size=%d",
                            child_id, child.depth, child.mean_score, child.pass_at_1,
                            child.total_tokens, eval_time, time.time()-child_start,
                            len(self.archive),
                        )
                        self._save_candidate_incremental(child, out_dir)
                        _persist_progress()
                        logger.debug("  Saved candidate %s to %s", child_id, out_dir)

                        # Stage 3 DOWNWARD RE-PROPAGATION (flag-gated, default OFF
                        # ⇒ byte-identical): when a depth>=3 child PLATEAUED /
                        # regressed vs its parent, the proximate cause may be an
                        # INTERMEDIATE layer, not the new top one. Regenerate an
                        # intermediate layer's injection from the full-chain
                        # failure traces + the above-layer injections as downstream
                        # feedback, REPLACE it in a NEW monotonic candidate, and
                        # log + classify a `downstream` SelfRepairEvent. NEVER fires
                        # on focus/consolidate resamples or depth<3 (gen0/depth-1
                        # parity preserved — d_t >= 2 only touches meta-layers).
                        if (
                            self.config.repropagation
                            and child_depth >= 3
                            and not focus_task
                            and self._should_repropagate(child)
                        ):
                            reprop, reprop_spent = (
                                await self._attempt_repropagation(
                                    child=child,
                                    tasks=tasks,
                                    iteration=iteration,
                                    temperature=temperature,
                                    archive_best_scores=archive_best_scores,
                                    result=result,
                                    out_dir=out_dir,
                                )
                            )
                            iter_tokens += reprop_spent
                            if reprop is not None:
                                any_added = True
                                _persist_progress()

                # If no new work was done (all candidates skipped on resume), skip
                # patience/convergence logic to avoid double-counting.
                if not any_new_work:
                    logger.info("  All candidates skipped (resume) — skipping patience check")
                    # Don't append to convergence_history: this iteration's entry
                    # is already in the checkpoint-restored history.
                    continue

                # Patience check: increment once per iteration if no archive-best improvement
                current_best = self.archive.best_mean_score
                result.convergence_history.append(current_best)

                if not any_added:
                    console.print("  [yellow]No candidates added this iteration[/yellow]")
                    logger.info("  No candidates added this iteration")

                # Per-task best (oracle upper bound). Average over ALL evaluated
                # tasks (missing → 0) so an evaluator crash on one task can't
                # silently inflate the reported oracle mean.
                ptb = self.archive.per_task_best_scores()
                ptb_mean = (
                    sum(ptb.get(t.task_id, 0.0) for t in tasks) / len(tasks)
                    if tasks else 0.0
                )
                ptb_sources = set(self.archive.per_task_best_sources().values())
                console.print(
                    f"  Per-task best mean: {ptb_mean:.3f} "
                    f"(from {len(ptb_sources)} different chains)"
                )

                iter_time = time.time() - iter_start
                # SRI formal: close anything this iteration left open. The sweep
                # is the guarantee behind "exactly one terminal row per nominal
                # slot" -- a branch that forgets to close becomes a VISIBLE
                # generation_error instead of a silently smaller denominator.
                _swept = self.sri.sweep_iteration(iteration)
                if _swept:
                    logger.warning(
                        "SRI: swept %d unclosed slot(s) in iteration %d as "
                        "generation_error", _swept, iteration)
                console.print(
                    f"  Iteration {iteration} done: tokens={iter_tokens:,}, time={iter_time:.1f}s, "
                    f"cumulative_tokens={result.total_tokens:,}"
                )

                # 1.5a + N7: an iteration improves if EITHER the archive-best OR
                # the oracle (per-task-best mean) rises by more than a magnitude-
                # aware margin (epsilon * frozen score_range). Tracking the oracle
                # keeps the search alive while different chains are still winning
                # new tasks even though best-mean is flat (the merge payoff). The
                # monotone-max best/oracle update is kept separate from this test.
                if self.archive.frozen_score_range is None:
                    self.archive.freeze_score_range()  # stable after warmup
                tol = self.config.epsilon * max(self.archive.score_range(), 1e-9)
                current_oracle = ptb_mean
                result.oracle_history.append(current_oracle)
                if self._iteration_improved(current_best, prev_best, current_oracle, prev_oracle, tol):
                    console.print(
                        f"  [green]Improved (best {prev_best:.3f}→{current_best:.3f}, "
                        f"oracle {prev_oracle:.3f}→{current_oracle:.3f}, tol={tol:.4f})[/green]"
                    )
                    patience_counter = 0
                    prev_best = max(prev_best, current_best)
                    prev_oracle = max(prev_oracle, current_oracle)
                    logger.info(
                        "  IMPROVED: best=%.3f, oracle=%.3f (tol=%.4f), iter_tokens=%d, iter_time=%.1fs",
                        current_best, current_oracle, tol, iter_tokens, iter_time,
                    )
                else:
                    patience_counter += 1
                    console.print(
                        f"  No improvement (patience {patience_counter}/{self._effective_patience})"
                    )
                    logger.info(
                        "  No improvement: best=%.3f, oracle=%.3f, patience=%d/%d, "
                        "iter_tokens=%d, iter_time=%.1fs",
                        current_best, current_oracle, patience_counter, self._effective_patience,
                        iter_tokens, iter_time,
                    )

                # --- Incremental convergence save (survives crashes) ---
                _persist_progress()

            total_time = time.time() - run_start
            # Both mid-loop breaks (budget / all-at-max-depth) fire after the
            # top-of-loop ``iteration += 1`` but before any breeding, so the
            # completed-generation count is ``iteration - 1`` there and
            # ``iteration`` on the normal patience / max-iterations exits.
            # total_iterations must report completed generations on EVERY exit
            # path (F033 sibling: len(convergence_history) == completed + 1).
            completed_iterations = (
                iteration - 1 if (budget_halted or max_depth_halted) else iteration
            )
            # The budget branch comes FIRST: a budget halt on the final
            # iteration would otherwise be mislabeled by the max_iterations
            # branch — the honest label wins.
            if budget_halted:
                console.print(f"\n[yellow]Stopped after {completed_iterations} completed iterations (daily budget headroom exhausted)[/yellow]")
                logger.info("Terminated: daily budget headroom exhausted at iteration %d", iteration)
            elif iteration >= self.config.max_iterations:
                console.print(f"\n[green]Reached max iterations ({self.config.max_iterations})[/green]")
                logger.info("Terminated: max_iterations=%d reached", self.config.max_iterations)
            elif patience_counter >= self._effective_patience:
                console.print(f"\n[green]Converged after {iteration} iterations (patience exhausted)[/green]")
                logger.info("Terminated: patience exhausted after %d iterations", iteration)
            else:
                console.print(f"\n[green]Stopped after {completed_iterations} completed iterations (all candidates at max depth)[/green]")
                logger.info("Terminated: all candidates at max_depth after %d completed iterations", completed_iterations)
            console.print(f"  Total run time: {total_time:.1f}s ({total_time/60:.1f}min)")
            logger.info(
                "Run complete: iterations=%d, archive=%d, best=%.3f, oracle=%.3f, "
                "tokens=%d, time=%.1fs (%.1fmin)",
                completed_iterations, len(self.archive), self.archive.best_mean_score,
                ptb_mean, result.total_tokens, total_time, total_time/60,
            )

            # Assemble result
            result.archive_size = len(self.archive)
            result.total_iterations = completed_iterations
            # N9a: tag whether this run actually iterated, so a cross-seed
            # aggregator can skip budget-starved stubs (the ARC/SWE phantom seed
            # dirs). Recomputed at end-of-run → a resumed-then-completed run
            # reports "completed". A run that terminated exactly as configured
            # is "completed" even at zero generations: ``iteration >=
            # max_iterations`` covers the deliberate --max-iterations 0
            # single-draw control, and the max-depth break keeps ``iteration >=
            # 1``. Only a budget halt (or a loop that never ran for another
            # reason) stamps an aborted status.
            if budget_halted:
                result.run_status = (
                    "aborted_mid" if completed_iterations >= 1 else "aborted_pre_iteration"
                )
            elif iteration >= 1 or iteration >= self.config.max_iterations:
                result.run_status = "completed"
            else:
                result.run_status = "aborted_pre_iteration"
            result.best_mean_score = self.archive.best_mean_score
            result.best_candidate_id = (
                self.archive.best_candidate.candidate_id
                if self.archive.best_candidate
                else ""
            )
            result.per_task_best_scores = self.archive.per_task_best_scores()
            ptb = result.per_task_best_scores
            # Warn if any evaluated task is missing from per_task_best_scores —
            # that means every candidate produced a non-finite score for it
            # (e.g. evaluator crash). The oracle mean would otherwise silently
            # average over a smaller set than the user expects.
            missing_tasks = [t.task_id for t in tasks if t.task_id not in ptb]
            if missing_tasks:
                logger.warning(
                    "Oracle: %d/%d tasks have no finite-scored candidate "
                    "(treating as 0.0 in oracle_mean_score): %s",
                    len(missing_tasks), len(tasks), missing_tasks[:10],
                )
            # Average over the full task set, counting missing tasks as 0.0,
            # so the oracle is an honest "best-across-the-archive per task" mean
            # rather than an inflated average over only the tasks that succeeded.
            result.oracle_mean_score = (
                sum(ptb.get(t.task_id, 0.0) for t in tasks) / len(tasks)
                if tasks else 0.0
            )

            # 4.1: build the deployable oracle by assembly (Ω_merge) — property-
            # gated, end-of-run, no LLM call / no re-eval. When it fires, the
            # synthesized candidate becomes the archive best (mean == oracle), so
            # the reported deployable number closes the linear-vs-oracle gap.
            merged = self._build_merged_candidate(
                tasks, result, force=self.config.consolidate,
            )
            if merged is not None:
                result.archive_size = len(self.archive)
                result.best_mean_score = self.archive.best_mean_score
                if self.archive.best_candidate is not None:
                    result.best_candidate_id = self.archive.best_candidate.candidate_id
                result.merge_candidate_id = merged.candidate_id
                # Persistence contract: summary.json / archive index reference
                # this candidate, so it must have a candidate dir on disk or a
                # --resume rebuild would silently drop the reported best.
                try:
                    self._save_candidate_incremental(merged, out_dir)
                except Exception:
                    logger.warning(
                        "Failed persisting merge candidate (non-fatal)",
                        exc_info=True,
                    )

            # Snapshot the LLMClient's cumulative usage to populate the outer
            # input/output split. The .total_tokens we've been accumulating
            # above is per-call totals returned from complete(); this read
            # supplies the prompt/completion breakdown the API returned. Both
            # numbers should agree on the total — log a warning if they drift.
            try:
                cu = getattr(self.llm_client, "cumulative_usage", None) or {}
                result.outer_prompt_tokens = int(cu.get("prompt", 0) or 0)
                result.outer_completion_tokens = int(cu.get("completion", 0) or 0)
                result.outer_calls = int(cu.get("calls", 0) or 0)
                cu_total = int(cu.get("total", 0) or 0)
                if cu_total and result.total_tokens and abs(cu_total - result.total_tokens) > 0:
                    logger.info(
                        "Outer-token reconciliation: cumulative_usage.total=%d vs "
                        "result.total_tokens=%d — using cumulative_usage for the "
                        "prompt/completion split, total_tokens unchanged.",
                        cu_total, result.total_tokens,
                    )
            except Exception:
                logger.exception("Failed reading LLMClient.cumulative_usage")

            result.archive_data = self.archive.to_dict()

            # External-agent run-level telemetry rollup (plan §7.8). ``None`` for
            # legacy / builtin runs, so summary.json is byte-for-byte unchanged.
            try:
                result.agent_telemetry_rollup = self._run_level_agent_telemetry_rollup()
            except Exception:
                logger.warning(
                    "Failed building run-level agent telemetry rollup (non-fatal)",
                    exc_info=True,
                )

            # Test-set evaluation (oracle per-task best solutions)
            console.print("\n[bold blue]═══ Test Set Evaluation (oracle per-task best) ═══[/bold blue]")
            logger.info("Starting test-set evaluation (oracle)")
            test_start = time.time()
            test_scores = await self._run_test_evaluation(tasks)
            test_time = time.time() - test_start
            if test_scores:
                result.test_scores = test_scores
                result.test_mean_score = sum(test_scores.values()) / len(test_scores)
                console.print(f"  Oracle test mean: {result.test_mean_score:.3f} ({test_time:.1f}s)")
                logger.info(
                    "Oracle test evaluation complete: mean=%.3f, tasks=%d, time=%.1fs",
                    result.test_mean_score, len(test_scores), test_time,
                )
            else:
                console.print("  [yellow]Skipped (test evaluation not available for this executor)[/yellow]")
                logger.info("Test evaluation skipped (no adapter support)")

            # Test-set evaluation (single best chain candidate)
            if self.archive.best_candidate:
                console.print("\n[bold blue]═══ Test Set Evaluation (single best chain) ═══[/bold blue]")
                logger.info("Starting test-set evaluation (chain)")
                chain_start = time.time()
                chain_test_scores = await self._run_chain_test_evaluation(tasks)
                chain_time = time.time() - chain_start
                if chain_test_scores:
                    result.chain_test_scores = chain_test_scores
                    result.chain_test_mean_score = (
                        sum(chain_test_scores.values()) / len(chain_test_scores)
                    )
                    console.print(
                        f"  Chain test mean: {result.chain_test_mean_score:.3f} ({chain_time:.1f}s)"
                    )
                    logger.info(
                        "Chain test evaluation complete: mean=%.3f, candidate=%s, time=%.1fs",
                        result.chain_test_mean_score,
                        self.archive.best_candidate.candidate_id,
                        chain_time,
                    )
                else:
                    console.print("  [yellow]Skipped (test evaluation not available)[/yellow]")
        finally:
            # Always sweep the external-agent run guard so a SIGINT /
            # KeyboardInterrupt (or any mid-run failure) still kills the OH
            # agent-server and any in-flight Docker leases, and tears down
            # their scratch dirs (plan §6.7). No-op for the legacy path
            # (the guard is None and is never leased).
            if self._run_guard is not None:
                try:
                    await self._run_guard.shutdown_sweep()
                except Exception:
                    logger.warning(
                        "External-agent run-guard shutdown sweep failed "
                        "(non-fatal)", exc_info=True,
                    )
            # Clean up the file handler to avoid leaks, and restore the
            # caller's logger level (see the setLevel above).
            root_logger.removeHandler(file_handler)
            file_handler.close()
            root_logger.setLevel(prev_level)

        return result

    def _uses_external_spine(self) -> bool:
        """Whether THIS run drives the external-agent spine rather than the legacy
        native path.

        The spine is used when ``base_solver`` is a genuine external kind
        (``openhands`` / ``terminus2`` — anything other than ``None`` / ``builtin``),
        OR when ``base_solver == "builtin"`` AND the bound adapter advertises that it
        wants ``builtin`` on the spine (``advertises_spine_builtin()`` — only the
        terminal_bench adapter does, where the native executor cannot run the legacy
        task layout). ``base_solver in (None, "builtin")`` with a NON-advertising
        adapter (CO-Bench, text-classification, …) stays on the legacy native path,
        byte-for-byte unchanged — so CO-Bench's ``builtin`` baseline never regresses.

        This single predicate replaces the scattered
        ``base_solver not in (None, "builtin")`` guards, so the seed / evaluate /
        gate / solver-build dispatch all agree on one routing rule. The rule
        itself lives in :func:`meta_n.core.spine_routing.uses_external_spine`
        (main.py's startup guards call the same function — one routing rule,
        two entry points).

        Returns:
            ``True`` if the spine + ``ExternalAgentSolver`` should drive this run.
        """
        return uses_external_spine(self.config.base_solver, self.adapter)

    def _init_external_agent_spine(self) -> None:
        """Construct the shared external-agent guards + telemetry (plan §6.7).

        Builds, exactly once per orchestrator, the three collaborators every
        :class:`ExternalAgentSolver` shares by reference:

        * ``self._run_guard`` — a :class:`DockerRunGuard` bounding the inner
          Docker concurrency to ``min(config.max_docker or min(parallel, 4),
          parallel)`` (clamped ``<= parallel`` per §6.4) under
          ``config.scratch_root``.
        * ``self._cost_guard`` — a :class:`CostGuard` adapter onto the *single*
          :class:`CostTracker` ledger the outer LLM client already uses, priced
          for the resolved inner model.
        * ``self._agent_telemetry`` — an :class:`AgentTelemetry` writer that
          creates ``<output_dir>/telemetry/`` and writes ``schema.json`` once.

        Refuses to start an external run without a cost ledger: external agents
        spend through their own LLM client, so a missing ``--daily-budget-usd``
        would leave that spend uncapped (plan §4.7, §6.7).

        Raises:
            RuntimeError: If ``config.base_solver`` is external but the LLM
                client has no ``cost_tracker`` (``--daily-budget-usd`` was not
                set), or no ``output_dir`` was configured.
        """
        # Lazy import: the external_agents spine is dependency-free (no openhands
        # / terminal_bench / docker), but we still defer it so the legacy path
        # never imports it at module load.
        from meta_n.core.external_agents.budget import CostGuard
        from meta_n.core.external_agents.concurrency import DockerRunGuard
        from meta_n.core.external_agents.telemetry import AgentTelemetry

        cost_tracker = getattr(self.llm_client, "cost_tracker", None)
        if cost_tracker is None:
            raise RuntimeError(
                "base_solver=%r requires a cost ledger so the external agent's "
                "own LLM spend is capped, but the LLM client has no "
                "cost_tracker. Launch with --daily-budget-usd > 0 (plan §4.7)."
                % self.config.base_solver
            )
        output_dir = self.config.output_dir
        if not output_dir:
            raise RuntimeError(
                "base_solver=%r requires an --output-dir so the telemetry tree "
                "(telemetry/agent_runs.jsonl + schema.json) can be written."
                % self.config.base_solver
            )

        # Inner Docker concurrency: default min(parallel, 4); always clamped to
        # <= parallel (only ``parallel`` tasks are ever in flight, so a higher
        # inner cap is unreachable — plan §6.4). Warn loudly if the operator
        # asked for more than parallel: they almost certainly want --parallel up.
        parallel = max(1, int(self.config.parallel))
        requested = self.config.max_docker or min(parallel, 4)
        if self.config.max_docker and self.config.max_docker > parallel:
            console.print(
                f"[yellow]Warning: --max-docker={self.config.max_docker} > "
                f"--parallel={parallel}; clamping to {parallel}. Only "
                f"{parallel} tasks run at once, so a higher inner Docker cap is "
                f"unreachable — raise --parallel instead.[/yellow]"
            )
            logger.warning(
                "max_docker=%d > parallel=%d; clamped to parallel "
                "(raise --parallel to widen Docker concurrency)",
                self.config.max_docker, parallel,
            )
        max_docker = min(requested, parallel)

        self._run_guard = DockerRunGuard(
            max_docker=max_docker, scratch_root=self.config.scratch_root
        )
        model = getattr(getattr(self.llm_client, "config", None), "model", "") or ""
        self._cost_guard = CostGuard(cost_tracker, model=model)
        self._agent_telemetry = AgentTelemetry(output_dir)
        logger.info(
            "External-agent spine ready: base_solver=%s, max_docker=%d, "
            "model=%s, telemetry=%s/telemetry",
            self.config.base_solver, max_docker, model, output_dir,
        )

    def _resolve_inner_backend_kwargs(self) -> dict:
        """Resolve the inner provider env (model / api_base / api_key / var name).

        Mirrors meta-n's outer :class:`LLMClient` resolution so the inner agent
        (litellm / LiteLLM) authenticates against the *same* provider with the
        *same* key (plan §7.10). Prefers the adapter's
        ``resolve_inner_provider_env(config) -> (var_name, var_value, base_url)``
        helper when present (planned for ``terminal_bench.py`` in Phase-1 — no
        adapter ships it in Phase-0); otherwise falls back to the resolved
        :class:`LLMConfig` directly.

        Returns:
            A kwargs dict forwarded verbatim to ``adapter.make_agent_backend``:
            ``{"model", "api_base", "api_key", "provider_env_var"}``.
        """
        llm_config = getattr(self.llm_client, "config", None)
        model = getattr(llm_config, "model", "") or ""
        api_base = getattr(llm_config, "base_url", None)
        api_key = getattr(llm_config, "api_key", None)
        provider_env_var = None

        resolver = getattr(self.adapter, "resolve_inner_provider_env", None)
        if callable(resolver):
            try:
                var_name, var_value, base_url = resolver(llm_config)
                provider_env_var = var_name
                if var_value:
                    api_key = var_value
                if base_url:
                    api_base = base_url
            except Exception:  # noqa: BLE001 - fall back to the raw config
                logger.warning(
                    "adapter.resolve_inner_provider_env failed; falling back to "
                    "the raw LLM config for inner backend auth",
                    exc_info=True,
                )

        return {
            "model": model,
            "api_base": api_base,
            "api_key": api_key,
            "provider_env_var": provider_env_var,
        }

    def _build_external_solver(self, candidate: Candidate):
        """Build an :class:`ExternalAgentSolver` for ``candidate`` (plan §2.3).

        Wires the adapter's per-benchmark factory hooks (backend / env provider /
        scorer) to the shared run/cost guards and telemetry. ``injected_codes``
        is the candidate's full chain (``[]`` for the gen0 vanilla baseline), so
        the injection mapper folds Ω's libraries onto the agent's prompt + staged
        helper files. The per-run telemetry coordinates (``generation`` /
        ``candidate_id``) ride in ``run_ctx`` so each ``AgentRunRecord`` lands on
        the right archive coordinate.

        Raises:
            RuntimeError: If the adapter does not provide the requested backend /
                env provider / scorer (i.e. this benchmark does not support the
                external base_solver).
        """
        from meta_n.core.external_agents import ExternalAgentSolver

        kind = self.config.base_solver
        if self.adapter is None:
            raise RuntimeError(
                f"base_solver={kind!r} requires a benchmark adapter exposing the "
                f"external-agent factory hooks, but the executor wraps none."
            )

        backend_kw = self._resolve_inner_backend_kwargs()
        # The terminal_bench 'builtin' control authors its bash script on the
        # meta-n side via the native Layer1Solver (one outer LLM call), so the
        # adapter's BuiltinTBBackend needs that solver + the outer LLM client (for
        # outer-token attribution). Other kinds run their own inner agent and read
        # neither. Pass them as additive kwargs; non-builtin backends ignore extras
        # they do not accept, but only builtin's factory declares them — so scope
        # the extras to the builtin kind to avoid an unexpected-kwarg TypeError.
        if kind == "builtin":
            backend_kw = {
                **backend_kw,
                "solver": self.solver,
                "llm_client": self.llm_client,
                "solver_language": self.solver_language,
            }
        backend = self.adapter.make_agent_backend(kind, **backend_kw)
        env_provider = self.adapter.make_env_provider(kind)
        scorer = self.adapter.make_scorer(kind)
        if backend is None or env_provider is None or scorer is None:
            raise RuntimeError(
                f"adapter {type(self.adapter).__name__} does not support "
                f"base_solver={kind!r} (backend={backend is not None}, "
                f"env_provider={env_provider is not None}, "
                f"scorer={scorer is not None})."
            )

        return ExternalAgentSolver(
            backend=backend,
            env_provider=env_provider,
            scorer=scorer,
            injected_codes=candidate.injected_codes,
            depth=candidate.depth,
            run_guard=self._run_guard,
            telemetry=self._agent_telemetry,
            cost_guard=self._cost_guard,
            adapter=self.adapter,
            solver_language=self.solver_language,
            max_turns=self.config.agentic_max_turns,
            token_budget=self.config.agentic_token_budget,
            time_limit_s=self.config.agentic_time_limit_s,
            max_budget_usd=self.config.agentic_max_budget_usd,
            run_ctx={
                "generation": candidate.iteration,
                "candidate_id": candidate.candidate_id,
                # S0.6 de-reap: the run root so ExternalAgentSolver can snapshot
                # the lease's ``agent_logs`` into ``archive/<cand>/agent_logs/``
                # before the lease is rmtree'd (the transcript pointer otherwise
                # dangles into a scratch dir outside output_dir — the FEAL
                # null-transcript finding).
                "output_dir": self.config.output_dir,
            },
        )

    def _code_library_is_live(self) -> bool:
        """Whether to prepend Ω's Python code_library helpers (roadmap v2 4.3).
        Per-benchmark static flag from the adapter; default True (CO-Bench/SWE
        opt out, where the measured call-rate is ~0).

        Adoption probe: ``force_code_library_live`` short-circuits to True (un-demote)
        so helpers are staged + callable even on families the adapter would demote.
        Default OFF ⇒ the adapter's static flag is honored byte-identically."""
        if self.config.force_code_library_live:
            return True
        try:
            if self.adapter is not None:
                return bool(self.adapter.code_library_is_live())
        except Exception:
            pass
        return True

    def _demote_python_library(
        self, merged_py: dict, injected_codes: "list[InjectedCode]"
    ) -> dict:
        """Zero the merged Python ``code_library`` on a DEMOTING family (the solver
        regenerates code inline, so prepending helpers is dead weight — CO-Bench/SWE).

        Default behavior is identical to the legacy ``merged_py = {}`` (return an
        empty dict, silently). Callers must guard this with ``not
        _code_library_is_live()`` exactly as before — when the family IS live this
        helper is never called, so the live path is byte-identical.

        T3.3/T3.4 (audit): the zeroing silently NO-OPs two USER-REQUESTED channels:
        a ``--seed-code-library`` seed (``source_depth==0``) and a
        ``--verified-code``-KEPT helper are both discarded here unless
        ``--force-code-library-live`` is set. So when either flag is on:
          * ``source_depth==0`` seed contributions are EXEMPTED (kept live) — a
            seeded known-good helper must survive to be tested at all;
          * any OTHER helper actually zeroed is named in a loud warning so the
            no-op surfaces (a verified-KEPT helper at depth>=2 still gets zeroed on
            a demoting family — pass ``--force-code-library-live`` to keep it).
        Neither flag set ⇒ the legacy silent ``{}`` (byte-identical).
        """
        if not (self.config.verified_code or self.config.seed_code_library):
            return {}
        exempt: dict[str, str] = {}
        for ic in (injected_codes or []):
            if getattr(ic, "source_depth", None) == 0:
                exempt.update(ic.code_library or {})
        kept = {k: exempt[k] for k in merged_py if k in exempt}
        overridden = [k for k in merged_py if k in exempt and merged_py[k] != exempt[k]]
        zeroed = [k for k in merged_py if k not in exempt]
        if zeroed:
            logger.warning(
                "code_library DEMOTED (family not code_library_is_live): zeroing "
                "%d Python helper(s) %s that --verified-code/--seed-code-library "
                "staged — they will NOT be prepended or callable. Pass "
                "--force-code-library-live to keep the live-helper channel on this "
                "family.",
                len(zeroed), sorted(zeroed),
            )
        if overridden:
            logger.warning(
                "code_library DEMOTED (seed/Omega name collision): %d Omega override(s) "
                "%s share a name with a --seed-code-library helper on a non-live family; "
                "staging the seed's known-good source instead of the Omega override. Pass "
                "--force-code-library-live to keep Omega's override.",
                len(overridden), sorted(overridden),
            )
        return kept

    # ------------------------------------------------------------------ #
    # Forensic improvement #2 — VERIFIED code_library + FORCED ADOPTION.  #
    # All gated on ``self.config.verified_code`` (default OFF). When OFF, #
    # _verify_and_filter_code_library is an identity returning the SAME   #
    # InjectedCode object (no sandbox, no copy) and the non-adoption       #
    # penalty is never consulted, so every surface is byte-identical.     #
    # ------------------------------------------------------------------ #

    def _make_heldout_verifier(self):
        """Resolve the held-out verifier for the verify-gate.

        Prefers the bound adapter's ``make_heldout_verifier()`` (a real
        SandboxedHeldoutVerifier on families with a value-oracle); falls back to
        the capability-preserving :class:`StubHeldoutVerifier` (KEEP + flag
        UNVERIFIED) when the adapter has no held-out harness. Only ever called when
        ``verified_code`` is ON.
        """
        if self.adapter is not None:
            try:
                verifier = self.adapter.make_heldout_verifier()
                if verifier is not None:
                    return verifier
            except Exception:
                logger.warning(
                    "verified_code: adapter.make_heldout_verifier() raised — "
                    "falling back to StubHeldoutVerifier",
                    exc_info=True,
                )
        # T2.2 (audit): no family-specific value-oracle ⇒ the DROP/verify gate is
        # INERT. StubHeldoutVerifier KEEPS every Ω helper (ran_in_sandbox=False /
        # UNVERIFIED) while forced adoption + the non-adoption penalty still fire,
        # so an UNVERIFIED helper is force-injected as if verified. Only co_bench
        # (crew) overrides make_heldout_verifier with a real SandboxedHeldoutVerifier
        # today. Warn (not INFO) so the inert gate surfaces on every other family.
        logger.warning(
            "verified_code: no held-out value-oracle for this family — using "
            "StubHeldoutVerifier (KEEP-ALL, ran_in_sandbox=False). The DROP/verify "
            "gate is INERT: no Ω helper is sandbox-checked or dropped; forced "
            "adoption still fires. Only co_bench (crew) ships a real "
            "SandboxedHeldoutVerifier.",
        )
        return StubHeldoutVerifier()

    async def _verify_and_filter_code_library(
        self,
        injected: InjectedCode,
        context_codes: "list[InjectedCode]",
        tasks: "list[TaskDescription]",
    ) -> "tuple[InjectedCode, set[str]]":
        """VERIFY-THEN-INJECT: keep only Ω Python helpers that pass a held-out check.

        Default OFF / no Python helpers ⇒ returns the SAME ``injected`` object
        UNCHANGED (no sandbox call, no copy, no log) so the candidate built
        downstream is byte-identical to HEAD. When ON, each helper is sandbox-
        executed (``--network none``) against the resolved held-out verifier and
        kept ONLY if it passes; the rest are dropped. ``pre_process`` and bash
        helpers always pass through. Returns ``(injected, sandbox_verified_names)``
        where the first element is a NEW ``InjectedCode`` (the omega object is
        replaced — never an archived candidate) only if something was dropped (else
        the original object untouched), and ``sandbox_verified_names`` (T3.2) is the
        subset of KEPT helper names that ACTUALLY ran in the ``--network none``
        sandbox (``ran_in_sandbox==True``) — i.e. genuinely verified, NOT merely
        stub-kept. The default OFF path returns ``(injected, set())``.
        """
        if not self.config.verified_code:
            return injected, set()
        if not injected.code_library:
            return injected, set()
        verifier = self._make_heldout_verifier()
        context_sources: list[str] = []
        for ic in (context_codes or []):
            context_sources.extend((ic.code_library or {}).values())
        task_id = tasks[0].task_id if tasks else ""
        kept: dict[str, str] = {}
        sandbox_verified: set[str] = set()
        for name, source in injected.code_library.items():
            # Match the verify candidate to what deploy ships: prepend_python_library
            # concatenates ALL of this injection's helpers, so an entry-point helper
            # that calls a sibling must see that sibling in the container file too.
            sib = [
                src for other, src in injected.code_library.items()
                if other != name
            ]
            try:
                res = verifier.verify(name, source, task_id, context_sources + sib)
            except Exception:
                logger.warning(
                    "verified_code: verifier.verify crashed for helper '%s' — "
                    "dropping (fail-closed)", name, exc_info=True,
                )
                continue
            if res.passed:
                kept[name] = source
                if res.ran_in_sandbox:
                    sandbox_verified.add(name)
                logger.info(
                    "verified_code: KEEP helper '%s' (ran_in_sandbox=%s) — %s",
                    name, res.ran_in_sandbox, res.evidence[:160],
                )
            else:
                logger.info(
                    "verified_code: DROP helper '%s' — failed held-out check "
                    "(ran_in_sandbox=%s) — %s",
                    name, res.ran_in_sandbox, res.evidence[:160],
                )
        # Dependency-aware keep: deploy prepends every KEPT helper together, so a
        # kept helper that name-calls a sibling needs that sibling present or it
        # NameErrors at runtime. Add back the transitive closure of injection
        # helpers name-called by any kept helper — even ones that individually
        # failed the single-entry-point oracle. A depended-upon sibling was never
        # verified AS an entry point, so it is NOT added to sandbox_verified (the
        # T3.2 penalty still bars per-task-best only on genuinely-verified names);
        # genuinely-dead helpers (called by nothing kept) stay dropped.
        from meta_n.core.adoption import scan_helper_calls

        all_names = list(injected.code_library)
        frontier = list(kept)
        while frontier:
            src = injected.code_library[frontier.pop()]
            called, _ = scan_helper_calls(src, all_names)
            for dep in called:
                if dep not in kept:
                    kept[dep] = injected.code_library[dep]
                    frontier.append(dep)
        # Stamp the sandbox-verified set onto the returned injection so the newest
        # layer carries its own genuinely-verified names into the child chain (the
        # override-aware penalty aggregation reads this per layer). Only reached on
        # the ON path with a non-empty library — the default-OFF early-returns above
        # stay unstamped, so that field stays default-empty and OFF is byte-identical.
        if kept == dict(injected.code_library):
            injected.sandbox_verified_names = sorted(sandbox_verified)
            return injected, sandbox_verified  # nothing dropped ⇒ identical object
        return injected.model_copy(update={
            "code_library": kept,
            "sandbox_verified_names": sorted(sandbox_verified),
        }), sandbox_verified

    def _nonadopting_verified_tasks(
        self,
        candidate: Candidate,
        injected: InjectedCode,
        sandbox_verified_names: "set[str] | None" = None,
    ) -> set:
        """Task ids whose fresh trace advertised a verified helper but DID NOT call it.

        The inline-re-derivation penalty: when ``verified_code`` is ON, a trace
        that had a verified helper STAGED (``utilities_available``) but re-derived
        it inline instead of calling it (``utilities_called`` excludes every
        verified name) is barred from updating per-task-best. Reuses the existing
        adoption signal (``populate_adoption_fields`` / ``scan_helper_calls``); a
        ``None`` ``utilities_called`` means UNMEASURABLE (demoted/no live helpers)
        and is never penalized.

        T3.2 (audit): ``sandbox_verified_names`` is the set of helper names that
        ACTUALLY ran in the sandbox (``ran_in_sandbox==True``). When supplied for a
        layer, its verified set is restricted to ``code_library ∩
        sandbox_verified_names`` so a STUB-KEPT (``ran_in_sandbox==False``)
        UNVERIFIED helper never bars per-task-best — a helper we never actually
        verified must not gate a real win. ``None`` (legacy callers / existing
        tests) ⇒ that layer treats every code_library key as verified.

        The LIVE verified set spans the FULL merged chain, not just the newest
        layer: a prior-generation helper that is still staged in the library the
        child solver saw must bar too. Layers are folded in order with
        last-writer-wins override semantics (mirroring ``merge_code_libraries``):
        for each helper a layer (re)defines, ADD it iff that layer verified it,
        else DISCARD it (a later unverified/stub redefinition clears an earlier
        verification; a prior helper a later layer never touches stays verified as
        its still-staged prior version). The newest layer (object-identity ``is
        injected`` at the real call site, or supplied via ``injected`` when
        ``candidate.injected_codes`` omits it — ad-hoc unit callers) uses the
        freshly-passed ``sandbox_verified_names``; every other layer uses its own
        stamped ``sandbox_verified_names``.
        """
        verified: set[str] = set()

        def _fold(ic: "InjectedCode", verified_names: "set[str] | None") -> None:
            # verified_names is None ⇒ this layer's every code_library key counts
            # as verified (legacy newest-only behavior for callers with no set).
            for name in (ic.code_library or {}):
                if verified_names is None or name in verified_names:
                    verified.add(name)
                else:
                    verified.discard(name)

        injected_in_chain = False
        for ic in candidate.injected_codes:
            if ic is injected:
                injected_in_chain = True
                _fold(ic, sandbox_verified_names)
            else:
                _fold(ic, set(ic.sandbox_verified_names))
        if not injected_in_chain:
            # Ad-hoc / unit callers pass the newest layer only via ``injected``
            # (candidate.injected_codes omits it); fold it last as the chain top.
            _fold(injected, sandbox_verified_names)

        if not verified:
            return set()
        barred: set = set()
        for trace in candidate.traces:
            called = trace.utilities_called
            if called is None:  # unmeasurable — never penalize
                continue
            # Liveness/staging is gated by the trace's utilities_available (empty
            # on the demoted CO-Bench/SWE path ⇒ no bars). Bar iff any STAGED
            # verified helper was re-derived inline (advertised but not called);
            # calling a sibling verified helper does not excuse ignoring another.
            advertised = verified & set(trace.utilities_available or [])
            if advertised - set(called):
                barred.add(trace.task_id)
        return barred

    @staticmethod
    def _iteration_improved(current_best, prev_best, current_oracle, prev_oracle, tol):
        """1.5a + N7: an iteration counts as improving iff the archive-best OR
        the oracle (per-task-best mean) rose by more than the magnitude-aware
        margin ``tol`` (= epsilon * score_range). Tracking the oracle keeps the
        search alive while different chains still win new tasks though best-mean
        is flat. On single-task families oracle == best, so this reduces to the
        best-only test; on a multi-task binary scale the oracle CAN rise while
        best stays flat (disjoint per-task wins) — exactly the case oracle
        tracking exists to catch."""
        return ((current_best - prev_best) > tol) or ((current_oracle - prev_oracle) > tol)

    def _build_solver_from_candidate(self, candidate: Candidate):
        """Reconstruct the solver chain from a candidate's injected_codes."""
        # External base_solver (openhands / terminus2, plus terminal_bench
        # 'builtin'): the per-candidate solver is a self-contained external agent
        # wrapped in ExternalAgentSolver, built from the adapter's factory hooks +
        # the shared guards. This branch sits ABOVE the use_agentic / MetaLayer
        # logic so it fully replaces the native solver for the external path.
        # ``None`` and a NON-advertised ``"builtin"`` (CO-Bench etc.) fall through
        # to the legacy native path unchanged (plan §2.3, §8.1).
        if self._uses_external_spine():
            return self._build_external_solver(candidate)
        if self.config.use_agentic:
            from meta_n.core.agentic_solver import AgenticSolver

            # R3-A: mirror the native branch (:2210-2211) so a --seed-code-library
            # (source_depth==0) seed / --verified-code-KEPT helper survives the demote
            # on a non-live family instead of being blanket-zeroed. None (live family)
            # => AgenticSolver skips its demote branch (full library kept); default
            # demoting family with no seed/verified flag => _demote_python_library
            # early-returns {} (no warning) => byte-identical to the legacy blanket zero.
            code_library_is_live = self._code_library_is_live()
            demoted_code_library = None
            if not code_library_is_live:
                merged_py, _ = merge_code_libraries(candidate.injected_codes)
                demoted_code_library = self._demote_python_library(
                    merged_py, candidate.injected_codes
                )
            return AgenticSolver(
                llm_client=self.llm_client,
                executor=self.executor,
                injected_codes=candidate.injected_codes,
                solver_language=self.solver_language,
                max_turns=self.config.agentic_max_turns,
                token_budget=self.config.agentic_token_budget,
                # F063 (§6b): real-spend cap; None (default) = off = byte-identical.
                spend_budget=self.config.agentic_spend_budget,
                # F060 (§6b): solve-time temperature; default 0.7 == the
                # historical constructor pin (byte-identical when unset).
                temperature=self.config.agentic_temperature,
                depth=candidate.depth,
                code_library_is_live=code_library_is_live,
                demoted_code_library=demoted_code_library,
                agentic_error_hints=self.config.agentic_error_hints,
                agentic_preamble=self.config.agentic_preamble,
                # Y4-C_callability-5 (R3-E completion): thread the E3 ablation
                # flag so the agentic pre_process chain honors it too — it
                # changes behavior only at depth>=3 (>=2 injected blocks, where
                # a shallower block could see a deeper block's emission as
                # outer_context). Default False == the ctor default, so the
                # default path is byte-identical.
                no_outer_context=self.config.no_outer_context,
            )
        solver = self.solver  # Layer1Solver
        n = len(candidate.injected_codes)
        merged_py, merged_bash = merge_code_libraries(candidate.injected_codes)
        # 4.3: per-benchmark gate — demote dead Python helpers where the solver
        # regenerates code (CO-Bench/SWE). Bash helpers + live families keep.
        # T3.3/T3.4: _demote_python_library is the legacy ``{}`` by default but,
        # under --verified-code/--seed-code-library, exempts source_depth==0 seeds
        # and warns about any helper it zeroes (surfacing the silent no-op).
        if not self._code_library_is_live():
            merged_py = self._demote_python_library(merged_py, candidate.injected_codes)
        for i, injected_code in enumerate(candidate.injected_codes):
            is_outermost = (i == n - 1)
            solver = MetaLayer(
                depth=i + 2,
                injected_code=injected_code,
                inner_solver=solver,
                executor=self.executor,
                merged_code_library=merged_py if is_outermost else None,
                merged_code_library_bash=merged_bash if is_outermost else None,
                max_retries=self.config.max_retries if is_outermost else 0,
                retry_threshold=self.config.retry_threshold if is_outermost else 0.5,
                solver_language=self.solver_language,
                no_outer_context=self.config.no_outer_context,
                # Forensic #2 FORCED ADOPTION: verified_code forces the foster
                # affordance ON (the solver MUST call the verified helper). When
                # verified_code is OFF this reduces to ``self.config.foster_adoption``
                # exactly (``X or False == X``), so the rendered prompt is identical.
                foster_adoption=self.config.foster_adoption or self.config.verified_code,
                # P1a DEPLOY FALLBACK — set ONLY on the outermost layer (the one
                # holding the merged verified library). Default False everywhere
                # else ⇒ byte-identical. Inert unless merged_code_library is
                # non-empty (live), so the CO-Bench demoted path no-ops.
                deploy_verified_code=(
                    self.config.deploy_verified_code if is_outermost else False
                ),
            )
        return solver

    def _crn_seed(self, task: TaskDescription, repeat_index: int) -> int | None:
        """Per-(task, repeat) Common-Random-Numbers seed for paired eval.

        Returns ``None`` when ``paired_eval`` is OFF (the default) so NO seed
        reaches the solver / wire — byte-identical to the legacy path. When ON,
        returns ``stable_crn_seed(config.seed, task.task_id, repeat_index)``: a
        pure deterministic hash with NO candidate input, so parent and child
        evaluating the same ``(task, repeat_index)`` get the SAME seed.

        Purity matters: this must NOT draw from ``self.rng`` (that would perturb
        Ω trace / gate sampling and break ``--resume`` via the checkpointed RNG
        state) and must be derived per-call inside the eval loop (never via
        instance/global state — the ``asyncio.gather`` fan-out below would race
        and cross-contaminate the pairing).
        """
        if not self.config.paired_eval:
            return None
        return stable_crn_seed(self.config.seed, task.task_id, repeat_index)

    async def _eval_solve_once(
        self, solver, task: TaskDescription, candidate: Candidate,
        *, seed: int | None = None,
    ):
        """One full evaluation solve of ``task`` → ``(trace, tokens)``.

        Mirrors the _evaluate_candidate dispatch: a spine/agentic/deep candidate
        goes through ``execute()``; a native depth-1 builtin goes through
        ``solve()`` + the executor. Factored so per-candidate repeated eval
        (``eval_repeats``) can invoke it R times to denoise the per-task score.

        ``seed`` (CRN / paired eval): when not ``None``, threaded into the native
        depth-1 ``solver.solve(task, seed=seed)`` so a seed-honouring backend
        correlates sampler luck across candidates. FIRST-CUT GAP (documented):
        the deep / agentic / external-spine branch returns earlier via
        ``execute()`` and is left UNSEEDED — widening the ``execute()`` contract
        is deferred follow-up (F1/F2). The motivating CO-Bench A/B is native
        depth-1, so this is sufficient for the stated measurement.
        """
        if (
            self.config.use_agentic
            or candidate.depth > 1
            or self._uses_external_spine()
        ):
            return await solver.execute(task)
        start = time.time()
        script, reasoning, tokens = await solver.solve(task, seed=seed)
        trace = await self.executor.execute(script, task)
        trace.depth = 1
        trace.reasoning = reasoning
        trace.duration_s = time.time() - start
        # S0.2: native depth-1 fresh-trace finalize site — one executor.execute.
        # The helper scan is gated on live helpers inside the populator; a
        # depth-1 candidate has no injected_codes (and CO-Bench demotes the
        # Python library), so utilities_called stays None there.
        merged_py, merged_bash = merge_code_libraries(candidate.injected_codes)
        if not self._code_library_is_live():
            # T3.3/T3.4: legacy ``{}`` by default; under verified/seed flags exempts
            # source_depth==0 seeds + warns. A depth-1 candidate has no
            # injected_codes, so this is the empty ``{}`` (byte-identical) here.
            merged_py = self._demote_python_library(merged_py, candidate.injected_codes)
        populate_adoption_fields(
            trace,
            command_count=1,
            merged_code_library=merged_py,
            merged_code_library_bash=merged_bash,
            executor=self.executor,
        )
        return trace, tokens

    @staticmethod
    def _sum_inner_traces(sample_traces) -> dict[str, int]:
        """Sum the inner-LLM token/call fields across an iterable of traces.

        Shared by the full-eval reuse/median accounting in
        ``_evaluate_candidate`` (H7: under ``eval_repeats>1`` the candidate's
        inner-token totals must cover ALL R solves' inner spend, not just the
        single median trace) and the gate-FAIL inner-token reconciliation
        (reaudit #6): a gate-rejected candidate is never evaluated, so its gate
        solves' inner_tokens would otherwise vanish from ``result.inner_total``.
        For a single-element / empty iterable this equals the trivial field sum,
        so every default path is byte-identical.
        """
        return {
            "inner_tokens": sum(
                int(getattr(t, "inner_tokens", 0) or 0) for t in sample_traces
            ),
            "inner_prompt_tokens": sum(
                int(getattr(t, "inner_prompt_tokens", 0) or 0)
                for t in sample_traces
            ),
            "inner_completion_tokens": sum(
                int(getattr(t, "inner_completion_tokens", 0) or 0)
                for t in sample_traces
            ),
            "inner_calls": sum(
                int(getattr(t, "inner_calls", 0) or 0) for t in sample_traces
            ),
        }

    async def _evaluate_candidate(
        self,
        candidate: Candidate,
        solver,
        tasks: list[TaskDescription],
        precomputed: "dict[str, Trace] | None" = None,
        *,
        precomputed_frozen: bool = False,
        crn_repeat_offset: int = 0,
    ) -> Candidate:
        """Run all tasks through the solver, populate candidate with results.

        ``precomputed`` (1.6): traces already produced by the gate check, keyed
        by task_id. A task with a precomputed trace is NOT re-solved — its gate
        trace is reused (outer tokens counted as 0 to avoid double-counting; the
        gate already executed it). The gate samples a random subset, so this is a
        MERGE — the remaining tasks are solved normally.

        ``precomputed_frozen`` (F035, §6b): True when ``precomputed`` is the
        consolidation inherit-frozen map, which must never be re-solved — it
        exempts those traces from the ``eval_repeats_gate_topup`` top-up below.

        crn_repeat_offset: base CRN repeat-index offset added to every _crn_seed
        call so distinct _evaluate_candidate invocations (e.g. regression-guard
        base-floor resamples) draw DISTINCT paired-eval seeds instead of colliding
        on index 0. Default 0 -> byte-identical for every existing caller.
        """
        n = len(tasks)
        sem = asyncio.Semaphore(self.config.parallel)
        completed = 0

        async def _solve_one(task: TaskDescription) -> tuple[Trace, int, dict]:
            nonlocal completed
            if precomputed and task.task_id in precomputed:
                reused = precomputed[task.task_id]
                # F035 (§6b): flag-gated gate-trace top-up. OFF (default), a
                # frozen consolidation map, or eval_repeats==1 keep the reuse
                # short-circuit below byte-identical (return without the sem).
                repeats = max(1, self.config.eval_repeats)
                topup = (
                    self.config.eval_repeats_gate_topup
                    and not precomputed_frozen
                    and repeats > 1
                )
                if not topup:
                    completed += 1
                    # T-R2.1: this gate-passing task is reused WITHOUT re-solving, so
                    # its only telemetry row is the gate-phase one that fair_comparison
                    # drops by default — re-stamp it as eval so the single physical run
                    # counts once in the per-agent A/B table. Spine-only (native paths
                    # emit no telemetry); a no-op when the precomputed trace did not
                    # come from a gate run (e.g. consolidation reuse — no cached row).
                    if self._uses_external_spine() and self._agent_telemetry is not None:
                        self._agent_telemetry.restamp_reused_gate(task, solver)
                    console.print(
                        f"      [{completed}/{n}] {task.task_id} "
                        f"[dim](reused gate trace, score={reused.score:.3f})[/dim]"
                    )
                    return reused, 0, self._sum_inner_traces([reused])
                # F035 flag-ON: the gate trace is sample 0; top up with R-1
                # fresh solves and take the median over the union, so this task
                # is denoised at the same R as every other. Under gate_repeats>1
                # sample 0 is the gate's median-of-gate_R (accepted approximation,
                # see the config docstring).
                if self._uses_external_spine() and self._agent_telemetry is not None:
                    # Sample 0 is the reused gate run — restamp exactly once as
                    # eval (same T-R2.1 contract as the reuse path above).
                    self._agent_telemetry.restamp_reused_gate(task, solver)
                # Top-ups run INSIDE the semaphore (real solves); the reuse path
                # above deliberately returns without acquiring it.
                async with sem:
                    spine = self._uses_external_spine()
                    # Sample 0 contributes 0 OUTER tokens (the gate already
                    # counted them — audit-#29 reconciliation) but its inner
                    # tokens ARE summed into inner_all below (H7: all R samples'
                    # inner spend).
                    samples: list[tuple[Trace, int]] = [(reused, 0)]
                    try:
                        # CRN seeds use repeat indices 1..R-1, matching the
                        # normal repeats branch — sample 0 corresponds to the
                        # gate's CRN index-0 draw (now seeded via _gate_solve_one).
                        # H6: the spine solver is stamped per top-up for distinct
                        # telemetry run_ids.
                        for r in range(1, repeats):
                            if spine:
                                solver.repeat_index = r
                            try:
                                samples.append(
                                    await self._eval_solve_once(
                                        solver, task, candidate,
                                        seed=self._crn_seed(task, r),
                                    )
                                )
                            except Exception as e:
                                # A top-up fault must never discard the
                                # already-measured samples (at minimum the
                                # gate sample): score this task over the
                                # partial union collected so far, with
                                # outer/inner tokens summed over what
                                # actually ran. (CancelledError is a
                                # BaseException and still propagates.)
                                logger.warning(
                                    "Gate top-up sample %d/%d raised for "
                                    "task=%s (candidate=%s): %s — falling "
                                    "back to the %d sample(s) collected so "
                                    "far",
                                    r, repeats - 1, task.task_id,
                                    candidate.candidate_id, e, len(samples),
                                )
                                break
                    finally:
                        if spine:
                            solver.repeat_index = 0
                    # ``samples`` is non-empty by construction — the gate trace
                    # is always sample 0 — so the median pick below can never
                    # raise, and no exception escapes to the gather fault
                    # handler from this branch.
                    tokens = sum(tk for _, tk in samples)  # == top-up outer spend only
                    inner_all = self._sum_inner_traces([tr for tr, _ in samples])
                    ordered = sorted(
                        (tr for tr, _ in samples),
                        key=lambda t: t.score if math.isfinite(t.score) else float("-inf"),
                    )
                    trace = ordered[len(ordered) // 2]  # median-scoring trace
                completed += 1
                # len(samples) - 1 == repeats - 1 unless the fault fallback
                # stopped the top-ups early — report what actually ran.
                console.print(
                    f"      [{completed}/{n}] {task.task_id} "
                    f"[dim](gate sample + {len(samples) - 1} top-up, median={trace.score:.3f})[/dim]"
                )
                return trace, tokens, inner_all
            async with sem:
                # 1.2/1.3: per-candidate repeated eval — median-of-R full solves
                # per task denoises the candidate's per-task score so selection /
                # best-promotion / merge act on signal, not a single noisy draw.
                # R=1 (default) is byte-identical to the prior single-eval path.
                repeats = max(1, self.config.eval_repeats)
                if repeats == 1:
                    # CRN seed is derived per (task, repeat) INSIDE the fan-out
                    # task and passed DOWN — never via instance/global state, or
                    # the asyncio.gather below would race and cross-contaminate
                    # the pairing. seed=None when paired_eval is off (no-op).
                    #
                    # Y4-P_benchmarks-5: a non-zero crn_repeat_offset marks a
                    # DISTINCT invocation over the SAME (generation, candidate,
                    # task, depth) run_id coordinates (the regression-guard
                    # base-floor resamples — the only non-zero caller), so
                    # stamp it as the spine telemetry repeat index (the H6
                    # pattern): start_record reads it in execute()'s
                    # synchronous prefix, so each resample row gets its own
                    # run_id instead of colliding with — being de-dup-dropped
                    # against or silently superseding — draw 0's row. Every
                    # concurrent _solve_one shares the same constant offset,
                    # so interleaved finally-resets are harmlessly re-stamped.
                    # Offset 0 (every other caller) never touches the solver —
                    # strictly mutation-free, byte-identical run_id basis.
                    stamp = self._uses_external_spine() and bool(crn_repeat_offset)
                    try:
                        if stamp:
                            solver.repeat_index = crn_repeat_offset
                        trace, tokens = await self._eval_solve_once(
                            solver, task, candidate,
                            seed=self._crn_seed(task, crn_repeat_offset),
                        )
                    finally:
                        if stamp:
                            solver.repeat_index = 0
                    inner_all = self._sum_inner_traces([trace])
                else:
                    # H6 companion: stamp the eval repeat index around each of the
                    # R solves so each sample lands on a DISTINCT telemetry run_id
                    # (compute_run_id folds a non-zero repeat_index); without it
                    # the R rows share coordinates and R-1 are dropped by the
                    # run_id de-dup. Only the spine's start_record reads it; the
                    # native/agentic paths ignore the attribute. r==0 keeps the
                    # basis byte-identical to the historical single-sample formula.
                    # Concurrency-safe: start_record reads repeat_index in execute's
                    # synchronous prefix (before any await), so a concurrent
                    # _solve_one cannot interleave between this stamp and that read.
                    # Y4-P_benchmarks-5: fold crn_repeat_offset into the stamp so a
                    # non-zero-offset invocation (base-floor resample r) occupies
                    # its own disjoint repeat window [rE, rE+E-1] — the same window
                    # arithmetic as the CRN seeds — instead of re-colliding with
                    # draw 0's indices. Default offset 0 ⇒ byte-identical stamps.
                    spine = self._uses_external_spine()
                    samples = []
                    try:
                        for r in range(repeats):
                            if spine:
                                solver.repeat_index = crn_repeat_offset + r
                            samples.append(
                                await self._eval_solve_once(
                                    solver, task, candidate,
                                    seed=self._crn_seed(task, crn_repeat_offset + r),
                                )
                            )
                    finally:
                        if spine:
                            solver.repeat_index = 0
                    tokens = sum(tk for _, tk in samples)
                    inner_all = self._sum_inner_traces([tr for tr, _ in samples])
                    ordered = sorted(
                        (tr for tr, _ in samples),
                        key=lambda t: t.score if math.isfinite(t.score) else float("-inf"),
                    )
                    trace = ordered[len(ordered) // 2]  # median-scoring trace
                completed += 1
                status = (
                    f"[green]✓ {trace.score:.3f}[/green]"
                    if trace.success
                    else f"[red]✗ {trace.error_summary[:50]}[/red]"
                )
                console.print(f"      [{completed}/{n}] {task.task_id} {status}")
                return trace, tokens, inner_all

        # Audit #46: isolate a single task's solver fault so it cannot abort the
        # whole run mid-iteration. The gate (_gate_check) and both test-eval paths
        # already tolerate a per-task raise; the full eval was the ONLY phase
        # without isolation, so one persistent solver.solve() failure (auth error,
        # oversized context for one task, API outage) killed the entire run and
        # discarded the in-progress generation. ``return_exceptions=True`` lets the
        # gather finish; we convert each real Exception into a failed Trace (score
        # 0.0, success=False) for that task — mirroring the gate/test handling —
        # while still propagating a genuine cancellation.
        raw_results = await asyncio.gather(
            *[_solve_one(t) for t in tasks], return_exceptions=True
        )
        results: list[tuple[Trace, int, dict]] = []
        for _task, _r in zip(tasks, raw_results):
            if isinstance(_r, BaseException):
                if isinstance(_r, asyncio.CancelledError):
                    raise _r
                logger.warning(
                    "Full-eval solve raised for task=%s (candidate=%s): %s — "
                    "recording a failed trace and continuing",
                    _task.task_id, candidate.candidate_id, _r,
                )
                _failed = Trace(
                    task_id=_task.task_id,
                    depth=candidate.depth,
                    success=False,
                    score=0.0,
                    error_summary=f"solve raised: {type(_r).__name__}: {_r}"[:500],
                )
                results.append((_failed, 0, self._sum_inner_traces([_failed])))
            else:
                results.append(_r)
        traces = [r[0] for r in results]
        # 6.3: tag each FAILED trace with a unified failure class for Ω guidance.
        for _tr in traces:
            if not _tr.success and not _tr.failure_class:
                _tr.failure_class = OmegaEngine._classify_error(_tr)
        total_tokens = sum(r[1] for r in results)

        candidate.traces = traces
        candidate.total_tokens += total_tokens
        # Accumulate inner-LLM tokens consumed by the executed scripts
        # themselves (e.g., classify solve() calling llm() per case). These
        # are real LLM cost and must be reported in summary.json alongside
        # outer-LLM tokens, not folded into the same total — analysis needs
        # to distinguish "tokens for code generation" from "tokens spent
        # by the generated code".
        # H7: accumulate over the ALL-R inner sums (`r[2]`) returned by
        # _solve_one, NOT over the median-only `traces` — under eval_repeats>1
        # each task really ran R full solves and spent R-fold inner tokens, so
        # summing only the median trace understates inner_total ~R-fold. At
        # eval_repeats=1 `r[2]` equals the single trace's fields → byte-identical.
        candidate.inner_tokens += sum(r[2]["inner_tokens"] for r in results)
        candidate.inner_prompt_tokens += sum(
            r[2]["inner_prompt_tokens"] for r in results
        )
        candidate.inner_completion_tokens += sum(
            r[2]["inner_completion_tokens"] for r in results
        )
        candidate.inner_calls += sum(r[2]["inner_calls"] for r in results)
        candidate.pass_at_1 = (
            sum(1 for t in traces if t.success) / len(traces) if traces else 0.0
        )
        # Filter non-finite scores from aggregates so NaN/inf from a buggy
        # evaluator can't propagate into candidate.mean_score, summary.json
        # (json.dumps emits literal NaN, which is invalid JSON), or downstream
        # parent selection. NaN traces are still recorded in candidate.traces
        # for observability; they're only excluded from numeric aggregates.
        finite_traces = [t for t in traces if math.isfinite(t.score)]
        if finite_traces:
            candidate.mean_score = (
                sum(t.score for t in finite_traces) / len(finite_traces)
            )
        else:
            candidate.mean_score = 0.0
        if len(finite_traces) < len(traces):
            dropped = [t.task_id for t in traces if not math.isfinite(t.score)]
            logger.warning(
                "Candidate %s: %d/%d traces had non-finite score; "
                "excluded from mean_score and per_task_scores. Tasks: %s",
                candidate.candidate_id, len(traces) - len(finite_traces),
                len(traces), dropped,
            )
        candidate.per_task_scores = {
            t.task_id: t.score for t in finite_traces
        }
        return candidate

    def _archive_score_ceiling(self) -> "float | None":
        """Reaudit #1 — the absolute score ceiling that
        ``Archive._fresh_headroom_signal`` (audit #5) compares raw per-task
        scores against: the bound benchmark adapter's ``score_scale()['hi']``.

        Mirrors the audit-#30 saturation-ceiling derivation in
        ``_within_task_focus`` (both delegate to ``_score_scale_hi``): a finite
        numeric ``hi`` IS the ceiling (unit / binary scales report 1.0, so the
        clause is byte-identical to the historical hardcoded 1.0); a continuous /
        unbounded scale (``hi=None``) or a missing / mock adapter returns ``None``
        so the absolute-ceiling clause is DROPPED and headroom relies solely on
        the per-task-best lag. Only consulted when ``within_task_depth_bonus > 0``
        (``--within-task-recursion``), so default runs are unaffected either way.

        Reads the adapter off ``self.executor`` (not ``self.adapter``) because
        the first call happens in ``__init__`` before ``self.adapter`` is bound.
        """
        return _score_scale_hi(getattr(self.executor, "adapter", None))

    def _archive_kwargs(self) -> dict:
        """The kwarg bundle every ``Archive`` construction / rebuild must share.

        Single source for the four sites (``__init__``, resume rebuild, and the
        two fresh-start fallbacks) so they can never drift apart. Reaudit #1 —
        plumbs the bound benchmark's ``score_scale()['hi']`` into
        ``Archive._fresh_headroom_signal`` (audit #5). ``None`` on a continuous
        scale drops the absolute-ceiling clause; consulted only when the depth
        bonus is live, so default runs are unaffected.

        The depth term requires BOTH ``within_task_recursion`` AND
        ``consolidate``: ``within_task_recursion`` is a MODIFIER scoped to
        consolidate mode (its DEEPEN directive + focus inheritance are dormant
        without ``--consolidate``), so its archive depth*headroom selection term
        must stay off there too — otherwise it would silently bias parent
        selection toward deep chains with the other two knobs inert. Without
        ``consolidate`` the term is 0.0, making ``_selection_weights``
        bit-identical to a plain non-consolidate run.
        """
        return {
            "novelty_alpha": self.config.novelty_alpha,
            "regression_guard": self.config.regression_guard,
            "within_task_depth_bonus": (
                _WITHIN_TASK_DEPTH_BETA
                if (self.config.within_task_recursion and self.config.consolidate)
                else 0.0
            ),
            "score_ceiling": self._archive_score_ceiling(),
        }

    def _outer_cumulative_total(self) -> int:
        """Audit #29 helper: the shared outer ``LLMClient``'s cumulative total
        token count (0 when no client / counter is wired).

        Used to measure the GATE phase's outer script-generation spend as a
        before/after delta. The gate solves are real outer LLM calls whose token
        counts were otherwise discarded (gate-pass reuses traces at 0 outer
        tokens; gate-fail counts only the Ω call), so result.total_tokens drifted
        below this authoritative counter — which already records every native /
        MetaLayer script-generation call. Returns 0 for the external-agent spine
        (those tokens are the inner/agent channel, not the outer counter), which
        is correct: they must not inflate the outer total_tokens.
        """
        cu = getattr(getattr(self, "llm_client", None), "cumulative_usage", None) or {}
        try:
            return int(cu.get("total", 0) or 0)
        except (TypeError, ValueError, AttributeError):
            return 0

    async def _gate_solve_one(
        self, solver, task: TaskDescription, candidate: Candidate,
        repeat_index: int = 0,
    ) -> Trace:
        """Solve a single gate task and return its trace.

        Delegates to ``_eval_solve_once`` for all non-spine candidates (so the
        gate trace carries the same depth/reasoning/duration + adoption-field
        stamping the full eval performs — required for gate-trace reuse); the
        spine branch keeps the gate-phase telemetry stamp:
        ``execution_phase='gate'`` gives the spine telemetry row a distinct
        run_id from the later full eval (de-dup correctness): a gate run and
        the eval run share identical run_id coordinates, so without the phase
        marker the eval row would de-dup against the gate row and
        agent_runs.jsonl would undercount. Restored in ``finally``.

        ``repeat_index`` threads the CRN ``(task, index)`` draw into the native
        solve so the reused gate trace IS that eval sample. ``_crn_seed`` returns
        ``None`` when ``paired_eval`` is OFF (default) ⇒ byte-identical to the
        legacy unseeded call. The spine branch takes no seed (``execute()`` gap,
        matching ``_eval_solve_once``).
        """
        if self._uses_external_spine():
            prev_phase = getattr(solver, "execution_phase", "eval")
            solver.execution_phase = "gate"
            try:
                trace, _ = await solver.execute(task)
            finally:
                solver.execution_phase = prev_phase
        else:
            trace, _ = await self._eval_solve_once(
                solver, task, candidate, seed=self._crn_seed(task, repeat_index)
            )
        return trace

    async def _gate_check(
        self,
        candidate: Candidate,
        parent: "Candidate | None",
        solver,
        tasks: list[TaskDescription],
    ) -> tuple[bool, int, dict[str, Trace]]:
        """Quality gate (roadmap v2 1.1 + 1.6). Returns
        ``(passed, num_tasks_tested, gate_traces)``.

        A gate task CLEARS iff ``trace.success`` AND (no parent baseline OR
        ``trace.score >= parent_score - gate_margin``) — a relative, scale-safe
        quality bar, not the old liveness rubber-stamp. The candidate passes on
        the first cleared task. Crash/exception still rejects (``continue`` +
        final ``False``). FAIL-OPEN: a task with no parent baseline (gen0
        children, missing score) falls back to the liveness check, so a missing
        baseline never rejects everything; ``gate_margin=None`` disables
        thresholding entirely (liveness gate, for ablation).

        Every solved gate trace is returned in ``gate_traces`` so the full
        evaluation can REUSE it instead of re-solving (1.6); with
        ``gate_repeats>1`` the median-scoring trace per task is kept. Deeper
        candidates get more gate tasks (2x at depth 3+) to reduce false rejects.
        """
        base = self.config.gate_tasks
        scale = min(candidate.depth - 1, 2)  # depth 2 → 1x, depth 3+ → 2x
        gate_n = min(base * scale, max(1, len(tasks) // 2))
        gate_tasks = self.rng.sample(tasks, gate_n)
        logger.debug(
            "Gate check: candidate=%s, depth=%d, testing %d tasks: %s",
            candidate.candidate_id, candidate.depth, gate_n,
            [t.task_id for t in gate_tasks],
        )

        margin = self.config.gate_margin  # None → thresholding disabled
        protect = self.config.protect_floor  # None → no per-task floor (legacy)
        repeats = max(1, self.config.gate_repeats)
        gate_traces: dict[str, Trace] = {}
        cleared_any = False

        spine = self._uses_external_spine()
        for task in gate_tasks:
            try:
                if repeats > 1 and spine:
                    # H6 companion: stamp repeat_index per gate sample so the R
                    # gate solves of one task get DISTINCT telemetry run_ids
                    # (the gate phase already separates gate rows from eval rows;
                    # the repeat index separates the R gate rows from each other).
                    # The first sample is index 0 → byte-identical basis with
                    # phase='gate'; restored to 0 in finally. The gate loop is
                    # sequential, so there is no concurrent repeat_index writer.
                    solver.repeat_index = 0
                trace = await self._gate_solve_one(
                    solver, task, candidate, repeat_index=0
                )
                if repeats > 1:
                    samples = [trace]
                    try:
                        for ri in range(1, repeats):
                            if spine:
                                solver.repeat_index = ri
                            samples.append(
                                await self._gate_solve_one(
                                    solver, task, candidate, repeat_index=ri
                                )
                            )
                    finally:
                        if spine:
                            solver.repeat_index = 0
                    # Audit #2: use the non-finite-safe key (NaN/inf → -inf) so a
                    # buggy evaluator's non-finite score cannot make list.sort's
                    # ordering undefined and pick an arbitrary "median" trace —
                    # matching the eval-path median guard in _evaluate_candidate.
                    samples.sort(
                        key=lambda t: t.score if math.isfinite(t.score) else float("-inf")
                    )
                    trace = samples[len(samples) // 2]  # median-scoring trace
            except Exception as e:
                logger.debug("  Gate task=%s exception: %s", task.task_id, e)
                continue

            gate_traces[task.task_id] = trace
            # base_score is the parent baseline for THIS task, computed
            # independently of the margin gate so the 6.2 floor works even when
            # margin thresholding is disabled.
            base_score = (
                parent.per_task_scores.get(task.task_id)
                if parent is not None else None
            )
            # 6.2 PROTECTION FLOOR: a hard per-task regression vetoes the
            # candidate outright — even if another gate task clears. This is the
            # fix for the gate admitting a candidate that tanks one task.
            if (
                protect is not None and base_score is not None
                and trace.score < base_score - protect
            ):
                logger.info(
                    "  Gate FLOOR veto on task=%s: score=%.3f < baseline %.3f - "
                    "floor %.3f", task.task_id, trace.score, base_score, protect,
                )
                return False, gate_n, gate_traces

            baseline = base_score if margin is not None else None
            cleared = trace.success and (
                baseline is None or trace.score >= baseline - margin
            )
            if cleared:
                cleared_any = True
                logger.debug(
                    "  Gate cleared on task=%s (score=%.3f, baseline=%s)",
                    task.task_id, trace.score, baseline,
                )
                # Legacy fast path: with the floor OFF, pass on the first clear
                # (byte-unchanged). With the floor ON, keep scanning so a later
                # gate task can still veto on a hard regression.
                if protect is None:
                    return True, gate_n, gate_traces
            else:
                logger.debug(
                    "  Gate task=%s not cleared (success=%s, score=%.3f, baseline=%s)",
                    task.task_id, trace.success, trace.score, baseline,
                )
        # Floor ON: passed iff something cleared AND nothing vetoed (no veto
        # would have returned above). Floor OFF: nothing cleared → reject.
        return (cleared_any if protect is not None else False), gate_n, gate_traces

    async def _attempt_within_layer_refine(
        self,
        *,
        parent: Candidate,
        buggy_child: Candidate,
        buggy_injection: InjectedCode,
        gate_traces: dict[str, Trace],
        tasks: list[TaskDescription],
        iteration: int,
        child_depth: int,
        temperature: float,
        previous_scores: "dict[str, float] | None",
        archive_best_scores: "dict[str, float] | None",
        result: EvolutionaryResult,
        out_dir: Path,
    ) -> "tuple[Candidate | None, int]":
        """Stage 2 WITHIN-LAYER REFINE: one extra Ω call to fix ``buggy_injection``.

        Called ONLY from the gate-fail branch when ``within_layer_refine`` is ON
        and the injection is non-empty (so gen0 / depth-1 / empty / focus
        resamples can never reach here). Makes one ``OmegaEngine.refine`` call,
        builds a NEW candidate from the corrected injection (monotonic — the
        rejected ``buggy_child`` is never mutated or archived), re-gates it, and
        KEEPS IT iff it now clears the gate. On success it is fully evaluated,
        gets a ``within_layer`` :class:`SelfRepairEvent`, and is archived + saved.

        Returns ``(refined_candidate | None, tokens_spent)`` — ``tokens_spent`` is
        what was added to ``result.total_tokens`` here (the refine Ω call, the
        re-gate's outer solve spend, plus the re-evaluation when the candidate was
        kept), for the caller's per-iteration ``iter_tokens`` tally. ``None`` ⇒ the
        refine produced an empty injection or still failed the gate; the original
        gate-fail outcome stands.
        """
        # The fix is motivated by the failure traces: prefer the gate traces
        # (fresh, carry eval_feedback / error_summary for the buggy injection);
        # fall back to the buggy child's own traces if the gate produced none.
        failure_traces = list(gate_traces.values()) or list(buggy_child.traces)
        if not failure_traces:
            return None, 0
        before_scores = {
            t.task_id: t.score
            for t in failure_traces
            if math.isfinite(t.score)
        }
        mean_before = (
            sum(before_scores.values()) / len(before_scores)
            if before_scores else 0.0
        )

        refine_start = time.time()
        console.print(
            f"    [yellow]Within-layer refine: fixing {buggy_child.candidate_id} "
            f"(depth={child_depth})...[/yellow]"
        )
        refined_injection, refine_tokens = await self.omega.refine(
            prev_injection=buggy_injection,
            child_traces=failure_traces,
            context_stack=parent.injected_codes,
            tasks=tasks,
            depth=child_depth,
            temperature=temperature,
            previous_scores=previous_scores,
            archive_best_scores=archive_best_scores,
            current_scores=before_scores or None,
            mean_before=mean_before,
            solver_language=self.solver_language,
            no_code_library=self.config.no_code_library,
            code_library_is_live=self._code_library_is_live(),
        )
        # E2 parity with generate(): defensively clear any code_library that
        # slipped through when the ablation strips it from the prompt.
        if self.config.no_code_library:
            refined_injection.code_library = {}
            refined_injection.code_library_bash = {}

        # Forensic #2 VERIFY-THEN-INJECT gate (default OFF ⇒ identity). Routed
        # here too so every InjectedCode-producing site is covered.
        refined_injection, refined_verified = await self._verify_and_filter_code_library(
            refined_injection, parent.injected_codes, tasks
        )

        # Empty refine ⇒ nothing to keep; count the Ω call and bail (the original
        # gate-fail outcome stands).
        if refined_injection.is_empty:
            result.total_tokens += refine_tokens
            console.print(
                f"    [yellow]Refine returned empty injection "
                f"(tokens={refine_tokens:,}) — keeping gate-fail[/yellow]"
            )
            return None, refine_tokens

        # NEW candidate (never mutate the rejected one). Parent is the ARCHIVED
        # parent the buggy child was bred from, so lineage / num_children rebuild
        # stay intact; the SelfRepairEvent records the buggy attempt as the
        # repaired source (provenance).
        refined_id = f"{buggy_child.candidate_id}_refine"
        refined = Candidate(
            candidate_id=refined_id,
            parent_id=parent.candidate_id,
            iteration=iteration,
            depth=child_depth,
            injected_codes=parent.injected_codes + [refined_injection],
            temperature_used=temperature,
            total_tokens=refine_tokens,
        )
        refined_solver = self._build_solver_from_candidate(refined)

        # KEEP THE BETTER: the refined injection must clear the same quality gate
        # the buggy one failed. If it still fails, the refine did not recover the
        # bug — fall back to the original gate-fail (count the Ω call only).
        # The re-gate's outer solve tokens are added to result.total_tokens below
        # AND must be threaded into the returned tokens_spent so the caller's
        # per-iteration iter_tokens stays in lock-step with the authoritative outer
        # counter (mirrors the main gate path, which adds gate_tokens to BOTH). 0
        # when gate_tasks == 0.
        regate_delta = 0
        if self.config.gate_tasks > 0:
            # Audit #29: count the refine re-gate's outer solve tokens too (covers
            # both pass and fail) via the cumulative_usage delta, so total_tokens
            # stays reconciled with the authoritative outer counter.
            _cu_before_gate2 = self._outer_cumulative_total()
            passed, _gate_n, gate2 = await self._gate_check(
                refined, parent, refined_solver, tasks
            )
            regate_delta = max(0, self._outer_cumulative_total() - _cu_before_gate2)
            result.total_tokens += regate_delta
            if not passed:
                result.total_tokens += refine_tokens
                console.print(
                    f"    [red]Refine still gate-FAILED (tokens={refine_tokens:,}) "
                    f"— keeping gate-fail[/red]"
                )
                return None, refine_tokens + regate_delta
        else:
            gate2 = {}

        refined = await self._evaluate_candidate(
            refined, refined_solver, tasks, precomputed=gate2
        )
        result.absorb_candidate_tokens(refined)

        # within_layer SelfRepairEvent — provenance of THIS construction.
        # SR1: score mean_after over the SAME keys as before_scores (the gate
        # subset) so mean_before→mean_after share a MATCHED denominator and the
        # delta is comparable; accepted is then the score-improvement verdict on
        # that matched set (NOT the keep decision — the refine was kept because it
        # cleared the gate, recorded separately as archived=True). The full
        # evaluated mean rides along in mean_after_full.
        refined_subset = {
            tid: float(refined.per_task_scores.get(tid, 0.0))
            for tid in before_scores
        }
        mean_after = (
            sum(refined_subset.values()) / len(refined_subset)
            if refined_subset else refined.mean_score
        )
        accepted = mean_after >= mean_before
        per_task_delta = {
            tid: refined_subset[tid] - float(before_scores.get(tid, 0.0))
            for tid in before_scores
        }
        event = SelfRepairEvent(
            candidate_id=refined.candidate_id,
            parent_candidate_id=buggy_child.candidate_id,
            granularity="within_layer",
            target_depth=child_depth,
            pre_code_hash=self._pre_process_hash(buggy_injection),
            post_code_hash=self._pre_process_hash(refined_injection),
            post_injection_ref=f"injected_code_d{child_depth}.json",
            mean_before=mean_before,
            mean_after=mean_after,
            mean_after_full=refined.mean_score,
            per_task_delta=per_task_delta,
            accepted=accepted,
            # The refine was archived because it CLEARED THE GATE (keep decision),
            # which is independent of the matched-denominator accepted verdict.
            archived=True,
            # Stage-3 classifier (the FEAL-bound test): was the repair a localized
            # error-correction or a novel rewrite? Pre failure class from the
            # gate-fail corpus; post from the (now mostly-passing) refined traces.
            classification=classify_repair(
                buggy_injection, refined_injection,
                pre_failure_class=self._dominant_failure_class(failure_traces),
                post_failure_class=self._dominant_failure_class(refined.traces),
            ),
            raw_omega_prompt=refined_injection.raw_omega_prompt,
            raw_omega_response=refined_injection.raw_omega_response,
        )
        refined.self_repair_events.append(event)

        # R2-CS-1: mirror the breed path's non-adoption penalty. When verified_code
        # is ON, a refined trace that re-derived a verified helper inline (never
        # called it) must not bank a per-task-best win the breed path would have
        # barred. Default OFF ⇒ the exact original unbarred add.
        if self.config.verified_code:
            self.archive.add(
                refined,
                bar_from_best=self._nonadopting_verified_tasks(
                    refined, refined_injection, refined_verified
                ),
            )
        else:
            self.archive.add(refined)
        self._save_candidate_incremental(refined, out_dir)
        console.print(
            f"    [green]Refine RECOVERED → {refined.candidate_id} "
            f"(subset mean {mean_before:.3f}→{mean_after:.3f}, "
            f"full {refined.mean_score:.3f}, {time.time()-refine_start:.1f}s)[/green]"
        )
        logger.info(
            "  Within-layer refine kept: %s (parent=%s, repaired=%s, depth=%d, "
            "subset mean_before=%.3f, mean_after=%.3f, full mean=%.3f, accepted=%s)",
            refined.candidate_id, parent.candidate_id, buggy_child.candidate_id,
            child_depth, mean_before, mean_after, refined.mean_score, accepted,
        )
        return refined, refined.total_tokens + regate_delta

    @staticmethod
    def _pre_process_hash(injected: InjectedCode) -> str:
        """sha256-prefix of an injection's ``pre_process`` (parity Trace.code_hash).

        The pre/post code-hash on a :class:`SelfRepairEvent` lets the Stage-3
        error-correction-vs-novel classifier (and any reviewer) tell whether a
        refine actually changed the code without inlining the blob.
        """
        return hashlib.sha256((injected.pre_process or "").encode()).hexdigest()[:16]

    @staticmethod
    def _dominant_failure_class(traces: "list[Trace] | None") -> str:
        """Most common ``classify_error`` class over the FAILING traces ("" if none).

        Feeds the Stage-3 error-correction-vs-novel classifier's failure-class
        signal. A fully-resolved (all-success) corpus returns "" so the classifier
        does not read a spurious failure-class shift from a clean repair.
        """
        from collections import Counter

        from meta_n.core.meta_layer import classify_error

        classes = [classify_error(t) for t in (traces or []) if not t.success]
        if not classes:
            return ""
        return Counter(classes).most_common(1)[0][0]

    def _should_repropagate(self, child: Candidate) -> bool:
        """Stage 3 trigger: did the new TOP layer plateau / regress vs its parent?

        The re-propagation rationale is "the top layer barely moved the score, so
        the bottleneck may be a LOWER layer — revise an intermediate one". The
        signal is the child's own delta vs its parent (the layer it added). Fires
        when that delta is a plateau/regression relative to a SCALE-AWARE epsilon
        (``<= 0.01`` on the unit scale, ``<= 1%`` of the observed score spread on
        non-unit scales) — the same plateau threshold the Ω novelty directive uses
        (``omega.py`` G5), so parity with G5 holds on any score magnitude. Returns
        ``False`` (never re-propagate) when there is no parent or no finite means
        to compare, or no INTERMEDIATE non-empty layer exists to revise.
        """
        if child.parent_id is None:
            return False
        parent = self.archive.find(child.parent_id)
        if parent is None:
            return False
        if not (math.isfinite(child.mean_score) and math.isfinite(parent.mean_score)):
            return False
        eps = OmegaEngine._scale_epsilon(child.per_task_scores, parent.per_task_scores)
        if (child.mean_score - parent.mean_score) > eps:
            return False  # the top layer genuinely improved — nothing to re-propagate
        return self._pick_repropagation_depth(child) is not None

    @staticmethod
    def _pick_repropagation_depth(child: Candidate) -> "int | None":
        """Pick the intermediate target depth ``d_t`` in ``[2, depth-1]`` to revise.

        Deterministic: the DEEPEST intermediate layer (closest to the plateaued
        top, the most proximate cause) whose injection is non-empty — scanning
        ``depth-1 .. 2``. ``d_t >= 2`` ALWAYS, so gen0 / depth-1 (the base solver)
        are NEVER touched. Returns ``None`` when no intermediate layer carries a
        non-empty injection (nothing to error-correct).
        """
        # injected_codes is 0-indexed: index i == depth i+2. The TOP layer is the
        # last entry (depth == child.depth); intermediates are depths [2, depth-1].
        for d_t in range(child.depth - 1, 1, -1):
            idx = d_t - 2
            if 0 <= idx < len(child.injected_codes):
                if not child.injected_codes[idx].is_empty:
                    return d_t
        return None

    async def _attempt_repropagation(
        self,
        *,
        child: Candidate,
        tasks: list[TaskDescription],
        iteration: int,
        temperature: float,
        archive_best_scores: "dict[str, float] | None",
        result: EvolutionaryResult,
        out_dir: Path,
    ) -> "tuple[Candidate | None, int]":
        """Stage 3 DOWNWARD RE-PROPAGATION: regenerate an intermediate layer of
        ``child`` and build a NEW monotonic candidate.

        Called ONLY from the post-archive.add depth>=3 branch when
        ``repropagation`` is ON and :meth:`_should_repropagate` fired. Picks
        ``d_t`` via :meth:`_pick_repropagation_depth`, rebuilds the context as the
        layers BELOW ``d_t``, feeds ``omega.generate`` the FULL-CHAIN failure
        traces (``child.traces``, contaminated by the layers above ``d_t``) plus
        the above-layer injections as ``downstream_injections`` (so the call is
        scoped to error-correction), REPLACES ``child.injected_codes[d_t-2]``, and
        evaluates the new chain. The result is ALWAYS archived (monotonic — the
        old ``child`` is retained); a ``downstream`` :class:`SelfRepairEvent` is
        logged + classified. ``accepted`` records whether the mean improved.

        Returns ``(reprop_candidate | None, tokens_spent)`` — ``tokens_spent`` is
        what was added to ``result.total_tokens`` here, for the caller's
        ``iter_tokens`` tally. ``None`` ⇒ the Ω regeneration returned an empty
        injection (nothing replaced); the original ``child`` stands.
        """
        d_t = self._pick_repropagation_depth(child)
        if d_t is None:  # defensive — _should_repropagate already checked
            return None, 0
        target_idx = d_t - 2
        below = list(child.injected_codes[:target_idx])      # layers below d_t
        above = list(child.injected_codes[target_idx + 1:])  # downstream feedback
        old_injection = child.injected_codes[target_idx]

        # Previous-layer scores for Ω's regression analysis: the pre-revision
        # baseline is the below-chain ancestor at depth d_t-1 (whose injected_codes
        # == ``below`` by monotonic breeding — Candidate.depth == 1+len(injected_codes)).
        # Its per_task_scores are the cumulative effect of layers d_t..top BEFORE
        # this revision, so Section C reports a REAL delta (mirrors the normal
        # breeding path's grandparent baseline). For d_t==2 there is no lower
        # meta-layer, so leave it None (the depth-2 render path). Falls back to the
        # child's own scores when the ancestor is unfindable (dangling lineage /
        # loaded archive / hand-built child) — byte-identical to the prior default.
        previous_scores = None
        if below:
            below_depth = d_t - 1
            anc = self.archive.find(child.parent_id)
            while anc is not None and anc.depth > below_depth:
                anc = self.archive.find(anc.parent_id)
            if anc is not None and anc.depth == below_depth and anc.per_task_scores:
                previous_scores = anc.per_task_scores or None
            else:
                previous_scores = child.per_task_scores or None

        reprop_start = time.time()
        console.print(
            f"    [yellow]Re-propagation: regenerating depth {d_t} of "
            f"{child.candidate_id} (top layer plateaued)...[/yellow]"
        )
        new_injection, reprop_tokens = await self.omega.generate(
            traces=child.traces,
            context_stack=below,
            tasks=tasks,
            depth=d_t,
            temperature=temperature,
            previous_scores=previous_scores,
            archive_best_scores=archive_best_scores,
            current_scores=child.per_task_scores or None,
            solver_language=self.solver_language,
            no_code_library=self.config.no_code_library,
            code_library_is_live=self._code_library_is_live(),
            downstream_injections=above or None,
        )
        if self.config.no_code_library:
            new_injection.code_library = {}
            new_injection.code_library_bash = {}

        # Forensic #2 VERIFY-THEN-INJECT gate (default OFF ⇒ identity). The
        # re-propagated layer's helpers are sandbox-checked like fresh breeding;
        # ``below`` is the context (lower-layer helpers a verified helper may call).
        new_injection, new_verified = await self._verify_and_filter_code_library(
            new_injection, below, tasks
        )

        if new_injection.is_empty:
            result.total_tokens += reprop_tokens
            console.print(
                f"    [yellow]Re-propagation returned empty injection "
                f"(tokens={reprop_tokens:,}) — keeping original[/yellow]"
            )
            return None, reprop_tokens

        # The regenerated layer must keep its original depth coordinate so the
        # rebuilt chain's source_depths stay consistent (omega parses at depth d_t).
        new_injection.source_depth = d_t
        # REPLACE (not augment) the intermediate layer; the chain length / depth
        # is unchanged — only the d_t layer's injection differs.
        new_chain = below + [new_injection] + above
        reprop_id = f"{child.candidate_id}_reprop_d{d_t}"
        reprop = Candidate(
            candidate_id=reprop_id,
            parent_id=child.candidate_id,  # parent = the revised child (monotonic)
            iteration=iteration,
            depth=child.depth,
            injected_codes=new_chain,
            temperature_used=temperature,
            total_tokens=reprop_tokens,
        )
        reprop_solver = self._build_solver_from_candidate(reprop)
        reprop = await self._evaluate_candidate(reprop, reprop_solver, tasks)
        result.absorb_candidate_tokens(reprop)

        per_task_delta = {
            tid: float(reprop.per_task_scores.get(tid, 0.0))
            - float(child.per_task_scores.get(tid, 0.0))
            for tid in child.per_task_scores
        }
        # SR2: accepted is the SCORE-IMPROVEMENT verdict on the matched denominator
        # (child vs reprop are both full means over the same task set), unified
        # with the within_layer granularity. archived is the keep decision —
        # re-propagation archives the reprop candidate unconditionally.
        accepted = (
            math.isfinite(reprop.mean_score)
            and reprop.mean_score >= child.mean_score
        )
        event = SelfRepairEvent(
            candidate_id=reprop.candidate_id,
            parent_candidate_id=child.candidate_id,
            granularity="downstream",
            target_depth=d_t,
            pre_code_hash=self._pre_process_hash(old_injection),
            post_code_hash=self._pre_process_hash(new_injection),
            post_injection_ref=f"injected_code_d{d_t}.json",
            mean_before=child.mean_score,
            mean_after=reprop.mean_score,
            mean_after_full=reprop.mean_score,
            per_task_delta=per_task_delta,
            accepted=accepted,
            archived=True,
            classification=classify_repair(
                old_injection, new_injection,
                pre_failure_class=self._dominant_failure_class(child.traces),
                post_failure_class=self._dominant_failure_class(reprop.traces),
            ),
            raw_omega_prompt=new_injection.raw_omega_prompt,
            raw_omega_response=new_injection.raw_omega_response,
        )
        reprop.self_repair_events.append(event)

        # R2-CS-1: mirror the breed path's non-adoption penalty on the
        # re-propagation add too. Default OFF ⇒ the exact original unbarred add.
        if self.config.verified_code:
            self.archive.add(
                reprop,
                bar_from_best=self._nonadopting_verified_tasks(
                    reprop, new_injection, new_verified
                ),
            )
        else:
            self.archive.add(reprop)
        child.num_children += 1  # reprop is an archived edge (parent_id=child); mirror breed path (L1340) + rebuild_from_disk so live == resumed
        self._save_candidate_incremental(reprop, out_dir)
        console.print(
            f"    [green]Re-propagation → {reprop.candidate_id} "
            f"(d{d_t}, mean {child.mean_score:.3f}→{reprop.mean_score:.3f}, "
            f"{event.classification}, {time.time()-reprop_start:.1f}s)[/green]"
        )
        logger.info(
            "  Re-propagation kept: %s (revised=%s, d_t=%d, mean_before=%.3f, "
            "mean_after=%.3f, accepted=%s, classification=%s)",
            reprop.candidate_id, child.candidate_id, d_t, child.mean_score,
            reprop.mean_score, accepted, event.classification,
        )
        return reprop, reprop.total_tokens

    async def _establish_base_floor(
        self,
        seed: Candidate,
        seed_solver,
        tasks: list[TaskDescription],
        result: EvolutionaryResult,
    ) -> None:
        """Forensic improvement #1 — run R seed-only resamples and install the
        base floor (best-of-R) + denoised focus scores (median-of-R).

        Two readouts of the SAME R draws, kept DISTINCT on purpose:
          * floor = best-of-R per task → the never-regress guarantee (per_task_best
            can never drop below what the base solver alone achieves);
          * focus = median-of-R per task → denoises the consolidate FOCUS pick so a
            degenerate one-draw 0.0 (e.g. crew_scheduling seed-42) no longer
            mis-selects a healthy task (true ~0.62) as the catastrophic focus.

        Draw 0 is the seed's OWN evaluation (already done); draws 1..R-1 are fresh
        resamples of the SAME seed solver. The resample candidates are NOT added to
        the archive — they are denoise draws, not archived chains — so the
        monotonic invariant holds (the stored ``gen0_seed`` candidate and its
        traces are never mutated; the floor index points at a fresh Trace).
        Only ever called when ``config.regression_guard`` is ON.
        """
        R = max(1, self.config.regression_guard_repeats)
        # task_id -> list of (score, trace) samples; draw 0 = the seed's own eval.
        samples: dict[str, list[tuple[float, Trace]]] = {}
        for tr in seed.traces:
            if math.isfinite(tr.score):
                samples.setdefault(tr.task_id, []).append((tr.score, tr))
        for r in range(1, R):
            # Audit #18: stamp the resample with the SEED's actual depth, not a
            # hardcoded 1. _eval_solve_once dispatches on candidate.depth (>1 →
            # execute(); ==1 → native solve(task, seed=)). With --seed-code-library
            # the seed_solver is a depth-2 MetaLayer chain whose solve() takes no
            # ``seed`` kwarg, so a depth=1 stamp mis-routed it into the native
            # branch and raised TypeError, aborting the run. Matching the seed's
            # depth keeps the bare-Layer1Solver case at depth 1 (byte-identical).
            rg_cand = Candidate(
                candidate_id=f"gen0_seed_rg{r}", iteration=0, depth=seed.depth
            )
            # Offset each resample past draw 0's [0, E-1] seed window (E =
            # max(1, eval_repeats)): resample r consumes indices [r*E, r*E+E-1],
            # so no CRN seed collides with draw 0 or across resamples.
            rg_cand = await self._evaluate_candidate(
                rg_cand, seed_solver, tasks,
                crn_repeat_offset=r * max(1, self.config.eval_repeats),
            )
            # Honest accounting: the resample really did spend base-solver tokens.
            result.absorb_candidate_tokens(rg_cand)
            for tr in rg_cand.traces:
                if math.isfinite(tr.score):
                    samples.setdefault(tr.task_id, []).append((tr.score, tr))
        base_floor: dict[str, float] = {}
        base_floor_traces: dict[str, Trace] = {}
        focus_scores: dict[str, float] = {}
        for tid, draws in samples.items():
            if not draws:
                continue
            # Dual-channel (success, score) pick — mirrors archive.py per-task-best
            # ranking so a valid-but-poor negative fit (success=True) is chosen as
            # the floor over a crash (success=False, clamped to 0.0). A plain
            # score-max would pick the crash's 0.0 on a negative-capable scale and
            # then block a genuine negative Ω gain at the raw-score clamp. On
            # non-negative scales any success scores strictly >0 while a crash
            # floors to 0.0, so the success term never flips the max ⇒ byte-identical.
            # RESIDUAL: if ALL R draws crash the max still picks a 0.0 crash and the
            # archive raw-score clamp would still block a later valid negative Ω —
            # honoring the dual-channel guarantee there needs a success-aware clamp
            # (archive.py), tracked as a follow-up; this fixes the any-valid-draw case.
            best_score, best_trace = max(
                draws, key=lambda st: (bool(st[1].success), st[0])
            )
            base_floor[tid] = best_score
            base_floor_traces[tid] = best_trace
            ordered = sorted(s for s, _ in draws)
            focus_scores[tid] = ordered[len(ordered) // 2]  # median-of-R
        self.archive.set_base_floor(base_floor, base_floor_traces)
        self._base_focus_scores = focus_scores
        logger.info(
            "Regression guard: base floor over R=%d seed resamples installed for "
            "%d tasks (best-of-R floor + median-of-R focus).",
            R, len(base_floor),
        )

    def _within_task_focus(
        self, parent: Candidate, cons_targets: list[str], k: int
    ) -> str | None:
        """Pick the focus task for a consolidation child (forensic improvement #3).

        OFF (``within_task_recursion`` False) ⇒ the EXACT existing round-robin
        pick: ``cons_targets[k % len(cons_targets)]`` when targets exist, else
        ``None`` — so flag-OFF breeding is byte-identical to HEAD.

        ON ⇒ INHERIT the parent's focus task so the SAME task is re-worked at this
        deeper layer (genuine within-task recursion: ``trace.depth`` compounds on
        one task, multiplicity >= 2). The parent's focus is recovered READ-ONLY as
        its single FRESH-solved task (``split_candidate_mean(parent).fresh_tasks``
        — the consolidate focus is exactly the one task a consolidation parent
        solves fresh), with NO new persisted Candidate field (monotonic; golden
        schema unchanged). Falls back to the round-robin pick when the parent did
        NOT fresh-solve exactly one task (e.g. the gen0 seed solves all tasks
        fresh, or a non-consolidate parent), or when that task is already SATURATED
        at the normalized ceiling (no headroom left to deepen) — so a finished task
        is released back to round-robin instead of starving the others.
        """
        round_robin = (
            cons_targets[k % len(cons_targets)] if cons_targets else None
        )
        if not self.config.within_task_recursion or not cons_targets:
            return round_robin
        # Local import: depth_attribution imports archive/meta_layer only (no
        # cycle with the orchestrator). Read-only over parent traces/scores.
        from meta_n.analysis.depth_attribution import split_candidate_mean

        fresh = split_candidate_mean(parent).fresh_tasks
        if len(fresh) != 1:
            # 0 fresh (nothing attributable) or >1 (the gen0 seed / a
            # non-consolidate parent) ⇒ no single task to inherit.
            return round_robin
        focus = fresh[0]
        # Bounded inheritance: release a SATURATED task back to round-robin so a
        # finished task does not monopolize the chain.
        #
        # Audit #30: the ceiling is the bound adapter's score_scale() hi, NOT a
        # hardcoded 1.0. On a continuous / unbounded scale (hi is None, e.g.
        # symbolic_regression / alphaevolve_math) "saturation" is undefined — a
        # genuinely finished task never reaches 1.0 — so the release must NEVER
        # fire there (else the chain monopolizes the task and starves the rest);
        # a unit-test mock's non-numeric stub likewise yields None instead of
        # crashing the ``score >= hi - 1e-9`` comparison below. A unit [0,1] /
        # binary adapter returns hi=1.0, so the release fires at exactly the old
        # threshold (byte-identical for the default scale).
        hi = _score_scale_hi(self.adapter)
        score = parent.per_task_scores.get(focus)
        if (
            hi is not None
            and score is not None
            and math.isfinite(score)
            and score >= hi - 1e-9
        ):
            return round_robin
        return focus

    def _effective_focus_scores(self, pool: list[str]) -> dict[str, float]:
        """F034 flag-ON ranking scores: denoised base as the noise floor,
        current per-task best once a task has genuinely improved past its
        best-of-R base floor. Monotone (archive PTB never drops), so no
        draw-noise is reintroduced; a saturated task ranks at the ceiling
        and rotates out."""
        ptb = self.archive.per_task_best_scores()
        floor = self.archive.base_floor_snapshot()  # {tid: (score, cid, trace)}
        out: dict[str, float] = {}
        for tid in pool:
            base = self._base_focus_scores.get(tid, 0.0)
            cur = ptb.get(tid)
            fl = floor.get(tid)
            if cur is None or (fl is not None and cur <= fl[0] + 1e-12):
                out[tid] = base          # no genuine improvement yet — keep denoised base
            else:
                out[tid] = max(base, cur)  # improved — rank at current level
        return out

    def _consolidation_targets(
        self, tasks: list[TaskDescription], n: int, iteration: int
    ) -> list[str]:
        """G9: pick the ``n`` task ids this generation's candidates will improve.

        Round-robin over the (sorted, for determinism) task set, rotated by
        ``iteration`` so every task gets improvement attempts across generations
        and none is starved — robust for small task sets where a headroom-only
        rule could hammer a single unimprovable task forever. Used ONLY in
        ``consolidate`` mode, where every non-target task is inherited at its
        per-task-best, so a candidate can only ever change its one target.

        Forensic improvement #1 — when ``regression_guard`` is ON and median-of-R
        denoised base scores exist, the pick is HEADROOM-based (lowest denoised
        base score first, tie-broken by an iteration-rotated round-robin within
        each tie group) instead of round-robin: this is the true highest-headroom
        task, so a one-draw 0.0 outlier (crew_scheduling seed-42) no longer
        drives focus. OFF / no denoised scores ⇒ byte-identical round-robin.

        F034 (§6b) — the guard's base ranking is gen0-FROZEN, so a task already
        improved to the ceiling would stay the focus forever (the mirror form of
        the round-robin warning above). ``focus_current_headroom`` re-ranks a
        task that has GENUINELY improved past its best-of-R base floor at its
        CURRENT per-task best (see :meth:`_effective_focus_scores`); tasks with
        no improvement keep the median-of-R denoised base ranking, so the gen0
        pick is unchanged even when ON. Default OFF ⇒ the frozen ranking below
        is byte-identical (the staleness is pinned in tests/test_regression_guard.py).
        """
        pool = sorted(t.task_id for t in tasks)
        if not pool:
            return []
        if self.config.regression_guard and self._base_focus_scores:
            if self.config.focus_current_headroom:
                scores = self._effective_focus_scores(pool)
            else:
                scores = {
                    tid: self._base_focus_scores.get(tid, 0.0) for tid in pool
                }
            ranked = sorted(pool, key=lambda tid: (scores[tid], tid))
            # Y4-P_benchmarks-3: rotate WITHIN each exact-score tie group by
            # iteration. A fixed task-id tie-break would pin the pick on the
            # alphabetically-first member of the lowest tie group — on a binary
            # scale every unimproved task ties at exactly 0.0, so a single
            # unimprovable task could be hammered forever (the round-robin
            # docstring's own warning above). Distinct scores form singleton
            # groups (offset ``anything % 1 == 0``), so continuous scales stay
            # byte-identical; cross-group order is preserved (lowest-score
            # groups still fill the picks first); the rotation depends only on
            # ``(iteration, n)``, so checkpoint resume stays deterministic.
            out: list[str] = []
            i = 0
            while i < len(ranked):
                j = i
                while j < len(ranked) and scores[ranked[j]] == scores[ranked[i]]:
                    j += 1
                group = ranked[i:j]
                off = (iteration * n) % len(group)
                out.extend(group[off:] + group[:off])
                i = j
            return out[: min(n, len(pool))]
        offset = (iteration * n) % len(pool)
        return [pool[(offset + i) % len(pool)] for i in range(min(n, len(pool)))]

    def _consolidation_precomputed(
        self, target: str, tasks: list[TaskDescription], child_depth: int
    ) -> dict[str, Trace]:
        """G9: the inherit-frozen map for a consolidation candidate targeting
        ``target`` — every OTHER task's per-task-best trace, so ``_evaluate_candidate``
        reuses it (no re-solve, 0 tokens) and ONLY ``target`` is solved fresh.

        A task with no archive trace yet (e.g. gen0 hasn't covered it) is omitted,
        so it gets solved normally rather than inherited — correct for the first
        generation before per-task bests exist.

        Under ``within_task_recursion`` the inherited copy's depth is clamped to
        strictly below ``child_depth`` so an inherited trace can never be bucketed
        as FRESH by ``split_candidate_mean`` (``d >= candidate.depth``): the child's
        stored map then has exactly ONE trace at ``child_depth`` — its own fresh
        focus solve (added by the caller, not in this map) — so its true focus is
        recoverable by ``_within_task_focus``. Gated behind the flag ⇒ every OFF
        path keeps the frozen trace depths byte-identical.
        """
        ptb_traces = self.archive.per_task_best_traces()
        out: dict[str, Trace] = {}
        for t in tasks:
            if t.task_id == target:
                continue
            tr = ptb_traces.get(t.task_id)
            if tr is not None and (tr.script or ""):
                # Deep-copy so we never mutate the archive's stored trace (the
                # _classify_error tagging in _evaluate_candidate writes in place),
                # and ZERO the inner-LLM cost: the inherited solve's tokens belong
                # to the candidate that ORIGINALLY produced it (already counted
                # there) — this candidate only pays for its one focus task. Outer
                # tokens are already 0 on the reused path. (Audit W5: HIGH
                # double-count + MEDIUM aliasing, fixed at the source.)
                frozen = tr.model_copy(deep=True)
                frozen.inner_tokens = 0
                frozen.inner_prompt_tokens = 0
                frozen.inner_completion_tokens = 0
                frozen.inner_calls = 0
                if self.config.within_task_recursion:
                    # Never let an inherited (same-or-deeper) sibling trace be
                    # mis-bucketed as this child's fresh solve; a shallower trace
                    # is unchanged (min keeps it).
                    frozen.depth = min(frozen.depth, child_depth - 1)
                out[t.task_id] = frozen
        return out

    def _build_merged_candidate(
        self, tasks: list[TaskDescription], result: EvolutionaryResult,
        force: bool = False,
    ) -> "Candidate | None":
        """4.1 Ω_merge (pure assembly): synthesize a deployable oracle that routes
        each task to its per-task-best frozen winner.

        No LLM call and no re-evaluation — the candidate is assembled directly
        from the archive's per-task bests, so its mean IS the oracle by
        construction at ~zero cost. Property-gated: a NO-OP on single-task
        benchmarks (vacuous) and when a single chain owns every per-task best
        (``len(sources) <= 1`` — the ONLY case where oracle == best holds by
        construction), plus the force-bypassable range-relative gap gate below.
        Returns the merged Candidate (already added to the archive, and
        excluded from the breedable pool) or ``None``.
        """
        sources = set(self.archive.per_task_best_sources().values())
        best_before = self.archive.best_mean_score
        # Y4-P_benchmarks-4: no score-scale ``kind`` gate. The former
        # ``kind == "binary"`` no-op arm was unreachable (no adapter declares
        # 'binary'; TB/SWE inherit 'unit' from BenchmarkAdapter.score_scale)
        # and its oracle==best premise is false on a multi-task binary scale
        # with disjoint per-task wins (two chains at {1,0} / {0,1} give oracle
        # 1.0 vs best 0.5) — exactly the case the merge exists to deploy. The
        # true oracle==best degeneracy is ``len(sources) <= 1``, gated here.
        if len(tasks) <= 1 or len(sources) <= 1:
            return None
        # ``force`` (G9 consolidate mode): always materialize the deployable
        # oracle, even when the gap is small — the whole point of consolidation
        # is to realize it. Otherwise keep the range-relative gap gate (the E1
        # 0.047 gap fell just under 0.05*range and the merge never fired).
        if not force and (
            result.oracle_mean_score - best_before <= 0.05 * self.archive.score_range()
        ):
            return None

        ptb_traces = self.archive.per_task_best_traces()
        task_solution_map: dict[str, str] = {}
        merged_traces: list[Trace] = []
        for t in tasks:
            tr = ptb_traces.get(t.task_id)
            if tr is None or not (tr.script or ""):
                continue
            # Deep-copy so the merged candidate never aliases the archive's
            # _best_per_task entries (in-place tagging, e.g. failure_class,
            # must not leak into the per-task-best index) — same discipline
            # as the consolidation-inherit path above.
            frozen = tr.model_copy(deep=True)
            task_solution_map[t.task_id] = frozen.script
            merged_traces.append(frozen)
        if not task_solution_map:
            return None

        merge_ic = InjectedCode(
            task_solution_map=task_solution_map,
            rationale="SYNTHESIZED:Ω_merge",
            source_depth=2,
        )
        merged = Candidate(
            candidate_id="merge_oracle",
            parent_id=None,
            iteration=result.total_iterations,
            depth=2,
            injected_codes=[merge_ic],
            traces=merged_traces,
            pass_at_1=(sum(1 for tr in merged_traces if tr.success) / len(merged_traces)
                       if merged_traces else 0.0),
            mean_score=result.oracle_mean_score,
            per_task_scores=self.archive.per_task_best_scores(),
        )
        self.archive.add(merged)
        logger.info(
            "Ω_merge: synthesized deployable oracle '%s' (mean=%.3f vs best=%.3f, "
            "%d contributing chains, %d routed tasks)",
            merged.candidate_id, merged.mean_score, best_before,
            len(sources), len(task_solution_map),
        )
        return merged

    @staticmethod
    def _format_test_eval_status(eval_result) -> str:
        """Green ✓ / red ✗ score cell shared by the [test] and [chain-test]
        console loops — one format string so the two reports can't drift."""
        return (
            f"[green]✓ {eval_result.score:.3f}[/green]"
            if eval_result.success
            else f"[red]✗ {eval_result.score:.3f}[/red]"
        )

    async def _run_test_evaluation(
        self, tasks: list[TaskDescription]
    ) -> dict[str, float]:
        """Re-evaluate oracle per-task best solutions on the test split.

        Only available for executors with an adapter that supports evaluate_test
        (e.g., COBenchExecutor, TextClassificationExecutor). The stored
        ``trace.script`` is re-executed against the held-out test cases via
        ``adapter.evaluate_test(task, trace.script)`` — the classification
        adapter's ``evaluate_test`` re-runs the evolved ``solve()`` on test
        examples internally, so no separate re-solve step is needed here.
        """
        if not hasattr(self.executor, "adapter") or not hasattr(
            self.executor.adapter, "evaluate_test"
        ):
            return {}

        adapter = self.executor.adapter
        best_traces = self.archive.per_task_best_traces()
        # Hoisted above the loop (O(n) once, not per task) — same index snapshot
        # for every reported "(from ...)" attribution.
        best_sources = self.archive.per_task_best_sources()
        test_scores: dict[str, float] = {}

        console.print(f"  Evaluating {len(tasks)} tasks on test split (oracle per-task best)...")
        for i, task in enumerate(tasks, 1):
            trace = best_traces.get(task.task_id)
            if not trace:
                logger.debug("  No best trace for task=%s — skipping test eval", task.task_id)
                continue
            try:
                source_id = best_sources.get(task.task_id, "unknown")

                # Re-exec the stored solution against the held-out test cases.
                # For CO-Bench this reuses the code directly; for classification
                # the adapter's evaluate_test re-runs the evolved solve() on the
                # test examples internally.
                eval_result = await adapter.evaluate_test(task, trace.script)

                test_scores[task.task_id] = eval_result.score
                status = self._format_test_eval_status(eval_result)
                console.print(f"    [{i}/{len(tasks)}] [test] {task.task_id} {status} (from {source_id})")
                logger.info(
                    "  Test eval: task=%s, score=%.3f, source=%s",
                    task.task_id, eval_result.score, source_id,
                )
            except Exception as e:
                console.print(f"    [{i}/{len(tasks)}] [test] {task.task_id} [red]error: {e}[/red]")
                logger.error("  Test eval error: task=%s, error=%s", task.task_id, e)
                test_scores[task.task_id] = 0.0

        return test_scores

    async def _run_chain_test_evaluation(
        self, tasks: list[TaskDescription]
    ) -> dict[str, float]:
        """Evaluate the single best-chain candidate on the test split.

        Unlike ``_run_test_evaluation`` which picks the per-task best candidate
        (oracle), this method uses one candidate for all tasks — the overall
        best candidate from the archive.
        """
        if not hasattr(self.executor, "adapter") or not hasattr(
            self.executor.adapter, "evaluate_test"
        ):
            return {}

        best = self.archive.best_candidate
        if not best:
            return {}

        adapter = self.executor.adapter
        needs_re_solve = hasattr(adapter, "get_test_task")
        # Only build solver chain when we need to re-solve (classification tasks).
        # For code-based tasks (CO-Bench), we reuse the stored trace scripts.
        solver = self._build_solver_from_candidate(best) if needs_re_solve else None
        # Index the candidate's traces by task_id for code-based reuse
        traces_by_task = {
            trace.task_id: trace for trace in best.traces
        } if best.traces else {}
        chain_test_scores: dict[str, float] = {}

        cid = best.candidate_id
        console.print(f"  Evaluating {len(tasks)} tasks on test split (chain: {cid})...")
        for i, task in enumerate(tasks, 1):
            try:
                if needs_re_solve:
                    test_task = adapter.get_test_task(task)
                    script, _, _ = await solver.solve(test_task)
                    eval_result = await adapter.evaluate_test(test_task, script)
                else:
                    trace = traces_by_task.get(task.task_id)
                    if not trace:
                        logger.debug(
                            "  No trace for chain candidate %s task=%s",
                            cid, task.task_id,
                        )
                        continue
                    eval_result = await adapter.evaluate_test(task, trace.script)

                chain_test_scores[task.task_id] = eval_result.score
                status = self._format_test_eval_status(eval_result)
                console.print(f"    [{i}/{len(tasks)}] [chain-test] {task.task_id} {status}")
                logger.info(
                    "  Chain test eval: task=%s, score=%.3f, candidate=%s, re_solved=%s",
                    task.task_id, eval_result.score, cid, needs_re_solve,
                )
            except Exception as e:
                console.print(
                    f"    [{i}/{len(tasks)}] [chain-test] {task.task_id} [red]error: {e}[/red]"
                )
                logger.error("  Chain test eval error: task=%s, error=%s", task.task_id, e)
                chain_test_scores[task.task_id] = 0.0

        return chain_test_scores

    def _select_temperature(self, k_index: int, iteration: int = 0) -> float:
        """Cycle through configured temperatures for diversity (roadmap v2 2.4).

        Decoupled from ``k`` alone: rotating by ``(iteration + k_index)`` means a
        single-candidate beam (K=1) still visits every configured temperature
        across iterations instead of being pinned to ``temps[0]`` forever.
        Deterministic rotation (no RNG draw, so the gate / parent RNG stream is
        unaffected — reproducibility).
        """
        temps = self.config.temperatures
        return temps[(iteration + k_index) % len(temps)]

    def _print_candidate(self, candidate: Candidate, label: str):
        """Print candidate summary."""
        n_success = sum(1 for t in candidate.traces if t.success)
        n_total = len(candidate.traces)
        console.print(
            f"  [{label}] depth={candidate.depth}, "
            f"{n_success}/{n_total} passed, "
            f"mean_score={candidate.mean_score:.3f}, "
            f"tokens={candidate.total_tokens}"
        )

    # ------------------------------------------------------------------ #
    # Persistence delegates (F253) — the implementations live in
    # meta_n/core/run_persistence.py::RunPersistence. Kept as one-line
    # delegating methods so every existing caller (run(), main.py, and the
    # tests that call or patch these private names on the orchestrator)
    # keeps working unchanged.
    # ------------------------------------------------------------------ #

    def _try_resume(self, out_dir: Path) -> dict | None:
        """Delegates to :meth:`RunPersistence.try_resume`."""
        return self._persistence.try_resume(out_dir)

    def _save_checkpoint(
        self,
        out_dir: Path,
        iteration: int,
        patience_counter: int,
        prev_best: float,
        total_tokens: int,
        convergence_history: list[float],
        prev_oracle: float = 0.0,
        oracle_history: list[float] | None = None,
        result: "EvolutionaryResult | None" = None,
    ):
        """Delegates to :meth:`RunPersistence.save_checkpoint`."""
        return self._persistence.save_checkpoint(
            out_dir, iteration, patience_counter, prev_best, total_tokens,
            convergence_history, prev_oracle=prev_oracle,
            oracle_history=oracle_history, result=result,
        )

    def _save_running_summary(
        self,
        out_dir: Path,
        iteration: int,
        result: EvolutionaryResult,
        run_start: float,
    ):
        """Delegates to :meth:`RunPersistence.save_running_summary`."""
        return self._persistence.save_running_summary(
            out_dir, iteration, result, run_start
        )

    def _save_candidate_incremental(self, candidate: Candidate, out_dir: Path):
        """Delegates to :meth:`RunPersistence.save_candidate_incremental`."""
        return self._persistence.save_candidate_incremental(candidate, out_dir)

    def _read_agent_run_rows(self) -> list[dict]:
        """Delegates to :meth:`RunPersistence.read_agent_run_rows`."""
        return self._persistence.read_agent_run_rows()

    @staticmethod
    def _aggregate_agent_rows(rows: list[dict]) -> dict:
        """Delegates to :meth:`RunPersistence.aggregate_agent_rows`."""
        return RunPersistence.aggregate_agent_rows(rows)

    def _candidate_agent_telemetry_rollup(self, candidate_id: str) -> dict | None:
        """Delegates to :meth:`RunPersistence.candidate_agent_telemetry_rollup`."""
        return self._persistence.candidate_agent_telemetry_rollup(candidate_id)

    def _run_level_agent_telemetry_rollup(self) -> dict | None:
        """Delegates to :meth:`RunPersistence.run_level_agent_telemetry_rollup`."""
        return self._persistence.run_level_agent_telemetry_rollup()

    def _write_run_config(
        self,
        out_dir: Path | str,
        run_config: dict | None,
        *,
        stage: str = "end",
    ) -> None:
        """Delegates to :meth:`RunPersistence.write_run_config`."""
        return self._persistence.write_run_config(out_dir, run_config, stage=stage)

    def save_results(
        self,
        result: EvolutionaryResult,
        output_dir: str | None = None,
        run_config: dict | None = None,
    ):
        """Save final results — delegates to :meth:`RunPersistence.save_results`.

        Candidates are already saved incrementally to self.config.output_dir
        during run(); this saves summary, convergence, per-task best, and
        lineage to the same directory."""
        return self._persistence.save_results(
            result, output_dir=output_dir, run_config=run_config
        )
