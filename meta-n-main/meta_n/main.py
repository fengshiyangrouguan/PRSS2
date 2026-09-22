"""CLI entry point for Meta^n."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import yaml
from rich.console import Console

from meta_n.core.base_executor import LocalExecutor
from meta_n.core.llm_client import LLMClient, LLMConfig
from meta_n.core.meta_layer import TaskDescription
from meta_n.core.omega import OmegaEngine
from meta_n.core.spine_routing import uses_external_spine
from meta_n.utils.cost_tracker import BudgetExceededError

console = Console()


def _gate_margin(s: str) -> float | None:
    """``--gate-margin`` parser: ``none``/``off`` disables thresholding (the
    legacy liveness gate, for clean ablation); anything else is a float."""
    return None if s.lower() in ("none", "off") else float(s)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Meta^n: Recursive Self-Improving Agent Framework",
        # No prefix abbreviation: apply_benchmark_config detects explicitly-set
        # flags by option-string match, so an abbreviated flag would be missed
        # and silently overridden by the benchmark-config YAML.
        allow_abbrev=False,
    )
    parser.add_argument(
        "--tasks",
        default=None,
        help="Path to tasks JSON file (for custom tasks)",
    )
    parser.add_argument(
        "--benchmark",
        default=None,
        choices=[
            "co_bench", "symptom2disease", "lawbench_charge", "terminal_bench",
            "alphaevolve_math", "symbolic_regression", "algotune", "arc_agi_2",
            "swe_bench_verified",
        ],
        help="Use a benchmark instead of a tasks file",
    )
    parser.add_argument(
        "--bench-data-dir",
        default=None,
        help="Benchmark data directory (default: ./data/<benchmark>)",
    )
    parser.add_argument(
        "--bench-tasks",
        nargs="*",
        default=None,
        help="Specific benchmark task names (default: all). Matching is "
             "per-benchmark: EXACT name for co_bench / terminal_bench / "
             "swe_bench_verified (swe_bench fails hard on zero matches); "
             "SUBSTRING for arc_agi_2 / alphaevolve_math / symbolic_regression "
             "/ algotune (a pattern may match multiple tasks — an over-match "
             "warning is logged).",
    )
    parser.add_argument(
        "--bench-limit",
        type=int,
        default=None,
        help="Limit number of benchmark tasks (for piloting)",
    )
    parser.add_argument(
        "--benchmark-config",
        default="auto",
        help="Benchmark-features YAML that turns ON meta-n's metacognition stack "
             "WITHOUT changing code defaults (use_archive / consolidate / "
             "regression_guard + companions, per-family foster_adoption, noisy-"
             "classification eval_repeats). Precedence: explicit CLI flag > this "
             "file's per-benchmark block > its 'defaults' block > code default. "
             "'auto' (default) loads the bundled meta_n/configs/benchmark_features.yaml; "
             "'none' skips it entirely to REPRODUCE the bare pre-metacognition "
             "behavior; any other value is a path to a custom YAML. "
             "Applies only to --benchmark runs; custom --tasks runs always "
             "keep bare code defaults. Per-key opt-out: every boolean this "
             "file sets has a --no-<flag> CLI form (e.g. --no-consolidate); "
             "--benchmark-config none drops ALL its defaults at once (a lone "
             "--no-use-archive would leave the archive-only companions on "
             "and be refused).",
    )
    parser.add_argument(
        "--model",
        default="anthropic/claude-sonnet-4-20250514",
        help="Model name (default: claude-sonnet via OpenRouter)",
    )
    parser.add_argument(
        "--base-url",
        default="https://openrouter.ai/api/v1",
        help="LLM API base URL",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key (default: OPENROUTER_API_KEY env var)",
    )
    parser.add_argument(
        "--exclude-providers",
        nargs="*",
        default=None,
        help="OpenRouter: exclude these providers (e.g., together)",
    )
    # Azure OpenAI backend. Activated by --model azure/<deployment>; the
    # endpoint and key come from AZURE_OPENAI_{ENDPOINT,API_KEY} env vars
    # (already set in .env for the csci8980 group key) unless overridden.
    parser.add_argument(
        "--azure-endpoint",
        default=None,
        help="Azure OpenAI endpoint URL (default: $AZURE_OPENAI_ENDPOINT). "
             "Only used when --model is prefixed with 'azure/'.",
    )
    parser.add_argument(
        "--azure-api-version",
        default="2024-12-01-preview",
        help="Azure OpenAI API version (default: 2024-12-01-preview).",
    )
    parser.add_argument(
        "--daily-budget-usd",
        type=float,
        default=None,
        help="USD spend cap per local day. Default: 500 for Azure (the daily "
             "envelope after the May 2026 TPM bump; the project-level $1450 "
             "budget remains the operator-enforced hard ceiling), 0 (off) for "
             "OpenRouter. Hard stop — raises BudgetExceededError when today's "
             "spend hits the cap. Override with $META_N_DAILY_BUDGET_USD.",
    )
    parser.add_argument(
        "--max-backend-requests",
        "--max-llm-calls",              # legacy alias, same destination
        dest="max_backend_requests",
        type=int,
        default=None,
        help="HARD STOP once this many HTTP requests have been issued to the "
             "model provider. Counted AT THE SEND SITE, before the request, so "
             "empty-content escalations and transport retries are all counted "
             "-- in the 2026-09-17 pilot 983 billed requests were invisible to "
             "the I/O log for exactly that reason. This is a BACKEND-REQUEST "
             "budget: not dollars, and not a count of logical llm() calls. "
             "Override with $META_N_MAX_BACKEND_REQUESTS.",
    )
    parser.add_argument(
        "--reduction-mode",
        choices=["official", "official_trace_k4", "predictive", "matched_k4",
                 "full"],
        default="official",
        help="Context reduction before every Omega call. `official` (default) "
             "is the historical path, byte-identical: it caps traces and the "
             "context stack under two SEPARATE budgets. `official_trace_k4` is "
             "THE MAIN BASELINE for the paired comparison: the same native path "
             "with the trace sampler capped at 4 and the stack then truncated by "
             "the SAME native rule, so it differs from `predictive` only in "
             "WHICH four traces reach Omega. `predictive` swaps in the RPBE "
             "adapter (frozen encoder + Gamma attention + PredictiveSelector) "
             "and REQUIRES $CODEBERT_PATH and --gamma-checkpoint. `matched_k4` "
             "is the RETIRED capacity ablation (at most 4 objects total over "
             "traces + stack, from the phase that had Gamma compress both): it "
             "is no longer the main comparison, because dropping the stack "
             "changes the recursion mechanism, not just the trace choice. "
             "`full` sends the unreduced context, bounded only by the model's "
             "hard limit. The adapter covers BOTH call sites: generate and "
             "refine.",
    )
    parser.add_argument(
        "--gamma-checkpoint",
        default=None,
        help="Path to the frozen Gamma produced by Phase B. REQUIRED by "
             "--reduction-mode predictive: Phase C must evaluate a TRAINED "
             "Gamma, so a missing/unreadable checkpoint aborts the run instead "
             "of silently scoring with a randomly initialised model. May also "
             "be given as $META_N_GAMMA_CHECKPOINT.",
    )
    parser.add_argument(
        "--no-test-eval",
        action="store_true",
        help="Never read the held-out TEST split during this run. The SRI formal "
             "protocol needs this on EVERY stage except the post-freeze `final` "
             "stage: a search run (or the shared-root run, which has no "
             "--sri-context and therefore no ledger to carry the setting) would "
             "otherwise evaluate the per-task oracle AND the best chain against "
             "test before the fork -- reading test data during the search phase "
             "and paying for a number the protocol does not use. Default off = "
             "byte-identical.",
    )
    parser.add_argument(
        "--sri-context",
        default=None,
        help="Path to a JSON SRI run context (formal protocol). Its presence "
             "ENABLES the proposal-slot ledger and the orchestrator hooks; "
             "without it every hook stays a strict no-op. Required keys: "
             "run_id, arm, backbone, cohort_id, search_seed, "
             "root_bundle_sha256, and ledger_path; optional: "
             "gamma_checkpoint_sha256, gate_reason, cohort, "
             "no_test_eval (default true for the formal arm stages).",
    )
    parser.add_argument(
        "--cost-ledger-dir",
        default=None,
        help="Directory for daily JSONL cost ledgers (default: ~/.meta_n_costs "
             "or $META_N_COST_LEDGER_DIR).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=16384,
        help="Max output tokens per LLM call (default: 16384)",
    )
    parser.add_argument(
        "--reasoning-effort",
        default=None,
        help="Sent as reasoning_effort on every LLM call (e.g. 'none' to stop "
             "a reasoning/QAT model like LM Studio Gemma-QAT from spending its "
             "token budget on hidden reasoning). Default None = field omitted "
             "(byte-identical requests). Applies to search AND subprocess-"
             "isolated test evaluation.",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=None,
        help="Per-call LLM request timeout in seconds. Default None = byte-"
             "identical fallback (env $META_N_LLM_REQUEST_TIMEOUT, else 360.0). "
             "Raise it (e.g. 1200) for slow local reasoning models that over-"
             "reason on heavy solve()-style prompts.",
    )
    parser.add_argument(
        "--omega-context-budget",
        type=int,
        default=None,
        help="Total INPUT-prompt token budget for the Omega context window "
             "(drives failure-trace / injected-code truncation). Unrelated to "
             "--max-tokens (the LLM OUTPUT cap). Default None = keep the built-in "
             "100k budget (existing runs byte-identical). Set this to the resolved "
             "model window (e.g. 32768 for a small local backbone) so truncation "
             "fires before the real context window silently overflows.",
    )
    parser.add_argument(
        "--symmetric-trace-sampling",
        # YAML-settable metacognition boolean: BooleanOptionalAction gives it a
        # --no- form so the OFF direction is reachable per-key when the bundled
        # benchmark config turns it on. default=False is MANDATORY on every
        # such conversion (BooleanOptionalAction otherwise defaults to None,
        # which would change config.json provenance / bare-path byte-identity).
        action=argparse.BooleanOptionalAction,
        default=False,
        help="F156: opt the Omega trace sampler into symmetric sampling — "
             "failures backfill leftover sample slots when successes are "
             "scarce, and token-budget eviction preserves the failure ratio "
             "across classes instead of evicting successes first. Default OFF "
             "preserves the historical sampling byte-identically for "
             "cross-run comparability. NOTE: the bundled benchmark config "
             "(--benchmark-config auto, the default) turns this ON for every "
             "benchmark; --benchmark-config none restores this code default.",
    )
    parser.add_argument(
        "--empty-retry-max-tokens",
        type=int,
        default=32768,
        help="One-shot escalation cap re-issued when an LLM call returns EMPTY "
             "content with finish_reason='length' (a reasoning/QAT model that "
             "burned its whole budget on hidden reasoning). Fires only when this "
             "is > --max-tokens; 0 disables. Re-issues at min(this, 4x the "
             "call's requested max_tokens). Default: 32768.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.02,
        help="Improvement threshold for termination (default: 0.02)",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=10,
        help="Maximum recursion depth (default: 10)",
    )
    parser.add_argument(
        "--output-dir",
        default="./experiments",
        help="Base output directory (default: ./experiments)",
    )
    parser.add_argument(
        "--exp-name",
        default=None,
        help="Experiment name (default: auto-generated from timestamp + model)",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Max concurrent tasks (default: 1 = sequential)",
    )
    parser.add_argument(
        "--local-exec",
        action="store_true",
        help="UNSUPPORTED — refused at startup (T2.5): never wired into "
             "executor selection; each benchmark builds its own executor. The "
             "--tasks path already uses LocalExecutor implicitly. Kept only so "
             "the refusal message is explanatory.",
    )
    parser.add_argument(
        "--instance-workers",
        type=int,
        default=0,
        help="Parallel instance workers per task for CO-Bench eval "
             "(0=cpu_count) and text-classification eval (0=4). Ignored by "
             "other benchmarks.",
    )
    parser.add_argument(
        "--n-few-shot",
        type=int,
        default=5,
        help="Number of few-shot examples for classification benchmarks (default: 5)",
    )
    parser.add_argument(
        "--max-val",
        type=int,
        default=50,
        help="Max validation examples for classification benchmarks (default: 50)",
    )
    parser.add_argument(
        "--max-test",
        type=int,
        default=None,
        help="Max TEST examples for classification benchmarks (default: None = "
             "the full test split). Cap it (e.g. 5) so the oracle/chain test "
             "evaluation fits inside the solve timeout on a slow local model — "
             "the full split (symptom2disease: 212 cases) otherwise times out.",
    )
    parser.add_argument(
        "--classify-balanced-json-fallback",
        action="store_true",
        help="classify runs only: when the evolved solve() returns a string "
             "instead of a dict, recover predictions by running the classify "
             "extraction chain (fenced -> flat regex -> string-aware "
             "brace-balanced case_<n> object, json.loads-validated) over that "
             "string at evaluation time. Default OFF — changes extraction on "
             "the classify path, so flip deliberately per-experiment.",
    )
    # Evolutionary orchestrator flags
    parser.add_argument(
        "--use-archive",
        # YAML-settable boolean (see --symmetric-trace-sampling): --no- form +
        # explicit default=False. CAUTION: --use-archive is now REQUIRED (the
        # linear orchestrator was retired); a lone --no-use-archive is refused
        # at startup. The bundled YAML enables it by default.
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use archive-based evolutionary orchestrator. NOTE: the bundled "
             "benchmark config (--benchmark-config auto, the default) turns "
             "this ON for every benchmark; --benchmark-config none restores "
             "this code default.",
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=1,
        help="Parents selected per iteration (B) (default: 1)",
    )
    parser.add_argument(
        "--beam-candidates",
        type=int,
        default=1,
        help="Children generated per parent (K) (default: 1)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=50,
        help="Max number of iterations (default: 50)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Stop after P iterations without improvement (default: 5)",
    )
    parser.add_argument(
        "--no-early-stop",
        action="store_true",
        help="5.4: never stop early on patience; run to --max-iterations (for "
             "runs that intentionally set patience >= max_iterations).",
    )
    parser.add_argument(
        "--gate-tasks",
        type=int,
        default=3,
        help="Tasks for gate check (0 = skip) (default: 3); skipped entirely "
             "under --consolidate (the default profile in the bundled "
             "benchmark config), which never runs the gate.",
    )
    parser.add_argument(
        "--gate-margin",
        type=_gate_margin,
        default=0.0,
        help="1.1: relative per-task gate threshold (score >= parent - margin). "
             "Pass 'none' to disable thresholding (the legacy liveness gate, "
             "for clean ablation). Default 0.0; skipped entirely under "
             "--consolidate (the default profile in the bundled benchmark "
             "config), which never runs the gate.",
    )
    parser.add_argument(
        "--gate-repeats",
        type=int,
        default=1,
        help="Median-over-R gate solves per task to denoise the gate (default: 1); "
             "skipped entirely under --consolidate (the default profile in the "
             "bundled benchmark config), which never runs the gate.",
    )
    parser.add_argument(
        "--eval-repeats",
        type=int,
        default=1,
        help="Median-over-R full-eval solves per task to denoise each candidate's "
             "per-task score, so selection/best/merge act on signal not one noisy "
             "draw (default: 1). NOTE: the bundled benchmark config sets this to "
             "3 for symptom2disease / lawbench_charge; --benchmark-config none "
             "restores the code default.",
    )
    parser.add_argument(
        "--eval-repeats-gate-topup",
        # YAML-settable boolean (see --symmetric-trace-sampling): --no- form +
        # explicit default=False.
        action=argparse.BooleanOptionalAction,
        default=False,
        help="F035 (§6b), modifier on --eval-repeats R>1: treat a GATE-"
             "precomputed trace as sample 0 and top it up with R-1 fresh solves "
             "(median over the union), so gate-sampled tasks are denoised at "
             "the same R as every other task. Consolidation inherit-frozen "
             "traces are never re-solved. Costs (R-1) extra full solves per "
             "gate-sampled task. Default OFF = the gate-reuse short-circuit is "
             "byte-identical.",
    )
    parser.add_argument(
        "--paired-eval",
        action="store_true",
        help="Common-random-numbers paired eval: evaluate every candidate's task "
             "under a seed derived from (--seed, task_id, repeat_index), identical "
             "across candidates, so the child-vs-parent comparison cancels shared "
             "LLM-sampler noise. Composes with --eval-repeats. NO-OP unless the "
             "backend honors a per-request seed (Azure/OpenAI yes; LM Studio no). "
             "Default off (byte-identical request payload).",
    )
    parser.add_argument(
        "--consolidate",
        # YAML-settable boolean (see --symmetric-trace-sampling): --no- form +
        # explicit default=False.
        action=argparse.BooleanOptionalAction,
        default=False,
        help="G9 targeted per-task consolidation: each candidate improves exactly "
             "ONE target task (round-robin by generation) while INHERITING every "
             "other task's per-task-best frozen score (no re-solve) — so gains are "
             "monotonic and collateral-free, and the deployable Ω_merge oracle is "
             "always materialized. Fixes the 'improve A, break B' thrash. Default off. "
             "NOTE: the bundled benchmark config (--benchmark-config auto, the "
             "default) turns this ON for every benchmark; --benchmark-config none "
             "restores this code default.",
    )
    parser.add_argument(
        "--protect-floor",
        type=float,
        default=None,
        help="6.2 per-task protection floor: the gate VETOES a candidate if any "
             "tested task drops more than this far below its parent baseline, even "
             "if another task clears (fixes the 'improve A, break B' admit). "
             "Default off (legacy pass-on-first-clear); skipped entirely under "
             "--consolidate (the default profile in the bundled benchmark "
             "config), which never runs the gate.",
    )
    parser.add_argument(
        "--no-inspiration",
        action="store_true",
        help="Disable cross-candidate inspiration in Omega prompt",
    )
    parser.add_argument(
        "--no-code-library",
        action="store_true",
        help="Ablation E2: disable code-library channel. Strips solver_lib:* "
             "from the Omega prompt and forces empty code_library on parsed output.",
    )
    parser.add_argument(
        "--no-outer-context",
        action="store_true",
        help="Ablation E3: disable inter-layer conditioning. Forces "
             "outer_context=\"\" in every MetaLayer's pre_process call.",
    )
    parser.add_argument(
        "--foster-adoption",
        # YAML-settable boolean (see --symmetric-trace-sampling): --no- form +
        # explicit default=False.
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Mechanism-0: REQUIRE the solver to CALL injected code_library "
             "helpers by name (+ wired solve() skeleton) instead of re-deriving "
             "them inline (default OFF = byte-identical helper-advertising). "
             "NOTE: the bundled benchmark config turns this ON for "
             "alphaevolve_math / symbolic_regression / algotune / arc_agi_2; "
             "--benchmark-config none restores the code default.",
    )
    parser.add_argument(
        "--force-code-library-live",
        action="store_true",
        help="Adoption probe: force the adapter's code_library_is_live() to be "
             "treated as True (un-demote) so Ω's Python helpers are staged + "
             "callable even on families that normally demote them (CO-Bench/SWE). "
             "Default OFF = byte-identical (CO-Bench still demotes).",
    )
    parser.add_argument(
        "--verified-code",
        action="store_true",
        help="Forensic #2 VERIFY-THEN-INJECT: sandbox-execute each Ω code_library "
             "helper against a held-out check (--network none) and keep it ONLY "
             "if it passes; drop dead/wrong helpers. Fuses forced adoption (the "
             "solver MUST call the verified helper) + an inline-re-derivation "
             "penalty. Default OFF = helpers injected as-is, no foster, no sandbox.",
    )
    parser.add_argument(
        "--seed-code-library",
        default=None,
        help="Path to a JSON {name: source} mapping seeded into the gen0 "
             "candidate's code_library BEFORE Ω runs, so a known-good helper is "
             "tested WITHOUT Ω authoring it. Each source passes the same "
             "utils/safety.py static AST gate as Ω code (load fails fast on a "
             "rejected source). Default None = byte-identical (no seeding; gen0 "
             "is the bare Layer1Solver baseline). The seed is a source_depth==0 "
             "contribution, so it is EXEMPTED from the CO-Bench/SWE demotion and "
             "stays live+advertised WITHOUT --force-code-library-live (only "
             "Ω-authored depth>=2 helpers still need force-live). Seeding "
             "stages+advertises only; it does NOT force adoption (add "
             "--foster-adoption / --deploy-verified-code for that).",
    )
    parser.add_argument(
        "--deploy-verified-code",
        action="store_true",
        help="DEPLOY FALLBACK (composes with --verified-code; default OFF). "
             "Post-process on the authored solve(): when a verified helper is "
             "staged but the solver's authored solve() is EMPTY (model over-"
             "reasoned) or does NOT call the helper by name (re-derived inline), "
             "REPLACE the body with a deterministic wrapper that calls the "
             "helper. HONEST SEMANTICS: this DEPLOYS verified code deterministi"
             "cally and BYPASSES model authoring on the focus task, so the "
             "resulting score equals the verified helper's score — it is the "
             "deploy fallback / feature, NOT a measurement of natural model "
             "adoption. INERT on CO-Bench unless --force-code-library-live is "
             "ALSO ON (the merged Python library is otherwise zeroed/demoted).",
    )
    parser.add_argument(
        "--regression-guard",
        # YAML-settable boolean (see --symmetric-trace-sampling): --no- form +
        # explicit default=False.
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Forensic #1: never ship a per-task regression — per-task-best := "
             "max(Ω, base seed-only resample best); the consolidate focus pick "
             "uses a median-of-R denoised base score (true highest-headroom) "
             "instead of a single noisy seed draw. Default OFF = byte-identical. "
             "NOTE: the bundled benchmark config (--benchmark-config auto, the "
             "default) turns this ON for every benchmark; --benchmark-config none "
             "restores this code default.",
    )
    parser.add_argument(
        "--regression-guard-repeats",
        type=int,
        default=3,
        help="Forensic #1: R seed-only resamples used to compute the base floor "
             "(best-of-R) and the denoised focus score (median-of-R). Only "
             "consulted when --regression-guard is ON (default: 3). Also set by "
             "the bundled benchmark config (3 in its defaults block — the same "
             "value; 1 for swe_bench_verified / terminal_bench).",
    )
    parser.add_argument(
        "--focus-current-headroom",
        # YAML-settable boolean (see --symmetric-trace-sampling): --no- form +
        # explicit default=False.
        action=argparse.BooleanOptionalAction,
        default=False,
        help="F034 (§6b), modifier on --regression-guard --consolidate: "
             "re-rank an improved task at its current per-task best so it "
             "rotates out of consolidation focus (default: frozen "
             "denoised-base ranking). NOTE: the bundled benchmark config "
             "(--benchmark-config auto, the default) turns this ON for every "
             "benchmark; --benchmark-config none restores this code default.",
    )
    parser.add_argument(
        "--temperatures",
        nargs="+",
        type=float,
        default=[0.5, 0.7, 0.9],
        help="Ω-generation temperature cycle for candidate diversity (roadmap "
             "v2 2.4) — applied to Omega generate/refine calls only, NOT to "
             "solve-time sampling (see --agentic-temperature). "
             "Default: 0.5 0.7 0.9.",
    )
    parser.add_argument(
        "--novelty-alpha",
        type=float,
        default=0.3,
        help="Exploration bonus weight in parent selection (default: 0.3)",
    )
    parser.add_argument(
        "--elite-rotation",
        # YAML-settable boolean (see --symmetric-trace-sampling): --no- form +
        # explicit default=False.
        action=argparse.BooleanOptionalAction,
        default=False,
        help="F006 (§6b): rotate the reserved-elite window in parent selection "
             "by generation, so with more distinct per-task winners than "
             "reserved slots every elite gets a reserved slot within "
             "len(elites)-1 consecutive generations (the archive-best keeps "
             "slot 0). Default OFF = byte-identical parent reservation "
             "(alphabetically-first per-task winners keep every reserved slot). "
             "Consulted only at --beam-width >= 2 (inert at the default 1, with "
             "a startup note); the bundled benchmark config leaves it unset — "
             "pair it with --beam-width 2+ or arm it in a custom config.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for evolutionary selection (default: 42)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=0,
        help="Self-debug retries per failing task (0=disabled, default: 0)",
    )
    parser.add_argument(
        "--llm-max-retries",
        type=int,
        default=3,
        help="TRANSPORT retries inside LLMClient -- connection, timeout, 5xx and "
             "throttle failures are re-issued up to this many times. Distinct "
             "from --max-retries, which is the evolutionary path's self-debug "
             "retry. Both arms must share one value, so the SRI profiles pin it "
             "and it is recorded in config.json (default: 3, the LLMConfig "
             "dataclass default it has always effectively been).",
    )
    parser.add_argument(
        "--retry-threshold",
        type=float,
        default=0.5,
        help="Score threshold for self-debug retry (default: 0.5)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a crashed run from the last checkpoint (requires --exp-name)",
    )
    # Agentic solver flags (Terminus 2-inspired)
    parser.add_argument(
        "--use-agentic",
        action="store_true",
        help="Use Terminus 2-inspired agentic solver (iterative observe-act loop) instead of single-shot",
    )
    parser.add_argument(
        "--agentic-max-turns",
        type=int,
        default=5,
        help="Max turns per task for agentic solver (default: 5)",
    )
    parser.add_argument(
        "--agentic-token-budget",
        type=int,
        default=100_000,
        help="Per-task context-size bound for the agentic solver (chars//4 "
             "estimate of the message list per call), NOT a spend cap — see "
             "--agentic-spend-budget (default: 100000)",
    )
    parser.add_argument(
        "--agentic-spend-budget",
        type=int,
        default=None,
        help="Cumulative real-spend token cap (outer LLM + inner "
             "llm()/llm_batch()) for the builtin agentic solver; default None "
             "= off. Consumed only with --use-agentic (the external spine has "
             "its own CostGuard).",
    )
    parser.add_argument(
        "--agentic-temperature",
        type=float,
        default=0.7,
        help="F060 (§6b): solve-time sampling temperature for the builtin "
             "agentic solver. Default 0.7 == the historical constructor pin "
             "(byte-identical when unset). Independent of --temperatures (the "
             "Ω-generation cycle). Consumed only with --use-agentic.",
    )
    parser.add_argument(
        "--agentic-error-hints",
        action="store_true",
        help="Stage 1 R1 (ORTHOGONAL base-agent floor-raiser, builtin "
             "AgenticSolver only): render a '### Likely cause: <class>' hint for "
             "actionable failure classes into the observation. Default OFF = "
             "byte-identical observation. Apply UNIFORMLY across baseline + Ω "
             "arms when measuring.",
    )
    parser.add_argument(
        "--agentic-preamble",
        action="store_true",
        help="Stage 1 R2 (ORTHOGONAL base-agent floor-raiser, builtin "
             "AgenticSolver only): inject a short behavioral preamble into the "
             "system prompt. Default OFF = byte-identical system prompt. Apply "
             "UNIFORMLY across baseline + Ω arms when measuring.",
    )
    parser.add_argument(
        "--within-layer-refine",
        # YAML-settable boolean (see --symmetric-trace-sampling): --no- form +
        # explicit default=False.
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Stage 2 WITHIN-LAYER REFINE: when a freshly-bred child's injection "
             "FAILS the quality gate, make ONE extra Ω call to FIX the bug in that "
             "injection (error-correction — keep the approach, fix the code) and "
             "keep the result as a NEW candidate iff it now clears the gate "
             "(monotonic; the rejected attempt is never mutated). Default OFF = "
             "byte-identical orchestration. Never fires on gen0 / depth-1 / empty "
             "injections.",
    )
    parser.add_argument(
        "--repropagation",
        action="store_true",
        help="Stage 3 DOWNWARD RE-PROPAGATION: when a depth>=3 child PLATEAUS / "
             "regresses vs its parent, regenerate an INTERMEDIATE layer's injection "
             "(d_t in [2, depth-1]) from the full-chain failure traces + the "
             "above-layer injections as downstream feedback, REPLACE that layer in "
             "a NEW monotonic candidate, re-evaluate, and log + classify a "
             "`downstream` self-repair event (error-correction vs novel). Default "
             "OFF = byte-identical orchestration. Never touches gen0 / depth-1.",
    )
    parser.add_argument(
        "--within-task-recursion",
        # YAML-settable boolean (see --symmetric-trace-sampling): --no- form +
        # explicit default=False.
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Forensic #3 WITHIN-TASK RECURSION (modifier on --consolidate; "
             "does not fire alongside --repropagation, which is scoped to "
             "non-consolidate breeding): remove the FOCUS-freeze so a deeper "
             "layer RE-WORKS the SAME task its parent worked (the Ω FOCUS directive "
             "becomes DEEPEN, the focus pick inherits the parent's fresh task, and "
             "the archive gains a depth*headroom selection term), building genuine "
             "within-task depth instead of router breadth. Default OFF = "
             "FOCUS-freeze on, round-robin focus, no depth term = byte-identical. "
             "NOTE: the bundled benchmark config (--benchmark-config auto, the "
             "default) turns this ON for every benchmark; --benchmark-config none "
             "restores this code default.",
    )
    # External-agent base solver flags (OpenHands / Terminus 2).
    # When --base-solver is openhands or terminus2 the evolutionary
    # orchestrator routes per-candidate execution through an
    # ExternalAgentSolver wired to the adapter's env-provider / scorer
    # factories. --daily-budget-usd > 0 is a hard prerequisite for those
    # backends (real-spend backstop, see §4.7 / §8.1).
    parser.add_argument(
        "--base-solver",
        default=None,
        choices=["builtin", "openhands", "terminus2"],
        help="Per-candidate base solver. 'builtin' (default None) keeps the "
             "legacy Layer1/agentic path; 'openhands'/'terminus2' route through "
             "ExternalAgentSolver. Requires --daily-budget-usd > 0 for the "
             "external backends.",
    )
    parser.add_argument(
        "--max-docker",
        type=int,
        default=None,
        help="Max concurrent inner Docker containers for external agents "
             "(default: min(--parallel, 4)). Clamped to <= --parallel.",
    )
    parser.add_argument(
        "--agent-time-limit",
        type=int,
        default=1200,
        help="Per-run wall-clock envelope (seconds) for external agents "
             "(default: 1200).",
    )
    parser.add_argument(
        "--agent-max-budget",
        type=float,
        default=0.5,
        help="Per-task USD budget for external agents — a DECLARED / pre-check "
             "ceiling, NOT a hard per-run stop. No backend enforces a per-run USD "
             "kill (OpenHands records max_budget_per_task without enforcing it; "
             "Terminus 2 never forwards a USD budget), so an admitted run may "
             "overshoot this value; the daily cap (--daily-budget-usd) is the real "
             "backstop, enforced at per-candidate admission + the generation "
             "boundary. Default: 0.5.",
    )
    parser.add_argument(
        "--scratch-root",
        default=None,
        help="Fast-disk root for external-agent lease workdirs "
             "(default: system temp).",
    )
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def _benchmark_config_path(choice: str | None) -> Path | None:
    """Resolve ``--benchmark-config`` to a YAML path, or ``None`` to skip.

    ``None`` / ``"none"`` → ``None`` (skip entirely: bare code defaults).
    ``"auto"`` / ``""``   → the bundled ``configs/benchmark_features.yaml``
                            (package-relative to this file:
                            ``Path(__file__).resolve().parent`` — shipped as
                            package data, so installed wheels resolve it too).
    anything else         → that literal path.
    """
    token = "none" if choice is None else str(choice).strip()
    if token.lower() == "none":
        return None
    if token.lower() in ("auto", ""):
        return Path(__file__).resolve().parent / "configs" / "benchmark_features.yaml"
    return Path(token)


def _user_set_dests(parser: argparse.ArgumentParser, argv: list[str]) -> set[str]:
    """Dests the user EXPLICITLY set on the CLI.

    A dest counts as user-set when any of its action's option strings appears
    in ``argv`` — either bare (``--eval-repeats 3``) or in the joined
    ``--eval-repeats=3`` form. Used so a CLI-explicit flag is never overwritten
    by the benchmark-features YAML.
    """
    def _present(opt: str) -> bool:
        return any(tok == opt or tok.startswith(opt + "=") for tok in argv)

    return {
        action.dest
        for action in parser._actions
        if any(_present(opt) for opt in (action.option_strings or []))
    }


def _validated_config_value(key: str, val, action):
    """Validate/coerce ONE benchmark-config value against its argparse action.

    ``action is None`` (no parser supplied, or a dest with no matching action)
    passes the value through unchanged — the lenient ``hasattr`` philosophy of
    :func:`apply_benchmark_config`. With an action, a wrong-typed YAML value
    raises ``ValueError`` (fail fast at startup) instead of being setattr'd
    silently — most importantly the string ``'false'`` on a store_true dest,
    which would otherwise read as truthy and INVERT the flag downstream.
    """
    if action is None:
        return val
    # Boolean flags FIRST (store_true/BooleanOptionalAction carry nargs=0, so
    # the list-dest short-circuit below must not swallow them): the string
    # 'false' would otherwise land truthy and INVERT the flag. The YAML-settable
    # booleans are BooleanOptionalAction (they carry a --no- opt-out form).
    if isinstance(
        action,
        (
            argparse._StoreTrueAction,
            argparse._StoreFalseAction,
            argparse.BooleanOptionalAction,
        ),
    ):
        if not isinstance(val, bool):
            raise ValueError(
                f"benchmark-config: key {key!r} expects a YAML boolean, "
                f"got {type(val).__name__} {val!r}"
            )
        return val
    # nargs dests (e.g. --temperatures nargs='+') take lists — no coercion.
    if action.nargs is not None:
        return val
    # Nullable dests (protect_floor / max_docker / agentic_spend_budget ...):
    # a YAML null on a default-None dest is the code default, pass as-is.
    if val is None and action.default is None:
        return val
    if action.type in (int, float):
        # bool FIRST: bool is an int subclass, so `regression_guard_repeats:
        # true` must be rejected, not silently become 1.
        if isinstance(val, bool):
            raise ValueError(
                f"benchmark-config: key {key!r} expects a YAML "
                f"{action.type.__name__}, got bool {val!r}"
            )
        if isinstance(val, action.type):
            return val
        if action.type is float and isinstance(val, int):
            return float(val)
        if isinstance(val, str):
            # Mirror argparse CLI semantics: '3' -> 3; 'three' fails fast.
            try:
                return action.type(val)
            except (ValueError, TypeError):
                raise ValueError(
                    f"benchmark-config: key {key!r} expects a YAML "
                    f"{action.type.__name__}, got str {val!r}"
                ) from None
        raise ValueError(
            f"benchmark-config: key {key!r} expects a YAML "
            f"{action.type.__name__}, got {type(val).__name__} {val!r}"
        )
    if action.type is not None and callable(action.type):
        # Non-class converter (e.g. --gate-margin's _gate_margin): coerce only
        # str values via the callable; pass anything else through unvalidated.
        # bool FIRST: PyYAML reads on/off/yes/no as booleans, and e.g. a raw
        # False on gate_margin is not-None downstream — the inverse of the
        # 'disable thresholding' the operator asked for.
        if isinstance(val, bool):
            raise ValueError(
                f"benchmark-config: key {key!r}: got YAML bool {val!r} "
                f"(PyYAML parses on/off/yes/no as booleans) — write a real "
                f"number, or a quoted string such as 'none'"
            )
        if isinstance(val, str):
            try:
                return action.type(val)
            except (ValueError, TypeError) as e:
                raise ValueError(
                    f"benchmark-config: key {key!r}: invalid value "
                    f"{val!r} ({e})"
                ) from None
        return val
    return val


#: The ONLY dests a benchmark-features YAML may set when parser validation is
#: armed (the production call site): the metacognition booleans + companion
#: ints the file header promises, plus the gate-family knobs its NOTE documents
#: for custom non-consolidate profiles ({consolidate: false, protect_floor:
#: 0.05, gate_tasks: ...}). Everything else that happens to be a real args
#: attr (model / api_key / resume / daily_budget_usd / ...) is warn+skipped —
#: run identity and credentials are CLI/env-only, never YAML-settable.
_BENCHMARK_CONFIG_SETTABLE_DESTS = frozenset({
    # metacognition booleans (all BooleanOptionalAction, default False)
    "use_archive", "consolidate", "regression_guard", "within_task_recursion",
    "within_layer_refine", "elite_rotation", "focus_current_headroom",
    "symmetric_trace_sampling", "eval_repeats_gate_topup", "foster_adoption",
    # companion ints (beam_width is elite_rotation's documented pairing: the
    # file header directs custom profiles to raise it in the SAME block)
    "regression_guard_repeats", "eval_repeats", "beam_width",
    # gate-family knobs: consumed only by the quality gate (which consolidate
    # mode skips) — settable so a custom per-bench block can pair
    # {consolidate: false} with a strict gate profile.
    "protect_floor", "gate_tasks", "gate_margin", "gate_repeats",
})


def apply_benchmark_config(
    args, user_set_dests, cfg_dict, *, warn=None, parser=None
) -> list[str]:
    """Apply a benchmark-features config onto ``args`` IN PLACE.

    Precedence (highest first): a dest the user set explicitly (in
    ``user_set_dests``) is NEVER overwritten; otherwise the merged block
    (``defaults`` updated by the ``args.benchmark`` per-benchmark block) sets it;
    otherwise the code/argparse default stands. Pure apart from the in-place
    ``setattr`` (no I/O, no parsing) so it is trivially unit-testable.

    Applies ONLY when ``args.benchmark`` is set: the file is per-BENCHMARK by
    contract, so on the ``--tasks`` (benchmark ``None``) path — and for
    hand-built namespaces lacking the attr — it returns ``[]`` and leaves
    ``args`` untouched (the historical pre-metacognition baseline).

    Never raises on a missing/unknown benchmark, on a YAML key that is not a
    real ``args`` attribute, or on a structurally malformed document (non-
    mapping top level / ``defaults`` block / per-benchmark block) — such keys
    and blocks are skipped (``warn`` is called with the skipped key name or a
    ``<block: ...>`` token when provided), and a valid ``defaults`` block still
    applies when only the per-benchmark block is malformed. Returns the SORTED
    list of dests actually applied (for logging / config.json provenance).

    ``parser`` (keyword-only, default ``None``) opts in to per-key validation
    against the parser's actions: a key outside
    :data:`_BENCHMARK_CONFIG_SETTABLE_DESTS` is warn+skipped (YAML can never
    set ``model`` / ``api_key`` / ``resume`` ...), and a wrong-typed YAML value
    raises ``ValueError`` BEFORE setattr (see :func:`_validated_config_value`),
    so a custom config fails at startup instead of silently misconfiguring the
    run. ``parser=None`` preserves the lenient parserless behavior
    byte-identically (hand-built namespaces).
    """
    if not cfg_dict:
        return []
    if not isinstance(cfg_dict, dict):
        # Structural malformation (e.g. a top-level YAML list): warn-and-degrade
        # to bare code defaults, same contract as the parse-error path.
        if warn is not None:
            warn(f"<top-level: expected mapping, got {type(cfg_dict).__name__}>")
        return []
    bench = getattr(args, "benchmark", None)
    if not bench:
        # Per-BENCHMARK file: the no-benchmark (--tasks / hand-built-namespace)
        # path keeps pre-YAML bare semantics. Blocks are keyed by benchmark
        # name, so this case could never be targeted from the YAML anyway.
        return []
    defaults = cfg_dict.get("defaults") or {}
    if not isinstance(defaults, dict):
        if warn is not None:
            warn(f"<defaults: expected mapping, got {type(defaults).__name__}>")
        defaults = {}
    per_bench = cfg_dict.get(bench) or {}
    if not isinstance(per_bench, dict):
        if warn is not None:
            warn(f"<{bench}: expected mapping, got {type(per_bench).__name__}>")
        per_bench = {}
    merged = {**defaults, **per_bench}
    actions_by_dest = (
        {a.dest: a for a in parser._actions} if parser is not None else {}
    )
    applied: list[str] = []
    for key, val in merged.items():
        if key in user_set_dests:
            continue  # CLI-explicit wins
        if not hasattr(args, key):
            # Unknown arg attr (typo'd YAML key, or a stale hand-built
            # namespace): skip, never raise.
            if warn is not None:
                warn(key)
            continue
        if parser is not None and key not in _BENCHMARK_CONFIG_SETTABLE_DESTS:
            # Real args attr, but not a YAML-settable dest (model / api_key /
            # resume ...): warn+skip, never setattr (same posture as the
            # unknown-key path — a benign extra key must not kill the run,
            # but YAML must never steer run identity or credentials).
            if warn is not None:
                warn(f"{key} (not settable via benchmark-config)")
            continue
        setattr(args, key, _validated_config_value(key, val, actions_by_dest.get(key)))
        applied.append(key)
    return sorted(applied)


def load_seed_code_library(path: str | None) -> dict[str, str] | None:
    """Load + validate a P1c ``--seed-code-library`` JSON {name: source} mapping.

    Returns ``None`` for a ``None`` path (no seeding ⇒ byte-identical baseline).
    Each source passes the same ``utils/safety.validate_code`` static AST gate Ω
    code passes; an invalid source raises ``ValueError`` at LOAD time (fail fast)
    so a broken/unsafe seed never reaches Docker staging.
    """
    if not path:
        return None
    from meta_n.utils.safety import validate_code

    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    if not isinstance(raw, dict):
        raise ValueError(
            f"--seed-code-library {path}: expected a JSON object "
            f"{{name: source}}, got {type(raw).__name__}"
        )
    seeded: dict[str, str] = {}
    for name, source in raw.items():
        if not isinstance(name, str) or not isinstance(source, str):
            raise ValueError(
                f"--seed-code-library {path}: every key and value must be a "
                f"string (offending key={name!r})"
            )
        ok, msg = validate_code(source)
        if not ok:
            raise ValueError(
                f"--seed-code-library {path}: helper '{name}' failed safety "
                f"validation: {msg}"
            )
        seeded[name] = source
    return seeded


def load_tasks(path: str) -> list[TaskDescription]:
    """Load tasks from JSON file."""
    with open(path) as f:
        data = json.load(f)
    return [TaskDescription(**t) for t in data]


def _resolve_backend_and_model(model_spec: str) -> tuple[str, str]:
    """Parse a ``--model`` argument into ``(backend, deployment_or_model)``.

    Conventions:
      * ``azure/<deployment>`` → Azure backend with the given deployment.
        The deployment name is what Azure expects as the ``model`` field
        on chat.completions.create, and is also the pricing key. Name
        deployments after the model family (``gpt-4.1``, ``gpt-5.2``).
      * Anything else → OpenRouter / OpenAI-compatible (legacy default).
    """
    if model_spec.startswith("azure/"):
        return "azure", model_spec[len("azure/"):]
    return "openrouter", model_spec


def _resolve_budget(args: argparse.Namespace, backend: str) -> tuple[float, str]:
    """Resolve daily USD cap and ledger dir from CLI > env > default.

    Default cap depends on backend: 500 USD for Azure, 0 (off) elsewhere.
    Original $295 default sat $5 below the $300 group-key ceiling. After
    the TPM bump in May 2026 the key supports a higher daily envelope;
    the project-level $1450 budget remains the hard ceiling and is
    enforced by the operator (not the per-process cap).
    """
    if args.daily_budget_usd is not None:
        cap = float(args.daily_budget_usd)
    elif os.environ.get("META_N_DAILY_BUDGET_USD"):
        cap = float(os.environ["META_N_DAILY_BUDGET_USD"])
    else:
        cap = 500.0 if backend == "azure" else 0.0
    if args.cost_ledger_dir:
        ledger_dir = args.cost_ledger_dir
    else:
        ledger_dir = os.environ.get("META_N_COST_LEDGER_DIR", "~/.meta_n_costs")
    return cap, ledger_dir


def foster_adoption_noop_reasons(args, adapter, external_spine) -> list[str]:
    """Reasons why ``--foster-adoption`` would be a silent no-op for THIS run.

    ``foster_adoption`` is consumed ONLY on the native Layer1Solver /
    ``MetaLayer`` path (``MetaLayer(foster_adoption=...)``). The external-agent
    spine branch and the ``AgenticSolver`` branch never receive it, and on a
    demoted-helper benchmark family the merged Python helpers are emptied
    before the ``MetaLayer`` is built (nothing to foster) unless
    ``--force-code-library-live`` un-demotes them. Returns a (possibly empty)
    list of human-readable reasons; an empty list means ``foster_adoption`` is
    actually wired in for this invocation. Pure / side-effect-free so it is
    unit-testable without spinning up an orchestrator.
    """
    reasons: list[str] = []
    if getattr(args, "use_agentic", False):
        reasons.append(
            "--use-agentic (the agentic solver path does not consume "
            "foster_adoption; only the native Layer1Solver/MetaLayer path does)"
        )
    if external_spine:
        reasons.append(
            "an external base-solver (the external-agent spine ignores "
            "foster_adoption; execution never reaches a MetaLayer)"
        )
    # Demoted-helper family without --force-code-library-live: the merged
    # Python helpers are emptied before the MetaLayer is built, so there is
    # nothing left for foster_adoption to require the solver to call.
    if not getattr(args, "force_code_library_live", False):
        try:
            live = getattr(adapter, "code_library_is_live", lambda: True)()
        except Exception:  # noqa: BLE001 - advisory only, never blocks startup
            live = True
        if adapter is not None and not live:
            reasons.append(
                "a demoted-helper benchmark family (code_library_is_live() is "
                "False, so the Python helpers are emptied before the MetaLayer "
                "is built); pass --force-code-library-live to keep them live"
            )
    return reasons


def deploy_verified_code_noop_note(args, solver_language: str) -> str | None:
    """R5: note when the ``MetaLayer`` language gate makes the flag inert.

    The deploy wrapper is Python source (``def solve(**kw): ...``), so
    ``MetaLayer._maybe_deploy_verified_helper`` deploys ONLY on
    python-language layers; bash/openevolve authored scripts pass through
    untouched. Returns a note (F229 house style: text starts with the flag
    token) or ``None`` when the flag is off or actually wired in.
    """
    if not getattr(args, "deploy_verified_code", False):
        return None
    if solver_language == "python":
        return None
    return (
        "--deploy-verified-code (the deploy wrapper is Python source; the "
        f"{solver_language}-language MetaLayer passes authored scripts "
        "through untouched)"
    )


def annotate_config_sourced(flag_texts: list[str], args) -> list[str]:
    """Suffix guard/no-op flag texts whose value came from the benchmark-
    features YAML (not the CLI) with their source file.

    :func:`evolutionary_noop_flag_notes` names flags the operator supposedly
    set — but under the default ``--benchmark-config auto`` a value may have
    been set by the YAML, and the warning should point at the file, not at a
    flag the operator never typed. Every entry it emits begins with one or more long
    CLI flag tokens (a joined note names several, comma-separated), each
    mapping to its argparse dest via ``rstrip(',')`` + ``lstrip('-')`` +
    ``'-'→'_'``; entries none of whose leading dests are in
    ``args.benchmark_config_applied`` pass through unchanged. A CLI-explicit
    flag is excluded from ``benchmark_config_applied`` by
    :func:`apply_benchmark_config` (user_set wins), so it can never be
    falsely attributed. With no applied entries (or the provenance attrs
    absent — bare path / hand-built namespaces) the output equals the input,
    keeping console bytes identical.
    """
    applied = set(getattr(args, "benchmark_config_applied", None) or [])
    src = getattr(args, "benchmark_config_resolved", "none")
    out: list[str] = []
    for text in flag_texts:
        dests: list[str] = []
        for tok in text.split(" "):
            if not tok.startswith("--"):
                break
            dests.append(tok.rstrip(",").lstrip("-").replace("-", "_"))
        out.append(
            f"{text} (set by benchmark-config: {src})"
            if any(d in applied for d in dests)
            else text
        )
    return out


def evolutionary_noop_flag_notes(args, external_spine: bool) -> list[str]:
    """Silent no-op evolutionary flags for THIS run's solver path (F229).

    Warn-only (``foster_adoption_noop_reasons`` house style): the agentic /
    external-spine knobs are consumed only by specific solver paths, so a
    non-default value on an ``--use-archive`` run that never engages that path
    is a silent no-op. The raw values stay recorded in config.json (deliberate
    — the gating flags ``use_agentic`` / ``base_solver`` are recorded alongside,
    so a reader can determine inertness). Returns ``[]`` off the archive path
    (a non-archive run is refused at startup — the linear orchestrator is
    retired). Pure / side-effect-free so it is unit-testable.
    """
    if not getattr(args, "use_archive", False):
        return []
    notes: list[str] = []
    use_agentic = getattr(args, "use_agentic", False)
    if not use_agentic:
        for cli, attr in (
            ("--agentic-error-hints", "agentic_error_hints"),
            ("--agentic-preamble", "agentic_preamble"),
        ):
            if getattr(args, attr, False):
                notes.append(
                    f"{cli} (consumed only by the builtin AgenticSolver "
                    f"(--use-agentic); the external spine never reads it)"
                )
        # F060 (§6b): solve-time temperature — builtin AgenticSolver only
        # (float default, so it does not fit the store_true loop above).
        if float(getattr(args, "agentic_temperature", 0.7)) != 0.7:
            notes.append(
                "--agentic-temperature (consumed only by the builtin "
                "AgenticSolver (--use-agentic); the external spine never "
                "reads it)"
            )
        # F063 (§6b): spend_budget is consumed ONLY by the builtin
        # AgenticSolver, so — like the temperature note above — this fires on
        # external-spine runs too (the spine has its own CostGuard).
        # None-default flag — is-not-None check (a store_true/int loop cannot
        # express a None default).
        if getattr(args, "agentic_spend_budget", None) is not None:
            notes.append(
                "--agentic-spend-budget (consumed only by --use-agentic; the "
                "external spine has its own CostGuard)"
            )
    if not use_agentic and not external_spine:
        for cli, attr, default in (
            ("--agentic-max-turns", "agentic_max_turns", 5),
            ("--agentic-token-budget", "agentic_token_budget", 100_000),
        ):
            if int(getattr(args, attr, default)) != default:
                notes.append(
                    f"{cli} (consumed only by --use-agentic or an external "
                    f"base-solver)"
                )
    if not external_spine:
        spine_note = (
            "consumed only when the external-agent spine is engaged "
            "(--base-solver openhands/terminus2, or builtin on terminal_bench)"
        )
        if int(getattr(args, "agent_time_limit", 1200)) != 1200:
            notes.append(f"--agent-time-limit ({spine_note})")
        if float(getattr(args, "agent_max_budget", 0.5)) != 0.5:
            notes.append(f"--agent-max-budget ({spine_note})")
        if getattr(args, "max_docker", None) is not None:
            notes.append(f"--max-docker ({spine_note})")
        if getattr(args, "scratch_root", None) is not None:
            notes.append(f"--scratch-root ({spine_note})")
    # R1-D_selection-4: evolutionary-strategy MODIFIER flags that are inert
    # without their base flag on the archive path (CLI-visible inertness, warn-
    # only — F229 house style).
    if getattr(args, "focus_current_headroom", False) and not (
        getattr(args, "regression_guard", False)
        and getattr(args, "consolidate", False)
    ):
        notes.append(
            "--focus-current-headroom (modifier consumed only under "
            "--regression-guard --consolidate; a base flag is absent)"
        )
    if getattr(args, "within_task_recursion", False) and not getattr(
        args, "consolidate", False
    ):
        notes.append(
            "--within-task-recursion (modifier consumed only under "
            "--consolidate; the DEEPEN directive, focus inheritance and the "
            "archive depth term are all dormant without it)"
        )
    if getattr(args, "eval_repeats_gate_topup", False):
        if int(getattr(args, "eval_repeats", 1) or 1) <= 1:
            notes.append(
                "--eval-repeats-gate-topup (modifier consumed only with "
                "--eval-repeats R>1)"
            )
        elif getattr(args, "consolidate", False):
            # Y4-S_stack-3: the top-up denoises REUSED GATE traces only, and
            # consolidate skips the gate (every child carries a focus task) —
            # dead even at R>1.
            notes.append(
                "--eval-repeats-gate-topup (tops up reused gate traces only; "
                "--consolidate skips the candidate gate entirely, so no gate "
                "trace ever exists to top up)"
            )
    # Y4-S_stack-1: within_layer_refine's ONLY entry point is the gate-FAIL
    # branch; it is dead when the gate never runs (consolidate mode) or is
    # explicitly skipped (gate_tasks<=0).
    if getattr(args, "within_layer_refine", False):
        if getattr(args, "consolidate", False):
            notes.append(
                "--within-layer-refine (consumed only in the quality-gate FAIL "
                "branch; --consolidate skips the gate — every child carries a "
                "focus task — so the refine hook can never fire)"
            )
        elif int(getattr(args, "gate_tasks", 3) or 0) <= 0:
            notes.append(
                "--within-layer-refine (consumed only in the quality-gate FAIL "
                "branch; --gate-tasks<=0 skips the gate, so the refine hook "
                "can never fire)"
            )
    if int(getattr(args, "regression_guard_repeats", 3) or 3) != 3 and not getattr(
        args, "regression_guard", False
    ):
        notes.append(
            "--regression-guard-repeats (consulted only when "
            "--regression-guard is ON)"
        )
    # Y4-P_benchmarks-8: elite_rotation is consulted only inside the n>=2
    # reserved-elite block of select_parents; the default --beam-width 1 never
    # reaches it. Gate is < 2, NOT < 3: at beam_width=2 the flag IS consulted
    # (it changes the reserved slot whenever the archive-best sits outside the
    # breedable pool), so a note there would be a false no-op claim.
    if getattr(args, "elite_rotation", False) and int(
        getattr(args, "beam_width", 1)
    ) < 2:
        notes.append(
            "--elite-rotation (consumed only inside the n>=2 reserved-elite "
            "block of select_parents; --beam-width 1 never reaches it — "
            "full rotation coverage needs --beam-width >= 3)"
        )
    # R1-E_omega_context-2: --no-outer-context is consumed on the native
    # MetaLayer path AND (since Y4-C_callability-5 wired AgenticSolver) the
    # agentic path — only the external spine never reaches either consumer.
    # At most one note (if/elif chain); external_spine takes precedence over
    # the depth check because the spine branch of _build_solver_from_candidate
    # wins the routing.
    if getattr(args, "no_outer_context", False):
        if external_spine:
            notes.append(
                "--no-outer-context (consumed only on the native MetaLayer / "
                "agentic paths; the external-agent spine never reaches either)"
            )
        elif int(getattr(args, "max_depth", 10)) <= 2:
            # Applies to native AND agentic: a depth-2 candidate carries a
            # single injected layer, so there is no inter-layer outer_context
            # to thread on either path (the first block always sees "").
            notes.append(
                "--no-outer-context (needs an inter-layer outer_context to "
                "ablate; --max-depth<=2 caps candidates at one injected layer, "
                "so the flag is inert)"
            )
    # R3-A_code_channel-2: --force-code-library-live toggles the native/agentic
    # demotion gate _code_library_is_live(); the external-agent spine never
    # consults it — its InjectionMapper always stages every safe Ω helper — so
    # the flag is inert on the spine. Native/agentic paths honor it (no note).
    if getattr(args, "force_code_library_live", False) and external_spine:
        notes.append(
            "--force-code-library-live (toggles the native/agentic demotion "
            "gate _code_library_is_live(); the external-agent spine never "
            "consults it — its InjectionMapper always stages every safe Ω "
            "helper regardless — so the flag has no effect on this run)"
        )
    # Consolidate mode moots two flag families: every consolidation child
    # carries a focus task, so the per-candidate quality gate (the ONLY
    # consumer of the gate knobs) never runs, and the re-propagation hook
    # (scoped to NON-consolidate breeding) never fires. Warn-only; every
    # clause is gated on the flag's own non-default value so the default
    # path prints nothing (getattr fallbacks keep hand-built namespaces
    # counting as default).
    if getattr(args, "consolidate", False):
        if getattr(args, "repropagation", False):
            notes.append(
                "--repropagation (scoped to NON-consolidate breeding; every "
                "consolidate candidate carries a focus_task so the reprop hook "
                "never fires; consolidate is ON by default via the "
                "benchmark-features YAML — use a custom YAML or "
                "--benchmark-config none to run it)"
            )
        _gate_flags: list[str] = []
        # gate_tasks=0 is excluded: an explicit gate skip is trivially
        # honored under consolidate, not a misleading no-op.
        if int(getattr(args, "gate_tasks", 3)) not in (3, 0):
            _gate_flags.append("--gate-tasks")
        if int(getattr(args, "gate_repeats", 1) or 1) != 1:
            _gate_flags.append("--gate-repeats")
        # 'none'/'off' → None counts as non-default.
        _cgm = getattr(args, "gate_margin", 0.0)
        if _cgm is None or float(_cgm) != 0.0:
            _gate_flags.append("--gate-margin")
        if getattr(args, "protect_floor", None) is not None:
            _gate_flags.append("--protect-floor")
        if _gate_flags:
            notes.append(
                ", ".join(_gate_flags)
                + " (the candidate gate is skipped entirely in --consolidate "
                "mode — the gate block runs only when no FOCUS task is set "
                "and consolidation always sets one — so these values are "
                "never consumed)"
            )
    return notes


def evolutionary_run_kwargs(args) -> dict:
    """The evolutionary-only key/value pairs shared by ``EvolutionaryConfig``
    and the ``--use-archive`` config.json provenance block (F048).

    Every key is a real ``EvolutionaryConfig`` field name; insertion order is
    FROZEN (it reproduces the historical ``run_config.update`` literal order,
    with the F031-wired keys appended) because ``json.dump`` preserves dict
    order — reordering would perturb config.json byte-identity. The single
    deliberate divergence between the two consumers is ``seed_code_library``:
    provenance records the PATH (this dict), while ``EvolutionaryConfig``
    receives the parsed+validated mapping (overridden at the call site).
    """
    return {
        "beam_width": args.beam_width,
        "beam_candidates": args.beam_candidates,
        "max_iterations": args.max_iterations,
        "patience": args.patience,
        "gate_tasks": args.gate_tasks,
        "gate_repeats": args.gate_repeats,
        "eval_repeats": args.eval_repeats,
        "paired_eval": args.paired_eval,
        "consolidate": args.consolidate,
        "protect_floor": args.protect_floor,
        "use_inspiration": not args.no_inspiration,
        "no_code_library": args.no_code_library,
        "no_outer_context": args.no_outer_context,
        "foster_adoption": args.foster_adoption,
        "force_code_library_live": args.force_code_library_live,
        "verified_code": args.verified_code,
        # #14: these two behavioral flags genuinely change orchestrator
        # behavior (deploy-fallback / gen0 library seeding) but had no
        # provenance channel. Default OFF (False / None) keeps config.json
        # byte-identical on the default path.
        "deploy_verified_code": args.deploy_verified_code,
        "seed_code_library": args.seed_code_library,
        "regression_guard": args.regression_guard,
        "regression_guard_repeats": args.regression_guard_repeats,
        "novelty_alpha": args.novelty_alpha,
        "seed": args.seed,
        "use_agentic": args.use_agentic,
        "agentic_max_turns": args.agentic_max_turns,
        "agentic_token_budget": args.agentic_token_budget,
        "agentic_error_hints": args.agentic_error_hints,
        "agentic_preamble": args.agentic_preamble,
        "within_layer_refine": args.within_layer_refine,
        "repropagation": args.repropagation,
        "within_task_recursion": args.within_task_recursion,
        "base_solver": args.base_solver,
        "agentic_time_limit_s": args.agent_time_limit,
        "agentic_max_budget_usd": args.agent_max_budget,
        "max_docker": args.max_docker,
        "scratch_root": args.scratch_root,
        # F031: newly CLI-wired knobs, appended (new provenance keys). getattr
        # keeps stale wrapper namespaces (scripts/meta_n_ag_gpt52_*.py) valid.
        "no_early_stop": getattr(args, "no_early_stop", False),
        "gate_margin": getattr(args, "gate_margin", 0.0),
        "temperatures": getattr(args, "temperatures", None) or [0.5, 0.7, 0.9],
        # Refine §6b F006: appended (new provenance key — F031 precedent);
        # getattr keeps stale wrapper namespaces valid.
        "elite_rotation": getattr(args, "elite_rotation", False),
        # F034 (§6b): appended provenance key.
        "focus_current_headroom": getattr(args, "focus_current_headroom", False),
        # F035 (§6b): appended provenance key.
        "eval_repeats_gate_topup": getattr(args, "eval_repeats_gate_topup", False),
        # F063/F060 (§6b): appended provenance keys (F031 pattern).
        "agentic_spend_budget": getattr(args, "agentic_spend_budget", None),
        "agentic_temperature": getattr(args, "agentic_temperature", 0.7),
    }


#: F197: the four OpenEvolve-family benchmarks share one construction shape —
#: benchmark -> (module, adapter class, default data_dir, task-filter kwarg).
#: Default dirs are carried verbatim here (do NOT rely on the adapters'
#: constructor defaults, even though they currently match). Imports stay lazy
#: at the use site (subprocess-bridge invariant untouched).
_OPENEVOLVE_FAMILY: dict[str, tuple[str, str, str, str]] = {
    "alphaevolve_math": (
        "meta_n.integrations.openevolve", "AlphaEvolveMathAdapter",
        "./data/openevolve/examples/alphaevolve_math_problems", "problem_names",
    ),
    "symbolic_regression": (
        "meta_n.integrations.openevolve", "SymbolicRegressionAdapter",
        "./data/openevolve/examples/symbolic_regression/problems", "problem_names",
    ),
    "algotune": (
        "meta_n.integrations.openevolve", "AlgoTuneAdapter",
        "./data/openevolve/examples/algotune", "task_names",
    ),
    "arc_agi_2": (
        "meta_n.integrations.arc_agi", "ARCAGI2Adapter",
        "./data/arc_agi_2", "task_ids",
    ),
}


def build_base_run_config(
    args, *, benchmark_name, solver_language, executor_name
) -> dict:
    """The base run_config keys shared by every run (H8).

    The evolutionary-only ablation provenance keys (``no_code_library`` /
    ``no_outer_context`` / ``foster_adoption`` / ``force_code_library_live``)
    are added by the archive block via :func:`evolutionary_run_kwargs`, not
    here, so this stays the minimal base. ``bench_tasks`` / ``seed`` ARE
    retained because task filtering happens upstream regardless.
    """
    return {
        "model": args.model,
        "base_url": args.base_url,
        "max_tokens": args.max_tokens,
        "empty_retry_max_tokens": args.empty_retry_max_tokens,
        "epsilon": args.epsilon,
        "max_depth": args.max_depth,
        "parallel": args.parallel,
        "tasks_file": args.tasks,
        "benchmark": benchmark_name,
        "solver_language": solver_language,
        "executor": executor_name,
        "timestamp": datetime.now().isoformat(),
        "n_few_shot": args.n_few_shot,
        "max_val": args.max_val,
        "max_retries": args.max_retries,
        # Transport retries (LLMClient), as distinct from max_retries above.
        # Recorded so the SRI runner's verify_run_config can prove both arms
        # ran the same retry policy instead of inheriting a dataclass default.
        # getattr: same reason as reasoning_effort -- hand-built arg namespaces
        # (gpt52 drivers, test helpers) predate the flag. Real runs always carry
        # it, because profile.render_pinned_flags emits it.
        "llm_max_retries": getattr(args, "llm_max_retries", 3),
        "retry_threshold": args.retry_threshold,
        # F6: bench task selection is provenance recorded in the base config;
        # the evolutionary-only ablation flags are added inside the archive
        # block (H8).
        "bench_tasks": args.bench_tasks,
        # #27: the seed deterministically selects WHICH tasks run
        # (terminal_bench / swe_bench load_tasks(seed_shuffle=args.seed)), so it
        # is task-subset provenance recorded in the base config.
        "seed": args.seed,
        # #62: result-affecting LLM/Omega knobs consumed on BOTH paths but
        # previously unrecorded. All default to a byte-identical no-op (None / 0),
        # so config.json on the default path is unchanged.
        "omega_context_budget": args.omega_context_budget,
        "exclude_providers": args.exclude_providers,
        "request_timeout": args.request_timeout,
        "instance_workers": args.instance_workers,
        # Reasoning-model control + test-split cap (result-affecting LLM/eval
        # knobs on BOTH paths; #62 treatment). getattr keeps hand-built arg
        # namespaces valid; defaults (None) leave config.json byte-identical.
        "reasoning_effort": getattr(args, "reasoning_effort", None),
        "max_test": getattr(args, "max_test", None),
        # F156 (§6b): result-affecting Omega-sampler knob, consumed on BOTH
        # paths via the shared OmegaEngine (#62 treatment). Additive provenance
        # key, default False; config.json is write-only (never read on
        # --resume) — absence == False on old configs.
        "symmetric_trace_sampling": getattr(args, "symmetric_trace_sampling", False),
        # F075 (§6b): result-affecting classify-extraction knob, consumed by
        # BOTH orchestrator paths via Layer1Solver (#62 treatment). Base-config
        # provenance like solver_language, NOT an EvolutionaryConfig field.
        "classify_balanced_json_fallback": args.classify_balanced_json_fallback,
        # Benchmark-features provenance (additive, BOTH paths). Sentinel
        # grammar for benchmark_config: a real path = the file was found AND
        # parsed (benchmark_config_applied may still be [] — e.g. an empty
        # file or a no-benchmark run); 'none' = operator opt-out
        # (--benchmark-config none), a --tasks-scoped skip, or a hand-built
        # namespace; 'parse-error:<path>' = the file existed but did not
        # parse to a mapping (run proceeded on bare code defaults). A missing
        # file never reaches here (startup hard-error). getattr keeps
        # hand-built namespaces valid (default: skipped).
        "benchmark_config": getattr(args, "benchmark_config_resolved", "none"),
        "benchmark_config_applied": getattr(args, "benchmark_config_applied", []),
    }


async def async_main(args: argparse.Namespace):
    """Async entry point."""
    # T2.5: --local-exec was never wired into executor selection (each benchmark
    # builds its own executor), so honoring it would require a per-adapter change
    # that some Docker-hard benchmarks cannot satisfy. Rather than let it remain a
    # silent no-op, refuse to start when it is passed. Inert (byte-identical) on
    # the default path where the flag is absent.
    # F032: every startup-validation refusal returns 1 (main() → exit code 1,
    # distinct from budget's 2 / argparse's 2-for-bad-syntax); success paths
    # fall off the end (None → exit 0).
    if getattr(args, "local_exec", False):
        console.print(
            "[red]Error: --local-exec is not supported — it is not wired into "
            "executor selection (each benchmark builds its own executor, and "
            "Docker-backed benchmarks such as terminal_bench / swe_bench require "
            "Docker). Remove --local-exec; it would otherwise be a silent "
            "no-op.[/red]"
        )
        return 1

    # --- Benchmark-features config (metacognition stack, ON by default) ---
    # Applied AFTER args.benchmark is resolved but BEFORE adapter construction /
    # evolutionary_run_kwargs / the no-archive startup guard, so a YAML-supplied
    # use_archive routes this run through the evolutionary orchestrator (and the
    # guard never trips). getattr keeps hand-built
    # wrapper/test namespaces byte-identical: a missing benchmark_config attr →
    # "none" → skip (same as --benchmark-config none). The provenance stash
    # (benchmark_config_resolved / benchmark_config_applied) is set on every path.
    cfg_choice = getattr(args, "benchmark_config", "none")
    cfg_path = _benchmark_config_path(cfg_choice)
    benchmark_config_resolved = "none"
    benchmark_config_applied: list[str] = []
    # Per-benchmark scope: the YAML's blocks key off args.benchmark, so a
    # custom --tasks run (benchmark None) keeps bare code defaults — same
    # provenance as --benchmark-config none. getattr: hand-built namespaces.
    if cfg_path is not None and getattr(args, "benchmark", None) is None:
        console.print(
            "  benchmark-config: skipped — per-benchmark scope; custom --tasks "
            "runs keep bare code defaults (pass flags explicitly to opt in)."
        )
        cfg_path = None
    if cfg_path is not None:
        # F032: a missing config file is a startup-validation failure for BOTH
        # 'auto' (the YAML ships as package data, so a miss implies a broken
        # install) and an explicit literal path (an operator typo). Refuse
        # rather than silently fork onto bare code defaults.
        if not cfg_path.exists():
            console.print(
                f"[red]Error: benchmark-config file not found: {cfg_path}. "
                f"Fix the path (or reinstall for 'auto'), or pass "
                f"--benchmark-config none for bare code defaults.[/red]"
            )
            return 1
        loaded = None
        parse_failed = False
        try:
            loaded = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001 - malformed YAML must not crash startup
            console.print(
                f"[yellow]benchmark-config: failed to parse {cfg_path}: {e}; "
                f"skipping.[/yellow]",
                soft_wrap=True,  # one-line diagnostic — don't width-wrap (CI-width brittleness)
            )
            parse_failed = True
        if parse_failed:
            benchmark_config_resolved = f"parse-error:{cfg_path}"
        else:
            # None ⇒ empty/comments-only document ⇒ {} (found, zero overrides
            # — records the real path + prints 'applied 0 overrides'). A
            # non-mapping document (list / scalar) is a parse-class failure:
            # warn + skip, never an AttributeError crash.
            loaded = {} if loaded is None else loaded
            if not isinstance(loaded, dict):
                console.print(
                    f"[yellow]benchmark-config: {cfg_path} did not parse to a "
                    f"mapping (got {type(loaded).__name__}); skipping.[/yellow]",
                    soft_wrap=True,  # one-line diagnostic — don't width-wrap (CI-width brittleness)
                )
                benchmark_config_resolved = f"parse-error:{cfg_path}"
            else:
                parser = build_parser()
                user_set = _user_set_dests(parser, sys.argv[1:])
                _skipped: list[str] = []
                # parser= arms per-key validation: a wrong-typed YAML value
                # raises ValueError (never warn-and-skip, which would silently
                # run bare defaults) — surfaced here as an F032 startup
                # refusal (return 1). Non-settable/unknown keys and malformed
                # blocks are warn+skipped into _skipped instead.
                try:
                    benchmark_config_applied = apply_benchmark_config(
                        args, user_set, loaded, warn=_skipped.append, parser=parser
                    )
                except ValueError as e:
                    console.print(f"[red]Error: {e} (in {cfg_path}). Fix the "
                                  f"value, or pass --benchmark-config none for "
                                  f"bare code defaults.[/red]")
                    return 1
                benchmark_config_resolved = str(cfg_path)
                console.print(
                    f"  benchmark-config: applied {len(benchmark_config_applied)} "
                    f"overrides for {getattr(args, 'benchmark', None) or '(none)'}: "
                    f"{benchmark_config_applied}"
                )
                if _skipped:
                    console.print(
                        f"  [yellow]benchmark-config: skipped unknown/malformed "
                        f"entries: {sorted(_skipped)}[/yellow]"
                    )
    # Provenance stash — read by build_base_run_config (additive config.json keys).
    args.benchmark_config_resolved = benchmark_config_resolved
    args.benchmark_config_applied = benchmark_config_applied

    # --- Setup LLM (needed before benchmark loading for llm() injection) ---
    backend, deployment_or_model = _resolve_backend_and_model(args.model)
    daily_budget_usd, cost_ledger_dir = _resolve_budget(args, backend)

    # The legacy linear MetaRecurseOrchestrator has been retired; the
    # evolutionary orchestrator (--use-archive) is the only run path. The
    # bundled benchmark config (applied earlier) enables it by default, so a
    # normal ``--benchmark X`` run is unaffected; only a deliberate opt-out
    # (``--benchmark-config none`` or a bare ``--tasks`` run) without
    # ``--use-archive`` reaches here. Refuse BEFORE the LLM client / adapter are
    # built so the operator gets this clean message rather than a downstream
    # credentials/adapter error. This one guard subsumes the former --resume /
    # --base-solver / evolutionary-flag no-archive refusals (all downstream,
    # unreachable once it fires).
    if not getattr(args, "use_archive", False):
        console.print(
            "[red]Error: --use-archive is required (the linear orchestrator has "
            "been retired). The bundled benchmark config enables it by default; "
            "pass --use-archive explicitly if you also pass --benchmark-config "
            "none or run with bare --tasks.[/red]"
        )
        return 1

    # Backend-request budget -> env, read by meta_n.rpbe.accounting at the
    # send site. Set BEFORE the client is built so the selection-request
    # guard is armed from the very first call.
    if getattr(args, "max_backend_requests", None) is not None:
        os.environ["META_N_MAX_BACKEND_REQUESTS"] = str(args.max_backend_requests)

    llm_config = LLMConfig(
        base_url=args.base_url,
        api_key=args.api_key,
        model=deployment_or_model,
        max_tokens=args.max_tokens,
        request_timeout=args.request_timeout,
        empty_content_retry_max_tokens=args.empty_retry_max_tokens,
        # TRANSPORT retries (default 3 = the dataclass default this replaces).
        # getattr: hand-built arg namespaces predate the flag.
        max_retries=getattr(args, "llm_max_retries", 3),
        exclude_providers=args.exclude_providers,
        # getattr keeps hand-built arg namespaces valid (test helpers + the
        # gpt52 driver scripts construct a Namespace directly).
        reasoning_effort=getattr(args, "reasoning_effort", None),
        backend=backend,
        azure_endpoint=args.azure_endpoint,
        azure_api_version=args.azure_api_version,
        daily_budget_usd=daily_budget_usd,
        cost_ledger_dir=cost_ledger_dir,
    )
    llm_client = LLMClient(llm_config)
    if llm_client.cost_tracker is not None:
        try:
            today = llm_client.cost_tracker.today_total_usd()
            console.print(
                f"  Cost tracker: backend={backend}, today=${today:.4f} / "
                f"${daily_budget_usd:.2f} cap, ledger={cost_ledger_dir}"
            )
        except Exception as e:  # noqa: BLE001
            console.print(f"  [yellow]Cost tracker startup read failed: {e}[/yellow]")

    # --- Load tasks ---
    benchmark_name = None
    # Pre-initialized so the external-spine capability probe below is
    # NameError-safe on the ``--tasks`` (no-adapter) path: getattr(None, ...)
    # resolves to the default lambda → False (legacy native routing).
    adapter = None

    if args.benchmark == "co_bench":
        from meta_n.integrations.co_bench import COBenchAdapter, COBenchExecutor

        data_dir = args.bench_data_dir or "./data/co_bench"
        adapter = COBenchAdapter(
            data_dir=data_dir,
            task_names=args.bench_tasks,
            instance_workers=args.instance_workers,
            llm_config_dict=asdict(llm_config),
        )
        tasks = adapter.load_tasks(limit=args.bench_limit)
        executor = COBenchExecutor(adapter)
        benchmark_name = "co_bench"
        solver_language = "python"
    elif args.benchmark in ("symptom2disease", "lawbench_charge"):
        from meta_n.integrations.text_classification import (
            TextClassificationAdapter,
            TextClassificationExecutor,
        )

        data_dir = args.bench_data_dir or "./data/text_classification"
        eval_workers = args.instance_workers if args.instance_workers > 0 else 4
        adapter = TextClassificationAdapter(
            dataset_name=args.benchmark,
            data_dir=data_dir,
            n_few_shot=args.n_few_shot,
            max_val=args.max_val,
            max_test=getattr(args, "max_test", None),
            llm_client=llm_client,
            eval_workers=eval_workers,
            # F075 (§6b): the LIVE parse site — a solve() that returns a
            # string gets one recovery pass through the classify extraction
            # chain (see TextClassificationAdapter._coerce_predictions).
            balanced_json_fallback=args.classify_balanced_json_fallback,
        )
        tasks = adapter.load_tasks(limit=args.bench_limit)
        executor = TextClassificationExecutor(adapter)
        benchmark_name = args.benchmark
        solver_language = "python"
    elif args.benchmark == "terminal_bench":
        from meta_n.integrations.terminal_bench import (
            TerminalBenchAdapter,
            TerminalBenchExecutor,
        )

        data_dir = args.bench_data_dir
        adapter = TerminalBenchAdapter(
            task_cache_dir=data_dir,
            task_names=args.bench_tasks,
            # R5: declared route lets load_tasks refuse a native run on the
            # legacy original-tasks layout BEFORE any solver spend.
            base_solver=args.base_solver,
        )
        await adapter.download()
        # Seed-stable shuffle of the task list so meta-n and DGM evaluate
        # on the SAME subset for the same ``--seed`` (DGM also shuffles
        # via random.Random(seed) — see baselines/dgm/src/task_terminal_bench.py).
        # Apples-to-apples per-task comparison requires identical subsets.
        tasks = adapter.load_tasks(
            limit=args.bench_limit, seed_shuffle=args.seed,
        )
        executor = TerminalBenchExecutor(adapter)
        benchmark_name = "terminal_bench"
        solver_language = "bash"
    elif args.benchmark == "swe_bench_verified":
        from meta_n.integrations.swe_bench import SWEBenchVerifiedAdapter
        from meta_n.integrations.terminal_bench import TerminalBenchExecutor

        data_dir = args.bench_data_dir or "./data/swe_bench_verified"
        adapter = SWEBenchVerifiedAdapter(
            task_cache_dir=data_dir,
            task_names=args.bench_tasks,
            # R5: same pre-spend legacy-layout refusal as terminal_bench.
            base_solver=args.base_solver,
        )
        await adapter.download()
        tasks = adapter.load_tasks(
            limit=args.bench_limit, seed_shuffle=args.seed,
        )
        executor = TerminalBenchExecutor(adapter)
        benchmark_name = "swe_bench_verified"
        solver_language = "bash"
    elif args.benchmark in _OPENEVOLVE_FAMILY:
        # F197: one uniform branch for the OpenEvolve family (adapter class,
        # default data_dir, filter kwarg vary; everything else is identical).
        import importlib

        from meta_n.integrations.openevolve import OpenEvolveExecutor

        mod, cls, default_dir, filter_kw = _OPENEVOLVE_FAMILY[args.benchmark]
        adapter_cls = getattr(importlib.import_module(mod), cls)
        data_dir = args.bench_data_dir or default_dir
        adapter = adapter_cls(data_dir=data_dir, **{filter_kw: args.bench_tasks})
        tasks = adapter.load_tasks(limit=args.bench_limit)
        executor = OpenEvolveExecutor(adapter)
        benchmark_name = args.benchmark
        solver_language = "openevolve"
    elif args.tasks:
        tasks = load_tasks(args.tasks)
        executor = LocalExecutor()
        solver_language = "bash"
    else:
        console.print("[red]Error: provide --tasks or --benchmark[/red]")
        return 1

    # Whether THIS run drives the external-agent spine. F042: main.py and the
    # orchestrator's ``_uses_external_spine()`` now share ONE predicate
    # (core/spine_routing.uses_external_spine) — engaged for a genuine external
    # kind (openhands / terminus2) OR for ``builtin`` on an adapter advertising
    # spine-builtin (only terminal_bench does). The three external-only startup
    # guards below gate on THIS, not a hardcoded base-solver pair, so a
    # spine-routed ``--base-solver builtin --benchmark terminal_bench`` is not
    # waved past them; non-advertising adapters (CO-Bench, classify) and the
    # ``--tasks`` (adapter=None) path stay legacy.
    external_spine = uses_external_spine(args.base_solver, adapter)

    if args.resume and not args.exp_name:
        console.print("[red]Error: --resume requires --exp-name to identify the run directory[/red]")
        return 1

    # External-agent base solvers spend real money inside Docker / remote
    # agents that land in the daily ledger. A positive daily cap is a hard
    # prerequisite (§4.7 / §8.1): refuse to start without it. Gated on
    # ``external_spine`` so a spine-routed builtin is covered too.
    if external_spine and not (daily_budget_usd and daily_budget_usd > 0):
        console.print(
            f"[red]Error: --base-solver {args.base_solver} requires "
            f"--daily-budget-usd > 0 (got {daily_budget_usd}). "
            f"Set --daily-budget-usd or $META_N_DAILY_BUDGET_USD.[/red]"
        )
        return 1

    # BUDGET-PRECHECK clarity (footgun fix, fix (c)): when the per-run agent
    # budget is larger than the day's headroom, the CostGuard ADMITS the run
    # rather than silently denying every one. Surface the arithmetic up front so
    # the operator understands the per-run budget is NOT clamped (no backend
    # enforces a per-run USD stop) and the daily cap is enforced at the
    # next-run admission / generation boundary, not mid-run. Gated on
    # ``external_spine`` so a spine-routed builtin is covered too.
    if external_spine and llm_client.cost_tracker is not None:
        try:
            tracker = llm_client.cost_tracker
            threshold = tracker.daily_cap_usd - tracker.reservation_usd
            headroom = threshold - tracker.today_total_usd()
            per_run = float(args.agent_max_budget)
            if per_run > headroom:
                console.print(
                    f"[yellow]Note: --agent-max-budget ${per_run:.2f} exceeds the "
                    f"day's remaining headroom ${headroom:.2f} (cap "
                    f"${tracker.daily_cap_usd:.2f} - reservation "
                    f"${tracker.reservation_usd:.2f} - today "
                    f"${tracker.today_total_usd():.2f}). Runs are still ADMITTED; "
                    f"the per-run budget is NOT clamped and no backend enforces a "
                    f"per-run USD stop, so an admitted run may overshoot. The "
                    f"daily cap (${threshold:.2f}) is enforced by denying further "
                    f"runs once headroom hits $0 (overshoot up to --parallel "
                    f"in-flight runs).[/yellow]"
                )
        except Exception as e:  # noqa: BLE001 - advisory only, never blocks startup
            console.print(f"[yellow]Budget headroom note skipped: {e}[/yellow]")

    # Clamp the inner Docker cap to the outer task parallelism. Running more
    # containers than concurrent tasks cannot help and only inflates resource
    # pressure, so warn and clamp down rather than silently honoring it.
    if args.max_docker is not None and args.max_docker > args.parallel:
        console.print(
            f"[yellow]Warning: --max-docker {args.max_docker} > --parallel "
            f"{args.parallel}; clamping --max-docker to {args.parallel}.[/yellow]"
        )
        args.max_docker = args.parallel

    # F5: --foster-adoption is consumed ONLY on the native Layer1Solver /
    # MetaLayer path. Warn (do NOT hard-reject, to avoid blocking any
    # currently-valid invocation) when it is combined with a path that
    # silently ignores it — the agentic solver, an external base-solver, or a
    # demoted-helper family without --force-code-library-live — so the operator
    # is not misled into believing adoption was enforced. Wholly gated on the
    # flag being set, so the default-OFF path emits nothing (byte-identical).
    if args.foster_adoption:
        _foster_noop = foster_adoption_noop_reasons(args, adapter, external_spine)
        if _foster_noop:
            console.print(
                "[yellow]Warning: --foster-adoption is a silent no-op with: "
                + "; ".join(_foster_noop)
                + ".[/yellow]"
            )

    # F229: warn-only surfacing of evolutionary flags that are silent no-ops on
    # THIS run's solver path (foster_adoption precedent). Wholly gated on
    # non-default values, so the default path prints nothing (byte-identical).
    # annotate_config_sourced: YAML-sourced flags carry their source file.
    _noop_notes = evolutionary_noop_flag_notes(args, external_spine)
    _deploy_note = deploy_verified_code_noop_note(args, solver_language)
    if _deploy_note:
        _noop_notes.append(_deploy_note)
    if _noop_notes:
        console.print(
            "[yellow]Warning: silent no-op flags for this run: "
            + "; ".join(annotate_config_sourced(_noop_notes, args))
            + ".[/yellow]"
        )

    console.print(f"[bold]Meta^n[/bold] — Loaded {len(tasks)} tasks")
    if backend == "azure":
        console.print(f"  Model: {args.model} (azure deployment={deployment_or_model})")
    else:
        console.print(f"  Model: {args.model}")
    console.print(f"  Epsilon: {args.epsilon}")
    console.print(f"  Max depth: {args.max_depth}")
    if benchmark_name:
        console.print(f"  Benchmark: {benchmark_name}")
    if args.use_agentic:
        console.print(f"  Solver: agentic {solver_language} (max_turns={args.agentic_max_turns}, budget={args.agentic_token_budget})")
    else:
        console.print(f"  Solver: {solver_language}")
    console.print(f"  Parallel: {args.parallel}")
    # F6: echo the bench task selection so a run's provenance is visible in the
    # console log. Bench-task filtering is echoed unconditionally; the ablation
    # echo is emitted inside the archive block below (H8) alongside the flags it
    # reports.
    console.print(
        f"  Bench tasks: {args.bench_tasks if args.bench_tasks else '(all)'}"
    )

    # --- Experiment output path ---
    if args.exp_name:
        exp_name = args.exp_name
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_short = args.model.split("/")[-1].replace(":", "_")
        prefix = benchmark_name or "custom"
        exp_name = f"{ts}_{prefix}_{model_short}"
    output_dir = f"{args.output_dir}/{exp_name}"
    console.print(f"  Output: {output_dir}")

    # --- Setup components ---
    # C2.3: thread a model-aware Omega INPUT-prompt budget. Default None keeps
    # the built-in 100k ContextBudget so existing runs stay byte-identical; only
    # when --omega-context-budget is supplied do we override max_tokens (e.g. to
    # a small local backbone's real window so truncation fires before overflow).
    omega_context_budget = None
    if getattr(args, "omega_context_budget", None):
        from meta_n.utils.context_manager import ContextBudget
        omega_context_budget = ContextBudget(max_tokens=args.omega_context_budget)
    # F156 (§6b): Default OFF = byte-identical historical trace sampling.
    omega = OmegaEngine(
        llm_client, context_budget=omega_context_budget,
        symmetric_trace_sampling=getattr(args, "symmetric_trace_sampling", False),
    )

    # --- RPBE context-reduction mode (opt-in; default OFF) ----------------
    # `official` is the historical path, byte-identical. The other modes swap
    # the engine's context reduction for the RPBE one, covering BOTH call
    # sites (generate and refine) because both go through the same two
    # methods. This touches only the entry point -- OmegaEngine's own body,
    # `_build_prompt`, the archive and the evaluator are untouched.
    _reduction_mode = getattr(args, "reduction_mode", "official") or "official"
    if _reduction_mode != "official":
        from meta_n.rpbe.context_reduction import ContextReducer, install
        _rpbe_enc = None
        _rpbe_gamma = None
        if _reduction_mode == "predictive":
            from meta_n.rpbe.encoder import FrozenItemEncoder
            from meta_n.rpbe.fusion import SlottedFusion
            _cb = os.environ.get("CODEBERT_PATH", "").strip()
            if not _cb or not os.path.isdir(_cb):
                raise SystemExit(
                    "reduction-mode=predictive needs the frozen encoder: set "
                    "CODEBERT_PATH to the codebert-base snapshot directory")
            _rpbe_enc = FrozenItemEncoder(_cb)
            # Phase C must evaluate a TRAINED, FROZEN Gamma -- so loading is
            # MANDATORY. A bare `SlottedFusion()` here would silently score the
            # entire run with a randomly initialised model and the reported
            # "Ours" number would be meaningless. A missing or unreadable
            # checkpoint aborts instead (GammaCheckpointError -> SystemExit).
            _gpath = (getattr(args, "gamma_checkpoint", None)
                      or os.environ.get("META_N_GAMMA_CHECKPOINT", "") or "").strip()
            if not _gpath:
                raise SystemExit(
                    "reduction-mode=predictive requires the trained Gamma: pass "
                    "--gamma-checkpoint <path> (or set META_N_GAMMA_CHECKPOINT). "
                    "Refusing to evaluate with a randomly initialised Gamma.")
            from meta_n.rpbe.checkpoint import (GammaCheckpointError,
                                                load_gamma_checkpoint)
            try:
                _rpbe_gamma, _gmeta = load_gamma_checkpoint(_gpath)
            except GammaCheckpointError as e:
                raise SystemExit("Gamma checkpoint rejected: {}".format(e))
            console.print(
                f"  Gamma checkpoint: {_gpath}")
            console.print(
                f"    checksum={_gmeta.checksum}  params={_gmeta.param_count}  "
                f"steps={_gmeta.steps}  status={_gmeta.status}  "
                f"fingerprint={_gmeta.fingerprint()}")
        # The reducer MUST share the engine's own ContextManager. Constructing
        # it without one gives the reducer a FRESH ContextManager(), whose
        # defaults silently replace the engine's: `symmetric_sampling` reverts
        # to False (the profile pins true), the seeded `rng` the orchestrator
        # writes, the token estimators and the stack-truncation rule are all
        # bypassed -- and the formal runner's config.json parity check cannot
        # see any of it, because it compares recorded keys, not live objects.
        #
        # Two sides have to line up, because `install` puts the ADAPTER in the
        # engine slot while the reducer keeps sampling from the base:
        #   * the reducer is built with `omega.context_manager` (the base), and
        #   * the adapter forwards `rng` / `symmetric_sampling` / `budget` to
        #     that same base -- the orchestrator writes `omega.context_manager
        #     .rng = self.rng` AFTER install, i.e. through the adapter
        #     (evolutionary_orchestrator.py:552, run_persistence.py:102/123).
        # Get either side wrong and the recorded search_seed is not the seed
        # the sampler used. self_test() asserts both, end to end.
        install(omega, ContextReducer(_reduction_mode, encoder=_rpbe_enc,
                                      fusion=_rpbe_gamma,
                                      context_manager=omega.context_manager))
        console.print(
            f"  Reduction mode: {_reduction_mode} (RPBE adapter installed; "
            f"covers generate AND refine)")

    # H8: the COMMON run_config omits the evolutionary-only ablation provenance
    # keys — they are re-added inside the use_archive block below so a linear
    # run's config.json does not falsely record ablations it cannot apply.
    run_config = build_base_run_config(
        args,
        benchmark_name=benchmark_name,
        solver_language=solver_language,
        executor_name=type(executor).__name__,
    )

    # F6: echo the ablation flags (evolutionary-only) now that we know this
    # run drives the archive orchestrator that actually applies them.
    console.print(
        f"  Ablations: no_code_library={args.no_code_library}, "
        f"no_outer_context={args.no_outer_context}, "
        f"foster_adoption={args.foster_adoption}, "
        f"force_code_library_live={args.force_code_library_live}, "
        f"verified_code={args.verified_code}, "
        f"regression_guard={args.regression_guard}"
    )
    from meta_n.core.evolutionary_orchestrator import (
        EvolutionaryConfig,
        EvolutionaryOrchestrator,
    )

    # F048: the evolutionary-only kwargs are built ONCE and shared by the
    # config constructor and the config.json provenance block (key order
    # frozen inside evolutionary_run_kwargs). seed_code_library is the one
    # deliberate override: PATH in provenance, parsed mapping in config.
    evo_kwargs = evolutionary_run_kwargs(args)
    evo_config = EvolutionaryConfig(
        epsilon=args.epsilon,
        max_depth=args.max_depth,
        output_dir=output_dir,
        parallel=args.parallel,
        max_retries=args.max_retries,
        retry_threshold=args.retry_threshold,
        **{
            **evo_kwargs,
            "seed_code_library": load_seed_code_library(args.seed_code_library),
        },
    )
    run_config.update({"orchestrator": "evolutionary", **evo_kwargs})
    console.print(f"  Orchestrator: evolutionary (B={args.beam_width}, K={args.beam_candidates}, max_iter={args.max_iterations}, patience={args.patience}, alpha={args.novelty_alpha})")

    orchestrator = EvolutionaryOrchestrator(
        llm_client, executor, omega, evo_config,
        solver_language=solver_language,
        # F075 (§6b): classify-path extraction knob (both orchestrator
        # paths consume it; provenance via build_base_run_config).
        balanced_json_fallback=args.classify_balanced_json_fallback,
    )
    # --- SRI formal instrumentation (opt-in; default OFF) ------------------
    # Constructing the ledger HERE is what actually enables it. Without this the
    # orchestrator keeps SRIHooks.disabled() no matter what the runner believes,
    # so a "formal" run produces no proposal_slots.jsonl at all while the runner
    # records a path to a file that does not exist.
    #
    # The no-test-eval switch is set FIRST and independently of the context: the
    # shared-root stage runs without --sri-context, so relying on the context to
    # carry it left the root evaluating the held-out test split twice.
    if getattr(args, "no_test_eval", False):
        orchestrator.sri_no_test_eval = True
    if getattr(args, "sri_context", None):
        from meta_n.sri.hooks import SRIHooks
        from meta_n.sri.ledger import SlotLedger
        _ctx = json.loads(Path(args.sri_context).read_text(encoding="utf-8"))
        _need = ("run_id", "arm", "backbone", "cohort_id", "search_seed",
                 "root_bundle_sha256", "ledger_path")
        # `search_seed` may legitimately be 0, so test for ABSENCE, not falsiness.
        _missing = [k for k in _need
                    if k not in _ctx or _ctx[k] in (None, "")]
        if _missing:
            raise SystemExit(
                "--sri-context is missing required key(s): {}".format(_missing))
        if _ctx["arm"] == "predictive" and not _ctx.get(
                "gamma_checkpoint_sha256"):
            raise SystemExit(
                "--sri-context arm=predictive requires gamma_checkpoint_sha256: "
                "the arm must record its Gamma identity (§4)")
        # The context names the TREATMENT; the process runs `_reduction_mode`.
        # A disagreement would log one arm's ledger while running the other's
        # mode -- caught here, before any paid call, rather than after.
        if _ctx.get("reduction_mode") not in (None, _reduction_mode):
            raise SystemExit(
                "--sri-context says reduction_mode={!r} but this process is "
                "running {!r}; the treatment would not be applied".format(
                    _ctx.get("reduction_mode"), _reduction_mode))
        _led = SlotLedger(
            _ctx["ledger_path"], run_id=_ctx["run_id"], arm=_ctx["arm"],
            backbone=_ctx["backbone"], cohort_id=_ctx["cohort_id"],
            search_seed=_ctx["search_seed"],
            root_bundle_sha256=_ctx["root_bundle_sha256"],
            gamma_checkpoint_sha256=_ctx.get("gamma_checkpoint_sha256"))
        orchestrator.sri = SRIHooks(
            ledger=_led, run_dir=Path(output_dir),
            gate_configured=orchestrator.config.gate_tasks,
            gate_reason=_ctx.get("gate_reason", ""),
            reduction_mode=_reduction_mode, cohort=_ctx.get("cohort") or [])
        orchestrator.sri_no_test_eval = bool(
            getattr(orchestrator, "sri_no_test_eval", False)
            or _ctx.get("no_test_eval", True))
        # §4/§10: the TREATMENT itself must be recorded in the run's own
        # artifacts. `reduction_mode` and the Gamma identity never reached
        # config.json, so a reader could not tell the two arms apart from the
        # artifact set -- and one arm's silently wrong mode was undetectable.
        # Additive and gated behind --sri-context, so a non-formal config.json
        # stays byte-identical.
        run_config["sri"] = {
            "run_id": _ctx["run_id"], "arm": _ctx["arm"],
            "backbone": _ctx["backbone"], "cohort_id": _ctx["cohort_id"],
            "search_seed": _ctx["search_seed"],
            "root_bundle_sha256": _ctx["root_bundle_sha256"],
            "reduction_mode": _reduction_mode,
            "gamma_checkpoint_sha256": _ctx.get("gamma_checkpoint_sha256"),
            "ledger_path": _ctx["ledger_path"],
            "no_test_eval": orchestrator.sri_no_test_eval,
            "cohort": list(_ctx.get("cohort") or []),
            "profile_sha256": _ctx.get("profile_sha256"),
            "sri_context_sha256": _ctx.get("sri_context_sha256"),
            "gate_reason": _ctx.get("gate_reason", ""),
        }
        console.print(
            "  SRI ledger: {}  arm={}  seed={}  no_test_eval={}".format(
                _ctx["ledger_path"], _ctx["arm"], _ctx["search_seed"],
                orchestrator.sri_no_test_eval))

    result = await orchestrator.run(tasks, resume=args.resume, run_config=run_config)

    # Summary — oracle mean comes from run(), which averages over ALL
    # tasks (missing-as-0.0), not the attempted-only subset denominator.
    console.print(f"\n[bold green]═══ Final Summary ═══[/bold green]")
    console.print(f"  Iterations: {result.total_iterations}")
    console.print(f"  Archive size: {result.archive_size}")
    console.print(f"  Best single-chain mean_score: {result.best_mean_score:.3f}")
    console.print(f"  Per-task best mean (oracle):  {result.oracle_mean_score:.3f}")
    if result.test_mean_score > 0:
        console.print(f"  Test mean score (oracle):     {result.test_mean_score:.3f}")
    console.print(f"  Best candidate: {result.best_candidate_id}")
    console.print(f"  Total tokens: {result.total_tokens:,}")
    console.print(f"  Per-task best scores:")
    for tid, score in sorted(result.per_task_best_scores.items()):
        console.print(f"    {tid}: {score:.3f}")

    orchestrator.save_results(result, run_config=run_config)
    _save_cost_summary(output_dir, llm_client)

    # Backend-request accounting -- the count the money fuse actually gates on.
    # Printed even when the request ledger is unset (then it reports zeros) so
    # the number is never a surprise after the fact.
    try:
        from meta_n.rpbe.accounting import format_accounting
        console.print()
        for _line in format_accounting().splitlines():
            console.print("  " + _line)
    except Exception as e:  # noqa: BLE001
        console.print(f"[yellow]accounting report failed: {e}[/yellow]")


def _save_cost_summary(output_dir: str, llm_client: LLMClient) -> None:
    """Write run-level cost_summary.json next to summary.json.

    Pulls from the LLMClient's in-process counters (outer-only — inner
    LLM calls live inside subprocess workers and are tracked in the
    daily ledger but not in the parent's cumulative_usage). The ledger
    file remains the cross-process source of truth for daily spend; this
    file just gives a per-run cost figure that's easy to scan.
    """
    cu = getattr(llm_client, "cumulative_usage", {}) or {}
    payload = {
        "outer_calls": int(cu.get("calls", 0) or 0),
        "outer_prompt_tokens": int(cu.get("prompt", 0) or 0),
        "outer_completion_tokens": int(cu.get("completion", 0) or 0),
        "outer_cached_tokens": int(cu.get("cached", 0) or 0),
        "outer_cost_usd": round(float(cu.get("cost_usd", 0.0) or 0.0), 6),
        "model": llm_client.config.model,
        "backend": llm_client.config.backend,
    }
    if llm_client.cost_tracker is not None:
        try:
            payload["daily_total_usd_at_save"] = round(
                llm_client.cost_tracker.today_total_usd(), 6
            )
            payload["daily_cap_usd"] = llm_client.cost_tracker.daily_cap_usd
            payload["ledger_path"] = str(llm_client.cost_tracker._today_path())
        except Exception:  # noqa: BLE001
            pass
    try:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "cost_summary.json"), "w") as f:
            json.dump(payload, f, indent=2)
    except OSError as e:
        console.print(f"[yellow]cost_summary write failed: {e}[/yellow]")
    console.print(
        f"  [bold]Outer LLM cost:[/bold] ${payload['outer_cost_usd']:.4f} "
        f"({payload['outer_calls']} calls, {payload['outer_prompt_tokens']:,}p / "
        f"{payload['outer_completion_tokens']:,}c)"
    )
    if "daily_total_usd_at_save" in payload:
        console.print(
            f"  [bold]Daily ledger:[/bold] ${payload['daily_total_usd_at_save']:.4f} / "
            f"${payload['daily_cap_usd']:.2f} cap (incl. inner LLM calls)"
        )


def main():
    args = parse_args()
    try:
        rc = asyncio.run(async_main(args))
    except BudgetExceededError as e:
        # Hard stop: today's spend cap reached. The orchestrator's
        # last checkpoint is already on disk (saved after each iteration),
        # so resume tomorrow with `--resume --exp-name <same>`.
        # DELIBERATE (F029/F038): save_results is NOT called on the hard-kill
        # path, so summary.json keeps the MID-RUN shape (no total_iterations /
        # run_status). Downstream completion checks are presence-based
        # (azure_full_*.sh key on 'total_iterations'); stamping a final summary
        # here would make a budget-killed run read as completed. If a stamped
        # abort summary is ever wanted, it is a persisted-schema decision that
        # must be audited against every presence-based consumer — see the
        # save_running_summary contract docstring (F038).
        console.print(f"\n[bold red]BUDGET CAP REACHED[/bold red] — {e}")
        console.print("[yellow]Resume with --resume --exp-name <same> after the daily reset.[/yellow]")
        sys.exit(2)
    # F032: startup-validation refusals return 1 (misconfiguration); success
    # paths return None and keep exiting 0.
    if rc:
        sys.exit(int(rc))


if __name__ == "__main__":
    main()
