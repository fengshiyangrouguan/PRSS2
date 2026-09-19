"""CO-Bench adapter — combinatorial optimization benchmark.

Self-contained evaluation using each task's config.py (load_data, eval_func,
norm_score, get_dev).  No external CO-Bench evaluation module required.

Data download:
    huggingface-cli download CO-Bench/CO-Bench --repo-type dataset --local-dir data

Usage:
    adapter = COBenchAdapter(data_dir="./data/co_bench")
    tasks   = adapter.load_tasks(limit=5)
"""

from __future__ import annotations

import ast as ast_mod
import asyncio
import contextlib
import functools
import importlib.util
import io
import logging
import math
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Any

from meta_n.core.external_agents.env import (
    AgentEnvProvider,
    EnvLease,
    Scorer,
    mirror_agent_tokens,
)
from meta_n.core.meta_layer import TaskDescription

# Compat re-exports: tests and baselines import/monkeypatch these via this
# module's namespace (e.g. ``co_bench._kill_process_tree``); keep them bound.
from meta_n.integrations._subprocess_utils import (  # noqa: F401
    _add_usage,
    _coerce_usage,
    _empty_usage,
    _kill_process_tree,
    _snapshot_log_offset,
    _tracker_usage,
    _usage_from_log_since,
    detach_process_group,
    run_process_with_timeout,
)
from meta_n.core.verified_code import SandboxedHeldoutVerifier, VerifyResult
from meta_n.integrations.benchmark import AdapterExecutor, BenchmarkAdapter, EvalResult

if TYPE_CHECKING:  # pragma: no cover - typing only; keeps the SDK-free import contract
    from meta_n.core.external_agents.backend import (
        AgentBackend,
        AgentRunResult,
    )

logger = logging.getLogger(__name__)

# All 36 CO-Bench task names (must match directory names in data/co_bench/)
CO_BENCH_TASKS = [
    "Aircraft landing",
    "Assignment problem",
    "Assortment problem",
    "Bin packing - one-dimensional",
    "Capacitated warehouse location",
    "Common due date scheduling",
    "Constrained guillotine cutting",
    "Constrained non-guillotine cutting",
    "Container loading",
    "Container loading with weight restrictions",
    "Corporate structuring",
    "Crew scheduling",
    "Equitable partitioning problem",
    "Euclidean Steiner problem",
    "Flow shop scheduling",
    "Generalised assignment problem",
    "Graph colouring",
    "Hybrid Reentrant Shop Scheduling",
    "Job shop scheduling",
    "Maximal independent set",
    "Multi-Demand Multidimensional Knapsack problem",
    "Multidimensional knapsack problem",
    "Open shop scheduling",
    "p-median - capacitated",
    "p-median - uncapacitated",
    "Packing unequal circles",
    "Packing unequal circles area",
    "Packing unequal rectangles and squares",
    "Packing unequal rectangles and squares area",
    "Resource constrained shortest path",
    "Set covering",
    "Set partitioning",
    "Travelling salesman problem",
    "Uncapacitated warehouse location",
    "Unconstrained guillotine cutting",
    "Vehicle routing: period routing",
]


def _task_id_from_name(name: str) -> str:
    """Convert task name to a safe task_id."""
    return name.lower().replace(" ", "_").replace("-", "_")


# ---------------------------------------------------------------------------
# Helpers for running solve+eval in a subprocess with timeout
# (usage-dict helpers + kill/timeout harness live in _subprocess_utils)
# ---------------------------------------------------------------------------

def _run_instance_in_process(
    config_path: str, instance: dict, solve_source: str, queue: mp.Queue,
    llm_config_dict: dict | None = None,
    inner_log_path: str | None = None,
):
    """Target for subprocess: compile solve, run it, evaluate, put result in queue.

    Queue protocol — 3-tuple ``(status, payload, usage_dict)``:
      status="ok"    → payload is the eval score
      status="error" → payload is a human-readable error string
      usage_dict     → {"total","prompt","completion","calls"} from the
                        per-instance LLMUsageTracker. Empty dict on a path
                        that never created a tracker (no llm_config_dict).

    ``inner_log_path``: when set, the inner llm() helper appends each call's
    messages + response to the JSONL at this path. flock makes concurrent
    instance subprocesses safe to share one file.
    """
    detach_process_group()
    tracker = None
    try:
        # Import eval_func from config
        spec = importlib.util.spec_from_file_location("_cfg", config_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        eval_func = getattr(mod, "eval_func")

        # Compile the submitted solve() with optional llm() helper. We
        # always allocate the tracker so usage_dict round-trips even when
        # solve() never calls llm().
        ns: dict[str, Any] = {}
        if llm_config_dict is not None:
            from meta_n.core.llm_helpers import (
                LLMUsageTracker,
                make_llm_func_from_config,
            )
            tracker = LLMUsageTracker()
            ns["llm"], ns["llm_batch"] = make_llm_func_from_config(
                llm_config_dict, tracker, log_path=inner_log_path,
            )
        exec(solve_source, ns)  # noqa: S102
        solve_fn = ns["solve"]

        # Suppress solver stdout/stderr (PuLP/CBC are verbose)
        with _capture_output():
            solution = solve_fn(**instance)
            # Explicit type check (mirrors the text_classification sibling):
            # a list/None return must report a readable message, not the
            # cryptic AttributeError from .items() below.
            if not isinstance(solution, dict):
                queue.put((
                    "error",
                    f"solve() returned {type(solution).__name__}, expected dict",
                    _tracker_usage(tracker) if tracker else _empty_usage(),
                ))
                return
            solution = {str(k): v for k, v in solution.items()}
            score = eval_func(**instance, **solution)

        queue.put(("ok", score, _tracker_usage(tracker) if tracker else _empty_usage()))
    except Exception as e:
        queue.put(("error", str(e), _tracker_usage(tracker) if tracker else _empty_usage()))


@contextlib.contextmanager
def _capture_output():
    """Suppress stdout/stderr (e.g. PuLP/CBC solver output)."""
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    try:
        yield
    finally:
        sys.stdout, sys.stderr = old_out, old_err


def _execution_contract(timeout_s: int) -> str:
    """The benchmark's hard runtime contracts, stated to the model.

    Meta^n previously left TWO things invisible that decide whether generated
    code can run at all. Both were measured as total failures (score 0.000,
    gate FAILED, no lineage) on CO-Bench / deepseek-flash and kimi-k2.7-code:

      1. RUNTIME LIMIT. `_run_with_timeout` executes each instance in its own
         process with a ``timeout`` second wall-clock limit and scores a
         timeout as **0.0**. Omega wrote a helper that returned the OPTIMAL bin
         count with a self-imposed 15 s budget -> 40/40 instances SIGKILLed.
         Changing ONLY that budget to 5 s gave 0/40 timeouts, score 0.4946.

      2. HELPER SANDBOX. Every ``code_library`` helper is screened by
         ``validate_library_function`` before it is ever prepended, and
         ``BLOCKED_IMPORTS`` (``meta_n.utils.safety``) rejects a helper that
         imports e.g. ``sys``, ``signal``, ``threading`` or ``io``. A helper
         rejected there is silently absent -- yet the layer's own
         ``pre_process`` may still tell the solver to call it -> NameError on
         every instance (160/160 here).

    Note the two interact: the runtime advice below says "bound your search
    with a deadline", and the obvious tools for that (``signal``, ``threading``)
    are themselves blocked. Both facts therefore have to be stated together.

    This text goes into the TASK DESCRIPTION, not into any prompt-injection
    path: seed generation, Omega generate/refine, and the Full / Official /
    Predictive reductions all present the same task text, so every mode sees
    exactly the same contract and the comparison stays fair. The evaluator and
    the validator themselves are UNCHANGED -- this only stops hiding rules the
    generated code already had to satisfy.
    """
    from meta_n.utils.safety import BLOCKED_IMPORTS
    blocked = ", ".join(sorted(BLOCKED_IMPORTS))
    return (
        "\n\n## Execution limits (HARD)\n"
        "- Each instance is evaluated in a separate process with a strict "
        "**{t}-second wall-clock limit**.\n"
        "- Exceeding it kills the process and that instance scores **0.0**.\n"
        "- The limit covers the ENTIRE runtime of one `solve(**instance_kwargs)` "
        "call: your parsing, every heuristic, every local-search loop, and every "
        "helper you call.\n"
        "- Plan for a safe margin: a timeout scores 0.0, which is strictly worse "
        "than simply returning an un-improved but valid answer. Bound any search "
        "with an explicit deadline well below {t} seconds and return the best "
        "solution found so far when it expires.\n"
        "- Do NOT reach for `signal`, `threading` or `sys` to implement that "
        "deadline -- they are unavailable (see below). Compare elapsed time "
        "with the `time` module instead.\n"
        "\n## Helper code restrictions (HARD)\n"
        "Any function you place in `code_library` is validated BEFORE it is "
        "made available, and a helper that fails validation is dropped "
        "silently -- callers then fail with `NameError`.\n"
        "- These modules may NOT be imported, at any level:\n"
        "  `{blocked}`\n"
        "- Plain arithmetic, `math`, `random`, `time`, `collections`, `heapq`, "
        "`itertools`, `functools`, `bisect` and the builtins are all fine.\n"
        "- Everything the helper needs must live inside the helper itself; do "
        "not rely on imports from the solver script or the surrounding process.\n"
        "- If you cannot implement a helper without a blocked module, write the "
        "logic directly in `solve()` instead of advertising a helper the solver "
        "would then fail to call.\n"
    ).format(t=int(timeout_s), blocked=blocked)


def _run_with_timeout(
    config_path: str, instance: dict, solve_source: str, timeout: int,
    llm_config_dict: dict | None = None,
    inner_log_path: str | None = None,
) -> tuple[str, Any, dict[str, int]]:
    """Run solve+eval in a subprocess with timeout.

    Returns 3-tuple: ``(status, payload, usage_dict)``.
      status="ok"    → payload is the eval score
      status="error" → payload is a human-readable error string
      usage_dict     → {"total","prompt","completion","calls"} aggregating
                        the per-instance inner-LLM tracker. On timeout/crash
                        (subprocess killed before reporting) partial usage is
                        reconstructed from the inner-log records THIS child
                        appended (pid-attributed — concurrent instance
                        subprocesses share one log file); zeros when no
                        inner-log is configured. Always a 4-key dict so
                        callers can ``_add_usage`` unconditionally.
    """
    # Snapshot the inner-log size before the child starts so a timeout/kill
    # can reconstruct just this subprocess's partial inner-LLM usage from the
    # records it appends (see _usage_from_log_since).
    log_start_offset = _snapshot_log_offset(inner_log_path)
    queue: mp.Queue = mp.Queue()
    return run_process_with_timeout(
        _run_instance_in_process,
        (config_path, instance, solve_source, queue, llm_config_dict,
         inner_log_path),
        timeout,
        queue=queue,
        on_timeout=lambda _elapsed, pid: (
            "error",
            f"Timeout ({timeout}s)",
            _usage_from_log_since(inner_log_path, log_start_offset, pid=pid),
        ),
        on_no_result=lambda _exc, pid: (
            "error",
            "No result from subprocess",
            _usage_from_log_since(inner_log_path, log_start_offset, pid=pid),
        ),
    )


# ---------------------------------------------------------------------------
# Self-contained evaluator
# ---------------------------------------------------------------------------

class _TaskEvaluator:
    """Evaluates a solve() function against a single CO-Bench task."""

    def __init__(
        self, task_name: str, data_dir: Path, timeout: int = 10,
        instance_workers: int = 2, llm_config_dict: dict | None = None,
        inner_log_path: str | None = None,
    ):
        self.task_name = task_name
        self.data_dir = data_dir
        self.timeout = timeout
        self._instance_workers = instance_workers
        self._llm_config_dict = llm_config_dict
        self._inner_log_path = inner_log_path
        self.task_dir = data_dir / task_name
        self.config_path = str(self.task_dir / "config.py")

        # Import config module
        spec = importlib.util.spec_from_file_location(
            f"cobench_{_task_id_from_name(task_name)}", self.config_path
        )
        self._module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self._module)

        self.load_data = getattr(self._module, "load_data")
        self.eval_func = getattr(self._module, "eval_func")
        self.norm_score = getattr(self._module, "norm_score", None)
        self.get_dev = getattr(self._module, "get_dev", None)

        # List all test case files (non-.py, non-__pycache__)
        self.all_cases = sorted(
            f for f in os.listdir(self.task_dir)
            if not (f.endswith(".py") or f == "__pycache__" or f == ".DS_Store")
        )

        # Dev/test split: dev_map maps file -> list of dev instance indices.
        # Dev eval: only files in dev_map, only dev indices within each file.
        # Test eval: all files, but skip dev indices (non-dev instances).
        self._dev_map = self.get_dev() if self.get_dev else None

    def evaluate(self, solve_source: str) -> dict:
        """
        Evaluate solve code against dev instances only (for search feedback).

        Returns dict with keys: dev_score, dev_feedback, raw_results
        """
        return self._evaluate_on_split(solve_source, split="dev")

    def evaluate_test(self, solve_source: str) -> dict:
        """
        Evaluate solve code against test instances only (for final reporting).

        Returns dict with keys: test_score, test_feedback, raw_results
        """
        return self._evaluate_on_split(solve_source, split="test")

    def _evaluate_on_split(self, solve_source: str, split: str = "dev") -> dict:
        """Run evaluation on dev or test split.

        Strategy: evaluate ALL instances within selected files (to preserve
        index alignment for norm_score), then normalize, then filter to the
        target instance indices.

        Dev: only files in dev_map → evaluate all instances → normalize → keep dev indices.
        Test: all files → evaluate all instances → normalize → keep non-dev indices.
        If no dev_map: evaluate all files and all instances for both splits.
        """
        results: dict[str, tuple[list, str | None]] = {}

        # Dev: only files that appear in dev_map (skip non-dev files entirely).
        # Test: all files.
        if split == "dev" and self._dev_map:
            case_files = [f for f in self.all_cases if f in self._dev_map]
        else:
            case_files = self.all_cases

        logger.debug(
            "Eval %s split for %s: %d files (of %d total), instance_workers=%d",
            split.upper(), self.task_name, len(case_files), len(self.all_cases),
            self._instance_workers,
        )

        # Phase 1: Load all data files
        file_instances: dict[str, list] = {}
        for case_file in case_files:
            file_path = str(self.task_dir / case_file)
            try:
                instances = self.load_data(file_path)
                if not isinstance(instances, (list, tuple)):
                    results[case_file] = ([], f"load_data returned {type(instances).__name__}")
                    continue
            except Exception as e:
                results[case_file] = ([], f"load_data error: {e}")
                continue
            file_instances[case_file] = list(instances)

        # Phase 2: Build work items — (case_file, idx, instance)
        work_items: list[tuple[str, int, dict]] = []
        for case_file, instances in file_instances.items():
            for idx, instance in enumerate(instances):
                work_items.append((case_file, idx, instance))

        total_instances = len(work_items)
        total_errors = 0
        total_timeouts = 0

        # Pre-allocate result structures to preserve index alignment
        for case_file, instances in file_instances.items():
            results[case_file] = ([None] * len(instances), None)

        # Aggregate inner-LLM usage across every instance subprocess for
        # this split. Surfaced on the EvalResult so summary.json's
        # ``inner_*`` fields cover CO-Bench solve()'s own llm() calls —
        # not just the orchestrator's outer calls.
        agg_usage = _empty_usage()

        if self._instance_workers <= 1 or total_instances <= 1:
            # Sequential evaluation
            for case_file, idx, instance in work_items:
                status, value, usage = _run_with_timeout(
                    self.config_path, instance, solve_source, self.timeout,
                    llm_config_dict=self._llm_config_dict,
                    inner_log_path=self._inner_log_path,
                )
                agg_usage = _add_usage(agg_usage, _coerce_usage(usage))
                scores_list = results[case_file][0]
                scores_list[idx] = value
                if status != "ok":
                    total_errors += 1
                    if "Timeout" in str(value):
                        total_timeouts += 1
        else:
            # Parallel instance evaluation using ThreadPoolExecutor.
            # Each thread calls _run_with_timeout which spawns its own Process.
            with ThreadPoolExecutor(max_workers=self._instance_workers) as pool:
                future_to_key = {}
                for case_file, idx, instance in work_items:
                    # kwargs match the sequential branch above. Switching
                    # to positional was fragile: any future signature
                    # change to _run_with_timeout would silently misroute
                    # arguments only on the parallel path.
                    future = pool.submit(
                        _run_with_timeout,
                        config_path=self.config_path,
                        instance=instance,
                        solve_source=solve_source,
                        timeout=self.timeout,
                        llm_config_dict=self._llm_config_dict,
                        inner_log_path=self._inner_log_path,
                    )
                    future_to_key[future] = (case_file, idx)

                for future in as_completed(future_to_key):
                    case_file, idx = future_to_key[future]
                    try:
                        status, value, usage = future.result()
                    except Exception as e:
                        status, value, usage = "error", f"Future exception: {e}", _empty_usage()
                    agg_usage = _add_usage(agg_usage, _coerce_usage(usage))

                    scores_list = results[case_file][0]
                    scores_list[idx] = value
                    if status != "ok":
                        total_errors += 1
                        if "Timeout" in str(value):
                            total_timeouts += 1

        if total_errors > 0:
            logger.debug(
                "Eval %s %s: %d instances, %d errors (%d timeouts)",
                self.task_name, split, total_instances, total_errors, total_timeouts,
            )

        # Normalize scores (needs full index alignment per file)
        if self.norm_score is not None:
            try:
                results = self.norm_score(results)
            except Exception as e:
                logger.warning("norm_score failed for %s: %s", self.task_name, e)
                # Fail CLOSED. norm_score maps raw objective values into the
                # comparative ~[0,1] scale; if it raises, the still-raw results
                # are MEANINGLESS as scores. Averaging them would mark a
                # candidate successful on huge raw magnitudes (and, for a
                # minimization task, INVERT the optimization direction). Mark
                # every per-file entry as an error so _average_score counts them
                # as 0 (matching the held-out Docker runner, which fails closed
                # on a norm_score fault) rather than scoring on un-normalized
                # objective values.
                results = {
                    case_file: (scores, err or f"norm_score failed: {e}")
                    for case_file, (scores, err) in results.items()
                }

        # NOW filter to target instance indices (after normalization)
        if self._dev_map:
            filtered: dict[str, tuple[list, str | None]] = {}
            for case_file, (scores, err) in results.items():
                if case_file not in self._dev_map:
                    if split == "test":
                        # File not in dev_map → all instances are test
                        filtered[case_file] = (scores, err)
                    # For dev split, file not in dev_map → skip (already filtered above)
                    continue

                dev_list = self._dev_map[case_file]
                if not dev_list:
                    # Original CO-Bench convention: empty list defaults to [0]
                    dev_list = [0]

                if split == "dev":
                    selected = [scores[i] for i in dev_list if i < len(scores)]
                else:
                    # Test: all indices NOT in dev
                    dev_set = set(dev_list)
                    selected = [s for i, s in enumerate(scores) if i not in dev_set]

                if selected or err:
                    # Keep files with selected instances AND errored files
                    # (load_data failure / non-list return set ``err`` with empty
                    # scores). Without ``or err`` an in-dev_map errored file's
                    # empty selection would be silently dropped here, so its
                    # ``err`` channel never reaches _average_score's
                    # ``if err: continue`` count-as-0 logic — inflating the
                    # average. The out-of-dev_map branch above already keeps
                    # errored files unconditionally; this makes the two symmetric.
                    filtered[case_file] = (selected, err)

            results = filtered

        # Compute average score
        avg_score = self._average_score(results)

        # Build feedback string
        feedback_lines = []
        for case, (scores, err) in results.items():
            if err:
                feedback_lines.append(f"{case} -> Error: {err}")
            else:
                score_strs = [
                    f"{float(s):.3f}" if isinstance(s, (int, float)) else str(s)
                    for s in scores[:10]
                ]
                feedback_lines.append(f"{case} -> Scores: {score_strs}")
        score_label = "Dev" if split == "dev" else "Test"
        feedback_lines.append(f"Avg {score_label} Score: {avg_score:.4f}")

        return {
            f"{split}_score": avg_score,
            f"{split}_feedback": "\n".join(feedback_lines),
            "raw_results": results,
            # Aggregate inner-LLM usage across all instance subprocesses.
            # COBenchAdapter.evaluate{,_test} reads this onto the EvalResult.
            "inner_usage": agg_usage,
        }

    def _average_score(self, results: dict) -> float:
        """Compute average score across cases.

        Matches original CO-Bench: error cases contribute 0 and still
        count toward the denominator (so errors pull the average down
        instead of being silently ignored). This holds at BOTH granularities:

        * file granularity — an ``err``-bearing file contributes 0 but still
          counts in ``n`` (the ``if err: continue`` path below); and
        * instance granularity — a solve()/timeout failure is recorded as a
          non-numeric error STRING (``_run_with_timeout`` returns
          ``("error", "Timeout (10s)"/<msg>, …)`` stored verbatim at
          ``scores_list[idx]``), and a non-finite score is meaningless. Both
          count as 0.0 in the per-file mean — matching the held-out gate runner
          (which appends 0.0 for infeasible/error instances) so the production
          dev/test scores and the gate's full-set metric use the SAME estimator.
          Dropping them would silently take the per-file mean over only the
          feasible subset, inflating the average.

        Only a genuine ``None`` "skip this instance" sentinel from a norm_score
        implementation is excluded from the per-file denominator entirely (it is
        not a failure — it is "no measurement"), so true errors are never
        silently voided.
        """
        if not results:
            return 0.0
        total = 0.0
        n = len(results)
        for _case, (scores, err) in results.items():
            if err:
                continue  # contributes 0 to total, but n still counts it
            contributions = []
            for x in scores:
                if x is None:
                    # Explicit norm_score "skip" sentinel — not a failure;
                    # excluded from this file's denominator.
                    continue
                if (
                    isinstance(x, (int, float))
                    and not isinstance(x, bool)
                    and math.isfinite(x)
                ):
                    contributions.append(float(x))
                else:
                    # Error/timeout string or non-finite score → counts as 0.0
                    # (pulls the mean down) instead of being dropped.
                    contributions.append(0.0)
            if contributions:
                total += sum(contributions) / len(contributions)
        return total / n if n > 0 else 0.0


# ---------------------------------------------------------------------------
# Forensic #2 held-out SANDBOX verifier (consulted ONLY under --verified-code).
# ---------------------------------------------------------------------------

#: The single CO-Bench task with a held-out sandbox runner (the crew bed used to
#: prove the verified-code channel). Other tasks fall back to the Stub keep.
_CREW_TASK_NAME = "Crew scheduling"
#: best-of-N gemma baseline on the crew TEST split (the held-out gate floor).
_CREW_BEST_OF_N = 0.657797836513118

#: Held-out value-oracle runner executed INSIDE the --network none container by
#: SandboxedHeldoutVerifier. SINGLE SOURCE OF TRUTH for the crew gate:
#: scripts/experiments/metan_crew_verified_test.py imports this constant as its
#: CREW_RUNNER — do not fork a copy there. It loads the candidate helper +
#: the crew config (the human value-oracle, read ONLY in-container), scores the
#: helper over the HELD-OUT (test) split, and PASSES iff the helper's FULL-SET
#: mean (zeros-included, matched to the full-set best-of-N gate) beats best-of-N
#: AND it is feasible on a majority of held-out instances.
#:
#: PRECONDITION (F120) — crew-solo only. The runner's split logic deliberately
#: diverges from ``_TaskEvaluator``:
#:   (a) an EMPTY dev list means "no dev instances" here (the ``set((dev_map or
#:       {}).get(fname, []) or [])`` line below), while the evaluator defaults
#:       an empty list to ``[0]`` (the "Original CO-Bench convention" branch in
#:       ``_TaskEvaluator._evaluate_solution``); and
#:   (b) the runner lists ``*.txt`` data files only, while the evaluator lists
#:       every non-``.py``/``__pycache__``/``.DS_Store`` entry.
#: Both divergences are UNREACHABLE for crew: its ``config.get_dev()`` returns a
#: non-empty index list for all 10 data files, and every crew data file is
#: ``.txt`` (both facts pinned by tests/test_r6b_co_bench_crew_gate.py, which
#: fails loudly if the dataset ever changes shape). This is sound ONLY under the
#: crew-solo gate in ``make_heldout_verifier``; anyone generalizing this runner
#: beyond crew MUST adopt the evaluator's ``[0]``-default convention and its
#: file filter (and re-baseline the gate). Do NOT edit the runner string itself:
#: its bytes must stay comparable with prior --verified-code gate runs.
_CREW_HELDOUT_RUNNER = r'''import sys, json, importlib.util, statistics

HELPER_NAME = sys.argv[1] if len(sys.argv) > 1 else "solve_crew_scheduling"
SPLIT = sys.argv[2] if len(sys.argv) > 2 else "test"          # held-out = test split
GATE = float(sys.argv[3]) if len(sys.argv) > 3 else 0.657797836513118

out = {"loaded": False, "helper_present": False, "n_total": 0, "n_feasible": 0,
       "feasibility_rate": 0.0, "full_mean": 0.0, "feasible_subset_mean": 0.0,
       "passed": False, "error": ""}


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


try:
    cfg = _load("/app/crew/config.py", "crew_cfg")
    cand = _load("/app/_candidate.py", "crew_cand")
    out["loaded"] = True
    helper = getattr(cand, HELPER_NAME, None)
    out["helper_present"] = callable(helper)
    if not callable(helper):
        print(json.dumps(out)); sys.exit(0)

    dev_map = cfg.get_dev() if hasattr(cfg, "get_dev") else None
    import os
    files = sorted(f for f in os.listdir("/app/crew")
                   if f.endswith(".txt"))
    per_file = []                      # mean feasible score per file (full)
    feas_scores = []                   # all feasible per-instance scores
    n_total = 0; n_feas = 0
    for fname in files:
        path = "/app/crew/" + fname
        try:
            instances = cfg.load_data(path)
        except Exception:
            continue
        # which indices are HELD-OUT for this split
        dev_idx = set((dev_map or {}).get(fname, []) or [])
        sel = []
        for idx in range(len(instances)):
            in_dev = idx in dev_idx
            if (SPLIT == "dev" and in_dev) or (SPLIT == "test" and not in_dev):
                sel.append(idx)
        if not sel:
            continue
        raw = [None] * len(instances)
        for idx in range(len(instances)):       # eval ALL for norm alignment
            inst = instances[idx]
            try:
                sol = helper(inst["N"], inst["K"], inst["time_limit"],
                             inst["tasks"], inst["arcs"])
                sol = {str(k): v for k, v in sol.items()}
                raw[idx] = cfg.eval_func(**inst, **sol)
            except Exception:
                raw[idx] = None
        results = {fname: (raw, None)}
        if hasattr(cfg, "norm_score"):
            results = cfg.norm_score(results)
        normed = results[fname][0]
        file_inst = []
        for idx in sel:
            n_total += 1
            s = normed[idx] if idx < len(normed) else None
            if isinstance(s, (int, float)) and s > 0:
                n_feas += 1
                feas_scores.append(float(s))
                file_inst.append(float(s))
            else:
                file_inst.append(0.0)
        if file_inst:
            per_file.append(statistics.mean(file_inst))
    out["n_total"] = n_total
    out["n_feasible"] = n_feas
    out["feasibility_rate"] = (n_feas / n_total) if n_total else 0.0
    out["full_mean"] = statistics.mean(per_file) if per_file else 0.0
    out["feasible_subset_mean"] = statistics.mean(feas_scores) if feas_scores else 0.0
    # PASS = the verified helper beats best-of-N on a MATCHED denominator (the
    # full-set mean, zeros-included — same denominator as the best-of-N GATE,
    # which is itself a full-set metric) AND is feasible on a majority of
    # held-out instances. (A feasible-subset mean would drop the zeros the GATE
    # keeps, comparing mismatched denominators — see audit T1.1.)
    out["passed"] = (out["full_mean"] > GATE
                     and out["feasibility_rate"] >= 0.5)
except Exception as e:
    import traceback
    out["error"] = traceback.format_exc()[-1500:]

print(json.dumps(out))
'''

#: The crew deploy contract keys — exactly the argument names ``_CREW_HELDOUT_RUNNER``
#: forwards positionally (``helper(inst["N"], inst["K"], inst["time_limit"],
#: inst["tasks"], inst["arcs"])``) and the keys CO-Bench passes to ``solve(**instance)``.
#: SINGLE SOURCE OF TRUTH for the crew positional key list: the verify gate uses it to
#: reject any helper the name-based deploy wrapper (``adoption._build_deploy_wrapper``)
#: could not bind, so every VERIFIED helper is also name-DEPLOYABLE.
_CREW_DEPLOY_KEYS = frozenset({"N", "K", "time_limit", "tasks", "arcs"})


def deploy_name_forwardable(source: str, name: str, keys: frozenset[str]) -> bool:
    """True iff the name-based deploy wrapper can bind ``name`` given ONLY ``keys``.

    Contract mirror of ``adoption._build_deploy_wrapper``: positional-only params are
    forwarded POSITIONALLY as the contiguous present-prefix (a gap breaks forwarding),
    everything else is forwarded BY KEYWORD, and ``**kwargs`` is irrelevant to
    required-arg binding. A helper is forwardable iff every argument the wrapper is
    OBLIGED to supply has a name in ``keys``:

    * any positional-only param name absent from ``keys`` -> not forwardable (an
      absent name breaks the positional present-prefix, stranding later params);
    * any REQUIRED (default-less) positional-or-keyword / keyword-only param name
      absent from ``keys`` -> not forwardable.

    Params WITH defaults whose name is absent are safe (they default identically under
    positional-verify and name-deploy). Fail-closed: an unparseable / missing ``def``
    returns False. Param taxonomy mirrors adoption.py:154-161 (posonlyargs, args,
    kwonlyargs, kwarg).
    """
    try:
        tree = ast_mod.parse(source)
    except SyntaxError:
        return False
    func = None
    for node in ast_mod.walk(tree):
        if (
            isinstance(node, (ast_mod.FunctionDef, ast_mod.AsyncFunctionDef))
            and node.name == name
        ):
            func = node
            break
    if func is None:
        return False
    a = func.args
    # Positional-only: forwarded positionally as a contiguous present-prefix, so ANY
    # absent name (defaulted or not) can strand a later param — require them all.
    for arg in a.posonlyargs:
        if arg.arg not in keys:
            return False
    # ``defaults`` align to the TAIL of (posonlyargs + args); the leading
    # len(combined) - len(defaults) positional params are REQUIRED.
    n_required_pos = len(a.posonlyargs) + len(a.args) - len(a.defaults)
    for i, arg in enumerate(a.args):
        combined_idx = len(a.posonlyargs) + i
        if combined_idx < n_required_pos and arg.arg not in keys:
            return False
    # Keyword-only params with no default (kw_defaults[i] is None) are required.
    for arg, default in zip(a.kwonlyargs, a.kw_defaults):
        if default is None and arg.arg not in keys:
            return False
    return True


class _SignatureGuardedVerifier(SandboxedHeldoutVerifier):
    """Crew verify gate that DROPS any helper the name-based deploy wrapper could not
    bind, BEFORE it reaches the sandbox.

    Restores the invariant "every verified helper is name-deployable": a helper that
    passes verify has a signature name-forwardable from ``_CREW_DEPLOY_KEYS``, so its
    deployed score equals its verified score. Convention-violating renamed helpers are
    rejected fail-closed (they can be re-authored with the canonical crew names).
    """

    def verify(
        self,
        name: str,
        source: str,
        task_id: str,
        context_sources: list[str],
    ) -> VerifyResult:
        if not deploy_name_forwardable(source, name, _CREW_DEPLOY_KEYS):
            return VerifyResult(
                passed=False,
                evidence="signature not name-forwardable from crew keys",
                ran_in_sandbox=False,
            )
        return super().verify(name, source, task_id, context_sources)


# ---------------------------------------------------------------------------
# Main adapter
# ---------------------------------------------------------------------------

class COBenchAdapter(BenchmarkAdapter):
    """Adapter for CO-Bench combinatorial optimization problems."""

    def __init__(
        self,
        data_dir: str = "./data/co_bench",
        timeout: int = 10,
        task_names: list[str] | None = None,
        instance_workers: int = 0,
        llm_config_dict: dict | None = None,
        inner_log_path: str | Path | None = None,
    ):
        self.data_dir = Path(data_dir)
        self.timeout = timeout
        self.task_names = task_names or CO_BENCH_TASKS
        self._llm_config_dict = llm_config_dict
        self._inner_log_path = str(inner_log_path) if inner_log_path else None
        self._evaluators: dict[str, _TaskEvaluator] = {}
        self._data_cache: dict[str, Any] = {}
        # Instance-level parallelism: env var > explicit > default.
        # Default capped at 8 to bound subprocess fan-out: each worker spawns
        # an mp.Process per instance, and unbounded cpu_count() (= 18 on this
        # box) caused FD exhaustion when multiple ablations ran concurrently.
        # Slowest-instance dominates per-task wall clock anyway, so >8 rarely
        # helps. Override with META_N_INSTANCE_WORKERS or the explicit arg.
        env_workers = os.environ.get("META_N_INSTANCE_WORKERS")
        if env_workers is not None:
            self.instance_workers = int(env_workers)
        elif instance_workers > 0:
            self.instance_workers = instance_workers
        else:
            self.instance_workers = min(os.cpu_count() or 4, 8)

    @property
    def name(self) -> str:
        return "co_bench"

    def split_type(self) -> str:
        # R5 flip: evaluate() scores the dev_map instance indices and
        # evaluate_test() the disjoint non-dev complement (see
        # _TaskEvaluator._evaluate_on_split) — a genuine held-out split.
        # NOTE for the overfit cluster (plan §3.3/§6.2/§6.4, reconciled R5):
        # this label describes SPLIT STRUCTURE only. CO-Bench's dev score is
        # the deterministic real objective (not a proxy), and the in-loop
        # re-solve machinery (get_test_task) does not exist here — gate that
        # machinery on capability, not on this label.
        return "held_out"

    def code_library_is_live(self) -> bool:
        # Measured helper call-rate ~0: the solver regenerates code inline rather
        # than importing the prepended helpers (roadmap v2 4.3) — demote them.
        return False

    def make_heldout_verifier(self):  # -> HeldoutVerifier | None
        """Forensic #2 — CO-Bench held-out SANDBOX verifier (P1b).

        Consulted ONLY when ``--verified-code`` is ON (the orchestrator's
        verify-gate is guarded by ``if not self.config.verified_code: return``),
        so with the flag OFF this override is never reached and behavior is
        byte-identical to the base ``None`` default.

        SCOPE: a real :class:`SandboxedHeldoutVerifier` is returned ONLY for the
        crew task (the one task with a sandbox value-oracle runner). The runner
        is crew-specific (it calls ``helper(N, K, time_limit, tasks, arcs)`` and
        scores via the crew ``config.py`` value-oracle), so returning it for
        OTHER tasks would fail-closed-DROP every helper on an unsupported task.
        For any non-crew task selection we return ``None`` ⇒ the orchestrator
        falls back to the capability-preserving Stub (keep + flag UNVERIFIED).

        GATE: the helper PASSES iff, on the HELD-OUT (test) split, its
        full-set mean (zeros-included, matched to the full-set best-of-N gate)
        beats best-of-N AND it is feasible on >= 50% of held-out instances
        (exactly the crew harness predicate). Fail-closed is
        structural: ``_docker_run_json`` maps timeout / non-JSON / exec-error to
        ``{"passed": False}`` and ``SandboxedHeldoutVerifier.verify`` wraps any
        crash to ``passed=False``, so a broken helper is always DROPPED.

        CAVEAT: a kept CO-Bench helper is still ZEROED at staging because
        ``code_library_is_live()`` is False — the DROP decision affects what is
        stored/penalized, NOT deployment, unless ``--force-code-library-live``
        is also ON. P1b is only end-to-end meaningful in the
        {--verified-code + --force-code-library-live} combo.

        PRECONDITION (F120): the runner's split conventions are only correct
        under this crew-solo gate — see the ``_CREW_HELDOUT_RUNNER`` header
        comment.
        """
        # The verifier is resolved ONCE per candidate and applied to the
        # candidate's GLOBAL code_library (the orchestrator iterates every helper
        # through this single verifier, not per-task). The runner is crew-specific
        # — it calls ``helper(N, K, time_limit, tasks, arcs)`` and scores via the
        # crew value-oracle — so it must gate ONLY a crew-only task selection. A
        # membership test (``_CREW_TASK_NAME not in ...``) would also fire for a
        # MULTI-task selection like ``[crew, "Set covering"]`` and then
        # fail-closed-DROP every non-crew helper against the crew oracle. Require
        # crew to be the SOLE task so any other (incl. multi-task) selection falls
        # back to None ⇒ the capability-preserving Stub (keep + flag UNVERIFIED).
        if list(self.task_names or []) != [_CREW_TASK_NAME]:
            return None

        # .resolve() coerces the Docker bind-mount SOURCE to absolute (a
        # Docker -v requirement); self.data_dir stays untouched so load_tasks
        # / _load_task_data remain byte-identical. The verifier is a
        # _SignatureGuardedVerifier (a SandboxedHeldoutVerifier subclass): it
        # rejects any helper not name-forwardable from _CREW_DEPLOY_KEYS BEFORE
        # the sandbox, so a verify PASS implies the name-deploy wrapper binds
        # identically (deployed score == verified score).
        crew_dir = (self.data_dir / _CREW_TASK_NAME).resolve()
        return _SignatureGuardedVerifier(
            image="python:3.12-slim",
            runner_src=_CREW_HELDOUT_RUNNER,
            mounts=[(str(crew_dir), "/app/crew")],
            success_predicate=lambda r: bool(r.get("passed")),
            runner_args=["test", repr(_CREW_BEST_OF_N)],
        )

    def load_tasks(self, limit: int | None = None) -> list[TaskDescription]:
        """Load CO-Bench problems as TaskDescription objects."""
        tasks = []
        names = self.task_names[:limit] if limit is not None else self.task_names

        for task_name in names:
            task_dir = self.data_dir / task_name
            if not task_dir.exists():
                logger.warning("Task directory not found: %s — skipping", task_dir)
                continue

            data = self._load_task_data(task_name)
            if data is None:
                continue

            task = TaskDescription(
                task_id=_task_id_from_name(task_name),
                description=data["problem_description"]
                + _execution_contract(self.timeout),
                metadata={
                    "benchmark": "co_bench",
                    "task_name": task_name,
                    "solve_template": data["solve_template"],
                    "solution_language": "python",
                    "instance_timeout_s": self.timeout,
                },
            )
            tasks.append(task)

        logger.info("Loaded %d CO-Bench tasks", len(tasks))
        return tasks

    async def _evaluate_split(
        self, task: TaskDescription, solution: str, split: str
    ) -> EvalResult:
        """Shared body of :meth:`evaluate` / :meth:`evaluate_test`.

        ``split`` selects the evaluator method (``evaluate`` for "dev",
        ``evaluate_test`` otherwise) and the ``{split}_score`` /
        ``{split}_feedback`` result keys; every EvalResult field and log line
        is otherwise identical between the two splits.
        """
        label = "Dev" if split == "dev" else "Test"
        task_name = task.metadata.get("task_name", "")
        if not task_name:
            return EvalResult(success=False, feedback="No task_name in metadata")

        try:
            evaluator = self._get_evaluator(task_name)
        except Exception as e:
            return EvalResult(success=False, feedback=f"Failed to load evaluator: {e}")

        # Run evaluation in thread pool (it spawns subprocesses internally)
        loop = asyncio.get_running_loop()
        eval_start = time.time()
        try:
            method = getattr(
                evaluator, "evaluate" if split == "dev" else "evaluate_test"
            )
            # functools.partial avoids the lambda late-binding hazard
            # (closures capture by reference; if `solution` were ever
            # hoisted to instance state, two concurrent calls would race).
            result = await loop.run_in_executor(
                None, functools.partial(method, solution)
            )
        except Exception as e:
            logger.warning(
                "%s eval error for %s: %s (%.1fs)",
                label, task_name, e, time.time()-eval_start,
            )
            return EvalResult(
                success=False,
                score=0.0,
                feedback=f"Evaluation error: {e}",
            )

        score = result.get(f"{split}_score", 0.0)
        logger.debug(
            "%s eval: %s → score=%.4f (%.1fs)",
            label, task_name, score, time.time()-eval_start,
        )
        usage = _coerce_usage(result.get("inner_usage", {}))
        return EvalResult(
            success=score > 0.0,
            score=score,
            raw_score=score,
            feedback=result.get(f"{split}_feedback", ""),
            inner_tokens=usage["total"],
            inner_prompt_tokens=usage["prompt"],
            inner_completion_tokens=usage["completion"],
            inner_calls=usage["calls"],
        )

    async def evaluate(self, task: TaskDescription, solution: str) -> EvalResult:
        """Evaluate a Python solve() function against CO-Bench dev instances."""
        return await self._evaluate_split(task, solution, "dev")

    async def evaluate_test(self, task: TaskDescription, solution: str) -> EvalResult:
        """Evaluate a Python solve() function against CO-Bench TEST instances.

        Use this for final reporting only — not during search.
        """
        return await self._evaluate_split(task, solution, "test")

    def _load_task_data(self, task_name: str) -> dict | None:
        """Load problem description and solve template from config.py."""
        if task_name in self._data_cache:
            return self._data_cache[task_name]

        config_path = self.data_dir / task_name / "config.py"
        if not config_path.exists():
            logger.warning("config.py not found for %s", task_name)
            return None

        try:
            spec = importlib.util.spec_from_file_location(
                f"cobench_config_{_task_id_from_name(task_name)}", config_path,
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            description = getattr(module, "DESCRIPTION", "")
            solve_source = self._extract_solve_template(config_path)
            problem_description = f"{description}\n\n# Implement in Solve Function\n\n{solve_source}"

            data = {
                "problem_description": problem_description,
                "solve_template": solve_source,
            }
            self._data_cache[task_name] = data
            return data
        except Exception as e:
            logger.warning("Failed to load task %s: %s", task_name, e)
            return None

    def _extract_solve_template(self, config_path: Path) -> str:
        """Extract the solve() function source from config.py."""
        source = config_path.read_text()
        try:
            tree = ast_mod.parse(source)
        except SyntaxError:
            return ""

        for node in ast_mod.walk(tree):
            if isinstance(node, ast_mod.FunctionDef) and node.name == "solve":
                lines = source.splitlines()
                start = node.lineno - 1
                end = node.end_lineno if node.end_lineno else start + 1
                return "\n".join(lines[start:end])

        return ""

    def _get_evaluator(self, task_name: str) -> _TaskEvaluator:
        """Get or create a task evaluator."""
        if task_name not in self._evaluators:
            self._evaluators[task_name] = _TaskEvaluator(
                task_name, self.data_dir, self.timeout,
                instance_workers=self.instance_workers,
                llm_config_dict=self._llm_config_dict,
                inner_log_path=self._inner_log_path,
            )
        return self._evaluators[task_name]

    # ------------------------------------------------------------------ #
    # External-agent factory hooks (plan §2.6, §5.9)                      #
    # ------------------------------------------------------------------ #
    # CO-Bench is the honest weak structural fit for the external-agent
    # spine: the agent's task is reframed to "author ``solve.py``" rather
    # than acting in a live container. Only the OpenHands backend is
    # supported here (its REST conversation-DELETE / in-process killpg hard
    # kill works without a Docker container); the Terminus 2 backend, which
    # drives a tmux/Docker session CO-Bench never provisions, is rejected at
    # construction with a clear ``ValueError`` (plan §5.9).

    def make_agent_backend(self, kind: str, **kw) -> "AgentBackend":
        """Build the authoring backend for the external CO-Bench path.

        Only ``"openhands"`` is supported: it has an enforceable hard kill
        (REST conversation DELETE, or — on the in-process path — a
        ``killpg`` on a ``start_new_session=True`` child) that does not rely
        on a Docker container, which CO-Bench's no-op host workspace cannot
        offer (plan §5.9). The Terminus 2 backend is unsupported here because
        it requires the tmux/Docker session this benchmark never stands up.

        Args:
            kind: Backend identifier; must be ``"openhands"``.
            **kw: Forwarded verbatim to :class:`OpenHandsBackend` (``model``,
                ``api_base``, ``api_key``, ``provider_env_var``, ``mode`` …).

        Returns:
            A configured :class:`OpenHandsBackend`.

        Raises:
            ValueError: If ``kind`` is not ``"openhands"`` (Terminus 2 and any
                other backend are unsupported on the CO-Bench authoring path).
        """
        if kind != "openhands":
            raise ValueError(
                "CO-Bench external authoring supports only the 'openhands' "
                f"backend (got {kind!r}); Terminus 2 needs a tmux/Docker "
                "session CO-Bench does not provide (plan §5.9)."
            )
        # Lazy import: the OpenHands SDK is NOT touched here (the backend
        # itself imports it lazily inside its run path), so this keeps the
        # external_agents package importable without ``openhands`` installed.
        from meta_n.core.external_agents.backends.openhands import OpenHandsBackend

        # Bind the runner's audit-copy filename to this benchmark's authoritative
        # ``_SOLVE_FILENAME`` (the same file ``COBenchEnvProvider.extract_solution``
        # reads), so the redundant runner copy and the spine's FS read can never
        # name different files if ``_SOLVE_FILENAME`` ever changes.
        kw.setdefault("solution_file", _SOLVE_FILENAME)
        # Opt into the Docker-sandboxed runner path when META_N_OH_SANDBOX is set
        # (the OH terminal tool then shells out against a container, not the bare
        # host). The backend default stays the proven host path; an explicit
        # caller-supplied ``sandbox`` kwarg always wins over the env toggle.
        if "sandbox" not in kw:
            toggle = os.environ.get("META_N_OH_SANDBOX", "").strip().lower()
            if toggle in ("1", "true", "yes", "on"):
                kw["sandbox"] = True
        return OpenHandsBackend(**kw)

    def make_env_provider(self, kind: str) -> "COBenchEnvProvider":
        """Build the no-op host-workspace env provider for CO-Bench.

        Args:
            kind: Backend identifier (accepted for parity with the other
                adapters; CO-Bench provisions the same host-workspace env
                regardless of which authoring backend runs).

        Returns:
            A :class:`COBenchEnvProvider` bound to this adapter.
        """
        return COBenchEnvProvider(self)

    def make_scorer(self, kind: str) -> "COBenchScorer":
        """Build the scorer that delegates to ``_evaluate_on_split``.

        Args:
            kind: Backend identifier (accepted for parity; scoring is
                backend-independent — the authored ``solve.py`` is evaluated
                identically however it was produced).

        Returns:
            A :class:`COBenchScorer` bound to this adapter.
        """
        return COBenchScorer(self)


class COBenchExecutor(AdapterExecutor):
    """Executor that evaluates Python code via CO-Bench scoring.

    Replaces LocalExecutor for CO-Bench tasks. Pure ``AdapterExecutor``
    defaults (``.4f`` score format, feedback-truncation error summary) —
    byte behavior pinned by tests/test_refine_benchmark.py goldens.
    """


# ---------------------------------------------------------------------------
# External-agent env provider + scorer (plan §2.5, §5.9)
# ---------------------------------------------------------------------------

#: Workspace-relative filename the authoring agent is expected to write. The
#: env provider stages the solve template here and reads the agent's authored
#: solution back from it in :meth:`COBenchEnvProvider.extract_solution`.
_SOLVE_FILENAME = "solve.py"


class _COBenchEnv:
    """Opaque per-run handle for the no-op host-workspace CO-Bench env.

    CO-Bench is the honest structural weak case for the external-agent spine
    (plan §5.9): there is no Docker container, so the "environment" is just an
    isolated host scratch directory under :attr:`EnvLease.workdir`. This handle
    bundles that directory and a mutable ``agent_pid`` slot the OpenHands
    in-process path can stamp with its child's pid so the lease's synchronous
    ``hard_kill`` can reap a wedged agent via :func:`_kill_process_tree`.

    Attributes:
        workspace_root: The per-run host scratch directory. The solve template
            is staged here and the agent's authored ``solve.py`` is read back
            from here.
        workspace_handle: The handle passed to the backend as
            ``AgentRunContext.workspace``. This is the env object **itself**
            (parity with TB's ``_TBTerminus2Env.workspace_handle = self``) so
            the backend's ``setattr(ctx.workspace, "agent_pid", pid)`` stamp
            lands on the slot the lease's ``hard_kill`` reads. The object is
            ``str``/``os.fspath``-able to ``str(workspace_root)`` so the
            backend's path-stringifying callsites still resolve the host path
            (the OpenHands backend treats it as the host workspace directory it
            operates in / copies files to).
        agent_pid: Pid of the agent's child process when the OpenHands
            in-process path launches one under ``start_new_session=True``; left
            ``None`` on the REST path (whose hard kill is the conversation
            DELETE). Read by the lease ``hard_kill`` closure.
    """

    def __init__(self, workspace_root: Path):
        self.workspace_root = workspace_root
        # Carry the env itself as the workspace handle (parity with TB's
        # ``_TBTerminus2Env.workspace_handle = self``). The backend stamps the
        # agent child pid via ``setattr(ctx.workspace, "agent_pid", pid)``
        # (openhands.py:347); if the handle were a bare ``str(workspace_root)``
        # that setattr would raise on the immutable str (swallowed by the
        # backend's ``contextlib.suppress``), leaving ``agent_pid`` None and the
        # synchronous ``hard_kill`` a dead no-op. Exposing ``self`` routes the
        # stamp onto this object's ``agent_pid`` slot that ``hard_kill`` reads.
        self.workspace_handle = self
        self.agent_pid: int | None = None

    def __fspath__(self) -> str:
        # ``os.fspath`` / ``Path(env_handle)`` resolve to the host workspace dir,
        # so the backend's path-stringifying callsites keep the same value they
        # got from the old ``str(workspace_root)`` handle.
        return str(self.workspace_root)

    def __str__(self) -> str:
        return str(self.workspace_root)


class COBenchEnvProvider(AgentEnvProvider):
    """No-op host-workspace env provider for the external CO-Bench path.

    CO-Bench does not provision a Docker container; the agent's reframed task is
    to *author* ``solve.py`` in an isolated host scratch directory (plan §5.9).
    This provider therefore:

    * creates the workspace under :attr:`EnvLease.workdir` and seeds it with the
      task's ``solve`` template so the agent has a starting point;
    * stages the injection plan's helper files into that workspace;
    * reads the authored ``solve.py`` back out as the solution; and
    * installs a synchronous :attr:`EnvLease.hard_kill` that reaps a wedged
      in-process agent via the existing :func:`_kill_process_tree` (a
      ``killpg`` on the child's session group), closing the "no enforceable
      kill on the no-Docker path" gap (plan §5.9, §6.5d). On the REST OpenHands
      path no child pid is recorded and the real hard kill is the conversation
      DELETE owned by the backend.

    The :class:`DockerRunGuard` still leases and cleans the scratch dir; this
    provider simply does no container work inside it.
    """

    def __init__(self, adapter: COBenchAdapter):
        self._adapter = adapter

    @contextlib.asynccontextmanager
    async def provision(self, task: TaskDescription, lease: EnvLease):
        """Stand up the no-op host workspace and install ``hard_kill``.

        Creates ``{lease.workdir}/workspace`` (``0o700`` via the guard's
        ``mkdtemp`` parent), seeds it with the task's ``solve`` template, and
        installs a synchronous ``lease.hard_kill`` that reaps any agent child
        pid stamped onto the env (the OpenHands in-process path) using the
        existing :func:`_kill_process_tree`. Yields the :class:`_COBenchEnv`.

        Args:
            task: The CO-Bench task being authored.
            lease: The held :class:`EnvLease` (scratch dir + cleanup-hook slots).

        Yields:
            A :class:`_COBenchEnv` whose ``workspace_handle`` the backend uses.
        """
        workspace_root = lease.workdir / "workspace"
        workspace_root.mkdir(parents=True, exist_ok=True)

        # Seed the authoring target with the task's solve template so the agent
        # edits a real starting point rather than an empty file.
        template = task.metadata.get("solve_template", "") if task.metadata else ""
        if template:
            (workspace_root / _SOLVE_FILENAME).write_text(template)

        env = _COBenchEnv(workspace_root)

        def _hard_kill() -> None:
            """Synchronously reap a wedged in-process agent (plan §5.9, §6.5d).

            Reuses :func:`_kill_process_tree` (``killpg`` once the child has
            ``setsid``'d, psutil-walk fallback otherwise). A no-op when no child
            pid was recorded (the REST path, whose hard kill is the conversation
            DELETE owned by the backend).
            """
            pid = env.agent_pid
            if pid is None:
                return
            try:
                _kill_process_tree(pid)
            except Exception:  # noqa: BLE001 - hard kill must never raise out
                logger.debug("CO-Bench hard_kill of pid %s failed", pid, exc_info=True)

        lease.hard_kill = _hard_kill

        try:
            yield env
        finally:
            # The DockerRunGuard owns ``rmtree`` of ``lease.workdir``; nothing
            # else to tear down for the no-op host workspace.
            pass

    async def stage_files(self, env: object, files: dict[str, str]) -> None:
        """Write the injection plan's helper files into the workspace.

        Args:
            env: The :class:`_COBenchEnv` from :meth:`provision`.
            files: Map of workspace-relative path -> contents (empty for the
                gen0 vanilla-agent baseline, a valid no-op).
        """
        if not files:
            return
        root: Path = env.workspace_root  # type: ignore[attr-defined]
        for rel, content in files.items():
            # Defense-in-depth (parity with OpenHandsBackend.stage_files and the
            # container-staging writers _runner_common.stage_helper_files /
            # _external_tb._staged_files): this writes to the BARE HOST (no
            # Docker), so reject an absolute or ``..``-traversing key that would
            # escape the workspace root and write an arbitrary host file.
            # Skip-with-warning, never abort the stage.
            rel_path = Path(rel)
            if rel_path.is_absolute() or ".." in rel_path.parts:
                logger.warning(
                    "COBenchEnvProvider.stage_files: skipping unsafe staged path %r",
                    rel,
                )
                continue
            target = root / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)

    async def extract_solution(self, env: object, run: "AgentRunResult") -> str:
        """Read the agent's authored ``solve.py`` out of the workspace.

        Args:
            env: The :class:`_COBenchEnv` from :meth:`provision`.
            run: The backend's :class:`AgentRunResult` (unused — the solution is
                always read from the filesystem for CO-Bench).

        Returns:
            The contents of ``solve.py`` (``""`` if the agent wrote nothing).
        """
        solve_path: Path = env.workspace_root / _SOLVE_FILENAME  # type: ignore[attr-defined]
        try:
            return solve_path.read_text()
        except OSError:
            return ""


def _agent_usage_kwargs(run: "AgentRunResult") -> dict[str, int]:
    """The four ``inner_*`` EvalResult kwargs mirroring the agent's token
    spend from ``run`` (the TBScorer convention — see COBenchScorer)."""
    tokens, prompt_tokens, completion_tokens, calls = mirror_agent_tokens(run)
    return {
        "inner_tokens": tokens,
        "inner_prompt_tokens": prompt_tokens,
        "inner_completion_tokens": completion_tokens,
        "inner_calls": calls,
    }


class COBenchScorer(Scorer):
    """Scorer that hands the authored ``solve.py`` to ``_evaluate_on_split``.

    The agent's authored source is passed **verbatim** to the unchanged CO-Bench
    evaluator (``_TaskEvaluator._evaluate_on_split`` at ``co_bench.py:347`),
    which runs the subprocess instance pool with its ``_kill_process_tree``
    guard, the dev/test split, and ``norm_score`` normalization (plan §5.9).
    ``norm_score()`` is therefore **always** deferred to meta-n's evaluator and
    never re-derived here. The dev split is scored (parity with
    :meth:`COBenchAdapter.evaluate`, the in-search feedback signal).

    The returned :class:`EvalResult` mirrors the agent's inner-token spend from
    ``run`` so cost analysis attributes it correctly (plan §5.8); CO-Bench's own
    ``solve()`` rarely calls ``llm()``, so ``EvalResult.inner_*`` from the
    evaluator and the agent's ``run.agent_*`` are distinct ledgers — the agent's
    authoring spend (``run.agent_*``) is the one surfaced here, matching the
    ``TBScorer`` convention.
    """

    def __init__(self, adapter: COBenchAdapter):
        self._adapter = adapter

    async def score(
        self,
        task: TaskDescription,
        env: object,
        solution: str,
        run: "AgentRunResult",
    ) -> EvalResult:
        """Score ``solution`` against the CO-Bench dev split; never raises.

        Delegates to the adapter's existing dev-split evaluator (which calls the
        unchanged ``_evaluate_on_split``) and folds the result into an
        :class:`EvalResult`, mirroring the agent's inner-token accounting from
        ``run``. An empty solution, a missing ``task_name``, or any evaluator
        fault scores as ``EvalResult(success=False, score=0.0)``.

        Args:
            task: The CO-Bench task being scored.
            env: The live :class:`_COBenchEnv` (unused — scoring runs the source
                in the evaluator's own subprocess pool, not the workspace).
            solution: The authored ``solve.py`` source from
                :meth:`COBenchEnvProvider.extract_solution`.
            run: The backend's :class:`AgentRunResult`, for mirroring tokens.

        Returns:
            An :class:`EvalResult` carrying the normalized dev score, native
            ``raw_score``, ``valid``/``feasible`` flags, and the agent's
            ``inner_*`` token spend.
        """
        # The valid/feasible combinations differ per branch (contract): only
        # the empty-solution branch sets feasible=False; the no-task-name /
        # evaluator-load / eval-error branches set valid=False alone; the
        # success branch sets valid=True, feasible=dev_score > 0.0.
        usage_kw = _agent_usage_kwargs(run)

        if not solution.strip():
            return EvalResult(
                success=False,
                score=0.0,
                feedback="Agent authored no solve.py",
                valid=False,
                feasible=False,
                **usage_kw,
            )

        task_name = task.metadata.get("task_name", "") if task.metadata else ""
        if not task_name:
            return EvalResult(
                success=False,
                score=0.0,
                feedback="No task_name in metadata",
                valid=False,
                **usage_kw,
            )

        try:
            evaluator = self._adapter._get_evaluator(task_name)
        except Exception as e:  # noqa: BLE001 - scorer must never raise out
            return EvalResult(
                success=False,
                score=0.0,
                feedback=f"Failed to load evaluator: {e}",
                valid=False,
                **usage_kw,
            )

        # Run the (subprocess-spawning) dev-split evaluation off the event loop;
        # _evaluate_on_split is the unchanged CO-Bench path (norm_score deferred
        # to the evaluator, _kill_process_tree guards wedged solve()).
        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(
                None,
                functools.partial(evaluator._evaluate_on_split, solution, split="dev"),
            )
        except Exception as e:  # noqa: BLE001 - scorer must never raise out
            return EvalResult(
                success=False,
                score=0.0,
                feedback=f"Evaluation error: {e}",
                valid=False,
                **usage_kw,
            )

        dev_score = result.get("dev_score", 0.0)
        return EvalResult(
            success=dev_score > 0.0,
            score=dev_score,
            raw_score=dev_score,
            feedback=result.get("dev_feedback", ""),
            valid=True,
            feasible=dev_score > 0.0,
            **usage_kw,
        )
