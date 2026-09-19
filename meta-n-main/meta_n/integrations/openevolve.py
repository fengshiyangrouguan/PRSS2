"""OpenEvolve benchmark adapters — AlphaEvolve Math, Symbolic Regression, AlgoTune.

All three domains share the same evaluation contract:
    evaluate(program_path: str) -> dict   with a ``combined_score`` key.

Data download:
    bash scripts/setup_openevolve.sh

Usage:
    adapter = AlphaEvolveMathAdapter(data_dir="./data/openevolve/examples/alphaevolve_math_problems")
    tasks   = adapter.load_tasks(limit=5)
"""

from __future__ import annotations

import asyncio
import functools
import importlib.util
import json
import logging
import math
import multiprocessing as mp
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None  # graceful: config.yaml parsing is best-effort

from meta_n.core.meta_layer import TaskDescription

# Compat re-export: _kill_process_tree was historically importable from here.
from meta_n.integrations._subprocess_utils import (  # noqa: F401
    _kill_process_tree,
    detach_process_group,
    run_process_with_timeout,
)
from meta_n.integrations.benchmark import AdapterExecutor, BenchmarkAdapter, EvalResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------

def _ensure_openevolve_stub(evaluator_dir: str) -> None:
    """Ensure ``from openevolve.evaluation_result import EvaluationResult`` resolves.

    AlgoTune evaluators import this simple dataclass. If the openevolve package
    isn't installed, we create a minimal stub in sys.modules by finding the real
    ``evaluation_result.py`` in the cloned OpenEvolve repo.
    """
    try:
        import openevolve.evaluation_result  # noqa: F401
        return  # already importable
    except (ImportError, ModuleNotFoundError):
        pass

    # Walk up from evaluator_dir to find openevolve/evaluation_result.py
    import types
    p = Path(evaluator_dir)
    for _ in range(10):  # limit depth
        candidate = p / "openevolve" / "evaluation_result.py"
        if candidate.exists():
            stub_pkg = types.ModuleType("openevolve")
            stub_pkg.__path__ = []
            sys.modules["openevolve"] = stub_pkg
            spec = importlib.util.spec_from_file_location(
                "openevolve.evaluation_result", str(candidate),
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            sys.modules["openevolve.evaluation_result"] = mod
            return
        if p.parent == p:
            break
        p = p.parent


def _run_openevolve_eval_in_process(
    evaluator_dir: str,
    program_source: str,
    score_key: str,
    queue: mp.Queue,
    extra_sys_paths: list[str] | None = None,
) -> None:
    """Subprocess target: import evaluator, run evaluate(), put result in queue."""
    detach_process_group()
    try:
        # Inject extra sys.path entries (e.g., AlgoTune repo)
        for p in (extra_sys_paths or []):
            if p not in sys.path:
                sys.path.insert(0, p)

        # Ensure openevolve.evaluation_result is importable (for AlgoTune)
        _ensure_openevolve_stub(evaluator_dir)

        # Write solution to a temp file (system temp dir — problem_dir may be read-only)
        fd, program_path = tempfile.mkstemp(suffix=".py", prefix="_meta_n_oe_")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(program_source)

            # Import evaluator module
            evaluator_path = os.path.join(evaluator_dir, "evaluator.py")
            spec = importlib.util.spec_from_file_location(
                "_oe_evaluator", evaluator_path,
            )
            evaluator_mod = importlib.util.module_from_spec(spec)
            # Ensure evaluator can do relative imports from its own directory
            if evaluator_dir not in sys.path:
                sys.path.insert(0, evaluator_dir)
            spec.loader.exec_module(evaluator_mod)

            result = evaluator_mod.evaluate(program_path)
            # Guard the score conversion: evaluators can return None for the
            # score key, or NaN/inf for buggy scoring code. float(None) raises
            # TypeError and crashes the subprocess; non-finite values poison
            # archive.mean_score downstream. Coerce both to 0.0 here.
            raw = result.get(score_key, result.get("combined_score", None))
            score, sentinel_raw = _coerce_eval_score(raw)
            # Serialize full result dict for feedback
            details = {k: v for k, v in result.items() if _is_json_serializable(v)}
            if sentinel_raw is not None:
                details["raw_failure_sentinel"] = sentinel_raw
            queue.put(("ok", {"score": score, "details": details}))
        finally:
            try:
                os.unlink(program_path)
            except OSError:
                pass
    except Exception as e:
        queue.put(("error", f"{type(e).__name__}: {e}"))


def _is_json_serializable(v: Any) -> bool:
    """Check if a value can be serialized to JSON."""
    try:
        json.dumps(v)
        return True
    except (TypeError, ValueError, OverflowError):
        return False


def _coerce_eval_score(raw) -> tuple[float, "float | None"]:
    """Coerce a raw evaluator score to a finite, sentinel-free float.

    Returns ``(score, raw_failure_sentinel)``. ``None`` / non-finite → ``0.0``;
    a large-negative FAILURE SENTINEL (``<= -1e8``, e.g. openevolve's ``-1e9``)
    → ``0.0`` with the raw value returned for feedback, so it does not poison
    mean-based aggregates / scale-invariant selection (roadmap v2 N4c). The run
    is still a failure (low score) — only the poisoning magnitude is neutralized.
    """
    try:
        score = float(raw) if raw is not None else 0.0
    except (TypeError, ValueError):
        return 0.0, None
    if not math.isfinite(score):
        return 0.0, None
    if score <= -1e8:
        return 0.0, score
    return score, None


def _run_openevolve_with_timeout(
    evaluator_dir: str,
    program_source: str,
    score_key: str,
    timeout: int,
    extra_sys_paths: list[str] | None = None,
) -> tuple[str, Any]:
    """Run OpenEvolve evaluation in a subprocess with timeout.

    Returns ("ok", {"score": float, "details": dict}) or ("error", msg).
    """
    queue: mp.Queue = mp.Queue()
    return run_process_with_timeout(
        _run_openevolve_eval_in_process,
        (evaluator_dir, program_source, score_key, queue, extra_sys_paths),
        timeout,
        queue=queue,
        on_timeout=lambda _elapsed, _pid: ("error", f"Timeout ({timeout}s)"),
        on_no_result=lambda _exc, _pid: ("error", "No result from subprocess"),
    )


# ---------------------------------------------------------------------------
# Base adapter
# ---------------------------------------------------------------------------

class OpenEvolveBaseAdapter(BenchmarkAdapter):
    """Shared base for all OpenEvolve-style benchmark adapters.

    ``problem_names`` filters discovered problems by SUBSTRING over the
    ``data_dir``-relative directory name, not exact match (so parent-dir
    fragments select whole families); a pattern that matches more than one
    problem logs an over-match warning.
    """

    def __init__(
        self,
        data_dir: str,
        timeout: int = 120,
        problem_names: list[str] | None = None,
    ):
        self.data_dir = Path(data_dir)
        self.timeout = timeout
        self._problem_names = problem_names
        self._score_key = "combined_score"

    def score_scale(self) -> dict:
        """OpenEvolve scores are continuous and unbounded (AlgoTune speedups,
        SR fitness that may go negative); the evaluator emits a finite -1e9
        sentinel for a hard failure. Scale-aware selection / stopping must
        normalize by the observed range and clamp this sentinel rather than
        averaging it as a real value (roadmap v2 N4)."""
        return {"kind": "continuous", "lo": None, "hi": None, "failure_sentinel": -1e9}

    def split_type(self) -> str:
        # Deterministic evaluator; evaluate_test delegates to evaluate (dev==test).
        return "dev_equals_test"

    # --- Failure-ranking hooks (OE-1 / OE-2 / OE-3, round-2) ---
    #
    # These two hooks localize the symbolic_regression failure-ranking fix to
    # the OpenEvolve family WITHOUT touching the shared archive per-task-best
    # loop or any [0,1]/unit-scale benchmark. The base implementations preserve
    # the historical (AlphaEvolve / AlgoTune) behavior byte-for-byte; only a
    # genuinely negative-capable scale (symbolic_regression) overrides them.

    def _failure_score(self) -> float:
        """Score assigned to a hard FAILURE / crash in meta-n's ``score``
        channel — the value that becomes ``trace.score`` and is AVERAGED into
        ``candidate.mean_score`` and every reported mean. A failure is ``0.0``
        (the historical value) on every OpenEvolve scale — unit [0,1],
        AlphaEvolve geometric constructions, AlgoTune speedups, and (post
        Reaudit #9) symbolic_regression — so all of these stay byte-identical.

        This is deliberately NOT the ``score_scale()`` ``failure_sentinel``
        (``-1e9``): that sentinel documents the *evaluator's* raw hard-failure
        output and, per the ``score_scale`` contract, must be CLAMPED rather than
        averaged. Routing it into ``score`` (OE-3) poisoned every reported mean;
        see ``SymbolicRegressionAdapter._failure_score`` (Reaudit #9)."""
        return 0.0

    def _eval_success(self, raw_score: float, is_sentinel_failure: bool) -> bool:
        """Whether a non-crashed ("ok") eval counts as a SUCCESS (drives
        ``trace.success`` → pass@1 + the 3:1 failure-biased Ω trace sampling).
        Default: a positive score, matching every [0,1]/non-negative continuous
        scale where 0.0 means "no improvement / failed". A negative-capable
        scale (symbolic_regression) overrides this with ran-vs-crashed so a
        valid-but-poor negative fit is NOT mis-sampled as a failure (OE-2)."""
        return raw_score > 0.0

    # --- Shared helpers ---

    def _discover_problems(self) -> list[dict]:
        """Auto-discover problem directories containing evaluator.py + initial_program.py."""
        if not self.data_dir.exists():
            logger.warning("Data directory not found: %s", self.data_dir)
            return []

        problems = []
        for evaluator_path in sorted(self.data_dir.rglob("evaluator.py")):
            problem_dir = evaluator_path.parent
            if not (problem_dir / "initial_program.py").exists():
                continue
            rel_name = str(problem_dir.relative_to(self.data_dir))
            if rel_name == ".":
                continue

            # Filter by explicit problem names if provided
            if self._problem_names is not None:
                if not any(pn in rel_name for pn in self._problem_names):
                    continue

            initial_src = self._read_file(problem_dir / "initial_program.py")
            config = self._load_config(problem_dir)
            func_name = self._extract_function_name(initial_src)
            deps = self._read_requirements(problem_dir)

            problems.append({
                "problem_dir": str(problem_dir),
                "name": rel_name,
                "function_name": func_name,
                "initial_program": initial_src,
                "config": config,
                "deps": deps,
                "timeout": config.get("timeout", self.timeout),
            })

        # Over-match visibility: the SUBSTRING filter above means a pattern
        # (even an exact problem name, e.g. "circle_packing" alongside
        # "circle_packing_v2") can select several problems. Selection stays
        # as documented; the warning is log-only.
        if self._problem_names is not None:
            for pn in self._problem_names:
                matched = [p["name"] for p in problems if pn in p["name"]]
                if len(matched) > 1:
                    logger.warning(
                        "--bench-tasks pattern %r substring-matched %d "
                        "OpenEvolve problems (%s%s); OpenEvolve matching is "
                        "substring, not exact",
                        pn, len(matched), ", ".join(matched[:3]),
                        ", ..." if len(matched) > 3 else "",
                    )
        return problems

    @staticmethod
    def _read_file(path: Path) -> str:
        """Read file contents or return empty string."""
        try:
            return path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return ""

    @staticmethod
    def _extract_function_name(source: str) -> str:
        """Extract the primary function name from initial_program.py.

        Looks for the first ``def`` inside EVOLVE-BLOCK markers, or the first
        top-level ``def`` if no markers are present.
        """
        # Try to extract from within EVOLVE-BLOCK markers first
        block_match = re.search(
            r"#\s*EVOLVE-BLOCK-START\s*\n(.*?)#\s*EVOLVE-BLOCK-END",
            source, re.DOTALL,
        )
        search_text = block_match.group(1) if block_match else source

        # Find first function definition
        match = re.search(r"^def\s+(\w+)\s*\(", search_text, re.MULTILINE)
        return match.group(1) if match else "solve"

    @staticmethod
    def _load_config(problem_dir: Path) -> dict:
        """Load config.yaml and extract relevant fields."""
        config_path = problem_dir / "config.yaml"
        if not config_path.exists() or yaml is None:
            return {}
        try:
            with open(config_path, encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
            # Extract key fields
            config: dict[str, Any] = {}
            prompt = raw.get("prompt", {})
            if isinstance(prompt, dict):
                config["system_message"] = prompt.get("system_message", "")
            evaluator = raw.get("evaluator", {})
            if isinstance(evaluator, dict):
                config["timeout"] = evaluator.get("timeout", 120)
            return config
        except Exception:
            return {}

    @staticmethod
    def _read_requirements(problem_dir: Path) -> list[str]:
        """Read requirements.txt and extract package names."""
        req_path = problem_dir / "requirements.txt"
        if not req_path.exists():
            return ["numpy"]
        try:
            lines = req_path.read_text(encoding="utf-8").strip().splitlines()
            return [re.split(r"[><=!]", line.strip())[0] for line in lines if line.strip() and not line.startswith("#")]
        except OSError:
            return ["numpy"]

    def _build_task_description(self, problem: dict) -> str:
        """Build human-readable task description for the solver."""
        parts = []

        # Use config system_message as the primary description.
        # Some configs (e.g. symbolic_regression) use single-quoted YAML strings
        # where `\n` is stored literally rather than as a newline; un-escape so
        # the model sees real line breaks and code fences instead of `\\n`.
        sys_msg = problem.get("config", {}).get("system_message", "")
        if sys_msg:
            if "\\n" in sys_msg:
                sys_msg = sys_msg.replace("\\n", "\n").replace("\\t", "\t")
            parts.append(sys_msg.strip())

        # Fallback description if no system_message
        if not sys_msg:
            parts.insert(0, f"## Problem: {problem['name']}\n\n"
                         f"Implement the function `{problem['function_name']}` as specified below.")

        # Include full initial_program.py as reference — this shows the solver
        # the complete contract: class structure, entry-point functions (run,
        # run_search, run_solver), is_solution methods, and return types.
        initial_prog = problem.get("initial_program", "").strip()
        if initial_prog:
            parts.append(
                "\n## Reference Implementation\n"
                "Your module MUST follow this exact structure — same function "
                "names, same return types. Improve the algorithm but keep the "
                "interface identical.\n"
                f"```python\n{initial_prog}\n```"
            )

        # Dependencies
        deps = problem.get("deps", ["numpy"])
        if deps:
            parts.append(f"\n## Available Libraries\n{', '.join(deps)}")

        return "\n\n".join(parts)

    def _make_task(self, problem: dict) -> TaskDescription:
        """Create a TaskDescription from discovered problem info."""
        # Sanitize name for task_id
        task_id = problem["name"].replace("/", "_").replace(" ", "_").replace("-", "_").lower()

        return TaskDescription(
            task_id=task_id,
            description=self._build_task_description(problem),
            metadata={
                "benchmark": self.name,
                "problem_name": problem["name"],
                "problem_dir": problem["problem_dir"],
                "function_name": problem["function_name"],
                "solution_language": "openevolve",
                "timeout": problem.get("timeout", self.timeout),
                "deps": problem.get("deps", ["numpy"]),
                "score_key": self._score_key,
            },
        )

    # --- BenchmarkAdapter interface ---

    def _get_extra_sys_paths(self) -> list[str] | None:
        """Return extra sys.path entries for subprocess evaluation. Override in subclasses."""
        return None

    async def evaluate(self, task: TaskDescription, solution: str) -> EvalResult:
        """Evaluate solution by writing to temp file and calling evaluator in subprocess."""
        problem_dir = task.metadata.get("problem_dir", "")
        if not problem_dir:
            return EvalResult(success=False, feedback="No problem_dir in metadata")

        score_key = task.metadata.get("score_key", self._score_key)
        timeout = task.metadata.get("timeout", self.timeout)
        extra_paths = self._get_extra_sys_paths()

        loop = asyncio.get_running_loop()
        eval_start = time.time()
        try:
            status, value = await loop.run_in_executor(
                None,
                functools.partial(
                    _run_openevolve_with_timeout,
                    problem_dir, solution, score_key, timeout, extra_paths,
                ),
            )
        except Exception as e:
            logger.warning(
                "Eval error for %s: %s (%.1fs)",
                task.task_id, e, time.time() - eval_start,
            )
            return EvalResult(success=False, score=0.0, feedback=f"Evaluation error: {e}")

        if status == "ok":
            raw_score = float(value.get("score", 0.0))
            details = value.get("details", {})
            # A hard FAILURE the evaluator emitted as its -1e9 sentinel was
            # clamped to 0.0 in-subprocess (``_coerce_eval_score``); it is
            # flagged here by ``raw_failure_sentinel`` in details. Its ``score``
            # is the clamped reporting floor (``_failure_score``) so the sentinel
            # never enters an averaged/reported metric (Reaudit #9 / score_scale
            # contract); ``success`` still reflects ran-vs-crashed via
            # ``_eval_success`` (OE-2).
            is_sentinel_failure = "raw_failure_sentinel" in details
            feedback = json.dumps(details, indent=2) if details else f"score={raw_score:.6f}"
            logger.debug(
                "Eval: %s -> score=%.6f (%.1fs)",
                task.task_id, raw_score, time.time() - eval_start,
            )
            score = self._failure_score() if is_sentinel_failure else raw_score
            return EvalResult(
                success=self._eval_success(raw_score, is_sentinel_failure),
                score=score,
                raw_score=raw_score,
                feedback=feedback,
            )
        else:
            logger.warning(
                "Eval failed for %s: %s (%.1fs)",
                task.task_id, value, time.time() - eval_start,
            )
            # A crash (timeout / import error / no result) is a hard failure.
            # Its score is the clamped reporting floor (``_failure_score`` = 0.0
            # on every OpenEvolve scale) — never the -1e9 sentinel — so it does
            # not poison any averaged/reported mean (Reaudit #9). ``raw_score``
            # matches ``score`` here, consistent with the ok-sentinel branch
            # above (both channels carry the clamped floor for a hard failure).
            fail_score = self._failure_score()
            return EvalResult(
                success=False, score=fail_score, raw_score=fail_score,
                feedback=f"Evaluation error: {value}",
            )

    async def evaluate_test(self, task: TaskDescription, solution: str) -> EvalResult:
        """Test-time re-eval. OpenEvolve-family benchmarks (AlphaEvolve, SR,
        AlgoTune) have NO held-out test split — each per-problem evaluator is
        deterministic and runs against a fixed instance every time. See
        ``baselines/openevolve/src/task_alphaevolve_math.py:120`` which makes
        the same observation for the upstream baseline's ``test_score()``.
        ``evaluate()`` already runs the canonical evaluator via
        ``_run_openevolve_with_timeout`` (subprocess + ``spec_from_file_location``
        + temp .py file — same path OE/godel baselines use). So
        ``evaluate_test = evaluate`` for these benchmarks. The two-name
        distinction exists for symmetry with adapters that DO have separate
        test splits (``co_bench``, ``arc_agi``, ``text_classification``); the
        evolutionary orchestrator gates the per-task-oracle test pass on
        ``hasattr(adapter, "evaluate_test")``, so defining this method is
        what populates ``test_mean_score`` / ``chain_test_mean_score`` in the
        final summary.json. For these benchmarks the two equal
        ``oracle_mean_score`` / ``best_mean_score`` by construction.
        """
        return await self.evaluate(task, solution)


# ---------------------------------------------------------------------------
# AlphaEvolve Math
# ---------------------------------------------------------------------------

class AlphaEvolveMathAdapter(OpenEvolveBaseAdapter):
    """Adapter for AlphaEvolve mathematical construction problems.

    ~16 problems spanning geometry (kissing number, circle packing, Heilbronn),
    algebra (matrix multiplication), and analysis (autocorrelation inequalities).
    """

    def __init__(
        self,
        data_dir: str = "./data/openevolve/examples/alphaevolve_math_problems",
        timeout: int = 120,
        problem_names: list[str] | None = None,
    ):
        super().__init__(data_dir, timeout, problem_names)

    @property
    def name(self) -> str:
        return "alphaevolve_math"

    def load_tasks(self, limit: int | None = None) -> list[TaskDescription]:
        problems = self._discover_problems()
        if not problems:
            logger.warning("No AlphaEvolve math problems found in %s", self.data_dir)
            return []

        tasks = [self._make_task(p) for p in problems]
        logger.info("Loaded %d AlphaEvolve math tasks", len(tasks))
        return tasks[:limit] if limit is not None else tasks


# ---------------------------------------------------------------------------
# Symbolic Regression
# ---------------------------------------------------------------------------

class SymbolicRegressionAdapter(OpenEvolveBaseAdapter):
    """Adapter for symbolic regression (equation discovery from data).

    Problems are factory-generated by data_api.py across physics, chemistry,
    biology, and materials science domains.
    """

    def __init__(
        self,
        data_dir: str = "./data/openevolve/examples/symbolic_regression/problems",
        timeout: int = 90,
        problem_names: list[str] | None = None,
    ):
        super().__init__(data_dir, timeout, problem_names)

    @property
    def name(self) -> str:
        return "symbolic_regression"

    # symbolic_regression is the one OpenEvolve domain with a genuinely
    # NEGATIVE-CAPABLE score: ``combined_score = -log10(mse + 1e-9)`` is
    # negative for any fit with MSE > 1 (a valid-but-poor equation). The two
    # base hooks below localize the round-2 failure-ranking fix here so the
    # non-negative AlphaEvolve / AlgoTune scales (and every [0,1]/unit
    # benchmark) stay byte-identical.

    def _failure_score(self) -> float:
        """Score assigned to a hard failure/crash in meta-n's ``score`` channel.

        Returns the CLAMPED reporting floor (``0.0``), NOT the ``score_scale()``
        ``failure_sentinel`` (``-1e9``). Reaudit #9 — regression from OE-3:
        meta-n's ``score`` is the value that becomes ``trace.score`` and is then
        AVERAGED into every reported metric — ``candidate.mean_score`` /
        ``best_mean_score``, ``per_task_best_scores`` / ``oracle_mean_score``,
        ``test_mean_score`` / ``chain_test_mean_score``, ``convergence_history``
        and the STOP-rule delta. The ``score_scale`` contract
        (``benchmark.py`` — ``failure_sentinel`` "must NOT be averaged as a real
        value") is therefore violated the instant the sentinel enters ``score``:
        one SR task crashed by the archive-best / by every candidate dragged all
        of those means to ~ ``-1e9 / N`` (~ ``-1e8``), an ~8-order-of-magnitude
        corruption of the headline metric. The clamped floor here is the same
        value ``_coerce_eval_score`` already produces and matches the
        AlphaEvolve / AlgoTune failure floor, so a crash contributes a bounded
        ``0.0`` to every reported mean (its pre-OE-3 behavior).

        NOTE: OE-3 originally routed a failure to ``-1e9`` so that a *valid*
        negative fit out-ranks a ``0.0``-looking crash in the archive per-task
        ``max()`` (OE-1). Preserving that requires keeping the sentinel in a
        RANKING-ONLY channel while CLAMPING it in the averaging sinks — a
        ranking-vs-reporting split that lives at those sinks, not in this single
        ``score`` value that feeds both. Since the ``score`` channel cannot be
        both un-averaged (ranking) and averaged-as-0.0 (reporting) at once, the
        reporting contract wins here; ``_eval_success`` (OE-2) still distinguishes
        a valid negative fit (success) from a crash (failure) for pass@1 / Ω
        trace sampling, independent of this score."""
        return 0.0

    def _eval_success(self, raw_score: float, is_sentinel_failure: bool) -> bool:
        """Success = ran-vs-crashed, NOT score > 0: a valid-but-poor negative
        fit ran fine and must not be sampled as a failure by Ω (OE-2)."""
        return not is_sentinel_failure

    def load_tasks(self, limit: int | None = None) -> list[TaskDescription]:
        problems = self._discover_problems()
        if not problems:
            logger.warning(
                "No symbolic regression problems found in %s. "
                "Run data_api.py to generate them.",
                self.data_dir,
            )
            return []

        tasks = [self._make_task(p) for p in problems]
        logger.info("Loaded %d symbolic regression tasks", len(tasks))
        return tasks[:limit] if limit is not None else tasks


# ---------------------------------------------------------------------------
# AlgoTune
# ---------------------------------------------------------------------------

ALGOTUNE_TASKS = [
    "affine_transform_2d",
    "convolve2d_full_fill",
    "eigenvectors_complex",
    "fft_cmplx_scipy_fftpack",
    "fft_convolution",
    "lu_factorization",
    "polynomial_real",
    "psd_cone_projection",
]


class AlgoTuneAdapter(OpenEvolveBaseAdapter):
    """Adapter for AlgoTune algorithmic speedup optimization.

    8 tasks where the solver writes a faster implementation of a known
    algorithm. Scored on speedup ratio vs reference implementation.

    Requires heavy dependencies (torch, jax, cvxpy, etc.) — degrades
    gracefully if not installed.
    """

    def __init__(
        self,
        data_dir: str = "./data/openevolve/examples/algotune",
        timeout: int = 200,
        task_names: list[str] | None = None,
        algotune_repo: str | None = None,
    ):
        # AlgoTune uses a different score key
        super().__init__(data_dir, timeout, task_names)
        self._score_key = "speedup_score"
        self.algotune_repo = algotune_repo or self._find_algotune_repo()
        self._available = self._check_dependencies()

    @property
    def name(self) -> str:
        return "algotune"

    @staticmethod
    def _find_algotune_repo() -> str | None:
        """Auto-detect AlgoTune repo location."""
        candidates = [
            Path("data/AlgoTune"),
            Path.home() / "github" / "AlgoTune",
        ]
        for p in candidates:
            if p.exists() and (p / "AlgoTuneTasks").exists():
                logger.info("Auto-detected AlgoTune repo at %s", p)
                return str(p.resolve())
        return None

    @staticmethod
    def _check_dependencies() -> bool:
        """Check if minimal AlgoTune dependencies are available."""
        try:
            import numpy  # noqa: F401
            import scipy  # noqa: F401
            return True
        except ImportError:
            logger.warning(
                "AlgoTune dependencies not installed (need at least numpy, scipy). "
                "Adapter will return empty task list."
            )
            return False

    def _get_extra_sys_paths(self) -> list[str] | None:
        """Provide AlgoTune repo path for subprocess sys.path."""
        if self.algotune_repo:
            return [self.algotune_repo]
        return None

    def load_tasks(self, limit: int | None = None) -> list[TaskDescription]:
        if not self._available:
            return []

        problems = self._discover_problems()
        if not problems:
            logger.warning("No AlgoTune tasks found in %s", self.data_dir)
            return []

        tasks = [self._make_task(p) for p in problems]
        logger.info("Loaded %d AlgoTune tasks", len(tasks))
        return tasks[:limit] if limit is not None else tasks


# ---------------------------------------------------------------------------
# Shared executor
# ---------------------------------------------------------------------------

class OpenEvolveExecutor(AdapterExecutor):
    """Executor for all OpenEvolve benchmark domains (also reused for ARC-AGI-2).

    Wraps adapter.evaluate() into a Trace object via the shared
    :class:`AdapterExecutor`, customizing the score precision (``.6f``,
    matching the evaluators' 6-decimal output) and synthesizing a structured
    ``error_summary`` from the JSON feedback the OpenEvolve evaluators emit.
    """

    _score_fmt = ".6f"

    def _error_summary(self, result: EvalResult) -> str:
        """Extract structured error message from JSON feedback if available."""
        if result.success:
            return ""
        error_summary = ""
        try:
            data = json.loads(result.feedback)
            if isinstance(data, dict):
                # Try explicit error key first
                error_summary = str(data.get("error", ""))[:500]
                # Synthesize from scores if no explicit error
                if not error_summary:
                    parts = []
                    if data.get("correctness_score", 1) == 0:
                        valid = data.get("baseline_comparison", {}).get("num_valid_solutions", "?")
                        total = data.get("baseline_comparison", {}).get("num_total_trials", "?")
                        parts.append(f"correctness=0 ({valid}/{total} valid)")
                    if data.get("combined_score", 1) == 0:
                        parts.append("combined_score=0")
                    error_summary = "; ".join(parts)[:500] if parts else ""
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
        if not error_summary:
            error_summary = result.feedback[:200]
        return error_summary
