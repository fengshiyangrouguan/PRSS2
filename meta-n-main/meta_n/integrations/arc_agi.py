"""ARC-AGI-2 benchmark adapter.

Loads tasks from the official ``arcprize/ARC-AGI-2`` GitHub repo (Apache-2.0).
Each task is a JSON file under ``data/training/`` (1000 tasks) or
``data/evaluation/`` (120 tasks) with the shape::

    {"train": [{"input": [[...]], "output": [[...]]}, ...],
     "test":  [{"input": [[...]], "output": [[...]]}, ...]}

The solver writes a Python module exposing two functions::

    transform_grid_attempt_1(grid: np.ndarray) -> np.ndarray
    transform_grid_attempt_2(grid: np.ndarray) -> np.ndarray

Both receive a 2D int array (values 0-9) and must return the same. Pass@2:
the task is solved if EITHER attempt matches the expected output exactly.

This adapter intentionally does NOT subclass ``OpenEvolveBaseAdapter`` —
that base assumes per-problem ``evaluator.py``/``initial_program.py`` files,
which the GitHub ARC-AGI-2 layout does not provide. We score grids
directly via numpy comparison instead.

**Train/test split.** ``evaluate()`` scores on TRAIN pairs (these are the
demonstration examples shown to the LLM in the prompt — the standard
in-the-loop signal during evolution). ``evaluate_test()`` scores on the
held-out TEST pairs (never shown to the LLM) and is reported once at
end-of-run for reference only. Train scoring is leaky by design (the LLM
has seen the answers), so a large train↔test gap signals overfitting.
Gold TEST output grids are held adapter-side (``self._test_pairs``) and
never enter ``task.metadata`` — Ω-injected pre_process has runtime read
access to metadata and its additional_context lands in the solver prompt
(the F127 invariant, see text_classification).

The pass@2 scoring functions are lifted (with attribution) from
``data/openevolve/examples/arc_benchmark/evaluator.py``.
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
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from meta_n.core.meta_layer import TaskDescription

# Compat re-export: _kill_process_tree was historically importable from here.
from meta_n.integrations._subprocess_utils import (  # noqa: F401
    _kill_process_tree,
    detach_process_group,
    run_process_with_timeout,
)
from meta_n.integrations.benchmark import BenchmarkAdapter, EvalResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pass@2 scoring (lifted from openevolve arc_benchmark/evaluator.py)
# ---------------------------------------------------------------------------

def pass_at_2_accuracy_single(
    attempts: list[np.ndarray],
    gt: np.ndarray,
) -> tuple[int, dict[int, Any]]:
    """Compute pass@2 for a single ARC test case.

    Returns (1 if any attempt is exactly correct else 0, per-attempt diagnostics).
    """
    assert len(attempts) == 2, "Expected exactly 2 attempts for pass@2"

    diagnostics: dict[int, Any] = {}
    passed = False

    for i, pred in enumerate(attempts):
        info: dict[str, Any] = {}
        if pred is None:
            info.update(
                size_match=False, pred_shape=None, gt_shape=tuple(gt.shape),
                incorrect_indices=None, perfect_match=False,
            )
            diagnostics[i] = info
            continue

        if pred.shape != gt.shape:
            info["size_match"] = False
            info["pred_shape"] = tuple(pred.shape)
            info["gt_shape"] = tuple(gt.shape)
            info["incorrect_indices"] = None
            attempt_passed = False
        else:
            info["size_match"] = True
            mask = pred != gt
            info["incorrect_indices"] = np.argwhere(mask).tolist()
            info["num_incorrect"] = int(mask.sum())
            attempt_passed = bool(mask.sum() == 0)

        info["perfect_match"] = attempt_passed
        diagnostics[i] = info
        passed = passed or attempt_passed

    return (1 if passed else 0), diagnostics


def pass_at_2_accuracy_multi_test(
    all_attempts: list[list[np.ndarray]],
    all_gt: list[np.ndarray],
) -> tuple[list[int], list[dict[int, Any]]]:
    """Apply pass_at_2_accuracy_single across multiple test cases."""
    assert len(all_attempts) == len(all_gt), "Mismatched test/gt counts"
    passes: list[int] = []
    diags: list[dict[int, Any]] = []
    for attempts, gt in zip(all_attempts, all_gt):
        p, d = pass_at_2_accuracy_single(attempts, gt)
        passes.append(p)
        diags.append(d)
    return passes, diags


def extract_failure_artifacts(diagnostics: dict[str, Any]) -> dict[str, Any]:
    """Convert one attempt's diagnostics into a brief feedback artifact."""
    if not diagnostics.get("size_match", False):
        return {
            "error_type": "SizeMismatch",
            "error_message": (
                f"Output shape {diagnostics.get('pred_shape')} != "
                f"expected {diagnostics.get('gt_shape')}"
            ),
            "suggestion": "Review your output size determination.",
        }
    return {
        "error_type": "IncorrectCells",
        "error_message": (
            f"{diagnostics.get('num_incorrect', '?')} incorrect cells"
        ),
        "suggestion": "Review per-cell logic; some cells differ from the target.",
    }


# ---------------------------------------------------------------------------
# Subprocess execution of solver-generated transform_grid_attempt_*
# ---------------------------------------------------------------------------

def _run_arc_attempts_in_process(
    solution_source: str,
    inputs_serialized: list[list[list[int]]],
    queue: mp.Queue,
) -> None:
    """Subprocess target.

    Imports the solver-written module, calls ``transform_grid_attempt_1`` and
    ``transform_grid_attempt_2`` on each input grid, and queues results
    serialized as nested int lists. Robust to per-input exceptions (the
    failed input gets a ``None`` placeholder; sibling inputs still score).
    """
    detach_process_group()

    try:
        fd, program_path = tempfile.mkstemp(suffix=".py", prefix="_meta_n_arc_")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(solution_source)

            spec = importlib.util.spec_from_file_location(
                "_arc_solution", program_path,
            )
            if spec is None or spec.loader is None:
                queue.put(("error", "Could not create module spec"))
                return
            mod = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(mod)
            except Exception as e:
                queue.put(("error", f"Module import failed: {type(e).__name__}: {e}"))
                return

            missing = [
                fn for fn in ("transform_grid_attempt_1", "transform_grid_attempt_2")
                if not hasattr(mod, fn)
            ]
            if missing:
                queue.put((
                    "error",
                    f"Module is missing required function(s): {', '.join(missing)}. "
                    f"Define both transform_grid_attempt_1(grid) and "
                    f"transform_grid_attempt_2(grid).",
                ))
                return

            results: list[list[Any]] = []
            for raw in inputs_serialized:
                grid_in = np.asarray(raw, dtype=np.int32)
                pair: list[Any] = []
                for fn_name in ("transform_grid_attempt_1", "transform_grid_attempt_2"):
                    fn = getattr(mod, fn_name)
                    # The whole attempt — call AND output conversion — must be
                    # protected: a malformed return (e.g. a 2D object/str array
                    # that makes np.isfinite raise TypeError) should null only
                    # THIS attempt, leaving sibling pairs/attempts to score
                    # (the documented per-input robustness contract).
                    try:
                        out = fn(grid_in)
                        arr = np.asarray(out)
                        if arr.ndim != 2:
                            pair.append(None)
                            continue
                        # Guard against float NaN/inf: numpy silently casts
                        # these to undefined ints (impl-defined; on x86
                        # typically a large negative). That would score as a
                        # "valid" wrong cell rather than a failed attempt.
                        if not np.issubdtype(arr.dtype, np.integer):
                            if not np.isfinite(arr).all():
                                pair.append(None)
                                continue
                        pair.append(arr.astype(int).tolist())
                    except Exception as e:
                        pair.append(None)
                        logger.debug("attempt %s raised: %s", fn_name, e)
                        continue
                results.append(pair)

            queue.put(("ok", results))
        finally:
            try:
                os.unlink(program_path)
            except OSError:
                pass
    except Exception as e:
        queue.put(("error", f"{type(e).__name__}: {e}"))


def _run_arc_attempts_with_timeout(
    solution_source: str,
    inputs: list[list[list[int]]],
    timeout: int,
) -> tuple[str, Any]:
    """Run a candidate's two attempts on a list of inputs, with timeout.

    Returns ("ok", [[a1, a2], ...]) where each attempt is a nested int list
    or ``None``, OR ("error", message).
    """
    queue: mp.Queue = mp.Queue()
    return run_process_with_timeout(
        _run_arc_attempts_in_process,
        (solution_source, inputs, queue),
        timeout,
        queue=queue,
        on_timeout=lambda _elapsed, _pid: ("error", f"Timeout ({timeout}s)"),
        on_no_result=lambda _exc, _pid: ("error", "No result from subprocess"),
    )


# ---------------------------------------------------------------------------
# Description rendering — train pairs only, never test
# ---------------------------------------------------------------------------

def _grid_to_ascii(grid: list[list[int]]) -> str:
    """Render a grid as one row per line, space-separated digits."""
    return "\n".join(" ".join(str(v) for v in row) for row in grid)


def _build_description(task_dict: dict[str, Any]) -> str:
    """Build the solver-facing prompt. Shows TRAIN pairs only."""
    train = task_dict.get("train", [])
    parts = [
        "## ARC-AGI-2 puzzle",
        "",
        "Each puzzle shows a small set of demonstration pairs. They share a "
        "common (but unstated) transformation rule that maps each input grid "
        "to its output grid. Infer the rule and implement it.",
        "",
        f"## Demonstration pairs ({len(train)} total)",
    ]

    for i, pair in enumerate(train):
        inp = pair["input"]
        out = pair["output"]
        h_in, w_in = len(inp), (len(inp[0]) if inp else 0)
        h_out, w_out = len(out), (len(out[0]) if out else 0)
        parts.extend([
            "",
            f"### Pair {i + 1}",
            f"Input ({h_in}x{w_in}):",
            "```",
            _grid_to_ascii(inp),
            "```",
            f"Output ({h_out}x{w_out}):",
            "```",
            _grid_to_ascii(out),
            "```",
        ])

    parts.extend([
        "",
        "## Your task",
        "Implement TWO Python functions in a single module:",
        "",
        "- `transform_grid_attempt_1(grid: np.ndarray) -> np.ndarray`",
        "- `transform_grid_attempt_2(grid: np.ndarray) -> np.ndarray`",
        "",
        "Each receives a 2D numpy int array (values 0-9) and must return a 2D "
        "numpy int array (values 0-9). The two attempts should embody "
        "DIFFERENT hypotheses about the transformation rule — pass@2 scoring "
        "means the puzzle is solved if EITHER attempt is exactly correct.",
        "",
        "Use numpy and the standard library. No `if __name__ == \"__main__\"` "
        "blocks; the evaluator imports your module and calls the functions "
        "directly.",
    ])
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class ARCAGI2Adapter(BenchmarkAdapter):
    """Adapter for the ARC-AGI-2 benchmark.

    Loads tasks from the official arcprize/ARC-AGI-2 GitHub layout
    (``<data_dir>/data/<split>/<task_id>.json``) and scores solver-written
    ``transform_grid_attempt_*`` modules via pass@2.

    ``task_ids`` filters by SUBSTRING over the file stem, not exact match; a
    pattern that matches more than one task logs an over-match warning.
    """

    def __init__(
        self,
        data_dir: str = "./data/arc_agi_2",
        split: str = "evaluation",
        task_ids: list[str] | None = None,
        timeout: int = 60,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.split = split
        self.split_dir = self.data_dir / "data" / split
        self.task_ids = task_ids
        self.timeout = timeout
        # Gold TEST pairs (input + output), keyed by task_id. Kept OUT of
        # task.metadata: Ω-injected pre_process can read metadata, so gold
        # test outputs there would leak into the solver prompt.
        self._test_pairs: dict[str, list[dict[str, Any]]] = {}

    @property
    def name(self) -> str:
        return "arc_agi_2"

    def split_type(self) -> str:
        # Dev = the few train-demo pairs, a PROXY for the hidden test grid;
        # high dev (train-demo-perfect) routinely fails test. Overfit-protection
        # here must use a held-out-stable signal, never raw dev.
        #
        # NOTE (R2-ARC-1): this signal is NOT YET CONSUMED by archive selection
        # — ``per_task_best_traces`` still ranks on the raw dev ``evaluate()``
        # score, so split-aware overfit protection is absent (deferred feature).
        # The only current reader is the orchestrator's run-start one-time
        # warning (EvolutionaryOrchestrator.run), which surfaces the disabled
        # protection. The reported test number stays honest (``evaluate_test``).
        return "proxy"

    # --- Loading ---

    def load_tasks(self, limit: int | None = None) -> list[TaskDescription]:
        if not self.split_dir.exists():
            logger.warning("ARC-AGI-2 split dir not found: %s", self.split_dir)
            return []

        files = sorted(self.split_dir.glob("*.json"))
        # Per-pattern stems matched by the SUBSTRING filter, counted over the
        # full pre-``limit`` file loop so the over-match warning reflects the
        # true filter semantics (never the truncated set).
        pattern_matches: dict[str, list[str]] = {}
        tasks: list[TaskDescription] = []
        for path in files:
            stem = path.stem
            if self.task_ids is not None:
                matched = [pattern for pattern in self.task_ids if pattern in stem]
                for pattern in matched:
                    pattern_matches.setdefault(pattern, []).append(stem)
                if not matched:
                    continue
            try:
                with open(path, encoding="utf-8") as f:
                    task_dict = json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                logger.warning("Skipping %s (load error: %s)", path, e)
                continue
            test_pairs = task_dict.get("test", [])
            self._test_pairs[stem] = test_pairs
            public_dict = dict(task_dict)
            public_dict["test"] = [
                {"input": pair["input"]} for pair in test_pairs
            ]
            tasks.append(TaskDescription(
                task_id=stem,
                description=_build_description(task_dict),
                metadata={
                    "benchmark": "arc_agi_2",
                    "task_dict": public_dict,
                    "timeout": self.timeout,
                    "solution_language": "openevolve",
                },
            ))

        for pattern, stems in pattern_matches.items():
            if len(stems) > 1:
                logger.warning(
                    "--bench-tasks pattern %r substring-matched %d ARC tasks "
                    "(%s%s); ARC matching is substring, not exact",
                    pattern, len(stems), ", ".join(stems[:3]),
                    ", ..." if len(stems) > 3 else "",
                )

        logger.info(
            "Loaded %d ARC-AGI-2 tasks from %s (split=%s)",
            len(tasks), self.split_dir, self.split,
        )
        return tasks[:limit] if limit is not None else tasks

    # --- Evaluation ---

    async def evaluate(self, task: TaskDescription, solution: str) -> EvalResult:
        """Score against TRAIN pairs (drives evolution)."""
        return await self._score_split(task, solution, "train")

    async def evaluate_test(self, task: TaskDescription, solution: str) -> EvalResult:
        """Score against TEST pairs (held-out, reference only)."""
        return await self._score_split(task, solution, "test")

    async def _score_split(
        self, task: TaskDescription, solution: str, split: str,
    ) -> EvalResult:
        if split == "test":
            # Gold test outputs live only adapter-side (never in metadata).
            pairs = self._test_pairs.get(task.task_id, [])
        else:
            task_dict = task.metadata.get("task_dict", {})
            pairs = task_dict.get(split, [])
        if not pairs:
            return EvalResult(
                success=False, score=0.0, raw_score=0.0,
                feedback=json.dumps({
                    "error": f"no {split} pairs",
                    "combined_score": 0.0,
                }),
            )

        timeout = int(task.metadata.get("timeout", self.timeout))
        inputs = [pair["input"] for pair in pairs]
        gts = [np.asarray(pair["output"], dtype=np.int32) for pair in pairs]

        loop = asyncio.get_running_loop()
        eval_start = time.time()
        try:
            status, value = await loop.run_in_executor(
                None,
                functools.partial(
                    _run_arc_attempts_with_timeout, solution, inputs, timeout,
                ),
            )
        except Exception as e:
            return EvalResult(
                success=False, score=0.0, raw_score=0.0,
                feedback=json.dumps({
                    "error": f"runner exception: {e}",
                    "combined_score": 0.0,
                }),
            )

        if status == "error":
            logger.warning(
                "ARC eval error task=%s split=%s: %s (%.1fs)",
                task.task_id, split, value, time.time() - eval_start,
            )
            return EvalResult(
                success=False, score=0.0, raw_score=0.0,
                feedback=json.dumps({"error": str(value), "combined_score": 0.0}),
            )

        # status == "ok"; value is list of [a1, a2] per input (entries may be None)
        all_attempts: list[list[np.ndarray]] = []
        for pair in value:
            converted: list[np.ndarray] = []
            for raw in pair:
                if raw is None:
                    converted.append(None)  # type: ignore[arg-type]
                else:
                    converted.append(np.asarray(raw, dtype=np.int32))
            all_attempts.append(converted)

        passes, diags = pass_at_2_accuracy_multi_test(all_attempts, gts)
        score = float(sum(passes)) / len(passes) if passes else 0.0
        if not math.isfinite(score):
            score = 0.0

        per_pair: list[dict[str, Any]] = []
        for i, (p, d) in enumerate(zip(passes, diags)):
            entry: dict[str, Any] = {"index": i, "pass_at_2": int(p)}
            if not p:
                # Add a brief failure artifact for the first failing attempt.
                first_attempt = d.get(0, {})
                entry["diagnostic"] = extract_failure_artifacts(first_attempt)
            per_pair.append(entry)

        feedback = json.dumps({
            "split": split,
            "score": score,
            "n_pairs": len(passes),
            "n_passed": int(sum(passes)),
            "per_pair": per_pair,
            "combined_score": score,
        })
        logger.debug(
            "ARC eval ok task=%s split=%s score=%.3f (%d/%d, %.1fs)",
            task.task_id, split, score, sum(passes), len(passes),
            time.time() - eval_start,
        )
        return EvalResult(
            success=score > 0.0,
            score=score,
            raw_score=score,
            feedback=feedback,
        )
