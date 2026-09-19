"""Capture the Stage-0b/2/3 deterministic orchestration + Omega-prompt goldens.

These goldens are the "all new flags OFF ⇒ byte-identical" reference for the
within-layer-refine (Stage 2) and re-propagation (Stage 3) work. They are
captured on HEAD (before any feature edit) using the fake-solver harness from
``tests/test_evolutionary_orchestrator.py`` — no LLM, no Docker, fully seeded.

Two artifacts are written under ``tests/golden/``:

  * ``orchestration_golden.json`` — the archive ``index.json`` plus every
    per-candidate ``summary.json`` from a seeded multi-generation micro-run,
    with volatile fields (``created_at``, ``elapsed_s``, wall timings) NORMALIZED
    so the diff is over orchestration STRUCTURE, not wall-clock noise.
  * ``omega_prompt_golden.txt`` — a single rendered ``OmegaEngine._build_prompt``
    output over deterministic Trace/InjectedCode/TaskDescription inputs, with the
    (future) downstream-feedback param ABSENT. The Stage-3 signature extension
    must keep this byte-identical when the new param is empty/absent.

Run: ``python tests/golden/capture_stage23_golden.py``  (writes/refreshes goldens)
Verify later (flags OFF): re-run capture into a temp dir and assert equality.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from meta_n.core.evolutionary_orchestrator import (
    EvolutionaryConfig,
    EvolutionaryOrchestrator,
)
from meta_n.core.meta_layer import InjectedCode, TaskDescription, Trace
from meta_n.core.omega import OmegaEngine

GOLDEN_DIR = Path(__file__).resolve().parent

# Volatile keys normalized out of the orchestration golden (wall-clock / pid-ish).
_VOLATILE = {"created_at", "elapsed_s"}


def _normalize(obj):
    """Recursively replace volatile values with a stable sentinel."""
    if isinstance(obj, dict):
        return {
            k: ("<normalized>" if k in _VOLATILE else _normalize(v))
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_normalize(v) for v in obj]
    return obj


def _deterministic_injection() -> InjectedCode:
    # A trivially-valid, side-effect-free pre_process so MetaLayer.execute runs a
    # real (validated) pre_process block deterministically.
    return InjectedCode(
        pre_process="additional_context = 'golden-hint'",
        rationale="golden deterministic injection",
        source_depth=2,
    )


def _stub_trace(task_id: str) -> Trace:
    return Trace(
        task_id=task_id,
        depth=1,
        script="echo golden",
        stdout="ok",
        success=True,
        score=0.5,
        reasoning="stub",
        exit_code=0,
    )


async def _run_orchestration_golden(out_dir: Path) -> dict:
    config = EvolutionaryConfig(
        output_dir=str(out_dir),
        max_depth=3,
        max_iterations=3,
        patience=3,
        gate_tasks=1,
        beam_width=1,
        beam_candidates=1,
        parallel=1,
        seed=42,
        # All NEW flags explicitly OFF (parity reference).
        # (within_layer_refine / repropagation do not exist yet on HEAD.)
    )
    orch = EvolutionaryOrchestrator(
        llm_client=MagicMock(),
        executor=MagicMock(),
        omega=MagicMock(),
        config=config,
        solver_language="bash",
    )
    orch.solver.solve = AsyncMock(return_value=("echo golden", "stub", 10))
    orch.executor.execute = AsyncMock(
        side_effect=lambda script, task: _stub_trace(task.task_id)
    )
    orch.omega.generate = AsyncMock(
        return_value=(_deterministic_injection(), 7)
    )

    tasks = [
        TaskDescription(task_id="task_a", description="Solve task_a"),
        TaskDescription(task_id="task_b", description="Solve task_b"),
    ]
    await orch.run(tasks)

    # Collect archive index + every per-candidate summary.json.
    archive_dir = out_dir / "archive"
    index = json.loads((archive_dir / "index.json").read_text())
    candidates = {}
    for cand_dir in sorted(p for p in archive_dir.iterdir() if p.is_dir()):
        summ = cand_dir / "summary.json"
        if summ.exists():
            candidates[cand_dir.name] = json.loads(summ.read_text())
    return {
        "archive_index": _normalize(index),
        "candidate_summaries": _normalize(candidates),
    }


def _omega_prompt_golden() -> str:
    engine = OmegaEngine(llm_client=MagicMock())
    traces = [
        Trace(
            task_id="task_a", depth=2, script="echo a", success=False, score=0.0,
            error_summary="ZeroDivisionError: division by zero",
            stderr="ZeroDivisionError", exit_code=1, reasoning="r-a",
        ),
        Trace(
            task_id="task_b", depth=2, script="echo b", success=True, score=1.0,
            reasoning="r-b", exit_code=0,
        ),
    ]
    context_stack = [
        InjectedCode(
            pre_process="additional_context = 'prior'",
            rationale="prior layer", source_depth=2,
        ),
    ]
    tasks = [
        TaskDescription(task_id="task_a", description="Solve task_a"),
        TaskDescription(task_id="task_b", description="Solve task_b"),
    ]
    # NOTE: the (future) downstream-feedback param is ABSENT here by construction.
    return engine._build_prompt(
        traces, context_stack, tasks, depth=3,
        previous_scores={"task_a": 0.4, "task_b": 0.9},
        archive_best_scores={"task_a": 0.6, "task_b": 1.0},
        solver_language="bash",
    )


def main() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        orchestration = asyncio.run(_run_orchestration_golden(Path(td)))
    (GOLDEN_DIR / "orchestration_golden.json").write_text(
        json.dumps(orchestration, indent=2, sort_keys=True)
    )
    (GOLDEN_DIR / "omega_prompt_golden.txt").write_text(_omega_prompt_golden())
    print("wrote orchestration_golden.json + omega_prompt_golden.txt")
    print("candidates:", sorted(orchestration["candidate_summaries"]))


if __name__ == "__main__":
    main()
