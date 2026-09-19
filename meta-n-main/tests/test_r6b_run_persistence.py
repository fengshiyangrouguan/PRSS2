"""Refine §6b — F038: the dual mid-run/final summary.json schema is a
load-bearing PROTOCOL, pinned here (documented on
RunPersistence.save_running_summary).

The DISCRIMINATOR is key presence: consumers (scripts/azure_full_*.sh,
run_tb2_single_shot_all.sh) treat a summary.json WITHOUT 'total_iterations' as
"run not completed". A crash or a hard BudgetExceededError kill leaves the
MID-RUN shape behind (main.py deliberately skips save_results — F029). So:
  * the mid-run shape must NEVER gain 'total_iterations' / 'run_status';
  * the final shape must never lose them (also frozen by
    tests/test_persistence_compat.py::FROZEN_SUMMARY_KEYS).

All tests are LLM-free / offline.
"""

import json
import time
from pathlib import Path
from unittest.mock import MagicMock

from meta_n.core.evolutionary_orchestrator import (
    EvolutionaryConfig,
    EvolutionaryOrchestrator,
    EvolutionaryResult,
)

#: The frozen MID-RUN summary.json shape (RunPersistence.save_running_summary).
#: As load-bearing as the FINAL shape — see the F038 contract docstring.
FROZEN_MIDRUN_SUMMARY_KEYS = {
    "iteration", "archive_size", "best_mean_score", "oracle_mean_score",
    "best_candidate_id", "per_task_best_scores", "total_tokens", "elapsed_s",
}


def _make_orch(tmp_path) -> EvolutionaryOrchestrator:
    return EvolutionaryOrchestrator(
        llm_client=MagicMock(),
        executor=MagicMock(),
        omega=MagicMock(),
        config=EvolutionaryConfig(output_dir=str(tmp_path)),
        solver_language="bash",
    )


def _write_midrun(tmp_path) -> dict:
    orch = _make_orch(tmp_path)
    orch._persistence.save_running_summary(
        Path(tmp_path), iteration=1, result=EvolutionaryResult(),
        run_start=time.time(),
    )
    return json.loads((tmp_path / "summary.json").read_text())


def test_midrun_summary_shape_frozen(tmp_path):
    payload = _write_midrun(tmp_path)
    assert set(payload.keys()) == FROZEN_MIDRUN_SUMMARY_KEYS


def test_midrun_summary_lacks_completion_discriminators(tmp_path):
    payload = _write_midrun(tmp_path)
    assert "total_iterations" not in payload
    assert "run_status" not in payload


def test_final_summary_carries_completion_discriminators():
    # Already implied by FROZEN_SUMMARY_KEYS; restated here so THIS file
    # documents both halves of the presence-based protocol.
    final = EvolutionaryResult().to_dict()
    assert "total_iterations" in final
    assert "run_status" in final


def test_final_overwrites_midrun_shape(tmp_path):
    # The overwrite direction of the protocol: save_running_summary then
    # save_results on the same dir leaves the FINAL shape on disk.
    orch = _make_orch(tmp_path)
    orch._persistence.save_running_summary(
        Path(tmp_path), iteration=1, result=EvolutionaryResult(),
        run_start=time.time(),
    )
    assert "total_iterations" not in json.loads(
        (tmp_path / "summary.json").read_text()
    )
    orch.save_results(EvolutionaryResult())
    payload = json.loads((tmp_path / "summary.json").read_text())
    assert "total_iterations" in payload
    assert "run_status" in payload
