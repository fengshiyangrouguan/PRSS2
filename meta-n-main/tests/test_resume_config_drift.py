"""Resume config-drift provenance (Y4-Y_mechanism-5) — persistence side.

Contract under test (RunPersistence):
  * ``save_checkpoint`` stamps ``run_config_snapshot`` (the orchestrator's
    stashed ``_run_config_snapshot`` minus the volatile ``timestamp``) into
    checkpoint.json — additive key, absent when no run_config was passed.
  * ``write_run_config`` appends ``resume_config_drift`` (the orchestrator's
    stashed ``_resume_config_drift``) to config.json in BOTH stages — absent
    on fresh runs / drift-free resumes so those files stay byte-identical,
    last-drift-wins on repeated drifted resumes.

The resume-branch COMPARE (checkpoint snapshot vs the resumed run_config →
yellow warning + drift stash + re-issued START dump) lives in
``EvolutionaryOrchestrator.run()`` and is exercised end-to-end in section (e):
the re-issued START dump makes the drift record durable in config.json BEFORE
any iteration work, so a drifted resume that crashes or is budget-killed
(save_results never runs) still leaves the machine-readable record; drift-free
resumes never rewrite config.json differently.
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from meta_n.core.evolutionary_orchestrator import (
    EvolutionaryConfig,
    EvolutionaryOrchestrator,
)
from meta_n.core.meta_layer import InjectedCode, TaskDescription, Trace


def _orch(output_dir, **cfg):
    defaults = dict(
        max_depth=3, parallel=1, beam_width=1, beam_candidates=1,
        gate_tasks=0, output_dir=str(output_dir),
    )
    defaults.update(cfg)
    return EvolutionaryOrchestrator(
        llm_client=MagicMock(), executor=MagicMock(), omega=MagicMock(),
        config=EvolutionaryConfig(**defaults), solver_language="bash",
    )


# --------------------------------------------------------------------------- #
# (a) checkpoint stamps run_config_snapshot
# --------------------------------------------------------------------------- #

def test_checkpoint_stamps_run_config_snapshot_without_timestamp(tmp_path):
    orch = _orch(tmp_path)
    orch._run_config_snapshot = {
        "model": "m",
        "consolidate": False,
        "benchmark_config": "configs/benchmark_features.yaml",
        "timestamp": "2026-07-04T00:00:00",
    }
    orch._save_checkpoint(tmp_path, 1, 0, 0.5, 100, [0.5])
    ck = json.loads((tmp_path / "checkpoint.json").read_text())
    assert ck["run_config_snapshot"] == {
        "model": "m",
        "consolidate": False,
        "benchmark_config": "configs/benchmark_features.yaml",
    }
    assert "timestamp" not in ck["run_config_snapshot"]


def test_checkpoint_omits_snapshot_when_no_run_config(tmp_path):
    # Direct callers (unit tests) never pass run_config → no stash → no key,
    # so pre-existing checkpoint consumers see a byte-identical file.
    orch = _orch(tmp_path)
    orch._save_checkpoint(tmp_path, 1, 0, 0.5, 100, [0.5])
    ck = json.loads((tmp_path / "checkpoint.json").read_text())
    assert "run_config_snapshot" not in ck


def test_checkpoint_omits_snapshot_when_run_config_is_none(tmp_path):
    # run(resume=..., run_config=None) stashes None → guarded, key absent.
    orch = _orch(tmp_path)
    orch._run_config_snapshot = None
    orch._save_checkpoint(tmp_path, 1, 0, 0.5, 100, [0.5])
    ck = json.loads((tmp_path / "checkpoint.json").read_text())
    assert "run_config_snapshot" not in ck


# --------------------------------------------------------------------------- #
# (b, persistence half) config.json carries resume_config_drift when stashed
# --------------------------------------------------------------------------- #

def test_write_run_config_carries_drift_in_both_stages(tmp_path):
    drift = {"consolidate": {"checkpoint": False, "resumed": True}}
    run_config = {"model": "m", "consolidate": True}

    start_dir = tmp_path / "start"
    start_dir.mkdir()
    orch = _orch(start_dir)
    orch._resume_config_drift = drift
    orch._write_run_config(start_dir, run_config, stage="start")
    out = json.loads((start_dir / "config.json").read_text())
    assert out["resume_config_drift"] == drift
    assert out["consolidate"] is True  # top-level keeps the RESUMED value

    end_dir = tmp_path / "end"
    end_dir.mkdir()
    orch = _orch(end_dir)
    orch._resume_config_drift = drift
    orch._write_run_config(end_dir, run_config, stage="end")
    out = json.loads((end_dir / "config.json").read_text())
    assert out["resume_config_drift"] == drift
    assert out["consolidate"] is True


def test_end_dump_key_order_preserved_with_drift_appended_last(tmp_path):
    # FROZEN key-order contract: the run_config keys keep their order; the
    # drift record is appended after them.
    run_config = {"model": "m", "benchmark": "co_bench", "consolidate": True}
    orch = _orch(tmp_path)
    orch._resume_config_drift = {"consolidate": {"checkpoint": False, "resumed": True}}
    orch._write_run_config(tmp_path, run_config, stage="end")
    out = json.loads((tmp_path / "config.json").read_text())
    assert list(out.keys()) == list(run_config.keys()) + ["resume_config_drift"]


def test_second_drifted_resume_overwrites_drift_record(tmp_path):
    # Last-drift-wins: a later START dump with a fresh drift replaces the
    # record a prior drifted resume left behind (merge-preserve notwithstanding).
    orch = _orch(tmp_path)
    orch._resume_config_drift = {"consolidate": {"checkpoint": False, "resumed": True}}
    orch._write_run_config(tmp_path, {"model": "m"}, stage="start")
    orch._resume_config_drift = {"eval_repeats": {"checkpoint": 1, "resumed": 3}}
    orch._write_run_config(tmp_path, {"model": "m"}, stage="start")
    out = json.loads((tmp_path / "config.json").read_text())
    assert out["resume_config_drift"] == {
        "eval_repeats": {"checkpoint": 1, "resumed": 3}
    }


# --------------------------------------------------------------------------- #
# (c) byte-compat guards — no drift → no new key, config.json unchanged
# --------------------------------------------------------------------------- #

def test_fresh_run_config_json_has_no_drift_key(tmp_path):
    run_config = {"model": "m", "consolidate": True, "bench_tasks": None}
    orch = _orch(tmp_path)
    orch._write_run_config(tmp_path, run_config, stage="start")
    assert json.loads((tmp_path / "config.json").read_text()) == run_config
    orch._write_run_config(tmp_path, run_config, stage="end")
    assert json.loads((tmp_path / "config.json").read_text()) == run_config


def test_empty_drift_dict_writes_no_key(tmp_path):
    # __init__ initialises the stash to {} — falsy → key absent, so a
    # drift-free resume is byte-identical to a fresh run's config.json.
    run_config = {"model": "m"}
    orch = _orch(tmp_path)
    orch._resume_config_drift = {}
    orch._write_run_config(tmp_path, run_config, stage="end")
    assert json.loads((tmp_path / "config.json").read_text()) == run_config


def test_legacy_orchestrator_without_stashes_is_unchanged(tmp_path):
    # No _run_config_snapshot / _resume_config_drift attributes at all
    # (pre-fix orchestrators, mocks): getattr defaults keep both writers on
    # their legacy byte-identical paths.
    orch = _orch(tmp_path)
    orch._save_checkpoint(tmp_path, 2, 1, 0.4, 50, [0.4])
    ck = json.loads((tmp_path / "checkpoint.json").read_text())
    assert "run_config_snapshot" not in ck
    run_config = {"model": "m"}
    orch._write_run_config(tmp_path, run_config, stage="end")
    assert json.loads((tmp_path / "config.json").read_text()) == run_config


# --------------------------------------------------------------------------- #
# (d) the resume-branch compare itself (_config_drift, orchestrator side)


def test_config_drift_reports_changed_added_and_removed_keys():
    from meta_n.core.evolutionary_orchestrator import _config_drift

    snap = {"consolidate": False, "regression_guard": False, "model": "m"}
    resumed = {
        "consolidate": True,       # changed (e.g. bundled YAML now applies)
        "model": "m",              # unchanged
        "benchmark_config": "/x",  # added key
        "timestamp": "ignored",    # volatile — excluded from the compare
    }
    drift = _config_drift(snap, resumed)
    assert drift == {
        "benchmark_config": {"checkpoint": None, "resumed": "/x"},
        "consolidate": {"checkpoint": False, "resumed": True},
        "regression_guard": {"checkpoint": False, "resumed": None},
    }
    assert list(drift) == sorted(drift)


def test_config_drift_free_resume_is_empty():
    from meta_n.core.evolutionary_orchestrator import _config_drift

    snap = {"model": "m", "consolidate": True}
    assert _config_drift(snap, {**snap, "timestamp": "t2"}) == {}


def test_orchestrator_initialises_drift_stashes(tmp_path):
    # The __init__ contract RunPersistence getattr-reads against.
    o = _orch(tmp_path)
    assert o._run_config_snapshot is None
    assert o._resume_config_drift == {}


# --------------------------------------------------------------------------- #
# (e) end-to-end: the re-issued START dump makes the drift record durable
# --------------------------------------------------------------------------- #

def _seed_run_mocks(orch):
    orch.solver.solve = AsyncMock(return_value=("echo hi", "r", 10))
    orch.executor.execute = AsyncMock(return_value=Trace(
        task_id="t1", success=True, score=0.5, script="echo hi"))
    orch.omega.generate = AsyncMock(return_value=(InjectedCode(), 50))


_TASKS = [TaskDescription(task_id="t1", description="x")]


async def test_drifted_resume_records_drift_before_any_iteration_work(tmp_path):
    cfg_a = {"model": "m", "consolidate": False}
    orch1 = _orch(tmp_path, max_iterations=1, patience=1)
    _seed_run_mocks(orch1)
    await orch1.run(_TASKS, run_config=cfg_a)

    # Drifted resume that CRASHES at the first post-resume checkpoint write
    # (mid-iteration 1) — save_results (the END dump) never runs, exactly the
    # budget-kill/crash scenario the re-issued START dump exists for.
    cfg_b = {"model": "m", "consolidate": True}
    orch2 = _orch(tmp_path, max_iterations=2, no_early_stop=True)
    _seed_run_mocks(orch2)
    orch2._save_checkpoint = MagicMock(side_effect=RuntimeError("disk died"))
    with pytest.raises(RuntimeError, match="disk died"):
        await orch2.run(_TASKS, resume=True, run_config=cfg_b)

    out = json.loads((tmp_path / "config.json").read_text())
    assert out["resume_config_drift"] == {
        "consolidate": {"checkpoint": False, "resumed": True}
    }
    assert out["consolidate"] is True  # top-level keeps the RESUMED value


async def test_drifted_resume_that_completes_still_carries_drift(tmp_path):
    cfg_a = {"model": "m", "consolidate": False}
    orch1 = _orch(tmp_path, max_iterations=1, patience=1)
    _seed_run_mocks(orch1)
    await orch1.run(_TASKS, run_config=cfg_a)

    cfg_b = {"model": "m", "consolidate": True}
    orch2 = _orch(tmp_path, max_iterations=1, patience=1, no_early_stop=True)
    _seed_run_mocks(orch2)
    # run() alone (no save_results): only the re-issued START dump can have
    # written the drift record.
    await orch2.run(_TASKS, resume=True, run_config=cfg_b)
    out = json.loads((tmp_path / "config.json").read_text())
    assert out["resume_config_drift"] == {
        "consolidate": {"checkpoint": False, "resumed": True}
    }


async def test_drift_free_resume_keeps_config_json_byte_identical(tmp_path):
    cfg = {"model": "m", "consolidate": False}
    orch1 = _orch(tmp_path, max_iterations=1, patience=1)
    _seed_run_mocks(orch1)
    await orch1.run(_TASKS, run_config=dict(cfg))
    before = (tmp_path / "config.json").read_bytes()

    orch2 = _orch(tmp_path, max_iterations=1, patience=1, no_early_stop=True)
    _seed_run_mocks(orch2)
    await orch2.run(_TASKS, resume=True, run_config=dict(cfg))
    assert (tmp_path / "config.json").read_bytes() == before
