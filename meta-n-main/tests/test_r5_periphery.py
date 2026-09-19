"""Regression tests for the R5 periphery fixes.

1. LocalExecutor timeout enforcement with spawned children (group kill +
   bounded post-kill drain) — execute() and _verify() return within
   ``timeout + _KILL_GRACE_S`` + slack no matter what the script spawned,
   and the whole process tree is gone afterwards.
2. Exp-6 abstraction scoring degrades honestly (None + warning) when no task
   descriptions are available, instead of a fake 'entirely generic' score.
3. metrics.py oracle_mean/oracle_gap use the run's full task universe
   (summary oracle_mean_score → seed-trace universe → subset + note).
4. atomic_json_dump fsyncs the tmp file before rename; behavior contract
   (round-trip, tmp cleanup, failure policy) unchanged.

All offline / LLM-free / no Docker.
"""

import asyncio
import json
import logging
import random
import subprocess
import time

import pytest

from meta_n.analysis.emergent_roles import EmergentRoleAnalyzer
from meta_n.analysis.metrics import ExperimentAnalyzer
from meta_n.core.base_executor import LocalExecutor
from meta_n.core.meta_layer import InjectedCode, TaskDescription
from meta_n.utils.atomic_io import atomic_json_dump


# --------------------------------------------------------------------------- #
# 1. LocalExecutor timeout with spawned children
# --------------------------------------------------------------------------- #

def _unique_sleep_secs() -> str:
    """A sleep duration unlikely to collide with anything else in ps output."""
    return f"31{random.randint(10**5, 10**6 - 1)}"


def _sleep_gone_from_ps(dur: str, wait_s: float = 5.0) -> bool:
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        out = subprocess.run(
            ["ps", "ax", "-o", "command"], capture_output=True, text=True
        ).stdout
        if f"sleep {dur}" not in out:
            return True
        time.sleep(0.1)
    return False


@pytest.fixture
def executor():
    return LocalExecutor()


@pytest.fixture
def simple_task():
    return TaskDescription(task_id="r5_timeout", description="timeout test")


class TestLocalExecutorTimeoutTreeKill:
    @pytest.mark.asyncio
    async def test_backgrounded_child_does_not_defeat_timeout(
        self, executor, simple_task
    ):
        dur = _unique_sleep_secs()
        script = f"sleep {dur} &\necho started\nsleep {dur}\n"
        t0 = time.monotonic()
        trace = await executor.execute(script, simple_task, timeout=1)
        elapsed = time.monotonic() - t0

        # timeout(1) + _KILL_GRACE_S(2) + slack — NOT the child's 31xxxx s.
        assert elapsed < 6, f"execute() took {elapsed:.1f}s despite timeout=1"
        assert trace.success is False
        assert "timed out" in trace.error_summary.lower()
        assert trace.stderr == "Timeout exceeded"
        # Salvaged output from before the kill.
        assert "started" in trace.stdout
        # Group kill: neither the backgrounded nor the foreground sleep survives.
        assert _sleep_gone_from_ps(dur), f"orphaned 'sleep {dur}' still running"

    @pytest.mark.asyncio
    async def test_verify_timeout_with_child_bounded(self, executor):
        dur = _unique_sleep_secs()
        t0 = time.monotonic()
        ok = await executor._verify(f"sleep {dur} &\nsleep {dur}\n", timeout=1)
        elapsed = time.monotonic() - t0

        assert ok is False
        assert elapsed < 6, f"_verify() took {elapsed:.1f}s despite timeout=1"
        assert _sleep_gone_from_ps(dur), f"orphaned 'sleep {dur}' still running"

    @pytest.mark.asyncio
    async def test_normal_script_unaffected(self, executor, simple_task):
        trace = await executor.execute("echo hello", simple_task)
        assert trace.success is True
        assert trace.exit_code == 0
        assert trace.stdout.strip() == "hello"

    @pytest.mark.asyncio
    async def test_normal_verify_unaffected(self, executor):
        assert await executor._verify("true", timeout=5) is True
        assert await executor._verify("false", timeout=5) is False


# --------------------------------------------------------------------------- #
# 2. Exp-6 abstraction: honest degradation without tasks
# --------------------------------------------------------------------------- #

@pytest.fixture
def _no_embedding_model(monkeypatch):
    def _raise(self):
        raise ImportError("embeddings disabled in tests")
    monkeypatch.setattr(EmergentRoleAnalyzer, "_get_embedding_model", _raise)


def _write_linear_run(root, *, with_tasks_file=False):
    d2 = root / "depth_2"
    d2.mkdir()
    (d2 / "injected_code.json").write_text(json.dumps({
        "pre_process": (
            "if task.task_id == 'symptom2disease_042':\n"
            "    additional_context = 'create the hello world file'"
        ),
        "rationale": "Handle the hardcoded task specially.",
        "source_depth": 2,
    }))
    config = {"tasks_file": None, "benchmark": "symptom2disease"}
    if with_tasks_file:
        tasks_path = root / "tasks.json"
        tasks_path.write_text(json.dumps([{
            "task_id": "symptom2disease_042",
            "description": "create the hello world file for the patient",
        }]))
        config["tasks_file"] = str(tasks_path)
    (root / "config.json").write_text(json.dumps(config))
    return root


class TestExp6AbstractionDegradation:
    def test_empty_tasks_yields_none_and_warns(
        self, tmp_path, caplog, _no_embedding_model
    ):
        _write_linear_run(tmp_path)
        analyzer = EmergentRoleAnalyzer(str(tmp_path))
        with caplog.at_level(logging.WARNING, logger="meta_n.analysis.emergent_roles"):
            result = analyzer.analyze()

        assert result.layer_profiles[0].abstraction_score is None
        assert result.abstraction_gradient == [None]
        assert any(
            "abstraction_score_exp6" in rec.message for rec in caplog.records
        )
        # Serialization tolerates None (null in role_analysis.json).
        payload = json.loads(json.dumps(result.to_dict()))
        assert payload["layer_profiles"][0]["abstraction_score_exp6"] is None
        assert payload["abstraction_gradient_exp6"] == [None]

    def test_explicit_empty_task_list_also_none(
        self, tmp_path, caplog, _no_embedding_model
    ):
        _write_linear_run(tmp_path)
        analyzer = EmergentRoleAnalyzer(str(tmp_path))
        with caplog.at_level(logging.WARNING, logger="meta_n.analysis.emergent_roles"):
            result = analyzer.analyze(tasks=[])
        assert result.abstraction_gradient == [None]
        assert any(
            rec.name == "meta_n.analysis.emergent_roles" for rec in caplog.records
        )

    def test_nonempty_tasks_numeric_behavior_unchanged(
        self, tmp_path, caplog, _no_embedding_model
    ):
        _write_linear_run(tmp_path, with_tasks_file=True)
        analyzer = EmergentRoleAnalyzer(str(tmp_path))
        with caplog.at_level(logging.WARNING, logger="meta_n.analysis.emergent_roles"):
            result = analyzer.analyze()

        score = result.layer_profiles[0].abstraction_score
        assert score is not None
        assert 0.0 <= score < 1.0  # task-specific refs present → not 'entirely generic'
        assert not [
            rec for rec in caplog.records
            if rec.name == "meta_n.analysis.emergent_roles"
        ]

    def test_direct_score_none_vs_numeric(self):
        analyzer = EmergentRoleAnalyzer("/tmp/fake")
        code = InjectedCode(
            pre_process="if task.task_id == 't1':\n    pass",
            rationale="specific",
            source_depth=2,
        )
        assert analyzer._compute_abstraction_score(code, []) is None
        task = TaskDescription(task_id="t1", description="some task description here now")
        numeric = analyzer._compute_abstraction_score(code, [task])
        assert isinstance(numeric, float)


# --------------------------------------------------------------------------- #
# 3. metrics.py oracle denominator
# --------------------------------------------------------------------------- #

def _write_evolutionary_run(
    root, *, summary_oracle=None, seed_task_ids=("t1", "t2", "t3")
):
    """per_task_best covers only t1/t2; t3 (when present) has no finite score."""
    summary = {
        "archive_size": 2,
        "total_iterations": 1,
        "best_mean_score": 0.4,
        "best_candidate_id": "gen0_seed",
        "per_task_best_scores": {"t1": 0.9, "t2": 0.5},
        "convergence_history": [0.4],
    }
    if summary_oracle is not None:
        summary["oracle_mean_score"] = summary_oracle
    (root / "summary.json").write_text(json.dumps(summary))
    (root / "config.json").write_text(json.dumps({"model": "test"}))

    archive = root / "archive"
    archive.mkdir()
    (archive / "index.json").write_text(json.dumps({
        "size": 2,
        "best_mean_score": 0.4,
        "best_candidate_id": "gen0_seed",
        "per_task_best": {
            "t1": {"score": 0.9, "candidate_id": "gen0_seed"},
            "t2": {"score": 0.5, "candidate_id": "gen1_pgen0_seed_k0"},
        },
        "candidates": [],
    }))
    if seed_task_ids:
        traces_dir = archive / "gen0_seed" / "traces"
        traces_dir.mkdir(parents=True)
        for tid in seed_task_ids:
            (traces_dir / f"{tid}.json").write_text(json.dumps({
                "task_id": tid, "depth": 1, "script": "x=1", "success": False,
                "score": 0.0, "error_summary": "", "stderr": "", "stdout": "",
            }))
    return root


class TestOracleDenominator:
    def test_summary_oracle_mean_score_preferred(self, tmp_path):
        _write_evolutionary_run(tmp_path, summary_oracle=0.4667)
        analyzer = ExperimentAnalyzer(tmp_path, use_embeddings=False)
        result = analyzer.compute_archive_diversity()
        assert result["oracle_mean"] == pytest.approx(0.4667)
        assert result["oracle_gap"] == pytest.approx(0.4667 - 0.4, abs=1e-4)
        assert "oracle_denominator_note" not in result

    def test_seed_trace_universe_missing_task_counts_as_zero(self, tmp_path):
        _write_evolutionary_run(tmp_path)  # no summary oracle; seed has t1..t3
        analyzer = ExperimentAnalyzer(tmp_path, use_embeddings=False)
        result = analyzer.compute_archive_diversity()
        # (0.9 + 0.5 + 0.0) / 3, NOT (0.9 + 0.5) / 2 = 0.7.
        assert result["oracle_mean"] == pytest.approx((0.9 + 0.5) / 3, abs=1e-4)
        assert result["oracle_gap"] == pytest.approx((0.9 + 0.5) / 3 - 0.4, abs=1e-4)
        assert "oracle_denominator_note" not in result

    def test_subset_fallback_notes_denominator(self, tmp_path):
        _write_evolutionary_run(tmp_path, seed_task_ids=())
        analyzer = ExperimentAnalyzer(tmp_path, use_embeddings=False)
        result = analyzer.compute_archive_diversity()
        assert result["oracle_mean"] == pytest.approx(0.7)
        assert "subset denominator" in result["oracle_denominator_note"]

    def test_complete_coverage_unchanged(self, tmp_path):
        # Seed universe == per_task_best keys → same value as before the fix.
        _write_evolutionary_run(tmp_path, seed_task_ids=("t1", "t2"))
        analyzer = ExperimentAnalyzer(tmp_path, use_embeddings=False)
        result = analyzer.compute_archive_diversity()
        assert result["oracle_mean"] == pytest.approx(0.7)
        assert "oracle_denominator_note" not in result


# --------------------------------------------------------------------------- #
# 4. atomic_json_dump fsync
# --------------------------------------------------------------------------- #

class TestAtomicJsonDumpFsync:
    def test_round_trip_and_tmp_cleanup(self, tmp_path):
        obj = {"a": 1, "b": [1, 2, 3], "nested": {"c": None}}
        path = tmp_path / "out.json"
        atomic_json_dump(path, obj)
        assert path.read_text() == json.dumps(obj, indent=2)
        assert not (tmp_path / "out.json.tmp").exists()

    def test_fsyncs_tmp_file_before_rename(self, tmp_path, monkeypatch):
        import os as os_mod
        import meta_n.utils.atomic_io as atomic_io_mod

        synced_fds = []
        real_fsync = os_mod.fsync

        def recording_fsync(fd):
            synced_fds.append(fd)
            return real_fsync(fd)

        monkeypatch.setattr(atomic_io_mod.os, "fsync", recording_fsync)
        path = tmp_path / "out.json"
        atomic_json_dump(path, {"v": 1})
        # At least the tmp-file fsync; the directory fsync is best-effort.
        assert len(synced_fds) >= 1
        assert json.loads(path.read_text()) == {"v": 1}

    def test_failure_still_removes_tmp_and_preserves_original(self, tmp_path):
        path = tmp_path / "out.json"
        path.write_text('{"original": true}')
        with pytest.raises(TypeError):
            atomic_json_dump(path, {"bad": object()})
        assert path.read_text() == '{"original": true}'
        assert not (tmp_path / "out.json.tmp").exists()
