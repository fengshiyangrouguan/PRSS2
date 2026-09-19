"""Regression tests for audit fixes in meta_n/integrations/terminal_bench.py.

Covers findings:
  39 — _read_reward must not let AttributeError escape on a non-dict reward.json
        (bare scalar / list / null); it should coerce a scalar and otherwise
        fall through to reward.txt, honoring the "returns 0.0 on any failure"
        contract.
  56 — the tb2_compose_* template tempdir must be reclaimed on GC / interpreter
        exit (a finalizer), not leaked because cleanup() is only called by tests.
  57 — one malformed task.toml must skip that task, not abort the whole
        benchmark load.
  70 — the held-out verifier tests must be copied into the container, and the
        reward dir made world-writable, ONLY AFTER the solve script has run
        (no test-set leak / no forgeable reward during the agent phase).

All tests are offline: no Docker, no network, no LLM. The execute() flow test
monkeypatches the module-level compose helpers so nothing is actually spawned.
"""

from __future__ import annotations

import asyncio
import gc
import tempfile
from pathlib import Path

import pytest

import meta_n.integrations.terminal_bench as tb
from meta_n.core.meta_layer import TaskDescription
from meta_n.integrations.terminal_bench import (
    TerminalBenchAdapter,
    TerminalBenchExecutor,
    _ExecResult,
    _read_reward,
)


# ---------------------------------------------------------------------------
# Finding 39 — _read_reward on a non-dict reward.json
# ---------------------------------------------------------------------------

class TestFinding39ScalarRewardJson:
    def test_bare_scalar_reward_json_is_coerced(self, tmp_path):
        """A bare scalar reward.json (e.g. ``1``) must be coerced, not raise.

        On the unfixed code ``json.loads("1").get(...)`` raises AttributeError
        which escapes _read_reward (not in the except tuple), so this call would
        raise instead of returning 1.0.
        """
        (tmp_path / "reward.json").write_text("1")
        assert _read_reward(tmp_path) == 1.0

    def test_bare_float_scalar_reward_json(self, tmp_path):
        (tmp_path / "reward.json").write_text("0.5")
        assert _read_reward(tmp_path) == 0.5

    def test_non_numeric_json_falls_through_to_reward_txt(self, tmp_path):
        """A list/null reward.json must fall through to reward.txt, not raise.

        On the unfixed code ``None.get`` / ``list.get`` raises AttributeError,
        skipping the reward.txt fallback entirely.
        """
        (tmp_path / "reward.json").write_text("null")
        (tmp_path / "reward.txt").write_text("0.75")
        assert _read_reward(tmp_path) == 0.75

    def test_list_reward_json_falls_through(self, tmp_path):
        (tmp_path / "reward.json").write_text("[1]")
        (tmp_path / "reward.txt").write_text("1")
        assert _read_reward(tmp_path) == 1.0

    def test_dict_reward_json_still_works(self, tmp_path):
        """Default protocol (object with a "reward" key) is unchanged."""
        (tmp_path / "reward.json").write_text('{"reward": 1}')
        assert _read_reward(tmp_path) == 1.0


# ---------------------------------------------------------------------------
# Finding 56 — compose-template tempdir is reclaimed (no leak)
# ---------------------------------------------------------------------------

class TestFinding56ComposeDirReclaimed:
    def test_finalizer_removes_compose_dir_on_gc(self):
        """Dropping the adapter (then gc) must remove its tb2_compose_* dir.

        On the unfixed code there is no finalizer/__del__/atexit, so the dir
        survives garbage collection and leaks for the process lifetime.
        """
        adapter = TerminalBenchAdapter()
        compose_dir = adapter._compose_dir
        assert compose_dir.exists()

        del adapter
        gc.collect()

        assert not compose_dir.exists()

    def test_explicit_cleanup_is_idempotent(self):
        adapter = TerminalBenchAdapter()
        compose_dir = adapter._compose_dir
        assert compose_dir.exists()

        adapter.cleanup()
        assert not compose_dir.exists()
        # A second call must not raise (finalizer already detached).
        adapter.cleanup()


# ---------------------------------------------------------------------------
# Finding 57 — one malformed task.toml skips that task, not the whole load
# ---------------------------------------------------------------------------

def _make_harbor_task(root: Path, name: str, toml_text: str) -> None:
    task = root / name
    (task / "environment").mkdir(parents=True)
    (task / "tests").mkdir()
    (task / "instruction.md").write_text("do the thing")
    (task / "task.toml").write_text(toml_text)


class TestFinding57MalformedTaskToml:
    def test_malformed_task_toml_is_skipped(self, tmp_path):
        """A corrupt task.toml must not abort the entire benchmark load.

        On the unfixed code the unguarded ``_parse_task_toml`` raises
        TOMLDecodeError straight out of load_tasks, crashing the run before any
        good task is returned.
        """
        cache = tmp_path / "cache"
        cache.mkdir()
        # "bad_task" sorts before "good_task", so it is parsed first.
        _make_harbor_task(cache, "bad_task", "this is = = not valid toml\n[[[\n")
        _make_harbor_task(cache, "good_task", "[environment]\ncpus = 2\n")

        adapter = TerminalBenchAdapter(task_cache_dir=str(cache))
        try:
            tasks = adapter.load_tasks()
        finally:
            adapter.cleanup()

        names = {t.metadata["task_name"] for t in tasks}
        assert "good_task" in names
        assert "bad_task" not in names


# ---------------------------------------------------------------------------
# Finding 70 — verifier tests staged + reward dir unlocked only AFTER solve
# ---------------------------------------------------------------------------

def _command_repr(args, kwargs) -> str:
    """Extract the compose command (list -> joined str, str -> str)."""
    cmd = kwargs.get("command")
    if cmd is None and len(args) >= 4:
        cmd = args[3]
    if isinstance(cmd, (list, tuple)):
        return " ".join(str(p) for p in cmd)
    return str(cmd)


class TestFinding70StagingOrder:
    def _run_execute_capturing_order(self, monkeypatch) -> list[str]:
        order: list[str] = []

        async def fake_run(*args, **kwargs):
            order.append(_command_repr(args, kwargs))
            return _ExecResult(return_code=0)

        async def fake_exec(*args, **kwargs):
            order.append(_command_repr(args, kwargs))
            return _ExecResult(stdout="", stderr="", return_code=0)

        monkeypatch.setattr(tb, "_run_compose_command", fake_run)
        monkeypatch.setattr(tb, "_compose_exec", fake_exec)

        tmp = Path(tempfile.mkdtemp())
        task_dir = tmp / "mytask"
        (task_dir / "environment").mkdir(parents=True)
        (task_dir / "tests").mkdir()
        (task_dir / "tests" / "test.sh").write_text("#!/bin/bash\necho hi\n")

        adapter = TerminalBenchAdapter()
        task_id = "mytask"
        adapter._task_dirs[task_id] = task_dir
        adapter._task_configs[task_id] = {
            "cpus": 1, "memory_mb": 2048, "allow_internet": True,
        }

        async def _noop_ensure(_task_id):
            return None

        adapter._ensure_image = _noop_ensure  # type: ignore[assignment]

        executor = TerminalBenchExecutor(adapter)
        executor._docker_checked = True  # skip docker preflight

        task = TaskDescription(
            task_id=task_id,
            description="solve it",
            metadata={
                "task_dir": str(task_dir),
                "timeout_sec": 1800,
                "verifier_timeout_sec": 900,
            },
        )

        try:
            asyncio.run(executor.execute("echo hello", task))
        finally:
            adapter.cleanup()

        return order

    def test_tests_and_reward_dir_staged_after_solve(self, monkeypatch):
        order = self._run_execute_capturing_order(monkeypatch)

        def first_index(predicate) -> int:
            for i, c in enumerate(order):
                if predicate(c):
                    return i
            return -1

        solve_idx = first_index(lambda c: c == "bash /tmp/solve.sh")
        tests_copy_idx = first_index(lambda c: "main:/tests/" in c)
        verifier_unlock_idx = first_index(
            lambda c: "chmod" in c and "/logs/verifier" in c
        )

        assert solve_idx >= 0, f"solve command not found in {order}"
        assert tests_copy_idx >= 0, f"tests copy not found in {order}"
        assert verifier_unlock_idx >= 0, f"verifier unlock not found in {order}"

        # The held-out tests must be copied AFTER the solve script runs.
        assert solve_idx < tests_copy_idx, (
            f"tests staged before solve (leak): solve@{solve_idx} "
            f"tests@{tests_copy_idx} order={order}"
        )
        # The reward dir must be made writable AFTER the solve script runs.
        assert solve_idx < verifier_unlock_idx, (
            f"reward dir unlocked before solve (forgeable): solve@{solve_idx} "
            f"unlock@{verifier_unlock_idx} order={order}"
        )

    def test_solve_phase_has_no_tests_and_locked_reward_dir(self, monkeypatch):
        """Belt-and-suspenders: nothing before solve touches /tests or unlocks
        the verifier reward dir."""
        order = self._run_execute_capturing_order(monkeypatch)
        solve_idx = next(
            i for i, c in enumerate(order) if c == "bash /tmp/solve.sh"
        )
        before_solve = order[:solve_idx]
        assert not any("main:/tests/" in c for c in before_solve), order
        assert not any(
            "chmod" in c and "/logs/verifier" in c for c in before_solve
        ), order


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
