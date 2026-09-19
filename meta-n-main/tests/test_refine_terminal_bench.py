"""Refinement regression tests for the terminal_bench integration.

Covers:
  * F118 — litellm provider-prefix routing lockstep between
    ``terminal_bench._LITELLM_PROVIDER_PREFIXES`` (T2 backend, parent process)
    and ``scripts/_runner_common.KNOWN_LITELLM_PROVIDERS`` (OH runners). A true
    merge is architecturally blocked (``_runner_common`` must stay stdlib-only
    for the external-agents venv; ``meta_n`` cannot import from ``scripts/``),
    so this mechanical gate enforces the identical-routing contract instead.
  * F122 — host-side purge of the bind-mounted ``/logs/verifier`` dir between
    the solve phase and the verifier phase. The chmod lockdown is inert when
    the task image's default user is root (CAP_DAC_OVERRIDE), so a solve
    script could pre-write a forged reward.json that ``_read_reward`` would
    pick up if test.sh crashed before writing its own reward.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from meta_n.integrations.terminal_bench import (
    TerminalBenchAdapter,
    TerminalBenchExecutor,
    _ExecResult,
    _litellm_route_model,
    _LITELLM_PROVIDER_PREFIXES,
    _purge_verifier_dir,
)

# scripts/ holds the shared runner base; add it so we can import it directly
# (same pattern as tests/external_agents/test_runner_common.py — the runners
# execute install-free under .venv_external_agents, so this is the only way).
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import _runner_common  # noqa: E402  (after sys.path insert)


# ---------------------------------------------------------------------------
# F118 — provider-prefix lockstep
# ---------------------------------------------------------------------------

class TestLitellmProviderLockstep:
    def test_provider_sets_are_lockstep(self):
        assert set(_LITELLM_PROVIDER_PREFIXES) == {
            p.rstrip("/") for p in _runner_common.KNOWN_LITELLM_PROVIDERS
        }, (
            "terminal_bench._LITELLM_PROVIDER_PREFIXES and "
            "_runner_common.KNOWN_LITELLM_PROVIDERS must stay the UNION of each "
            "other so OpenHands and Terminus 2 route a given model id "
            "IDENTICALLY (both sets are hand-synchronized; a drift here means "
            "one backend would misroute a model the other routes correctly)."
        )

    def test_route_model_functional_parity(self):
        """The two routers must produce the same litellm model id.

        ``route_model``'s ``base_url`` gate is deliberately different (it
        no-ops without a custom endpoint), so parity is asserted with a
        non-empty base_url — the configuration under which both backends
        actually route.
        """
        for model in (
            "google/gemma-4-31b-qat",   # pricing-key slash, NOT a provider
            "openai/gpt-5.2",           # already-routed provider prefix
            "anthropic/claude-sonnet-4",  # provider prefix in both sets
            "gpt-5.2",                  # bare model name
        ):
            assert _litellm_route_model(model) == _runner_common.route_model(
                model, "http://x:1234/v1"
            ), f"OH/T2 routing diverged for model id {model!r}"

    def test_empty_model_divergence_is_pinned(self):
        """'' is deliberately OUTSIDE the parity contract.

        The T2 parent path short-circuits an empty model (backend defaults
        apply), while route_model unconditionally prefixes once base_url is
        set. Neither backend routes an empty model in practice; pin both
        behaviors so an unnoticed change reopens the discussion.
        """
        assert _litellm_route_model("") == ""
        assert _runner_common.route_model("", "http://x:1234/v1") == "openai/"


# ---------------------------------------------------------------------------
# F122 — verifier-dir anti-forgery purge
# ---------------------------------------------------------------------------

class TestPurgeVerifierDir:
    def test_purge_removes_files_dirs_symlinks(self, tmp_path):
        verifier = tmp_path / "verifier"
        verifier.mkdir()
        (verifier / "reward.json").write_text('{"reward": 1}')
        sub = verifier / "subdir"
        sub.mkdir()
        (sub / "nested.txt").write_text("x")
        (verifier / "dangling").symlink_to(tmp_path / "no-such-target")

        removed = _purge_verifier_dir(verifier)

        assert sorted(removed) == ["dangling", "reward.json", "subdir"]
        assert list(verifier.iterdir()) == []

    def test_purge_empty_dir_returns_empty(self, tmp_path):
        verifier = tmp_path / "verifier"
        verifier.mkdir()
        assert _purge_verifier_dir(verifier) == []

    def test_purge_missing_dir_returns_empty(self, tmp_path):
        assert _purge_verifier_dir(tmp_path / "absent") == []

    def test_purge_does_not_follow_symlinked_dir(self, tmp_path):
        """A symlink to a dir outside the verifier dir is unlinked, never
        rmtree'd through (the target's contents must survive)."""
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "keep.txt").write_text("keep")
        verifier = tmp_path / "verifier"
        verifier.mkdir()
        (verifier / "link").symlink_to(outside)

        removed = _purge_verifier_dir(verifier)

        assert removed == ["link"]
        assert (outside / "keep.txt").read_text() == "keep"


@pytest.fixture
def fake_task_dir(tmp_path):
    """Minimal TerminalBench task directory (mirrors tests/test_terminal_bench.py)."""
    task_dir = tmp_path / "purge-task-001"
    task_dir.mkdir()
    (task_dir / "task.toml").write_text(
        "[environment]\ncpus = 1\nmemory_mb = 1024\n"
        "[agent]\ntimeout_sec = 60\n"
        "[verifier]\ntimeout_sec = 30\n"
    )
    (task_dir / "instruction.md").write_text("# Task\nDo the thing.")
    env_dir = task_dir / "environment"
    env_dir.mkdir()
    (env_dir / "Dockerfile").write_text("FROM ubuntu:22.04\n")
    tests_dir = task_dir / "tests"
    tests_dir.mkdir()
    (tests_dir / "test.sh").write_text("echo 1 > /logs/verifier/reward.txt\n")
    return task_dir


@pytest.fixture
def adapter(fake_task_dir):
    a = TerminalBenchAdapter(task_cache_dir=str(fake_task_dir.parent))
    yield a
    a.cleanup()


@pytest.fixture
def task_desc(adapter):
    tasks = adapter.load_tasks()
    assert len(tasks) == 1
    return tasks[0]


class TestForgedRewardPurge:
    """End-to-end (mocked compose) proof that the purge closes the root-user
    forgery hole without touching honest runs. ``_read_reward`` is NOT
    patched — the score comes from whatever survives in the bind-mounted
    verifier dir."""

    async def _run(self, adapter, task_desc, tmp_path, mock_exec):
        executor = TerminalBenchExecutor(adapter)
        adapter._image_built.add(task_desc.task_id)

        host_logs = tmp_path / "logs"
        host_logs.mkdir()

        async def mock_compose(compose_files, project_name, project_dir,
                               command, **kwargs):
            return _ExecResult(return_code=0)

        with patch(
            "meta_n.integrations.terminal_bench._run_compose_command",
            side_effect=mock_compose,
        ), patch(
            "meta_n.integrations.terminal_bench._compose_exec",
            side_effect=mock_exec,
        ), patch(
            "meta_n.integrations.terminal_bench.tempfile.mkdtemp",
            return_value=str(host_logs),
        ), patch.object(
            executor, "_preflight", new_callable=AsyncMock,
        ):
            return await executor.execute("echo hi", task_desc), host_logs

    @pytest.mark.asyncio
    async def test_forged_reward_from_solve_phase_is_purged(
        self, adapter, task_desc, tmp_path
    ):
        """A reward.json pre-written during the solve phase (root bypassing the
        chmod lockdown) must NOT survive into _read_reward when the verifier
        writes nothing (e.g. test.sh crashed)."""
        async def mock_exec(compose_files, project_name, project_dir,
                            command, **kwargs):
            if command == "bash /tmp/solve.sh":
                # Simulate the root-user solve phase forging a reward through
                # the bind mount, despite the mode lockdown.
                (tmp_path / "logs" / "verifier" / "reward.json").write_text(
                    '{"reward": 1}'
                )
            return _ExecResult(return_code=0)

        trace, _ = await self._run(adapter, task_desc, tmp_path, mock_exec)

        assert trace.score == 0.0
        assert trace.success is False

    @pytest.mark.asyncio
    async def test_honest_run_unaffected_by_purge(
        self, adapter, task_desc, tmp_path
    ):
        """A reward written by the verifier phase (after the unlock) survives:
        the purge fires only BETWEEN the solve and verifier phases."""
        async def mock_exec(compose_files, project_name, project_dir,
                            command, **kwargs):
            if command.startswith("/tests/test.sh"):
                (tmp_path / "logs" / "verifier" / "reward.json").write_text(
                    '{"reward": 1}'
                )
            return _ExecResult(return_code=0)

        trace, _ = await self._run(adapter, task_desc, tmp_path, mock_exec)

        assert trace.score == 1.0
        assert trace.success is True
