"""Tests for TerminalBench 2.0 adapter and executor."""

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from meta_n.core.meta_layer import TaskDescription, Trace
from meta_n.integrations.benchmark import EvalResult
from meta_n.integrations.terminal_bench import (
    TerminalBenchAdapter,
    TerminalBenchExecutor,
    _ExecResult,
    _read_reward,
    _sanitize_compose_name,
    _sanitize_image_name,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_task_dir(tmp_path):
    """Create a minimal TerminalBench task directory."""
    task_dir = tmp_path / "test-task-001"
    task_dir.mkdir()

    # task.toml
    (task_dir / "task.toml").write_text(
        '[environment]\n'
        'cpus = 2\n'
        'memory_mb = 4096\n'
        'allow_internet = false\n'
        '\n'
        '[agent]\n'
        'timeout_sec = 600\n'
        '\n'
        '[verifier]\n'
        'timeout_sec = 60\n'
    )

    # instruction.md
    (task_dir / "instruction.md").write_text(
        "# Test Task\n\nCreate a file called /tmp/output.txt with 'hello' in it."
    )

    # environment/Dockerfile
    env_dir = task_dir / "environment"
    env_dir.mkdir()
    (env_dir / "Dockerfile").write_text("FROM ubuntu:22.04\n")

    # tests/test.sh
    tests_dir = task_dir / "tests"
    tests_dir.mkdir()
    (tests_dir / "test.sh").write_text(
        '#!/bin/bash\n'
        'if [ -f /tmp/output.txt ]; then\n'
        '  echo 1 > /logs/verifier/reward.txt\n'
        'else\n'
        '  echo 0 > /logs/verifier/reward.txt\n'
        'fi\n'
    )

    # solution/solve.sh (oracle)
    sol_dir = task_dir / "solution"
    sol_dir.mkdir()
    (sol_dir / "solve.sh").write_text("echo hello > /tmp/output.txt\n")

    return task_dir


@pytest.fixture
def fake_task_dir_with_compose(fake_task_dir):
    """Task directory with a custom docker-compose.yaml."""
    compose_content = (
        "services:\n"
        "  main:\n"
        "    environment:\n"
        "      - MY_VAR=hello\n"
        "  db:\n"
        "    image: postgres:15\n"
    )
    (fake_task_dir / "environment" / "docker-compose.yaml").write_text(
        compose_content
    )
    return fake_task_dir


@pytest.fixture
def adapter(tmp_path, fake_task_dir):
    """TerminalBenchAdapter with a single fake task."""
    # Point adapter at the parent of the fake task directory
    a = TerminalBenchAdapter(task_cache_dir=str(fake_task_dir.parent))
    yield a
    a.cleanup()


@pytest.fixture
def task_desc(adapter):
    """Load the single fake task."""
    tasks = adapter.load_tasks()
    assert len(tasks) == 1
    return tasks[0]


# ---------------------------------------------------------------------------
# Sanitization helpers
# ---------------------------------------------------------------------------

class TestSanitize:
    def test_compose_name_basic(self):
        assert _sanitize_compose_name("my-task") == "my-task"

    def test_compose_name_uppercase(self):
        assert _sanitize_compose_name("My Task") == "my-task"

    def test_compose_name_special_chars(self):
        assert _sanitize_compose_name("task/with:special") == "task-with-special"

    def test_compose_name_leading_nonalpha(self):
        assert _sanitize_compose_name("-bad-start") == "0-bad-start"

    def test_image_name_basic(self):
        assert _sanitize_image_name("tb2-my_task") == "tb2-my_task"

    def test_image_name_dots_allowed(self):
        assert _sanitize_image_name("tb2-v1.0") == "tb2-v1.0"


# ---------------------------------------------------------------------------
# Reward parsing
# ---------------------------------------------------------------------------

class TestReadReward:
    def test_reward_txt_one(self, tmp_path):
        (tmp_path / "reward.txt").write_text("1\n")
        assert _read_reward(tmp_path) == 1.0

    def test_reward_txt_zero(self, tmp_path):
        (tmp_path / "reward.txt").write_text("0\n")
        assert _read_reward(tmp_path) == 0.0

    def test_reward_txt_fractional(self, tmp_path):
        (tmp_path / "reward.txt").write_text("0.75\n")
        assert _read_reward(tmp_path) == 0.75

    def test_reward_json(self, tmp_path):
        (tmp_path / "reward.json").write_text('{"reward": 1}')
        assert _read_reward(tmp_path) == 1.0

    def test_reward_json_preferred_over_txt(self, tmp_path):
        (tmp_path / "reward.json").write_text('{"reward": 0.5}')
        (tmp_path / "reward.txt").write_text("1\n")
        # JSON is checked first
        assert _read_reward(tmp_path) == 0.5

    def test_missing_reward(self, tmp_path):
        assert _read_reward(tmp_path) == 0.0

    def test_empty_reward_txt(self, tmp_path):
        (tmp_path / "reward.txt").write_text("")
        assert _read_reward(tmp_path) == 0.0

    def test_malformed_reward_txt(self, tmp_path):
        (tmp_path / "reward.txt").write_text("not_a_number\n")
        assert _read_reward(tmp_path) == 0.0

    def test_malformed_reward_json(self, tmp_path):
        (tmp_path / "reward.json").write_text("{bad json")
        # Falls through to reward.txt (missing) -> 0.0
        assert _read_reward(tmp_path) == 0.0


# ---------------------------------------------------------------------------
# Adapter: task loading
# ---------------------------------------------------------------------------

class TestTerminalBenchAdapter:
    def test_properties(self, adapter):
        assert adapter.name == "terminal_bench"

    def test_load_tasks(self, adapter, fake_task_dir):
        tasks = adapter.load_tasks()
        assert len(tasks) == 1
        task = tasks[0]
        assert "test-task-001" in task.task_id
        assert "Create a file" in task.description
        assert task.metadata["benchmark"] == "terminal_bench"
        assert task.metadata["cpus"] == 2
        assert task.metadata["memory_mb"] == 4096
        assert task.metadata["timeout_sec"] == 600
        assert task.metadata["verifier_timeout_sec"] == 60
        assert task.metadata["allow_internet"] is False
        assert task.metadata["solution_language"] == "bash"

    def test_load_tasks_limit(self, tmp_path):
        """Create multiple tasks and verify limit works."""
        for i in range(5):
            task_dir = tmp_path / f"task-{i:03d}"
            task_dir.mkdir()
            (task_dir / "task.toml").write_text("[environment]\n")
            (task_dir / "instruction.md").write_text(f"Task {i}")
            (task_dir / "environment").mkdir()
            (task_dir / "environment" / "Dockerfile").write_text("FROM ubuntu:22.04\n")
            (task_dir / "tests").mkdir()
            (task_dir / "tests" / "test.sh").write_text("echo 1 > /logs/verifier/reward.txt\n")

        a = TerminalBenchAdapter(task_cache_dir=str(tmp_path))
        tasks = a.load_tasks(limit=3)
        assert len(tasks) == 3
        a.cleanup()

    def test_load_tasks_filter(self, tmp_path):
        """Verify task_names filter."""
        for name in ["task-a", "task-b", "task-c"]:
            task_dir = tmp_path / name
            task_dir.mkdir()
            (task_dir / "task.toml").write_text("[environment]\n")
            (task_dir / "instruction.md").write_text(f"Task {name}")
            (task_dir / "environment").mkdir()
            (task_dir / "environment" / "Dockerfile").write_text("FROM ubuntu:22.04\n")
            (task_dir / "tests").mkdir()
            (task_dir / "tests" / "test.sh").write_text("echo ok\n")

        a = TerminalBenchAdapter(
            task_cache_dir=str(tmp_path),
            task_names=["task-a", "task-c"],
        )
        tasks = a.load_tasks()
        ids = [t.metadata["task_name"] for t in tasks]
        assert "task-a" in ids
        assert "task-c" in ids
        assert "task-b" not in ids
        a.cleanup()

    def test_load_tasks_skips_invalid(self, tmp_path):
        """Tasks missing instruction.md or environment/ are skipped."""
        # Valid task
        valid = tmp_path / "valid"
        valid.mkdir()
        (valid / "task.toml").write_text("[environment]\n")
        (valid / "instruction.md").write_text("Valid task")
        (valid / "environment").mkdir()
        (valid / "environment" / "Dockerfile").write_text("FROM ubuntu:22.04\n")
        (valid / "tests").mkdir()
        (valid / "tests" / "test.sh").write_text("echo ok\n")

        # Invalid: no instruction.md
        no_inst = tmp_path / "no-instruction"
        no_inst.mkdir()
        (no_inst / "task.toml").write_text("[environment]\n")
        (no_inst / "environment").mkdir()
        (no_inst / "tests").mkdir()

        a = TerminalBenchAdapter(task_cache_dir=str(tmp_path))
        tasks = a.load_tasks()
        assert len(tasks) == 1
        assert tasks[0].metadata["task_name"] == "valid"
        a.cleanup()

    def test_parse_task_toml_defaults(self, tmp_path):
        """Missing fields get sensible defaults."""
        toml_path = tmp_path / "task.toml"
        toml_path.write_text("[environment]\n")  # minimal

        a = TerminalBenchAdapter(task_cache_dir=str(tmp_path))
        config = a._parse_task_toml(toml_path)
        assert config["cpus"] == 1
        assert config["memory_mb"] == 2048
        assert config["allow_internet"] is True
        assert config["timeout_sec"] == 1800
        assert config["verifier_timeout_sec"] == 900
        a.cleanup()


# ---------------------------------------------------------------------------
# Adapter: compose file ordering
# ---------------------------------------------------------------------------

class TestComposeFiles:
    def test_simple_task(self, adapter, task_desc):
        """Simple task: base + prebuilt + no-network (allow_internet=false)."""
        files = adapter._get_compose_files(task_desc.task_id, prebuilt=True)
        names = [f.name for f in files]
        assert names[0] == "compose-base.yaml"
        assert names[1] == "compose-prebuilt.yaml"
        # allow_internet=false in our fake task -> no-network appended
        assert "compose-no-network.yaml" in names

    def test_build_mode(self, adapter, task_desc):
        """Build mode uses compose-build.yaml instead of prebuilt."""
        files = adapter._get_compose_files(task_desc.task_id, prebuilt=False)
        names = [f.name for f in files]
        assert names[1] == "compose-build.yaml"

    def test_task_with_compose(self, adapter, fake_task_dir_with_compose):
        """Task with its own docker-compose.yaml includes it in the chain."""
        tasks = adapter.load_tasks()
        task = tasks[0]
        files = adapter._get_compose_files(task.task_id, prebuilt=True)
        names = [f.name for f in files]
        assert "docker-compose.yaml" in names

    def test_internet_allowed(self, tmp_path):
        """Task with allow_internet=true omits no-network."""
        task_dir = tmp_path / "internet-task"
        task_dir.mkdir()
        (task_dir / "task.toml").write_text(
            "[environment]\nallow_internet = true\n"
        )
        (task_dir / "instruction.md").write_text("Task with internet")
        (task_dir / "environment").mkdir()
        (task_dir / "environment" / "Dockerfile").write_text("FROM ubuntu:22.04\n")
        (task_dir / "tests").mkdir()
        (task_dir / "tests" / "test.sh").write_text("echo ok\n")

        a = TerminalBenchAdapter(task_cache_dir=str(tmp_path))
        tasks = a.load_tasks()
        files = a._get_compose_files(tasks[0].task_id, prebuilt=True)
        names = [f.name for f in files]
        assert "compose-no-network.yaml" not in names
        a.cleanup()


# ---------------------------------------------------------------------------
# Adapter: image management (mocked Docker)
# ---------------------------------------------------------------------------

class TestImageManagement:
    @pytest.mark.asyncio
    async def test_image_tag(self, adapter):
        tag = adapter._image_tag("my-task")
        assert tag == "tb2-my-task"

    @pytest.mark.asyncio
    async def test_ensure_image_skips_if_cached(self, adapter, task_desc):
        """If image is in _image_built set, skip everything."""
        adapter._image_built.add(task_desc.task_id)
        # Should not raise (no Docker calls needed)
        await adapter._ensure_image(task_desc.task_id)

    @pytest.mark.asyncio
    async def test_ensure_image_checks_docker_store(self, adapter, task_desc):
        """If image exists in Docker store, add to _image_built and skip build."""
        with patch(
            "meta_n.integrations.terminal_bench.TerminalBenchAdapter._image_exists",
            new_callable=AsyncMock,
            return_value=True,
        ):
            await adapter._ensure_image(task_desc.task_id)
            assert task_desc.task_id in adapter._image_built

    @pytest.mark.asyncio
    async def test_ensure_image_builds_when_missing(self, adapter, task_desc):
        """If image doesn't exist, build it."""
        with patch(
            "meta_n.integrations.terminal_bench.TerminalBenchAdapter._image_exists",
            new_callable=AsyncMock,
            return_value=False,
        ), patch(
            "meta_n.integrations.terminal_bench._run_compose_command",
            new_callable=AsyncMock,
            return_value=_ExecResult(return_code=0),
        ) as mock_compose:
            await adapter._ensure_image(task_desc.task_id)
            assert task_desc.task_id in adapter._image_built

            # Should have called compose with "build" command
            calls = mock_compose.call_args_list
            build_call = [c for c in calls if "build" in c.kwargs.get("command", [])]
            assert len(build_call) >= 1

    @pytest.mark.asyncio
    async def test_concurrent_builds_share_lock(self, adapter, task_desc):
        """Two concurrent _ensure_image calls should only build once."""
        build_count = 0

        async def mock_compose(**kwargs):
            nonlocal build_count
            if "build" in kwargs.get("command", []):
                build_count += 1
                await asyncio.sleep(0.01)  # Simulate build time
            return _ExecResult(return_code=0)

        with patch(
            "meta_n.integrations.terminal_bench.TerminalBenchAdapter._image_exists",
            new_callable=AsyncMock,
            return_value=False,
        ), patch(
            "meta_n.integrations.terminal_bench._run_compose_command",
            side_effect=mock_compose,
        ):
            await asyncio.gather(
                adapter._ensure_image(task_desc.task_id),
                adapter._ensure_image(task_desc.task_id),
            )
            assert build_count == 1


# ---------------------------------------------------------------------------
# Executor (mocked Docker)
# ---------------------------------------------------------------------------

class TestTerminalBenchExecutor:
    @pytest.mark.asyncio
    async def test_execute_success(self, adapter, task_desc):
        """Successful execution: reward=1 -> score=1.0, success=True."""
        executor = TerminalBenchExecutor(adapter)
        adapter._image_built.add(task_desc.task_id)

        call_log: list[str] = []

        async def mock_compose(compose_files, project_name, project_dir,
                               command, **kwargs):
            cmd_str = " ".join(command)
            call_log.append(cmd_str)
            return _ExecResult(stdout="", stderr="", return_code=0)

        async def mock_exec(compose_files, project_name, project_dir,
                            command, **kwargs):
            call_log.append(f"exec:{command}")
            return _ExecResult(stdout="output", stderr="", return_code=0)

        with patch(
            "meta_n.integrations.terminal_bench._run_compose_command",
            side_effect=mock_compose,
        ), patch(
            "meta_n.integrations.terminal_bench._compose_exec",
            side_effect=mock_exec,
        ), patch(
            "meta_n.integrations.terminal_bench._read_reward",
            return_value=1.0,
        ), patch.object(
            executor, "_preflight", new_callable=AsyncMock,
        ):
            trace = await executor.execute("echo hello", task_desc)

        assert trace.score == 1.0
        assert trace.success is True
        assert trace.task_id == task_desc.task_id
        assert trace.script == "echo hello"

        # Verify compose down was called with --remove-orphans AND --volumes
        # (anonymous volumes from VOLUME-declaring task images must not leak).
        down_cmds = [c for c in call_log if "down" in c]
        assert len(down_cmds) >= 1
        assert any("--remove-orphans" in c for c in down_cmds)
        assert any("--volumes" in c for c in down_cmds)

        # Verify no --rmi in any command
        for cmd in call_log:
            assert "--rmi" not in cmd

    @pytest.mark.asyncio
    async def test_execute_failure(self, adapter, task_desc):
        """Failed execution: reward=0 -> score=0.0, success=False."""
        executor = TerminalBenchExecutor(adapter)
        adapter._image_built.add(task_desc.task_id)

        async def mock_compose(*args, **kwargs):
            return _ExecResult(return_code=0)

        async def mock_exec(*args, **kwargs):
            return _ExecResult(stdout="", stderr="Error: not found", return_code=1)

        with patch(
            "meta_n.integrations.terminal_bench._run_compose_command",
            side_effect=mock_compose,
        ), patch(
            "meta_n.integrations.terminal_bench._compose_exec",
            side_effect=mock_exec,
        ), patch(
            "meta_n.integrations.terminal_bench._read_reward",
            return_value=0.0,
        ), patch.object(
            executor, "_preflight", new_callable=AsyncMock,
        ):
            trace = await executor.execute("bad-command", task_desc)

        assert trace.score == 0.0
        assert trace.success is False
        assert "Error" in trace.stderr or "Error" in trace.error_summary

    @pytest.mark.asyncio
    async def test_execute_image_build_error(self, adapter, task_desc):
        """Image build failure -> Trace with score=0."""
        executor = TerminalBenchExecutor(adapter)

        with patch.object(
            adapter, "_ensure_image",
            new_callable=AsyncMock,
            side_effect=RuntimeError("Build failed"),
        ):
            trace = await executor.execute("echo hello", task_desc)

        assert trace.score == 0.0
        assert trace.success is False
        assert "build failed" in trace.error_summary.lower()

    @pytest.mark.asyncio
    async def test_cleanup_on_exception(self, adapter, task_desc):
        """Container teardown must happen even if execution raises."""
        executor = TerminalBenchExecutor(adapter)
        adapter._image_built.add(task_desc.task_id)

        down_called = False

        async def mock_compose(compose_files, project_name, project_dir,
                               command, **kwargs):
            nonlocal down_called
            if command[0] == "down":
                down_called = True
                return _ExecResult(return_code=0)
            if "up" in command:
                raise RuntimeError("Container start failed")
            return _ExecResult(return_code=0)

        with patch(
            "meta_n.integrations.terminal_bench._run_compose_command",
            side_effect=mock_compose,
        ), patch.object(
            executor, "_preflight", new_callable=AsyncMock,
        ):
            trace = await executor.execute("echo hello", task_desc)

        # down should have been called despite the exception
        assert down_called
        assert trace.score == 0.0

    @pytest.mark.asyncio
    async def test_no_rmi_in_any_command(self, adapter, task_desc):
        """Verify --rmi never appears in any compose command."""
        executor = TerminalBenchExecutor(adapter)
        adapter._image_built.add(task_desc.task_id)

        all_commands: list[list[str]] = []

        async def mock_compose(compose_files, project_name, project_dir,
                               command, **kwargs):
            all_commands.append(command)
            return _ExecResult(return_code=0)

        async def mock_exec(*args, **kwargs):
            return _ExecResult(return_code=0)

        with patch(
            "meta_n.integrations.terminal_bench._run_compose_command",
            side_effect=mock_compose,
        ), patch(
            "meta_n.integrations.terminal_bench._compose_exec",
            side_effect=mock_exec,
        ), patch(
            "meta_n.integrations.terminal_bench._read_reward",
            return_value=1.0,
        ), patch.object(
            executor, "_preflight", new_callable=AsyncMock,
        ):
            await executor.execute("echo hello", task_desc)

        for cmd in all_commands:
            assert "--rmi" not in cmd, f"Found --rmi in command: {cmd}"


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

class TestDownload:
    @pytest.mark.asyncio
    async def test_download_skips_existing(self, adapter):
        """download() is a no-op when task_cache_dir already has content."""
        # adapter fixture already points at a dir with tasks
        await adapter.download()  # Should not raise

    @pytest.mark.asyncio
    async def test_download_raises_without_harbor(self):
        """download() raises RuntimeError when harbor is not installed
        and no task_cache_dir is provided."""
        a = TerminalBenchAdapter(task_cache_dir=None)
        with patch.dict(
            "sys.modules", {"harbor.models.job.config": None}
        ):
            # This should raise because task_cache_dir is None and harbor
            # can't be imported (mocked to None)
            with pytest.raises((RuntimeError, ImportError)):
                await a.download()
        a.cleanup()


# ---------------------------------------------------------------------------
# CLI route for --base-solver terminus2 (legacy task.yaml load + litellm route)
# ---------------------------------------------------------------------------

class TestTerminus2CliRoute:
    """Reconcile the meta-n CLI route with the Terminus 2 subprocess runner.

    Two shims make ``meta-n --benchmark terminal_bench --base-solver terminus2``
    work end-to-end against the ``original-tasks`` baseline the runner reads:

    1. ``load_tasks`` falls back to the legacy ``task.yaml`` layout (the harbor
       ``task.toml`` scan finds nothing there) carrying the on-disk folder name
       as ``task_name`` (the field the env provider forwards to the runner).
    2. ``make_agent_backend('terminus2', ...)`` litellm-routes the un-prefixed
       cost-ledger model id so a custom OpenAI-compatible ``api_base`` is used.
    """

    def _write_legacy_task(self, root, name="hello-world"):
        d = root / name
        (d / "tests").mkdir(parents=True)
        (d / "task.yaml").write_text(
            "instruction: |-\n"
            "  Create /app/hello.txt with 'Hello, world!'.\n"
            "difficulty: easy\n"
            "category: file-operations\n"
            "max_agent_timeout_sec: 900.0\n"
            "max_test_timeout_sec: 180.0\n"
        )
        (d / "Dockerfile").write_text("FROM scratch\n")
        return d

    def test_load_tasks_legacy_yaml_layout(self, tmp_path):
        self._write_legacy_task(tmp_path)
        adapter = TerminalBenchAdapter(task_cache_dir=str(tmp_path))
        tasks = adapter.load_tasks()
        adapter.cleanup()
        assert len(tasks) == 1
        t = tasks[0]
        assert t.task_id == "hello-world"
        # The on-disk folder name is load-bearing: the env provider forwards it
        # to the runner as the runner_task_id.
        assert t.metadata["task_name"] == "hello-world"
        assert t.metadata["layout"] == "legacy_yaml"
        assert "Hello, world!" in t.description

    def test_env_provider_runner_task_id_from_legacy_metadata(self, tmp_path):
        self._write_legacy_task(tmp_path)
        adapter = TerminalBenchAdapter(task_cache_dir=str(tmp_path))
        tasks = adapter.load_tasks()
        provider = adapter.make_env_provider("terminus2")
        adapter.cleanup()
        assert provider._runner_task_id(tasks[0]) == "hello-world"

    def test_load_tasks_name_filter_and_limit_legacy(self, tmp_path):
        for n in ("aaa", "bbb", "ccc"):
            self._write_legacy_task(tmp_path, name=n)
        adapter = TerminalBenchAdapter(
            task_cache_dir=str(tmp_path), task_names=["bbb", "ccc"]
        )
        tasks = adapter.load_tasks(limit=1)
        adapter.cleanup()
        assert len(tasks) == 1
        assert tasks[0].task_id in {"bbb", "ccc"}

    def test_make_agent_backend_litellm_routes_unprefixed_model(self, tmp_path):
        self._write_legacy_task(tmp_path)
        adapter = TerminalBenchAdapter(task_cache_dir=str(tmp_path))
        backend = adapter.make_agent_backend(
            "terminus2",
            model="google/gemma-4-31b-qat",
            api_base="http://127.0.0.1:1234/v1",
            api_key="dummy",
            provider_env_var="OPENAI_API_KEY",
        )
        adapter.cleanup()
        # Un-prefixed pricing key -> litellm-routed openai/ form for the runner.
        assert backend._model == "openai/google/gemma-4-31b-qat"

    def test_make_agent_backend_preserves_existing_provider_prefix(self, tmp_path):
        adapter = TerminalBenchAdapter(task_cache_dir=str(tmp_path))
        backend = adapter.make_agent_backend(
            "terminus2",
            model="openai/google/gemma-4-31b-qat",
            api_base="http://127.0.0.1:1234/v1",
        )
        adapter.cleanup()
        # Idempotent: an already-routed model is left untouched.
        assert backend._model == "openai/google/gemma-4-31b-qat"
