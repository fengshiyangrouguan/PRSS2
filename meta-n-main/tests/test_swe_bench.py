"""Tests for SWE-bench Verified adapter.

Mirrors tests/test_terminal_bench.py patterns. SWE-bench's runtime
(TerminalBenchExecutor) is already covered by test_terminal_bench.py;
this file only tests the SWE-bench-specific overrides:
  - _parse_memory_string conversion helper
  - SWEBenchVerifiedAdapter.name + dataset_name
  - _parse_task_toml memory-string handling
  - load_tasks metadata augmentation from tests/config.json
  - get_language_instructions SWE-bench branch
"""

import json

import pytest

from meta_n.core.agentic_prompts import (
    LANG_INSTRUCTIONS_BASH,
    LANG_INSTRUCTIONS_BASH_SWEBENCH,
    get_language_instructions,
)
from meta_n.integrations.swe_bench import (
    SWEBenchVerifiedAdapter,
    _parse_memory_string,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_swebench_task_dir(tmp_path):
    """Create a minimal SWE-bench Verified task directory matching what
    harbor's swebench adapter generates."""
    task_dir = tmp_path / "django__django-11265"
    task_dir.mkdir()

    # task.toml — SWE-bench-style: memory='4G' string, storage='10G', long timeouts
    (task_dir / "task.toml").write_text(
        '[metadata]\n'
        'author_name = "unknown"\n'
        'author_email = "unknown"\n'
        'difficulty = "15 min - 1 hour"\n'
        'category = "debugging"\n'
        'tags = ["debugging", "swe-bench"]\n'
        '\n'
        '[verifier]\n'
        'timeout_sec = 3000\n'
        '\n'
        '[agent]\n'
        'timeout_sec = 3000\n'
        '\n'
        '[environment]\n'
        'build_timeout_sec = 1800.0\n'
        'cpus = 1\n'
        "memory = '4G'\n"
        "storage = '10G'\n"
    )

    (task_dir / "instruction.md").write_text(
        "Using exclude on annotated FilteredRelation doesn't work\n"
        "Description\n\nIt looks like using exclude on queryset...\n"
    )

    env_dir = task_dir / "environment"
    env_dir.mkdir()
    (env_dir / "Dockerfile").write_text(
        "FROM swebench/sweb.eval.x86_64.django_1776_django-11265:latest\n"
        "RUN mkdir -p /logs\n"
    )

    tests_dir = task_dir / "tests"
    tests_dir.mkdir()
    (tests_dir / "test.sh").write_text(
        '#!/bin/bash\nexit 0\n'
    )
    (tests_dir / "config.json").write_text(json.dumps({
        "instance_id": "django__django-11265",
        "repo": "django/django",
        "base_commit": "1234abcd",
        "version": "3.0",
        "FAIL_TO_PASS": ["tests/test_a.py::test_one", "tests/test_a.py::test_two"],
        "PASS_TO_PASS": [f"tests/test_b.py::test_{i}" for i in range(5)],
        "patch": "diff --git a/file.py b/file.py\n",
        "test_patch": "diff --git a/tests.py b/tests.py\n",
        "problem_statement": "Issue body",
        "difficulty": "15 min - 1 hour",
        "hints_text": "",
        "created_at": "2019-04-23T13:48:13Z",
        "environment_setup_commit": "1234abcd",
    }))

    sol_dir = task_dir / "solution"
    sol_dir.mkdir()
    (sol_dir / "solve.sh").write_text("# oracle\n")

    return task_dir


@pytest.fixture
def adapter(fake_swebench_task_dir):
    """SWEBenchVerifiedAdapter pointed at the fake task's parent dir."""
    a = SWEBenchVerifiedAdapter(task_cache_dir=str(fake_swebench_task_dir.parent))
    yield a
    a.cleanup()


# ---------------------------------------------------------------------------
# Memory string parser
# ---------------------------------------------------------------------------

class TestParseMemoryString:
    @pytest.mark.parametrize("value,expected", [
        ("4G", 4096),
        ("4Gi", 4096),
        ("8G", 8192),
        ("512M", 512),
        ("512Mi", 512),
        ("2g", 2048),       # case insensitive
        ("2.5G", 2560),     # fractional
        ("1024", 1024),     # bare number = MB
        (1024, 1024),       # passthrough int
        (1024.0, 1024),     # passthrough float
        ("1024M", 1024),
        ("  4G  ", 4096),   # whitespace OK
    ])
    def test_valid(self, value, expected):
        assert _parse_memory_string(value) == expected

    def test_none_returns_default(self):
        assert _parse_memory_string(None) == 2048
        assert _parse_memory_string(None, default_mb=8192) == 8192

    def test_garbage_returns_default(self):
        assert _parse_memory_string("garbage") == 2048
        assert _parse_memory_string("garbage", default_mb=1024) == 1024
        assert _parse_memory_string("") == 2048


# ---------------------------------------------------------------------------
# Adapter properties and task loading
# ---------------------------------------------------------------------------

class TestSWEBenchVerifiedAdapter:
    def test_properties(self, adapter):
        assert adapter.name == "swe_bench_verified"
        assert adapter._dataset_name == "swebench-verified"

    def test_load_tasks_metadata(self, adapter):
        tasks = adapter.load_tasks()
        assert len(tasks) == 1
        task = tasks[0]
        # Inherited metadata (with parent fixes applied)
        assert task.metadata["benchmark"] == "swe_bench_verified"  # parent fix A
        assert task.metadata["task_name"] == "django__django-11265"
        assert task.metadata["solution_language"] == "bash"
        assert task.metadata["cpus"] == 1
        assert task.metadata["memory_mb"] == 4096  # parsed from '4G'
        assert task.metadata["timeout_sec"] == 3000
        assert task.metadata["verifier_timeout_sec"] == 3000
        assert task.metadata["build_timeout_sec"] == 1800  # parent fix B
        # SWE-bench-specific metadata from tests/config.json
        assert task.metadata["instance_id"] == "django__django-11265"
        assert task.metadata["repo"] == "django/django"
        assert task.metadata["base_commit"] == "1234abcd"
        assert task.metadata["version"] == "3.0"
        assert task.metadata["fail_to_pass_count"] == 2
        assert task.metadata["pass_to_pass_count"] == 5
        # Env-notes contract — adapter owns its env hints, Ω reads them generically
        env_notes = task.metadata.get("omega_env_notes")
        assert isinstance(env_notes, str) and env_notes
        # Spot-check the constraints we want Ω to honor
        assert "Do NOT create new" in env_notes and "venv" in env_notes
        assert "ripgrep" in env_notes or "`rg`" in env_notes
        assert "/testbed" in env_notes
        assert "conda" in env_notes.lower()

    def test_memory_mb_wins_when_both_present(self, tmp_path):
        """Explicit memory_mb should take precedence over memory string."""
        task_dir = tmp_path / "task-x"
        task_dir.mkdir()
        (task_dir / "task.toml").write_text(
            "[environment]\n"
            "memory_mb = 8192\n"
            "memory = '4G'\n"
        )
        (task_dir / "instruction.md").write_text("x")
        (task_dir / "environment").mkdir()
        (task_dir / "environment" / "Dockerfile").write_text("FROM ubuntu:22.04\n")
        (task_dir / "tests").mkdir()
        (task_dir / "tests" / "test.sh").write_text("echo ok\n")

        a = SWEBenchVerifiedAdapter(task_cache_dir=str(tmp_path))
        tasks = a.load_tasks()
        assert tasks[0].metadata["memory_mb"] == 8192
        a.cleanup()

    def test_load_tasks_limit(self, tmp_path):
        for i in range(4):
            task_dir = tmp_path / f"task-{i:03d}"
            task_dir.mkdir()
            (task_dir / "task.toml").write_text("[environment]\nmemory = '4G'\n")
            (task_dir / "instruction.md").write_text(f"Task {i}")
            (task_dir / "environment").mkdir()
            (task_dir / "environment" / "Dockerfile").write_text("FROM x\n")
            (task_dir / "tests").mkdir()
            (task_dir / "tests" / "test.sh").write_text("echo ok\n")

        a = SWEBenchVerifiedAdapter(task_cache_dir=str(tmp_path))
        tasks = a.load_tasks(limit=2)
        assert len(tasks) == 2
        a.cleanup()

    def test_load_tasks_filter(self, tmp_path):
        for name in ["sympy__sympy-1", "django__django-2", "sympy__sympy-3"]:
            task_dir = tmp_path / name
            task_dir.mkdir()
            (task_dir / "task.toml").write_text("[environment]\nmemory = '4G'\n")
            (task_dir / "instruction.md").write_text("x")
            (task_dir / "environment").mkdir()
            (task_dir / "environment" / "Dockerfile").write_text("FROM x\n")
            (task_dir / "tests").mkdir()
            (task_dir / "tests" / "test.sh").write_text("echo ok\n")

        a = SWEBenchVerifiedAdapter(
            task_cache_dir=str(tmp_path),
            task_names=["sympy__sympy-1", "sympy__sympy-3"],
        )
        tasks = a.load_tasks()
        names = sorted(t.metadata["task_name"] for t in tasks)
        assert names == ["sympy__sympy-1", "sympy__sympy-3"]
        a.cleanup()

    def test_load_tasks_skips_invalid(self, tmp_path):
        """Tasks missing instruction.md or environment/ are skipped."""
        valid = tmp_path / "valid"
        valid.mkdir()
        (valid / "task.toml").write_text("[environment]\nmemory = '4G'\n")
        (valid / "instruction.md").write_text("ok")
        (valid / "environment").mkdir()
        (valid / "environment" / "Dockerfile").write_text("FROM x\n")
        (valid / "tests").mkdir()
        (valid / "tests" / "test.sh").write_text("echo ok\n")

        broken = tmp_path / "no-instruction"
        broken.mkdir()
        (broken / "task.toml").write_text("[environment]\n")
        (broken / "environment").mkdir()
        (broken / "tests").mkdir()

        a = SWEBenchVerifiedAdapter(task_cache_dir=str(tmp_path))
        tasks = a.load_tasks()
        assert len(tasks) == 1
        assert tasks[0].metadata["task_name"] == "valid"
        a.cleanup()

    def test_config_json_missing_does_not_crash(self, tmp_path):
        """If tests/config.json is absent, load_tasks should still succeed
        with no SWE-bench fields in metadata."""
        task_dir = tmp_path / "task-no-config"
        task_dir.mkdir()
        (task_dir / "task.toml").write_text("[environment]\nmemory = '4G'\n")
        (task_dir / "instruction.md").write_text("x")
        (task_dir / "environment").mkdir()
        (task_dir / "environment" / "Dockerfile").write_text("FROM x\n")
        (task_dir / "tests").mkdir()
        (task_dir / "tests" / "test.sh").write_text("echo ok\n")
        # no config.json

        a = SWEBenchVerifiedAdapter(task_cache_dir=str(tmp_path))
        tasks = a.load_tasks()
        assert len(tasks) == 1
        assert "instance_id" not in tasks[0].metadata
        a.cleanup()

    def test_config_json_malformed_does_not_crash(self, tmp_path):
        """If tests/config.json is unparseable JSON, load_tasks logs a
        warning and continues with no SWE-bench fields in metadata.
        Regression test for the json.JSONDecodeError branch."""
        task_dir = tmp_path / "task-bad-config"
        task_dir.mkdir()
        (task_dir / "task.toml").write_text("[environment]\nmemory = '4G'\n")
        (task_dir / "instruction.md").write_text("x")
        (task_dir / "environment").mkdir()
        (task_dir / "environment" / "Dockerfile").write_text("FROM x\n")
        (task_dir / "tests").mkdir()
        (task_dir / "tests" / "test.sh").write_text("echo ok\n")
        (task_dir / "tests" / "config.json").write_text("{not valid json")

        a = SWEBenchVerifiedAdapter(task_cache_dir=str(tmp_path))
        tasks = a.load_tasks()
        assert len(tasks) == 1
        assert "instance_id" not in tasks[0].metadata
        # Core TB2 metadata should still be present
        assert tasks[0].metadata["memory_mb"] == 4096
        a.cleanup()

    @pytest.mark.asyncio
    async def test_download_stages_multi_hash_paths(self, tmp_path, monkeypatch):
        """Regression test for the dispersal bug: harbor content-addresses
        each task to its own hash dir, so a multi-task download returns
        result.paths spanning DIFFERENT parent dirs. download() must
        symlink-stage them into one location so load_tasks sees them
        uniformly. Without this, only the first task's hash dir is
        captured and the other tasks are silently dropped."""
        from pathlib import PosixPath

        # Two tasks in two SEPARATE hash dirs (mimicking real harbor cache)
        hash_a = tmp_path / "AAAA"
        hash_b = tmp_path / "BBBB"
        hash_a.mkdir()
        hash_b.mkdir()
        task_a = hash_a / "django__django-13741"
        task_b = hash_b / "sympy__sympy-13798"
        for td in (task_a, task_b):
            td.mkdir()
            (td / "task.toml").write_text("[environment]\nmemory = '4G'\n")
            (td / "instruction.md").write_text(f"x: {td.name}")
            (td / "environment").mkdir()
            (td / "environment" / "Dockerfile").write_text(f"FROM x-{td.name}\n")
            (td / "tests").mkdir()
            (td / "tests" / "test.sh").write_text("echo ok\n")

        class _FakeId:
            def __init__(self, p): self.path = PosixPath(p)

        class _FakeTaskConfig:
            def __init__(self, p): self._id = _FakeId(p)
            def get_task_id(self): return self._id

        class _FakeDatasetConfig:
            def __init__(self, name): pass
            async def get_task_configs(self):
                return [
                    _FakeTaskConfig("datasets/swebench-verified/django__django-13741"),
                    _FakeTaskConfig("datasets/swebench-verified/sympy__sympy-13798"),
                ]

        class _FakeResult:
            paths = [task_a, task_b]  # different parent hash dirs

        class _FakeTaskClient:
            async def download_tasks(self, ids):
                return _FakeResult()

        # Route the staging dir to tmp so we don't pollute the user's
        # real ~/.cache/meta_n/ during tests.
        staging_root = tmp_path / "staging_home"
        monkeypatch.setenv("HOME", str(staging_root))

        import sys, types
        fake_models = types.ModuleType("harbor.models.job.config")
        fake_models.DatasetConfig = _FakeDatasetConfig
        fake_client = types.ModuleType("harbor.tasks.client")
        fake_client.TaskClient = _FakeTaskClient
        monkeypatch.setitem(sys.modules, "harbor.models.job.config", fake_models)
        monkeypatch.setitem(sys.modules, "harbor.tasks.client", fake_client)

        a = SWEBenchVerifiedAdapter(task_cache_dir=None)
        await a.download()

        # Critical: cache_dir is the staging dir, BOTH tasks visible
        expected_staging = staging_root / ".cache" / "meta_n" / "swebench_verified"
        assert a._task_cache_dir == expected_staging, (
            f"cache_dir should be {expected_staging}, got {a._task_cache_dir}"
        )
        # Both task names symlinked into staging
        assert (expected_staging / "django__django-13741").is_symlink()
        assert (expected_staging / "sympy__sympy-13798").is_symlink()
        # Symlinks resolve to original harbor hash dirs
        assert (expected_staging / "django__django-13741").resolve() == task_a.resolve()
        assert (expected_staging / "sympy__sympy-13798").resolve() == task_b.resolve()
        # load_tasks sees BOTH tasks (not just the first)
        tasks = a.load_tasks()
        names = sorted(t.metadata["task_name"] for t in tasks)
        assert names == ["django__django-13741", "sympy__sympy-13798"], (
            f"expected both tasks loaded, got {names}"
        )
        a.cleanup()

    @pytest.mark.asyncio
    async def test_download_filter_exact_match_not_substring(self, tmp_path, monkeypatch):
        """task_names filter must match the exact task_name (last path
        segment), NOT do substring search. Regression test for the case
        where a short id like 'django-13741' would otherwise also match
        'django-137410' or any other id containing those characters."""
        from pathlib import Path, PosixPath

        # Route the staging dir to tmp so we don't pollute the user's
        # real ~/.cache/meta_n/ during tests (download() runs end-to-end
        # and _stage_downloaded writes symlinks under Path.home()).
        staging_root = tmp_path / "staging_home"
        monkeypatch.setenv("HOME", str(staging_root))

        fake_hash_dir = tmp_path / "BBBB"
        fake_hash_dir.mkdir()
        # Create both tasks on disk so load_tasks can find them
        for tname in ("django__django-13741", "django__django-137410"):
            td = fake_hash_dir / tname
            td.mkdir()
            (td / "task.toml").write_text("[environment]\nmemory = '4G'\n")
            (td / "instruction.md").write_text("x")
            (td / "environment").mkdir()
            (td / "environment" / "Dockerfile").write_text("FROM x\n")
            (td / "tests").mkdir()
            (td / "tests" / "test.sh").write_text("echo ok\n")

        captured_ids: list = []

        class _FakeId:
            def __init__(self, p): self.path = PosixPath(p)

        class _FakeTaskConfig:
            def __init__(self, p): self._id = _FakeId(p)
            def get_task_id(self): return self._id

        class _FakeDatasetConfig:
            def __init__(self, name): pass
            async def get_task_configs(self):
                return [
                    _FakeTaskConfig("datasets/swebench-verified/django__django-13741"),
                    _FakeTaskConfig("datasets/swebench-verified/django__django-137410"),
                    _FakeTaskConfig("datasets/swebench-verified/sympy__sympy-13798"),
                ]

        class _FakeResult:
            paths = [fake_hash_dir / "django__django-13741"]

        class _FakeTaskClient:
            async def download_tasks(self, ids):
                captured_ids.extend(ids)
                return _FakeResult()

        import sys, types
        fake_models = types.ModuleType("harbor.models.job.config")
        fake_models.DatasetConfig = _FakeDatasetConfig
        fake_client = types.ModuleType("harbor.tasks.client")
        fake_client.TaskClient = _FakeTaskClient
        monkeypatch.setitem(sys.modules, "harbor.models.job.config", fake_models)
        monkeypatch.setitem(sys.modules, "harbor.tasks.client", fake_client)

        a = SWEBenchVerifiedAdapter(
            task_cache_dir=None,
            task_names=["django__django-13741"],  # exactly one task
        )
        await a.download()

        # Filter passed exactly 1 task to download_tasks, not 2 (would
        # be 2 under buggy substring matching since '13741' ⊂ '137410')
        assert len(captured_ids) == 1, (
            f"expected 1 task in filtered download, got {len(captured_ids)}"
        )
        # The staging dir the download used must live under this test's tmp
        # sandbox — never the user's real ~/.cache/meta_n/.
        assert Path.home() == staging_root
        assert a._task_cache_dir == (
            staging_root / ".cache" / "meta_n" / "swebench_verified"
        )
        assert a._task_cache_dir.is_relative_to(tmp_path)
        assert (a._task_cache_dir / "django__django-13741").is_symlink()
        a.cleanup()

    @pytest.mark.asyncio
    async def test_download_filter_no_match_raises(self, tmp_path, monkeypatch):
        """If the task_names filter matches zero configs, download() must
        raise — not silently fall back to downloading all 500. A silent
        fallback wastes disk and confuses the user."""
        from pathlib import PosixPath

        class _FakeId:
            def __init__(self, p): self.path = PosixPath(p)

        class _FakeTaskConfig:
            def __init__(self, p): self._id = _FakeId(p)
            def get_task_id(self): return self._id

        class _FakeDatasetConfig:
            def __init__(self, name): pass
            async def get_task_configs(self):
                return [
                    _FakeTaskConfig("datasets/swebench-verified/django__django-13741"),
                ]

        class _FakeTaskClient:
            async def download_tasks(self, ids):
                raise AssertionError("download_tasks should not be called on empty filter")

        import sys, types
        fake_models = types.ModuleType("harbor.models.job.config")
        fake_models.DatasetConfig = _FakeDatasetConfig
        fake_client = types.ModuleType("harbor.tasks.client")
        fake_client.TaskClient = _FakeTaskClient
        monkeypatch.setitem(sys.modules, "harbor.models.job.config", fake_models)
        monkeypatch.setitem(sys.modules, "harbor.tasks.client", fake_client)

        a = SWEBenchVerifiedAdapter(
            task_cache_dir=None,
            task_names=["does__not-exist-99999"],
        )
        with pytest.raises(RuntimeError, match="matched 0 of"):
            await a.download()
        a.cleanup()

    def test_parse_task_toml_no_environment_section(self, tmp_path):
        """task.toml without [environment] is degenerate but shouldn't crash —
        memory_mb falls back to the parent's default and no SWE-bench-style
        memory string parsing kicks in."""
        a = SWEBenchVerifiedAdapter(task_cache_dir=str(tmp_path))
        toml_path = tmp_path / "task.toml"
        toml_path.write_text("[metadata]\ncategory = 'debugging'\n")

        cfg = a._parse_task_toml(toml_path)
        assert cfg["memory_mb"] == 2048  # parent default
        assert cfg["cpus"] == 1
        assert cfg["category"] == "debugging"
        a.cleanup()


# ---------------------------------------------------------------------------
# get_language_instructions SWE-bench branch
# ---------------------------------------------------------------------------

class TestGetLanguageInstructionsSWEBench:
    def test_swebench_metadata_returns_swebench_block(self):
        out = get_language_instructions(
            "bash", {"benchmark": "swe_bench_verified"}
        )
        assert out is LANG_INSTRUCTIONS_BASH_SWEBENCH
        assert "/testbed" in out
        assert "FAIL_TO_PASS" in out

    def test_no_metadata_returns_generic_bash(self):
        out = get_language_instructions("bash", None)
        assert out is LANG_INSTRUCTIONS_BASH

    def test_empty_metadata_returns_generic_bash(self):
        out = get_language_instructions("bash", {})
        assert out is LANG_INSTRUCTIONS_BASH

    def test_other_benchmark_returns_generic_bash(self):
        out = get_language_instructions(
            "bash", {"benchmark": "terminal_bench"}
        )
        assert out is LANG_INSTRUCTIONS_BASH

    def test_swebench_metadata_but_non_bash_language_falls_back(self):
        """SWE-bench is bash-only; if someone passes python language,
        they get the python prompt regardless of benchmark metadata."""
        from meta_n.core.agentic_prompts import LANG_INSTRUCTIONS_PYTHON
        out = get_language_instructions(
            "python", {"benchmark": "swe_bench_verified"}
        )
        assert out is LANG_INSTRUCTIONS_PYTHON
