"""Tests for the base executor."""

import pytest

from meta_n.core.base_executor import LocalExecutor
from meta_n.core.meta_layer import TaskDescription


@pytest.fixture
def executor():
    return LocalExecutor()


@pytest.fixture
def simple_task():
    return TaskDescription(task_id="test_001", description="Test task")


class TestLocalExecutor:
    @pytest.mark.asyncio
    async def test_echo(self, executor, simple_task):
        trace = await executor.execute("echo hello", simple_task)
        assert trace.task_id == "test_001"
        assert trace.stdout.strip() == "hello"
        assert trace.exit_code == 0
        assert trace.success is True
        assert trace.duration_s > 0

    @pytest.mark.asyncio
    async def test_failure(self, executor, simple_task):
        trace = await executor.execute("exit 1", simple_task)
        assert trace.exit_code == 1
        assert trace.success is False

    @pytest.mark.asyncio
    async def test_stderr(self, executor, simple_task):
        trace = await executor.execute("echo error >&2; exit 1", simple_task)
        assert "error" in trace.stderr
        assert trace.success is False
        assert trace.error_summary != ""

    @pytest.mark.asyncio
    async def test_timeout(self, executor, simple_task):
        trace = await executor.execute("sleep 10", simple_task, timeout=1)
        assert trace.success is False
        assert "timed out" in trace.error_summary.lower()

    @pytest.mark.asyncio
    async def test_multiline_script(self, executor, simple_task):
        script = "#!/bin/bash\nset -e\nX=42\necho $X"
        trace = await executor.execute(script, simple_task)
        assert trace.stdout.strip() == "42"
        assert trace.success is True

    @pytest.mark.asyncio
    async def test_verification_script_pass(self, executor):
        task = TaskDescription(
            task_id="verify_pass",
            description="Create a file",
            verification_script="test -f /tmp/test_executor_verify.txt",
        )
        trace = await executor.execute(
            "echo 'hi' > /tmp/test_executor_verify.txt", task
        )
        assert trace.success is True

    @pytest.mark.asyncio
    async def test_verification_script_fail(self, executor):
        task = TaskDescription(
            task_id="verify_fail",
            description="Create a file",
            verification_script="test -f /tmp/nonexistent_file_xyz.txt",
        )
        trace = await executor.execute("echo 'did not create the file'", task)
        assert trace.success is False

    @pytest.mark.asyncio
    async def test_code_hash(self, executor, simple_task):
        trace = await executor.execute("echo test", simple_task)
        assert len(trace.code_hash) == 16
