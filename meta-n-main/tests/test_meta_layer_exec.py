"""Tests for MetaLayer execution logic."""

import pytest

from meta_n.core.base_executor import LocalExecutor
from meta_n.core.meta_layer import InjectedCode, MetaLayer, TaskDescription


class MockSolver:
    """Mock solver that returns a fixed script."""

    def __init__(self, script: str = "echo hello"):
        self.script = script
        self.last_additional_context = ""

    async def solve(self, task: TaskDescription, additional_context: str = "") -> tuple[str, str, int]:
        self.last_additional_context = additional_context
        return self.script, "mock reasoning", 50


@pytest.fixture
def executor():
    return LocalExecutor()


@pytest.fixture
def task():
    return TaskDescription(
        task_id="test_001",
        description="Create hello.txt",
        verification_script="test -f /tmp/meta_layer_test.txt",
    )


class TestMetaLayerExecution:
    @pytest.mark.asyncio
    async def test_execute_no_injection(self, executor, task):
        solver = MockSolver(script="touch /tmp/meta_layer_test.txt")
        code = InjectedCode()  # empty
        layer = MetaLayer(depth=2, injected_code=code, inner_solver=solver, executor=executor)

        trace, tokens = await layer.execute(task)
        assert trace.success is True
        assert trace.depth == 2
        assert tokens == 50

    @pytest.mark.asyncio
    async def test_execute_with_pre_process(self, executor, task):
        solver = MockSolver(script="touch /tmp/meta_layer_test.txt")
        code = InjectedCode(
            pre_process="additional_context = 'Remember to use touch command'",
        )
        layer = MetaLayer(depth=2, injected_code=code, inner_solver=solver, executor=executor)

        trace, tokens = await layer.execute(task)
        assert solver.last_additional_context == "Remember to use touch command"
        assert trace.success is True

    @pytest.mark.asyncio
    async def test_execute_pre_process_error_graceful(self, executor):
        """Pre-process errors should not crash execution."""
        task = TaskDescription(task_id="t1", description="test")
        solver = MockSolver(script="echo still works")
        code = InjectedCode(
            pre_process="raise ValueError('broken')",
        )
        layer = MetaLayer(depth=2, injected_code=code, inner_solver=solver, executor=executor)

        trace, _ = await layer.execute(task)
        # Should still execute despite pre_process error
        assert trace.stdout.strip() == "still works"

class TestMetaLayerSafety:
    """Safety validation integration tests."""

    @pytest.mark.asyncio
    async def test_dangerous_pre_process_rejected(self, executor):
        """pre_process with dangerous imports is skipped."""
        task = TaskDescription(task_id="t1", description="test")
        solver = MockSolver(script="echo safe")
        code = InjectedCode(
            pre_process="import os; additional_context = os.environ.get('SECRET', '')",
        )
        layer = MetaLayer(depth=2, injected_code=code, inner_solver=solver, executor=executor)

        trace, _ = await layer.execute(task)
        # Should still run fine — dangerous pre_process is skipped
        assert trace.stdout.strip() == "safe"
        assert solver.last_additional_context == ""  # pre_process was skipped

    @pytest.mark.asyncio
    async def test_safe_code_still_works(self, executor):
        """Safe injected code executes normally."""
        task = TaskDescription(task_id="t1", description="compute something")
        solver = MockSolver(script="echo placeholder")
        code = InjectedCode(
            pre_process="additional_context = 'hint: use arithmetic'",
        )
        layer = MetaLayer(depth=2, injected_code=code, inner_solver=solver, executor=executor)

        trace, _ = await layer.execute(task)
        assert solver.last_additional_context == "hint: use arithmetic"
