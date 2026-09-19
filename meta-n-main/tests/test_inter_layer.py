"""Tests for inter-layer communication via outer_context."""

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
def task():
    return TaskDescription(task_id="test_inter", description="Test assignment problem")


@pytest.fixture
def executor():
    return LocalExecutor()


class TestOuterContext:
    @pytest.mark.asyncio
    async def test_outer_context_passed_to_inner(self, task, executor):
        """Inner MetaLayer's pre_process should see outer_context from outer layer."""
        inner_code = InjectedCode(
            pre_process=(
                'additional_context = f"inner saw: {outer_context[:20]}"'
            ),
        )
        outer_code = InjectedCode(
            pre_process='additional_context = "STRATEGY: use greedy"',
        )

        solver = MockSolver(script="echo done")
        inner_layer = MetaLayer(depth=2, injected_code=inner_code, inner_solver=solver, executor=executor)
        outer_layer = MetaLayer(depth=3, injected_code=outer_code, inner_solver=inner_layer, executor=executor)

        await outer_layer.execute(task)
        # The solver should see both contexts
        ctx = solver.last_additional_context
        assert "STRATEGY: use greedy" in ctx
        assert "inner saw: STRATEGY: use gre" in ctx

    @pytest.mark.asyncio
    async def test_outer_context_empty_at_outermost(self, task, executor):
        """Outermost MetaLayer.execute() should have empty outer_context."""
        code = InjectedCode(
            pre_process='additional_context = f"outer_ctx_len={len(outer_context)}"',
        )
        solver = MockSolver(script="echo done")
        layer = MetaLayer(depth=2, injected_code=code, inner_solver=solver, executor=executor)

        await layer.execute(task)
        assert "outer_ctx_len=0" in solver.last_additional_context

    @pytest.mark.asyncio
    async def test_inner_adapts_to_outer(self, task, executor):
        """Inner layer should change behavior based on outer_context."""
        inner_code = InjectedCode(
            pre_process=(
                'if "greedy" in outer_context.lower():\n'
                '    additional_context = "TACTIC: sort by cost, assign greedily"\n'
                'else:\n'
                '    additional_context = "TACTIC: use default approach"'
            ),
        )
        outer_code_greedy = InjectedCode(
            pre_process='additional_context = "STRATEGY: greedy approach"',
        )
        outer_code_exact = InjectedCode(
            pre_process='additional_context = "STRATEGY: exact algorithm"',
        )

        # Test with greedy strategy
        solver = MockSolver(script="echo done")
        inner = MetaLayer(depth=2, injected_code=inner_code, inner_solver=solver, executor=executor)
        outer = MetaLayer(depth=3, injected_code=outer_code_greedy, inner_solver=inner, executor=executor)
        await outer.execute(task)
        assert "sort by cost, assign greedily" in solver.last_additional_context

        # Test with exact strategy — same inner code, different outer
        solver2 = MockSolver(script="echo done")
        inner2 = MetaLayer(depth=2, injected_code=inner_code, inner_solver=solver2, executor=executor)
        outer2 = MetaLayer(depth=3, injected_code=outer_code_exact, inner_solver=inner2, executor=executor)
        await outer2.execute(task)
        assert "use default approach" in solver2.last_additional_context

    @pytest.mark.asyncio
    async def test_backward_compat_no_outer_context_reference(self, task, executor):
        """Old pre_process code that doesn't reference outer_context should still work."""
        old_code = InjectedCode(
            pre_process='additional_context = "simple hint"',
        )
        solver = MockSolver(script="echo done")
        layer = MetaLayer(depth=2, injected_code=old_code, inner_solver=solver, executor=executor)

        await layer.execute(task)
        assert "simple hint" in solver.last_additional_context

    @pytest.mark.asyncio
    async def test_three_layer_context_chain(self, task, executor):
        """Three-layer nesting: outer_context accumulates through the chain."""
        d2_code = InjectedCode(
            pre_process='additional_context = f"d2 sees {len(outer_context)} chars"',
        )
        d3_code = InjectedCode(
            pre_process='additional_context = "d3 strategy"',
        )
        d4_code = InjectedCode(
            pre_process='additional_context = "d4 meta-strategy"',
        )

        solver = MockSolver(script="echo done")
        d2 = MetaLayer(depth=2, injected_code=d2_code, inner_solver=solver, executor=executor)
        d3 = MetaLayer(depth=3, injected_code=d3_code, inner_solver=d2, executor=executor)
        d4 = MetaLayer(depth=4, injected_code=d4_code, inner_solver=d3, executor=executor)

        await d4.execute(task)
        ctx = solver.last_additional_context
        # d4's context should be present
        assert "d4 meta-strategy" in ctx
        # d3's context should be present
        assert "d3 strategy" in ctx
        # d2 should have seen accumulated outer_context (d4 + d3)
        assert "d2 sees" in ctx
