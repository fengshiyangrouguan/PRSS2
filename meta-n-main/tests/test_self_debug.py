"""Tests for the self-debug retry loop."""

import pytest

from meta_n.core.meta_layer import InjectedCode, MetaLayer, TaskDescription, Trace


class MockSolver:
    """Mock solver that returns a fixed script."""

    def __init__(self, script: str = "echo hello"):
        self.script = script
        self.call_count = 0
        self.last_additional_context = ""

    async def solve(self, task: TaskDescription, additional_context: str = "") -> tuple[str, str, int]:
        self.call_count += 1
        self.last_additional_context = additional_context
        return self.script, f"reasoning #{self.call_count}", 100


class MockExecutor:
    """Mock executor that returns configurable scores per call."""

    def __init__(self, scores: list[float]):
        self.scores = scores
        self.call_count = 0

    async def execute(self, script: str, task: TaskDescription, timeout: int = 30) -> Trace:
        idx = min(self.call_count, len(self.scores) - 1)
        score = self.scores[idx]
        self.call_count += 1
        return Trace(
            task_id=task.task_id,
            script=script,
            score=score,
            success=score > 0.0,
            stderr="error output" if score < 0.5 else "",
            error_summary="something went wrong" if score < 0.5 else "",
        )


@pytest.fixture
def task():
    return TaskDescription(task_id="test_retry", description="Test task")


class TestSelfDebug:
    @pytest.mark.asyncio
    async def test_retry_disabled(self, task):
        """max_retries=0 should not retry."""
        executor = MockExecutor(scores=[0.1])
        solver = MockSolver(script="bad code")
        layer = MetaLayer(
            depth=2,
            injected_code=InjectedCode(),
            inner_solver=solver,
            executor=executor,
            max_retries=0,
            retry_threshold=0.5,
        )
        trace, tokens = await layer.execute(task)
        assert trace.score == 0.1
        assert solver.call_count == 1
        assert executor.call_count == 1

    @pytest.mark.asyncio
    async def test_no_retry_above_threshold(self, task):
        """Score above threshold should not trigger retry."""
        executor = MockExecutor(scores=[0.7])
        solver = MockSolver(script="ok code")
        layer = MetaLayer(
            depth=2,
            injected_code=InjectedCode(),
            inner_solver=solver,
            executor=executor,
            max_retries=2,
            retry_threshold=0.5,
        )
        trace, tokens = await layer.execute(task)
        assert trace.score == 0.7
        assert solver.call_count == 1

    @pytest.mark.asyncio
    async def test_retry_improves(self, task):
        """Retry should keep the improved trace."""
        executor = MockExecutor(scores=[0.2, 0.8])
        solver = MockSolver(script="code")
        layer = MetaLayer(
            depth=2,
            injected_code=InjectedCode(),
            inner_solver=solver,
            executor=executor,
            max_retries=1,
            retry_threshold=0.5,
        )
        trace, tokens = await layer.execute(task)
        assert trace.score == 0.8
        assert solver.call_count == 2
        assert executor.call_count == 2

    @pytest.mark.asyncio
    async def test_retry_no_improvement(self, task):
        """If retry doesn't improve, keep original (best)."""
        executor = MockExecutor(scores=[0.3, 0.2])
        solver = MockSolver(script="code")
        layer = MetaLayer(
            depth=2,
            injected_code=InjectedCode(),
            inner_solver=solver,
            executor=executor,
            max_retries=1,
            retry_threshold=0.5,
        )
        trace, tokens = await layer.execute(task)
        assert trace.score == 0.3  # kept original

    @pytest.mark.asyncio
    async def test_retry_keeps_best_across_multiple(self, task):
        """With multiple retries, keep the best score seen."""
        executor = MockExecutor(scores=[0.1, 0.4, 0.2])
        solver = MockSolver(script="code")
        layer = MetaLayer(
            depth=2,
            injected_code=InjectedCode(),
            inner_solver=solver,
            executor=executor,
            max_retries=2,
            retry_threshold=0.5,
        )
        trace, tokens = await layer.execute(task)
        assert trace.score == 0.4  # best of 0.1, 0.4, 0.2

    @pytest.mark.asyncio
    async def test_retry_stops_at_threshold(self, task):
        """Should stop retrying once threshold is reached."""
        executor = MockExecutor(scores=[0.2, 0.6, 0.9])  # 3rd shouldn't be reached
        solver = MockSolver(script="code")
        layer = MetaLayer(
            depth=2,
            injected_code=InjectedCode(),
            inner_solver=solver,
            executor=executor,
            max_retries=3,
            retry_threshold=0.5,
        )
        trace, tokens = await layer.execute(task)
        assert trace.score == 0.6
        assert solver.call_count == 2  # initial + 1 retry (stopped at threshold)
        assert executor.call_count == 2

    @pytest.mark.asyncio
    async def test_tokens_accumulated(self, task):
        """Total tokens should include all retry attempts."""
        executor = MockExecutor(scores=[0.1, 0.2])
        solver = MockSolver(script="code")  # returns 100 tokens each call
        layer = MetaLayer(
            depth=2,
            injected_code=InjectedCode(),
            inner_solver=solver,
            executor=executor,
            max_retries=1,
            retry_threshold=0.5,
        )
        trace, tokens = await layer.execute(task)
        assert tokens == 200  # 100 initial + 100 retry

    @pytest.mark.asyncio
    async def test_debug_context_includes_error(self, task):
        """Debug context should contain error info from failed trace."""
        executor = MockExecutor(scores=[0.1, 0.8])
        solver = MockSolver(script="code")
        layer = MetaLayer(
            depth=2,
            injected_code=InjectedCode(),
            inner_solver=solver,
            executor=executor,
            max_retries=1,
            retry_threshold=0.5,
        )
        await layer.execute(task)
        # The retry call should have received debug context
        ctx = solver.last_additional_context
        assert "Self-Debug Round 1" in ctx
        assert "scored 0.100" in ctx
        assert "error output" in ctx or "something went wrong" in ctx

    @pytest.mark.asyncio
    async def test_max_retries_respected(self, task):
        """Should not exceed max_retries attempts."""
        executor = MockExecutor(scores=[0.1, 0.1, 0.1, 0.1, 0.1])
        solver = MockSolver(script="code")
        layer = MetaLayer(
            depth=2,
            injected_code=InjectedCode(),
            inner_solver=solver,
            executor=executor,
            max_retries=2,
            retry_threshold=0.5,
        )
        trace, tokens = await layer.execute(task)
        assert solver.call_count == 3  # initial + 2 retries
        assert executor.call_count == 3
