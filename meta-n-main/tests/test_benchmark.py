"""Tests for benchmark abstraction and CO-Bench adapter."""

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from meta_n.core.meta_layer import TaskDescription, Trace
from meta_n.integrations.benchmark import BenchmarkAdapter, EvalResult
from meta_n.integrations.co_bench import (
    CO_BENCH_TASKS,
    COBenchAdapter,
    COBenchExecutor,
    _task_id_from_name,
)


# --- EvalResult tests ---


class TestEvalResult:
    def test_success(self):
        r = EvalResult(success=True, score=0.85, feedback="Good", raw_score=42.0)
        assert r.success
        assert r.score == 0.85
        assert r.raw_score == 42.0

    def test_failure(self):
        r = EvalResult(success=False, feedback="Error: timeout")
        assert not r.success
        assert r.score == 0.0


# --- Helper tests ---


class TestHelpers:
    def test_task_id_from_name(self):
        assert _task_id_from_name("Travelling salesman problem") == "travelling_salesman_problem"
        assert _task_id_from_name("Bin packing") == "bin_packing"
        assert _task_id_from_name("p-median problems") == "p_median_problems"

    def test_all_tasks_listed(self):
        assert len(CO_BENCH_TASKS) == 36
        assert "Bin packing - one-dimensional" in CO_BENCH_TASKS
        assert "Travelling salesman problem" in CO_BENCH_TASKS


# --- COBenchAdapter tests (with fake data dir) ---


class TestCOBenchAdapter:
    @pytest.fixture
    def fake_data_dir(self):
        """Create a minimal fake CO-Bench data directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a fake "Bin packing" task
            task_dir = Path(tmpdir) / "Bin packing"
            task_dir.mkdir()

            config_py = task_dir / "config.py"
            config_py.write_text('''
DESCRIPTION = """The Bin Packing Problem: Given a set of items with weights and a bin capacity, find the minimum number of bins needed to pack all items."""

def load_data(path):
    return {"items": [3, 5, 2, 7, 1], "capacity": 10}

def eval_func(items, capacity, bins, **kwargs):
    return len(bins)

def solve(**kwargs):
    """
    Args:
        items (list): List of item weights
        capacity (int): Bin capacity
    Returns:
        dict: {'bins': [[item_indices], ...]}
    """
    items = kwargs['items']
    return {'bins': [[i] for i in range(len(items))]}
''')

            # Create a fake test case directory
            (task_dir / "test_0").mkdir()

            yield tmpdir

    def test_load_tasks(self, fake_data_dir):
        adapter = COBenchAdapter(
            data_dir=fake_data_dir,
            task_names=["Bin packing"],
        )
        tasks = adapter.load_tasks()

        assert len(tasks) == 1
        task = tasks[0]
        assert task.task_id == "bin_packing"
        assert "Bin Packing" in task.description
        assert task.metadata["benchmark"] == "co_bench"
        assert task.metadata["solution_language"] == "python"
        assert "solve" in task.metadata["solve_template"]

    def test_load_tasks_with_limit(self, fake_data_dir):
        adapter = COBenchAdapter(
            data_dir=fake_data_dir,
            task_names=["Bin packing"],
        )
        tasks = adapter.load_tasks(limit=0)
        assert len(tasks) == 0

    def test_load_tasks_missing_dir(self):
        adapter = COBenchAdapter(
            data_dir="/nonexistent",
            task_names=["Bin packing"],
        )
        tasks = adapter.load_tasks()
        assert len(tasks) == 0

    def test_properties(self):
        adapter = COBenchAdapter()
        assert adapter.name == "co_bench"

    def test_extract_solve_template(self, fake_data_dir):
        adapter = COBenchAdapter(data_dir=fake_data_dir, task_names=["Bin packing"])
        config_path = Path(fake_data_dir) / "Bin packing" / "config.py"
        template = adapter._extract_solve_template(config_path)
        assert "def solve(" in template
        assert "kwargs" in template

    def test_load_task_data_caching(self, fake_data_dir):
        adapter = COBenchAdapter(data_dir=fake_data_dir, task_names=["Bin packing"])
        data1 = adapter._load_task_data("Bin packing")
        data2 = adapter._load_task_data("Bin packing")
        assert data1 is data2  # same object (cached)


# --- COBenchExecutor tests ---


class TestCOBenchExecutor:
    @pytest.mark.asyncio
    async def test_execute_success(self):
        adapter = MagicMock()
        adapter.evaluate = AsyncMock(
            return_value=EvalResult(success=True, score=0.75, raw_score=3.0, feedback="OK")
        )

        executor = COBenchExecutor(adapter)
        task = TaskDescription(
            task_id="bin_packing",
            description="Pack items",
            metadata={"task_name": "Bin packing"},
        )

        trace = await executor.execute("def solve(**kwargs): ...", task)

        assert trace.success is True
        assert trace.score == 0.75
        assert "0.7500" in trace.stdout
        assert trace.exit_code == 0
        assert trace.duration_s > 0

    @pytest.mark.asyncio
    async def test_execute_failure(self):
        adapter = MagicMock()
        adapter.evaluate = AsyncMock(
            return_value=EvalResult(success=False, score=0.0, feedback="Runtime error: division by zero")
        )

        executor = COBenchExecutor(adapter)
        task = TaskDescription(
            task_id="tsp",
            description="Solve TSP",
            metadata={"task_name": "Travelling salesman problem"},
        )

        trace = await executor.execute("def solve(**kwargs): 1/0", task)

        assert trace.success is False
        assert trace.score == 0.0
        assert trace.exit_code == 1
        assert "division by zero" in trace.stderr

    @pytest.mark.asyncio
    async def test_execute_partial_score(self):
        """A solution that runs but scores poorly."""
        adapter = MagicMock()
        adapter.evaluate = AsyncMock(
            return_value=EvalResult(success=True, score=0.15, raw_score=1.5, feedback="Suboptimal")
        )

        executor = COBenchExecutor(adapter)
        task = TaskDescription(
            task_id="bin_packing",
            description="Pack items",
            metadata={"task_name": "Bin packing"},
        )

        trace = await executor.execute("def solve(**kwargs): ...", task)
        assert trace.success is True
        assert trace.score == 0.15


# --- Trace score field tests ---


class TestTraceScore:
    def test_trace_default_score(self):
        t = Trace(task_id="t1")
        assert t.score == 0.0

    def test_trace_with_score(self):
        t = Trace(task_id="t1", success=True, score=0.85)
        assert t.score == 0.85

    def test_trace_serialization_includes_score(self):
        t = Trace(task_id="t1", score=0.42, script="x")
        data = t.model_dump()
        assert data["score"] == 0.42

        restored = Trace(**data)
        assert restored.score == 0.42
