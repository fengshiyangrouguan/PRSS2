"""Tests for the ARC-AGI-2 benchmark adapter."""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from meta_n.integrations.arc_agi import (
    ARCAGI2Adapter,
    _build_description,
    pass_at_2_accuracy_multi_test,
    pass_at_2_accuracy_single,
)


# ---------------------------------------------------------------------------
# Test data fixtures
# ---------------------------------------------------------------------------

# Three minimal ARC-AGI-2 tasks. Each follows the official GitHub layout:
# {"train": [{"input", "output"}, ...], "test": [{"input", "output"}, ...]}.

# Task t1: identity transformation. Input == Output for both train and test.
T1 = {
    "train": [
        {"input": [[1, 2], [3, 4]], "output": [[1, 2], [3, 4]]},
        {"input": [[5, 6], [7, 8]], "output": [[5, 6], [7, 8]]},
    ],
    "test": [
        {"input": [[9, 0], [1, 2]], "output": [[9, 0], [1, 2]]},
    ],
}

# Task t2: rotate 90 clockwise.
T2 = {
    "train": [
        # 1 2     3 1
        # 3 4 ->  4 2
        {"input": [[1, 2], [3, 4]], "output": [[3, 1], [4, 2]]},
    ],
    "test": [
        {"input": [[5, 6], [7, 8]], "output": [[7, 5], [8, 6]]},
    ],
}

# Task t3: identity, used in additional tests.
T3 = {
    "train": [{"input": [[0]], "output": [[0]]}],
    "test": [{"input": [[1]], "output": [[1]]}],
}


@pytest.fixture
def fake_arc_dir():
    """Create a fake ARC-AGI-2 data tree with 3 evaluation tasks."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        eval_dir = base / "data" / "evaluation"
        eval_dir.mkdir(parents=True)
        (eval_dir / "t1.json").write_text(json.dumps(T1))
        (eval_dir / "t2.json").write_text(json.dumps(T2))
        (eval_dir / "t3.json").write_text(json.dumps(T3))
        yield tmpdir


# Hand-crafted candidate solutions used across tests.

IDENTITY_SOLUTION = """
import numpy as np
def transform_grid_attempt_1(grid):
    return np.asarray(grid, dtype=np.int32)
def transform_grid_attempt_2(grid):
    return np.asarray(grid, dtype=np.int32)
"""

ROT90_THEN_IDENTITY = """
import numpy as np
def transform_grid_attempt_1(grid):
    return np.rot90(np.asarray(grid), k=-1).astype(np.int32)
def transform_grid_attempt_2(grid):
    return np.asarray(grid, dtype=np.int32)
"""

MISSING_FUNCTION_SOLUTION = """
import numpy as np
def transform_grid_attempt_1(grid):
    return np.asarray(grid, dtype=np.int32)
# transform_grid_attempt_2 deliberately missing
"""

INVALID_RETURN_SOLUTION = """
import numpy as np
def transform_grid_attempt_1(grid):
    return None
def transform_grid_attempt_2(grid):
    return [1, 2, 3]
"""

INFINITE_LOOP_SOLUTION = """
import numpy as np
def transform_grid_attempt_1(grid):
    while True:
        pass
def transform_grid_attempt_2(grid):
    return np.asarray(grid, dtype=np.int32)
"""


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def test_load_tasks_finds_evaluation_files(fake_arc_dir):
    a = ARCAGI2Adapter(data_dir=fake_arc_dir)
    tasks = a.load_tasks()
    assert len(tasks) == 3
    assert sorted(t.task_id for t in tasks) == ["t1", "t2", "t3"]
    # Metadata round-trips the full task dict
    t1 = next(t for t in tasks if t.task_id == "t1")
    assert "task_dict" in t1.metadata
    assert t1.metadata["task_dict"]["train"][0]["input"] == [[1, 2], [3, 4]]


def test_load_tasks_respects_task_ids_filter(fake_arc_dir):
    a = ARCAGI2Adapter(data_dir=fake_arc_dir, task_ids=["t1"])
    tasks = a.load_tasks()
    assert [t.task_id for t in tasks] == ["t1"]


def test_load_tasks_respects_limit(fake_arc_dir):
    a = ARCAGI2Adapter(data_dir=fake_arc_dir)
    tasks = a.load_tasks(limit=2)
    assert len(tasks) == 2


def test_load_tasks_empty_dir_returns_empty():
    with tempfile.TemporaryDirectory() as tmpdir:
        a = ARCAGI2Adapter(data_dir=tmpdir)
        assert a.load_tasks() == []


# ---------------------------------------------------------------------------
# Description renderer — train pairs only
# ---------------------------------------------------------------------------

def test_description_contains_train_pairs_only():
    desc = _build_description(T1)
    # Train pair input "1 2" appears
    assert "1 2" in desc
    # Train pair input from second pair "5 6" appears
    assert "5 6" in desc
    # Test pair input "9 0" must NOT appear
    assert "9 0" not in desc
    # Header indicates demo pair count, not test
    assert "2 total" in desc
    # Contract for the two functions is present
    assert "transform_grid_attempt_1" in desc
    assert "transform_grid_attempt_2" in desc


# ---------------------------------------------------------------------------
# Pass@2 scoring (lifted functions)
# ---------------------------------------------------------------------------

def test_pass_at_2_logic_either_attempt():
    """If attempt_1 fails but attempt_2 matches, pass@2 is 1."""
    gt = np.array([[1, 2], [3, 4]])
    wrong = np.array([[0, 0], [0, 0]])
    pass_, _ = pass_at_2_accuracy_single([wrong, gt], gt)
    assert pass_ == 1
    pass_, _ = pass_at_2_accuracy_single([gt, wrong], gt)
    assert pass_ == 1
    pass_, _ = pass_at_2_accuracy_single([wrong, wrong], gt)
    assert pass_ == 0


def test_pass_at_2_handles_size_mismatch():
    gt = np.array([[1, 2], [3, 4]])
    smaller = np.array([[1]])
    pass_, diags = pass_at_2_accuracy_single([smaller, smaller], gt)
    assert pass_ == 0
    assert diags[0]["size_match"] is False


def test_pass_at_2_handles_none_attempt():
    gt = np.array([[1, 2], [3, 4]])
    pass_, diags = pass_at_2_accuracy_single([None, gt], gt)
    assert pass_ == 1  # second attempt salvages it
    pass_, diags = pass_at_2_accuracy_single([None, None], gt)
    assert pass_ == 0


def test_pass_at_2_multi_test_aggregates():
    gts = [np.array([[1]]), np.array([[2]])]
    attempts = [
        [np.array([[1]]), np.array([[1]])],  # match
        [np.array([[0]]), np.array([[0]])],  # no match
    ]
    passes, _ = pass_at_2_accuracy_multi_test(attempts, gts)
    assert passes == [1, 0]


# ---------------------------------------------------------------------------
# evaluate() / evaluate_test()
# ---------------------------------------------------------------------------

def test_evaluate_perfect_solution_train(fake_arc_dir):
    a = ARCAGI2Adapter(data_dir=fake_arc_dir, task_ids=["t1"])
    [task] = a.load_tasks()
    result = asyncio.run(a.evaluate(task, IDENTITY_SOLUTION))
    assert result.success is True
    assert result.score == pytest.approx(1.0)
    payload = json.loads(result.feedback)
    assert payload["combined_score"] == pytest.approx(1.0)
    assert payload["split"] == "train"


def test_evaluate_test_uses_test_pairs(fake_arc_dir):
    a = ARCAGI2Adapter(data_dir=fake_arc_dir, task_ids=["t1"])
    [task] = a.load_tasks()
    result = asyncio.run(a.evaluate_test(task, IDENTITY_SOLUTION))
    assert result.score == pytest.approx(1.0)
    assert json.loads(result.feedback)["split"] == "test"


def test_train_test_split_is_real(fake_arc_dir):
    """A solution that solves train but not test should show the gap.

    On task t2 (rotate 90 CW), `ROT90_THEN_IDENTITY` solves train via
    attempt_1 and is also correct on test. We construct a deliberately
    failing solution to verify the split: identity-only on a non-identity
    task fails BOTH train and test (score=0).
    """
    a = ARCAGI2Adapter(data_dir=fake_arc_dir, task_ids=["t2"])
    [task] = a.load_tasks()
    # Identity-only on rotate-90 task: attempt_2 (identity) fails, attempt_1 (also identity) fails
    solution_identity = """
import numpy as np
def transform_grid_attempt_1(grid):
    return np.asarray(grid, dtype=np.int32)
def transform_grid_attempt_2(grid):
    return np.asarray(grid, dtype=np.int32)
"""
    train_r = asyncio.run(a.evaluate(task, solution_identity))
    test_r = asyncio.run(a.evaluate_test(task, solution_identity))
    assert train_r.score == pytest.approx(0.0)
    assert test_r.score == pytest.approx(0.0)

    # ROT90_THEN_IDENTITY solves both train and test
    train_r2 = asyncio.run(a.evaluate(task, ROT90_THEN_IDENTITY))
    test_r2 = asyncio.run(a.evaluate_test(task, ROT90_THEN_IDENTITY))
    assert train_r2.score == pytest.approx(1.0)
    assert test_r2.score == pytest.approx(1.0)


def test_missing_functions_returns_error_feedback(fake_arc_dir):
    a = ARCAGI2Adapter(data_dir=fake_arc_dir, task_ids=["t1"])
    [task] = a.load_tasks()
    result = asyncio.run(a.evaluate(task, MISSING_FUNCTION_SOLUTION))
    assert result.success is False
    assert result.score == pytest.approx(0.0)
    payload = json.loads(result.feedback)
    assert "error" in payload
    assert "transform_grid_attempt_2" in payload["error"]


def test_invalid_return_type_handled(fake_arc_dir):
    a = ARCAGI2Adapter(data_dir=fake_arc_dir, task_ids=["t1"])
    [task] = a.load_tasks()
    result = asyncio.run(a.evaluate(task, INVALID_RETURN_SOLUTION))
    # Both attempts return invalid types → pass@2 must be 0; the task is
    # not "successful" but the call should NOT raise.
    assert result.success is False
    assert result.score == pytest.approx(0.0)


@pytest.mark.slow
def test_subprocess_timeout(fake_arc_dir):
    """`while True: pass` in attempt_1 must time out within a few seconds."""
    a = ARCAGI2Adapter(data_dir=fake_arc_dir, task_ids=["t1"], timeout=2)
    [task] = a.load_tasks()
    result = asyncio.run(a.evaluate(task, INFINITE_LOOP_SOLUTION))
    assert result.success is False
    payload = json.loads(result.feedback)
    assert "Timeout" in payload.get("error", "") or payload.get("combined_score") == 0.0


# ---------------------------------------------------------------------------
# Orchestrator integration gate
# ---------------------------------------------------------------------------

def test_evaluate_test_method_present():
    """The orchestrator's `_run_test_evaluation` checks for this method via
    hasattr(); ensure it's there so the test-set report populates.
    """
    a = ARCAGI2Adapter()
    assert hasattr(a, "evaluate_test")
    assert callable(a.evaluate_test)
