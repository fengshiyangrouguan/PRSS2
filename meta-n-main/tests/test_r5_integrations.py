"""R5 regression tests for the integrations cluster.

Covers three audited defects:

* ``run_process_with_timeout`` deadlocked on mp-queue payloads >= ~64KiB
  (parent joined before draining; the child blocked in its feeder thread and
  a successful eval was misreported as a timeout). The harness now drains
  the queue before joining.
* ARC-AGI-2 gold TEST output grids were exposed via
  ``task.metadata["task_dict"]`` — readable by Ω-injected pre_process, whose
  additional_context lands in the solver prompt. Gold test outputs now live
  only adapter-side (``ARCAGI2Adapter._test_pairs``).
* ``COBenchAdapter.split_type()`` carried a factually false "delegates to
  evaluate" comment (comment-only fix; the value is a deliberate P0.4
  classification and stays ``dev_equals_test``).
"""

from __future__ import annotations

import asyncio
import inspect
import json
import multiprocessing as mp
import os
import subprocess
import time
from pathlib import Path

import pytest

import meta_n.integrations._subprocess_utils as su
import meta_n.integrations.arc_agi as arc
from meta_n.core.meta_layer import TaskDescription
from meta_n.integrations.arc_agi import ARCAGI2Adapter
from meta_n.integrations.co_bench import COBenchAdapter

_PAYLOAD_BYTES = 256 * 1024


def _put_big_payload(queue):
    su.detach_process_group()
    queue.put(("ok", b"x" * _PAYLOAD_BYTES))


def _hang_with_grandchild(pid_file, queue):
    su.detach_process_group()
    proc = subprocess.Popen(["sleep", "60"])
    Path(pid_file).write_text(str(proc.pid))
    time.sleep(60)


def _assert_pid_dead(pid, timeout_s=5.0):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            os.kill(pid, 0)
        except (ProcessLookupError, PermissionError):
            return
        time.sleep(0.05)
    pytest.fail(f"pid {pid} still alive after kill ladder")


# ---------------------------------------------------------------------------
# run_process_with_timeout: drain-then-join
# ---------------------------------------------------------------------------

def test_large_payload_returned_promptly_not_misreported_as_timeout():
    queue: mp.Queue = mp.Queue()
    start = time.time()
    out = su.run_process_with_timeout(
        _put_big_payload,
        (queue,),
        30,
        queue=queue,
        on_timeout=lambda elapsed, pid: ("timeout", elapsed),
        on_no_result=lambda exc, pid: ("noresult", exc),
    )
    elapsed = time.time() - start
    assert out == ("ok", b"x" * _PAYLOAD_BYTES)
    assert elapsed < 15


def test_genuine_timeout_still_reports_timeout_and_kills_tree(tmp_path):
    pid_file = tmp_path / "grandchild.pid"
    seen: dict[str, object] = {}

    def on_timeout(elapsed, pid):
        seen["pid"] = pid
        return ("timeout", elapsed)

    queue: mp.Queue = mp.Queue()
    start = time.time()
    out = su.run_process_with_timeout(
        _hang_with_grandchild,
        (str(pid_file), queue),
        2,
        queue=queue,
        on_timeout=on_timeout,
        on_no_result=lambda exc, pid: ("noresult", exc),
    )
    elapsed = time.time() - start
    assert out[0] == "timeout"
    assert elapsed < 15
    _assert_pid_dead(int(seen["pid"]))
    if pid_file.exists():
        _assert_pid_dead(int(pid_file.read_text()))


def test_crashed_child_still_reports_no_result_quickly():
    solution = "import os\nos._exit(0)\n"
    start = time.time()
    status, value = arc._run_arc_attempts_with_timeout(solution, [[[1]]], 30)
    elapsed = time.time() - start
    assert status == "error"
    assert value == "No result from subprocess"
    assert elapsed < 15


def test_arc_large_grid_result_survives_the_pipe():
    solution = (
        "import numpy as np\n"
        "def transform_grid_attempt_1(grid):\n"
        "    return np.ones((300, 300), dtype=int)\n"
        "def transform_grid_attempt_2(grid):\n"
        "    return np.zeros((300, 300), dtype=int)\n"
    )
    start = time.time()
    status, value = arc._run_arc_attempts_with_timeout(solution, [[[1]]], 30)
    elapsed = time.time() - start
    assert status == "ok"
    assert len(value) == 1
    assert len(value[0][0]) == 300
    assert elapsed < 25


# ---------------------------------------------------------------------------
# ARC-AGI-2: gold TEST outputs stay out of task.metadata
# ---------------------------------------------------------------------------

_TRAIN_PAIR = {"input": [[5, 6], [7, 8]], "output": [[8, 7], [6, 5]]}
_TEST_INPUT = [[1, 2], [3, 4]]
_GOLD_TEST_OUTPUT = [[4, 3], [2, 1]]

_FLIP_SOLUTION = (
    "import numpy as np\n"
    "def transform_grid_attempt_1(grid):\n"
    "    return grid[::-1, ::-1]\n"
    "def transform_grid_attempt_2(grid):\n"
    "    return grid\n"
)

_IDENTITY_SOLUTION = (
    "import numpy as np\n"
    "def transform_grid_attempt_1(grid):\n"
    "    return grid\n"
    "def transform_grid_attempt_2(grid):\n"
    "    return grid\n"
)


@pytest.fixture
def arc_data_dir(tmp_path):
    split_dir = tmp_path / "data" / "evaluation"
    split_dir.mkdir(parents=True)
    task = {"train": [_TRAIN_PAIR], "test": [
        {"input": _TEST_INPUT, "output": _GOLD_TEST_OUTPUT},
    ]}
    (split_dir / "cafebabe.json").write_text(json.dumps(task))
    return tmp_path


def _walk(obj):
    yield obj
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _walk(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _walk(v)


def test_gold_test_output_unreachable_under_metadata(arc_data_dir):
    a = ARCAGI2Adapter(data_dir=str(arc_data_dir))
    [task] = a.load_tasks()
    assert all(node != _GOLD_TEST_OUTPUT for node in _walk(task.metadata))
    test_pairs = task.metadata["task_dict"]["test"]
    assert [set(p) for p in test_pairs] == [{"input"}]
    assert test_pairs[0]["input"] == _TEST_INPUT
    assert task.metadata["task_dict"]["train"] == [_TRAIN_PAIR]


def test_evaluate_test_scores_from_adapter_side_gold(arc_data_dir):
    a = ARCAGI2Adapter(data_dir=str(arc_data_dir))
    [task] = a.load_tasks()

    good = asyncio.run(a.evaluate_test(task, _FLIP_SOLUTION))
    assert good.success is True
    assert good.score == 1.0

    bad = asyncio.run(a.evaluate_test(task, _IDENTITY_SOLUTION))
    assert bad.success is False
    assert bad.score == 0.0

    train = asyncio.run(a.evaluate(task, _FLIP_SOLUTION))
    assert train.score == 1.0


def test_evaluate_test_unknown_task_reports_no_pairs(arc_data_dir):
    a = ARCAGI2Adapter(data_dir=str(arc_data_dir))
    a.load_tasks()
    stranger = TaskDescription(
        task_id="not_loaded", description="", metadata={"task_dict": {}},
    )
    result = asyncio.run(a.evaluate_test(stranger, _FLIP_SOLUTION))
    assert result.success is False
    assert "no test pairs" in result.feedback


# ---------------------------------------------------------------------------
# CO-Bench: split_type comment no longer claims delegation
# ---------------------------------------------------------------------------

def test_co_bench_split_type_value_pinned_and_comment_fixed():
    assert COBenchAdapter.split_type(None) == "held_out"
    src = inspect.getsource(COBenchAdapter.split_type)
    assert "delegates to evaluate" not in src
