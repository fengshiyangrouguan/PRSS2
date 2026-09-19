"""Regression tests for audit fixes in meta_n/integrations/arc_agi.py.

Covers:
  - Finding 59: per-attempt output conversion must run inside the per-attempt
    try/except so a malformed return (e.g. a 2D object/str array that makes
    np.isfinite raise TypeError) nulls only THAT attempt -- sibling
    attempts/pairs still score (the documented robustness contract). On the
    un-fixed code the TypeError escapes the loop and the WHOLE split is queued
    as ("error", ...).
  - Finding 41: `import sys` was dead and removed; the module must import
    cleanly and not expose a bound `sys` name.

LLM-free / offline: calls the subprocess target in-process with a fake queue.
No LM Studio, no Docker, no network, no real multiprocessing.
"""

from meta_n.integrations import arc_agi


class _FakeQueue:
    """Minimal stand-in for mp.Queue capturing put() payloads."""

    def __init__(self) -> None:
        self.items: list = []

    def put(self, item) -> None:
        self.items.append(item)


def test_malformed_attempt_nulls_only_that_attempt():
    """Finding 59: a 2D object-dtype return (np.isfinite -> TypeError) must
    null only that attempt; the sibling attempt and other pairs still score."""
    solution = (
        "import numpy as np\n"
        "def transform_grid_attempt_1(grid):\n"
        "    return np.array([[1, None], [2, 3]], dtype=object)\n"
        "def transform_grid_attempt_2(grid):\n"
        "    return grid\n"
    )
    inputs = [[[0, 0], [0, 0]]]
    q = _FakeQueue()

    arc_agi._run_arc_attempts_in_process(solution, inputs, q)

    assert len(q.items) == 1
    status, value = q.items[0]
    # Pre-fix: TypeError escaped the loop -> ("error", ...). Post-fix: ("ok", ...).
    assert status == "ok", f"expected ok, got {status!r}: {value!r}"
    assert len(value) == 1
    attempt_1, attempt_2 = value[0]
    # The malformed attempt is nulled...
    assert attempt_1 is None
    # ...but the valid sibling attempt still scores.
    assert attempt_2 == [[0, 0], [0, 0]]


def test_string_attempt_nulls_only_that_attempt():
    """Finding 59 variant: a 2D string array also makes np.isfinite raise."""
    solution = (
        "import numpy as np\n"
        "def transform_grid_attempt_1(grid):\n"
        "    return grid\n"
        "def transform_grid_attempt_2(grid):\n"
        "    return np.array([['a', 'b'], ['c', 'd']])\n"
    )
    inputs = [[[5, 6], [7, 8]]]
    q = _FakeQueue()

    arc_agi._run_arc_attempts_in_process(solution, inputs, q)

    status, value = q.items[0]
    assert status == "ok", f"expected ok, got {status!r}: {value!r}"
    attempt_1, attempt_2 = value[0]
    assert attempt_1 == [[5, 6], [7, 8]]
    assert attempt_2 is None


def test_sys_import_removed():
    """Finding 41: dead `import sys` removed -- module has no bound `sys` name."""
    assert not hasattr(arc_agi, "sys")
