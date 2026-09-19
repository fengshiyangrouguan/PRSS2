"""Refinement regression tests for meta_n/integrations/_subprocess_utils.py.

Covers F110/F112/F183 (shared home for the usage-dict quartet +
_kill_process_tree + _usage_from_log_since, with compat re-exports from the
old homes) and F111/F184 (the consolidated mp.Process-with-timeout harness —
each integration wrapper must keep its exact status/payload[/usage] tuples).

Offline: the subprocess tests run tiny inline sources in real mp.Process
children (the same isolation production uses); no LLM, no Docker, no network.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import meta_n.integrations._subprocess_utils as su
import meta_n.integrations.arc_agi as arc
import meta_n.integrations.co_bench as cb
import meta_n.integrations.openevolve as oe
import meta_n.integrations.text_classification as tc

_ZEROS = {"total": 0, "prompt": 0, "completion": 0, "calls": 0}


# ---------------------------------------------------------------------------
# F110/F112/F183 — one shared home, compat re-exports from the old homes
# ---------------------------------------------------------------------------


def test_helpers_identical_across_old_homes():
    """The moved helpers stay importable from co_bench AND text_classification
    and are the SAME objects as the _subprocess_utils ones (so monkeypatching
    either module namespace, and old `from co_bench import ...`, keep working)."""
    for name in (
        "_empty_usage",
        "_tracker_usage",
        "_coerce_usage",
        "_add_usage",
        "_kill_process_tree",
        "_usage_from_log_since",
    ):
        assert getattr(cb, name) is getattr(su, name), name
        assert getattr(tc, name) is getattr(su, name), name


def test_kill_process_tree_reexported_from_openevolve_and_arc():
    assert oe._kill_process_tree is su._kill_process_tree
    assert arc._kill_process_tree is su._kill_process_tree


def test_integration_imports_no_longer_drag_external_agents():
    """Importing arc_agi / openevolve / text_classification must not pull in
    meta_n.core.external_agents (nor co_bench) transitively any more.

    Fresh-interpreter subprocess so this can't pass by accident via modules
    another test already imported.
    """
    code = textwrap.dedent(
        """
        import sys
        import meta_n.integrations.arc_agi
        import meta_n.integrations.openevolve
        import meta_n.integrations.text_classification
        bad = [m for m in sys.modules if m.startswith("meta_n.core.external_agents")]
        assert not bad, f"external_agents leaked: {bad}"
        assert "meta_n.integrations.co_bench" not in sys.modules, "co_bench leaked"
        """
    )
    subprocess.run([sys.executable, "-c", code], check=True, timeout=120)


# ---------------------------------------------------------------------------
# F111/F184 — exact wrapper tuples (byte-identical failure payloads)
# ---------------------------------------------------------------------------

_CB_CONFIG = textwrap.dedent(
    """
    def load_data(path):
        return [{"x": 1}]

    def eval_func(**kwargs):
        return 1.5
    """
)


def _write_cb_config(tmp_path) -> str:
    (tmp_path / "config.py").write_text(_CB_CONFIG)
    return str(tmp_path / "config.py")


def test_cobench_wrapper_ok_tuple(tmp_path):
    cfg = _write_cb_config(tmp_path)
    out = cb._run_with_timeout(cfg, {"x": 1}, "def solve(**kw):\n    return {}\n", 10)
    assert out == ("ok", 1.5, _ZEROS)


def test_cobench_wrapper_timeout_tuple(tmp_path):
    cfg = _write_cb_config(tmp_path)
    solve = "import time\ndef solve(**kw):\n    time.sleep(30)\n    return {}\n"
    out = cb._run_with_timeout(cfg, {"x": 1}, solve, 1)
    assert out == ("error", "Timeout (1s)", _ZEROS)


def test_cobench_wrapper_no_result_tuple(tmp_path):
    cfg = _write_cb_config(tmp_path)
    solve = "import os\ndef solve(**kw):\n    os._exit(0)\n"
    out = cb._run_with_timeout(cfg, {"x": 1}, solve, 5)
    assert out == ("error", "No result from subprocess", _ZEROS)


def test_textclass_wrapper_ok_tuple():
    code = "def solve(cases, labels, few_shot):\n    return {k: labels[0] for k in cases}\n"
    out = tc._run_solve_with_timeout(code, {"case_0": "x"}, ["A"], [], None, 10)
    assert out == ("ok", {"case_0": "A"}, _ZEROS)


def test_textclass_wrapper_timeout_tuple():
    code = "import time\ndef solve(cases, labels, few_shot):\n    time.sleep(30)\n"
    status, payload, usage = tc._run_solve_with_timeout(
        code, {"case_0": "x"}, ["A"], [], None, 1,
    )
    assert status == "error"
    # tc's message deliberately carries the elapsed clock (unlike siblings).
    assert payload.startswith("Timeout (configured=1s, elapsed=")
    assert payload.endswith("s)")
    assert usage == _ZEROS


def test_textclass_wrapper_no_result_tuple():
    code = "import os\ndef solve(cases, labels, few_shot):\n    os._exit(0)\n"
    out = tc._run_solve_with_timeout(code, {"case_0": "x"}, ["A"], [], None, 5)
    assert out == ("error", "No result from subprocess (Empty: )", _ZEROS)


def _write_oe_evaluator(tmp_path, body: str) -> str:
    (tmp_path / "evaluator.py").write_text(body)
    return str(tmp_path)


def test_openevolve_wrapper_ok_tuple(tmp_path):
    d = _write_oe_evaluator(
        tmp_path, "def evaluate(program_path):\n    return {'combined_score': 0.5}\n"
    )
    out = oe._run_openevolve_with_timeout(d, "x = 1\n", "combined_score", 10)
    assert out == ("ok", {"score": 0.5, "details": {"combined_score": 0.5}})


def test_openevolve_wrapper_timeout_tuple(tmp_path):
    d = _write_oe_evaluator(
        tmp_path,
        "import time\ndef evaluate(program_path):\n    time.sleep(30)\n",
    )
    out = oe._run_openevolve_with_timeout(d, "x = 1\n", "combined_score", 1)
    assert out == ("error", "Timeout (1s)")


def test_openevolve_wrapper_no_result_tuple(tmp_path):
    d = _write_oe_evaluator(
        tmp_path, "import os\ndef evaluate(program_path):\n    os._exit(0)\n"
    )
    out = oe._run_openevolve_with_timeout(d, "x = 1\n", "combined_score", 5)
    assert out == ("error", "No result from subprocess")


_ARC_OK_SOLUTION = (
    "def transform_grid_attempt_1(g):\n    return g\n"
    "def transform_grid_attempt_2(g):\n    return g\n"
)


def test_arc_wrapper_ok_tuple():
    grid = [[1, 2], [3, 4]]
    out = arc._run_arc_attempts_with_timeout(_ARC_OK_SOLUTION, [grid], 10)
    assert out == ("ok", [[grid, grid]])


def test_arc_wrapper_timeout_tuple():
    out = arc._run_arc_attempts_with_timeout(
        "import time\ntime.sleep(30)\n" + _ARC_OK_SOLUTION, [[[1]]], 1,
    )
    assert out == ("error", "Timeout (1s)")


def test_arc_wrapper_no_result_tuple():
    out = arc._run_arc_attempts_with_timeout("import os\nos._exit(0)\n", [[[1]]], 5)
    assert out == ("error", "No result from subprocess")
