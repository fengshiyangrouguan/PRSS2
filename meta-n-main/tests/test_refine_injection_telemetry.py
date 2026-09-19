"""Refinement regression tests for meta_n/analysis/injection_telemetry.py.

Covers:
- F161: the documented CLI (`python -m meta_n.analysis.injection_telemetry`)
  exists — main() prints the report JSON and honors --output.
- F165: the bash bare-word fallback in helper_called_in accepts a cached
  ``body_is_python`` verdict, and candidate_helper_stats parse-checks each
  solver body at most once (not once per advertised helper).

All offline / LLM-free / no Docker / no network.
"""

import json
from collections import Counter

from meta_n.analysis import injection_telemetry
from meta_n.analysis.injection_telemetry import (
    candidate_helper_stats,
    helper_called_in,
    main,
)

MARKER = "# --- end injected code library ---"
LIB = "# --- injected code library ---\ndef robust_zscore(v):\n    return v\n" + MARKER
BASH_BODY = "#!/bin/bash\nset -e\ncompute_route alpha beta\n"


# --------------------------------------------------------------------------- #
# F161 — CLI entry point
# --------------------------------------------------------------------------- #

def test_module_exposes_main_entry_point():
    assert hasattr(injection_telemetry, "main")
    assert callable(injection_telemetry.main)


def _write_run_dir(tmp_path, make_candidate, make_injected, make_trace,
                   write_candidate_dir):
    script = LIB + "\ndef solve(**kwargs):\n    return robust_zscore(5)\n"
    ic = make_injected(
        code_library={"robust_zscore": "def robust_zscore(v):\n    return v"}
    )
    c = make_candidate(
        "gen1_b0_k0", depth=2, injected_codes=[ic],
        traces=[make_trace("task_a", score=0.5, script=script)],
    )
    write_candidate_dir(tmp_path / "archive", c)
    return tmp_path


def test_cli_prints_report_json(
    tmp_path, capsys, make_candidate, make_injected, make_trace,
    write_candidate_dir,
):
    run_dir = _write_run_dir(
        tmp_path, make_candidate, make_injected, make_trace, write_candidate_dir
    )
    rc = main([str(run_dir)])
    assert rc == 0
    report = json.loads(capsys.readouterr().out)
    assert "overall_call_rate" in report
    assert report["overall_call_rate"] == 1.0


def test_cli_output_flag_writes_file(
    tmp_path, capsys, make_candidate, make_injected, make_trace,
    write_candidate_dir,
):
    run_dir = _write_run_dir(
        tmp_path, make_candidate, make_injected, make_trace, write_candidate_dir
    )
    out_path = tmp_path / "report.json"
    rc = main([str(run_dir), "--output", str(out_path)])
    assert rc == 0
    written = json.loads(out_path.read_text())
    printed = json.loads(capsys.readouterr().out)
    assert written == printed
    assert out_path.read_text().endswith("\n")


# --------------------------------------------------------------------------- #
# F165 — cached parse verdict
# --------------------------------------------------------------------------- #

def test_bash_fallback_uses_cached_parse_flag(
    tmp_path, monkeypatch, make_candidate, make_injected, make_trace,
    write_candidate_dir,
):
    # Reference results BEFORE patching (the un-cached and cached paths must
    # return identical booleans).
    expected_hit = helper_called_in("compute_route", BASH_BODY, Counter())
    expected_miss = helper_called_in("unrelated", BASH_BODY, Counter())
    assert expected_hit is True and expected_miss is False

    calls = {"n": 0}
    real = injection_telemetry._parses_as_python

    def counting(body):
        calls["n"] += 1
        return real(body)

    monkeypatch.setattr(injection_telemetry, "_parses_as_python", counting)

    # With a caller-supplied verdict the parse is never re-run.
    assert helper_called_in(
        "compute_route", BASH_BODY, Counter(), body_is_python=False
    ) is expected_hit
    assert helper_called_in(
        "unrelated", BASH_BODY, Counter(), body_is_python=False
    ) is expected_miss
    assert calls["n"] == 0

    # candidate_helper_stats over a 3-helper bash candidate: the body is
    # parse-checked at most once (previously once per advertised helper).
    ic = make_injected(
        pre_process=None,
        code_library_bash={
            "compute_route": "compute_route() { :; }",
            "helper_b": "helper_b() { :; }",
            "helper_c": "helper_c() { :; }",
        },
    )
    c = make_candidate(
        "gen1_bash", depth=2, injected_codes=[ic],
        traces=[make_trace("task_a", score=0.5, script=BASH_BODY)],
    )
    cdir = write_candidate_dir(tmp_path / "archive", c)

    calls["n"] = 0
    stats = candidate_helper_stats(cdir)
    assert calls["n"] <= 1  # one body → at most one parse check
    # Behavior unchanged: the invoked helper is counted, the others are not.
    assert stats["compute_route"]["tasks_called"] == ["task_a"]
    assert stats["helper_b"]["tasks_called"] == []
    assert stats["helper_c"]["tasks_called"] == []


def test_default_path_behavior_identical():
    """No cached flag → byte-identical semantics for every existing regime."""
    py_body = "def solve(task, llm):\n    return get_swap_delta(1, 2)\n"
    assert helper_called_in("get_swap_delta", py_body) is True
    assert helper_called_in("compute_route", BASH_BODY) is True
    assert helper_called_in("unrelated", BASH_BODY) is False
    # Python body with zero calls: bare-word fallback correctly NOT taken.
    py_mention = "# compute_route is mentioned but never called\nx = 1\n"
    assert helper_called_in("compute_route", py_mention) is False
