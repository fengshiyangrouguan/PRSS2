"""Helper call-rate & injection-diversity telemetry tests (P0.5, §3.5)."""

from meta_n.analysis.injection_telemetry import (
    candidate_helper_stats,
    candidate_layer_diversity,
    helper_called_in,
    jaccard,
    run_injection_report,
    solver_body,
)

MARKER = "# --- end injected code library ---"
LIB = "# --- injected code library ---\ndef robust_zscore(v):\n    return v\n" + MARKER


def _script(call=True):
    body = ("def solve(**kwargs):\n    return robust_zscore(5)\n" if call
            else "def solve(**kwargs):\n    return 5\n")
    return LIB + "\n" + body


def test_solver_body_strips_library_prefix():
    body = solver_body(_script(call=True))
    assert "robust_zscore(5)" in body
    assert "def robust_zscore" not in body  # the def lives in the stripped prefix


def test_no_marker_returns_whole_script():
    assert solver_body("def solve(): return 1") == "def solve(): return 1"


def test_helper_called_after_marker_counts():
    assert helper_called_in("robust_zscore", solver_body(_script(call=True))) is True


def test_helper_not_called_does_not_count():
    assert helper_called_in("robust_zscore", solver_body(_script(call=False))) is False


def test_def_in_library_prefix_not_counted():
    # The only mention of robust_zscore is its own def, before the marker.
    body = solver_body(LIB + "\ndef solve(**kwargs):\n    return 5\n")
    assert helper_called_in("robust_zscore", body) is False


def test_bash_staged_file_call_counts():
    body = "set -e\npython3 /tmp/_lib_foo.py 1 2 3\n"
    assert helper_called_in("foo", body) is True
    assert helper_called_in("bar", body) is False
    # word-boundary: _lib_foo.py must not match a helper named 'fo'
    assert helper_called_in("fo", body) is False


def test_jaccard_basic():
    assert jaccard({"a", "b"}, {"a", "c"}) == 1 / 3
    assert jaccard(set(), set()) == 1.0
    assert jaccard({"a"}, {"b"}) == 0.0


def test_candidate_helper_stats_roundtrip(
    tmp_path, make_candidate, make_injected, make_trace, write_candidate_dir
):
    ic = make_injected(code_library={"robust_zscore": "def robust_zscore(v):\n    return v"})
    c = make_candidate(
        "gen1_b0_k0", depth=2, injected_codes=[ic],
        traces=[make_trace("task_a", score=0.5, script=_script(call=True)),
                make_trace("task_b", score=0.4, script=_script(call=False))],
    )
    cdir = write_candidate_dir(tmp_path / "archive", c)
    stats = candidate_helper_stats(cdir)
    assert stats["robust_zscore"]["tasks_called"] == ["task_a"]
    assert stats["robust_zscore"]["call_rate"] == 0.5  # called on 1 of 2 tasks


def test_run_report_aggregates(
    tmp_path, make_candidate, make_injected, make_trace, write_candidate_dir
):
    ic = make_injected(code_library={"robust_zscore": "def robust_zscore(v):\n    return v"})
    c = make_candidate(
        "gen1_b0_k0", depth=2, injected_codes=[ic],
        traces=[make_trace("task_a", script=_script(call=True))],
    )
    write_candidate_dir(tmp_path / "archive", c)
    report = run_injection_report(tmp_path)
    assert report["advertised_helpers"] == 1
    assert report["called_helpers"] == 1
    assert report["overall_call_rate"] == 1.0


def test_layer_diversity_consecutive_jaccard(
    tmp_path, make_candidate, make_injected, make_trace, write_candidate_dir
):
    ic1 = make_injected(code_library={"a": "def a(): pass", "b": "def b(): pass"},
                        rationale="use local search to perturb the assignment")
    ic2 = make_injected(code_library={"a": "def a(): pass", "c": "def c(): pass"},
                        rationale="use simulated annealing with restarts")
    c = make_candidate(
        "gen2", depth=3, injected_codes=[ic1, ic2],
        traces=[make_trace("task_a", script=_script(call=True))],
    )
    cdir = write_candidate_dir(tmp_path / "archive", c)
    div = candidate_layer_diversity(cdir)
    assert div["n_layers"] == 2
    assert div["helper_name_jaccards"] == [1 / 3]  # {a,b} vs {a,c}
    assert 0.0 <= div["rationale_cosines"][0] < 1.0  # different rationales
