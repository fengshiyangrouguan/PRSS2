"""attribute_utilities — non-invasive bash/python matching, None vs [] (§12.1)."""

from __future__ import annotations

from meta_n.core.external_agents.telemetry import attribute_utilities


# --- the load-bearing None-vs-[] distinction -------------------------------


def test_unavailable_attribution_returns_none_not_empty():
    called, counts = attribute_utilities(
        ["wc -l x"], ["count_lines"], attribution_available=False
    )
    # None == unmeasurable on this backend (NOT zero usage).
    assert called is None
    assert counts == {}


def test_available_but_none_used_returns_empty_list():
    called, counts = attribute_utilities(
        ["ls -la", "cat file"], ["count_lines"], attribution_available=True
    )
    # [] == measured, none of the injected utilities were called.
    assert called == []
    assert counts == {}


def test_available_with_no_utilities_returns_empty_list():
    called, counts = attribute_utilities(
        ["ls"], [], attribution_available=True
    )
    assert called == []
    assert counts == {}


# --- bash helper attribution -----------------------------------------------


def test_bash_helper_matched_as_command_word():
    called, counts = attribute_utilities(
        ["count_lines input.txt", "echo done"],
        ["count_lines"],
        attribution_available=True,
    )
    assert called == ["count_lines"]
    assert counts["count_lines"] == 1


def test_bash_helper_word_boundary_no_substring_match():
    # A helper named `parse` must NOT match `parser` (word-boundary match).
    called, counts = attribute_utilities(
        ["parser --flag input"], ["parse"], attribution_available=True
    )
    assert called == []


def test_bash_helper_counts_multiple_invocations():
    called, counts = attribute_utilities(
        ["count_lines a", "count_lines b", "count_lines c"],
        ["count_lines"],
        attribution_available=True,
    )
    assert counts["count_lines"] == 3


# --- python helper attribution ---------------------------------------------


def test_python_helper_matched_by_filename():
    called, counts = attribute_utilities(
        ["python3 helpers/greet.py World"],
        ["greet"],
        attribution_available=True,
    )
    assert called == ["greet"]


def test_python_helper_matched_by_from_import_token():
    # `from helpers import greet` contains no filename — the import token match
    # is required to avoid a false negative.
    called, counts = attribute_utilities(
        ["python3 -c 'from helpers import greet; print(greet(1))'"],
        ["greet"],
        attribution_available=True,
    )
    assert called == ["greet"]


def test_python_helper_matched_by_dotted_call():
    called, counts = attribute_utilities(
        ["python3 -c 'import helpers; helpers.greet(5)'"],
        ["greet"],
        attribution_available=True,
    )
    assert called == ["greet"]


# --- false-positive avoidance ----------------------------------------------


def test_generic_solve_name_no_false_positive_on_bare_command():
    # A helper named `solve` must NOT match an unrelated bare `solve` command;
    # generic names only count via the helpers/-prefixed python form.
    called, counts = attribute_utilities(
        ["./solve --input data", "run benchmark"],
        ["solve", "run"],
        attribution_available=True,
    )
    assert called == []


def test_generic_solve_name_counts_only_via_helpers_prefix():
    called, counts = attribute_utilities(
        ["python3 helpers/solve.py"],
        ["solve"],
        attribution_available=True,
    )
    assert called == ["solve"]


def test_mixed_bash_and_python_helpers_sorted():
    called, counts = attribute_utilities(
        [
            "count_lines a.txt",
            "python3 -c 'from helpers import greet; greet(1)'",
        ],
        ["greet", "count_lines"],
        attribution_available=True,
    )
    # Result list is sorted.
    assert called == ["count_lines", "greet"]


# --- F1: advertised BASH-mode invocations (were false-negated before F1) ----


def test_python_helper_via_lib_bash_form():
    # The advertised bash invocation of a PYTHON helper is
    # `python3 ./helpers/_lib_<name>.py <args>` (injection.py _HELPERS_NOTE_BASH +
    # build_python_lib_file). Before F1 the `_lib_` prefix false-negated this.
    called, counts = attribute_utilities(
        ["python3 ./helpers/_lib_solve_countdown.py 462 3 8"],
        ["solve_countdown"],
        attribution_available=True,
    )
    assert called == ["solve_countdown"]
    assert counts["solve_countdown"] == 1


def test_bash_helper_via_source_form():
    # The advertised bash invocation of a BASH helper is `source ./helpers/<name>.sh`
    # then call the function. Before F1 the `source` line false-negated because the
    # name is followed by `.` (the word-boundary lookahead rejects it).
    called, counts = attribute_utilities(
        ["source ./helpers/normalize.sh"],
        ["normalize"],
        attribution_available=True,
    )
    assert called == ["normalize"]
    assert counts["normalize"] == 1


def test_bash_helper_source_then_function_call_counts_both():
    # Sourcing then calling the function should register on BOTH lines.
    called, counts = attribute_utilities(
        ["source ./helpers/normalize.sh", "normalize input.txt"],
        ["normalize"],
        attribution_available=True,
    )
    assert called == ["normalize"]
    assert counts["normalize"] == 2


def test_generic_name_via_lib_form_safe():
    # A generic-named python helper run via the `_lib_` bash form DOES register
    # (helpers/-anchored, so it is safe), while a bare `solve` command does NOT.
    called, counts = attribute_utilities(
        ["python3 ./helpers/_lib_solve.py", "./solve --input data"],
        ["solve"],
        attribution_available=True,
    )
    assert called == ["solve"]
    assert counts["solve"] == 1


# --- H16: generic-named helper via the advertised helpers/<name>.sh form ----


def test_generic_sh_helper_file_form():
    # A GENERIC name (run/solve/main/test) registers on its advertised
    # `source ./helpers/<name>.sh` invocation — the helpers/-anchored .sh file form
    # is generic-SAFE, so it counts even though the bare-word bash rule stays gated
    # off generic names.
    called, counts = attribute_utilities(
        ["source ./helpers/run.sh"], ["run"], attribution_available=True
    )
    assert called == ["run"]
    assert counts == {"run": 1}


def test_generic_sh_form_does_not_reopen_bare_word_false_positive():
    # The .sh file form is additive: a bare generic command word (no helpers/ .sh
    # anchor) must STILL not match (the bare word-boundary rule remains gated off
    # generic names).
    called, counts = attribute_utilities(
        ["run benchmark", "main --flag"], ["run", "main"], attribution_available=True
    )
    assert called == []


def test_generic_sh_and_lib_forms_both_count_for_generic_name():
    # A generic 'solve' helper invoked via BOTH advertised forms (the bash .sh file
    # and the python _lib_ file) registers on each.
    called, counts = attribute_utilities(
        ["source ./helpers/solve.sh", "python3 ./helpers/_lib_solve.py 1 2"],
        ["solve"],
        attribution_available=True,
    )
    assert called == ["solve"]
    assert counts["solve"] == 2


def test_lib_form_does_not_false_positive_on_bare_token():
    # The `_lib_` filename pattern is helpers/-anchored: a bare command word that
    # merely shares the name must NOT match via the python-helper path.
    called, counts = attribute_utilities(
        ["normalize input.txt"],  # bash function-word form, no source line
        ["normalize"],
        attribution_available=True,
    )
    # This still matches via the bash word-boundary rule (correct), but a path-
    # unrelated token like `_lib_normalize` substring must not over-count.
    assert called == ["normalize"]
    assert counts["normalize"] == 1
