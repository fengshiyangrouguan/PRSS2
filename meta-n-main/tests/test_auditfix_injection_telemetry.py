"""Regression tests for audit fixes in meta_n/analysis/injection_telemetry.py.

Findings:
  * 13 — helper_called_in over-counts adoption: the python (AST) channel must
         EXCLUDE a name shadowed by a local ``def <name>`` in the solver body
         (mirrors scan_helper_calls' def-shadow guard).
  * 72 — bash-helper adoption is structurally undercounted: the file-form
         detector must match ``helpers/(?:_lib_)?<name>.(?:py|sh)`` (``.sh`` and
         the ``helpers/`` spine form), and a bash body (no Python parse) must
         also count the bare-word ``\\bname\\b`` call form.
  * 24 — redundant_generation_rate inflated to 1.0 by helper-empty layers: a
         consecutive pair of EMPTY helper keysets (jaccard(∅,∅)=1.0) must be
         EXCLUDED from the diversity Jaccard list, so a run that generated zero
         helpers is not reported as maximal redundant re-emission.

All offline / LLM-free / no Docker / no network.
"""

import json

from meta_n.analysis.injection_telemetry import (
    candidate_layer_diversity,
    helper_called_in,
    run_injection_report,
)


# --------------------------------------------------------------------------- #
# Finding 13 — def-shadow guard on the python AST channel
# --------------------------------------------------------------------------- #

def test_finding13_local_def_shadow_not_counted_as_adoption():
    """A solver that DEFINES its own ``get_swap_delta`` and calls it must NOT be
    counted as adopting the injected helper of the same name."""
    body = (
        "def get_swap_delta(a, b):\n"
        "    return a - b\n"
        "\n"
        "def solve(task, llm):\n"
        "    return get_swap_delta(1, 2)\n"
    )
    # Pre-fix: _python_calls counts get_swap_delta -> returns True (false positive).
    assert helper_called_in("get_swap_delta", body) is False


def test_finding13_genuine_call_without_local_def_still_counted():
    """A genuine call to an injected helper (no local def) is still adoption."""
    body = (
        "def solve(task, llm):\n"
        "    return get_swap_delta(1, 2)\n"
    )
    assert helper_called_in("get_swap_delta", body) is True


# --------------------------------------------------------------------------- #
# Finding 72 — bash-helper adoption detection
# --------------------------------------------------------------------------- #

def test_finding72_spine_sh_file_form_counted():
    """A bash helper sourced via ``./helpers/<name>.sh`` (the spine staging +
    system-suffix form) must be detected — previously a flat zero."""
    body = "source ./helpers/myhelper.sh\nmyhelper --flag arg\n"
    assert helper_called_in("myhelper", body) is True


def test_finding72_helpers_py_without_lib_prefix_counted():
    """``./helpers/<name>.py`` (no ``_lib_`` prefix) is also a staged form."""
    body = "python3 ./helpers/cracker.py input.txt\n"
    assert helper_called_in("cracker", body) is True


def test_finding72_inline_lib_sh_counted():
    """Inline ``/tmp/_lib_<name>.sh`` form is matched (``.sh``, not just ``.py``)."""
    body = "bash /tmp/_lib_router.sh\n"
    assert helper_called_in("router", body) is True


def test_finding72_bash_bare_word_call_counted():
    """A bash body (does not parse as Python) that invokes a prepended helper by
    bare name must be counted via the bare-word fallback."""
    body = "#!/bin/bash\nset -e\ncompute_route alpha beta\n"
    assert helper_called_in("compute_route", body) is True


def test_finding72_inline_lib_py_still_counted_no_regression():
    """The original inline ``_lib_<name>.py`` form keeps working."""
    body = "python3 /tmp/_lib_solver.py\n"
    assert helper_called_in("solver", body) is True


def test_finding72_unrelated_name_not_overmatched():
    """A python body that never references the helper is not a false positive."""
    body = "def solve(task, llm):\n    return 42\n"
    assert helper_called_in("myhelper", body) is False


# --------------------------------------------------------------------------- #
# Finding 24 — both-empty layer pairs excluded from diversity Jaccard
# --------------------------------------------------------------------------- #

def _write_layer(cdir, depth, code_library=None, code_library_bash=None, rationale=""):
    blob = {
        "code_library": code_library or {},
        "code_library_bash": code_library_bash or {},
        "rationale": rationale,
    }
    (cdir / f"injected_code_d{depth}.json").write_text(json.dumps(blob))


def test_finding24_both_empty_layers_excluded_from_jaccards(tmp_path):
    """Two consecutive helper-EMPTY layers must yield NO Jaccard entry (not a
    1.0 entry)."""
    cdir = tmp_path / "cand0"
    cdir.mkdir()
    _write_layer(cdir, 0, rationale="layer zero")
    _write_layer(cdir, 1, rationale="layer one")

    div = candidate_layer_diversity(cdir)
    assert div["n_layers"] == 2
    # Pre-fix: jaccard(∅,∅) == 1.0 produced [1.0]; post-fix the both-empty pair
    # is excluded entirely.
    assert div["helper_name_jaccards"] == []


def test_finding24_redundant_rate_none_when_no_helpers_generated(tmp_path):
    """A run where every layer ships zero helpers reports redundant_generation_rate
    as None (no helper-bearing transitions), NOT 1.0."""
    archive = tmp_path / "archive"
    cdir = archive / "cand0"
    cdir.mkdir(parents=True)
    (cdir / "summary.json").write_text("{}")
    _write_layer(cdir, 0)
    _write_layer(cdir, 1)
    _write_layer(cdir, 2)

    report = run_injection_report(tmp_path)
    assert report["redundant_generation_rate"] is None


def test_finding24_helper_bearing_pairs_still_scored(tmp_path):
    """A transition that DOES ship helpers is still scored: identical helper sets
    on both sides -> jaccard 1.0 (genuine re-emission)."""
    cdir = tmp_path / "cand0"
    cdir.mkdir()
    _write_layer(cdir, 0, code_library={"foo": "..."})
    _write_layer(cdir, 1, code_library={"foo": "..."})

    div = candidate_layer_diversity(cdir)
    assert div["helper_name_jaccards"] == [1.0]


def test_finding24_partial_empty_pair_scored_zero(tmp_path):
    """An empty -> non-empty transition (union non-empty) IS scored, at 0.0 —
    only BOTH-empty pairs are excluded."""
    cdir = tmp_path / "cand0"
    cdir.mkdir()
    _write_layer(cdir, 0)  # empty
    _write_layer(cdir, 1, code_library={"foo": "..."})  # non-empty

    div = candidate_layer_diversity(cdir)
    assert div["helper_name_jaccards"] == [0.0]
