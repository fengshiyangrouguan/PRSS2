"""Render-all-paths guard for the Ω prompt assembly (G1–G7 global refinements).

The Ω prompt is rendered by a SINGLE call site (``OmegaEngine._build_prompt`` →
``template.format(...)``) but across THREE template/branch combinations:
  * depth 2, no baseline      → OMEGA_PROMPT
  * depth 2, with baseline    → OMEGA_PROMPT + score-comparison section
  * depth ≥3, with baseline   → OMEGA_PROMPT_META (structured summary)
each crossed with the ``no_code_library`` ablation (solver_lib slots blanked).

``str.format`` raises ``KeyError`` if a template references a ``{slot}`` the
kwargs don't supply. So when a global refinement adds a new ``{slot}`` to one
template, it MUST be supplied in the single ``.format`` kwargs AND kept
consistent across templates. This test renders every path so such a mismatch
fails loudly here instead of mid-run.
"""

from unittest.mock import MagicMock

import pytest

from meta_n.core.meta_layer import TaskDescription, Trace
from meta_n.core.omega import OmegaEngine


def _engine() -> OmegaEngine:
    return OmegaEngine(llm_client=MagicMock())


def _tasks(n: int = 3) -> list[TaskDescription]:
    return [
        TaskDescription(task_id=f"t{i}", description=f"description {i}", metadata={})
        for i in range(n)
    ]


def _traces() -> list[Trace]:
    return [
        Trace(
            task_id="t0", success=False, score=0.10,
            script="def solve(data):\n    pass",
            stderr="Timeout: solve timed out after 5s", error_summary="timed out",
        ),
        Trace(
            task_id="t1", success=True, score=0.80,
            script="def solve(data):\n    return 1",
        ),
        Trace(
            task_id="t2", success=False, score=0.0,
            script="def solve(data):\n    import scipy",
            stderr="ModuleNotFoundError: No module named 'scipy'",
            error_summary="no module named scipy",
        ),
    ]


_PREV = {"t0": 0.05, "t1": 0.70, "t2": 0.00}
_ARCHIVE_BEST = {"t0": 0.60, "t1": 0.90, "t2": 0.40}


@pytest.mark.parametrize("no_lib", [False, True])
@pytest.mark.parametrize(
    "depth,previous_scores,expect_marker",
    [
        (2, None, "one layer in a recursive"),          # OMEGA_PROMPT, no baseline
        (2, _PREV, "one layer in a recursive"),          # OMEGA_PROMPT + comparison
        (3, _PREV, "higher-order meta-layer"),           # OMEGA_PROMPT_META
    ],
)
def test_build_prompt_renders_every_path(depth, previous_scores, expect_marker, no_lib):
    """Every (template × baseline × ablation) combination renders without a
    KeyError and selects the expected template."""
    eng = _engine()
    prompt = eng._build_prompt(
        _traces(), [], _tasks(), depth,
        previous_scores=previous_scores,
        archive_best_scores=_ARCHIVE_BEST,
        solver_language="python",
        no_code_library=no_lib,
    )
    assert isinstance(prompt, str)
    assert len(prompt) > 200
    assert expect_marker in prompt
    # G1: continuous-score objective headline replaces the bare pass@1 line.
    assert "mean score" in prompt
    assert "PRIMARY OBJECTIVE" in prompt
    # G2: with archive-best gaps present, the headroom/opportunity table renders.
    assert "Opportunity" in prompt
    # G6: the generalization directive (no task_id-literal branching) is present.
    assert "GENERALIZE" in prompt
    # G7: depth-3+ uses the grounded rationale schema (Diagnosis/Intervention).
    if expect_marker == "higher-order meta-layer":
        assert "## Diagnosis" in prompt and "## Intervention" in prompt
        assert "What Previous Layers Tried" not in prompt
    # The ablation must drop the solver_lib schema; the default must keep it.
    if no_lib:
        assert "solver_lib:<name>" not in prompt
    else:
        assert "solver_lib" in prompt


def test_helper_usage_flags_dead_helpers():
    """G4: a helper the solver actually calls is reported with its call rate; a
    helper defined but never called is flagged DEAD. Counted over the full
    (library-prefix-stripped) solver scripts."""
    from meta_n.core.meta_layer import InjectedCode

    eng = _engine()
    stack = [
        InjectedCode(
            code_library={"foo": "def foo(d):\n    return d", "bar": "def bar(d):\n    return d"},
            source_depth=3,
        )
    ]
    traces = [
        Trace(task_id="t0", success=True, score=0.9, script="def solve(d):\n    return foo(d)"),
        Trace(task_id="t1", success=False, score=0.0, script="def solve(d):\n    return 1"),
    ]
    section = eng._helper_usage_section(traces, stack)
    assert "`foo`" in section and "called in 1/2" in section
    assert "`bar`" in section and "DEAD" in section
    # No helpers in the active stack → empty section (no spurious callout).
    assert eng._helper_usage_section(traces, []) == ""


def test_helper_usage_suppressed_when_library_demoted():
    """P8(a): when code_library_is_live is False the helpers are NEVER prepended
    to the solver (CO-Bench/SWE), so a ~0 call-rate is trivially true and the
    'DEAD — remove it' feedback misleads Ω. Suppress the whole section."""
    from meta_n.core.meta_layer import InjectedCode

    eng = _engine()
    stack = [InjectedCode(code_library={"foo": "def foo(d):\n    return d"}, source_depth=3)]
    traces = [Trace(task_id="t0", success=True, score=0.9, script="def solve(d):\n    return 1")]
    # Live (default) → the never-called helper is flagged DEAD.
    assert "DEAD" in eng._helper_usage_section(traces, stack)
    # Demoted → no callout at all.
    assert eng._helper_usage_section(traces, stack, code_library_is_live=False) == ""


def test_helper_usage_excludes_self_definition():
    """P8(b): a solver that DEFINES its own same-named function shadows the
    injected helper (Python scoping) — references bind to the solver's own def,
    not our helper — so that script must NOT count as a call (the observed
    equitable get_swap_delta false positive)."""
    from meta_n.core.meta_layer import InjectedCode

    eng = _engine()
    stack = [InjectedCode(code_library={"foo": "def foo(d):\n    return d"}, source_depth=3)]
    traces = [
        # Defines its OWN foo → shadows the injected helper, NOT a call.
        Trace(task_id="t0", success=True, score=0.9,
              script="def foo(d):\n    return d * 2\n\ndef solve(d):\n    return foo(d)"),
        # Calls the injected foo directly (no local def) → a real call.
        Trace(task_id="t1", success=True, score=0.9,
              script="def solve(d):\n    return foo(d)"),
    ]
    section = eng._helper_usage_section(traces, stack)
    # Only the second script counts → 1/2, not the pre-fix 2/2.
    assert "called in 1/2" in section


def test_helper_usage_counts_lib_and_sh_forms():
    """H16/F1: a solver that invokes the advertised helper via its FILE form —
    ``python3 ./helpers/_lib_foo.py`` or ``source ./helpers/bar.sh`` — IS calling
    it, so the helpers/-anchored filename form must count (not DEAD). Before H16
    the bare word-boundary regex missed the path and both were flagged DEAD,
    mis-steering Ω into pruning a LIVE helper."""
    from meta_n.core.meta_layer import InjectedCode

    eng = _engine()
    stack = [
        InjectedCode(
            code_library={
                "foo": "def foo(d):\n    return d",
                "bar": "def bar(d):\n    return d",
            },
            source_depth=3,
        )
    ]
    traces = [
        Trace(task_id="t0", success=True, score=0.9,
              script="python3 ./helpers/_lib_foo.py < in.json"),
        Trace(task_id="t1", success=True, score=0.9,
              script="source ./helpers/bar.sh && bar"),
    ]
    section = eng._helper_usage_section(traces, stack)
    assert "DEAD" not in section
    assert "`foo`" in section and "`bar`" in section
    assert section.count("called in 1/2") == 2


def test_helper_usage_file_form_not_subject_to_def_shadow():
    """The ``helpers/_lib_<name>.py`` invocation counts even when the SAME script
    also ``def <name>``s — a path reference is not a python rebind, so the
    def-shadow exclusion (which drops the bare-word form) must NOT drop it."""
    from meta_n.core.meta_layer import InjectedCode

    eng = _engine()
    stack = [InjectedCode(code_library={"foo": "def foo(d):\n    return d"}, source_depth=3)]
    traces = [
        Trace(task_id="t0", success=True, score=0.9,
              script="def foo(d):\n    return d*2\n# run: python3 ./helpers/_lib_foo.py"),
    ]
    section = eng._helper_usage_section(traces, stack)
    assert "called in 1/1" in section and "DEAD" not in section


def test_helper_usage_appended_to_context_stack_in_build_prompt():
    """The G4 callout reaches the rendered prompt via the injected-code section
    (no new template slot)."""
    eng = _engine()
    prompt = eng._build_prompt(
        _traces(), [], _tasks(), 2,
        previous_scores=_PREV, archive_best_scores=_ARCHIVE_BEST,
        solver_language="python",
        helper_usage="## Helper Usage\n- `widget`: called in 0/3 solver scripts — DEAD.\n",
    )
    assert "Helper Usage" in prompt and "widget" in prompt


def test_novelty_directive_fires_only_on_deep_plateau():
    """G5: the plateau directive fires at depth ≥3 when the previous layer's
    delta is ≤ +0.01, and stays silent otherwise (depth 2, real gain, or no
    baseline)."""
    eng = _engine()
    prev = {"t0": 0.50, "t1": 0.60}
    flat = {"t0": 0.50, "t1": 0.60}     # delta 0.0 → plateau
    better = {"t0": 0.90, "t1": 0.90}   # delta large → productive
    assert "PLATEAU RISK" in eng._novelty_directive(flat, prev, depth=3)
    assert eng._novelty_directive(better, prev, depth=3) == ""   # real gain
    assert eng._novelty_directive(flat, prev, depth=2) == ""     # not 1st Ω layer
    assert eng._novelty_directive(flat, None, depth=3) == ""     # no baseline


def test_novelty_directive_reaches_prompt_on_plateau():
    """A depth-3 plateau surfaces the directive in the rendered prompt."""
    eng = _engine()
    flat = {"t0": 0.10, "t1": 0.80, "t2": 0.00}
    prompt = eng._build_prompt(
        _traces(), [], _tasks(), 3,
        previous_scores=flat, archive_best_scores=_ARCHIVE_BEST,
        current_scores=flat,  # current == previous → plateau
        solver_language="python",
    )
    assert "PLATEAU RISK" in prompt


def test_build_prompt_bash_and_openevolve_langs_render():
    """The bash and openevolve solver_lib variants also render cleanly."""
    eng = _engine()
    for lang in ("bash", "openevolve", "python"):
        prompt = eng._build_prompt(
            _traces(), [], _tasks(), 2,
            previous_scores=_PREV, archive_best_scores=_ARCHIVE_BEST,
            solver_language=lang, no_code_library=False,
        )
        assert isinstance(prompt, str) and len(prompt) > 200
