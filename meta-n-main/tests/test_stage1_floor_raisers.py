"""Stage 1 OH ORTHOGONAL floor-raisers — behavior tests for R1 + R2.

R1 = ``--agentic-error-hints`` (error-hint taxonomy in the observation).
R2 = ``--agentic-preamble`` (behavioral preamble in the system prompt).

Both default OFF. The byte-identity-when-OFF *gate* lives in
``tests/test_stage1_golden.py`` (recomputes the HEAD goldens). This module
asserts the *ON* behaviour and the taxonomy↔classify_error contract, plus a
second OFF byte-identity check that does NOT depend on the captured goldens
(renders OFF vs the raw constant directly).
"""

from __future__ import annotations

import pytest

from meta_n.core.agentic_prompts import (
    AGENTIC_PREAMBLE,
    AGENTIC_SYSTEM_PROMPT,
    ERROR_HINTS,
    OBSERVATION_TEMPLATE,
    error_hint,
    get_language_instructions,
)
from meta_n.core.agentic_solver import AgenticSolver
from meta_n.core.meta_layer import TaskDescription, Trace, classify_error
from tests.stage1_golden_harness import _trace_fixtures

# Real classify_error classes (meta_layer.py:115-142) — single source of truth.
ACTIONABLE = {
    "Numeric instability",
    "Dependency error",
    "Timeout",
    "Constraint violation",
    "Indexing error",
    "Syntax error",
    "Format/parse error",
}
NON_ACTIONABLE = {
    "Unknown error",
    "Runtime error",
    "Turn starvation",
    "Environment fault",
}
ALL_CLASSES = ACTIONABLE | NON_ACTIONABLE


def _solver(**kw) -> AgenticSolver:
    """No-LLM / no-Docker solver — the two render methods are pure."""
    return AgenticSolver(
        llm_client=None,
        executor=None,
        injected_codes=None,
        solver_language="python",
        max_turns=5,
        **kw,
    )


# --------------------------------------------------------------------------- #
# Taxonomy ↔ classify_error contract (R1)
# --------------------------------------------------------------------------- #
def test_error_hints_keys_are_real_classify_error_classes():
    assert set(ERROR_HINTS) <= ALL_CLASSES
    assert set(ERROR_HINTS) == ACTIONABLE


def test_error_hint_empty_for_non_actionable_and_unknown_keys():
    for cls in NON_ACTIONABLE:
        assert error_hint(cls) == ""
    assert error_hint("") == ""
    assert error_hint("not a real class") == ""


def test_error_hint_nonempty_for_every_actionable_class():
    for cls in ACTIONABLE:
        assert error_hint(cls).strip()


def test_classify_error_map_partitions_fixtures_into_known_classes():
    """Every golden fixture resolves to one of the 11 real classes, and the
    fixtures collectively exercise the full actionable + non-actionable set."""
    observed = {classify_error(t) for _, t, _ in _trace_fixtures()}
    assert observed <= ALL_CLASSES
    assert ACTIONABLE <= observed
    assert NON_ACTIONABLE <= observed


# --------------------------------------------------------------------------- #
# R1 — observation hints
# --------------------------------------------------------------------------- #
def test_r1_off_is_constant_verbatim_render():
    """With the flag OFF, _build_observation == OBSERVATION_TEMPLATE.format(...)
    with NO error_hints field (proves no stray bytes, independent of goldens)."""
    off = _solver()  # default OFF
    trace = Trace(
        task_id="t", score=0.0, exit_code=124, success=False,
        stdout="", stderr="Process timed out after 60s", duration_s=60.0,
    )
    expected = OBSERVATION_TEMPLATE.format(
        turn=2, max_turns=5, score=0.0, exit_code=124, duration_s=60.0,
        stdout="(empty)", stderr="Process timed out after 60s",
        eval_feedback_section="",
    )
    assert off._build_observation(trace, 2) == expected
    assert "Likely cause" not in off._build_observation(trace, 2)


@pytest.mark.parametrize("key,trace,turn", _trace_fixtures())
def test_r1_on_hint_matches_actionability(key, trace, turn):
    """ON: a failing actionable class renders '### Likely cause: <cls>'; a
    success or non-actionable class renders byte-identically to OFF."""
    off = _solver()
    on = _solver(agentic_error_hints=True)
    off_render = off._build_observation(trace, turn)
    on_render = on._build_observation(trace, turn)

    cls = classify_error(trace)
    actionable_failure = (not trace.success) and cls in ACTIONABLE
    if actionable_failure:
        assert f"### Likely cause: {cls}" in on_render
        assert error_hint(cls) in on_render
        # The hint sits BEFORE the trailing "Analyze the results" coda.
        assert on_render.index("### Likely cause:") < on_render.index("Analyze the results")
    else:
        # No hint -> ON must be byte-identical to OFF.
        assert on_render == off_render
        assert "Likely cause" not in on_render


def test_r1_on_hint_precedes_eval_feedback_section():
    """When both a hint and eval feedback are present, the hint comes first."""
    on = _solver(agentic_error_hints=True)
    trace = Trace(
        task_id="t", score=0.0, exit_code=1, success=False,
        stdout="", stderr="IndexError: list index out of range",
        eval_feedback="case_3 mismatch: expected A got B", duration_s=0.6,
    )
    render = on._build_observation(trace, 2)
    assert "### Likely cause: Indexing error" in render
    assert "### Evaluation Details" in render
    assert render.index("### Likely cause:") < render.index("### Evaluation Details")


# --------------------------------------------------------------------------- #
# R2 — system-prompt preamble
# --------------------------------------------------------------------------- #
def test_r2_off_is_constant_verbatim_render():
    off = _solver()  # default OFF
    task = TaskDescription(task_id="x", description="Solve it.", metadata={})
    expected = AGENTIC_SYSTEM_PROMPT.format(
        task_description="Solve it.",
        language_instructions=get_language_instructions("python", {}),
        context_section="",
        language="python",
    )
    assert off._build_system_message(task, "") == expected
    assert "## How to work" not in off._build_system_message(task, "")


def test_r2_on_injects_preamble_in_correct_slot():
    on = _solver(agentic_preamble=True)
    task = TaskDescription(task_id="x", description="Solve it.", metadata={})
    render = on._build_system_message(task, "")
    assert AGENTIC_PREAMBLE.strip() in render
    # After the role line, before "## Task", and before "## OUTPUT FORMAT"
    # (so OUTPUT FORMAT + Hard rules stay last / most-salient).
    assert render.index("output executable code.") < render.index("## How to work")
    assert render.index("## How to work") < render.index("## Task")
    assert render.index("## How to work") < render.index("## OUTPUT FORMAT")
    assert render.index("## How to work") < render.index("## Hard rules")


def test_r2_does_not_mutate_module_constant():
    """The preamble is injected at render time only — the module constant has
    no preamble regardless of how many renders run."""
    on = _solver(agentic_preamble=True)
    task = TaskDescription(task_id="x", description="Solve it.", metadata={})
    on._build_system_message(task, "")
    assert "## How to work" not in AGENTIC_SYSTEM_PROMPT


# --------------------------------------------------------------------------- #
# Flags are independent and default OFF
# --------------------------------------------------------------------------- #
def test_flags_default_off():
    s = _solver()
    assert s.agentic_error_hints is False
    assert s.agentic_preamble is False
