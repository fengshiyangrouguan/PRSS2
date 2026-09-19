"""Failure-class classification tests (Step 7, 6.3) + the new Trace fields."""

from meta_n.core.meta_layer import Trace
from meta_n.core.omega import OmegaEngine

_c = OmegaEngine._classify_error


def _t(error_summary="", stderr="", terminated_by=""):
    return Trace(task_id="t", success=False, error_summary=error_summary,
                 stderr=stderr, terminated_by=terminated_by)


def test_turn_starvation_from_terminated_by():
    assert _c(_t(terminated_by="max_turns")) == "Turn starvation"


def test_turn_starvation_from_text():
    assert _c(_t(error_summary="hit max turns limit")) == "Turn starvation"


def test_environment_fault():
    assert _c(_t(terminated_by="env_error")) == "Environment fault"


def test_numeric_instability():
    assert _c(_t(error_summary="ZeroDivisionError: division by zero")) == "Numeric instability"
    assert _c(_t(error_summary="combined_score was nan")) == "Numeric instability"


def test_timeout():
    assert _c(_t(error_summary="Process timed out after 30s")) == "Timeout"


def test_dependency():
    assert _c(_t(error_summary="ModuleNotFoundError: no module named foo")) == "Dependency error"


def test_constraint_not_misread_as_numeric():
    # 'infeasible' contains the substring 'inf' but must NOT be Numeric instability.
    assert _c(_t(error_summary="solution infeasible: capacity exceeded")) == "Constraint violation"


def test_format_parse():
    assert _c(_t(error_summary="AgenticSolver produced no executable code")) == "Format/parse error"


def test_terminated_by_takes_precedence_over_text():
    # text says 'timed out' but the structured signal is max_turns → starvation wins
    assert _c(_t(error_summary="timed out waiting", terminated_by="max_turns")) == "Turn starvation"


def test_runtime_fallback():
    assert _c(_t(error_summary="ValueError: bad value at line 3")) == "Runtime error"


def test_empty_is_unknown():
    assert _c(_t()) == "Unknown error"


# --- the new Trace fields are additive and round-trip ---

def test_new_trace_fields_roundtrip():
    t = Trace(task_id="t", failure_class="Turn starvation", terminated_by="max_turns")
    t2 = Trace.model_validate(t.model_dump())
    assert t2.failure_class == "Turn starvation"
    assert t2.terminated_by == "max_turns"


def test_legacy_trace_without_new_fields_defaults():
    t = Trace.model_validate({"task_id": "t", "success": True, "score": 1.0})
    assert t.failure_class == ""
    assert t.terminated_by == ""
