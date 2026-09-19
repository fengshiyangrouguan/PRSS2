"""TerminatedBy mapping tables — coverage + unknown fallback (§12.1)."""

from __future__ import annotations

import pytest

from meta_n.core.external_agents.terminated import (
    _OH_STATUS_TO_ENUM,
    _T2_FAILURE_TO_ENUM,
    TerminatedBy,
    from_oh_status,
    from_t2_failure,
)


# The enum members the backends must each be able to *produce* via their tables.
# (BUDGET_DENIED is set by the spine pre-check, not a backend native status, and
# UNKNOWN is the fallback — both are still reachable, asserted separately.)
_CORE_MEMBERS = {
    TerminatedBy.COMPLETED,
    TerminatedBy.MAX_TURNS,
    TerminatedBy.TIMEOUT,
    TerminatedBy.ENV_ERROR,
    TerminatedBy.PARSE_ERROR,
    TerminatedBy.CONTEXT_LEN,
    TerminatedBy.AGENT_ERROR,
}


def test_oh_table_covers_core_members():
    produced = set(_OH_STATUS_TO_ENUM.values())
    # OpenHands' in-run cap is a USD ceiling, so its soft budget stop maps to
    # BUDGET_USD (not TOKEN_BUDGET); assert that member is produced too.
    expected = _CORE_MEMBERS | {TerminatedBy.BUDGET_USD}
    missing = expected - produced
    assert not missing, f"OH table never produces: {missing}"


def test_t2_table_covers_core_members():
    produced = set(_T2_FAILURE_TO_ENUM.values())
    # T2 has no ENV_ERROR-from-agent in core flow except test_timeout; include it.
    # T2's inner budget kill (the _BudgetedLiteLLM wrapper) emits 'token_budget'.
    core = _CORE_MEMBERS | {TerminatedBy.TOKEN_BUDGET}
    missing = core - produced
    assert not missing, f"T2 table never produces: {missing}"


def test_oh_table_covers_every_classifier_token():
    """REGRESSION: every token ``_classify_oh_error`` can emit MUST be a key in
    ``_OH_STATUS_TO_ENUM``, so a future catch-all token can never silently fall
    through to UNKNOWN (the original ``agent_error`` -> UNKNOWN gap)."""
    from meta_n.core.external_agents.backends.openhands import _classify_oh_error

    # The full set of failure_mode tokens `_classify_oh_error` can return.
    emit_tokens = {
        "budget_exhausted",
        "max_iterations",
        "context_window_exceeded",
        "parse_error",
        "env_error",
        "agent_timeout",
        "agent_error",
    }
    # Cross-check the literal-return set is reachable by reading the function: feed
    # representative exceptions and confirm the returned tokens are all in-table.
    samples = [
        RuntimeError("budget exceeded"),
        RuntimeError("max_iterations reached"),
        RuntimeError("context window length exceeded"),
        ValueError("invalid json parse error"),
        ConnectionError("connection refused to server"),
        TimeoutError("operation timeout"),
        RuntimeError("agent stuck in a loop"),  # catch-all -> agent_error
    ]
    for exc in samples:
        emit_tokens.add(_classify_oh_error(exc))
    for tok in emit_tokens:
        assert tok in _OH_STATUS_TO_ENUM, (
            f"_classify_oh_error can emit {tok!r} but it is not a key in "
            f"_OH_STATUS_TO_ENUM (would silently fall through to UNKNOWN)"
        )


def test_every_terminatedby_value_is_a_string():
    for m in TerminatedBy:
        assert isinstance(m.value, str)
        # str-backed enum: the member IS its JSON value.
        assert str(m) == m.value


def test_aliases_resolve_to_canonical_members():
    assert TerminatedBy.CONFIRMED is TerminatedBy.COMPLETED
    assert TerminatedBy.PERFECT_SCORE is TerminatedBy.COMPLETED
    assert TerminatedBy.CONTEXT is TerminatedBy.CONTEXT_LEN


# --- from_oh_status --------------------------------------------------------


@pytest.mark.parametrize(
    "status,expected",
    [
        ("finished", TerminatedBy.COMPLETED),
        ("error", TerminatedBy.AGENT_ERROR),
        ("stuck", TerminatedBy.AGENT_ERROR),  # genuine failure (looped)
        # The runner rewrites the SDK's overloaded ERROR+MaxIterationsReached to
        # this status so the iteration cap is MAX_TURNS, not AGENT_ERROR.
        ("max_iterations", TerminatedBy.MAX_TURNS),
        ("agent_timeout", TerminatedBy.TIMEOUT),
        ("context_window_exceeded", TerminatedBy.CONTEXT_LEN),
        ("budget_exhausted", TerminatedBy.BUDGET_USD),  # OH USD cap (soft)
        ("agent_error", TerminatedBy.AGENT_ERROR),  # classifier catch-all
        ("FINISHED", TerminatedBy.COMPLETED),  # case-insensitive
    ],
)
def test_from_oh_status_known(status, expected):
    assert from_oh_status(status) is expected


def test_from_oh_status_unknown_is_unknown():
    assert from_oh_status("not-a-real-status") is TerminatedBy.UNKNOWN
    assert from_oh_status(None) is TerminatedBy.UNKNOWN
    assert from_oh_status("") is TerminatedBy.UNKNOWN


# --- OH terminated_by nuance: finished/max-turns/error are DISTINCT ---------


def test_oh_finished_run_is_completed_not_agent_error():
    """REGRESSION (OH terminated_by footgun): a FINISHED run (no exception,
    failure_mode=None) that merely scored low must telemeter COMPLETED — never
    AGENT_ERROR. ``idle`` (the post-run resting state) is also a clean stop."""
    assert from_oh_status("finished") is TerminatedBy.COMPLETED
    assert from_oh_status("idle") is TerminatedBy.COMPLETED


def test_oh_max_iterations_is_max_turns_not_agent_error():
    """REGRESSION: the SDK overloads ``execution_status=ERROR`` for BOTH a real
    fault AND hitting ``max_iteration_per_run``. The runner disambiguates the
    iteration cap to the ``max_iterations`` token, which must map to MAX_TURNS
    (the agent hit the ceiling), NOT AGENT_ERROR (a genuine failure)."""
    assert from_oh_status("max_iterations") is TerminatedBy.MAX_TURNS


def test_oh_genuine_error_is_agent_error():
    """A real in-step exception (status stays the raw ``error`` token, or the
    classifier's ``agent_error`` catch-all) is the ONLY path to AGENT_ERROR."""
    assert from_oh_status("error") is TerminatedBy.AGENT_ERROR
    assert from_oh_status("agent_error") is TerminatedBy.AGENT_ERROR


def test_oh_term_for_finished_and_caps_distinct():
    """``_term_for`` (the backend's soft-failure-mode mapper) agrees with the
    table: None (success) -> COMPLETED, max_iterations -> MAX_TURNS."""
    from meta_n.core.external_agents.backends.openhands import _term_for

    assert _term_for(None) is TerminatedBy.COMPLETED
    assert _term_for("max_iterations") is TerminatedBy.MAX_TURNS
    assert _term_for("agent_error") is TerminatedBy.AGENT_ERROR


# --- from_t2_failure -------------------------------------------------------


@pytest.mark.parametrize(
    "fm,expected",
    [
        (None, TerminatedBy.COMPLETED),  # success path
        ("agent_timeout", TerminatedBy.TIMEOUT),
        ("token_budget", TerminatedBy.TOKEN_BUDGET),
        ("context_length_exceeded", TerminatedBy.CONTEXT_LEN),
        ("max_episodes", TerminatedBy.MAX_TURNS),
        ("fatal_llm_parse_error", TerminatedBy.PARSE_ERROR),
    ],
)
def test_from_t2_failure_known(fm, expected):
    assert from_t2_failure(fm) is expected


def test_from_t2_failure_unknown_is_unknown():
    assert from_t2_failure("nonsense-mode") is TerminatedBy.UNKNOWN


def test_from_t2_failure_enum_like_value():
    class FM:
        value = "agent_timeout"

    assert from_t2_failure(FM()) is TerminatedBy.TIMEOUT
