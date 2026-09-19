"""Regression tests for audit fix #34 (terminated.py).

Finding 34: `_T2_FAILURE_TO_ENUM` carried a dead `None -> COMPLETED` key whose
docstring claimed the backend indexed the table with a bare ``None`` via
``.get(None, ...)``. No such lookup path exists: ``from_t2_failure`` short-circuits
to COMPLETED whenever ``failure_mode is None`` *before* touching the table, and
``_normalize`` never yields ``None`` as a lookup key for a present entry. The dead
key is removed; behavior on every reachable path is unchanged.

These tests are LLM-free / offline (pure import + dict/function assertions).
"""

from meta_n.core.external_agents.terminated import (
    TerminatedBy,
    _T2_FAILURE_TO_ENUM,
    from_t2_failure,
)


def test_t2_table_has_no_dead_none_key():
    """The dead bare-``None`` key must be gone (failed on un-fixed code)."""
    assert None not in _T2_FAILURE_TO_ENUM
    # every remaining key is a normalized lowercase string
    assert all(isinstance(k, str) for k in _T2_FAILURE_TO_ENUM)


def test_none_failure_mode_still_maps_to_completed():
    """Behavior on the success path is preserved by the short-circuit."""
    assert from_t2_failure(None) is TerminatedBy.COMPLETED


def test_none_token_still_covers_failuremode_none():
    """The ``"none"`` string token still resolves FailureMode.NONE to COMPLETED."""
    assert _T2_FAILURE_TO_ENUM["none"] is TerminatedBy.COMPLETED
    assert from_t2_failure("none") is TerminatedBy.COMPLETED
    assert from_t2_failure("NONE") is TerminatedBy.COMPLETED


def test_unmapped_failure_mode_falls_back_to_unknown():
    assert from_t2_failure("totally_unrecognized") is TerminatedBy.UNKNOWN
