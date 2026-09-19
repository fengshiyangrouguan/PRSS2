"""Refinement regression tests for meta_n/analysis/telemetry.py.

Covers:
- F162: ``unwrap_record`` is the public alias of the (retained) private
  ``_unwrap_record`` and round-trips both envelope shapes; it is exported via
  ``__all__`` for external tooling (scripts/build_ab_summary.py).

``meta_n.analysis.telemetry`` hard-depends on pandas (the ``analysis`` extra);
skipped wholesale when pandas is absent, mirroring
tests/external_agents/test_telemetry_load.py.
"""

from __future__ import annotations

import pytest

pytest.importorskip("pandas")  # analysis-extra dependency; skip if absent

from meta_n.analysis import telemetry  # noqa: E402
from meta_n.analysis.telemetry import _unwrap_record, unwrap_record  # noqa: E402


def test_unwrap_record_public_alias():
    assert unwrap_record is _unwrap_record
    assert "unwrap_record" in telemetry.__all__

    # LLMIOLogger envelope: the record is nested under extra.record.
    record = {"run_id": "r1", "agent": "terminus2", "score": 0.5}
    assert unwrap_record({"extra": {"record": record}}) == record

    # Bare top-level record (no envelope): accepted via the run_id key.
    bare = {"run_id": "r2", "agent": "builtin"}
    assert unwrap_record(bare) is bare


def test_unwrap_record_strictness_unchanged():
    """Read-side strictness is deliberate (F194 step 3 stays report-only):
    an un-keyable row (no extra.record, no run_id) is dropped, unlike the
    writer-side ``... or obj`` fallback in core external_agents telemetry."""
    assert unwrap_record({"message": "not a record"}) is None
    assert unwrap_record({"extra": {}}) is None
    assert unwrap_record("not a mapping") is None
