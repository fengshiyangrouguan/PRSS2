"""Regression tests for audit findings 15 and 45 in cost_tracker.py.

Both concern the 80% soft-warn banner in ``CostTracker.assert_under_cap``:

* Finding 15: the soft-warn point was a fixed ``0.80 * daily_cap_usd``,
  decoupled from the effective threshold ``daily_cap_usd - reservation_usd``.
  When ``reservation_usd >= 0.20 * daily_cap_usd`` the warn interval is empty
  and the banner can never precede the hard stop. Fix anchors the warn to
  ``0.80 * threshold``.
* Finding 45: the ``_warned_80pct`` latch was process-lifetime and never reset
  on a day rollover, so a multi-day process warns only on day 1. Fix clears the
  latch when the local day key changes.

All tests are offline (no LLM / Docker / network): they write tiny JSONL ledger
files directly and inspect ``logging`` records.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from meta_n.utils.cost_tracker import CostTracker


def _write_ledger(path: Path, cost_usd: float) -> None:
    path.write_text(json.dumps({"cost_usd": cost_usd}) + "\n", encoding="utf-8")


def _warn_banners(caplog) -> list[logging.LogRecord]:
    return [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and "(>=80%" in r.getMessage()
    ]


def test_soft_warn_precedes_hard_stop_with_reservation(tmp_path, caplog):
    """Finding 15: with a reservation >= 20% of the cap, the 80% banner must
    still fire on a spend that is below the hard-stop threshold.

    cap=5, reservation=1.0 -> threshold=4.0. Fixed warn_at = 0.80*4.0 = 3.2.
    A spend of $3.5 is above warn_at (warns) but below threshold (no raise).
    On the original code warn_at = 5*0.80 = 4.0 > 3.5, so no banner fires at
    all before the hard stop -> this test fails on the un-fixed code.
    """
    tracker = CostTracker(tmp_path, daily_cap_usd=5.0, reservation_usd=1.0)
    _write_ledger(tracker._today_path(), 3.5)

    with caplog.at_level(logging.WARNING, logger="meta_n.utils.cost_tracker"):
        tracker.assert_under_cap()  # must NOT raise (3.5 < threshold 4.0)

    assert _warn_banners(caplog), (
        "80% soft-warn banner should fire before the hard stop when a "
        "non-trivial reservation pulls the threshold below 0.8*cap"
    )


def test_soft_warn_resets_across_day_rollover(tmp_path, caplog):
    """Finding 45: the soft-warn latch must reset on a day rollover so the
    banner fires once per day rather than once per process."""
    tracker = CostTracker(tmp_path, daily_cap_usd=5.0, reservation_usd=0.0)

    # --- Day 1 ---
    tracker._today_key = lambda: "2026-01-01"  # type: ignore[method-assign]
    _write_ledger(tracker._today_path(), 4.5)  # >= 0.8*5 = 4.0, < 5.0
    with caplog.at_level(logging.WARNING, logger="meta_n.utils.cost_tracker"):
        tracker.assert_under_cap()
        tracker.assert_under_cap()  # same day: must NOT re-warn (latch)
    day1 = _warn_banners(caplog)
    assert len(day1) == 1, "expected exactly one banner on day 1"

    caplog.clear()

    # --- Day 2 (rollover): fresh daily budget, latch must reset ---
    tracker._today_key = lambda: "2026-01-02"  # type: ignore[method-assign]
    _write_ledger(tracker._today_path(), 4.5)
    with caplog.at_level(logging.WARNING, logger="meta_n.utils.cost_tracker"):
        tracker.assert_under_cap()
    assert _warn_banners(caplog), (
        "soft-warn banner should fire again on day 2 after a day rollover"
    )


def test_hard_stop_unchanged_for_default_scale(tmp_path):
    """Sanity: the hard-stop enforcement (the actual budget guard) is
    unaffected by the warn-point change -- still raises at the threshold."""
    from meta_n.utils.cost_tracker import BudgetExceededError

    tracker = CostTracker(tmp_path, daily_cap_usd=5.0, reservation_usd=1.0)
    _write_ledger(tracker._today_path(), 4.0)  # == threshold (5 - 1)
    with pytest.raises(BudgetExceededError):
        tracker.assert_under_cap()
