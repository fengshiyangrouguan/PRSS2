"""Regression tests for the SPINE cost-integrity fix (NaN/negative cost).

Finding (F2/F5): the ``x or 0.0`` sanitization idiom on the native-USD
ingestion path does NOT coerce NaN (NaN is truthy) or a negative value, so a
non-finite/negative agent cost propagates verbatim into the on-disk ledger.
Once a single ``"cost_usd": NaN`` line is appended, ``today_total_usd()`` sums
it and returns NaN for the rest of the day, and BOTH spine gates fail OPEN
(``precheck`` and ``headroom_exhausted`` compare against NaN, which is always
False), silently disabling the daily cap for the whole day and across restart.

The fix has three points, each covered below:
  1. Ingestion clamp at the choke points ``CostGuard.record`` /
     ``CostTracker.record_usd`` (non-finite/negative -> $0).
  2. Gates fail CLOSED: ``CostGuard.precheck`` / ``headroom_exhausted`` treat a
     non-finite ``today_total_usd()`` as exhausted/deny.
  3. Read-side defense: ``today_total_usd()`` skips an already-persisted
     non-finite per-line cost (``json.loads`` parses literal ``NaN`` without
     raising, so the existing except does not catch it).

All tests are fully offline: no Docker, no LLM, no network. They exercise the
real ``CostTracker`` against a tmp ledger directory and the real ``CostGuard``.
"""

from __future__ import annotations

import json
import math

from meta_n.core.external_agents.backend import AgentRunResult
from meta_n.core.external_agents.budget import CostGuard
from meta_n.utils.cost_tracker import CostTracker

# A model that IS in PRICING (self-hosted $0 entry) so CostGuard's fail-fast
# construction check passes without needing a network or an override env var.
_PRICEABLE_MODEL = "google/gemma-4-31b-qat"


def _make_tracker(tmp_path, cap: float = 100.0, reservation: float = 1.0) -> CostTracker:
    return CostTracker(
        ledger_dir=tmp_path / "ledger",
        daily_cap_usd=cap,
        reservation_usd=reservation,
    )


# --- Point 1: ingestion clamp ------------------------------------------------


def test_record_usd_clamps_nan_to_zero(tmp_path):
    tracker = _make_tracker(tmp_path)
    tracker.record_usd(_PRICEABLE_MODEL, float("nan"))
    total = tracker.today_total_usd()
    # Pre-fix: NaN is written verbatim and the sum is NaN.
    assert math.isfinite(total)
    assert total == 0.0


def test_record_usd_clamps_negative_to_zero(tmp_path):
    tracker = _make_tracker(tmp_path)
    tracker.record_usd(_PRICEABLE_MODEL, -5.0)
    total = tracker.today_total_usd()
    # Pre-fix: -5.0 is truthy, survives ``or 0.0``, and manufactures phantom
    # headroom by subtracting from the day total.
    assert total == 0.0


def test_record_usd_clamps_inf_to_zero(tmp_path):
    tracker = _make_tracker(tmp_path)
    tracker.record_usd(_PRICEABLE_MODEL, float("inf"))
    total = tracker.today_total_usd()
    assert math.isfinite(total)
    assert total == 0.0


def test_record_usd_happy_path_unchanged(tmp_path):
    # HAPPY PATH: a normal finite cost is recorded byte-identically.
    tracker = _make_tracker(tmp_path)
    ret = tracker.record_usd(_PRICEABLE_MODEL, 1.234567, prompt_tokens=10)
    assert ret == 1.234567
    assert tracker.today_total_usd() == round(1.234567, 6)
    line = tracker._today_path().read_text().splitlines()[0]  # noqa: SLF001
    rec = json.loads(line)
    assert rec["cost_usd"] == round(1.234567, 6)
    assert rec["prompt_tokens"] == 10


def test_record_usd_zero_cost_local_model_byte_identical(tmp_path):
    # $0 local-model path must stay byte-identical (cost stays 0.0, one line).
    tracker = _make_tracker(tmp_path)
    tracker.record_usd(_PRICEABLE_MODEL, 0.0)
    rec = json.loads(tracker._today_path().read_text().splitlines()[0])  # noqa: SLF001
    assert rec["cost_usd"] == 0.0
    assert tracker.today_total_usd() == 0.0


def test_costguard_record_clamps_nan_native_cost(tmp_path):
    tracker = _make_tracker(tmp_path)
    guard = CostGuard(tracker, model=_PRICEABLE_MODEL)
    run = AgentRunResult(cost_usd=float("nan"), cost_basis="native_usd")
    guard.record(run)
    total = tracker.today_total_usd()
    # Pre-fix: the NaN OpenHands cost poisons the ledger and today_total_usd().
    assert math.isfinite(total)
    assert total == 0.0


def test_costguard_record_native_happy_path(tmp_path):
    # HAPPY PATH: a normal native USD cost is folded in verbatim.
    tracker = _make_tracker(tmp_path)
    guard = CostGuard(tracker, model=_PRICEABLE_MODEL)
    run = AgentRunResult(cost_usd=0.5, cost_basis="native_usd")
    guard.record(run)
    assert tracker.today_total_usd() == round(0.5, 6)


# --- Point 2: gates fail CLOSED on a non-finite today ------------------------


def test_precheck_denies_when_today_is_non_finite(tmp_path, monkeypatch):
    tracker = _make_tracker(tmp_path)
    guard = CostGuard(tracker, model=_PRICEABLE_MODEL)
    monkeypatch.setattr(tracker, "today_total_usd", lambda: float("nan"))
    # Pre-fix: headroom = threshold - NaN = NaN; ``NaN <= 0.0`` is False, so the
    # run is ADMITTED (returns False). Post-fix: fail closed -> deny (True).
    assert guard.precheck(2.0) is True


def test_headroom_exhausted_true_when_today_is_non_finite(tmp_path, monkeypatch):
    tracker = _make_tracker(tmp_path)
    guard = CostGuard(tracker, model=_PRICEABLE_MODEL)
    monkeypatch.setattr(tracker, "today_total_usd", lambda: float("nan"))
    # Pre-fix: ``NaN >= (cap - reservation)`` is False -> not exhausted, dispatch
    # continues unbounded. Post-fix: treat as exhausted (True).
    assert guard.headroom_exhausted() is True


def test_precheck_happy_path_admits(tmp_path):
    # HAPPY PATH: a finite under-threshold day admits the run.
    tracker = _make_tracker(tmp_path)
    guard = CostGuard(tracker, model=_PRICEABLE_MODEL)
    assert guard.precheck(2.0) is False


def test_headroom_exhausted_happy_path_false(tmp_path):
    tracker = _make_tracker(tmp_path)
    guard = CostGuard(tracker, model=_PRICEABLE_MODEL)
    assert guard.headroom_exhausted() is False


# --- Point 3: read-side defense against an already-persisted NaN line --------


def test_today_total_skips_persisted_nan_line(tmp_path):
    tracker = _make_tracker(tmp_path)
    path = tracker._today_path()  # noqa: SLF001
    path.parent.mkdir(parents=True, exist_ok=True)
    # Emulate a poisoned ledger written by an older/buggy build: json.dumps with
    # the default allow_nan=True serialises float('nan') to the literal ``NaN``,
    # and json.loads parses it back WITHOUT raising.
    good = json.dumps({"model": "m", "cost_usd": 2.5})
    poisoned = json.dumps({"model": "m", "cost_usd": float("nan")})
    inf_line = json.dumps({"model": "m", "cost_usd": float("inf")})
    good2 = json.dumps({"model": "m", "cost_usd": 1.5})
    path.write_text("\n".join([good, poisoned, inf_line, good2]) + "\n")
    total = tracker.today_total_usd()
    # Pre-fix: total is NaN (the poisoned line taints the sum). Post-fix: the two
    # non-finite lines are skipped and only the finite costs are summed.
    assert math.isfinite(total)
    assert total == 4.0
