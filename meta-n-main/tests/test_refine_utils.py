"""Regression tests for the C8_utils refinement wave (B10b specs).

Covers:
  * F142 — azure_compat._record_from_response: generic record errors are
    swallowed; a BudgetExceededError would propagate (BaseException contract).
  * F146/F186 — flock_append_bytes consolidation: ledger / llm_io lines are
    byte-identical to the pre-refactor implementation; short-write loop and
    cross-process no-torn-lines behavior preserved at the helper level.
  * F150 — list_azure_deployments._format_table consolidation: golden strings
    frozen from the pre-refactor implementation.
  * F154 — CostTracker fail-fast on reservation >= cap / negative reservation.
  * F155 — cost_report: non-finite cost_usd cannot poison totals; --utc flag.
  * F163 — pricing-override parse cache (raw-keyed) + no-file total regression.

All tests are pure-local; no Azure / OpenAI / Docker calls.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import meta_n.utils.cost_tracker as cost_tracker_mod
import meta_n.utils.flock_append as flock_append_mod
from meta_n.utils.azure_compat import _record_from_response, install_cost_tracking
from meta_n.utils.cost_report import _read_ledger, _summarise, _today
from meta_n.utils.cost_tracker import (
    BudgetExceededError,
    CostTracker,
    get_pricing,
)
from meta_n.utils.flock_append import flock_append_bytes
from meta_n.utils.list_azure_deployments import _format_models, _format_probes
from meta_n.utils.llm_io_logger import LLMIOLogger


# ---------------------------------------------------------------------------
# F142 — _record_from_response error containment
# ---------------------------------------------------------------------------

def _fake_response(prompt_tokens=10, completion_tokens=5):
    return SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            prompt_tokens_details=None,
        )
    )


class _RaisingTracker:
    """Tracker stub whose record() raises a chosen exception."""

    def __init__(self, exc: BaseException):
        self._exc = exc

    def record(self, **kwargs):
        raise self._exc


def test_record_from_response_swallows_generic_record_errors(caplog):
    """A RuntimeError from tracker.record must not escape the accounting
    path — the call returns normally with a warning logged."""
    import logging
    tracker = _RaisingTracker(RuntimeError("disk full"))
    with caplog.at_level(logging.WARNING, logger="meta_n.utils.azure_compat"):
        _record_from_response(tracker, "gpt-4.1", _fake_response())
    assert any("failed to record cost" in r.message for r in caplog.records)


def test_record_from_response_would_not_swallow_budget_error():
    """BudgetExceededError derives from BaseException, so the accounting
    path's ``except Exception`` cannot absorb it — even without an explicit
    re-raise clause (removed as dead code in F142)."""
    tracker = _RaisingTracker(BudgetExceededError("cap hit"))
    with pytest.raises(BudgetExceededError):
        _record_from_response(tracker, "gpt-4.1", _fake_response())


# ---------------------------------------------------------------------------
# F146/F186 — persisted ledger line byte-identity after consolidation
# ---------------------------------------------------------------------------

def test_record_line_bytes_exact(tmp_path):
    """Exact bytes frozen from the pre-refactor record() implementation."""
    t = CostTracker(ledger_dir=tmp_path, daily_cap_usd=100.0)
    with patch("meta_n.utils.cost_tracker.time.time", return_value=1234567890.5), \
         patch("meta_n.utils.cost_tracker.os.getpid", return_value=4242):
        t.record("gpt-4.1", 1000, 200, cached_tokens=50, extra={"task": "test"})
    path = next(Path(tmp_path).glob("*.jsonl"))
    assert path.read_bytes() == (
        b'{"ts": 1234567890.5, "model": "gpt-4.1", "prompt_tokens": 1000, '
        b'"completion_tokens": 200, "cached_tokens": 50, "cost_usd": 0.0036, '
        b'"pid": 4242, "extra": {"task": "test"}}\n'
    )


def test_record_usd_line_bytes_exact(tmp_path):
    """Exact bytes frozen from the pre-refactor record_usd() implementation,
    including ``extra`` key ordering."""
    t = CostTracker(ledger_dir=tmp_path, daily_cap_usd=100.0)
    with patch("meta_n.utils.cost_tracker.time.time", return_value=1234567890.5), \
         patch("meta_n.utils.cost_tracker.os.getpid", return_value=4242):
        t.record_usd(
            "custom-ext-model", 1.234567891,
            prompt_tokens=10, completion_tokens=5, cached_tokens=1,
            extra={"source": "openhands", "basis": "native_usd"},
        )
    path = next(Path(tmp_path).glob("*.jsonl"))
    assert path.read_bytes() == (
        b'{"ts": 1234567890.5, "model": "custom-ext-model", "prompt_tokens": 10, '
        b'"completion_tokens": 5, "cached_tokens": 1, "cost_usd": 1.234568, '
        b'"pid": 4242, "extra": {"source": "openhands", "basis": "native_usd"}}\n'
    )


# ---------------------------------------------------------------------------
# F146 — flock_append_bytes helper semantics
# ---------------------------------------------------------------------------

class _OsProxy:
    """Forward everything to the real os module except ``write``. Scoped
    replacement of flock_append's ``os`` attribute — avoids patching the
    process-global os.write."""

    def __init__(self, write_fn):
        self._write_fn = write_fn

    def write(self, fd, data):
        return self._write_fn(fd, data)

    def __getattr__(self, name):
        return getattr(os, name)


def test_short_writes_are_looped(tmp_path, monkeypatch):
    """POSIX permits partial writes; the helper must loop until every byte
    lands (1 byte per os.write call here)."""
    real_write = os.write
    monkeypatch.setattr(
        flock_append_mod, "os",
        _OsProxy(lambda fd, data: real_write(fd, data[:1])),
    )
    target = tmp_path / "short.jsonl"
    payload = b'{"k": "0123456789abcdef"}\n'
    flock_append_bytes(target, payload)
    assert target.read_bytes() == payload


def test_zero_write_raises_oserror(tmp_path, monkeypatch):
    """A zero-byte write (closed fd / full disk) must raise, not spin."""
    monkeypatch.setattr(
        flock_append_mod, "os", _OsProxy(lambda fd, data: 0),
    )
    with pytest.raises(OSError, match=r"os\.write returned 0 after"):
        flock_append_bytes(tmp_path / "zero.jsonl", b"payload\n")


def _flock_append_worker(path: str, tag: str, n_lines: int):
    """Module-level (picklable for spawn) appender used by the mp test."""
    from meta_n.utils.flock_append import flock_append_bytes as fab
    payload = (json.dumps({"tag": tag, "fill": "x" * 120}) + "\n").encode("utf-8")
    for _ in range(n_lines):
        fab(path, payload)


def test_concurrent_appends_no_torn_lines(tmp_path):
    """Two processes appending through the helper produce only whole lines
    (helper-level mirror of the CostTracker cross-process test)."""
    target = tmp_path / "concurrent.jsonl"
    per_proc = 100
    ctx = mp.get_context("spawn")
    procs = [
        ctx.Process(target=_flock_append_worker, args=(str(target), tag, per_proc))
        for tag in ("a", "b")
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=30)
        assert p.exitcode == 0
    lines = [ln for ln in target.read_text().splitlines() if ln.strip()]
    assert len(lines) == 2 * per_proc
    tags = [json.loads(ln)["tag"] for ln in lines]  # every line parseable
    assert sorted(set(tags)) == ["a", "b"]


# ---------------------------------------------------------------------------
# F146 — LLMIOLogger still appends parseable lines and swallows errors
# ---------------------------------------------------------------------------

def test_log_appends_parseable_line_and_swallows_errors(tmp_path, monkeypatch, capsys):
    target = tmp_path / "llm_io" / "io.jsonl"
    logger = LLMIOLogger(target, source="test")
    logger.log(
        messages=[{"role": "user", "content": "hi"}],
        response="ok", model="m", prompt_tokens=1, completion_tokens=2,
        total_tokens=3, extra={"k": "v"},
    )
    lines = target.read_text().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["source"] == "test"
    assert rec["messages"] == [{"role": "user", "content": "hi"}]
    assert rec["response"] == "ok"
    assert rec["extra"] == {"k": "v"}

    # An append failure must be swallowed (stderr warning only) — losing a
    # log line must never sink a run.
    def _boom(path, data):
        raise RuntimeError("append blew up")

    monkeypatch.setattr("meta_n.utils.llm_io_logger.flock_append_bytes", _boom)
    logger.log(response="second")  # must not raise
    assert "failed to append" in capsys.readouterr().err
    assert len(target.read_text().splitlines()) == 1  # nothing new landed


# ---------------------------------------------------------------------------
# F150 — table formatter goldens (frozen from the pre-refactor output)
# ---------------------------------------------------------------------------

def test_format_models_golden():
    models = [
        {"id": "gpt-4.1", "status": "succeeded"},
        {"id": "some-unpriced-model", "status": "preview"},
        {"id": "gpt-5.2", "status": "succeeded"},
    ]
    assert _format_models(models) == (
        "Model id             Status     $ in / out / cached / 1M\n"
        "-------------------  ---------  ------------------------\n"
        "gpt-4.1              succeeded  $2.00 / $8.00           \n"
        "some-unpriced-model  preview    (unknown)               \n"
        "gpt-5.2              succeeded  $1.75 / $14.00 / $0.175 "
    )


def test_format_probes_golden():
    probes = [
        {"deployment": "gpt-4.1", "deployed": True, "info": "gpt-4.1-2025-04-14"},
        {"deployment": "gpt-5", "deployed": False, "info": "404 (no such deployment)"},
    ]
    assert _format_probes(probes) == (
        "Deployment  Deployed?  Notes / model echoed    \n"
        "----------  ---------  ------------------------\n"
        "gpt-4.1     yes        gpt-4.1-2025-04-14      \n"
        "gpt-5       no         404 (no such deployment)"
    )


def test_format_empty_cases():
    assert _format_models([]) == "(no chat-capable models returned)"
    assert _format_probes([]) == "(no deployment probes ran)"


# ---------------------------------------------------------------------------
# F154 — reservation misconfiguration fails fast at construction
# ---------------------------------------------------------------------------

def test_reservation_at_cap_raises_valueerror(tmp_path):
    with pytest.raises(ValueError, match="must be smaller than"):
        CostTracker(ledger_dir=tmp_path, daily_cap_usd=1.0, reservation_usd=1.0)


def test_reservation_above_cap_raises_valueerror(tmp_path):
    with pytest.raises(ValueError, match="must be smaller than"):
        CostTracker(ledger_dir=tmp_path, daily_cap_usd=0.5, reservation_usd=1.0)


def test_negative_reservation_raises_valueerror(tmp_path):
    with pytest.raises(ValueError, match="must be >= 0"):
        CostTracker(ledger_dir=tmp_path, daily_cap_usd=10.0, reservation_usd=-0.1)


def test_install_cost_tracking_small_budget_fails_fast(tmp_path):
    """A sub-$1 daily budget with azure_compat's default $1 reservation must
    raise a clear ValueError at install time, NOT a misleading
    BudgetExceededError ('$0.0000 >= $-0.5000') on the first call."""
    from unittest.mock import MagicMock
    client = MagicMock()
    client.chat.completions.create = lambda **kw: None
    # Ensure the sentinel identity check isn't tripped by MagicMock.
    setattr(client, "_meta_n_cost_tracking_installed", None)
    with pytest.raises(ValueError, match="must be smaller than"):
        install_cost_tracking(
            client, model="gpt-4.1", daily_budget_usd=0.5,
            cost_ledger_dir=str(tmp_path),
        )


def test_valid_reservation_still_constructs_and_passes_empty_cap_check(tmp_path):
    t = CostTracker(ledger_dir=tmp_path, daily_cap_usd=10.0, reservation_usd=1.0)
    t.assert_under_cap()  # empty ledger → no raise
    # cap <= 0 keeps its pre-guard behavior (no new failure mode).
    CostTracker(ledger_dir=tmp_path, daily_cap_usd=0.0, reservation_usd=2.0)


# ---------------------------------------------------------------------------
# F155 — cost_report NaN hardening + --utc day selection
# ---------------------------------------------------------------------------

def test_summarise_skips_nonfinite_cost():
    rows = [
        {"model": "m", "prompt_tokens": 10, "completion_tokens": 5,
         "cached_tokens": 0, "cost_usd": 1.5},
        {"model": "m", "cost_usd": float("nan")},
        {"model": "m", "cost_usd": float("inf")},
        {"model": "m", "cost_usd": "not_a_number"},
    ]
    s = _summarise(rows)
    # Poisoned rows still count as calls; only their cost contributes 0.
    assert s["total"]["calls"] == 4
    assert s["by_model"]["m"]["calls"] == 4
    assert s["total"]["cost_usd"] == pytest.approx(1.5)
    assert s["by_model"]["m"]["cost_usd"] == pytest.approx(1.5)


def test_summarise_matches_tracker_total_on_poisoned_ledger(tmp_path):
    """The report and the enforcement path must agree about a poisoned file."""
    t = CostTracker(ledger_dir=tmp_path, daily_cap_usd=100.0)
    t.record("gpt-4.1", 1000, 200)
    path = next(Path(tmp_path).glob("*.jsonl"))
    with open(path, "a") as f:
        f.write('{"model": "x", "cost_usd": NaN}\n')  # json.loads parses NaN
    summary = _summarise(_read_ledger(path))
    assert summary["total"]["cost_usd"] == pytest.approx(t.today_total_usd())


def test_utc_flag_selects_utc_day_file(tmp_path, monkeypatch, capsys):
    # Unit level: _today returns a date, UTC vs local basis.
    assert isinstance(_today(True), date)
    assert _today(True) == datetime.now(timezone.utc).date()
    assert _today(False) == date.today()
    # CLI level: --utc selects the UTC day's ledger file path.
    from meta_n.utils.cost_report import main
    monkeypatch.setattr(
        "sys.argv",
        ["cost_report", "--utc", "--ledger-dir", str(tmp_path)],
    )
    main()
    out = capsys.readouterr().out
    assert datetime.now(timezone.utc).date().isoformat() in out


# ---------------------------------------------------------------------------
# F163 — pricing-override parse cache + no-file total regression
# ---------------------------------------------------------------------------

class _CountingJson:
    """Forward everything to the real json module, counting loads() calls."""

    def __init__(self):
        self.loads_count = 0

    def loads(self, *args, **kwargs):
        self.loads_count += 1
        return json.loads(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(json, name)


def test_pricing_override_cache_hits_on_same_raw(monkeypatch):
    monkeypatch.setattr(cost_tracker_mod, "_PRICING_OVERRIDE_CACHE", None)
    monkeypatch.setenv(
        "META_N_PRICING_OVERRIDE_JSON",
        '{"cache-model": {"input": 1.5, "output": 6.0}}',
    )
    counter = _CountingJson()
    monkeypatch.setattr(cost_tracker_mod, "json", counter)
    assert get_pricing("cache-model").input_per_M == 1.5
    assert get_pricing("cache-model").input_per_M == 1.5
    assert counter.loads_count == 1  # second call served from the raw-keyed cache


def test_pricing_override_cache_invalidates_on_env_change(monkeypatch):
    monkeypatch.setattr(cost_tracker_mod, "_PRICING_OVERRIDE_CACHE", None)
    monkeypatch.setenv(
        "META_N_PRICING_OVERRIDE_JSON",
        '{"cache-model": {"input": 1.5, "output": 6.0}}',
    )
    assert get_pricing("cache-model").input_per_M == 1.5
    monkeypatch.setenv(
        "META_N_PRICING_OVERRIDE_JSON",
        '{"cache-model": {"input": 2.5, "output": 6.0}}',
    )
    assert get_pricing("cache-model").input_per_M == 2.5  # changed raw → re-parse


def test_malformed_override_warns_once_per_value(monkeypatch, caplog):
    import logging
    monkeypatch.setattr(cost_tracker_mod, "_PRICING_OVERRIDE_CACHE", None)
    monkeypatch.setenv("META_N_PRICING_OVERRIDE_JSON", "{not json")
    with caplog.at_level(logging.WARNING, logger="meta_n.utils.cost_tracker"):
        for _ in range(3):
            get_pricing("gpt-4.1")  # known model; override lookup runs first
    warns = [r for r in caplog.records
             if "malformed META_N_PRICING_OVERRIDE_JSON" in r.message]
    assert len(warns) == 1
    # A DIFFERENT malformed value is a cache miss and warns again.
    monkeypatch.setenv("META_N_PRICING_OVERRIDE_JSON", "{still not json")
    with caplog.at_level(logging.WARNING, logger="meta_n.utils.cost_tracker"):
        get_pricing("gpt-4.1")
    warns = [r for r in caplog.records
             if "malformed META_N_PRICING_OVERRIDE_JSON" in r.message]
    assert len(warns) == 2


def test_today_total_zero_when_file_absent(tmp_path):
    """Post-F163 regression: the no-file path is handled by the
    FileNotFoundError handler (the racy pre-check was removed)."""
    t = CostTracker(ledger_dir=tmp_path, daily_cap_usd=10.0)
    assert not any(Path(tmp_path).glob("*.jsonl"))
    assert t.today_total_usd() == 0.0
