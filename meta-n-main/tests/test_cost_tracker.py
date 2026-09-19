"""Stress tests for ``meta_n.utils.cost_tracker``.

Covers what the README of the module promises:

* concurrent threads / processes do not corrupt the ledger
* the cap fires before an over-budget call lands
* ``BudgetExceededError`` is a ``BaseException`` subclass and so passes
  through ``except Exception`` blocks
* ``today_total_usd`` reads only the current day's file (no spillover
  from yesterday)
* pricing math is exact for known models

These are pure-local tests; no Azure / OpenAI calls are made.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import threading
import time
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from meta_n.utils.cost_tracker import (
    PRICING,
    BudgetExceededError,
    CostTracker,
    ModelPricing,
    compute_cost_usd,
    get_pricing,
)


# ---------------------------------------------------------------------------
# Pricing math
# ---------------------------------------------------------------------------

def test_pricing_table_complete():
    """Sanity: the documented Azure deployments are in the table."""
    for name in ("gpt-4.1", "gpt-5.2", "gpt-4.1-mini"):
        assert name in PRICING


def test_compute_cost_usd_gpt41_exact():
    # 1M input @ $2 + 1M output @ $8 = $10
    assert compute_cost_usd("gpt-4.1", 1_000_000, 1_000_000) == pytest.approx(10.0)
    # 14p + 2c (matches our smoke-test reading)
    expected = 14 / 1e6 * 2.0 + 2 / 1e6 * 8.0
    assert compute_cost_usd("gpt-4.1", 14, 2) == pytest.approx(expected)


def test_compute_cost_usd_cached_discount():
    # gpt-5.2: $1.75 input, $0.175 cached, $14 output
    # 100K total prompt, of which 100K cached, 50K output
    # = 0 uncached × 1.75 + 100K × 0.175 + 50K × 14 (per million)
    expected = 0 + 100_000 / 1e6 * 0.175 + 50_000 / 1e6 * 14.0
    assert compute_cost_usd("gpt-5.2", 100_000, 50_000, cached_tokens=100_000) == pytest.approx(expected)


def test_compute_cost_usd_cached_clamps_to_prompt():
    # cached > prompt should clamp at prompt (Azure won't bill negatively)
    cost = compute_cost_usd("gpt-5.2", 100, 50, cached_tokens=10_000)
    # All 100 prompt tokens cached, 50 completion at full rate
    expected = 100 / 1e6 * 0.175 + 50 / 1e6 * 14.0
    assert cost == pytest.approx(expected)


def test_unknown_model_raises_with_hint():
    with pytest.raises(KeyError, match="No pricing for model"):
        compute_cost_usd("not-a-real-model", 100, 100)


def test_pricing_override_env(monkeypatch):
    monkeypatch.setenv("META_N_PRICING_OVERRIDE_JSON",
                       '{"my-deployment": {"input": 1.5, "output": 6.0}}')
    p = get_pricing("my-deployment")
    assert p.input_per_M == 1.5
    assert p.output_per_M == 6.0
    # Original table still wins for known names
    assert get_pricing("gpt-4.1").input_per_M == 2.0


# ---------------------------------------------------------------------------
# BudgetExceededError MUST inherit from BaseException
# ---------------------------------------------------------------------------

def test_budget_exceeded_is_baseexception_not_exception():
    """The whole point of the BaseException design is to pass through
    ``except Exception``. If someone "fixes" it back to Exception,
    every benchmark integration's broad except will absorb the signal
    and the hard stop becomes a soft stop."""
    assert issubclass(BudgetExceededError, BaseException)
    assert not issubclass(BudgetExceededError, Exception)

    err = BudgetExceededError("test")
    caught = False
    try:
        try:
            raise err
        except Exception:  # noqa: BLE001
            caught = True  # this branch must NOT execute
    except BaseException as e:
        # OK — it bypassed Exception and reached us
        assert e is err
    assert caught is False


# ---------------------------------------------------------------------------
# Single-process ledger correctness
# ---------------------------------------------------------------------------

def test_record_and_read_back(tmp_path):
    t = CostTracker(ledger_dir=tmp_path, daily_cap_usd=100.0, reservation_usd=0.0)
    cost1 = t.record("gpt-4.1", 1000, 200)
    cost2 = t.record("gpt-5.2", 2000, 100)
    total = t.today_total_usd()
    assert total == pytest.approx(cost1 + cost2)


def test_record_writes_jsonl_format(tmp_path):
    t = CostTracker(ledger_dir=tmp_path, daily_cap_usd=100.0)
    t.record("gpt-4.1", 1000, 200, cached_tokens=50, extra={"task": "test"})
    path = next(Path(tmp_path).glob("*.jsonl"))
    lines = path.read_text().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["model"] == "gpt-4.1"
    assert rec["prompt_tokens"] == 1000
    assert rec["completion_tokens"] == 200
    assert rec["cached_tokens"] == 50
    assert rec["cost_usd"] > 0
    assert rec["extra"] == {"task": "test"}
    assert rec["pid"] == os.getpid()


def test_torn_lines_are_ignored(tmp_path):
    """A truncated tail line must not sink the whole sum."""
    t = CostTracker(ledger_dir=tmp_path, daily_cap_usd=100.0)
    t.record("gpt-4.1", 1000, 200)
    path = next(Path(tmp_path).glob("*.jsonl"))
    with open(path, "a") as f:
        f.write('{"this is not valid json\n')
        f.write('\n')  # blank line
        f.write('{"cost_usd": "not_a_number"}\n')
    # First (valid) record's cost still sums correctly; bad lines skipped.
    expected = compute_cost_usd("gpt-4.1", 1000, 200)
    assert t.today_total_usd() == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Cap enforcement
# ---------------------------------------------------------------------------

def test_assert_under_cap_passes_when_empty(tmp_path):
    t = CostTracker(ledger_dir=tmp_path, daily_cap_usd=10.0, reservation_usd=1.0)
    t.assert_under_cap()  # no spend → no raise


def test_assert_under_cap_fires_at_threshold(tmp_path):
    t = CostTracker(ledger_dir=tmp_path, daily_cap_usd=1.0, reservation_usd=0.01)
    # Push spend right to threshold
    # cap=$1, reservation=$0.01, threshold=$0.99. Need >= $0.99.
    # gpt-4.1: 0.5M input + 0 output = $1.00
    t.record("gpt-4.1", 500_000, 0)  # exactly $1.00 spent
    with pytest.raises(BudgetExceededError):
        t.assert_under_cap()


def test_assert_under_cap_below_threshold_does_not_fire(tmp_path):
    t = CostTracker(ledger_dir=tmp_path, daily_cap_usd=1.0, reservation_usd=0.10)
    # Spend $0.50 — well below $0.90 threshold
    t.record("gpt-4.1", 250_000, 0)
    t.assert_under_cap()  # no raise


def test_soft_warn_fires_once_per_process(tmp_path, caplog):
    import logging
    t = CostTracker(ledger_dir=tmp_path, daily_cap_usd=1.0, reservation_usd=0.0)
    # Push to 85% (above 80% warn, below 100%)
    t.record("gpt-4.1", 425_000, 0)  # $0.85
    with caplog.at_level(logging.WARNING, logger="meta_n.utils.cost_tracker"):
        t.assert_under_cap()  # warn fires
        t.assert_under_cap()  # warn must NOT fire again
    warn_lines = [r for r in caplog.records if "Daily spend at" in r.message]
    assert len(warn_lines) == 1


# ---------------------------------------------------------------------------
# Concurrency: threads
# ---------------------------------------------------------------------------

def test_concurrent_threads_record_no_corruption(tmp_path):
    """100 threads each record 10 lines → 1000 lines total, sum exact."""
    t = CostTracker(ledger_dir=tmp_path, daily_cap_usd=1_000_000.0)
    n_threads = 100
    per_thread = 10

    def worker():
        for _ in range(per_thread):
            t.record("gpt-4.1", 100, 50)

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    # File should have exactly n_threads * per_thread valid JSON lines
    path = next(Path(tmp_path).glob("*.jsonl"))
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    assert len(lines) == n_threads * per_thread
    # Each line parseable (no torn writes)
    for ln in lines:
        json.loads(ln)
    # Sum exact
    one_call_cost = compute_cost_usd("gpt-4.1", 100, 50)
    expected = n_threads * per_thread * one_call_cost
    assert t.today_total_usd() == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Concurrency: cross-process
# ---------------------------------------------------------------------------

def _proc_worker_record(ledger_dir, n_calls):
    """Module-level (picklable for spawn) record-spammer used by mp tests."""
    from meta_n.utils.cost_tracker import CostTracker as CT
    t = CT(ledger_dir=ledger_dir, daily_cap_usd=1_000_000.0)
    for _ in range(n_calls):
        t.record("gpt-4.1", 100, 50)


def test_concurrent_processes_record_no_corruption(tmp_path):
    """4 subprocesses each record 50 lines → 200 lines total, sum exact."""
    n_procs = 4
    per_proc = 50
    ctx = mp.get_context("spawn")
    procs = [
        ctx.Process(target=_proc_worker_record, args=(str(tmp_path), per_proc))
        for _ in range(n_procs)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=30)
        assert p.exitcode == 0

    path = next(Path(tmp_path).glob("*.jsonl"))
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    assert len(lines) == n_procs * per_proc
    for ln in lines:
        json.loads(ln)  # every line parseable

    t = CostTracker(ledger_dir=tmp_path, daily_cap_usd=1_000_000.0)
    one_call_cost = compute_cost_usd("gpt-4.1", 100, 50)
    expected = n_procs * per_proc * one_call_cost
    assert t.today_total_usd() == pytest.approx(expected)


def _proc_worker_check_cap(ledger_dir, cap, result_q):
    """Subprocess that calls ``assert_under_cap`` and reports outcome."""
    from meta_n.utils.cost_tracker import CostTracker as CT, BudgetExceededError as BEE
    t = CT(ledger_dir=ledger_dir, daily_cap_usd=cap, reservation_usd=0.0)
    try:
        t.assert_under_cap()
        result_q.put(("ok", None))
    except BEE as e:
        result_q.put(("budget_exceeded", str(e)[:120]))
    except BaseException as e:  # noqa: BLE001
        result_q.put(("other", repr(e)))


def test_subprocess_sees_parents_writes(tmp_path):
    """Parent records spend up to the cap; subprocess opens a fresh
    CostTracker on the same ledger_dir and observes the parent's writes."""
    parent_t = CostTracker(ledger_dir=tmp_path, daily_cap_usd=1.0)
    parent_t.record("gpt-4.1", 500_000, 0)  # $1.00 — at cap

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_proc_worker_check_cap, args=(str(tmp_path), 1.0, q))
    p.start()
    p.join(timeout=10)
    outcome, info = q.get(timeout=5)
    assert outcome == "budget_exceeded", f"got ({outcome}, {info})"


# ---------------------------------------------------------------------------
# Subprocess BaseException propagation
# ---------------------------------------------------------------------------

def _proc_worker_raises_budget_under_except_exception(ledger_dir, result_q):
    """Subprocess that calls assert_under_cap inside a broad
    ``except Exception`` block, mimicking the benchmark integrations.
    The budget signal MUST escape the broad-catch and crash the
    subprocess, mirroring how it would propagate through co_bench /
    openevolve / text_classification subprocess wrappers."""
    from meta_n.utils.cost_tracker import CostTracker as CT
    t = CT(ledger_dir=ledger_dir, daily_cap_usd=0.0001, reservation_usd=0.0)
    # Pre-spend to push over.
    t.record("gpt-4.1", 100, 50)
    try:
        try:
            t.assert_under_cap()
            result_q.put(("no_raise", None))
            return
        except Exception as e:  # noqa: BLE001
            # If BudgetExceededError were a regular Exception, we'd land here.
            result_q.put(("absorbed_by_except_exception", repr(e)))
            return
    except BaseException as e:  # noqa: BLE001
        # This is where a BaseException-derived budget signal lands.
        result_q.put(("escaped_to_baseexception", type(e).__name__))
        return


def test_budget_exception_escapes_except_exception_in_subprocess(tmp_path):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(
        target=_proc_worker_raises_budget_under_except_exception,
        args=(str(tmp_path), q),
    )
    p.start()
    p.join(timeout=10)
    outcome, info = q.get(timeout=5)
    assert outcome == "escaped_to_baseexception", (
        f"BudgetExceededError got absorbed by `except Exception` — "
        f"this defeats the hard stop. outcome=({outcome}, {info})"
    )
    assert info == "BudgetExceededError"


# ---------------------------------------------------------------------------
# Day rollover
# ---------------------------------------------------------------------------

def test_today_only_reads_today(tmp_path):
    """Yesterday's spend must not count toward today's cap."""
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    today = date.today().isoformat()
    # Pre-create yesterday's ledger with a $50 line.
    yfile = tmp_path / f"{yesterday}.jsonl"
    yfile.write_text(json.dumps({
        "ts": time.time() - 86400,
        "model": "gpt-4.1",
        "prompt_tokens": 25_000_000,
        "completion_tokens": 0,
        "cached_tokens": 0,
        "cost_usd": 50.0,
        "pid": 1,
    }) + "\n")

    t = CostTracker(ledger_dir=tmp_path, daily_cap_usd=10.0, reservation_usd=0.10)
    # Today is empty — no spend.
    assert t.today_total_usd() == 0.0
    # Cap check passes.
    t.assert_under_cap()


def test_day_rollover_path_changes_with_local_date(tmp_path):
    """When the local date changes, ``_today_path`` returns the new file."""
    t = CostTracker(ledger_dir=tmp_path, daily_cap_usd=10.0)

    class FrozenDateTime:
        @staticmethod
        def now(tz=None):
            from datetime import datetime as _dt
            return _dt(2026, 1, 1, 12, 0, 0)

    with patch("meta_n.utils.cost_tracker.datetime", FrozenDateTime):
        p1 = t._today_path()
    assert p1.name == "2026-01-01.jsonl"

    class FrozenDateTime2:
        @staticmethod
        def now(tz=None):
            from datetime import datetime as _dt
            return _dt(2026, 1, 2, 0, 30, 0)

    with patch("meta_n.utils.cost_tracker.datetime", FrozenDateTime2):
        p2 = t._today_path()
    assert p2.name == "2026-01-02.jsonl"
    assert p1 != p2


# ---------------------------------------------------------------------------
# Pre-call timing: assertion must fire BEFORE the API call
# ---------------------------------------------------------------------------

def test_llm_client_asserts_before_calling_api(tmp_path):
    """Build an LLMClient with a tracker pre-loaded over the cap; the
    very first ``complete`` must raise without ever touching the SDK
    client. This guarantees no inadvertent spend after the cap is hit."""
    from meta_n.core.llm_client import LLMClient, LLMConfig
    import asyncio

    # Pre-spend so any new client built against this dir is over cap.
    seed = CostTracker(ledger_dir=tmp_path, daily_cap_usd=1.0, reservation_usd=0.0)
    seed.record("gpt-4.1", 500_000, 0)  # $1.00 spent

    cfg = LLMConfig(
        backend="openrouter",  # No real API call should happen anyway
        api_key="sk-fake",
        model="gpt-4.1",
        daily_budget_usd=1.0,
        cost_ledger_dir=str(tmp_path),
        cost_reservation_usd=0.0,
        max_retries=0,
    )
    c = LLMClient(cfg)
    # Replace the SDK client with a sentinel that explodes if called
    # — proves we never reached the network.
    class _Boom:
        chat = type("c", (), {"completions": type("cc", (), {
            "create": staticmethod(lambda **kw: (_ for _ in ()).throw(
                AssertionError("API call escaped despite over-cap state"),
            ))
        })()})
    c._client = _Boom()  # type: ignore[assignment]
    with pytest.raises(BudgetExceededError):
        asyncio.run(c.complete([{"role": "user", "content": "hi"}]))


def test_llm_client_rechecks_cap_between_retry_attempts(tmp_path):
    """First attempt fails with a transient error, second attempt would
    normally proceed — but if the cap was breached during the backoff,
    the recheck must fire BudgetExceededError instead of letting the
    second attempt spend money."""
    from meta_n.core.llm_client import LLMClient, LLMConfig
    from openai import APIConnectionError
    import asyncio

    cfg = LLMConfig(
        backend="openrouter",
        api_key="sk-fake",
        model="gpt-4.1",
        daily_budget_usd=10.0,
        cost_ledger_dir=str(tmp_path),
        cost_reservation_usd=0.0,
        max_retries=1,        # one retry → loop runs twice
        retry_base_delay=0.0, # no backoff in test
        retry_max_delay=0.0,
    )
    c = LLMClient(cfg)

    call_count = {"n": 0}

    async def fake_create(**kw):
        call_count["n"] += 1
        if call_count["n"] == 1:
            # Mimic a transient connection error so the retry loop kicks in.
            raise APIConnectionError(request=None)  # type: ignore[arg-type]
        # If we reach here, the second attempt fired despite the cap
        # — that's the bug we're guarding against.
        raise AssertionError("second attempt fired without re-checking cap")

    class _Stub:
        class chat:
            class completions:
                create = staticmethod(fake_create)

    c._client = _Stub()  # type: ignore[assignment]

    # Between the failed first attempt and the retry, simulate another
    # process pushing the ledger over the cap.
    async def driver():
        # Schedule a "side process" to push the cap mid-retry: we hook
        # into the sleep that the retry path performs by patching it.
        original_sleep = asyncio.sleep

        async def sleep_then_burn(delay):
            # Other process spends $20 — ledger now over cap.
            CostTracker(
                ledger_dir=tmp_path, daily_cap_usd=10.0, reservation_usd=0.0,
            ).record("gpt-4.1", 10_000_000, 0)  # $20
            await original_sleep(0)

        with patch("meta_n.core.llm_client.asyncio.sleep", sleep_then_burn):
            await c.complete([{"role": "user", "content": "hi"}])

    with pytest.raises(BudgetExceededError):
        asyncio.run(driver())
    # First attempt fired (raised APIConnectionError), recheck caught it
    # before the second attempt could even reach fake_create.
    assert call_count["n"] == 1


def test_reservation_bounds_overshoot(tmp_path):
    """With reservation_usd set to the worst-case single-call cost,
    the ledger total after a final allowed call cannot exceed the cap
    by more than that reservation. Sanity check on the bookkeeping."""
    cap = 10.0
    reservation = 0.50  # claimed worst-case single call

    t = CostTracker(ledger_dir=tmp_path, daily_cap_usd=cap, reservation_usd=reservation)
    # Spend up to cap-reservation
    while t.today_total_usd() + 0.40 < cap - reservation:
        t.record("gpt-4.1", 200_000, 0)  # $0.40 each

    # Last allowed call must be under threshold (cap - reservation)
    t.assert_under_cap()
    t.record("gpt-4.1", 200_000, 0)  # one more $0.40 call lands

    final = t.today_total_usd()
    # Total is at most threshold + one_call_cost (= cap - reservation + 0.40 < cap)
    # because the reservation is large enough to absorb the call.
    assert final <= cap, f"overshoot: ${final:.4f} > ${cap}"
    # The next call would block.
    with pytest.raises(BudgetExceededError):
        t.assert_under_cap()


# ---------------------------------------------------------------------------
# Pricing override + model alias survival across process
# ---------------------------------------------------------------------------

def _proc_worker_with_override(ledger_dir, env_value, result_q):
    os.environ["META_N_PRICING_OVERRIDE_JSON"] = env_value
    from meta_n.utils.cost_tracker import (
        compute_cost_usd as _cost,
        CostTracker as CT,
    )
    t = CT(ledger_dir=ledger_dir, daily_cap_usd=10.0)
    t.record("custom-model", 1_000_000, 0)
    result_q.put(t.today_total_usd())


def test_pricing_override_propagates_to_subprocess(tmp_path):
    """Subprocess inherits parent env, so an override declared in the
    parent shell flows to mp.Process workers automatically."""
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    env_value = '{"custom-model": {"input": 5.0, "output": 0}}'
    p = ctx.Process(
        target=_proc_worker_with_override,
        args=(str(tmp_path), env_value, q),
    )
    p.start()
    p.join(timeout=10)
    total = q.get(timeout=5)
    # 1M input × $5 = $5
    assert total == pytest.approx(5.0)
