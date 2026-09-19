"""CostGuard — precheck arithmetic, record routing, fail-fast pricing (§12.1)."""

from __future__ import annotations

import json

import pytest

from meta_n.core.external_agents.backend import AgentRunResult
from meta_n.core.external_agents.budget import CostGuard
from meta_n.utils.cost_tracker import CostTracker, compute_cost_usd

from .conftest import make_task


def _ledger_lines(tracker: CostTracker) -> list[dict]:
    path = tracker._today_path()  # noqa: SLF001 - test introspection
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


# --- fail-fast model coverage ----------------------------------------------


def test_unpriceable_model_raises_at_construction(cost_tracker):
    with pytest.raises(RuntimeError) as ei:
        CostGuard(cost_tracker, model="totally-unknown-model-xyz")
    msg = str(ei.value)
    assert "totally-unknown-model-xyz" in msg
    # Names the override escape hatch (clear error, not silent zero-pricing).
    assert "META_N_PRICING_OVERRIDE_JSON" in msg


def test_none_tracker_raises_at_construction():
    with pytest.raises(RuntimeError):
        CostGuard(None, model="gpt-5.2")


def test_priceable_model_constructs(cost_tracker):
    guard = CostGuard(cost_tracker, model="gpt-5.2")
    assert guard.model == "gpt-5.2"


# --- precheck arithmetic ---------------------------------------------------


def test_precheck_admits_when_under_threshold(cost_guard):
    # cap=100, reservation=1 -> threshold=99. today=0, budget=2 -> projected 2 < 99.
    assert cost_guard.precheck(2.0) is False


def test_precheck_clamps_and_admits_when_budget_exceeds_headroom(tmp_path):
    # NEW semantics (BUDGET-PRECHECK footgun fix): a per-run budget larger than
    # the remaining headroom is CLAMPED-and-ADMITTED, not denied — the hard daily
    # cap still stops the run once headroom is actually consumed.
    tracker = CostTracker(ledger_dir=tmp_path / "c", daily_cap_usd=10.0, reservation_usd=1.0)
    guard = CostGuard(tracker, model="gpt-5.2")
    # threshold = 9.0; today = 0 -> headroom 9.0 > 0. budget 9.5 > headroom ->
    # clamp-and-admit (False), NOT deny.
    assert guard.precheck(9.5) is False
    # budget 8.0 <= headroom 9.0 -> admit without clamping.
    assert guard.precheck(8.0) is False


def test_precheck_warns_per_run_budget_is_not_a_hard_stop(tmp_path, caplog):
    # T2.3: --agent-max-budget is a DECLARED/precheck ceiling, NOT a hard per-run
    # USD stop. When the requested per-run budget exceeds the day's headroom the
    # run is admitted (False) but a WARNING surfaces that it is not clamped/
    # enforced — so the operator is never misled into thinking it caps a run.
    tracker = CostTracker(ledger_dir=tmp_path / "c", daily_cap_usd=10.0, reservation_usd=1.0)
    guard = CostGuard(tracker, model="gpt-5.2")
    with caplog.at_level("WARNING", logger="meta_n.core.external_agents.budget"):
        assert guard.precheck(9.5) is False  # 9.5 > headroom 9.0 -> admit + warn
    text = caplog.text
    assert "no backend enforces a per-run USD stop" in text
    assert "not clamped" in text.lower()


def test_precheck_silent_when_within_headroom(tmp_path, caplog):
    # T2.3 off-path: a per-run budget within headroom admits with NO warning
    # (the no-hard-stop notice fires only on the overshoot branch).
    tracker = CostTracker(ledger_dir=tmp_path / "c", daily_cap_usd=10.0, reservation_usd=1.0)
    guard = CostGuard(tracker, model="gpt-5.2")
    with caplog.at_level("WARNING", logger="meta_n.core.external_agents.budget"):
        assert guard.precheck(8.0) is False  # 8.0 <= headroom 9.0
    assert "per-run USD stop" not in caplog.text


def test_precheck_boundary_strictly_greater(tmp_path):
    tracker = CostTracker(ledger_dir=tmp_path / "c", daily_cap_usd=10.0, reservation_usd=1.0)
    guard = CostGuard(tracker, model="gpt-5.2")
    # Exactly at the threshold (9.0) with no prior spend -> headroom 9.0 > 0 -> admit.
    assert guard.precheck(9.0) is False


def test_precheck_denies_only_when_headroom_exhausted(tmp_path):
    # The hard daily cap stays meaningful: once today's spend has consumed the
    # day's headroom (today >= cap - reservation), EVERY run is denied.
    tracker = CostTracker(ledger_dir=tmp_path / "c", daily_cap_usd=10.0, reservation_usd=1.0)
    guard = CostGuard(tracker, model="gpt-5.2")
    # Spend right up to the threshold (9.0): headroom == 0 -> deny regardless of
    # the per-run budget (even a $0 budget is denied — the day is done).
    tracker.record_usd("gpt-5.2", 9.0, extra={"source": "test"})
    assert guard.precheck(0.0) is True
    assert guard.precheck(0.01) is True


def test_precheck_admits_with_headroom_despite_prior_spend(tmp_path):
    # With headroom still remaining, a per-run budget that would *project* over
    # the threshold is clamped-and-admitted (old behavior denied this).
    tracker = CostTracker(ledger_dir=tmp_path / "c", daily_cap_usd=10.0, reservation_usd=1.0)
    guard = CostGuard(tracker, model="gpt-5.2")
    tracker.record_usd("gpt-5.2", 8.5, extra={"source": "test"})
    # today=8.5, threshold=9.0 -> headroom 0.5 > 0. budget 1.0 > headroom ->
    # clamp-and-admit (False), NOT deny.
    assert guard.precheck(1.0) is False


def test_precheck_never_raises_even_on_broken_tracker(cost_tracker):
    guard = CostGuard(cost_tracker, model="gpt-5.2")

    class Broken:
        daily_cap_usd = 10.0
        reservation_usd = 1.0

        def today_total_usd(self):
            raise RuntimeError("ledger unreadable")

    guard.cost_tracker = Broken()
    # Must not raise; a broken pre-check admits the run (hard caps still apply).
    assert guard.precheck(2.0) is False


# --- record routing --------------------------------------------------------


def test_record_native_usd_appends_one_line(cost_guard, cost_tracker):
    run = AgentRunResult(
        cost_usd=0.42,
        cost_basis="native_usd",
        agent_prompt_tokens=100,
        agent_completion_tokens=50,
    )
    cost_guard.record(run, task=make_task(task_id="t1"), solver=None)
    lines = _ledger_lines(cost_tracker)
    assert len(lines) == 1
    # The native USD is appended verbatim and reflected in today_total_usd().
    assert lines[0]["cost_usd"] == pytest.approx(0.42)
    assert cost_tracker.today_total_usd() == pytest.approx(0.42)
    assert lines[0]["extra"]["basis"] == "native_usd"


def test_record_t2_prices_from_tokens(cost_tracker):
    guard = CostGuard(cost_tracker, model="gpt-5.2")
    run = AgentRunResult(
        cost_usd=0.0,  # T2 reports no native cost
        cost_basis="priced_from_tokens",
        agent_prompt_tokens=1_000_000,
        agent_completion_tokens=500_000,
        agent_cached_tokens=0,
    )
    guard.record(run, task=make_task(), solver=None)
    expected = compute_cost_usd("gpt-5.2", 1_000_000, 500_000, 0)
    assert cost_tracker.today_total_usd() == pytest.approx(expected)
    lines = _ledger_lines(cost_tracker)
    assert lines[0]["extra"]["basis"] == "priced_from_tokens"


def test_record_never_raises_on_internal_failure(cost_guard):
    # record() swallows internal errors (run already scored). A run object whose
    # attribute access blows up must not propagate.
    class Boom:
        cost_basis = "native_usd"

        def __getattr__(self, name):
            raise RuntimeError("attr access failed")

    # Should not raise.
    cost_guard.record(Boom(), task=make_task(), solver=None)


# --- BUDGET-PRECHECK footgun: small --daily-budget-usd is usable out of box --


def test_small_daily_budget_admits_default_agent_budget_end_to_end(tmp_path):
    """REGRESSION (footgun fix): ``--daily-budget-usd 1`` with a per-task
    ``--agent-max-budget`` ABOVE the day (here 2.0) must ADMIT a $0-priced local
    model run with no manual env overrides.

    Exercises the real wiring: an ``LLMConfig`` with ``daily_budget_usd=1``
    builds the ``CostTracker`` (now reservation 0.0 by default), and the
    ``CostGuard`` precheck for the local Gemma model admits a $2 per-task budget
    even though $2 > the $1 day — it is clamped-and-admitted, not denied. (The
    CLI default ``--agent-max-budget`` is now 0.5; this test deliberately uses a
    larger 2.0 to prove the clamp-and-admit path still holds for budgets that
    exceed the day.)
    """
    from meta_n.core.llm_client import LLMConfig, LLMClient

    cfg = LLMConfig(
        base_url="http://127.0.0.1:1234/v1",
        api_key="dummy",
        model="google/gemma-4-31b-qat",  # $0-priced local model (real PRICING entry)
        daily_budget_usd=1.0,
        cost_ledger_dir=str(tmp_path / "ledger"),
    )
    # Default reservation is now 0.0 -> the cap IS the cap (no silent shrink).
    assert cfg.cost_reservation_usd == 0.0

    client = LLMClient(cfg)
    assert client.cost_tracker is not None
    assert client.cost_tracker.daily_cap_usd == pytest.approx(1.0)
    assert client.cost_tracker.reservation_usd == pytest.approx(0.0)

    guard = CostGuard(client.cost_tracker, model=cfg.model)
    # A $2 per-task budget exceeds the $1 day, but with full headroom the run is
    # admitted (clamp-and-admit), NOT silently denied.
    assert guard.precheck(2.0) is False


def test_default_reservation_does_not_shrink_small_budget(tmp_path):
    # With the old default reservation of 1.0, a --daily-budget-usd of 1 left
    # threshold == 0, denying everything. The new 0.0 default keeps the whole $1.
    tracker = CostTracker(ledger_dir=tmp_path / "c", daily_cap_usd=1.0)
    assert tracker.reservation_usd == pytest.approx(0.0)
    guard = CostGuard(tracker, model="google/gemma-4-31b-qat")
    # Full $1 headroom: a $0-priced local run is admitted.
    assert guard.precheck(2.0) is False
    # And the hard cap is still real once the day's headroom is spent.
    tracker.record_usd("gpt-5.2", 1.0, extra={"source": "test"})
    assert guard.precheck(0.0) is True


def test_explicit_reservation_still_honored_and_caps(tmp_path):
    # Operators who DO want a margin still get one: an explicit reservation
    # shrinks the headroom and the hard cap fires earlier, as before.
    tracker = CostTracker(ledger_dir=tmp_path / "c", daily_cap_usd=10.0, reservation_usd=3.0)
    guard = CostGuard(tracker, model="gpt-5.2")
    # threshold = 7.0. Spend to the threshold -> deny.
    tracker.record_usd("gpt-5.2", 7.0, extra={"source": "test"})
    assert guard.precheck(0.0) is True


# --- headroom_exhausted: the generation-boundary daily-cap halt -------------


def test_headroom_exhausted_false_with_budget_left(tmp_path):
    tracker = CostTracker(ledger_dir=tmp_path / "c", daily_cap_usd=10.0, reservation_usd=1.0)
    guard = CostGuard(tracker, model="gpt-5.2")
    assert guard.headroom_exhausted() is False
    tracker.record_usd("gpt-5.2", 5.0, extra={"source": "test"})  # under threshold
    assert guard.headroom_exhausted() is False


def test_headroom_exhausted_true_once_threshold_reached(tmp_path):
    tracker = CostTracker(ledger_dir=tmp_path / "c", daily_cap_usd=10.0, reservation_usd=1.0)
    guard = CostGuard(tracker, model="gpt-5.2")
    tracker.record_usd("gpt-5.2", 9.0, extra={"source": "test"})  # == threshold 9.0
    assert guard.headroom_exhausted() is True
    # Re-reads fresh each call (not a cached pre-batch total): more spend stays True.
    tracker.record_usd("gpt-5.2", 1.0, extra={"source": "test"})
    assert guard.headroom_exhausted() is True


def test_headroom_exhausted_never_raises_on_broken_tracker(cost_tracker):
    guard = CostGuard(cost_tracker, model="gpt-5.2")

    class Broken:
        daily_cap_usd = 10.0
        reservation_usd = 1.0

        def today_total_usd(self):
            raise RuntimeError("ledger unreadable")

    guard.cost_tracker = Broken()
    # A broken read is treated as "not exhausted" (the per-task precheck re-checks).
    assert guard.headroom_exhausted() is False
