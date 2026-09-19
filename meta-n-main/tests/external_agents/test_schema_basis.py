"""fair_comparison — basis-split tokens, cost breakout, unmeasured OH (§12.1).

Unlike the rest of the external_agents suite, these tests exercise
``meta_n.analysis.telemetry.fair_comparison`` which hard-depends on ``pandas``
(the ``analysis`` extra). They are skipped wholesale when pandas is absent so the
Phase-0 install-free gate stays green on a minimal runner.
"""

from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")  # analysis-extra dependency; skip if absent

from meta_n.analysis.telemetry import fair_comparison


def _frame(rows):
    return pd.DataFrame(rows)


def _row(agent, token_basis, cost_basis, **kw):
    base = dict(
        agent=agent,
        token_basis=token_basis,
        cost_basis=cost_basis,
        total_tokens=0,
        inner_tokens=0,
        inner_calls=0,
        cost_usd=0.0,
        steps=0,
        score=0.0,
        success=False,
    )
    base.update(kw)
    return base


def test_token_basis_kept_in_separate_columns():
    df = _frame(
        [
            _row("builtin", "outer", "priced_from_tokens", total_tokens=1000),
            _row("terminus2", "inner", "priced_from_tokens", total_tokens=2000,
                 inner_tokens=2000),
        ]
    )
    out = fair_comparison(df)
    # builtin's outer tokens land in outer_total_tokens, NOT inner.
    assert out.loc["builtin", "outer_total_tokens"] == 1000
    assert out.loc["builtin", "inner_total_tokens"] == 0
    # terminus2's inner tokens land in inner_total_tokens, NOT outer.
    assert out.loc["terminus2", "inner_total_tokens"] == 2000
    assert out.loc["terminus2", "outer_total_tokens"] == 0


def test_no_mixed_basis_aggregation_across_agents():
    # The split columns mean an outer row is never summed under an "inner" header.
    df = _frame(
        [
            _row("builtin", "outer", "priced_from_tokens", total_tokens=500),
            _row("openhands", "inner", "native_usd", total_tokens=300, inner_tokens=300),
        ]
    )
    out = fair_comparison(df)
    assert out.loc["builtin", "inner_total_tokens"] == 0
    assert out.loc["openhands", "outer_total_tokens"] == 0


def test_cost_basis_broken_out():
    df = _frame(
        [
            _row("openhands", "inner", "native_usd", cost_usd=0.50),
            _row("terminus2", "inner", "priced_from_tokens", cost_usd=0.20),
        ]
    )
    out = fair_comparison(df)
    # OH's native USD shows under cost_native_usd; T2's priced under the other.
    assert out.loc["openhands", "cost_native_usd"] == pytest.approx(0.50)
    assert out.loc["openhands", "cost_priced_from_tokens"] == pytest.approx(0.0)
    assert out.loc["terminus2", "cost_priced_from_tokens"] == pytest.approx(0.20)
    assert out.loc["terminus2", "cost_native_usd"] == pytest.approx(0.0)
    # Total cost is still sum-valid (both land in the same USD ledger).
    assert out.loc["openhands", "cost_usd"] == pytest.approx(0.50)


def test_openhands_agent_calls_annotated_unmeasured():
    df = _frame(
        [
            _row("openhands", "inner", "native_usd", inner_calls=7),
            _row("terminus2", "inner", "priced_from_tokens", inner_calls=4),
        ]
    )
    out = fair_comparison(df)
    # OH's per-call count is not measurable -> "unmeasured" sentinel (not an int).
    assert out.loc["openhands", "agent_calls"] == "unmeasured"
    # T2 reports a real summed count.
    assert out.loc["terminus2", "agent_calls"] == 4


def test_steps_column_present_and_summed():
    df = _frame(
        [
            _row("terminus2", "inner", "priced_from_tokens", steps=3),
            _row("terminus2", "inner", "priced_from_tokens", steps=5),
        ]
    )
    out = fair_comparison(df)
    assert out.loc["terminus2", "steps_turns_approx"] == 8


def test_success_and_score_aggregates():
    df = _frame(
        [
            _row("terminus2", "inner", "priced_from_tokens", success=True, score=1.0),
            _row("terminus2", "inner", "priced_from_tokens", success=False, score=0.0),
        ]
    )
    out = fair_comparison(df)
    assert out.loc["terminus2", "n_runs"] == 2
    assert out.loc["terminus2", "n_success"] == 1
    assert out.loc["terminus2", "success_rate"] == pytest.approx(0.5)
    assert out.loc["terminus2", "mean_score"] == pytest.approx(0.5)


def test_empty_frame_returns_empty():
    assert fair_comparison(pd.DataFrame()).empty
    # A frame without an `agent` column is also empty.
    assert fair_comparison(pd.DataFrame([{"x": 1}])).empty


# --- fairness upgrade: consumed inner_calls + cap_bound + wall_s + notes ------


def test_inner_calls_measured_for_every_agent():
    """Both OH and T2 now emit a real inner_calls count, so the measured
    ``inner_calls`` column is a true number for BOTH (unlike legacy agent_calls,
    which still shows OH's ``"unmeasured"`` sentinel)."""
    df = _frame(
        [
            _row("openhands", "inner", "native_usd", inner_calls=7),
            _row("terminus2", "inner", "priced_from_tokens", inner_calls=4),
        ]
    )
    out = fair_comparison(df)
    # New measured column: real ints for both agents.
    assert out.loc["openhands", "inner_calls"] == 7
    assert out.loc["terminus2", "inner_calls"] == 4
    # Legacy column unchanged (back-compat): OH still annotated unmeasured.
    assert out.loc["openhands", "agent_calls"] == "unmeasured"
    assert out.loc["terminus2", "agent_calls"] == 4


def test_inner_calls_masked_for_outer_basis_rows():
    """The ``inner_calls`` fairness surface is basis-masked: an outer-basis
    (builtin) row contributes 0 inner calls even if the row carries a non-zero
    ``inner_calls`` value (legacy telemetry that mislabeled outer authoring calls
    onto the inner axis). Mirrors ``inner_tokens`` masking and keeps the surface
    robust against historical rows."""
    df = _frame(
        [
            # Legacy-shaped builtin row: outer basis but a stray inner_calls=2 from
            # the old finish_record that stamped authoring calls onto the inner axis.
            _row("builtin", "outer", "priced_from_tokens", total_tokens=1069,
                 inner_tokens=1069, inner_calls=2),
            _row("terminus2", "inner", "priced_from_tokens", inner_calls=4),
        ]
    )
    out = fair_comparison(df)
    # builtin's outer authoring calls do NOT leak onto the inner-call axis.
    assert out.loc["builtin", "inner_calls"] == 0
    # ...and neither do its outer authoring tokens (the existing masking).
    assert out.loc["builtin", "inner_tokens"] == 0
    assert out.loc["builtin", "inner_total_tokens"] == 0
    # The inner-basis agent's real inner calls are untouched.
    assert out.loc["terminus2", "inner_calls"] == 4


def test_cap_bound_derived_from_terminated_by():
    """``terminated_by`` in {token_budget, max_turns, timeout, budget_*} means a
    cap bound the run; completed / agent_error do not."""
    df = _frame(
        [
            _row("openhands", "inner", "native_usd", terminated_by="token_budget"),
            _row("openhands", "inner", "native_usd", terminated_by="completed"),
            _row("terminus2", "inner", "priced_from_tokens",
                 terminated_by="max_turns"),
            _row("terminus2", "inner", "priced_from_tokens",
                 terminated_by="agent_error"),
        ]
    )
    out = fair_comparison(df)
    # OH: 1 of 2 runs cap-bound (token_budget); the completed one is not.
    assert out.loc["openhands", "n_cap_bound"] == 1
    assert out.loc["openhands", "cap_bound_rate"] == pytest.approx(0.5)
    # T2: 1 of 2 cap-bound (max_turns); agent_error is a failure, not a cap.
    assert out.loc["terminus2", "n_cap_bound"] == 1
    assert out.loc["terminus2", "cap_bound_rate"] == pytest.approx(0.5)


def test_terminated_by_breakdown_rendered():
    df = _frame(
        [
            _row("openhands", "inner", "native_usd", terminated_by="completed"),
            _row("openhands", "inner", "native_usd", terminated_by="completed"),
            _row("openhands", "inner", "native_usd", terminated_by="token_budget"),
        ]
    )
    out = fair_comparison(df)
    # Ordered by descending count then name: completed:2 before token_budget:1.
    assert out.loc["openhands", "terminated_by"] == "completed:2,token_budget:1"


def test_wall_s_consumed_summed_and_mean():
    df = _frame(
        [
            _row("terminus2", "inner", "priced_from_tokens", wall_s=10.0),
            _row("terminus2", "inner", "priced_from_tokens", wall_s=20.0),
        ]
    )
    out = fair_comparison(df)
    assert out.loc["terminus2", "wall_s"] == pytest.approx(30.0)
    assert out.loc["terminus2", "wall_s_mean"] == pytest.approx(15.0)


def test_builtin_outer_basis_note_present():
    """The builtin (outer-basis) row carries an explicit not-comparable note; the
    inner-basis agents do not."""
    df = _frame(
        [
            _row("builtin", "outer", "priced_from_tokens", total_tokens=1000),
            _row("terminus2", "inner", "priced_from_tokens", total_tokens=2000,
                 inner_tokens=2000),
        ]
    )
    out = fair_comparison(df)
    assert "NOT comparable" in out.loc["builtin", "notes"]
    assert out.loc["terminus2", "notes"] == ""


def test_cap_bound_defaults_when_no_terminated_by_column():
    """A legacy frame with no ``terminated_by`` column degrades to 0 cap-bound
    (all 'unknown') rather than raising."""
    df = _frame([_row("terminus2", "inner", "priced_from_tokens")])
    out = fair_comparison(df)
    assert out.loc["terminus2", "n_cap_bound"] == 0
    assert out.loc["terminus2", "terminated_by"] == "unknown:1"


def test_agent_name_not_unique_across_benchmarks_blends_inner_axes():
    """Pins the LEGACY-row behavior of the benchmark discriminator (F157): this
    frame carries NO ``benchmark`` column (rows predating schema v2), so the
    table keeps the plain ``agent`` index and a cross-benchmark concat collapses
    the CO-Bench ``openhands`` (native_usd) and the TB ``openhands``
    (priced_from_tokens) into ONE row that BLENDS the inner token/call/score
    axes — while the cost-basis breakout stays separate. New-format rows carry
    ``benchmark`` and split to an ``(agent, benchmark)`` MultiIndex instead
    (see tests/test_r6b_telemetry_schema.py)."""
    df = _frame(
        [
            # CO-Bench OpenHandsBackend row: native-USD cost basis.
            _row("openhands", "inner", "native_usd", total_tokens=1000,
                 inner_tokens=1000, inner_calls=5, cost_usd=0.10, score=1.0),
            # TB OpenHandsTBBackend row: priced-from-tokens cost basis. SAME name.
            _row("openhands", "inner", "priced_from_tokens", total_tokens=2000,
                 inner_tokens=2000, inner_calls=8, cost_usd=0.05, score=0.0),
        ]
    )
    out = fair_comparison(df)
    # Both rows collapse into ONE 'openhands' agent row.
    assert list(out.index) == ["openhands"]
    assert out.loc["openhands", "n_runs"] == 2
    # The inner token / call axes BLEND across the two distinct backends (the
    # known cross-benchmark hazard the caveat documents).
    assert out.loc["openhands", "inner_total_tokens"] == 3000
    assert out.loc["openhands", "inner_calls"] == 13
    assert out.loc["openhands", "mean_score"] == pytest.approx(0.5)
    # ...but the cost-basis breakout stays SEPARATE (the one axis that survives).
    assert out.loc["openhands", "cost_native_usd"] == pytest.approx(0.10)
    assert out.loc["openhands", "cost_priced_from_tokens"] == pytest.approx(0.05)
