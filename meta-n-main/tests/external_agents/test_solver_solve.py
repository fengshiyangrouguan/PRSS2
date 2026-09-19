"""ExternalAgentSolver.solve() — the re-solve SolverProtocol seam (§12.1).

``solve()`` matches meta-n's ``SolverProtocol.solve`` shape so the chain-test /
``needs_re_solve`` path can drive an external agent the way it drives a native
solver. No benchmark CURRENTLY both supports an external ``base_solver`` AND
triggers a re-solve (``needs_re_solve`` is ``hasattr(adapter, "get_test_task")``,
a method only the classification adapters define — and those have no external
backend), so ``solve()`` and the ``parent_run_id`` child-lineage it threads are a
FORWARD-LOOKING seam. These tests lock that contract directly (the doc note on
``solve()`` documents the seam; this pins its behavior) so a future re-solving
external-backed adapter inherits a tested, never-raise re-solve path.

Install-free: ``meta_n`` + stdlib only, no Docker / SDK / LLM. Reuses the
conftest fakes (the same harness ``test_never_raise.py`` drives ``execute()``).
"""

from __future__ import annotations

import asyncio

import pytest

from meta_n.core.external_agents.backend import AgentRunResult
from meta_n.core.external_agents.solver import ExternalAgentSolver
from meta_n.core.external_agents.terminated import TerminatedBy

from .conftest import (
    FakeBackend,
    FakeEnvProvider,
    FakeScorer,
    make_task,
    read_run_records,
)


class _RecordingBackend(FakeBackend):
    """A success-path backend that records the ``Prompt.prefix`` it received, so a
    test can prove ``solve()`` threads ``additional_context`` onto the prompt."""

    def __init__(self):
        super().__init__()
        self.seen_prefix: str | None = None

    async def run(self, ctx, tel, rec) -> AgentRunResult:
        self.seen_prefix = ctx.prompt.prefix
        return await super().run(ctx, tel, rec)


def _solver(backend, run_guard, telemetry):
    return ExternalAgentSolver(
        backend=backend,
        env_provider=FakeEnvProvider(),
        scorer=FakeScorer(),
        injected_codes=[],
        depth=1,
        run_guard=run_guard,
        telemetry=telemetry,
        cost_guard=None,
        adapter=object(),
        time_limit_s=None,
    )


# --- the (script, reasoning, tokens) contract ------------------------------


@pytest.mark.asyncio
async def test_solve_returns_script_reasoning_tokens_triple(run_guard, telemetry):
    backend = _RecordingBackend()
    solver = _solver(backend, run_guard, telemetry)

    result = await solver.solve(make_task(), additional_context="LAYER-2 GUIDANCE")

    # The SolverProtocol shape: a (script, reasoning, tokens) triple.
    assert isinstance(result, tuple) and len(result) == 3
    script, reasoning, tokens = result
    assert isinstance(script, str) and isinstance(reasoning, str)
    assert isinstance(tokens, int)
    # The env provider's extracted solution becomes Trace.script; the backend's
    # reasoning_summary becomes Trace.reasoning (FakeEnvProvider / FakeBackend
    # defaults).
    assert script == "print('hi')"
    assert reasoning == "did it"
    # FakeBackend is not outer_token_mode, so the outer-token int is 0.
    assert tokens == 0


@pytest.mark.asyncio
async def test_solve_threads_additional_context_into_prompt_prefix(run_guard, telemetry):
    """The distinguishing behavior vs ``execute()``: the inter-layer
    ``additional_context`` is injected as the ``Prompt.prefix`` (the live half of
    the seam)."""
    backend = _RecordingBackend()
    solver = _solver(backend, run_guard, telemetry)

    await solver.solve(make_task(), additional_context="HIGHER-LAYER PREFIX")
    assert backend.seen_prefix == "HIGHER-LAYER PREFIX"


@pytest.mark.asyncio
async def test_solve_empty_context_yields_empty_prefix(run_guard, telemetry):
    backend = _RecordingBackend()
    solver = _solver(backend, run_guard, telemetry)
    # The default (no inter-layer context) is the gen0 vanilla prefix.
    await solver.solve(make_task())
    assert backend.seen_prefix == ""


# --- never-raise (the SolverProtocol re-solve must honor it too) -----------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exc",
    [RuntimeError("backend exploded"), asyncio.TimeoutError()],
    ids=["Exception", "TimeoutError"],
)
async def test_solve_never_propagates_for_fault(exc, run_guard, telemetry):
    backend = FakeBackend(raise_in="run", exc=exc)
    solver = _solver(backend, run_guard, telemetry)

    # Must NOT raise — degrades to an empty-script triple (mirrors execute()).
    script, reasoning, tokens = await solver.solve(make_task())
    assert script == ""
    assert tokens == 0
    # A telemetry row was still written for the degraded re-solve.
    assert len(read_run_records(telemetry.output_dir)) == 1


@pytest.mark.asyncio
async def test_solve_reraises_cancelled_error(run_guard, telemetry):
    backend = FakeBackend(raise_in="run", exc=asyncio.CancelledError())
    solver = _solver(backend, run_guard, telemetry)
    with pytest.raises(asyncio.CancelledError):
        await solver.solve(make_task())


# --- the parent_run_id lineage seam (currently unpopulated) ----------------


@pytest.mark.asyncio
async def test_solve_records_run_with_null_parent_run_id(run_guard, telemetry):
    """``solve()`` passes ``parent_run_id=None`` today (no re-solving external
    adapter exists), so the recorded row's ``parent_run_id`` is honestly null —
    pinning the documented state of the child-lineage seam so a future change that
    starts populating it is a deliberate, test-visible edit."""
    backend = _RecordingBackend()
    solver = _solver(backend, run_guard, telemetry)
    await solver.solve(make_task(), additional_context="ctx")

    rows = read_run_records(telemetry.output_dir)
    assert len(rows) == 1
    # The field exists on the schema but is null until a re-solve child populates it.
    assert rows[0].get("parent_run_id") is None


# --- H5: outer wait_for envelope strictly exceeds the backend hard timeout ----


def _solver_with_limit(backend, run_guard, telemetry, *, time_limit_s):
    return ExternalAgentSolver(
        backend=backend,
        env_provider=FakeEnvProvider(),
        scorer=FakeScorer(),
        injected_codes=[],
        depth=1,
        run_guard=run_guard,
        telemetry=telemetry,
        cost_guard=None,
        adapter=object(),
        time_limit_s=time_limit_s,
    )


@pytest.mark.asyncio
async def test_outer_envelope_strictly_exceeds_backend_hard_timeout(
    run_guard, telemetry, monkeypatch
):
    """H5: the outer ``asyncio.wait_for`` must wrap the run in an envelope that
    STRICTLY EXCEEDS the backend's own hard timeout (``_bridge.hard_timeout``)
    plus grace, so the backend's hard kill wins the race and a winding-down run
    is recorded with its real tokens — not cut off as a spurious 0-token TIMEOUT
    at the SOFT limit (the pre-H5 bug)."""
    from meta_n.core.external_agents import solver as solver_mod
    from meta_n.core.external_agents._bridge import hard_timeout

    captured: dict[str, object] = {}
    real_wait_for = asyncio.wait_for

    async def _spy(awaitable, timeout):
        captured["timeout"] = timeout
        return await real_wait_for(awaitable, timeout=timeout)

    monkeypatch.setattr(solver_mod.asyncio, "wait_for", _spy)

    solver = _solver_with_limit(
        FakeBackend(), run_guard, telemetry, time_limit_s=120.0
    )
    await solver.execute(make_task())

    backend_hard = hard_timeout(120.0)
    assert captured["timeout"] == backend_hard + solver_mod._OUTER_GRACE_S
    # The load-bearing invariant: the outer wall is STRICTLY ABOVE the backend's
    # own hard timeout, so the backend hard-kill (which returns recoverable
    # tokens) fires first.
    assert captured["timeout"] > backend_hard


@pytest.mark.asyncio
async def test_outer_envelope_none_when_no_time_limit(
    run_guard, telemetry, monkeypatch
):
    """H5: ``time_limit_s=None`` ⇒ envelope ``None`` (no outer wall) —
    byte-identical to the historical unbounded path."""
    from meta_n.core.external_agents import solver as solver_mod

    captured: dict[str, object] = {}
    real_wait_for = asyncio.wait_for

    async def _spy(awaitable, timeout):
        captured["timeout"] = timeout
        return await real_wait_for(awaitable, timeout=timeout)

    monkeypatch.setattr(solver_mod.asyncio, "wait_for", _spy)

    solver = _solver_with_limit(
        FakeBackend(), run_guard, telemetry, time_limit_s=None
    )
    await solver.execute(make_task())
    assert captured["timeout"] is None


@pytest.mark.asyncio
async def test_timeout_finisher_preserves_recoverable_tokens(run_guard, telemetry):
    """H5 fold: if the outer envelope DOES fire, agent tokens/cost already
    stamped on the record survive into the written row — the degraded finisher
    only zeroes score/success/attribution, never the token/cost fields."""
    solver = _solver_with_limit(
        FakeBackend(), run_guard, telemetry, time_limit_s=10.0
    )
    rec = telemetry.start_record(make_task(), solver, None)
    rec.total_tokens = 1234
    rec.prompt_tokens = 1000
    rec.completion_tokens = 234
    rec.inner_tokens = 234
    rec.cost_usd = 0.05

    telemetry.finish_timeout(make_task(), rec, 1)

    rows = read_run_records(telemetry.output_dir)
    assert len(rows) == 1
    row = rows[0]
    assert row["terminated_by"] == TerminatedBy.TIMEOUT.value
    # tokens/cost preserved (the fold) ...
    assert row["total_tokens"] == 1234
    assert row["cost_usd"] == 0.05
    # ... while score/success are zeroed (degraded).
    assert row["success"] is False
    assert row["score"] == 0.0
