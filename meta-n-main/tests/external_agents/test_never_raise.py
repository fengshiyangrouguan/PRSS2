"""Never-raise invariant — fault injection + gather-cancel survival (§12.1).

The spine's ``execute()`` runs inside the orchestrator's ``asyncio.gather`` with
**no** ``return_exceptions=True`` (evolutionary_orchestrator.py:811), so a raised
exception would cancel sibling tasks. Isolation depends on ``execute()`` *never*
propagating for ``Exception`` / ``asyncio.TimeoutError`` / ``BudgetExceededError``
— it must return a degraded ``(Trace, 0)`` — while ``asyncio.CancelledError`` must
still propagate (cooperative cancellation).
"""

from __future__ import annotations

import asyncio

import pytest

from meta_n.core.external_agents.solver import ExternalAgentSolver
from meta_n.core.meta_layer import Trace

from .conftest import (
    BudgetExceededError,
    FakeBackend,
    FakeEnvProvider,
    FakeScorer,
    make_task,
    read_run_records,
)


def _solver(backend, run_guard, telemetry, *, env_provider=None, cost_guard=None,
            time_limit_s=None, output_dir=None):
    return ExternalAgentSolver(
        backend=backend,
        env_provider=env_provider or FakeEnvProvider(),
        scorer=FakeScorer(),
        injected_codes=[],
        depth=1,
        run_guard=run_guard,
        telemetry=telemetry,
        cost_guard=cost_guard,
        adapter=object(),
        time_limit_s=time_limit_s,
    )


# --- the three non-propagating faults --------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exc",
    [
        RuntimeError("backend exploded"),
        asyncio.TimeoutError(),
        BudgetExceededError("cap reached"),
    ],
    ids=["Exception", "TimeoutError", "BudgetExceededError"],
)
async def test_execute_never_propagates_for_fault(exc, run_guard, telemetry, tmp_path):
    backend = FakeBackend(raise_in="run", exc=exc)
    solver = _solver(backend, run_guard, telemetry)
    # Must NOT raise — returns a degraded (Trace, 0).
    trace, tokens = await solver.execute(make_task())
    assert isinstance(trace, Trace)
    assert trace.success is False
    assert trace.score == 0.0
    assert tokens == 0
    # A telemetry row was still written for the degraded run.
    records = read_run_records(telemetry.output_dir)
    assert len(records) == 1


@pytest.mark.asyncio
async def test_provision_fault_is_degraded_not_propagated(run_guard, telemetry):
    backend = FakeBackend()  # would succeed if it ran
    solver = _solver(
        backend, run_guard, telemetry, env_provider=FakeEnvProvider(provision_error=True)
    )
    trace, tokens = await solver.execute(make_task())
    assert trace.success is False
    assert tokens == 0
    # The backend never ran because provision failed before it.
    assert backend.run_called == 0


# --- CancelledError must propagate -----------------------------------------


@pytest.mark.asyncio
async def test_execute_reraises_cancelled_error(run_guard, telemetry):
    backend = FakeBackend(raise_in="run", exc=asyncio.CancelledError())
    solver = _solver(backend, run_guard, telemetry)
    with pytest.raises(asyncio.CancelledError):
        await solver.execute(make_task())


# --- success path still works ----------------------------------------------


@pytest.mark.asyncio
async def test_success_path_returns_scored_trace(run_guard, telemetry):
    backend = FakeBackend()
    solver = _solver(backend, run_guard, telemetry)
    trace, tokens = await solver.execute(make_task())
    assert trace.success is True
    assert trace.score == 1.0
    # FakeBackend is not outer_token_mode -> outer int is 0 (inner spend rides
    # in Trace.inner_*).
    assert tokens == 0


# --- the gather-cancel invariant -------------------------------------------


@pytest.mark.asyncio
async def test_sibling_survives_when_one_run_faults(telemetry, tmp_path):
    """Two solvers under asyncio.gather (NO return_exceptions): one backend
    raises, and because execute() swallows it, the sibling still completes."""
    from meta_n.core.external_agents.concurrency import DockerRunGuard

    scratch = tmp_path / "scratch"
    scratch.mkdir()
    guard = DockerRunGuard(max_docker=2, scratch_root=str(scratch))

    failing = _solver(
        FakeBackend(raise_in="run", exc=RuntimeError("boom")),
        guard,
        telemetry,
    )
    ok = _solver(FakeBackend(), guard, telemetry)

    # gather WITHOUT return_exceptions — proves no exception escapes execute().
    results = await asyncio.gather(
        failing.execute(make_task(task_id="fails")),
        ok.execute(make_task(task_id="works")),
    )
    (fail_trace, _), (ok_trace, _) = results
    assert fail_trace.success is False
    assert ok_trace.success is True


@pytest.mark.asyncio
async def test_real_cancel_during_gather_propagates(telemetry, tmp_path):
    """A genuine CancelledError raised by one run cancels the gather (it is the
    one exception that must propagate), proving the re-raise is wired."""
    from meta_n.core.external_agents.concurrency import DockerRunGuard

    scratch = tmp_path / "scratch"
    scratch.mkdir()
    guard = DockerRunGuard(max_docker=2, scratch_root=str(scratch))
    cancelling = _solver(
        FakeBackend(raise_in="run", exc=asyncio.CancelledError()), guard, telemetry
    )
    with pytest.raises(asyncio.CancelledError):
        await asyncio.gather(cancelling.execute(make_task()))
