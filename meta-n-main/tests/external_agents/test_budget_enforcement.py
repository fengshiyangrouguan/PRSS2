"""Daily-cap enforcement on the external path (audit blocker #1).

The external backends enforce no per-run USD stop, so the daily cap is enforced
entirely meta-n-side by (a) the per-task admission precheck and (b) a fresh
headroom re-check inside the lease, immediately before ``backend.run``, that
aborts a task admitted before its siblings recorded spend. These tests drive the
real ``ExternalAgentSolver`` lifecycle with the in-memory Fake collaborators.
"""

from __future__ import annotations

import pytest

from meta_n.core.external_agents.solver import ExternalAgentSolver
from meta_n.core.external_agents.terminated import TerminatedBy
from meta_n.core.meta_layer import Trace

from .conftest import FakeBackend, FakeEnvProvider, FakeScorer, make_task, read_run_records


def _solver(backend, run_guard, telemetry, cost_guard, env_provider=None):
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
        time_limit_s=None,
    )


class _FakeGuard:
    """Stand-in CostGuard whose two gates are scriptable per-call."""

    def __init__(self, *, precheck_denies=False, exhausted_sequence=None):
        self._precheck_denies = precheck_denies
        # headroom_exhausted() returns the next value from this list each call;
        # the last value repeats once exhausted.
        self._seq = list(exhausted_sequence or [False])
        self.precheck_calls = 0
        self.headroom_calls = 0
        self.recorded = 0

    def precheck(self, max_budget_usd):
        self.precheck_calls += 1
        return self._precheck_denies

    def headroom_exhausted(self):
        self.headroom_calls += 1
        idx = min(self.headroom_calls - 1, len(self._seq) - 1)
        return self._seq[idx]

    def record(self, run, task=None, solver=None):
        self.recorded += 1


@pytest.mark.asyncio
async def test_precheck_denial_short_circuits_before_lease(run_guard, telemetry):
    backend = FakeBackend()
    guard = _FakeGuard(precheck_denies=True)
    solver = _solver(backend, run_guard, telemetry, guard)
    trace, tokens = await solver.execute(make_task())
    assert isinstance(trace, Trace)
    assert trace.success is False
    assert tokens == 0
    # The precheck denied before any work — backend never ran, no spend recorded.
    assert backend.run_called == 0
    assert guard.recorded == 0
    rows = read_run_records(telemetry.output_dir)
    assert rows[0]["terminated_by"] == TerminatedBy.BUDGET_DENIED.value


@pytest.mark.asyncio
async def test_intra_lease_recheck_aborts_when_day_fills_up(run_guard, telemetry):
    """Admission precheck passes (headroom False at admission), but the day fills
    up while the run waits for the inner Docker slot — the fresh re-check inside
    the lease aborts WITHOUT provisioning or running the backend."""
    backend = FakeBackend()
    env = FakeEnvProvider()
    # precheck path does NOT call headroom_exhausted; the spine calls
    # cost_guard.precheck() first (admit), then headroom_exhausted() in
    # _run_leased before backend.run. Script: first headroom call -> True (abort).
    guard = _FakeGuard(precheck_denies=False, exhausted_sequence=[True])
    solver = _solver(backend, run_guard, telemetry, guard, env_provider=env)
    trace, tokens = await solver.execute(make_task())
    assert trace.success is False
    assert tokens == 0
    # The backend never ran and the env was never provisioned.
    assert backend.run_called == 0
    assert env.last_env is None
    rows = read_run_records(telemetry.output_dir)
    assert rows[0]["terminated_by"] == TerminatedBy.BUDGET_DENIED.value


@pytest.mark.asyncio
async def test_recheck_passes_when_headroom_remains(run_guard, telemetry):
    """With headroom still present at the intra-lease re-check, the run proceeds
    normally and spend is recorded."""
    backend = FakeBackend()
    guard = _FakeGuard(precheck_denies=False, exhausted_sequence=[False])
    solver = _solver(backend, run_guard, telemetry, guard)
    trace, tokens = await solver.execute(make_task())
    assert trace.success is True
    assert backend.run_called == 1
    assert guard.recorded == 1  # spend folded into the ledger
