"""DockerRunGuard — lease lifecycle, sweep, sanitize, validation (§12.1)."""

from __future__ import annotations

import asyncio
import re

import pytest

from meta_n.core.external_agents._bridge import sanitize_compose_name
from meta_n.core.external_agents.concurrency import DockerRunGuard
from meta_n.core.external_agents.env import EnvLease

from .conftest import make_task


# --- construction ----------------------------------------------------------


def test_max_docker_below_one_raises():
    with pytest.raises(ValueError):
        DockerRunGuard(max_docker=0)
    with pytest.raises(ValueError):
        DockerRunGuard(max_docker=-3)


def test_max_docker_one_ok():
    guard = DockerRunGuard(max_docker=1)
    assert guard.max_docker == 1


# --- session sanitization ----------------------------------------------------
# The lease call site now calls the canonical ``sanitize_compose_name`` directly
# (the ``_sanitize`` wrapper was a pure delegate, so the former Item #7
# delegation-equality test is true by construction and was removed). These
# assertions pin the session-name shape the lease mints.


def test_sanitize_produces_valid_compose_project_name():
    out = sanitize_compose_name("ext-Task/With Spaces.AND_CAPS-1234")
    # Lower-case, starts alphanumeric, only [a-z0-9_-].
    assert out == out.lower()
    assert re.match(r"^[a-z0-9]", out)
    assert re.fullmatch(r"[a-z0-9_-]+", out)


def test_sanitize_prepends_zero_when_not_alnum_start():
    # After the canonical sanitizer strips a leading ``-``/``_`` the remainder
    # may already be alnum-leading (then no ``0`` prefix). A string that is
    # ALL-special reduces to the ``"0"`` fallback, which is alnum-leading.
    assert re.match(r"^[a-z0-9]", sanitize_compose_name("-leading-dash"))
    assert sanitize_compose_name("---") == "0"  # all-special → fallback
    assert re.match(r"^[a-z0-9]", sanitize_compose_name("___"))


def test_sanitize_collapses_strips_and_is_idempotent():
    assert sanitize_compose_name("ext-Foo  Bar--x_") == "ext-foo-bar-x"
    s = "ext-Task/With Spaces.AND_CAPS-1234"
    assert sanitize_compose_name(sanitize_compose_name(s)) == sanitize_compose_name(s)


def test_sanitize_common_case_unchanged():
    """The common ``ext-{task}-{uuid8}`` lease-session shape round-trips."""
    assert (
        sanitize_compose_name("ext-aircraft_landing-ab12cd34")
        == "ext-aircraft_landing-ab12cd34"
    )


# --- lease lifecycle -------------------------------------------------------


@pytest.mark.asyncio
async def test_lease_acquires_and_releases_semaphore_and_rmtrees(tmp_path):
    guard = DockerRunGuard(max_docker=2, scratch_root=str(tmp_path))
    task = make_task()
    before = guard._sem._value  # noqa: SLF001 - test introspection
    async with guard.lease(task) as lease:
        # Slot consumed while held.
        assert guard._sem._value == before - 1  # noqa: SLF001
        assert lease.workdir.exists()
        assert lease.session in guard._inflight
        workdir = lease.workdir
    # Released on exit: slot back, registry empty, scratch dir removed.
    assert guard._sem._value == before  # noqa: SLF001
    assert lease.session not in guard._inflight
    assert not workdir.exists()


@pytest.mark.asyncio
async def test_lease_calls_hard_kill_in_finally(tmp_path):
    guard = DockerRunGuard(max_docker=1, scratch_root=str(tmp_path))
    calls = []
    async with guard.lease(make_task()) as lease:
        lease.hard_kill = lambda: calls.append("killed")
    assert calls == ["killed"]


@pytest.mark.asyncio
async def test_lease_hard_kill_runs_even_on_error(tmp_path):
    guard = DockerRunGuard(max_docker=1, scratch_root=str(tmp_path))
    calls = []
    with pytest.raises(RuntimeError):
        async with guard.lease(make_task()) as lease:
            lease.hard_kill = lambda: calls.append("killed")
            raise RuntimeError("body failed")
    assert calls == ["killed"]
    # Registry still cleaned even though the body raised.
    assert lease.session not in guard._inflight


@pytest.mark.asyncio
async def test_lease_hard_kill_exception_is_swallowed(tmp_path):
    guard = DockerRunGuard(max_docker=1, scratch_root=str(tmp_path))

    def boom():
        raise RuntimeError("kill failed")

    # A raising hard_kill must not leak out of the lease release.
    async with guard.lease(make_task()) as lease:
        lease.hard_kill = boom
    assert lease.session not in guard._inflight


# --- shutdown_sweep --------------------------------------------------------


@pytest.mark.asyncio
async def test_shutdown_sweep_runs_hard_kill_then_teardown(tmp_path):
    guard = DockerRunGuard(max_docker=1, scratch_root=str(tmp_path))
    order = []
    lease = EnvLease(workdir=tmp_path / "wd", session="ext-x-1", task_id="x")
    lease.hard_kill = lambda: order.append("kill")

    async def teardown():
        order.append("teardown")

    lease.teardown = teardown
    # Register an in-flight lease, then sweep.
    guard._inflight[lease.session] = lease  # noqa: SLF001
    await guard.shutdown_sweep()
    assert order == ["kill", "teardown"]


@pytest.mark.asyncio
async def test_shutdown_sweep_swallows_one_bad_teardown(tmp_path):
    guard = DockerRunGuard(max_docker=1, scratch_root=str(tmp_path))
    swept = []

    bad = EnvLease(workdir=tmp_path / "a", session="ext-a-1", task_id="a")

    async def bad_teardown():
        raise RuntimeError("teardown failed")

    bad.teardown = bad_teardown

    good = EnvLease(workdir=tmp_path / "b", session="ext-b-1", task_id="b")

    async def good_teardown():
        swept.append("b")

    good.teardown = good_teardown

    guard._inflight = {bad.session: bad, good.session: good}  # noqa: SLF001
    # Should not raise; the good lease still tears down.
    await guard.shutdown_sweep()
    assert "b" in swept


@pytest.mark.asyncio
async def test_shutdown_sweep_bounds_a_wedged_teardown(tmp_path, monkeypatch):
    """A teardown that hangs forever must not stall reaping the other leases:
    the sweep bounds each lease and runs them concurrently."""
    import meta_n.core.external_agents.concurrency as conc

    monkeypatch.setattr(conc, "_SWEEP_TIMEOUT_S", 0.05)
    guard = DockerRunGuard(max_docker=2, scratch_root=str(tmp_path))
    swept = []

    wedged = EnvLease(workdir=tmp_path / "w", session="ext-w-1", task_id="w")

    async def wedged_teardown():
        await asyncio.sleep(100)  # never returns within the bound

    wedged.teardown = wedged_teardown

    good = EnvLease(workdir=tmp_path / "g", session="ext-g-1", task_id="g")

    async def good_teardown():
        swept.append("g")

    good.teardown = good_teardown

    guard._inflight = {wedged.session: wedged, good.session: good}  # noqa: SLF001
    # Returns promptly despite the wedged lease (bounded + concurrent).
    await asyncio.wait_for(guard.shutdown_sweep(), timeout=5)
    assert "g" in swept  # the good lease tore down regardless


@pytest.mark.asyncio
async def test_shutdown_sweep_bounds_a_blocking_hard_kill(tmp_path, monkeypatch):
    """A synchronous hard_kill that blocks is run off-thread and bounded, so it
    cannot stall the sweep."""
    import threading

    import meta_n.core.external_agents.concurrency as conc

    monkeypatch.setattr(conc, "_SWEEP_TIMEOUT_S", 0.05)
    guard = DockerRunGuard(max_docker=1, scratch_root=str(tmp_path))
    lease = EnvLease(workdir=tmp_path / "wd", session="ext-x-1", task_id="x")
    # A real Event released after the sweep returns, so the off-thread worker
    # does not linger past the test (a bare wait(100) would leak the thread).
    release = threading.Event()
    entered = threading.Event()

    def _blocking_kill():
        entered.set()
        release.wait(30)

    lease.hard_kill = _blocking_kill
    guard._inflight = {lease.session: lease}  # noqa: SLF001
    try:
        # Bounded: returns despite the blocking hard_kill still running.
        await asyncio.wait_for(guard.shutdown_sweep(), timeout=5)
        assert entered.wait(5)  # the kill was actually dispatched off-thread
    finally:
        release.set()  # let the worker thread exit promptly


@pytest.mark.asyncio
async def test_inner_semaphore_bounds_concurrency(tmp_path):
    guard = DockerRunGuard(max_docker=1, scratch_root=str(tmp_path))
    held = []
    max_seen = 0

    async def worker():
        nonlocal max_seen
        async with guard.lease(make_task()):
            held.append(1)
            max_seen = max(max_seen, len(held))
            await asyncio.sleep(0.01)
            held.pop()

    await asyncio.gather(worker(), worker(), worker())
    # With max_docker=1 only one lease is ever held at a time.
    assert max_seen == 1
