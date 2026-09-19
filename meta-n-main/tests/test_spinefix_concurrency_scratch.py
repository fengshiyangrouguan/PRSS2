"""Regression tests for the DockerRunGuard scratch-dir leak (SPINE F1/F3).

Offline: no Docker, no LLM, no network. The bug was that the provisioning
window in ``DockerRunGuard.lease`` (chmod 0o700, sanitize, EnvLease build, and
the ``async with self._lock`` registration) ran OUTSIDE the try/finally that
rmtrees the ``mkdtemp`` scratch dir. So a chmod ``OSError`` (e.g. a
``--scratch-root`` on a chmod-unsupported volume) or a ``CancelledError`` at the
lock await leaked the scratch dir: the finally never ran, and because the lease
was not yet in ``_inflight``, ``shutdown_sweep`` could not reap it either.

Post-fix: mkdtemp stays before the try, but everything after it is under an
outer try whose finally rmtrees the workdir, so no scratch dir survives a failed
lease attempt.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

import pytest

from meta_n.core.external_agents import concurrency
from meta_n.core.external_agents.concurrency import DockerRunGuard


@dataclass
class _Task:
    task_id: str


def _scratch_children(root: Path) -> list[Path]:
    return [p for p in root.iterdir() if p.name.startswith("ext_agent_")]


def test_chmod_failure_does_not_leak_scratch_dir(tmp_path, monkeypatch):
    """A chmod OSError during provisioning must not leave the scratch dir behind."""
    root = tmp_path / "scratch"
    guard = DockerRunGuard(max_docker=1, scratch_root=str(root))

    # Fail exactly at the 0o700 chmod, the first provisioning step after mkdtemp.
    def _boom(*_a, **_k):
        raise OSError("chmod not supported on this volume")

    monkeypatch.setattr(concurrency.os, "chmod", _boom)

    async def _run() -> None:
        with pytest.raises(OSError):
            async with guard.lease(_Task("t1")):
                pytest.fail("lease body must not be entered when chmod fails")

    asyncio.run(_run())

    # No ext_agent_ scratch dir may survive the failed lease attempt.
    assert _scratch_children(root) == []
    # And it must not have been registered as in-flight (nothing for the sweep).
    assert guard._inflight == {}


def test_cancel_at_lock_await_does_not_leak_scratch_dir(tmp_path, monkeypatch):
    """A CancelledError at the registration await must not leak the scratch dir."""
    root = tmp_path / "scratch"
    guard = DockerRunGuard(max_docker=1, scratch_root=str(root))

    real_lock = guard._lock

    class _CancelOnFirstAcquire:
        """Lock proxy that raises CancelledError the first time it is awaited.

        Simulates the task being cancelled exactly at the
        ``async with self._lock: self._inflight[session] = lease`` await point,
        i.e. before the lease is ever registered.
        """

        def __init__(self) -> None:
            self._fired = False

        async def __aenter__(self):
            if not self._fired:
                self._fired = True
                raise asyncio.CancelledError()
            return await real_lock.__aenter__()

        async def __aexit__(self, *exc):
            return await real_lock.__aexit__(*exc)

    monkeypatch.setattr(guard, "_lock", _CancelOnFirstAcquire())

    async def _run() -> None:
        with pytest.raises(asyncio.CancelledError):
            async with guard.lease(_Task("t2")):
                pytest.fail("lease body must not be entered when cancelled")

    asyncio.run(_run())

    assert _scratch_children(root) == []


def test_happy_path_still_provisions_and_cleans(tmp_path):
    """Sanity: a normal lease provisions a 0o700 workdir and rmtrees it on exit."""
    root = tmp_path / "scratch"
    guard = DockerRunGuard(max_docker=1, scratch_root=str(root))

    async def _run() -> Path:
        async with guard.lease(_Task("ok")) as lease:
            assert lease.workdir.is_dir()
            assert lease.session in guard._inflight
            return lease.workdir

    workdir = asyncio.run(_run())

    assert not workdir.exists()
    assert _scratch_children(root) == []
    assert guard._inflight == {}
