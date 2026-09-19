"""Inner concurrency + resource guard for external-agent runs.

This is WAVE 2 of the external-agents integration (plan §6). It owns the single
:class:`DockerRunGuard` that bounds and isolates the *inner* unit of work an
external agent performs, independent of meta-n's *outer* candidate-evaluation
concurrency.

Two-layer model (plan §6.1)
---------------------------
* **Outer (reused, zero edit):** ``EvolutionaryOrchestrator._evaluate_candidate``
  already gates simultaneous candidates with an ``asyncio.Semaphore(parallel)``
  + ``asyncio.gather``. Each :class:`ExternalAgentSolver.execute` runs *inside*
  that gather, so ``--parallel N`` already caps simultaneous agent runs.
* **Inner (new, this module):** ``--max-docker N`` is a second semaphore,
  clamped ``<= --parallel``, that bounds how many Docker sandboxes (or, for the
  no-Docker CO-Bench path, how many host scratch leases) one candidate's gather
  may hold open at once, and guarantees per-run isolation + cleanup.

One :class:`DockerRunGuard` is constructed per orchestrator and shared by
reference across every :class:`ExternalAgentSolver` (plan §6.7). Because
candidates are evaluated sequentially, the guard only ever arbitrates one
candidate's gather width at a time.

Isolation & cleanup (plan §6.3, §6.5)
-------------------------------------
Every lease gets a private host scratch directory
(``mkdtemp(prefix="ext_agent_")``, mode ``0o700`` — parity with the tb2 lease
``mkdtemp`` in ``terminal_bench.py``) and a compose-safe session name
(``ext-{task_id}-{uuid}`` — same naming parity), which the
env provider reuses as the Docker Compose project name so the container,
network, and volume all share the ``ext-`` prefix that
``scripts/reap_external_agents.sh`` matches.

Cleanup is layered so a hard kill always fires on *every* exit path (plan §6.5d):
the lease ``finally`` invokes the provider-set synchronous ``hard_kill`` closure
(forcibly stopping a wedged ``to_thread`` worker that ``asyncio.wait_for`` could
not interrupt), pops the lease from the in-flight registry, then
``shutil.rmtree``s the scratch dir. :meth:`DockerRunGuard.shutdown_sweep`
provides the graceful-shutdown / SIGINT backstop: for each still-in-flight lease
it runs ``hard_kill`` (sync) then ``teardown`` (async). Every step swallows its
own errors so cleanup of one lease never blocks another.

This module imports only the standard library and the WAVE-1 ``.env`` sibling
(for :class:`EnvLease`); it pulls in no external SDK, so the ``external_agents``
package stays importable without ``openhands``, ``terminal_bench`` or ``docker``
installed.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path
from uuid import uuid4

from ._bridge import sanitize_compose_name
from .env import EnvLease

__all__ = ["DockerRunGuard"]

logger = logging.getLogger(__name__)

#: Per-lease wall bound (s) for ``shutdown_sweep`` hard_kill / teardown so one
#: wedged docker daemon or blocking kill cannot stall reaping the other leases.
#: Generous enough for the T2 ``_force_compose_down`` subprocesses (30/60/30s
#: internal bounds) to finish their own waits before this outer bound fires.
_SWEEP_TIMEOUT_S = 120.0


class DockerRunGuard:
    """Semaphore-gated lease manager for isolated external-agent runs.

    Bounds the number of concurrently-open inner sandboxes/scratch leases to
    ``max_docker`` and guarantees per-run isolation and cleanup. One guard is
    shared by reference across every :class:`ExternalAgentSolver` in an
    orchestrator (plan §6.7).

    The guard itself is Docker-agnostic: it provisions only the host scratch
    directory and a compose-safe session name, then yields an :class:`EnvLease`
    that the env provider fills in with backend-specific ``teardown`` /
    ``hard_kill`` closures. The guard invokes those closures on cleanup but never
    talks to Docker (or any SDK) directly.

    Attributes:
        max_docker: The configured inner concurrency cap (``>= 1``).
    """

    def __init__(self, max_docker: int, *, scratch_root: str | None = None) -> None:
        """Build a guard with an inner concurrency cap of ``max_docker``.

        Args:
            max_docker: Maximum number of leases held open at once. Must be
                ``>= 1``; the orchestrator clamps it to ``<= --parallel`` before
                construction (plan §6.4).
            scratch_root: Optional parent directory for per-lease scratch dirs;
                ``None`` uses the system temp root. A dedicated fast/large
                scratch volume can be pointed here for big runs.

        Raises:
            ValueError: If ``max_docker < 1``.
        """
        if max_docker < 1:
            raise ValueError(f"max_docker must be >= 1, got {max_docker}")
        self.max_docker = max_docker
        self._sem = asyncio.Semaphore(max_docker)
        self._scratch_root = scratch_root
        # Ensure the scratch root exists so per-lease ``mkdtemp(dir=...)`` cannot
        # FileNotFoundError into a silent all-deny: an operator passing a fresh
        # ``--scratch-root`` (or a test pointing at a tmp subdir) should not have
        # every lease fail. ``None`` uses the system temp root, which always
        # exists. Idempotent + best-effort; a genuinely unwritable root still
        # surfaces at the first lease.
        if scratch_root:
            try:
                os.makedirs(scratch_root, exist_ok=True)
            except OSError:
                logger.warning(
                    "could not pre-create scratch_root=%s; per-lease mkdtemp may "
                    "fail (the spine degrades such runs rather than raising)",
                    scratch_root,
                )
        # session -> lease, so shutdown_sweep can tear down anything still open.
        self._inflight: dict[str, EnvLease] = {}
        self._lock = asyncio.Lock()  # guards _inflight

    @contextlib.asynccontextmanager
    async def lease(self, task, *, time_limit_s: float | None = None):
        """Acquire an isolated lease for one external-agent run.

        Blocks on the inner semaphore until a slot is free, then provisions a
        private ``0o700`` scratch directory and a compose-safe session name and
        yields an :class:`EnvLease`. The env provider is expected to populate the
        yielded lease's ``teardown`` / ``hard_kill`` closures while it holds it.

        On exit (success, error, or cancellation) the ``finally`` block always:
        runs ``lease.hard_kill`` (if set) to stop a wedged worker, removes the
        lease from the in-flight registry, and ``rmtree``s the scratch directory.
        The semaphore slot is released by the ``async with`` unwind. Every
        cleanup step swallows its own errors so one failing teardown never leaks
        the slot or blocks siblings (plan §6.5).

        Args:
            task: The task being run; only ``task.task_id`` is read, for the
                scratch dir / session naming.
            time_limit_s: Soft wall budget forwarded for logging/diagnostics; the
                guard does not itself enforce it (the spine wraps the run in a
                hard ``asyncio.wait_for``).

        Yields:
            EnvLease: The provisioned lease (scratch ``workdir``, ``session``,
            ``task_id``), ready for the provider to attach teardown/kill hooks.
        """
        async with self._sem:
            # ``mkdtemp`` stays OUTSIDE the try: if it fails there is nothing to
            # clean up. Everything that follows — the ``0o700`` chmod, the
            # sanitize, the ``EnvLease`` build, and the ``async with self._lock``
            # registration (an await point) — provisions or references ``workdir``
            # and so MUST run under the try whose finally rmtrees it. Otherwise a
            # chmod ``OSError`` (e.g. ``--scratch-root`` on a chmod-unsupported
            # volume) or a ``CancelledError`` at the lock await leaks the scratch
            # dir: the finally never runs and, not yet in ``_inflight``,
            # ``shutdown_sweep`` cannot reap it either.
            workdir = Path(
                tempfile.mkdtemp(prefix="ext_agent_", dir=self._scratch_root)
            )
            # ``session``/``lease`` may still be unset if provisioning fails before
            # they are assigned; keep the finally null-safe on both.
            session: str | None = None
            lease: EnvLease | None = None
            t0 = time.monotonic()
            try:
                # Force 0o700: the scratch dir is bind-mounted into the agent
                # container and may hold traces/keys; a permissive umask would leak
                # it on a shared host (parity with the tb2 lease mkdtemp in
                # terminal_bench.py). Session naming goes straight through the
                # single canonical sanitizer (_bridge.sanitize_compose_name).
                os.chmod(workdir, 0o700)
                session = sanitize_compose_name(f"ext-{task.task_id}-{uuid4().hex[:8]}")
                lease = EnvLease(workdir=workdir, session=session, task_id=task.task_id)
                async with self._lock:
                    self._inflight[session] = lease
                logger.debug(
                    "lease acquire session=%s slots=%d/%d time_limit_s=%s",
                    session,
                    self.max_docker - self._sem._value,  # noqa: SLF001 - diagnostic only
                    self.max_docker,
                    time_limit_s,
                )
                yield lease
            finally:
                # 6.5d: hard-kill any wedged worker the wall timeout could not
                # interrupt. Sync closure, set by the provider. ``lease`` may be
                # None if provisioning failed before it was built.
                if lease is not None and lease.hard_kill is not None:
                    try:
                        lease.hard_kill()
                    except Exception:  # never let cleanup raise
                        logger.debug(
                            "hard_kill raised on lease release session=%s",
                            session,
                            exc_info=True,
                        )
                # Only pop if registration actually happened (session set).
                if session is not None:
                    async with self._lock:
                        self._inflight.pop(session, None)
                shutil.rmtree(workdir, ignore_errors=True)
                logger.debug(
                    "lease release session=%s dur=%.1fs",
                    session,
                    time.monotonic() - t0,
                )

    async def shutdown_sweep(self) -> None:
        """Hard-kill and tear down every still-in-flight lease.

        The graceful-shutdown / SIGINT backstop (plan §6.7): the orchestrator
        calls this in its ``run()`` ``finally`` to reap leases whose lease
        context did not unwind normally (e.g. an interrupted gather). For each
        lease still registered it runs the synchronous ``hard_kill`` then awaits
        the asynchronous ``teardown`` (idempotent ``compose down`` for Docker
        providers). Per-lease failures are logged and swallowed so one bad
        teardown never aborts the sweep.

        A snapshot of the in-flight registry is taken under the lock; leases are
        torn down outside the lock so a concurrent lease ``finally`` can still
        pop itself without deadlocking.

        Each lease's teardown is bounded by an :func:`asyncio.wait_for` and the
        leases are reaped concurrently via :func:`asyncio.gather`, so one wedged
        teardown (e.g. a stuck docker daemon, or a blocking ``hard_kill`` psutil
        walk) cannot stall reaping the rest on shutdown.
        """
        async with self._lock:
            leases = list(self._inflight.values())
        if not leases:
            return

        async def _one(lease: EnvLease) -> None:
            try:
                if lease.hard_kill is not None:
                    # hard_kill is synchronous; bound it via a thread so a blocking
                    # psutil walk / wedged kill cannot stall this lease's reap.
                    await asyncio.wait_for(
                        asyncio.to_thread(lease.hard_kill), timeout=_SWEEP_TIMEOUT_S
                    )
                if lease.teardown is not None:
                    await asyncio.wait_for(
                        lease.teardown(), timeout=_SWEEP_TIMEOUT_S
                    )
            except Exception as e:  # one bad teardown must not abort the sweep
                logger.warning("sweep failed %s: %s", lease.session, e)

        await asyncio.gather(
            *(_one(lease) for lease in leases), return_exceptions=True
        )
