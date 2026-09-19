"""Environment & scoring interfaces for the external-agent spine.

This module owns the *per-benchmark* boundary of the external-agents
integration: how a run is given an isolated place to work (the
:class:`AgentEnvProvider`), how the resulting solution is read back out, and how
that solution is scored (the :class:`Scorer`). It also defines the canonical
:class:`EnvLease` dataclass — the single record handed between the concurrency
guard, the env provider and the spine.

The three abstractions:

* :class:`EnvLease` — the lease the :class:`DockerRunGuard` hands to a run: an
  isolated host scratch directory, an OS-isolation session name, the task id, and
  two provider-installed cleanup hooks (a synchronous ``hard_kill`` and an async
  ``teardown``). This is the **canonical** definition; ``concurrency.py`` imports
  it from here rather than re-declaring it (plan §2.5, §6.2 — single source).
* :class:`AgentEnvProvider` — an async-context-manager strategy that *provisions*
  an environment for a task inside a lease, *stages* the injection plan's helper
  files into it, and *extracts* the final solution text out of it once the agent
  has run. Concrete providers live next to their benchmark adapter
  (``COBenchEnvProvider`` in ``co_bench.py``;
  ``TBTerminus2EnvProvider``/``TBExternalEnvProvider`` in
  ``terminal_bench.py``), not here.
* :class:`Scorer` — turns ``(task, env, solution, run)`` into an
  :class:`~meta_n.integrations.benchmark.EvalResult` carrying the normalized
  ``score``, the benchmark-native ``raw_score``, and the new ``valid`` /
  ``feasible`` flags. Concrete scorers (e.g.
  ``TBTerminus2Scorer``/``TBExternalScorer``) also live next to their adapter
  so they can reach ``_read_reward`` / ``_evaluate_on_split`` in-module.

Design notes
------------
This file is WAVE 1: it imports only the standard library at runtime. The one
meta-n type it references — :class:`~meta_n.integrations.benchmark.EvalResult` —
is pulled in **only** under :data:`typing.TYPE_CHECKING`, so the
``external_agents`` package imports cleanly without ``openhands``,
``terminal_bench`` or ``docker`` installed. Any external SDK used by a concrete
provider/scorer is imported lazily inside that subclass's methods (and those
subclasses live in the adapter modules), never here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, AsyncIterator, Awaitable, Callable

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids runtime import cycles
    from meta_n.core.meta_layer import TaskDescription
    from meta_n.integrations.benchmark import EvalResult

    from .backend import AgentRunResult


__all__ = [
    "EnvLease",
    "AgentEnvProvider",
    "Scorer",
    "mirror_agent_tokens",
]


@dataclass
class EnvLease:
    """An isolated, time-bounded slot in which one external-agent run executes.

    A lease is minted by :class:`~meta_n.core.external_agents.concurrency.DockerRunGuard`
    (one per run, under the inner ``--max-docker`` semaphore) and threaded through
    the spine into the env provider. It bundles the host scratch directory, the
    OS-isolation session name, the task id, and the two cleanup hooks that a
    provider installs after it has provisioned real resources.

    This is the **single canonical** definition of ``EnvLease`` (plan §2.5, §6.2);
    ``concurrency.py`` imports it from this module rather than re-declaring it, so
    there is exactly one source of truth for the lease shape.

    Attributes:
        workdir: Per-run host scratch directory (created with ``mkdtemp`` /
            ``0o700`` by the guard; ``rmtree``'d on lease release). Used as the
            bind-mount root / staging area for the agent's workspace.
        session: Compose-safe, sanitized session name
            (``ext-{task_id}-{uuid4()[:8]}``). Drives container/network/volume
            naming for OS-level isolation and the crash-leak reaper's
            ``--filter name=ext-`` match. It is **not** the telemetry ``run_id``.
        task_id: The originating task's id, kept on the lease for logging and the
            shutdown sweep.
        teardown: Provider-installed async cleanup closure (e.g. idempotent
            ``compose down --remove-orphans --volumes``), or ``None`` before the
            provider has provisioned anything. Awaited during
            :meth:`DockerRunGuard.shutdown_sweep`.
        hard_kill: Provider-installed **synchronous** kill closure that forcibly
            stops the unit of work even when an ``asyncio`` timeout cannot
            interrupt a ``to_thread`` worker (e.g. ``container.kill()``,
            conversation DELETE, ``killpg(pgid, SIGKILL)``), or ``None``. Invoked
            in both the lease ``finally`` and the shutdown sweep (plan §6.5d).
    """

    workdir: Path
    session: str
    task_id: str
    teardown: Callable[[], Awaitable[None]] | None = None
    hard_kill: Callable[[], None] | None = None


class AgentEnvProvider(ABC):
    """Strategy that provisions, stages, and reads back a per-benchmark environment.

    A provider is the benchmark-specific half of the external-agent spine: it
    knows how to stand up an isolated environment for a task (a Docker compose
    project for TerminalBench, a host scratch dir for CO-Bench), how to write the
    injection plan's helper files into it, and how to recover the agent's final
    solution (a diff, a ``solve.py``, …) afterwards.

    Concrete providers live next to their adapter (``COBenchEnvProvider`` in
    ``co_bench.py``; ``TBTerminus2EnvProvider``/``TBExternalEnvProvider`` in
    ``terminal_bench.py``) so they can reach module-private helpers
    (``_ensure_image``, ``_read_reward``, ``_evaluate_on_split``) without
    import gymnastics. Any external SDK such a
    provider needs is imported lazily inside its own methods.

    Lifecycle, driven by :class:`ExternalAgentSolver._run_leased`::

        async with provider.provision(task, lease) as env:
            await provider.stage_files(env, plan.staged_files)
            run = await backend.run(ctx, ...)          # agent acts on ``env``
            solution = await provider.extract_solution(env, run)
            evalr = await scorer.score(task, env, solution, run)
    """

    @abstractmethod
    @asynccontextmanager
    async def provision(
        self, task: "TaskDescription", lease: EnvLease
    ) -> AsyncIterator[object]:
        """Provision an isolated environment for ``task`` inside ``lease``.

        Implemented as an ``@asynccontextmanager``: it yields an opaque, provider
        -specific ``env`` handle (e.g. an object bundling the container/session
        handle, a ``workspace_handle`` for the backend, and a verifier directory)
        and **must** guarantee cleanup in its ``finally`` block. Providers should
        also install ``lease.teardown`` and ``lease.hard_kill`` here, once the
        real resources exist, so the guard's lease ``finally`` and
        :meth:`DockerRunGuard.shutdown_sweep` can force-clean wedged runs.

        Args:
            task: The task being solved.
            lease: The :class:`EnvLease` providing the scratch dir, session name,
                and the cleanup-hook slots to populate.

        Yields:
            An opaque, provider-specific environment handle consumed by
            :meth:`stage_files`, :meth:`extract_solution`, the backend, and the
            scorer.
        """
        raise NotImplementedError
        yield  # pragma: no cover - makes this an (abstract) async generator

    @abstractmethod
    async def stage_files(self, env: object, files: dict[str, str]) -> None:
        """Write the injection plan's helper files into the provisioned env.

        Args:
            env: The environment handle yielded by :meth:`provision`.
            files: Map of workspace-relative path (e.g. ``helpers/foo.py``) to
                file contents, taken from :attr:`InjectionPlan.staged_files`. An
                empty map (the gen0 vanilla-agent baseline) is a valid no-op.
        """
        raise NotImplementedError

    @abstractmethod
    async def extract_solution(self, env: object, run: "AgentRunResult") -> str:
        """Read the agent's final solution out of the provisioned env.

        Returns the benchmark-native solution text — a unified diff for
        TerminalBench / SWE-bench, the contents of ``solve.py`` for CO-Bench, etc.
        — which the spine records as ``Trace.script`` and the :class:`Scorer`
        evaluates.

        Args:
            env: The environment handle yielded by :meth:`provision`.
            run: The :class:`AgentRunResult` returned by the backend, available in
                case the solution must be reconstructed from the transcript /
                artifacts rather than read from the filesystem.

        Returns:
            The solution text (``""`` when the agent produced nothing).
        """
        raise NotImplementedError


class Scorer(ABC):
    """Strategy that scores an extracted solution against a benchmark task.

    A scorer maps ``(task, env, solution, run)`` onto an
    :class:`~meta_n.integrations.benchmark.EvalResult`, which carries the binary
    ``success``, the normalized ``score``, the benchmark-native ``raw_score``, the
    new ``valid`` / ``feasible`` flags, human-readable ``feedback``, and the
    mirrored ``inner_*`` agent-token accounting. The spine's
    :meth:`AgentTelemetry.build_trace` then folds ``evalr.score`` / ``evalr.success``
    into the :class:`~meta_n.core.meta_layer.Trace`.

    Concrete scorers (e.g. ``TBTerminus2Scorer``/``TBExternalScorer``,
    ``COBenchScorer``) live next
    to their adapter and delegate to existing meta-n evaluation helpers
    (``_read_reward`` for the compose reward file, ``_evaluate_on_split`` for
    CO-Bench) rather than reimplementing scoring logic.
    """

    @abstractmethod
    async def score(
        self,
        task: "TaskDescription",
        env: object,
        solution: str,
        run: "AgentRunResult",
    ) -> "EvalResult":
        """Score ``solution`` for ``task`` and return an :class:`EvalResult`.

        Implementations must never raise: a failed/empty solution scores as
        ``EvalResult(success=False, score=0.0, ...)``. The returned result should
        mirror the agent's inner-token spend
        (``inner_tokens``/``inner_prompt_tokens``/…) from ``run`` so cost analysis
        attributes it correctly (plan §5.8).

        Args:
            task: The task being scored.
            env: The environment handle yielded by
                :meth:`AgentEnvProvider.provision` (still live — the verifier may
                need to run *inside* it).
            solution: The solution text from
                :meth:`AgentEnvProvider.extract_solution`.
            run: The backend's :class:`AgentRunResult`, for mirroring token
                accounting and surfacing failure context.

        Returns:
            An :class:`~meta_n.integrations.benchmark.EvalResult`.
        """
        raise NotImplementedError


def mirror_agent_tokens(run: object) -> tuple[int, int, int, int]:
    """Read the agent-token quad defensively off an :class:`AgentRunResult`.

    Canonical helper for the :class:`Scorer` mirroring contract above: every
    scorer copies the agent's authoring spend from ``run`` into the
    ``EvalResult.inner_*`` fields, and each field must degrade to ``0`` when
    the attribute is missing or ``None`` (fake runs in tests, partial backend
    results).

    Args:
        run: The backend's :class:`AgentRunResult` (or any duck-typed stand-in).

    Returns:
        ``(agent_tokens, agent_prompt_tokens, agent_completion_tokens,
        agent_calls)`` as ints; missing/``None`` attributes coerce to ``0``.
    """
    return (
        int(getattr(run, "agent_tokens", 0) or 0),
        int(getattr(run, "agent_prompt_tokens", 0) or 0),
        int(getattr(run, "agent_completion_tokens", 0) or 0),
        int(getattr(run, "agent_calls", 0) or 0),
    )
