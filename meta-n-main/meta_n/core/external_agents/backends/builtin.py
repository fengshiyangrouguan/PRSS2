"""Builtin backend — the A/B control that reproduces meta-n's native solver path.

This is WAVE 3 of the external-agents integration (plan §2.3 FIX, L191-199). The
:class:`BuiltinBackend` is the one backend that does **not** drive an external
agent: it wraps meta-n's own :class:`~meta_n.core.solver.Layer1Solver` /
:class:`~meta_n.core.agentic_solver.AgenticSolver` (or a ``MetaLayer`` chain built
on top of them) so that running it *through the external-agent spine* exercises a
same-harness control whose only difference from the OpenHands / Terminus 2
backends is the wrapper plumbing, not the solver.

Reachability (read this before wiring)
--------------------------------------
``BuiltinBackend`` is **not** reachable via ``--base-solver builtin`` through the
``EvolutionaryOrchestrator``. By design (and matching the orchestrator's own
``EvolutionaryConfig.base_solver`` docstring), ``base_solver == "builtin"`` keeps
the legacy native ``Layer1Solver`` / ``AgenticSolver`` / ``MetaLayer``-chain
dispatch byte-for-byte unchanged — the orchestrator never instantiates this class
for that flag, and the spine collaborators (run guard / cost guard / telemetry)
are not even constructed. ``BuiltinBackend`` is therefore reachable only by code
that drives the spine directly: the ``ExternalAgentSolver`` harness and the
unit/parity tests (e.g. ``tests/external_agents/test_builtin_parity.py``). It is
kept as the spine-side A/B control so a future phase that routes the native
solver through the spine has a ready, tested same-harness baseline; until that
wiring exists, treat it as test/harness-only.

Why it is the token-accounting exception
-----------------------------------------
Every other backend runs an agent that talks to its *own* LLM client, so its
spend is reported as ``Trace.inner_*`` and the spine returns ``0`` outer tokens
(``outer_token_mode=False``). The builtin backend is different: the wrapped
:class:`Layer1Solver`/:class:`AgenticSolver` call through the **outer**
:class:`~meta_n.core.llm_client.LLMClient` (the same ``cumulative_usage`` ledger
the orchestrator reads into ``summary.json``). Reporting those as *inner* tokens
would double-count them against the outer ledger. So this backend sets
``outer_token_mode = True`` and reports the native outer token counts in the
``AgentRunResult.agent_*`` fields; the spine, seeing ``outer_token_mode``, returns
them as the native *outer* int rather than remapping to inner, and the telemetry
row is stamped ``token_basis="outer"`` (plan §2.3, §7.11). ``cost_basis`` is
``"priced_from_tokens"`` — the builtin path is priced from those token counts via
the same ledger as the rest of meta-n.

How it integrates with the spine
--------------------------------
The native solver chain is supplied at construction (the orchestrator builds it
with its existing ``_build_solver_from_candidate`` logic, so the injected-code
chain is byte-for-byte the legacy one — the spine's ``InjectionMapper`` plan is
*not* used to re-derive behavior here; that mapping targets the agent backends).
``run`` recovers the :class:`~meta_n.core.meta_layer.TaskDescription` from the
opaque ``ctx.workspace`` handle, routes to ``solve()`` (depth-1, single-shot) or
``execute()`` (depth>1 / agentic) exactly as the orchestrator's
``_evaluate_candidate`` does, snapshots the outer ``LLMClient.cumulative_usage``
across the call to recover the prompt/completion/cached/calls split, and returns
a uniform :class:`AgentRunResult`. The env provider / scorer still run around this
(staging is a no-op for builtin, scoring delegates to the adapter), so the row's
``score``/``success`` are the scorer's, while the std streams and reasoning come
from the native ``Trace``.

Import discipline
-----------------
Like every file under ``external_agents/``, this module imports cleanly without
``openhands`` / ``terminal_bench`` / ``docker`` installed: it touches only WAVE
0-1 siblings (:mod:`..backend`, :mod:`..terminated`) and the in-tree native
solvers. ``run`` never raises (the spine relies on this to keep the evaluation
``gather`` alive); any failure is encoded in the returned result's
``terminated_by`` / ``failure_mode``, except :class:`asyncio.CancelledError`,
which propagates.
"""

from __future__ import annotations

import asyncio
import logging
import time

from .._outer_token_attribution import snapshot_usage, usage_delta
from ..backend import (
    AgentBackend,
    AgentRunContext,
    AgentRunResult,
    task_from_workspace,
)
from ..terminated import TerminatedBy

logger = logging.getLogger("meta_n.external_agents.backends.builtin")

__all__ = ["BuiltinBackend"]


class BuiltinBackend(AgentBackend):
    """Wraps the native ``Layer1Solver``/``AgenticSolver`` as an A/B control.

    Reachable only via the spine harness/tests, **not** via ``--base-solver
    builtin`` through the orchestrator (that flag stays on the legacy native
    dispatch — see the module docstring's "Reachability" note). Reproduces the
    legacy ``EvolutionaryOrchestrator`` solver path (same outer
    :class:`~meta_n.core.llm_client.LLMClient`, same ``solve()``/``execute()``
    routing) so it is a same-harness baseline for the external agent backends.
    Unique among backends in reporting **native outer tokens**
    (``outer_token_mode = True``); the spine returns those as the outer int rather
    than remapping to ``Trace.inner_*`` (plan §2.3 FIX, §7.11).

    Args:
        solver: The pre-built native solver for this candidate — a
            :class:`~meta_n.core.solver.Layer1Solver`, a
            :class:`~meta_n.core.agentic_solver.AgenticSolver`, or a ``MetaLayer``
            chain wrapping one. Built by the orchestrator's existing
            ``_build_solver_from_candidate`` so the injected-code chain is
            byte-for-byte the legacy one.
        llm_client: The outer LLM client whose ``cumulative_usage`` ledger the
            wrapped solver calls through. Snapshotted across the run to recover the
            prompt/completion/cached/calls split for telemetry. ``None`` falls back
            to the bare token total returned by the solver.
        use_agentic: Whether ``solver`` is the agentic loop (always routed via
            ``execute()``). Mirrors ``EvolutionaryConfig.use_agentic``.
        solver_language: ``"bash"`` (default) or ``"python"``. Carried for
            parity with the orchestrator's solver-build kwargs and stored as
            metadata only; it does **not** affect routing. The single-shot vs
            chain/agentic dispatch in ``_invoke_native`` branches solely on
            ``self.use_agentic or self.depth > 1`` (matching ``BuiltinTBBackend``'s
            honest "metadata only" note for the same field).
        executor: The base executor used to run a depth-1 single-shot script
            (``Layer1Solver.solve`` only produces the script; the legacy path then
            runs it through ``executor.execute``). Unused for the agentic / chain
            ``execute()`` paths, which run the executor themselves.
        depth: The candidate's solver depth, used to pick the single-shot
            (``depth == 1``) vs chain (``depth > 1``) route, mirroring the
            orchestrator's ``_evaluate_candidate`` dispatch.
    """

    name = "builtin"
    #: Native outer-token accounting — returned as the outer int, not inner.
    outer_token_mode = True

    def __init__(
        self,
        *,
        solver: object,
        llm_client: object | None = None,
        use_agentic: bool = False,
        solver_language: str = "bash",
        executor: object | None = None,
        depth: int = 1,
    ) -> None:
        self.solver = solver
        self.llm_client = llm_client
        self.use_agentic = bool(use_agentic)
        self.solver_language = solver_language  # metadata only; not read for routing
        self.executor = executor
        self.depth = int(depth)

    async def run(
        self,
        ctx: AgentRunContext,
        tel: object,
        rec: object,
    ) -> AgentRunResult:
        """Drive the native solver for one task and return a uniform result.

        Routes exactly as the legacy ``_evaluate_candidate`` does: the agentic
        loop and any depth>1 chain go through ``solver.execute(task)`` (which runs
        the executor internally and returns a populated ``Trace``); a plain
        depth-1 ``Layer1Solver`` goes through ``solver.solve(task)`` followed by
        ``executor.execute(script, task)``. The outer ``LLMClient.cumulative_usage``
        ledger is snapshotted across the call so the prompt/completion/cached/calls
        split can be reported even though the native solvers return only a total.

        Never raises except :class:`asyncio.CancelledError`; any other failure is
        encoded as :attr:`TerminatedBy.AGENT_ERROR` (or
        :attr:`TerminatedBy.ENV_ERROR` when the task could not be recovered).

        Args:
            ctx: Immutable run inputs; ``ctx.workspace`` carries the task.
            tel: The active telemetry (unused here — the builtin path has no
                per-step stream to log; kept for the uniform backend signature).
            rec: The run record opened by the spine (unused here; the spine folds
                the returned result into it).

        Returns:
            An :class:`AgentRunResult` with the native outer tokens in the
            ``agent_*`` fields and ``cost_basis="priced_from_tokens"``.
        """
        del tel, rec  # uniform signature; builtin path needs neither here
        start = time.monotonic()

        # Shared task-recovery helper (backend.task_from_workspace): the wrapped
        # native solvers need the full TaskDescription (they read task.metadata
        # to pick a prompt/language), and a nested handle needs only task_id here.
        task = task_from_workspace(ctx.workspace)
        if task is None:
            logger.warning(
                "builtin backend: no task recoverable from ctx.workspace (%r); "
                "returning env_error result",
                type(ctx.workspace).__name__,
            )
            return AgentRunResult(
                cost_basis="priced_from_tokens",
                wall_s=time.monotonic() - start,
                terminated_by=TerminatedBy.ENV_ERROR,
                failure_mode="env_error",
                attribution_available=False,
            )

        before = self._usage_snapshot()
        try:
            trace, outer_tokens = await self._invoke_native(task, ctx)
        except asyncio.CancelledError:  # cooperative cancel must propagate
            raise
        except BaseException as exc:  # noqa: BLE001 - run() must never raise
            logger.warning(
                "builtin backend: native solver raised for task=%s: %r",
                getattr(task, "task_id", "?"),
                exc,
            )
            # Recover any outer spend the wrapped solver already incurred before it
            # raised (e.g. an authoring LLM call that succeeded, then the executor
            # or a depth-1-with-no-executor path failed). Without this, the outer
            # ledger delta is dropped and the degraded row records real spend as
            # $0/0-token. Mirrors BuiltinTBBackend._apply's stamping (§2.3 FIX);
            # each component is clamped at 0 by usage_delta.
            e_prompt, e_completion, e_total, e_cached, e_calls = self._usage_delta(
                before
            )
            return AgentRunResult(
                agent_tokens=int(e_total or 0),
                agent_prompt_tokens=int(e_prompt or 0),
                agent_completion_tokens=int(e_completion or 0),
                agent_cached_tokens=int(e_cached or 0),
                agent_calls=int(e_calls or 0),
                cost_usd=0.0,
                cost_basis="priced_from_tokens",
                wall_s=time.monotonic() - start,
                terminated_by=TerminatedBy.AGENT_ERROR,
                # AgentRunResult has no error_summary field (it lives on the
                # record/Trace); carry the cause in failure_mode, truncated. The
                # spine's telemetry folds failure_mode into the record's
                # error_summary on the degraded path.
                failure_mode=f"agent_error: {type(exc).__name__}: {exc}"[:200],
                attribution_available=False,
            )

        delta = self._usage_delta(before)
        prompt_tokens, completion_tokens, total_tokens, cached_tokens, calls = delta

        # Prefer the solver's returned token total (the legacy contract) and fall
        # back to the cumulative-usage delta only when the ledger is unavailable.
        agent_tokens = int(outer_tokens or 0) or int(total_tokens or 0)

        success = bool(getattr(trace, "success", False))
        return AgentRunResult(
            transcript=str(getattr(trace, "stdout", "") or ""),
            reasoning_summary=str(getattr(trace, "reasoning", "") or ""),
            stdout_tail=str(getattr(trace, "stdout", "") or ""),
            stderr_tail=str(getattr(trace, "stderr", "") or ""),
            artifacts_path="",
            agent_tokens=agent_tokens,
            agent_prompt_tokens=int(prompt_tokens or 0),
            agent_completion_tokens=int(completion_tokens or 0),
            agent_cached_tokens=int(cached_tokens or 0),
            agent_calls=int(calls or 0),
            cost_usd=0.0,
            cost_basis="priced_from_tokens",
            wall_s=float(getattr(trace, "duration_s", 0.0) or (time.monotonic() - start)),
            steps=int(calls or 0),
            command_history=[],
            # The native solver does not expose a command stream we can attribute
            # against (helpers ride in via prepend, not as issued commands), so
            # attribution is honestly *unmeasurable* here, not empty (§7.6).
            attribution_available=False,
            terminated_by=(
                TerminatedBy.COMPLETED if success else TerminatedBy.AGENT_ERROR
            ),
            failure_mode=None if success else (
                str(getattr(trace, "error_summary", "") or "")[:200] or "agent_error"
            ),
            native_handle=trace,
        )

    # -- internals -----------------------------------------------------------

    async def _invoke_native(
        self, task: object, ctx: AgentRunContext
    ) -> tuple[object, int]:
        """Run the wrapped native solver, mirroring ``_evaluate_candidate`` routing.

        The agentic loop and any depth>1 ``MetaLayer`` chain expose
        ``execute(task) -> (Trace, tokens)`` and run the executor internally. A
        plain depth-1 :class:`Layer1Solver` exposes ``solve(task) ->
        (script, reasoning, tokens)``; the legacy path then runs that script
        through the executor and stamps the depth/reasoning/duration onto the
        resulting ``Trace`` (the orchestrator's ``_evaluate_candidate`` depth-1
        single-shot path).

        Args:
            task: The recovered :class:`~meta_n.core.meta_layer.TaskDescription`.
            ctx: The run context (unused beyond routing — kept for symmetry with
                the agent backends).

        Returns:
            ``(trace, outer_tokens)`` — the native ``Trace`` and the outer token
            total the solver reported.
        """
        del ctx
        if self.use_agentic or self.depth > 1:
            # Agentic loop / MetaLayer chain: execute() runs the executor itself.
            return await self.solver.execute(task)

        # Depth-1 single-shot Layer1Solver: solve() yields a script; the legacy
        # path runs it through the executor and finalizes the Trace (parity with
        # the orchestrator's _evaluate_candidate depth-1 single-shot path).
        # Monotonic clock: duration_s is a pure delta and must not jump with NTP.
        start = time.monotonic()
        script, reasoning, tokens = await self.solver.solve(task)
        if self.executor is None:
            raise RuntimeError(
                "builtin backend: depth-1 single-shot path requires an executor "
                "to run the solver's script"
            )
        trace = await self.executor.execute(script, task)
        trace.depth = 1
        trace.reasoning = reasoning
        trace.duration_s = time.monotonic() - start
        return trace, int(tokens or 0)

    def _usage_snapshot(self) -> dict[str, int | float]:
        """Snapshot the outer ``LLMClient.cumulative_usage`` ledger, or zeros.

        Thin wrapper over the shared
        :func:`~meta_n.core.external_agents._outer_token_attribution.snapshot_usage`
        (single-sourced with :class:`BuiltinTBBackend`).
        """
        return snapshot_usage(self.llm_client)

    def _usage_delta(
        self, before: dict[str, int | float]
    ) -> tuple[int, int, int, int, int]:
        """Compute the prompt/completion/total/cached/calls delta since ``before``.

        Delegates to the shared
        :func:`~meta_n.core.external_agents._outer_token_attribution.usage_delta`.
        The native solvers return only a total token count, but the outer client
        accumulates the full split on every ``complete()``. Under ``--parallel >
        1`` the cumulative ledger is shared, so this delta is an approximation that
        can absorb a sibling task's concurrent spend; the authoritative per-run
        *total* still comes from the solver's returned int (used as
        ``agent_tokens``), and the split is best-effort. Each component is clamped
        at ``0`` so a racy negative delta never leaks.

        Args:
            before: The :meth:`_usage_snapshot` taken before the native run.

        Returns:
            ``(prompt_tokens, completion_tokens, total_tokens, cached_tokens,
            calls)``.
        """
        return usage_delta(before, self._usage_snapshot())
