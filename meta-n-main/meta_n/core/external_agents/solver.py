"""The external-agent spine — :class:`ExternalAgentSolver` (plan §2.3).

This is WAVE 3 of the external-agents integration: the single object that
presents meta-n's *solver contract* (``execute``/``solve``) for a self-contained
external agent (the builtin Layer1Solver/AgenticSolver wrapper, OpenHands, or
Terminus 2). The orchestrator's dispatch (the ``_evaluate_candidate`` solver
dispatch) treats anything exposing ``execute(task) -> (Trace, int)`` as a peer of
``AgenticSolver``, so this wrapper *is a solver* — not a ``BaseExecutor`` (an
executor takes a finished script; an agent emits none, plan §2.1).

Lifecycle (the ordering this class owns, plan §2.3)::

    budget pre-check        # CostGuard.precheck — never raises
      -> injection.build    # InjectionMapper -> InjectionPlan
      -> run_guard.lease    # DockerRunGuard — scratch dir + isolation + cleanup
      -> env.provision      # AgentEnvProvider — stand up the per-benchmark env
      -> stage files        # write helpers/ into the env
      -> backend.run        # the pluggable agent strategy drives the task
      -> scorer.score       # EvalResult (score / success / valid / feasible)
      -> build Trace        # score, inner tokens, reasoning, script
      -> telemetry.record   # flock-append the AgentRunRecord; feed the ledger

The actual agent is driven via a pluggable
:class:`~meta_n.core.external_agents.backend.AgentBackend`; the per-benchmark
environment and scoring are supplied by adapter factory hooks
(:class:`~meta_n.core.external_agents.env.AgentEnvProvider` /
:class:`~meta_n.core.external_agents.env.Scorer`).

NEVER-RAISE invariant (plan §6.5)
---------------------------------
The orchestrator's evaluation ``asyncio.gather`` (in ``_evaluate_candidate``)
runs with ``return_exceptions=True`` and degrades a raised exception into a
failed Trace — but an escape from :meth:`execute` would still lose the run's
telemetry row and diagnostics. We therefore enforce never-raise as a hard
invariant: :meth:`_execute_with_plan` wraps the injection-plan build AND the
whole lease body in ``try/except BaseException`` (so it also catches
``BudgetExceededError`` and ``asyncio.TimeoutError``), re-raising **only**
``asyncio.CancelledError`` (cooperative cancellation must propagate). Every other
failure becomes a degraded ``(Trace, 0)`` with a written telemetry row.

Token contract (plan §2.3 FIX, §7.1)
------------------------------------
:meth:`execute` returns ``(Trace, 0)`` for OpenHands / Terminus 2 because those
agents use their *own* LLM client; their spend rides in ``Trace.inner_*``. The
**builtin backend is the exception** (``outer_token_mode=True``): its calls flow
through the outer ``LLMClient``, so the spine returns its native *outer* token
count as the int rather than remapping to inner. The discriminator is a property
of the backend, not of the spine.

Pure-import
-----------
This module imports only stdlib + zero-/wave-1/2 ``external_agents`` siblings and
``meta_n.core.meta_layer``; it touches **no** external SDK (``openhands`` /
``terminal_bench`` / ``docker``) at module scope or in its own methods — those
are imported lazily inside the concrete backends/providers it merely orchestrates.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Optional

from meta_n.core.meta_layer import SandboxMarker, Trace

from ._bridge import hard_timeout
from .backend import AgentRunContext
from .injection import InjectionMapper
from .terminated import TerminatedBy

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids runtime import cycles
    from meta_n.core.meta_layer import InjectedCode, TaskDescription

    from .backend import AgentBackend, InjectionPlan
    from .budget import CostGuard
    from .concurrency import DockerRunGuard
    from .env import AgentEnvProvider, EnvLease, Scorer
    from .telemetry import AgentRunRecord, AgentTelemetry

logger = logging.getLogger("meta_n.external_agents.solver")

__all__ = ["ExternalAgentSolver"]

#: H5: extra slack (s) added on top of the backend's OWN hard envelope
#: (``_bridge.hard_timeout(time_limit_s)``) before the outer ``asyncio.wait_for``
#: fires. The outer envelope must STRICTLY EXCEED the backend hard timeout plus
#: provision + scoring + lease teardown so the backend's hard kill wins the race
#: and returns a degraded-but-token-bearing ``AgentRunResult`` (recorded via the
#: normal ``finish_record`` path, so TIMEOUT-vs-COMPLETED and token/cost are
#: correct). The outer ``wait_for`` is only a true last-resort wall behind it.
_OUTER_GRACE_S = 300.0


class ExternalAgentSolver:
    """Presents meta-n's solver contract for a self-contained external agent.

    Owns the lifecycle ordering (budget pre-check -> inject -> lease -> provision
    -> stage -> run -> score -> Trace -> telemetry) and the never-raise invariant
    the evaluation ``gather`` depends on. The agent itself is driven by a
    pluggable :class:`~meta_n.core.external_agents.backend.AgentBackend`; the
    per-benchmark environment/scoring are injected via the adapter factory hooks.

    One solver is built per candidate by
    ``EvolutionaryOrchestrator._build_solver_from_candidate``; the shared
    ``run_guard`` / ``cost_guard`` / ``telemetry`` are passed by reference so a
    single :class:`DockerRunGuard` semaphore and a single cost ledger span the
    whole run (plan §6.7).

    Args:
        backend: The :class:`AgentBackend` strategy that drives the agent.
        env_provider: The :class:`AgentEnvProvider` that provisions/stages/reads
            the per-benchmark environment.
        scorer: The :class:`Scorer` that turns a solution into an ``EvalResult``.
        injected_codes: The candidate chain's ``InjectedCode`` blocks (shallowest
            -> deepest). ``[]`` is the gen0 vanilla-agent baseline.
        depth: The candidate's solver depth (1 for gen0).
        run_guard: The shared :class:`DockerRunGuard` (inner ``--max-docker``
            semaphore + isolation + cleanup).
        telemetry: The shared :class:`AgentTelemetry` writer.
        cost_guard: The shared :class:`CostGuard` (pre-check + ledger record), or
            ``None`` to skip budget enforcement.
        adapter: The benchmark adapter that produced ``backend``/``env_provider``/
            ``scorer`` (kept for parity with the orchestrator's executor, plan
            §2.6); not otherwise used by the spine.
        solver_language: ``"python"`` (default) or ``"bash"`` — routes the
            injection mapper's library-description formatter.
        max_turns: Per-run agent turn/episode ceiling forwarded to the backend.
        token_budget: Per-run inner token budget forwarded to the backend.
        time_limit_s: Soft wall-clock budget forwarded to the backend (which sets
            its OWN hard envelope ``_bridge.hard_timeout(time_limit_s)``). The
            outer ``asyncio.wait_for`` is a last-resort wall set STRICTLY ABOVE
            that backend hard timeout (``hard_timeout(time_limit_s) +
            _OUTER_GRACE_S``) so the backend's hard kill wins the race and reports
            recoverable tokens (H5). ``None`` means no soft limit and no outer
            wall (unbounded; the lease/guard still clean up).
        max_budget_usd: Per-run USD budget forwarded to the backend and used by
            the pre-check.
        run_ctx: Optional mapping carrying the per-run telemetry coordinates
            ``generation`` / ``candidate_id`` (the orchestrator updates these per
            candidate). Surfaced as solver attributes so
            :meth:`AgentTelemetry.start_record` reads them via ``getattr``.
    """

    def __init__(
        self,
        *,
        backend: "AgentBackend",
        env_provider: "AgentEnvProvider",
        scorer: "Scorer",
        injected_codes: "list[InjectedCode]",
        depth: int,
        run_guard: "DockerRunGuard",
        telemetry: "AgentTelemetry",
        cost_guard: "Optional[CostGuard]",
        adapter: object,
        solver_language: str = "python",
        max_turns: int = 16,
        token_budget: int = 1_000_000,
        time_limit_s: "float | None" = None,
        max_budget_usd: float = 2.0,
        run_ctx: "Optional[dict]" = None,
    ) -> None:
        self.backend = backend
        self.env_provider = env_provider
        self.scorer = scorer
        self.injected_codes = injected_codes
        self.depth = int(depth)
        self.run_guard = run_guard
        self.telemetry = telemetry
        self.cost_guard = cost_guard
        self.adapter = adapter
        # Benchmark coordinate for telemetry (adapter.name is a property; a broken
        # test double must not sink solver construction).
        try:
            self.benchmark = str(getattr(adapter, "name", "") or "")
        except Exception:  # noqa: BLE001
            self.benchmark = ""
        self.solver_language = solver_language
        self.max_turns = int(max_turns)
        self.token_budget = int(token_budget)
        self.time_limit_s = time_limit_s
        self.max_budget_usd = float(max_budget_usd)

        # Telemetry coordinates (read off ``self`` by ``start_record`` via
        # getattr with safe fallbacks). The orchestrator updates these per
        # candidate via ``run_ctx`` so each AgentRunRecord lands on the right
        # (generation, candidate_id) coordinate (telemetry §7.3).
        ctx = run_ctx or {}
        self.run_ctx = ctx
        self.generation = int(ctx.get("generation", 0) or 0)
        self.candidate_id = str(ctx.get("candidate_id", "") or "")
        # S0.6 de-reap target root: snapshot the lease ``agent_logs`` into
        # ``<output_dir>/archive/<candidate_id>/agent_logs/<run_id>/`` before the
        # lease scratch dir is rmtree'd. Falls back to the telemetry writer's
        # output_dir (the single source of truth for relpath pointers) when the
        # orchestrator did not thread it through ``run_ctx`` (e.g. a unit test
        # that built the solver directly).
        self.output_dir = ctx.get("output_dir") or getattr(
            telemetry, "output_dir", None
        )

        # Build the injection mapper once: merge the candidate's libraries and
        # bind the language + a sandbox marker (the agent runs helpers in its own
        # sandbox, so the description formatters skip host-side checks, plan §3).
        self._injection = InjectionMapper(
            injected_codes, solver_language, SandboxMarker()
        )

    # -- solver contract -----------------------------------------------------

    async def execute(self, task: "TaskDescription") -> "tuple[Trace, int]":
        """Run ``task`` end-to-end and return ``(Trace, outer_tokens)``.

        The primary solver entry point dispatched by the orchestrator. Builds the
        injection plan with no inter-layer context and drives the full lifecycle.

        The returned outer-token int is ``0`` for OpenHands / Terminus 2 (their
        spend is reported as ``Trace.inner_*``) and the backend's native *outer*
        token count for the builtin backend (``outer_token_mode=True``) — see the
        token contract in the module docstring. ``_evaluate_candidate`` sums the
        ``Trace.inner_*`` separately (§7.1).

        Args:
            task: The task to solve.

        Returns:
            ``(Trace, outer_tokens)``. Never raises except
            :class:`asyncio.CancelledError`; every other failure yields a degraded
            ``(Trace(success=False, score=0), 0)``.
        """
        return await self._execute_with_plan(task, "", parent_run_id=None)

    async def solve(
        self, task: "TaskDescription", additional_context: str = ""
    ) -> "tuple[str, str, int]":
        """Re-solve ``task`` with inter-layer ``additional_context``.

        Matches meta-n's ``SolverProtocol.solve`` shape so the chain-test /
        ``needs_re_solve`` path can drive an external agent the same way it drives
        a native solver. The ``additional_context`` becomes the injected
        ``Prompt.prefix``; the run produces a full :class:`Trace`, from which the
        ``(script, reasoning, tokens)`` triple is unpacked.

        FORWARD-LOOKING SEAM (not exercised by any current benchmark): the
        orchestrator only calls ``solve()`` when ``needs_re_solve`` is true, which
        is ``hasattr(adapter, "get_test_task")`` — a method only the classification
        adapters define, and those have no external backend. So no benchmark today
        both supports an external ``base_solver`` AND triggers a re-solve. The
        ``parent_run_id`` lineage threaded through :meth:`_execute_with_plan` (and
        recorded on ``AgentRunRecord.parent_run_id``) is the matching seam for a
        recorded re-solve child; both ``execute()`` and ``solve()`` currently pass
        ``parent_run_id=None``, so the child-lineage field is wired but unpopulated
        until a re-solving external-backed adapter exists.

        Args:
            task: The task to (re-)solve.
            additional_context: Higher-layer guidance prepended to the agent's
                initial instruction (the ``Prompt.prefix``).

        Returns:
            ``(script, reasoning, tokens)`` — ``trace.script``, ``trace.reasoning``
            and the outer-token int from :meth:`execute`'s contract. Never raises
            except :class:`asyncio.CancelledError`.
        """
        trace, tokens = await self._execute_with_plan(
            task, additional_context, parent_run_id=None
        )
        return trace.script, trace.reasoning, tokens

    # -- lifecycle core ------------------------------------------------------

    async def _execute_with_plan(
        self,
        task: "TaskDescription",
        additional_context: str,
        parent_run_id: Optional[str],
    ) -> "tuple[Trace, int]":
        """Drive the full lifecycle for one task under the never-raise invariant.

        Opens the telemetry record, builds the injection plan (inside the
        guarded lifecycle, so a build fault degrades to a recorded row), runs
        the soft per-run budget pre-check, then leases an isolated environment
        and runs the agent inside a hard ``asyncio.wait_for`` envelope. The
        whole lease body is wrapped in ``try/except BaseException`` so a
        ``BudgetExceededError`` (a ``BaseException``), an env/agent fault, or a
        ``provision`` failure becomes a degraded ``(Trace, 0)`` rather than
        escaping :meth:`execute` (plan §6.5). Only
        :class:`asyncio.CancelledError` propagates.

        Args:
            task: The task being solved.
            additional_context: Inter-layer context folded into the injection
                plan's ``Prompt.prefix`` (``""`` for the primary ``execute()``
                path).
            parent_run_id: The originating run_id when this is a re-solve child,
                else ``None``.

        Returns:
            ``(Trace, outer_tokens)`` per the :meth:`execute` token contract.
        """
        rec = self.telemetry.start_record(task, self, parent_run_id)
        # The plan build runs INSIDE the guarded lifecycle, AFTER start_record:
        # a pathological Ω-generated helper source can crash the build (e.g. an
        # ``ast.parse`` MemoryError escapes every ``except SyntaxError``), and
        # an escape here would violate the module invariant — no telemetry row,
        # no diagnostics. A build fault is a degraded, RECORDED failure; we
        # deliberately do NOT fall back to an empty plan, which would silently
        # score a vanilla agent as the candidate.
        try:
            plan = self._injection.build(task, additional_context)
        except BaseException as exc:  # noqa: BLE001 - never-raise covers the build
            if isinstance(exc, asyncio.CancelledError):
                raise
            logger.warning(
                "[%s] injection build failed task=%s: %s",
                self.backend.name,
                getattr(task, "task_id", "?"),
                exc,
            )
            return self.telemetry.finish_error(task, rec, exc, self.depth), 0
        # Stash the available utilities now so attribution can run at finish even
        # if the backend exposes a command stream but the plan is otherwise empty.
        rec.utilities_available = list(plan.utilities_available)
        rec.pre_process_ran = bool(plan.pre_process_ran)

        # Per-run soft budget pre-check (plan §4.7.3-4). CostGuard.precheck()
        # NEVER raises BudgetExceededError — it returns a decision so we honor the
        # never-raise invariant rather than canceling the gather. A True decision
        # short-circuits to a degraded BUDGET_DENIED record without running.
        if self.cost_guard is not None and self.cost_guard.precheck(self.max_budget_usd):
            rec.terminated_by = TerminatedBy.BUDGET_DENIED.value
            return self.telemetry.finish_budget_denied(task, rec, self.depth), 0

        try:
            async with self.run_guard.lease(
                task, time_limit_s=self.time_limit_s
            ) as lease:
                # H5: the outer envelope must STRICTLY EXCEED the backend's own
                # hard timeout (``_bridge.hard_timeout(time_limit_s)`` ==
                # ``max(soft*1.25, soft+120)``, used identically inside the _tb /
                # openhands backends) plus provision + scoring + lease teardown
                # slack. If the outer ``wait_for`` fired at the SOFT limit it
                # would cancel a backend that is winding down — recording a
                # spurious TIMEOUT with 0 tokens / 0 cost. With the enlarged
                # envelope the backend's hard kill wins, ``_run_leased`` returns a
                # degraded-but-token-bearing result, and the normal
                # ``finish_record`` path records TIMEOUT-vs-COMPLETED + spend
                # correctly. ``time_limit_s is None`` ⇒ envelope ``None`` ⇒ no
                # outer wall (byte-identical to the prior unbounded path).
                envelope = (
                    None
                    if self.time_limit_s is None
                    else hard_timeout(self.time_limit_s) + _OUTER_GRACE_S
                )
                try:
                    return await asyncio.wait_for(
                        self._run_leased(task, plan, lease, rec),
                        timeout=envelope,
                    )
                except asyncio.TimeoutError:
                    # Outer last-resort wall expired (the backend hard kill should
                    # have fired first). The lease ``finally`` (and the provider's
                    # hard_kill) tear down the wedged work; we record a degraded
                    # TIMEOUT row here (plan §2.3, §6.5a). _finish_degraded only
                    # zeroes score/success/attribution — any recoverable
                    # agent_tokens / cost a backend stamped on ``rec`` is
                    # PRESERVED into the written row rather than reported as a
                    # spurious 0-token timeout (H5 fold).
                    logger.warning(
                        "[%s] run timed out after %ss (envelope=%ss) task=%s",
                        self.backend.name,
                        self.time_limit_s,
                        envelope,
                        getattr(task, "task_id", "?"),
                    )
                    # Snapshot the lease ``agent_logs`` while the lease (and its
                    # scratch dir) still exists, so the degraded TIMEOUT row
                    # carries a resolvable ``transcript_ptr`` instead of null —
                    # the backends stage instruction/suffix (and any partial
                    # runner output) there BEFORE spawning. Best-effort /
                    # never-raises; an empty dir keeps an honest ``None``.
                    self._dereap_logs(rec, lease)
                    rec.terminated_by = TerminatedBy.TIMEOUT.value
                    return self.telemetry.finish_timeout(task, rec, self.depth), 0
                except BaseException as exc:
                    # In-lease fault: snapshot the diagnostics NOW — the outer
                    # handler below runs AFTER the ``async with`` rmtree'd the
                    # lease scratch dir. CancelledError is deliberately NOT
                    # snapshotted (cooperative teardown stays prompt, and no
                    # row is written on cancellation). The outer handler still
                    # owns the degraded row.
                    if not isinstance(exc, asyncio.CancelledError):
                        self._dereap_logs(rec, lease)
                    raise
        except BaseException as exc:  # noqa: BLE001 - never-raise invariant (§6.5)
            # Cooperative cancellation must propagate so the orchestrator can
            # tear the run down; everything else (incl. BudgetExceededError, a
            # BaseException, and any provision/backend/scorer fault that slipped
            # the inner guards) becomes a degraded (Trace, 0).
            if isinstance(exc, asyncio.CancelledError):
                raise
            logger.warning(
                "[%s] run failed task=%s: %s",
                self.backend.name,
                getattr(task, "task_id", "?"),
                exc,
            )
            return self.telemetry.finish_error(task, rec, exc, self.depth), 0

    async def _run_leased(
        self,
        task: "TaskDescription",
        plan: "InjectionPlan",
        lease: "EnvLease",
        rec: "AgentRunRecord",
    ) -> "tuple[Trace, int]":
        """Run the agent inside a held lease: provision -> stage -> run -> score.

        Provisions the per-benchmark environment (an async context manager that
        guarantees its own teardown), stages the injection plan's helper files,
        builds the immutable :class:`AgentRunContext`, drives the backend (which
        never raises), collects post-hoc metrics, extracts the solution, and
        scores it. The resulting :class:`Trace` is built, the real spend is fed
        into the single cost ledger, and the telemetry row is written.

        Timing is split into provision / agent / score so the record can carry
        ``provision_s`` / ``agent_s`` / ``score_s`` (set on ``rec`` *before*
        :meth:`AgentTelemetry.finish_record`, which otherwise defaults ``agent_s``
        to the whole wall).

        Args:
            task: The task being solved.
            plan: The :class:`InjectionPlan` to inject + stage.
            lease: The held :class:`EnvLease` (scratch dir, session, cleanup hooks).
            rec: The open :class:`AgentRunRecord` for this run.

        Returns:
            ``(Trace, outer_tokens)`` — the outer-token int per the backend's
            ``outer_token_mode`` (0 for OH/T2; native outer tokens for builtin).
        """
        # Per-task daily-cap re-check (plan §4.7.4). The admission precheck ran
        # before we blocked on the inner Docker semaphore; while we waited, other
        # in-flight runs may have recorded spend and exhausted the day. Re-read
        # the ledger fresh HERE — immediately before standing up the env / driving
        # the backend — and abort without provisioning if the headroom is gone, so
        # a day that filled up during the wait does not launch yet another
        # (uncapped) agent run. Concurrent in-flight runs can still overshoot by
        # at most ``parallel`` runs; there is no in-run USD kill (see budget.py).
        if (
            self.cost_guard is not None
            and self.cost_guard.headroom_exhausted()
        ):
            logger.warning(
                "[%s] daily headroom exhausted before backend.run task=%s; "
                "aborting run without provisioning (budget backstop).",
                self.backend.name,
                getattr(task, "task_id", "?"),
            )
            rec.terminated_by = TerminatedBy.BUDGET_DENIED.value
            return self.telemetry.finish_budget_denied(task, rec, self.depth), 0

        t_lease = time.monotonic()
        async with self.env_provider.provision(task, lease) as env:
            rec.provision_s = max(0.0, time.monotonic() - t_lease)

            await self.env_provider.stage_files(env, plan.staged_files)

            ctx = AgentRunContext(
                instruction=getattr(task, "description", "") or "",
                prompt=plan.prompt,
                # The env yields an opaque handle; the backend receives the
                # provider's ``workspace_handle`` (plan §2.3 L98).
                workspace=getattr(env, "workspace_handle", env),
                time_limit_s=self.time_limit_s,
                max_turns=self.max_turns,
                token_budget=self.token_budget,
                max_budget_usd=self.max_budget_usd,
                logging_dir=lease.workdir / "agent_logs",
            )

            # backend.run / collect_metrics never raise (plan §2.4); errors are
            # encoded in run.terminated_by / run.failure_mode.
            t_agent = time.monotonic()
            run = await self.backend.run(ctx, self.telemetry, rec)
            run = await self.backend.collect_metrics(env, run)
            rec.agent_s = max(0.0, time.monotonic() - t_agent)

            # Feed real OH/T2 spend into the SAME ledger the outer client uses so
            # the daily / project cap sees agent spend (plan §4.7, §7.2) AS SOON AS
            # it is known — immediately after ``collect_metrics`` finalizes the
            # token/cost counts and BEFORE the post-run steps (extract_solution /
            # scorer.score) that are NOT contractually never-raise. If one of those
            # raises, the run already spent real money (the agent ran), but the
            # failure escapes to ``_execute_with_plan``'s ``BaseException`` handler
            # as a degraded ``finish_error`` that records NOTHING to the ledger —
            # so recording here is what keeps that already-incurred spend from
            # being silently dropped past the daily cap. ``cost_guard.record`` never
            # raises out.
            #
            # The builtin (outer_token_mode) backend is the exception: its spend
            # ALREADY flows through the outer LLMClient, which appends its own ledger
            # line inside complete() (CostTracker.record). Calling cost_guard.record
            # here would price the SAME token counts a SECOND time (the
            # priced_from_tokens branch re-derives compute_cost_usd), double-billing
            # every builtin call against today_total_usd() and tripping the daily/
            # project cap at half the real budget. The budget.py docstring already
            # states builtin spend "does not pass through this guard at all" — so we
            # honor that contract here rather than folding it twice (plan §4.7).
            #
            # Degraded paths that never reach (or reach with an empty result) this
            # fold under-count the ledger — see budget.py "Ledger undercount bound".
            if self.cost_guard is not None and not self.backend.outer_token_mode:
                self.cost_guard.record(run, task, self)

            solution = await self.env_provider.extract_solution(env, run)
            # Stamp the authoritative solution onto the run so build_trace's
            # ``_solution_text`` prefers it over the raw transcript (telemetry §5).
            try:
                run.solution = solution  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001 - run is a dataclass; this should hold
                pass

            t_score = time.monotonic()
            evalr = await self.scorer.score(task, env, solution, run)
            rec.score_s = max(0.0, time.monotonic() - t_score)

        # Build the Trace outside the env context (the env is torn down on exit).
        trace = self.telemetry.build_trace(
            task, run, evalr, self.depth, self.backend.outer_token_mode
        )

        # S0.6 de-reap — MUST run BEFORE finish_record. The lease ``agent_logs``
        # dir lives under the scratch root (concurrency.py mkdtemp), OUTSIDE
        # output_dir, and is rmtree'd the moment this lease releases — leaving
        # ``transcript_ptr`` dangling (the FEAL null-transcript finding). Snapshot
        # it into the archive NOW, while the lease is still held, and repoint
        # ``rec.transcript_ptr``/``agent_logs_ptr`` at the durable archive copy.
        # Doing this BEFORE finish_record is what makes the resolvable pointer
        # actually reach ``agent_runs.jsonl``: finish_record serializes the row
        # exactly once and now honors the already-set pointer instead of clobbering
        # it back to the (None) scratch-relative path. Best-effort / never-raises
        # (stdlib only); a no-logs backend leaves the pointers honestly None.
        self._dereap_logs(rec, lease)

        # Write the AgentRunRecord (flock-append, de-duped by run_id). Never raises.
        # (OH/T2 spend was already folded into the cost ledger right after
        # collect_metrics, above, so a post-run scoring fault cannot drop it.)
        self.telemetry.finish_record(rec, run, evalr, lease)

        # Outer-return tokens: 0 for OH/T2 (spend is INNER, §7.1); for the builtin
        # backend (outer_token_mode=True) return its native outer token count so
        # cross-baseline summary.json stays apples-to-apples with use_agentic.
        outer_tokens = int(run.agent_tokens or 0) if self.backend.outer_token_mode else 0
        return trace, outer_tokens

    def _dereap_logs(self, rec: "AgentRunRecord", lease: "EnvLease") -> None:
        self.telemetry.dereap_agent_logs(
            rec,
            lease.workdir / "agent_logs",
            self.candidate_id,
            rec.run_id,
            output_dir=self.output_dir,
        )
