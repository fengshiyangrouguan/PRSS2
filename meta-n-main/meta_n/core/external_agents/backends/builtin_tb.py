"""TB-mode builtin backend — the REAL same-container control for OH vs T2.

This is the terminal-bench analogue of :class:`BuiltinBackend` (the CO-Bench /
host-workspace native control) and a sibling of
:class:`~meta_n.core.external_agents.backends.terminus2.Terminus2Backend` /
:class:`~meta_n.core.external_agents.backends.openhands_tb.OpenHandsTBBackend`. It
gives meta-n's own :class:`~meta_n.core.solver.Layer1Solver` a same-container,
same-verifier control on terminal-bench so ``builtin`` / ``openhands`` /
``terminus2`` all run the IDENTICAL tasks scored by the IDENTICAL terminal-bench
verifier.

Why a subprocess bridge (not the legacy native path)
-----------------------------------------------------
The native ``TerminalBenchExecutor`` CANNOT run the legacy ``original-tasks``
layout (it needs the harbor-derived ``environment/`` + ``task.toml`` + ``test.sh``
shape these tasks lack), so a real same-container builtin control must run through
the SAME terminal-bench :class:`Harness` the OH / T2 runners use. ``terminal_bench``
and meta-n have conflicting pins, so meta-n's process must **never** import it.
This backend therefore drives the run via a *subprocess bridge* to
``scripts/builtin_tb_runner.py`` under the external-agents interpreter
(``.venv_external_agents/bin/python``), reusing the ENTIRE
:class:`_ExternalTBBackend` machinery (request/result JSON contract, hard-timeout
SIGKILL-group teardown, label-scoped ``docker rm``, never-raise) verbatim.

The two-sided split (the design's whole point)
----------------------------------------------
Unlike OH / T2 — which author *and* execute their script INSIDE the venv runner
with the agent's own (inner) LLM — the builtin control authors its bash script on
the **meta-n side**:

1. :meth:`_prepare_run` calls the native :class:`Layer1Solver` (ONE outer LLM
   call through meta-n's outer :class:`~meta_n.core.llm_client.LLMClient`) to turn
   the task instruction (+ any inter-layer injection context) into a bash script.
   No LLM ever runs in the venv runner — it only *executes* the pre-authored
   script inside the harness-provisioned TB container and scores it.
2. The authored script + the injection's staged helper files ride into the request
   JSON; the runner writes the script into the container and runs it, then runs
   terminal-bench's verifier and emits the SHARED t2-style result JSON
   (``is_resolved`` is the binary reward; its token fields are 0 — there is no
   inner LLM).

Token accounting (the outer-mode exception)
--------------------------------------------
Because the only LLM spend is the meta-n-side authoring call — which already flows
through the outer ``LLMClient`` and is recorded once on the shared cost ledger —
this backend is the terminal-bench twin of :class:`BuiltinBackend`'s
``outer_token_mode = True`` exception: it reports the outer authoring tokens in the
``agent_*`` fields, the spine returns them as the native *outer* int (NOT remapped
to ``Trace.inner_*``), and the spine SKIPS ``cost_guard.record`` (the outer client
already billed them — folding again would double-count). The telemetry row is
stamped ``token_basis="outer"`` / ``cost_basis="priced_from_tokens"`` exactly like
the CO-Bench builtin control (plan §2.3 FIX, §7.11). ``native_score`` /
``native_resolved`` come from the runner's ``is_resolved``, so the shared
``TBExternalScorer`` works unchanged.

Import discipline
-----------------
Like every file under ``external_agents/``, this module imports cleanly without
``terminal_bench`` / ``docker`` installed: it touches only the stdlib, the
zero-/wave-1 siblings (``_external_tb`` / ``backend`` / ``terminated``), and the
in-tree native solver (which itself imports no external SDK). The whole
terminal-bench drive happens in the venv child; meta-n never imports it.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .._outer_token_attribution import snapshot_usage, usage_delta
from ..backend import AgentRunContext, AgentRunResult, task_from_workspace
from ..terminated import from_t2_failure
from ._external_tb import _ExternalTBBackend, _RunPrep

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..terminated import TerminatedBy

logger = logging.getLogger(__name__)

__all__ = ["BuiltinTBBackend"]


class BuiltinTBBackend(_ExternalTBBackend):
    """Drive meta-n's native ``Layer1Solver`` on terminal-bench (same-container).

    Selected per candidate by the orchestrator (``--base-solver builtin`` on the
    terminal_bench benchmark, which routes builtin through the spine ONLY for this
    adapter) and handed to
    :class:`~meta_n.core.external_agents.solver.ExternalAgentSolver`. The script is
    authored on the meta-n side (one outer LLM call); the harness-provisioned TB
    container execution + verifier scoring happen in the venv child. :meth:`run`
    (inherited from :class:`_ExternalTBBackend`) honors the never-raise contract.

    Attributes:
        name: ``"builtin"`` — the telemetry ``agent`` coordinate (shared with the
            CO-Bench builtin control; the benchmark disambiguates).
        outer_token_mode: ``True`` — the authoring call's outer tokens are returned
            as the native outer int and NOT re-billed by the spine's cost guard
            (the outer ``LLMClient`` already recorded them), exactly like
            :class:`BuiltinBackend` (plan §2.3 FIX, §7.11).
    """

    name: str = "builtin"
    #: Native outer-token accounting (the authoring call rides on the outer ledger).
    outer_token_mode: bool = True

    _RUNNER_MODULE = "builtin_tb_runner"
    _REQUEST_FILENAME = "builtin_tb_request.json"
    _RESULT_FILENAME = "builtin_tb_result.json"
    _RUN_LABEL_PREFIX = "builtin-tb"
    _TRIAL_COMPOSE_PROJECT = "builtin-tb-bridge"
    _LOG_NAME = "BuiltinTBBackend"

    def __init__(
        self,
        *,
        solver: object,
        llm_client: object | None = None,
        solver_language: str = "bash",
        **kwargs,
    ) -> None:
        """Configure the bridge plus the meta-n-side authoring collaborators.

        Args:
            solver: The native :class:`~meta_n.core.solver.Layer1Solver` used to
                author the task's bash script on the meta-n side (its ``solve()``
                emits the script through the outer ``LLMClient``; nothing is
                executed locally — the venv runner runs it in the container).
            llm_client: The outer LLM client whose ``cumulative_usage`` ledger the
                authoring call flows through. Snapshotted across :meth:`_prepare_run`
                to recover the prompt/completion/cached/calls split for telemetry.
                ``None`` falls back to the bare token total ``solve()`` returns.
            solver_language: Carried for parity with the orchestrator's
                solver-build kwargs; the authoring ``Layer1Solver`` already pins its
                own language (terminal-bench is ``"bash"``), and the synthesized task
                stamps ``solution_language="bash"``, so this is metadata only.
            **kwargs: The shared :class:`_ExternalTBBackend` construction kwargs
                (``model`` / ``api_base`` / ``api_key`` / ``venv_python`` /
                ``runner_dir`` / ``tasks_dir`` / ``provider_env_var`` / …). NOTE
                ``model`` / ``api_base`` here describe the *inner* model the runner
                would use — but the builtin runner runs NO LLM, so they are inert
                routing metadata only; the real authoring model is the outer
                ``llm_client``'s.
        """
        super().__init__(**kwargs)
        self._solver = solver
        self._llm_client = llm_client
        self._solver_language = solver_language

    # ------------------------------------------------------- subclass hooks --
    async def _prepare_run(self, ctx: AgentRunContext) -> _RunPrep:
        """Author the bash script (one outer LLM call) before the subprocess.

        Recovers the task (a full ``TaskDescription`` off the workspace handle when
        present, else a synthetic one built from ``ctx.instruction`` +
        ``ctx.workspace.task_id`` — the shared TB env provider carries only the
        task id), runs the native :class:`Layer1Solver` once with the composed
        injection context as ``additional_context``, snapshots the outer
        ``LLMClient.cumulative_usage`` delta across the call to recover the token
        split, and returns a :class:`_RunPrep` that (a) ships the authored script
        in the request (``bash_script``) and (b) stamps the outer authoring tokens
        onto the parsed runner result.

        Never raises (honoring the :meth:`run` contract): an authoring failure
        yields a prep with an empty script (the runner runs nothing → unresolved),
        carrying any partial outer spend so the run is not mispriced at $0.
        """
        # Shared task-recovery helper (backend.task_from_workspace). The shared
        # TB env provider's handle carries only task_id (not the full task), so
        # the common case is None and a task is synthesized below; a nested
        # candidate must expose description AND task_id (the builtin-parity
        # ``_Env.workspace_handle = task`` shape) so the authored script reads
        # the real ``task.metadata`` / ``task.description``.
        task = task_from_workspace(
            ctx.workspace, nested_requires=("description", "task_id")
        )
        if task is None:
            task = self._synthesize_task(ctx)

        # Composed inter-layer injection (system_suffix + prefix) → the solver's
        # additional_context. Empty at depth 1 → a byte-identical vanilla authoring.
        additional_context = self._compose_instruction_context(ctx)

        before = self._usage_snapshot()
        script = ""
        tokens = 0
        try:
            script, _reasoning, tokens = await self._solver.solve(
                task, additional_context=additional_context
            )
        except Exception:  # noqa: BLE001 - _prepare_run must not raise out of run()
            logger.exception(
                "%s: Layer1Solver authoring failed for task=%s",
                self._LOG_NAME, getattr(task, "task_id", "?"),
            )

        # Recover the prompt/completion/cached/calls split from the outer ledger
        # delta (the solver returns only a total). Best-effort under --parallel
        # (the shared ledger can absorb a sibling's concurrent spend); the
        # per-run authoritative TOTAL is the solver's returned int.
        p_tok, c_tok, total_tok, cached_tok, calls = self._usage_delta(before)
        outer_total = int(tokens or 0) or int(total_tok or 0)

        def _apply(result: AgentRunResult) -> AgentRunResult:
            """Stamp the outer authoring-token attribution onto the runner result.

            The runner ran NO LLM (its token fields are 0), so the ``agent_*``
            spend is entirely the meta-n-side authoring call. We set the totals
            here; the spine (seeing ``outer_token_mode=True``) returns them as the
            native outer int and skips ``cost_guard.record`` (the outer client
            already billed them — plan §2.3 FIX).
            """
            result.agent_tokens = outer_total
            result.agent_prompt_tokens = int(p_tok or 0)
            result.agent_completion_tokens = int(c_tok or 0)
            result.agent_cached_tokens = int(cached_tok or 0)
            result.agent_calls = int(calls or 0)
            result.cost_usd = 0.0
            result.cost_basis = "priced_from_tokens"
            return result

        return _RunPrep(request_fields={"bash_script": script}, apply_to_result=_apply)

    def _extra_request_fields(self, ctx: AgentRunContext, max_episodes: int) -> dict:
        """No STATIC request extras — the builtin request is shaped by the prep.

        Unlike Terminus 2 (``parser_name`` + ``additional_context``) / OpenHands
        (``system_message_suffix`` + ``max_iterations``), the builtin control's only
        request extra is the PRE-AUTHORED ``bash_script``, which is computed per-run
        in :meth:`_prepare_run` and merged via ``_RunPrep.request_fields`` (it can't
        be static — it depends on the task + the authoring LLM call). The injection
        context is already folded into the script's ``additional_context`` at
        authoring time, so there is nothing to add here. Returns ``{}`` (overriding
        the abstract base so the inherited ``_run_bridge`` does not ``raise``).
        """
        del ctx, max_episodes
        return {}

    def _resolve_terminated(self, failure_mode: object) -> "TerminatedBy":
        """Resolve a native runner failure tag to :class:`TerminatedBy`.

        The builtin runner emits the same vocabulary as the t2 runner
        (``token_budget`` is impossible here — no inner LLM — but ``env_error`` /
        ``agent_timeout`` / clean tags occur), so deferring to the canonical
        :func:`~meta_n.core.external_agents.terminated.from_t2_failure` covers
        every tag (including the runner/bridge-emitted ``env_error``, now in the
        shared table) and falls back to :attr:`TerminatedBy.UNKNOWN` for an
        unmapped one.
        """
        return from_t2_failure(failure_mode)

    # ---------------------------------------------------------- internals ----
    def _synthesize_task(self, ctx: AgentRunContext) -> object:
        """Build a native ``TaskDescription`` from the run context.

        The shared TB env provider's handle carries only ``task_id`` (not the full
        task), so the authoring solver is fed a task synthesized from
        ``ctx.instruction`` (the task's natural-language description) + the handle's
        ``task_id``, with ``solution_language="bash"`` so ``Layer1Solver.solve``
        picks the bash prompt. Imported at call scope to keep this module free of
        any heavy import at load (parity with the package's import discipline).
        """
        from meta_n.core.meta_layer import TaskDescription

        task_id = str(getattr(ctx.workspace, "task_id", "") or "") or "task"
        return TaskDescription(
            task_id=task_id,
            description=str(getattr(ctx, "instruction", "") or ""),
            metadata={"benchmark": "terminal_bench", "solution_language": "bash"},
        )

    def _usage_snapshot(self) -> dict[str, int | float]:
        """Snapshot the outer ``LLMClient.cumulative_usage`` ledger, or zeros.

        Thin wrapper over the shared
        :func:`~meta_n.core.external_agents._outer_token_attribution.snapshot_usage`
        (single-sourced with :class:`BuiltinBackend`).
        """
        return snapshot_usage(self._llm_client)

    def _usage_delta(
        self, before: dict[str, int | float]
    ) -> tuple[int, int, int, int, int]:
        """Compute the prompt/completion/total/cached/calls delta since ``before``.

        Delegates to the shared
        :func:`~meta_n.core.external_agents._outer_token_attribution.usage_delta`.
        The native solver returns only a total; the outer client accumulates the
        full split on every ``complete()``. Each component is clamped at ``0`` so a
        racy negative delta (shared ledger under ``--parallel``) never leaks
        (parity with :class:`BuiltinBackend._usage_delta`).
        """
        return usage_delta(before, self._usage_snapshot())
