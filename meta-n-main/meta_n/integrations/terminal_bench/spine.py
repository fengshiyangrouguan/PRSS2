"""External-agent (Terminus 2 / OpenHands / builtin-TB) env provider + scorer.

Split verbatim out of the old single-file
``meta_n/integrations/terminal_bench.py`` module (this was the tail
section behind the mid-file ``# noqa: E402`` import block, which now
lives at a normal module top).
"""

from __future__ import annotations

import contextlib
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

from meta_n.core.external_agents.env import (
    AgentEnvProvider,
    EnvLease,
    Scorer,
    mirror_agent_tokens,
)
from meta_n.core.external_agents.terminated import TerminatedBy
from meta_n.core.meta_layer import TaskDescription
from meta_n.integrations.benchmark import EvalResult

if TYPE_CHECKING:  # pragma: no cover — annotation-only, avoids a cycle
    from meta_n.integrations.terminal_bench.adapter import (
        TerminalBenchAdapter,  # noqa: F401
    )

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# External-agent (Terminus 2) env provider + scorer — SUBPROCESS-BRIDGE SHIMS.
#
# The Terminus 2 backend shells out to ``scripts/t2_runner.py`` (under the
# external-agents interpreter), which drives terminal-bench's OWN single-task
# trial machinery end-to-end — container build, tmux session, the verifier, and
# the binary ``is_resolved`` reward. So the meta-n-side env provider and scorer
# do NO container work and run NO verifier: the provider only conveys the
# ``task_id`` + ``staged_files`` the backend folds into the request JSON, and the
# scorer reads the already-computed binary reward back off the ``AgentRunResult``.
#
# Both import only stdlib + the zero-/wave-1 ``external_agents`` siblings, so this
# module stays importable with ``terminal_bench`` absent from meta-n's env.
# ---------------------------------------------------------------------------


class _TBTerminus2Env:
    """Opaque per-run handle for the Terminus 2 subprocess-bridge path.

    Carries exactly what the backend needs to compose its request JSON — the
    runner's task id and the staged-helper map — plus the host scratch dir. It is
    *not* a live container: the runner provisions and tears down the real
    terminal-bench Docker stack itself.

    Attributes:
        task_id: The terminal-bench task folder name the runner's ``TrialHandler``
            reads (``<tasks_dir>/<task_id>/task.yaml``). Prefers the unsanitized
            ``task_name`` so it matches the on-disk folder; falls back to the
            sanitized ``TaskDescription.task_id``.
        staged_files: Map of workspace-relative path → contents to stage into the
            container (the injection plan's helpers); empty for the gen0 baseline.
        workspace_root: The lease's host scratch dir (carried for parity; the
            runner writes its own artifacts under the backend's logging dir).
        run_label: The lease's per-run-unique session name
            (``ext-{task_id}-{uuid4()[:8]}``), used by the backend as the trial's
            compose project so concurrent T2 trials never collide on the global
            ``container_name`` / project and a hard-timeout sweep only removes THIS
            run's container (never a sibling's). Sanitized to a Docker-safe label.
        workspace_handle: The object handed to the backend as
            ``AgentRunContext.workspace`` — itself, so ``ctx.workspace.task_id`` /
            ``ctx.workspace.staged_files`` / ``ctx.workspace.run_label`` resolve.
        agent_pid: Pid of the runner subprocess once :meth:`Terminus2Backend.run`
            has spawned it (``None`` before/after). Stamped onto the handle so the
            lease-level ``hard_kill`` / ``shutdown_sweep`` can SIGKILL a wedged
            runner whose ``run()`` frame is no longer on the stack (parity with
            OpenHands' ``_COBenchEnv.agent_pid`` seam). Cleared in ``run()``'s
            ``finally`` so a later hard_kill cannot reap a reused pid.
    """

    def __init__(
        self,
        task_id: str,
        workspace_root: Path,
        run_label: str = "",
        task: "TaskDescription | None" = None,
    ):
        self.task_id = task_id
        self.staged_files: dict[str, str] = {}
        self.agent_pid: int | None = None
        self.workspace_root = workspace_root
        self.run_label = run_label
        # The full ``TaskDescription`` (additive — the Terminus 2 / OpenHands
        # backends ignore it; the BUILTIN-TB backend reads it off the handle to
        # author its one-shot bash script via the native ``Layer1Solver`` before
        # bridging to the runner). ``None`` only when provisioned without a task
        # (defensive; the spine always passes one).
        self.task = task
        # The backend reads task_id / staged_files / run_label (and, for builtin,
        # ``task``) off the handle.
        self.workspace_handle = self


class TBTerminus2EnvProvider(AgentEnvProvider):
    """Thin env provider for the Terminus 2 subprocess bridge (the runner owns env).

    Provisions only a host scratch dir and a handle carrying the runner task id +
    staged files; the real Docker environment, verifier and reward are owned by
    ``scripts/t2_runner.py`` inside the external-agents venv. No ``lease.teardown``
    / ``lease.hard_kill`` is installed here — the backend SIGKILLs the runner's
    process group and force-removes the orphan trial container on a hard timeout,
    and the runner's ``spin_up_terminal`` ``finally`` is the primary teardown.
    """

    def __init__(self, adapter: "TerminalBenchAdapter"):
        self._adapter = adapter

    def _runner_task_id(self, task: TaskDescription) -> str:
        """Return the on-disk task folder name the runner should load.

        Prefers the unsanitized ``task_name`` from metadata (the folder name under
        ``original-tasks``); falls back to the sanitized ``task_id``.
        """
        meta = getattr(task, "metadata", None) or {}
        return str(meta.get("task_name") or task.task_id)

    @contextlib.asynccontextmanager
    async def provision(self, task: TaskDescription, lease: EnvLease):  # type: ignore[override]
        """Provision the no-op host workspace and yield the bridge handle.

        Args:
            task: The terminal-bench task being solved.
            lease: The held :class:`EnvLease` (scratch dir + cleanup-hook slots).

        Yields:
            A :class:`_TBTerminus2Env` whose ``workspace_handle`` (itself) the
            backend reads ``task_id`` / ``staged_files`` off.
        """
        workspace_root = lease.workdir / "workspace"
        try:
            workspace_root.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        # The lease session (``ext-{task_id}-{uuid4()[:8]}``, already Docker-safe)
        # is the per-run-unique label the backend uses as the trial's compose
        # project, so concurrent T2 trials never collide on the global container
        # name / project (the prior hardcoded ``t2-bridge`` collided under
        # ``--parallel`` and let one run's timeout sweep kill a sibling's container).
        run_label = str(getattr(lease, "session", "") or "")
        env = _TBTerminus2Env(
            self._runner_task_id(task), workspace_root, run_label=run_label,
            task=task,
        )

        # Install the lease-level cleanup hooks so DockerRunGuard.shutdown_sweep
        # (and the lease ``finally``) can reap a wedged runner/container even when
        # a CancelledError is delivered while the backend is parked in a non-run()
        # await (collect_metrics / extract_solution / score), where run()'s own
        # except/finally teardown no longer fires. Without these the sweep skipped
        # this lease (both hooks None) and the runner subprocess + its trial
        # container could leak.
        def _hard_kill() -> None:
            """Synchronously SIGKILL the runner's process group, if still alive.

            Reads the pid Terminus2Backend.run() stamped onto the env handle
            (cleared in run()'s finally). ``start_new_session=True`` makes
            pgid == pid, so one killpg reaps the runner and any child. Never
            raises; a no-op when no pid is recorded or the process is gone.
            """
            import signal

            pid = getattr(env, "agent_pid", None)
            if pid is None:
                return
            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                pass
            except Exception:  # noqa: BLE001 - hard kill must never raise out
                logger.debug(
                    "TBTerminus2 hard_kill of pid %s failed", pid, exc_info=True
                )

        async def _teardown() -> None:
            """Force-remove THIS run's orphan trial container/volume/network.

            Delegates to the backend's label-scoped
            ``Terminus2Backend._force_compose_down(run_label)`` so a wedged runner
            that skipped its own ``spin_up_terminal`` teardown does not leak its
            Docker stack. Best-effort, never raises (the backend swallows errors).
            Lazy import keeps terminal_bench import-light and avoids a cycle.
            """
            from meta_n.core.external_agents.backends.terminus2 import (
                Terminus2Backend,
            )

            with contextlib.suppress(Exception):
                await Terminus2Backend._force_compose_down(run_label)

        lease.hard_kill = _hard_kill
        lease.teardown = _teardown

        try:
            yield env
        finally:
            # The DockerRunGuard owns ``rmtree`` of ``lease.workdir``; the runner
            # owns its own Docker teardown on a clean exit. The lease hooks above
            # are the backstop for the cancelled / wedged path.
            pass

    async def stage_files(self, env: object, files: dict[str, str]) -> None:
        """Record the helper files for the backend to forward into the container.

        The actual copy into the live container happens inside the runner
        (``InjectedTerminus2._stage_helper_files`` → ``copy_to_container``); here
        we only stash the map on the env handle so the backend can include it in
        the request JSON. An empty map (gen0 baseline) is a valid no-op.

        Args:
            env: The :class:`_TBTerminus2Env` from :meth:`provision`.
            files: Map of workspace-relative path → contents.
        """
        if not files:
            return
        try:
            env.staged_files = {str(k): str(v) for k, v in files.items()}  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001 - staging-record must never raise
            logger.warning("TBTerminus2EnvProvider could not record staged files")

    async def extract_solution(self, env: object, run: "object") -> str:  # type: ignore[override]
        """Return the agent's authored artifact as solution text.

        Terminal-bench scores the *container state* via its own verifier (run
        inside the runner), not a returned diff, so there is no file to read back
        on the host. We surface the post-agent pane tail (carried on the run) as a
        human-readable solution stand-in for the ``Trace.script`` slot.

        Args:
            env: The :class:`_TBTerminus2Env` from :meth:`provision`.
            run: The backend's ``AgentRunResult``.

        Returns:
            The agent's stdout-tail text (``""`` when unavailable).
        """
        return str(getattr(run, "stdout_tail", "") or "")


class TBTerminus2Scorer(Scorer):
    """Scorer that reads the runner's already-computed binary reward off the run.

    The Terminus 2 runner drives terminal-bench's native verifier and folds the
    binary ``is_resolved`` / reward outcome into the ``AgentRunResult``'s
    ``native_resolved`` / ``native_score``. The scorer derives success/score
    DIRECTLY from that verifier signal (``success = native_resolved`` /
    ``native_score > 0``), identical to native TB scoring, and keeps
    ``terminated_by == COMPLETED`` only as a coherence cross-check (NOT the source
    of truth, so a future status-only change cannot fabricate a passing score).
    ``norm_score`` stays deferred to meta-n's evaluator. The result mirrors the
    agent's inner-token spend from ``run`` for cost attribution (plan §5.8).
    """

    def __init__(self, adapter: "TerminalBenchAdapter"):
        self._adapter = adapter

    async def score(
        self,
        task: TaskDescription,
        env: object,
        solution: str,
        run: "object",
    ) -> EvalResult:
        """Score the run from its terminal state; never raises.

        Args:
            task: The terminal-bench task being scored.
            env: The :class:`_TBTerminus2Env` (unused — scoring already ran inside
                the runner's verifier).
            solution: The extracted solution stand-in (unused for scoring).
            run: The backend's ``AgentRunResult`` carrying the binary outcome.

        Returns:
            An :class:`EvalResult` with the binary score + mirrored inner tokens.
        """
        agent_tokens, agent_prompt, agent_completion, agent_calls = (
            mirror_agent_tokens(run)
        )

        terminated_by = getattr(run, "terminated_by", None)
        failure_mode = getattr(run, "failure_mode", None)

        # Derive success/score DIRECTLY from the runner's authoritative verifier
        # signal (the binary reward / is_resolved the verifier produced), NOT
        # transitively from the agent-status ``terminated_by`` enum (plan §5.8).
        # ``terminated_by == COMPLETED`` is kept only as a coherence cross-check so
        # a future status-only change cannot fabricate a passing score.
        native_score = getattr(run, "native_score", None)
        native_resolved = getattr(run, "native_resolved", None)
        if native_resolved is not None:
            resolved = bool(native_resolved)
        elif native_score is not None:
            resolved = float(native_score) > 0.0
        else:  # no verifier signal available — fall back to the status enum
            resolved = terminated_by == TerminatedBy.COMPLETED
        score = (
            float(native_score)
            if native_score is not None
            else (1.0 if resolved else 0.0)
        )

        # Coherence cross-check: surface a warning when the verifier signal and the
        # status enum disagree (they should not under the current coupling).
        status_resolved = terminated_by == TerminatedBy.COMPLETED
        if status_resolved != resolved:
            logger.warning(
                "TBTerminus2Scorer verifier/status mismatch: native_resolved=%s "
                "native_score=%s terminated_by=%s",
                native_resolved, native_score, terminated_by,
            )

        feedback = (
            "resolved (all unit tests passed)"
            if resolved
            else f"unresolved (score={score}, terminated_by={terminated_by}, "
            f"failure_mode={failure_mode})"
        )

        # valid: a verifier result was actually obtained (not an env/parse fault);
        # feasible: TB has no separate feasibility notion (binary pass/fail is the
        # success signal), so it is always True — do NOT mirror success here, which
        # would mislabel a well-formed unresolved run as infeasible (plan §5.8).
        valid = terminated_by not in (
            TerminatedBy.ENV_ERROR,
            TerminatedBy.PARSE_ERROR,
        )

        return EvalResult(
            success=resolved,
            score=score,
            raw_score=score,
            feedback=feedback,
            valid=valid,
            feasible=True,
            inner_tokens=agent_tokens,
            inner_prompt_tokens=agent_prompt,
            inner_completion_tokens=agent_completion,
            inner_calls=agent_calls,
        )


# ---------------------------------------------------------------------------
# Shared external env-provider / scorer aliases.
#
# The Terminus 2 env provider and scorer are backend-AGNOSTIC: the provider only
# carries ``task_id`` / ``staged_files`` / ``run_label`` into the request JSON,
# and the scorer derives success from ``native_resolved`` / ``native_score`` (a
# binary verifier signal both the Terminus 2 AND the OpenHands-on-TB backends
# populate identically from ``is_resolved``). The teardown hook the provider
# installs is also backend-agnostic — it sweeps Docker by THIS run's compose
# project label, and ``OpenHandsTBBackend._force_compose_down`` /
# ``Terminus2Backend._force_compose_down`` share that label-scoped contract.
#
# So both ``--base-solver terminus2`` and ``--base-solver openhands`` on
# terminal_bench use the SAME env provider + scorer. These aliases give them the
# backend-neutral names ``make_env_provider`` / ``make_scorer`` dispatch to,
# while the original ``TBTerminus2*`` names stay as-is for back-compat (existing
# tests + the integration spike import them directly).
# ---------------------------------------------------------------------------
TBExternalEnvProvider = TBTerminus2EnvProvider
TBExternalScorer = TBTerminus2Scorer
