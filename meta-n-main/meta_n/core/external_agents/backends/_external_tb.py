"""Shared base for the terminal-bench external-agent subprocess backends.

The Terminus 2 and OpenHands-on-terminal-bench backends drive a *different* agent
SDK but share the ENTIRE subprocess-bridge control flow: write a request JSON,
launch a standalone runner under the external-agents interpreter in its own
process group, wait under a hard timeout, parse the runner's result JSON into an
:class:`AgentRunResult`, and on a hard timeout SIGKILL the runner group + force a
label-scoped ``docker rm -f -v`` of any orphan trial container. Historically each
backend carried a byte-identical ~190-line ``run()`` plus identical
result-parsing / teardown / timeout / never-raise helpers, so a fix to the
hard-timeout result-recovery path or the never-raise contract had to be made
twice and could silently diverge (audit ``reuse-simplify``).

:class:`_ExternalTBBackend` single-sources all of that. A concrete backend
overrides only what genuinely differs:

* class attrs :attr:`name` and the bridge identity constants
  :attr:`_RUNNER_MODULE`, :attr:`_REQUEST_FILENAME`, :attr:`_RESULT_FILENAME`,
  :attr:`_RUN_LABEL_PREFIX`, :attr:`_TRIAL_COMPOSE_PROJECT`, :attr:`_LOG_NAME`;
* :meth:`_resolve_terminated` — the native-failure-tag → :class:`TerminatedBy`
  resolver (``from_t2_failure`` + the T2 overrides vs ``from_oh_status``);
* :meth:`_extra_request_fields` — the backend-specific request keys (T2's
  ``parser_name`` + ``additional_context`` vs OH's ``system_message_suffix`` +
  ``max_iterations``);
* :meth:`_extra_result_fields` — the 2-3 backend-specific result fields
  (``reasoning_summary`` / ``agent_cached_tokens``).

This module imports **only the standard library** plus the zero-/wave-1 sibling
``backend`` / ``terminated`` / ``_bridge`` modules — it has no top-level (or
method-local) SDK import, so the package stays importable with the SDKs absent.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable
from uuid import uuid4

from .._bridge import (
    TEARDOWN_WAIT_S as _TEARDOWN_WAIT_S,  # single source (teardown reap bound)
    hard_timeout as _shared_hard_timeout,
    sanitize_compose_name,
    scrubbed_child_env,
    sigkill_group,
)
from ..backend import AgentBackend, AgentRunContext, AgentRunResult
from ..terminated import TerminatedBy

if TYPE_CHECKING:  # pragma: no cover - typing only; never imported at runtime
    from ..telemetry import AgentRunRecord, AgentTelemetry

logger = logging.getLogger(__name__)

__all__ = ["_ExternalTBBackend"]

#: Docker label key Compose v2 stamps on every container it creates.
_COMPOSE_PROJECT_LABEL = "com.docker.compose.project"

# ``_TEARDOWN_WAIT_S`` (bounds every post-SIGKILL ``proc.wait()`` reap so a
# child stuck in uninterruptible sleep cannot hang the hard-timeout / cancel /
# finally unwind) is single-sourced as ``_bridge.TEARDOWN_WAIT_S`` and imported
# above; the module-global alias keeps per-module monkeypatching working.

#: Failure tags every backend maps to a clean (non-failure) ``failure_mode=None``.
_CLEAN_FAILURE_TAGS = (None, "", "none", "unset")


def _tb_keep_images() -> bool:
    """Whether to PRESERVE per-task TB2 Docker images on disk after each run.

    Read in the (non-scrubbed) main process and folded into the request JSON so
    it reliably crosses the subprocess-bridge boundary to the runner, which maps
    it to the harbor harness ``cleanup`` flag: keep_images → ``cleanup=False``.
    The harness's unconditional ``compose down`` still reaps CONTAINERS either
    way (docker_compose_manager.stop); only the extra ``down --rmi all`` (image
    deletion) + buildx prune are skipped, so images survive and a later run with
    ``no_rebuild`` reuses them. Default off (env unset) preserves prior cleanup
    behavior. Enable with ``META_N_TB_KEEP_IMAGES=1``.
    """
    return os.environ.get("META_N_TB_KEEP_IMAGES", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


@dataclass
class _RunPrep:
    """Per-run preparation a subclass may compute BEFORE the subprocess launches.

    The default (no-prep) instance is what the Terminus 2 / OpenHands backends
    use — empty request extras and an identity result mapper, so their inherited
    :meth:`_ExternalTBBackend.run` is byte-for-byte unchanged. The builtin-on-TB
    backend overrides :meth:`_ExternalTBBackend._prepare_run` to (a) author the
    task's bash script on the meta-n side via the native ``Layer1Solver`` (ONE
    outer LLM call) — folded into ``request_fields`` so the runner executes the
    pre-authored script with NO LLM in the venv — and (b) carry the outer
    authoring-token attribution into ``apply_to_result``, so the parsed runner
    result (whose own token fields are 0) is stamped with the meta-n-side outer
    spend (``outer_token_mode=True``).

    Attributes:
        request_fields: Extra keys merged into the request JSON the runner reads
            (e.g. ``{"bash_script": "..."}``). Empty for the no-prep default.
        apply_to_result: A mapper applied to EVERY :class:`AgentRunResult` the base
            ``run`` returns (the scored, timeout, and degraded paths alike), so the
            authoring-token attribution rides on the result regardless of how the
            run ended. The identity default leaves the result untouched.
    """

    request_fields: dict[str, Any] = field(default_factory=dict)
    apply_to_result: Callable[[AgentRunResult], AgentRunResult] = (
        lambda result: result
    )


class _ExternalTBBackend(AgentBackend):
    """Abstract subprocess-bridge backend for a terminal-bench external agent.

    Holds the full ``run()`` control flow and the result/teardown/timeout helpers
    that the Terminus 2 and OpenHands-on-TB backends share verbatim. Token spend
    is the agent's *inner* spend (``outer_token_mode = False``): the solver returns
    ``0`` outer tokens and the agent's tokens ride in ``Trace.inner_*``.

    Subclass contract: set the class attrs below and override
    :meth:`_resolve_terminated`, :meth:`_extra_request_fields`, and
    :meth:`_extra_result_fields`. Everything else (the never-raise ``run``, the
    result parsing, the SIGKILL-group + label-scoped ``docker rm`` teardown, the
    hard-timeout formula) is inherited unchanged.
    """

    #: Stable backend identifier / telemetry ``agent`` coordinate. Subclass sets it.
    name: str = "external-tb"
    outer_token_mode: bool = False

    #: Child module the bridge launches as ``-m <_RUNNER_MODULE>``.
    _RUNNER_MODULE: str = ""
    #: Request / result JSON filenames written under the run's logging dir.
    _REQUEST_FILENAME: str = "request.json"
    _RESULT_FILENAME: str = "result.json"
    #: Per-run compose-project uuid prefix (``<prefix>-<uuid8>``) for a non-leased
    #: caller, and the fallback compose project when no run label is available.
    _RUN_LABEL_PREFIX: str = "ext"
    _TRIAL_COMPOSE_PROJECT: str = "ext-bridge"
    #: Human-readable backend name used in log lines.
    _LOG_NAME: str = "ExternalTBBackend"

    def __init__(
        self,
        *,
        model: str,
        api_base: str,
        api_key: str | None = None,
        venv_python: str,
        runner_dir: str,
        tasks_dir: str,
        temperature: float = 0.7,
        max_episodes: int = 8,
        max_output_tokens: int = 4096,
        test_timeout_sec: float = 180.0,
        provider_env_var: str | None = None,
    ) -> None:
        """Configure the subprocess bridge (no agent SDK import here).

        Args:
            model: litellm-routed inner model id (e.g.
                ``openai/google/gemma-4-31b-qat``).
            api_base: OpenAI-compatible base URL for the inner LLM.
            api_key: Inner-LLM API key. ``None``/empty → the runner uses the
                ``OPENAI_API_KEY=dummy`` the local endpoint accepts.
            venv_python: Path to the external-agents interpreter (the only env with
                the SDK).
            runner_dir: Directory containing the runner module (added to the child
                ``PYTHONPATH`` so ``-m <_RUNNER_MODULE>`` resolves).
            tasks_dir: Directory holding the terminal-bench task folders.
            temperature: Inner-model sampling temperature.
            max_episodes: Agent episode/iteration ceiling (an independent backstop
                alongside the wall-clock budget); must be in ``[1, 8]``. ``8`` is
                the single safety ceiling — the same bound :meth:`_resolve_max_episodes`
                and the runners (``MAX_EPISODES_CEILING``) clamp to at runtime, so
                the accepted construction range equals the effective range.
            max_output_tokens: Forced ``max_tokens`` floor on every inner call
                (clamped to ``>= 4096`` by the runner).
            test_timeout_sec: Fallback verifier (test) timeout forwarded to the
                harness when the task declares no ``max_test_timeout_sec``
                (see :meth:`_resolve_test_timeout_sec`).
            provider_env_var: Name of the provider's API-key env var (accepted for
                parity with the orchestrator's inner-backend kwargs).

        Raises:
            ValueError: if ``max_episodes`` is outside ``[1, 8]``.
        """
        if not (1 <= max_episodes <= 8):
            raise ValueError(
                f"max_episodes must be in [1, 8], got {max_episodes!r}"
            )
        self._model = model
        self._api_base = api_base
        self._api_key = api_key
        self._venv_python = venv_python
        self._runner_dir = runner_dir
        self._tasks_dir = tasks_dir
        self._temperature = float(temperature)
        self._max_episodes = int(max_episodes)
        self._max_output_tokens = max(int(max_output_tokens), 4096)
        self._test_timeout_sec = float(test_timeout_sec)
        self._provider_env_var = provider_env_var

    # ------------------------------------------------------- subclass hooks --
    async def _prepare_run(self, ctx: AgentRunContext) -> _RunPrep:
        """Per-run preparation computed BEFORE the subprocess launches.

        The default does nothing (empty request extras + identity result mapper),
        so the Terminus 2 / OpenHands backends inherit ``run`` unchanged. The
        builtin-on-TB backend overrides this to author the task's bash script on
        the meta-n side (ONE outer ``Layer1Solver`` call) and carry the outer
        authoring-token attribution into the result; see :class:`_RunPrep`.

        MUST NOT raise out of ``run`` — an override should fold any failure into a
        prep that still launches the runner (or returns a clean empty prep) rather
        than propagating, to honor the never-raise contract. ``run`` itself only
        awaits this; it does not guard it, so an override owns that guard.

        Args:
            ctx: The immutable run context (instruction / prompt / workspace).

        Returns:
            A :class:`_RunPrep`; the base default is the no-op prep.
        """
        del ctx
        return _RunPrep()

    def _resolve_terminated(self, failure_mode: object) -> TerminatedBy:
        """Resolve a native failure tag → :class:`TerminatedBy` (subclass-specific).

        Terminus 2 / builtin defer to ``from_t2_failure``; the OH-TB backend
        defers to ``from_oh_status``.
        """
        raise NotImplementedError

    def _extra_request_fields(self, ctx: AgentRunContext, max_episodes: int) -> dict:
        """Backend-specific keys folded into the request dict.

        Terminus 2 returns ``{"parser_name", "additional_context"}``; the OH-TB
        backend returns ``{"system_message_suffix", "max_iterations"}``.
        """
        raise NotImplementedError

    def _extra_result_fields(self, data: dict[str, Any]) -> dict:
        """Backend-specific :class:`AgentRunResult` kwargs from the runner dict.

        Default is the Terminus 2 shape (no reasoning summary, no cached tokens);
        the OH-TB backend overrides to surface ``reasoning_summary`` (the agent's
        last message) and ``agent_cached_tokens`` (the prompt-cache read count).
        """
        return {"reasoning_summary": "", "agent_cached_tokens": 0}

    # ------------------------------------------------------------------ run --
    async def run(
        self,
        ctx: AgentRunContext,
        tel: "AgentTelemetry",
        rec: "AgentRunRecord",
    ) -> AgentRunResult:
        """Drive one trial in a subprocess; never raises.

        Computes the per-run :meth:`_prepare_run` (a no-op for Terminus 2 / OH;
        the meta-n-side script authoring + outer-token attribution for builtin),
        then delegates the whole subprocess-bridge control flow to
        :meth:`_run_bridge`, and finally applies the prep's result mapper to
        WHATEVER result that returns (scored / timeout / degraded alike) so the
        builtin authoring tokens ride on the result regardless of how the run
        ended. The bridge body is single-sourced (teardown / hard-timeout /
        never-raise) and untouched by the prep seam.

        Returns:
            A uniform :class:`AgentRunResult`. Never raises except
            :class:`asyncio.CancelledError`.
        """
        # Monotonic clock: t0 feeds only wall_s deltas (never an absolute
        # timestamp), so an NTP step must not inflate/negate them.
        t0 = time.monotonic()
        # Per-run prep BEFORE the subprocess: the builtin backend authors the
        # script (one outer LLM call) here. The base default is a no-op prep, so
        # the OH/T2 path is byte-for-byte unchanged. CancelledError must propagate;
        # any other prep fault is folded into a degraded result (never-raise).
        try:
            prep = await self._prepare_run(ctx)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - run() must never raise
            logger.exception("%s _prepare_run failed", self._LOG_NAME)
            return self._failed_result("agent_error", ctx, t0, error=str(exc))

        result = await self._run_bridge(ctx, tel, rec, t0, prep)
        # Stamp the prep's attribution (e.g. builtin outer authoring tokens) onto
        # whatever the bridge returned. Identity by default. Never raises out.
        try:
            return prep.apply_to_result(result)
        except Exception:  # noqa: BLE001 - result mapping must never raise out
            logger.exception("%s prep.apply_to_result failed", self._LOG_NAME)
            return result

    async def _run_bridge(
        self,
        ctx: AgentRunContext,
        tel: "AgentTelemetry",
        rec: "AgentRunRecord",
        t0: float,
        prep: "_RunPrep",
    ) -> AgentRunResult:
        """The single-sourced subprocess-bridge control flow (never raises).

        Writes a request JSON (merging ``prep.request_fields``), launches the
        runner under the external-agents interpreter in its own process group,
        waits under a hard timeout, then parses the runner's result JSON into an
        :class:`AgentRunResult`. On a hard timeout the runner's process group is
        SIGKILLed and a belt-and-suspenders ``docker rm -f`` removes any orphan
        trial container.

        Args:
            ctx: Immutable run inputs.
            tel: Active telemetry (unused here; uniform signature).
            rec: The open run record (unused here; the spine folds the result).
            t0: The wall-clock start stamped by :meth:`run` (shared so timing is
                measured from before the prep, not after).
            prep: The per-run :class:`_RunPrep` from :meth:`_prepare_run`; its
                ``request_fields`` are merged into the request JSON.

        Returns:
            A uniform :class:`AgentRunResult` (the prep's result mapper is applied
            by the :meth:`run` wrapper, NOT here).
        """
        del tel, rec  # uniform signature; the bridge needs neither directly
        workdir = ctx.logging_dir
        try:
            workdir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.error("%s could not create workdir: %s", self._LOG_NAME, exc)
            return self._failed_result("env_error", ctx, t0, error=str(exc))

        req_path = workdir / self._REQUEST_FILENAME
        res_path = workdir / self._RESULT_FILENAME

        task_id = self._task_id(ctx)
        staged_files = self._staged_files(ctx)
        # Per-run-unique compose project: the lease session off the workspace
        # handle (sanitized), or a uuid fallback for a non-leased caller. Used for
        # BOTH the runner's trial identity and the scoped hard-timeout teardown.
        run_label = self._run_label(ctx)
        # Episode cap: honor a per-candidate ctx.max_turns when the orchestrator
        # sets one, clamped to the runner's [1, 8] safety ceiling.
        max_episodes = self._resolve_max_episodes(ctx)

        request = {
            "task_id": task_id,
            "tasks_dir": self._tasks_dir,
            "model_name": self._model,
            "api_base": self._api_base,
            "temperature": self._temperature,
            "max_episodes": max_episodes,
            "max_output_tokens": self._max_output_tokens,
            "staged_files": staged_files,
            "agent_timeout_sec": float(ctx.time_limit_s or 900),
            "test_timeout_sec": self._resolve_test_timeout_sec(ctx),
            "output_dir": str(workdir),
            "run_label": run_label,
            "token_budget": int(getattr(ctx, "token_budget", 0) or 0),
            "no_rebuild": False,
            # Preserve per-task TB2 images on disk (containers still reaped) when
            # META_N_TB_KEEP_IMAGES is set — runner maps this to harness cleanup.
            "keep_images": _tb_keep_images(),
        }
        request.update(self._extra_request_fields(ctx, max_episodes))
        # Per-run prep extras (e.g. the builtin backend's pre-authored bash
        # script) override the static request keys for this run.
        request.update(prep.request_fields)
        try:
            req_path.write_text(json.dumps(request))
        except OSError as exc:
            logger.error("%s could not write request JSON: %s", self._LOG_NAME, exc)
            return self._failed_result("env_error", ctx, t0, error=str(exc))

        cmd = [
            self._venv_python,
            "-m",
            self._RUNNER_MODULE,
            "--request",
            str(req_path),
            "--result",
            str(res_path),
        ]
        # SAFETY: build the child env from the SCRUBBED allowlist only — never
        # ``os.environ`` wholesale — so meta-n's OpenRouter / Azure / Anthropic
        # secrets never leak into the runner process. The runner's real credentials
        # (api_base, model_name) arrive in the request JSON; only OPENAI_API_KEY is
        # injected here, plus PYTHONPATH so ``-m <runner>`` resolves.
        env = scrubbed_child_env({"OPENAI_API_KEY": self._api_key or "dummy"})
        # Prepend the runner dir to PYTHONPATH so ``-m <runner>`` resolves.
        existing_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            f"{self._runner_dir}{os.pathsep}{existing_pp}"
            if existing_pp
            else self._runner_dir
        )

        proc: asyncio.subprocess.Process | None = None
        err_bytes = b""
        # Whether the runner's ``communicate()`` returned cleanly (the happy path).
        # Gates the ``finally`` belt-and-suspenders sweep OFF for a clean run whose
        # runner already tore its own trial containers down, so a successful trial
        # is left byte-for-byte untouched.
        runner_completed = False
        # Whether THIS run's compose resources were ALREADY force-swept by the
        # timeout / cancel branch, so the ``finally`` catch-all does not sweep twice.
        torn_down = False
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                env=env,
                cwd=self._runner_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,  # own process group for group-SIGKILL
            )
            # Stamp the runner pid onto the workspace handle so the lease-level
            # hard_kill (installed by the env provider) / shutdown_sweep can SIGKILL
            # a wedged runner whose run() frame is no longer on the stack. Cleared
            # in finally.
            with contextlib.suppress(Exception):
                setattr(ctx.workspace, "agent_pid", proc.pid)
            try:
                _out, err_bytes = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=self._hard_timeout(ctx.time_limit_s),
                )
                runner_completed = True
            except asyncio.TimeoutError:
                # Hard wall-clock envelope expired: kill the whole runner group,
                # then force-remove any orphan trial container the killed runner
                # may have left (its ``finally`` teardown was skipped). Scope the
                # sweep to THIS run's compose project so a sibling concurrent
                # trial's live container is never collateral-killed.
                self._sigkill_group(proc)
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(proc.wait(), timeout=_TEARDOWN_WAIT_S)
                await self._force_compose_down(run_label)
                # Flag the inline sweep so the ``finally`` catch-all below does not
                # force-down this run's compose project a second time.
                torn_down = True
                # The runner's OWN in-process agent timeout often fires before
                # meta-n's hard wall, in which case it wrote a result JSON with
                # partial inner-token spend. Prefer that (so the token basis
                # round-trips and CostGuard does not price a real-token run at $0);
                # fall back to a zeroed timeout result if it wrote nothing.
                if res_path.exists():
                    data, parsed_ok = self._read_result_parsed(res_path, b"")
                    result = self._to_run_result(data, ctx, t0)
                    # Apply the wall-clock TIMEOUT override only when the run did
                    # NOT actually resolve: a run whose tests genuinely PASSED
                    # before the hard wall keeps COMPLETED / native_resolved=True
                    # (otherwise we would write success=True + terminated=TIMEOUT,
                    # an internally contradictory telemetry row). It fires for a
                    # clean-tagged unresolved run AND for a result file that was
                    # not a genuinely-parsed dict (so an unreadable/garbled file
                    # on the timeout path is classified as the meta-n wall-clock
                    # timeout, not the synthesized env_error/parse_error tag).
                    if not result.native_resolved and (
                        not parsed_ok or result.failure_mode in _CLEAN_FAILURE_TAGS
                    ):
                        result.failure_mode = "agent_timeout"
                        result.terminated_by = TerminatedBy.TIMEOUT
                    return result
                return self._failed_result("agent_timeout", ctx, t0)
        except asyncio.CancelledError:
            # Cooperative cancellation must propagate; clean up the child first.
            if proc is not None:
                self._sigkill_group(proc)
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(proc.wait(), timeout=_TEARDOWN_WAIT_S)
            # Flag BEFORE the sweep so the ``finally`` catch-all never repeats it,
            # even if a second cancel interrupts the await below. Shield the sweep
            # so that second cancel cannot abort the force-down mid-teardown (the
            # shielded coroutine still runs to completion), and bound the wait so a
            # wedged docker daemon cannot hang the cancel unwind forever.
            torn_down = True
            with contextlib.suppress(Exception):
                await asyncio.wait_for(
                    asyncio.shield(self._force_compose_down(run_label)),
                    timeout=_TEARDOWN_WAIT_S,
                )
            raise
        except Exception as exc:  # noqa: BLE001 - run() must never raise
            logger.exception("%s subprocess launch failed", self._LOG_NAME)
            return self._failed_result("env_error", ctx, t0, error=str(exc))
        finally:
            # Clear the stamped pid so a later lease.hard_kill / shutdown_sweep
            # cannot reap an unrelated process that reused the dead runner's pid.
            with contextlib.suppress(Exception):
                setattr(ctx.workspace, "agent_pid", None)
            # Defensive: if the process is somehow still alive (e.g. communicate
            # returned but the group lingers), reap it.
            if proc is not None and proc.returncode is None:
                self._sigkill_group(proc)
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(proc.wait(), timeout=_TEARDOWN_WAIT_S)
            # Belt-and-suspenders sweep of THIS run's orphan trial container /
            # network. The timeout / cancel branches sweep inline; a NON-timeout,
            # non-cancel fault (a launch / communicate error routed to the generic
            # ``except Exception`` above) would otherwise SKIP the sweep and leak
            # the SIGKILLed runner's compose resources. Sweep here too when the
            # runner was launched, did NOT complete cleanly, and was not already
            # swept (``torn_down``) — so the timeout/cancel paths are not swept
            # twice and the happy path (``runner_completed``) is left untouched.
            # Shielded so a second cancel arriving mid-teardown cannot interrupt
            # the force-down, and bounded so a wedged docker daemon cannot hang the
            # unwind.
            if proc is not None and not runner_completed and not torn_down:
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(
                        asyncio.shield(self._force_compose_down(run_label)),
                        timeout=_TEARDOWN_WAIT_S,
                    )

        # Parse the result JSON (the authoritative source; stdout is diagnostic).
        # Belt-and-suspenders: this pair sits OUTSIDE the launch try/except, so a
        # future field-shape bug in _read_result/_to_run_result would otherwise
        # escape run() and break the never-raise contract. Fold any such fault
        # into a degraded parse_error result instead of propagating.
        try:
            data = self._read_result(res_path, err_bytes)
            return self._to_run_result(data, ctx, t0)
        except Exception as exc:  # noqa: BLE001 - run() must never raise
            logger.exception("%s result mapping failed", self._LOG_NAME)
            return self._failed_result("parse_error", ctx, t0, error=str(exc))

    # ----------------------------------------------------- request builders --
    def _task_id(self, ctx: AgentRunContext) -> str:
        """Read the task id off the workspace handle (provider contract)."""
        return str(getattr(ctx.workspace, "task_id", "") or "")

    def _staged_files(self, ctx: AgentRunContext) -> dict[str, str]:
        """Read the staged-helper map off the workspace handle (provider contract).

        Drops any key that is absolute or contains ``..`` so a malicious/buggy
        injection cannot ride a path-traversal key across the bridge into the
        runner's host-side staging (defense in depth — the runner re-validates).
        """
        files = getattr(ctx.workspace, "staged_files", None)
        if not isinstance(files, dict):
            return {}
        safe: dict[str, str] = {}
        for k, v in files.items():
            key = str(k)
            parts = Path(key).parts
            if Path(key).is_absolute() or ".." in parts:
                logger.warning(
                    "%s dropping unsafe staged_files key (traversal): %r",
                    self._LOG_NAME, key,
                )
                continue
            safe[key] = str(v)
        return safe

    def _run_label(self, ctx: AgentRunContext) -> str:
        """Per-run-unique compose project label for the trial + scoped teardown.

        Reads the lease session off the workspace handle (``run_label``), sanitized
        to a Docker-safe project name; falls back to a fresh uuid for a non-leased
        caller so two concurrent runs never share a project even without a lease.
        """
        raw = str(getattr(ctx.workspace, "run_label", "") or "")
        if raw:
            return self._sanitize_compose_label(raw)
        return f"{self._RUN_LABEL_PREFIX}-{uuid4().hex[:8]}"

    def _resolve_test_timeout_sec(self, ctx: AgentRunContext) -> float:
        """Per-task verifier timeout for the runner's ``global_test_timeout_sec``.

        The task's declared ``max_test_timeout_sec`` (carried on the env handle's
        ``task`` metadata by the legacy loader) wins when it is a positive number,
        so a task whose test suite legitimately needs more than the bridge default
        is not clamped to it; the construction-fixed ``test_timeout_sec`` applies
        only when the task declares none.
        """
        task = getattr(ctx.workspace, "task", None)
        meta = getattr(task, "metadata", None) or {}
        try:
            declared = float(meta.get("max_test_timeout_sec") or 0)
        except (TypeError, ValueError):
            declared = 0.0
        if declared > 0:
            return declared
        return self._test_timeout_sec

    def _resolve_max_episodes(self, ctx: AgentRunContext) -> int:
        """Honor a per-candidate ``ctx.max_turns`` episode cap, clamped to [1, 8].

        The orchestrator may choose a per-candidate episode ceiling via
        ``ctx.max_turns``; when set we use it (clamped to the runner's safety
        ceiling), otherwise we fall back to the construction-fixed
        ``self._max_episodes``.
        """
        turns = getattr(ctx, "max_turns", None)
        if turns:
            return max(1, min(int(turns), 8))
        return max(1, min(int(self._max_episodes), 8))

    def _compose_instruction_context(self, ctx: AgentRunContext) -> str:
        """Fold the injection surfaces (``system_suffix`` + ``prefix``) into one block.

        Both backends concatenate the two injection fields; Terminus 2 forwards the
        block as ``additional_context`` (no suffix slot), OH-TB as
        ``system_message_suffix`` (a real suffix slot). At depth 1 both injection
        fields are empty → ``""`` (a byte-identical vanilla run).
        """
        parts = [p for p in (ctx.prompt.system_suffix, ctx.prompt.prefix) if p]
        return "\n\n".join(parts)

    def _sanitize_compose_label(self, label: str) -> str:
        """Reduce an arbitrary run label to a Docker-Compose-safe project name.

        Thin wrapper over the canonical :func:`_bridge.sanitize_compose_name` that
        pins the backend's fallback project for an empty label.
        """
        return sanitize_compose_name(label, fallback=self._TRIAL_COMPOSE_PROJECT)

    # ---------------------------------------------------------- result parse --
    def _read_result(self, res_path: Path, err_bytes: bytes) -> dict[str, Any]:
        """Read + parse the runner's result JSON, or synthesize a degraded dict.

        Thin wrapper over :meth:`_read_result_parsed` that drops the ``parsed_ok``
        flag — used on the normal-completion path where only the dict matters.
        """
        data, _parsed_ok = self._read_result_parsed(res_path, err_bytes)
        return data

    def _read_result_parsed(
        self, res_path: Path, err_bytes: bytes
    ) -> tuple[dict[str, Any], bool]:
        """Read + parse the runner's result JSON; return ``(dict, parsed_ok)``.

        The result is read from the file (not stdout), so LM Studio / harness log
        noise on the runner's streams cannot corrupt the parse. The returned
        ``parsed_ok`` flag is ``True`` only when the file contained a genuine JSON
        object; it is ``False`` for a missing, unparseable, OR *non-object* file
        (valid JSON whose top level is ``null`` / a list / a scalar), which lets
        the hard-timeout branch tell a real runner verdict apart from a synthesized
        one.

        A missing/unparseable/non-object file becomes a synthetic ``ok=false`` dict
        carrying the tail of the runner's stderr for diagnosis. The non-dict guard
        prevents ``_to_run_result`` from calling ``.get`` on the parsed value and
        raising ``AttributeError`` out of :meth:`run` (violating the never-raise
        contract). A non-object body is a malformed runner output (a parse-class
        fault), so ``failure_mode`` is ``parse_error``; a missing/unreadable file is
        an ``env_error`` (the runner never produced output).
        """
        if res_path.exists():
            try:
                parsed = json.loads(res_path.read_text())
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("%s result JSON unparseable: %s", self._LOG_NAME, exc)
            else:
                if isinstance(parsed, dict):
                    return parsed, True
                logger.warning(
                    "%s result JSON was not an object (got %s); "
                    "degrading to parse_error.",
                    self._LOG_NAME, type(parsed).__name__,
                )
                tail = (err_bytes or b"").decode(errors="replace")[-2000:]
                return {
                    "ok": False,
                    "reward": 0.0,
                    "is_resolved": False,
                    "failure_mode": "parse_error",
                    # A runner that produced no parseable object captured no
                    # commands, so attribution is unmeasurable (H3 contract);
                    # declare it False rather than letting _to_run_result default
                    # the missing key to True ("attribution-wired").
                    "attribution_available": False,
                    "error": tail or "runner result JSON was not an object",
                }, False
        tail = (err_bytes or b"").decode(errors="replace")[-2000:]
        return {
            "ok": False,
            "reward": 0.0,
            "is_resolved": False,
            "failure_mode": "env_error",
            # A runner that wrote no result JSON captured nothing; attribution is
            # unmeasurable (H3 contract), not wired -> False.
            "attribution_available": False,
            "error": tail or "runner wrote no result JSON",
        }, False

    def _to_run_result(
        self, data: dict[str, Any], ctx: AgentRunContext, t0: float
    ) -> AgentRunResult:
        """Map a runner result dict → :class:`AgentRunResult`.

        Reward is the binary ``is_resolved`` (identical to native TB scoring);
        ``norm_score`` stays deferred to meta-n's evaluator. Tokens come from the
        agent's authoritative totals (``0`` on an agent abort). ``terminated_by`` is
        resolved from the native ``failure_mode`` via :meth:`_resolve_terminated` (a
        clean resolved run → :attr:`TerminatedBy.COMPLETED`). Backend-specific result
        fields come from :meth:`_extra_result_fields`.
        """
        ok = bool(data.get("ok", False))
        is_resolved = bool(data.get("is_resolved", False))
        in_tok = int(data.get("total_input_tokens", 0) or 0)
        out_tok = int(data.get("total_output_tokens", 0) or 0)
        raw_fm = data.get("failure_mode", None)
        # The runner's authoritative binary reward (the verifier signal). Carried
        # onto the result so the scorer derives success FROM the verifier, not
        # transitively from the agent-status enum.
        native_reward = float(data.get("reward", 1.0 if is_resolved else 0.0) or 0.0)

        # terminated_by: a clean resolved run is COMPLETED; otherwise fold the
        # native failure mode. A clean/empty tag on an *unresolved* run means the
        # agent ran fine, tests ran+parsed fine, but the unit tests simply did not
        # pass — a "ran-but-wrong" outcome, which is a clean completion-with-score-0
        # (UNKNOWN), NOT an agent error.
        if ok and is_resolved:
            term = TerminatedBy.COMPLETED
        elif raw_fm in _CLEAN_FAILURE_TAGS:
            # ran-but-wrong: completion with score 0, not an agent error.
            term = TerminatedBy.UNKNOWN
        else:
            term = self._resolve_terminated(raw_fm)

        # Surface failure_mode only when it is a real (non-clean) tag. When the run
        # is COMPLETED (ok && is_resolved) we null any coexisting non-clean tag for
        # consistency: a (COMPLETED, agent_timeout) pair would read as contradictory
        # in telemetry.
        if ok and is_resolved:
            failure_mode = None
        else:
            failure_mode = None if raw_fm in _CLEAN_FAILURE_TAGS else str(raw_fm)

        command_history = [str(c) for c in (data.get("command_history") or [])]
        wall = float(data.get("wall_s", 0.0) or 0.0) or (time.monotonic() - t0)

        # H13: surface the synthesized-degraded dict's ``error`` (the runner's
        # traceback tail, set by _read_result_parsed) into stderr_tail when BOTH
        # the agent and test panes are empty — otherwise a parse_error/env_error
        # degraded row carried no inspectable stderr at all (the panes are ""),
        # and the traceback was silently dropped. stdout_tail stays the agent pane.
        post_agent = str(data.get("post_agent_pane", "") or "")
        post_test = str(data.get("post_test_pane", "") or "")
        err_tail = str(data.get("error", "") or "")
        stderr_tail = (
            post_test if post_test else (post_agent if post_agent else err_tail[-2000:])
        )

        return AgentRunResult(
            transcript=self._read_transcript(data.get("transcript_path", "")),
            stdout_tail=post_agent,
            stderr_tail=stderr_tail,
            artifacts_path=str(ctx.logging_dir),
            agent_tokens=in_tok + out_tok,
            agent_prompt_tokens=in_tok,
            agent_completion_tokens=out_tok,
            agent_calls=int(data.get("agent_calls", 0) or 0),
            cost_usd=0.0,
            cost_basis="priced_from_tokens",
            wall_s=wall,
            steps=int(data.get("steps", 0) or 0),
            command_history=command_history,
            # attribution_available reflects whether in-process command capture was
            # WIRED on this backend, NOT whether any command happened to be
            # captured (H3 producer contract). Each _tb runner now ALWAYS emits an
            # explicit ``attribution_available`` key: builtin_tb → bool(script);
            # oh_tb → True iff the _TBSessionExecutor was installed; t2 → True iff
            # the send_keys wrapper was installed; every error/degraded payload →
            # False. The ``True`` default here is retained ONLY as a legacy
            # fallback for an OLD runner that predates the key. command_count
            # (== len(command_history), stamped by finish_record) is then the
            # measured-zero (>0, called==[]) vs lost-stream (==0) discriminator.
            attribution_available=bool(data.get("attribution_available", True)),
            terminated_by=term,
            failure_mode=failure_mode,
            native_score=native_reward,
            native_resolved=(is_resolved if ok else None),
            native_handle=None,
            **self._extra_result_fields(data),
        )

    @staticmethod
    def _read_transcript(transcript_path: Any) -> str:
        """Render a best-effort transcript from the agent logging dir, or ``""``.

        The runner reports ``transcript_path`` as the agent logging directory the
        harness wrote. We concatenate its text files (bounded) so the ``Trace``
        carries a readable transcript; a missing/empty dir yields ``""``.
        """
        if not transcript_path:
            return ""
        try:
            root = Path(str(transcript_path))
            if not root.exists():
                return ""
            if root.is_file():
                return root.read_text(errors="replace")[:200_000]
            parts: list[str] = []
            budget = 200_000
            for p in sorted(root.rglob("*")):
                if budget <= 0:
                    break
                if p.is_file():
                    with contextlib.suppress(OSError):
                        chunk = p.read_text(errors="replace")[:budget]
                        parts.append(f"# {p.name}\n{chunk}")
                        budget -= len(chunk)
            return "\n\n".join(parts)
        except OSError:
            return ""

    # ------------------------------------------------------- failure helpers --
    def _failed_result(
        self,
        failure_mode: str,
        ctx: AgentRunContext,
        t0: float,
        *,
        error: str | None = None,
    ) -> AgentRunResult:
        """Build a degraded :class:`AgentRunResult` for a pre-/post-run failure.

        Used on a hard timeout, a subprocess-launch failure, or a missing result
        file — paths where there is no scored runner outcome to map. Tokens are
        ``0`` and the ``failure_mode`` folds through :meth:`_resolve_terminated`.
        """
        return AgentRunResult(
            transcript="",
            stdout_tail="",
            stderr_tail=(error or "")[-2000:],
            artifacts_path=str(ctx.logging_dir),
            agent_tokens=0,
            agent_prompt_tokens=0,
            agent_completion_tokens=0,
            agent_calls=0,
            cost_usd=0.0,
            cost_basis="priced_from_tokens",
            wall_s=time.monotonic() - t0,
            steps=0,
            command_history=[],
            attribution_available=False,
            terminated_by=self._resolve_terminated(failure_mode),
            failure_mode=failure_mode,
            native_handle=None,
            **self._extra_result_fields({}),
        )

    def _hard_timeout(self, soft: float | None) -> float:
        """Hard wall-clock ceiling for the runner subprocess (shared formula)."""
        return _shared_hard_timeout(soft)

    @staticmethod
    def _sigkill_group(proc: "asyncio.subprocess.Process") -> None:
        """SIGKILL the runner's whole process group (shared helper).

        A no-op if the process already exited (returncode-guarded against a
        pid/pgid-reuse race) or its group is gone. Never raises.
        """
        sigkill_group(proc)

    @staticmethod
    def _kill_if_alive(proc: "asyncio.subprocess.Process | None") -> None:
        """SIGKILL a still-running CLI child (best-effort, never raises).

        Used on a ``wait_for`` timeout in :meth:`_force_compose_down`: when the
        bounded ``communicate()`` is cancelled the underlying docker CLI process
        keeps running (and holds its pipe fds), so a wedged daemon would leak a
        detached process on every teardown. Kill it instead, mirroring the
        sibling ``openhands._docker_rm`` (``if rm.returncode is None: rm.kill()``).
        """
        if proc is not None and proc.returncode is None:
            with contextlib.suppress(Exception):
                proc.kill()

    @classmethod
    async def _force_compose_down(cls, run_label: str | None = None) -> None:
        """Belt-and-suspenders: force-remove THIS run's orphan trial resources.

        After a hard-timeout SIGKILL the runner may have skipped its terminal
        provisioning ``finally`` (which calls ``Terminal.stop()`` ->
        ``compose down --rmi all --volumes``), so an orphan container / anonymous
        volume / compose network could linger. We scope the sweep to THIS run's
        compose project (``run_label`` — the lease session), NEVER the shared
        constant, so a concurrent sibling trial's live container is never killed.

        Parity with the clean-exit teardown: ``docker rm -f -v`` (drop the container
        AND its anonymous volumes), then best-effort remove the compose-created
        network(s) — enumerated by the project label (so a custom compose network is
        reaped too) plus the conventional ``<project>_default`` as a backstop.
        Best-effort, never raises.
        """
        project = (
            sanitize_compose_name(run_label, fallback=cls._TRIAL_COMPOSE_PROJECT)
            if run_label
            else cls._TRIAL_COMPOSE_PROJECT
        )
        label = f"{_COMPOSE_PROJECT_LABEL}={project}"
        ps: "asyncio.subprocess.Process | None" = None
        try:
            ps = await asyncio.create_subprocess_exec(
                "docker", "ps", "-aq", "--filter", f"label={label}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            out, _ = await asyncio.wait_for(ps.communicate(), timeout=30)
        except Exception:  # noqa: BLE001 - best-effort teardown, never raises
            # Kill the hung CLI client so a wedged daemon does not leak it.
            cls._kill_if_alive(ps)
            return
        ids = [cid for cid in out.decode(errors="replace").split() if cid]
        if ids:
            rm: "asyncio.subprocess.Process | None" = None
            try:
                rm = await asyncio.create_subprocess_exec(
                    # -v: also remove anonymous volumes (parity with the normal
                    # ``compose down --volumes`` path; some tasks declare VOLUMEs).
                    "docker", "rm", "-f", "-v", *ids,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                await asyncio.wait_for(rm.communicate(), timeout=60)
            except Exception:  # noqa: BLE001 - best-effort teardown, never raises
                cls._kill_if_alive(rm)
        # Sweep the compose-created network(s) so a killed runner that skipped its
        # teardown does not leak them. Enumerate by the compose project label
        # (mirroring the container path above) rather than guessing the name:
        # ``<project>_default`` misses a task whose compose declares a CUSTOM
        # network (compose names it ``<project>_<custom>``), so a label query reaps
        # renamed/custom networks in-process instead of relying on the out-of-band
        # reaper.
        netls: "asyncio.subprocess.Process | None" = None
        try:
            netls = await asyncio.create_subprocess_exec(
                "docker", "network", "ls", "-q", "--filter", f"label={label}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            net_out, _ = await asyncio.wait_for(netls.communicate(), timeout=30)
        except Exception:  # noqa: BLE001 - best-effort teardown, never raises
            cls._kill_if_alive(netls)
            net_out = b""
        net_ids = [nid for nid in net_out.decode(errors="replace").split() if nid]
        # Compose does not always label the default network, so also remove the
        # conventionally-named ``<project>_default`` as a backstop.
        net_targets = net_ids + [f"{project}_default"]
        net: "asyncio.subprocess.Process | None" = None
        try:
            net = await asyncio.create_subprocess_exec(
                "docker", "network", "rm", *net_targets,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(net.communicate(), timeout=30)
        except Exception:  # noqa: BLE001 - best-effort teardown, never raises
            cls._kill_if_alive(net)
