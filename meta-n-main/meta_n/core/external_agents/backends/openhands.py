"""OpenHands agent backend — SUBPROCESS BRIDGE (plan §4.3–4.8, locked contract).

This is the :class:`~meta_n.core.external_agents.backend.AgentBackend` strategy
that drives an OpenHands agent for one task. meta-n's process **never imports
``openhands``**: the pins conflict with meta-n's env, so the SDK lives only in
the separate venv ``.venv_external_agents``. Instead, :meth:`OpenHandsBackend.run`
spawns ``.venv_external_agents/bin/python`` running a **standalone runner script**
(``scripts/oh_runner.py``) that imports the OH SDK in-process, drives one
``LocalConversation``, and writes a single result-JSON to a path meta-n chose.
meta-n reads that JSON and folds it into an :class:`AgentRunResult`.

Architecture (baked in)
-----------------------
There is **no** agent-server, no port, no ``/api/conversations``, no HTTP
polling, no ``_cancel_remote`` DELETE. The lone live run path is
``_mode == "subprocess"``: the in-process OH call sequence runs *inside* the
subprocess, not in meta-n. Hard-kill is ``killpg`` on the subprocess group
(``start_new_session=True`` ⇒ ``os.killpg(os.getpgid(pid), SIGKILL)``), backed by
two independent routes: the timeout/cancel path here, and the existing
synchronous ``_COBenchEnv.agent_pid`` / ``lease.hard_kill`` seam in
``co_bench.py``, which this backend stamps the pid onto.

Token/cost accounting is *inner* (``outer_token_mode = False``): OpenHands' LLM
calls go through litellm inside the subprocess, never meta-n's ``LLMClient``, so
spend is read **from the result-JSON** (``accumulated_cost_usd`` →
``cost_basis="native_usd"``; ``0.0`` for the local Gemma/dummy LLM is correct and
needs no entry in ``PRICING``). :meth:`collect_metrics` re-reads tokens/cost from
the result-file (``native_handle``), since no live conversation object crosses the
venv boundary.

Budget enforcement (plan §4.6 — SDK caveat)
-------------------------------------------
The per-run USD ceiling (``ctx.max_budget_usd``) is plumbed end-to-end:
:meth:`run` forwards ``--max-budget-usd`` to the runner, which stamps it onto
``LLM.metrics.max_budget_per_task``. **meta-n does NOT rely on the SDK enforcing
that field inside the conversation loop** — it was recorded-but-unenforced as of
the originally-verified 1.28.0 SDK (no ``accumulated_cost`` comparison), and the
pinned 1.31.0 field is still stamped without meta-n depending on an in-run SDK
kill. The over-budget contract is therefore enforced by the spine's ``CostGuard``
admission pre-check plus the iteration / wall-clock caps — NOT by an in-run SDK
kill. The value still round-trips so it becomes a hard stop automatically if the
SDK enforces it. (The
``budget_exhausted → TerminatedBy.BUDGET_USD`` soft-stop branch in
:func:`_classify_oh_error` covers the case where the SDK *does* raise a budget
error, e.g. a future SDK or a litellm-level cost limit.)

Inner token budget (the OH↔T2 fairness fix)
-------------------------------------------
Separately from the USD ceiling, :meth:`run` forwards ``ctx.token_budget`` as the
runner's ``--token-budget``. The runner DOES enforce that one in-loop: a per-call
hook (``oh_runner._install_token_budget_killswitch``) raises once cumulative inner
prompt+completion tokens exceed the budget, ending the run as
``status="token_budget"`` → :attr:`TerminatedBy.TOKEN_BUDGET`. This gives OpenHands
the same inner cumulative-token stop Terminus 2 already had, so an A/B is not
biased by OH running to its iteration cap while T2 stops on tokens. The residual
overshoot is bounded by one inner call (the call that first crosses the budget
completes); see the runner for details. ``0`` disables the cap.

HARD RULE — **no ``openhands`` import anywhere in this module** (top-level *or*
lazy). The SDK is only ever touched by the runner subprocess. This module imports
cleanly with ``openhands`` ABSENT from meta-n's env.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from .._bridge import (
    TEARDOWN_WAIT_S as _TEARDOWN_WAIT_S,  # single source (teardown reap bound)
    hard_timeout as _shared_hard_timeout,
    scrubbed_child_env,
    sigkill_group,
)
from ..backend import AgentBackend, AgentRunContext, AgentRunResult
from ..terminated import TerminatedBy, from_oh_status
from ..terminated import _OH_STATUS_TO_ENUM  # canonical table (single source)
from ..terminated import _normalize as _normalize_status  # single source (rule)

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids runtime import cycles
    from ..telemetry import AgentRunRecord, AgentTelemetry


__all__ = [
    "OpenHandsBackend",
    "_classify_oh_error",
    "_term_for",
    "_reconcile_termination",
    "_FAILURE_TERMINAL_STATES",
    "_OH_STATUS_TO_ENUM",
]

logger = logging.getLogger(__name__)

#: Repo root (…/meta-n) — three parents up from this file
#: (backends/ -> external_agents/ -> core/ -> meta_n/ -> meta-n).
_REPO_ROOT = Path(__file__).resolve().parents[4]

#: Interpreter that has the OpenHands SDK installed (NEVER meta-n's own python).
_DEFAULT_VENV_PY = _REPO_ROOT / ".venv_external_agents" / "bin" / "python"

#: The standalone runner the venv python executes by path (never imported here).
_DEFAULT_RUNNER = _REPO_ROOT / "scripts" / "oh_runner.py"

#: CO-Bench hard cap on agent iterations (safety: host-shelling terminal tool).
_MAX_ITERATIONS_CAP = 8

#: Floor for the slow local reasoning model — below this it truncates at
#: ``finish_reason=length`` with empty content (the SLOW-reasoning trap).
_MIN_OUTPUT_TOKENS = 4096

#: SAFETY (plan §4.8): the curated allowlist of env vars the OH runner subprocess
#: may inherit is single-sourced in ``external_agents._bridge`` (``SAFE_ENV_KEYS``)
#: and consumed there by ``scrubbed_child_env`` — this module never re-exports it.
#: The OH agent's terminal tool runs on the BARE HOST (CO-Bench is a no-op host
#: workspace, no Docker), so a single ``env`` / ``printenv`` / ``cat ~/.env``
#: would exfiltrate every secret in meta-n's environment; we therefore build the
#: child env from THIS allowlist only — never ``os.environ`` wholesale.

# ---------------------------------------------------------------------------
# Docker sandbox (opt-in) — run the runner INSIDE a container, not on the host
# ---------------------------------------------------------------------------
#: Default image carrying openhands-sdk 1.31.0 (built from scripts/oh_sandbox/,
#: pins sourced from .venv_external_agents so it is byte-consistent with the
#: proven host venv). The image must be (re)built before the opt-in sandbox path
#: is used — build it once with:
#:   docker build -t meta-n-oh-sandbox:1.31.0 scripts/oh_sandbox
_DEFAULT_SANDBOX_IMAGE = "meta-n-oh-sandbox:1.31.0"

#: Container-side mount points. The host workspace (where the agent authors the
#: solution file) lands at ``/workspace``; the run dir (instruction / suffix /
#: result files the backend stages) lands at ``/run``; the runner script is
#: bind-mounted read-only so edits to oh_runner.py need no image rebuild.
_SANDBOX_WORKSPACE = "/workspace"
_SANDBOX_RUNDIR = "/run"
_SANDBOX_RUNNER = "/opt/oh/oh_runner.py"

#: Host names a container cannot reach (loopback) → the Docker host gateway.
#: ``--add-host=host.docker.internal:host-gateway`` makes the rewritten URL
#: resolve to the host on Linux/macOS Docker Desktop alike.
_HOST_GATEWAY_ALIAS = "host.docker.internal"
_LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "0.0.0.0", "::1", "[::1]")

# ``_TEARDOWN_WAIT_S`` (bounds every teardown ``await`` — the ``_reap``
# ``proc.wait()`` and the ``docker rm -f`` ``rm.wait()`` — so a wedged docker
# daemon or an un-reapable child cannot stall the timeout/cancel branch) is
# single-sourced as ``_bridge.TEARDOWN_WAIT_S`` and imported above; the
# module-global alias keeps per-module monkeypatching working.


class OpenHandsBackend(AgentBackend):
    """Drive an OpenHands agent for one task via the subprocess bridge.

    meta-n imports nothing from ``openhands``. :meth:`run` spawns the venv python
    running ``scripts/oh_runner.py``, waits with a hard timeout, ``killpg``-reaps
    the whole process group on timeout/cancel, then parses the runner's result
    JSON into an :class:`AgentRunResult`.

    Attributes:
        name: ``"openhands"`` — the telemetry ``agent`` coordinate.
        outer_token_mode: ``False`` (inner-token accounting; native USD cost).
    """

    name = "openhands"
    outer_token_mode = False

    #: The one live run path. ``"rest"``/``"in_process"`` are accepted as
    #: constructor aliases (callers may still pass them) but collapse to the
    #: single subprocess bridge — there is no other path.
    _MODES = ("subprocess", "rest", "in_process")

    def __init__(
        self,
        *,
        model: str,
        api_base: str | None = None,
        api_key: str | None = None,
        provider_env_var: str | None = None,
        mode: str = "subprocess",
        poll_interval_s: float = 2.0,
        venv_bin: str | None = None,
        venv_python: str | os.PathLike[str] | None = None,
        runner_script: str | os.PathLike[str] | None = None,
        max_output_tokens: int = _MIN_OUTPUT_TOKENS,
        local_default: bool = True,
        solution_file: str = "solve.py",
        sandbox: bool = False,
        sandbox_image: str = _DEFAULT_SANDBOX_IMAGE,
        docker_bin: str = "docker",
    ) -> None:
        """Configure the backend (no ``openhands`` import — the SDK is venv-only).

        Args:
            model: litellm model id passed to the runner (default local target
                ``google/gemma-4-31b-qat``).
            api_base: Provider ``base_url`` forwarded to the runner's ``--base-url``
                (default local LM Studio ``http://127.0.0.1:1234/v1``).
            api_key: Provider API key. Held opaquely; passed to the runner via the
                ``OH_API_KEY`` env var (argv carries ``"-"``), never logged.
            provider_env_var: Name of the env var litellm reads for the resolved
                backend; passed through to the subprocess env when set.
            mode: Accepted for caller compatibility; all values collapse to the
                single subprocess bridge.
            poll_interval_s: Retained for signature compatibility (unused — there
                is no REST polling).
            venv_bin: Optional directory of the OH venv's binaries; ``python`` is
                resolved under it when ``venv_python`` is not given.
            venv_python: Explicit path to the OH venv interpreter. Defaults to
                ``<repo>/.venv_external_agents/bin/python``.
            runner_script: Explicit path to ``oh_runner.py``. Defaults to
                ``<repo>/scripts/oh_runner.py``.
            max_output_tokens: Runner ``--max-output-tokens`` (floored at
                ``4096`` for the slow reasoning model).
            local_default: When ``True`` and ``api_base``/``api_key`` are unset,
                default to the local LM Studio endpoint and ``"dummy"`` key.
            solution_file: Workspace-relative authored target the runner audit-copies
                and the spine authoritatively reads. Sourced from the env provider's
                canonical filename (CO-Bench ``_SOLVE_FILENAME``) so the runner's
                audit copy and the spine's FS read can never name different files.
            sandbox: When ``True``, run ``oh_runner.py`` INSIDE a Docker container
                (``sandbox_image``) with the workspace + run dir bind-mounted and
                the LLM ``base_url`` rewritten from loopback to
                ``host.docker.internal`` — so the OH terminal tool shells out
                against the container, never the bare host. When ``False`` (the
                default) the proven host path runs unchanged (env-scrub fallback).
            sandbox_image: Docker image tag carrying ``openhands-sdk`` for the
                sandboxed path (built from ``scripts/oh_sandbox/``).
            docker_bin: Docker CLI executable (overridable for podman/nerdctl).

        Raises:
            ValueError: If ``mode`` is not one of :attr:`_MODES`.
        """
        if mode not in self._MODES:
            raise ValueError(
                f"OpenHandsBackend mode must be one of {self._MODES}, got {mode!r}"
            )
        self.model = model
        if api_base is None and local_default:
            api_base = "http://127.0.0.1:1234/v1"
        self.api_base = api_base
        if api_key is None and local_default:
            api_key = "dummy"
        self._api_key = api_key
        self.provider_env_var = provider_env_var
        # ``mode`` / ``poll_interval_s`` / ``venv_bin`` are accepted for caller
        # compatibility only: ``mode`` is validated above (single live path),
        # ``poll_interval_s`` has no REST loop to drive, and ``venv_bin`` is
        # consumed locally below to resolve ``self._py``. None are stored, since
        # nothing reads them back after construction.

        # Resolve the venv interpreter + runner script (host-side only).
        if venv_python is not None:
            self._py = str(venv_python)
        elif venv_bin is not None:
            self._py = str(Path(venv_bin) / "python")
        else:
            self._py = str(_DEFAULT_VENV_PY)
        self._runner = str(runner_script) if runner_script is not None else str(_DEFAULT_RUNNER)

        self._max_output_tokens = max(int(max_output_tokens), _MIN_OUTPUT_TOKENS)
        self.solution_file = str(solution_file or "solve.py")

        # Docker sandbox (opt-in). When enabled the runner executes inside the
        # container; the existing host path remains the default fallback.
        self.sandbox = bool(sandbox)
        self.sandbox_image = str(sandbox_image or _DEFAULT_SANDBOX_IMAGE)
        self.docker_bin = str(docker_bin or "docker")

    # ------------------------------------------------------------------ run --
    async def run(
        self,
        ctx: AgentRunContext,
        tel: "AgentTelemetry",
        rec: "AgentRunRecord",
    ) -> AgentRunResult:
        """Drive one OpenHands run via the subprocess bridge; never raises.

        Writes the composed instruction + system suffix into the run's logging
        dir, spawns the venv python running ``oh_runner.py`` in its own process
        group, waits under a hard timeout, ``killpg``-reaps the group on
        timeout/cancel, then parses the result JSON into an
        :class:`AgentRunResult`. Honors the ABC invariant: every failure except
        :class:`asyncio.CancelledError` is folded into the result.

        Args:
            ctx: Immutable run inputs (instruction, prompt, workspace, limits).
            tel: Active telemetry (unused on the subprocess path).
            rec: The run record started by the spine for this run.

        Returns:
            An :class:`AgentRunResult` with ``cost_basis="native_usd"`` and the
            result-file path as ``native_handle``.
        """
        t0 = time.monotonic()
        # Workspace handle: _COBenchEnv exposes ``workspace_handle`` (a host path);
        # fall back to ``ctx.workspace`` stringified for other providers.
        ws = str(getattr(ctx.workspace, "workspace_handle", ctx.workspace))

        # Pre-spawn existence check: a missing venv interpreter / runner script is
        # a pure installation/env fault, NOT an agent failure. Catching it here
        # (rather than letting create_subprocess_exec raise a FileNotFoundError
        # that _classify_oh_error would mislabel ``agent_error``) keeps a broken
        # install from polluting agent-quality telemetry (plan §4.8).
        if not Path(self._py).exists():
            logger.error("OH venv interpreter missing: %s", self._py)
            return self._result_from_failure(
                ctx, ws, "env_error", time.monotonic() - t0
            )
        if not Path(self._runner).exists():
            logger.error("OH runner script missing: %s", self._runner)
            return self._result_from_failure(
                ctx, ws, "env_error", time.monotonic() - t0
            )

        run_dir = Path(ctx.logging_dir)
        try:
            run_dir.mkdir(parents=True, exist_ok=True)
            instr_f = run_dir / "instruction.txt"
            instr_f.write_text(self._compose_instruction(ctx))
            suffix_f = run_dir / "system_suffix.txt"
            suffix_f.write_text(ctx.prompt.system_suffix or "")
        except OSError as exc:  # cannot stage inputs — env fault, not agent
            logger.warning("OH run could not stage inputs: %s", exc)
            return self._result_from_failure(ctx, ws, "env_error", time.monotonic() - t0)
        res_f = run_dir / "oh_result.json"

        # Build the runner argv + child env for either the host path (default) or
        # the opt-in Docker sandbox. ``container_name`` is non-empty only on the
        # sandbox path and lets the timeout/cancel branches ``docker rm -f`` the
        # container (killpg only reaps the docker CLI client, not the daemon-side
        # container).
        argv, env, container_name = self._build_invocation(
            ctx, ws, run_dir, instr_f, suffix_f, res_f
        )

        failure_mode: str | None = None
        proc: asyncio.subprocess.Process | None = None
        # Runner stdout/stderr captured off the PIPE by communicate(). Retained
        # across the finally: stderr lets a degraded row (no/garbled result JSON)
        # surface the runner's traceback instead of dropping it (parity with
        # _ExternalTBBackend); stdout carries the runner's last-ditch result
        # payload when ``--result-file`` could not be written (``oh_runner.main``
        # emits the full JSON to stdout on that path).
        out_bytes: bytes = b""
        err_bytes: bytes = b""
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                start_new_session=True,  # own process group -> killpg reaps the tree
            )
            # Stamp the pid so the synchronous lease.hard_kill seam
            # (_COBenchEnv.agent_pid in co_bench.py) can reap a wedged child on
            # the orchestrator's shutdown/SIGINT sweep.
            with contextlib.suppress(Exception):
                setattr(ctx.workspace, "agent_pid", proc.pid)

            try:
                out_bytes, err_bytes = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=self._hard_timeout(ctx.time_limit_s),
                )
            except asyncio.TimeoutError:
                failure_mode = "agent_timeout"
                await self._reap(proc, container_name)
        except asyncio.CancelledError:
            # Cooperative cancellation: hard-kill the group, then propagate.
            if proc is not None:
                await self._reap(proc, container_name)
            raise
        except BaseException as exc:  # noqa: BLE001 — never raises out of run()
            if proc is not None:
                await self._reap(proc, container_name)
            failure_mode = _classify_oh_error(exc)
            logger.warning(
                "OpenHands subprocess failed task=%s: %s",
                str(ctx.instruction)[:60],
                exc,
            )
        finally:
            # Clear the stamped pid so a later lease.hard_kill can't reap an
            # unrelated process that reused the dead child's pid.
            with contextlib.suppress(Exception):
                setattr(ctx.workspace, "agent_pid", None)
            # Best-effort container sweep: even on the normal-completion path a
            # ``--rm`` container is already gone, but a wedged daemon-side
            # container that outlived the CLI client is force-removed here so no
            # sandbox is ever leaked (the task's "always tear down" rule). Shield
            # the teardown so a CancelledError delivered DURING the unwind cannot
            # abort the removal and leak the container (the await is also bounded
            # so a wedged daemon cannot stall the finally indefinitely).
            if container_name:
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(
                        asyncio.shield(self._docker_rm(container_name)),
                        timeout=_TEARDOWN_WAIT_S,
                    )

        data = self._parse_result(res_f)
        if data is None:
            data = self._parse_stdout_payload(out_bytes)
        if data is None:
            failure_mode = failure_mode or "parse_error"
            data = {}

        status = data.get("status")
        if failure_mode is None and data.get("error"):
            failure_mode = _classify_oh_error(RuntimeError(str(data["error"])))

        if failure_mode == "agent_timeout":
            terminated = TerminatedBy.TIMEOUT
        elif failure_mode is not None:
            terminated = _term_for(failure_mode)
        else:
            terminated = from_oh_status(status)

        # Reconcile the (terminated_by, failure_mode) pair so they are never
        # contradictory. The SDK overloads ``execution_status = ERROR`` for both
        # genuine in-step failures *and* the now-disambiguated max-iterations cap
        # (the runner rewrites the latter to ``status="max_iterations"``), and a
        # STUCK / ERROR status reaches here with ``failure_mode is None`` because
        # the runner only sets ``error`` when ``conv.run()`` actually raised. Left
        # unreconciled that yields the forbidden ``failure_mode=None`` paired with
        # ``terminated_by=AGENT_ERROR``. AGENT_ERROR is reserved for a genuine
        # failure, so whenever the mapped terminal state IS a failure we ensure a
        # matching ``failure_mode`` is carried; conversely a non-failure terminal
        # state (COMPLETED / MAX_TURNS / …) keeps ``failure_mode=None``.
        terminated, failure_mode = _reconcile_termination(
            terminated, failure_mode, status
        )

        # ``(... or [])`` not ``.get(key, [])``: a present-but-null
        # ``command_history`` field returns None from ``.get(key, default)`` and
        # iterating None raises TypeError — which would break the never-raise
        # contract for arbitrary/hand-edited result JSON.
        cmds = [
            str(c.get("command", ""))
            for c in (data.get("command_history") or [])
            if isinstance(c, dict)
        ]
        # Capture-wiredness (H3 producer contract, parity with the TB runners):
        # the runner's event harvest ran iff the payload carries the
        # ``command_history`` key WITH ``error`` unset. Key presence alone is not
        # enough — ``oh_runner.main``'s exception payload also ships
        # ``command_history: []`` but the harvest never ran there. A wired run
        # with zero commands is a measured zero (``utilities_called=[]``), not
        # unmeasurable (``None``).
        capture_wired = "command_history" in data and data.get("error") is None
        pt = int(data.get("prompt_tokens", 0) or 0)
        ct = int(data.get("completion_tokens", 0) or 0)
        cr = int(data.get("cache_read_tokens", 0) or 0)  # optional; read defensively

        # Surface the runner's captured stderr (its traceback) on a degraded row so
        # the diagnostic is not silently dropped — matching the H13 fix in
        # _ExternalTBBackend. Only attach it when something failed; a clean run
        # keeps stderr_tail="" (no behavior change on the success path).
        stderr_tail = ""
        if failure_mode is not None and err_bytes:
            stderr_tail = err_bytes.decode("utf-8", errors="replace")[-2000:]

        return AgentRunResult(
            transcript=self._render_transcript(data),
            reasoning_summary=str(data.get("last_message", ""))[:4000],
            stdout_tail="",
            stderr_tail=stderr_tail,
            artifacts_path=ws,
            agent_tokens=pt + ct,
            agent_prompt_tokens=pt,
            agent_completion_tokens=ct,
            agent_cached_tokens=cr,
            agent_calls=int(data.get("agent_calls", 0) or 0),  # len(metrics.token_usages)
            cost_usd=float(data.get("accumulated_cost_usd", 0.0) or 0.0),
            cost_basis="native_usd",
            wall_s=time.monotonic() - t0,
            steps=len(cmds),
            command_history=cmds,
            attribution_available=capture_wired,
            terminated_by=terminated,
            failure_mode=failure_mode,
            native_handle=str(res_f),  # the JSON path; no live conv crosses the venv
        )

    # ------------------------------------------------------- collect_metrics --
    async def collect_metrics(
        self, env: object, run: AgentRunResult
    ) -> AgentRunResult:
        """Re-read native cost/token usage off the result JSON; never raises.

        No live conversation crosses the venv boundary, so the metric read is
        relocated into the runner and shipped as JSON. ``native_handle`` is the
        result-file path; this re-parses it for ``accumulated_cost_usd`` and the
        token counts. Any parse failure returns the unmodified ``run`` (the
        ABC's never-raise contract), so a missing/garbled JSON never aborts a
        task batch.

        Args:
            env: The live environment yielded by the env provider (unused).
            run: The result produced by :meth:`run`.

        Returns:
            ``run`` augmented from the result JSON, or unmodified ``run`` if the
            JSON could not be read. (``run`` already carries these values from
            :meth:`run`; this is the idempotent post-hoc reconciliation.)
        """
        from dataclasses import replace

        try:
            data = self._parse_result(run.native_handle)
            if not data:
                return run
            pt = int(data.get("prompt_tokens", 0) or 0)
            ct = int(data.get("completion_tokens", 0) or 0)
            cr = int(data.get("cache_read_tokens", 0) or 0)  # optional; defensive
            return replace(
                run,
                agent_prompt_tokens=pt,
                agent_completion_tokens=ct,
                agent_cached_tokens=cr,
                agent_tokens=pt + ct,
                agent_calls=int(data.get("agent_calls", 0) or 0),
                cost_usd=float(data.get("accumulated_cost_usd", 0.0) or 0.0),
                cost_basis="native_usd",
            )
        except Exception as exc:  # noqa: BLE001 — defensive, never-raise (§4.4)
            logger.warning("OH collect_metrics failed (unmodified): %s", exc)
            return run

    # ------------------------------------------------------- subprocess plumbing
    @staticmethod
    def _killpg(proc: "asyncio.subprocess.Process") -> None:
        """Hard-kill the subprocess's whole process group via the shared helper.

        ``start_new_session=True`` gives the runner its own group, so the helper
        SIGKILLs the runner *and* any child the OH terminal tool shelled out on
        the host. A benign no-op when the process has already exited (the helper
        is returncode-guarded against a pid/pgid-reuse race). Never raises.
        """
        sigkill_group(proc)

    # ----------------------------------------------- invocation construction --
    def _runner_flags(
        self,
        ctx: AgentRunContext,
        workspace: str,
        instr_f: object,
        suffix_f: object,
        res_f: object,
        base_url: str,
    ) -> list[str]:
        """Build the ``oh_runner.py`` flag list, parameterized by path basis.

        Used by BOTH the host and sandbox invocations; only the path values
        (``workspace`` / file args) and ``base_url`` differ between them, so the
        flag set is identical and authored exactly once.
        """
        return [
            "--workspace", workspace,
            "--instruction-file", str(instr_f),
            "--system-suffix-file", str(suffix_f),
            "--result-file", str(res_f),
            "--model", self.model,
            "--base-url", base_url or "",
            "--api-key", "-",  # runner reads OH_API_KEY (no key in ps/argv)
            "--max-output-tokens", str(self._max_output_tokens),
            "--max-iterations", str(min(int(ctx.max_turns), _MAX_ITERATIONS_CAP)),
            "--max-budget-usd", str(float(ctx.max_budget_usd or 0.0)),
            # Per-run cumulative INNER-token kill switch (OH↔T2 fairness parity).
            # The runner enforces this in-loop (meta-n does not rely on the SDK
            # enforcing the USD cap above), so an over-budget OH run stops on
            # tokens — like Terminus
            # 2 — instead of running to the iteration cap. 0 = disabled.
            "--token-budget", str(int(getattr(ctx, "token_budget", 0) or 0)),
            "--solution-file", self.solution_file,
        ]

    def _build_invocation(
        self,
        ctx: AgentRunContext,
        ws: str,
        run_dir: Path,
        instr_f: Path,
        suffix_f: Path,
        res_f: Path,
    ) -> tuple[list[str], dict[str, str], str]:
        """Return ``(argv, env, container_name)`` for the host OR sandbox path.

        Host path (default): the venv python runs the runner directly; env is the
        scrubbed allowlist (plan §4.8) so the host-shelling terminal tool sees no
        secrets. ``container_name`` is ``""``.

        Sandbox path (``self.sandbox``): the same runner runs INSIDE a container
        with the workspace + run dir bind-mounted and the LLM ``base_url``
        rewritten from loopback to ``host.docker.internal`` (a container cannot
        reach the host loopback). The OH terminal tool then shells out against the
        container, removing the host-shell-out risk entirely — so the env-scrub is
        no longer the only thing standing between the agent and meta-n's secrets.
        ``container_name`` is the unique ``--name`` so the timeout/cancel paths can
        ``docker rm -f`` it.
        """
        # The child env handed to whichever process we spawn (venv python on the
        # host path; the docker CLI on the sandbox path). SAFETY (plan §4.8):
        # SCRUBBED allowlist only — never ``os.environ`` wholesale — so OpenRouter
        # / Azure / Anthropic keys never reach the agent. The provider key the
        # runner needs is injected separately as OH_API_KEY (and, on the host
        # path, the resolved provider env var).
        env = scrubbed_child_env(
            {
                "OPENHANDS_SUPPRESS_BANNER": "1",
                "OH_API_KEY": self._api_key or "dummy",
            }
        )

        if not self.sandbox:
            if self.provider_env_var and self._api_key:
                env[self.provider_env_var] = self._api_key
            argv = [self._py, self._runner, *self._runner_flags(
                ctx, ws, instr_f, suffix_f, res_f, self.api_base or ""
            )]
            return argv, env, ""

        # --- sandbox path -------------------------------------------------------
        # Container-side paths: workspace -> /workspace, run dir -> /run, runner
        # script -> /opt/oh/oh_runner.py (all bind-mounted below). The runner
        # round-trips the solution at ``{workspace}/{solution-file}``, i.e.
        # ``/workspace/<solution_file>``; that file lands on the host at
        # ``ws/<solution_file>`` through the bind mount — exactly where the spine's
        # authoritative FS read expects it.
        c_instr = f"{_SANDBOX_RUNDIR}/{instr_f.name}"
        c_suffix = f"{_SANDBOX_RUNDIR}/{suffix_f.name}"
        c_res = f"{_SANDBOX_RUNDIR}/{res_f.name}"
        base_url = self._rewrite_base_url(self.api_base or "")

        runner_flags = self._runner_flags(
            ctx, _SANDBOX_WORKSPACE, c_instr, c_suffix, c_res, base_url
        )
        # Unique container name (pid + uuid4) so the timeout/cancel paths can
        # ``docker rm -f`` exactly this container and concurrent runs never collide.
        container_name = f"meta-n-oh-{os.getpid()}-{uuid.uuid4().hex[:12]}"

        docker_argv = [
            self.docker_bin, "run", "--rm",
            "--name", container_name,
            # Loopback in the rewritten base_url resolves to the host gateway.
            "--add-host", f"{_HOST_GATEWAY_ALIAS}:host-gateway",
            # Bind mounts: workspace (rw — agent authors here), run dir (rw —
            # result JSON written here), runner script (ro — no rebuild on edit).
            "-v", f"{ws}:{_SANDBOX_WORKSPACE}",
            "-v", f"{run_dir}:{_SANDBOX_RUNDIR}",
            "-v", f"{self._runner}:{_SANDBOX_RUNNER}:ro",
            # The provider key crosses into the container as OH_API_KEY only — the
            # rest of meta-n's env is NOT forwarded (no ``--env-file``, no
            # ``-e`` of os.environ), preserving the secret-isolation invariant.
            "-e", "OH_API_KEY",
            "-e", "OPENHANDS_SUPPRESS_BANNER=1",
            self.sandbox_image,
            "python", _SANDBOX_RUNNER,
            *runner_flags,
        ]
        return docker_argv, env, container_name

    @staticmethod
    def _rewrite_base_url(base_url: str) -> str:
        """Rewrite a loopback ``base_url`` host to ``host.docker.internal``.

        A container cannot reach the host's ``127.0.0.1`` / ``localhost``; the
        Docker host gateway alias (paired with ``--add-host=...:host-gateway``)
        routes the call back to the LM Studio / provider endpoint on the host. A
        non-loopback URL (a real remote provider) is returned unchanged.
        """
        if not base_url:
            return base_url
        for host in _LOOPBACK_HOSTS:
            # Match the host as a URL authority component (``//127.0.0.1`` or
            # ``//127.0.0.1:1234``) so we never rewrite an unrelated substring.
            if f"//{host}/" in base_url or f"//{host}:" in base_url \
                    or base_url.rstrip("/").endswith(f"//{host}"):
                return base_url.replace(host, _HOST_GATEWAY_ALIAS, 1)
        return base_url

    async def _reap(
        self, proc: "asyncio.subprocess.Process", container_name: str
    ) -> None:
        """Hard-kill the spawned process group AND (sandbox) force-rm the container.

        ``killpg`` reaps the spawned process (the venv python on the host path, or
        the docker CLI client on the sandbox path), but on the sandbox path the
        actual workload runs in a daemon-side container that ``killpg`` does not
        touch — so ``docker rm -f <name>`` is issued to guarantee no sandbox is
        leaked. Never raises.

        Both awaits are bounded by ``_TEARDOWN_WAIT_S`` (matching ``run()``'s
        finally sweep) so an un-reapable child or a wedged docker daemon cannot
        stall the timeout/cancel branch that drives this method indefinitely.
        """
        self._killpg(proc)
        # Bound the reap so a child that ignored SIGKILL (e.g. uninterruptible
        # sleep) cannot hang the timeout/cancel unwind forever.
        with contextlib.suppress(Exception):
            await asyncio.wait_for(proc.wait(), timeout=_TEARDOWN_WAIT_S)
        if container_name:
            await self._docker_rm(container_name)

    async def _docker_rm(self, container_name: str) -> None:
        """``docker rm -f <name>`` — force-remove a (possibly already-gone) container.

        Best-effort and never-raise: a ``--rm`` container that already exited is
        gone and this is a harmless no-op (non-zero exit swallowed); a wedged
        container that outlived its CLI client is force-removed here.

        The ``rm.wait()`` is bounded by ``_TEARDOWN_WAIT_S`` so a wedged docker
        daemon (the ``docker rm`` client blocking on the socket) cannot stall
        teardown forever; on expiry the hung CLI client is SIGKILLed (so it is
        not left as a zombie) and the timeout is swallowed like any other
        teardown error.
        """
        rm: "asyncio.subprocess.Process | None" = None
        try:
            rm = await asyncio.create_subprocess_exec(
                self.docker_bin, "rm", "-f", container_name,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(rm.wait(), timeout=_TEARDOWN_WAIT_S)
        except Exception as exc:  # noqa: BLE001 — teardown must never raise
            # On a timeout the docker-CLI client is still alive and blocked on
            # the daemon socket; kill it so we don't leak a zombie waiter.
            if rm is not None and rm.returncode is None:
                with contextlib.suppress(Exception):
                    rm.kill()
            logger.debug("docker rm -f %s failed (ignored): %s", container_name, exc)

    def _parse_result(self, res_f: object) -> dict | None:
        """Parse the runner's result JSON; returns ``{}``/``None``, never raises.

        Args:
            res_f: The result-file path (or ``None``).

        Returns:
            The parsed dict on success, or ``None`` if the file is
            missing/empty/garbled (caller treats ``None`` as a parse error).
        """
        if res_f is None:
            return None
        try:
            text = Path(str(res_f)).read_text()
        except OSError:
            return None
        if not text.strip():
            return None
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return None
        return data if isinstance(data, dict) else None

    @staticmethod
    def _parse_stdout_payload(out_bytes: bytes) -> dict | None:
        """Parse the runner's last-ditch stdout payload; never raises.

        ``oh_runner.main`` emits the full result JSON on stdout (and exits 1)
        when it cannot write ``--result-file``. The runner otherwise keeps
        stdout empty (banner suppressed, visualizer off), so the captured bytes
        are either empty or exactly that payload.

        Args:
            out_bytes: The child's captured stdout (possibly empty).

        Returns:
            The parsed dict, or ``None`` if stdout carries no parseable payload.
        """
        if not out_bytes:
            return None
        try:
            data = json.loads(out_bytes.decode("utf-8", errors="replace").strip())
        except (json.JSONDecodeError, ValueError):
            return None
        return data if isinstance(data, dict) else None

    def _hard_timeout(self, soft: float | None) -> float:
        """Hard wall-clock ceiling for the runner subprocess (shared formula).

        Parity with the Terminus 2 backstop — both delegate to
        ``external_agents._bridge.hard_timeout`` so the formula is single-sourced.
        """
        return _shared_hard_timeout(soft)

    # ----------------------------------------------------------- rendering ----
    def _compose_instruction(self, ctx: AgentRunContext) -> str:
        """Fold the injection prompt prefix into the task instruction.

        The injected ``system_suffix`` is delivered separately via
        ``AgentContext.system_message_suffix`` (the OH injection slot), so only
        the inter-layer ``prefix`` is prepended here; the raw task comes last. At
        depth 1 both injection fields are empty, making this byte-identical to a
        vanilla OH run.

        Args:
            ctx: The run context carrying ``prompt`` and ``instruction``.

        Returns:
            The composed instruction string.
        """
        parts = [p for p in (ctx.prompt.prefix,) if p]
        parts.append(ctx.instruction)
        return "\n\n".join(parts)

    @staticmethod
    def _render_transcript(data: dict) -> str:
        """Render a readable transcript from the result-JSON command history.

        Pairs each TerminalAction command with its captured output (the runner
        ships ``command_history`` as ``[{command, output}]``), then appends the
        agent's last message. Returns ``""`` when there is nothing to render.

        Args:
            data: The parsed runner result dict (possibly empty).

        Returns:
            The rendered transcript text.
        """
        lines: list[str] = []
        for c in (data.get("command_history") or []):  # tolerate present-but-null
            if not isinstance(c, dict):
                continue
            cmd = str(c.get("command", ""))
            out = str(c.get("output", ""))
            lines.append(f"$ {cmd}")
            if out:
                lines.append(out)
        last = str(data.get("last_message", ""))
        if last:
            lines.append(last)
        return "\n".join(lines)

    def _result_from_failure(
        self, ctx: AgentRunContext, ws: str, failure_mode: str, wall: float
    ) -> AgentRunResult:
        """Build a zeroed :class:`AgentRunResult` for a pre-spawn failure.

        Used when meta-n cannot even stage inputs / spawn the runner (an env
        fault), so there is no result JSON to parse.
        """
        return AgentRunResult(
            transcript="",
            reasoning_summary="",
            artifacts_path=ws,
            cost_basis="native_usd",
            wall_s=wall,
            command_history=[],
            attribution_available=False,
            terminated_by=_term_for(failure_mode),
            failure_mode=failure_mode,
            native_handle=None,
        )


# ---------------------------------------------------------------------------
# Module-level classifiers (install-free, pure — no openhands import)
# ---------------------------------------------------------------------------
def _classify_oh_error(e: BaseException) -> str:
    """Classify an OpenHands run exception into a soft ``failure_mode`` string.

    Budget-stop and iteration-cap signals are mapped to **soft** failure modes
    (plan §4.6): a budget stop becomes ``"budget_exhausted"`` (which
    :func:`_term_for` maps to :attr:`TerminatedBy.BUDGET_USD` — OpenHands' in-run
    cap is a per-run USD ceiling, not a token cap) so the run still proceeds to
    scoring (partial credit possible) rather than being aborted or zeroed.
    Spawn-layer faults (missing / non-executable interpreter) map to
    ``"env_error"``; context-window and parse failures are recognized too;
    everything else falls back to ``"agent_error"`` (which maps to
    :attr:`TerminatedBy.AGENT_ERROR`).

    The classifier matches on the exception's type name and message text (also
    used for the ``error`` string the runner ships in its result JSON), both of
    which are stable across the runner/SDK boundary.

    Args:
        e: The exception raised by the OpenHands run path (or wrapping the
            runner's ``error`` string).

    Returns:
        A short, stable ``failure_mode`` token (a key of
        :data:`~meta_n.core.external_agents.terminated._OH_STATUS_TO_ENUM`).
    """
    text = f"{type(e).__name__}: {e}".lower()
    # Spawn-layer faults: a missing / non-executable / wrong-arch venv interpreter
    # is a pure installation/env fault, NOT an agent-quality failure (plan §4.8).
    # These arise only from ``create_subprocess_exec`` (the runner's own result
    # JSON never carries them), so there is no false-positive risk for genuine
    # agent failures surfaced via the result ``error`` string.
    if (
        isinstance(e, (FileNotFoundError, PermissionError))
        or "exec format" in text
        or "no such file or directory" in text
        or "permission denied" in text
    ):
        return "env_error"
    # Budget / iteration caps → SOFT stops (plan §4.6): proceed to scoring.
    if "budget" in text or "max_budget" in text or "cost limit" in text:
        return "budget_exhausted"
    if "max_iteration" in text or "iteration limit" in text or "max turns" in text:
        return "max_iterations"
    # Context window exhaustion.
    if "context" in text and ("window" in text or "length" in text or "token" in text):
        return "context_window_exceeded"
    # Output/parse failures.
    if "parse" in text or "jsondecode" in text or "invalid json" in text:
        return "parse_error"
    # Server / connection faults (server crash ≠ agent failure, §4.8).
    if (
        "connection" in text
        or "connectionrefused" in text
        or "server" in text
        or "503" in text
        or "502" in text
        or "500" in text
    ):
        return "env_error"
    if "timeout" in text or "timederror" in text:
        return "agent_timeout"
    return "agent_error"


def _term_for(failure_mode: str | None) -> TerminatedBy:
    """Map a soft ``failure_mode`` string to a :class:`TerminatedBy` member.

    Delegates to the canonical
    :func:`~meta_n.core.external_agents.terminated.from_oh_status` so the single
    :data:`~meta_n.core.external_agents.terminated._OH_STATUS_TO_ENUM` table is
    the only source of truth (it covers both raw ``execution_status`` values and
    the soft ``failure_mode`` strings this module emits, plan §7.5). ``None``
    (the success path) maps to :attr:`TerminatedBy.COMPLETED`; any unmapped
    string falls back to :attr:`TerminatedBy.UNKNOWN`.

    Args:
        failure_mode: The token from :func:`_classify_oh_error`, or ``None``.

    Returns:
        The corresponding :class:`TerminatedBy` member.
    """
    if failure_mode is None:
        return TerminatedBy.COMPLETED
    return from_oh_status(failure_mode)


#: Terminal states that represent a *genuine* in-agent failure — the only ones
#: for which a non-None ``failure_mode`` must be carried (and which a successful
#: run must never land on). ``MAX_TURNS`` / ``TIMEOUT`` / ``BUDGET_USD`` are
#: NON-failure stops: the agent simply ran out of its iteration / wall-clock /
#: USD envelope and the run still proceeds to scoring, so they keep
#: ``failure_mode=None``. ``UNKNOWN`` is indeterminate (not a positive failure
#: signal) and is likewise left with ``failure_mode=None``.
_FAILURE_TERMINAL_STATES: frozenset[TerminatedBy] = frozenset(
    {
        TerminatedBy.AGENT_ERROR,
        TerminatedBy.ENV_ERROR,
        TerminatedBy.PARSE_ERROR,
        TerminatedBy.CONTEXT_LEN,
    }
)


def _reconcile_termination(
    terminated: TerminatedBy,
    failure_mode: str | None,
    status: object,
) -> tuple[TerminatedBy, str | None]:
    """Make the ``(terminated_by, failure_mode)`` pair mutually consistent.

    Two invariants are enforced so telemetry can never contradict itself:

    * **A failure terminal state implies a failure_mode.** If ``terminated`` is in
      :data:`_FAILURE_TERMINAL_STATES` (``AGENT_ERROR`` / ``ENV_ERROR`` /
      ``PARSE_ERROR`` / ``CONTEXT_LEN``) but ``failure_mode`` is ``None`` — the
      shape produced by a bare ``execution_status = STUCK``/``ERROR`` whose
      ``ConversationRunError`` did not surface an ``error`` string — a stable
      ``failure_mode`` is synthesized from the status so the forbidden
      ``failure_mode=None`` + ``AGENT_ERROR`` pairing can never be written.
    * **A non-failure terminal state clears only failure-flavored tokens.**
      ``COMPLETED`` / ``MAX_TURNS`` (the iteration cap) / ``TIMEOUT`` /
      ``BUDGET_USD`` / ``UNKNOWN`` are not genuine failures: a finished-but-low-
      score run, or one that merely hit its iteration / wall-clock / USD ceiling,
      is a *non-error* stop whose success is decided by the scorer
      (``native_score`` / ``native_resolved``), not by ``terminated_by``. A stale
      ``failure_mode`` that *itself* maps INTO a failure state (e.g. an
      ``"agent_error"`` token paired with a non-failure terminal) is cleared to
      ``None`` to avoid a contradictory pair; a *descriptive* non-failure token
      that maps to a non-failure state (e.g. ``"max_iterations"`` -> MAX_TURNS) is
      intentionally retained.

    Args:
        terminated: The mapped :class:`TerminatedBy` from the status / failure_mode.
        failure_mode: The soft failure token, or ``None``.
        status: The raw runner ``status`` value (used to synthesize a token).

    Returns:
        The reconciled ``(terminated_by, failure_mode)`` tuple.
    """
    if terminated in _FAILURE_TERMINAL_STATES:
        if not failure_mode:
            # Synthesize a token from the native status (e.g. "stuck", "error")
            # so the failure carries a non-None failure_mode. _normalize-style
            # lowercasing keeps it a stable, in-table key where possible.
            token = _normalize_status(status)
            failure_mode = token or "agent_error"
        return terminated, failure_mode

    # Non-failure terminal state: a lingering failure_mode that itself maps to a
    # NON-failure state (e.g. "max_iterations" → MAX_TURNS) is allowed to stay,
    # but a failure-flavored token paired with a non-failure terminal state would
    # be contradictory, so clear it.
    if failure_mode is not None and from_oh_status(failure_mode) in _FAILURE_TERMINAL_STATES:
        failure_mode = None
    return terminated, failure_mode


# ``_normalize_status`` is the canonical ``terminated._normalize`` (imported
# above, single source): both reduce a raw status value to a stripped,
# lower-cased lookup token (or ``None``).
