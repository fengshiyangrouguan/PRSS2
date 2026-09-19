#!/usr/bin/env python
"""Standalone OpenHands-on-terminal-bench subprocess bridge for meta-n.

WHAT THIS IS
------------
This is the FULL runner (promoted from ``oh_tb_runner_spike.py``) that lets an
OpenHands ``Agent`` drive a real **terminal_bench** task container and be scored by
terminal_bench's own verifier. It is the OpenHands analogue of
``scripts/t2_runner.py`` (Terminus 2): same ``--request`` / ``--result`` JSON
contract, same result-JSON schema, so meta-n's shared result parser
(``backends.terminus2._map_result``) consumes both unchanged.

It runs **only** under
``<repo>/.venv_external_agents/bin/python`` (the venv
that has BOTH ``openhands`` 1.28 and ``terminal_bench`` 0.2.18). meta-n NEVER
imports this module — meta-n's own env has neither SDK, and their pins conflict
with meta-n's. meta-n shells out to::

    /Users/.../.venv_external_agents/bin/python scripts/oh_tb_runner.py \
        --request <request.json> --result <result.json>

CRITICAL INVARIANT (mirrors t2_runner): ``import terminal_bench`` / ``import
openhands`` (and everything that pulls them in) happens ONLY inside ``_run`` /
helper functions, NEVER at module top level, so that even if meta-n's interpreter
imported this module it would not drag the conflicting SDKs into meta-n's process.
The module top level is deliberately limited to the stdlib.

DESIGN (option-c: OpenHands AS a terminal_bench BaseAgent)
----------------------------------------------------------
1. ``_TBSessionExecutor(ToolExecutor)``: routes one OH ``TerminalAction`` into the
   provisioned ``TmuxSession`` (the REAL task container) via
   ``session.send_command(TerminalCommand(...))`` + ``get_incremental_output()``,
   appends a sentinel ``echo`` to approximate ``$?``, and returns a
   ``TerminalObservation``. This is what makes OH commands land inside the same
   Docker container the verifier inspects.
2. ``_TBTerminalTool(TerminalTool)``: a TerminalTool subclass whose ``create``
   wires OUR session executor (the registry only passes ``conv_state``; the live
   session is handed in via a class attribute). Registered under name ``terminal``
   so the OH Agent resolves OUR tool. file_editor is intentionally DROPPED — file
   writes fold into shell heredocs in the container.
3. ``InjectedOpenHandsTBAgent(BaseAgent)``: builds the LLM (local Gemma; the
   ``_route_model`` litellm prefix; the >=4096 ``max_output_tokens`` floor; the
   ported token-budget kill switch), injects ``system_message_suffix``, stages
   helper files into the container, drives a bounded ``conv.run()`` loop, and
   returns an ``AgentResult`` with token totals from
   ``llm.metrics.accumulated_token_usage``.
4. ``_run`` builds ``Harness(... agent_import_path='oh_tb_runner:Injected...',
   task_ids=[task_id], cleanup=True, ...)`` and calls ``harness._run_trial`` —
   IDENTICAL provisioning to t2_runner — then maps ``results`` + the captured
   ``AgentResult`` into the SAME result-JSON schema t2_runner writes.

SAFETY / TEARDOWN / ROBUSTNESS
------------------------------
* OH commands execute INSIDE the Docker container ``spin_up_terminal`` builds; the
  host is never touched by agent commands (the executor only relays into the
  session). Helper-file staging is the one host touch and is path-validated.
* ``spin_up_terminal``'s ``finally`` tears the container down (``cleanup=True``).
* Bounded by three independent stops: ``max_iteration_per_run`` (<= ceiling), the
  ported token-budget kill switch, and the harness's
  ``asyncio.wait_for(global_agent_timeout_sec)`` — plus meta-n's outer SIGKILL of
  this runner's process group.
* On ANY exception we STILL write a valid result JSON (``ok=false`` /
  ``status="error"`` / ``reward=0.0``) and exit 0, so meta-n always has a file to
  parse. We exit nonzero only if we cannot even write the result file.
"""

# ---------------------------------------------------------------------------
# MODULE TOP LEVEL: STDLIB ONLY.
# Never import terminal_bench / openhands / litellm here -- only inside functions,
# so this module is import-safe even from meta-n's interpreter (which lacks the
# SDKs). This mirrors the t2_runner invariant exactly.
# ---------------------------------------------------------------------------
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

# Shared, stdlib-only runner helpers (single-sourced across the three runners; see
# scripts/_runner_common.py — a top-level module on the child PYTHONPATH). Imported
# at module top so this module stays stdlib-only at module scope (openhands /
# terminal_bench / litellm are still imported lazily inside ``_run`` and the agent
# class). Re-exported under the historical private names so existing references
# keep resolving to the SAME class/function objects.
from _runner_common import (  # noqa: E402  (top-level module on the child PYTHONPATH)
    KNOWN_LITELLM_PROVIDERS as _KNOWN_LITELLM_PROVIDERS,
    accumulated_inner_tokens as _accumulated_inner_tokens,
    classify_runner_error as _classify_runner_error,
    error_result_payload as _error_result_payload,
    failure_mode_str as _failure_mode_str,
    install_token_budget_killswitch as _install_token_budget_killswitch,
    is_token_budget_error as _is_token_budget_error,
    route_model as _route_model,
    safe_read_text as _safe_read_text,
    sanitize_run_label as _sanitize_run_label_common,
    stage_helper_files as _stage_helper_files_common,
    write_result as _write_result,
)


def _sanitize_run_label(label) -> str:
    """OH-TB wrapper over the shared ``sanitize_run_label`` (pins the ``oh-tb`` stem).

    Compose project names must match ``[a-z0-9][a-z0-9_-]*``. meta-n passes the
    already-sanitized lease session; this is a defensive pass plus an
    ``oh-tb-{uuid}`` fallback so two concurrent runs never share a project (which
    would collide on the globally unique ``container_name``).
    """
    return _sanitize_run_label_common(label, fallback="oh-tb")


# Keep the OpenHands SDK banner off stdout so nothing pollutes a parse of this
# process (meta-n reads the result solely from the --result file).
os.environ.setdefault("OPENHANDS_SUPPRESS_BANNER", "1")


# ---------------------------------------------------------------------------
# Constants (locked routing + endpoint facts; mirror t2_runner / oh_runner).
# ---------------------------------------------------------------------------
DEFAULT_API_BASE = "http://127.0.0.1:1234/v1"
DEFAULT_MODEL = "google/gemma-4-31b-qat"
# The local LM Studio model is a SLOW reasoning model: without a high max_tokens
# it truncates at finish_reason="length" with empty content (litellm ->
# OutputLengthExceededError). We force >= 4096 on every inner call.
DEFAULT_MAX_OUTPUT_TOKENS = 4096
# Hard cap on OH iterations regardless of what the request asks for (safety bound),
# mirroring t2_runner's MAX_EPISODES_CEILING.
MAX_ITERATIONS_CEILING = 8
# Fallback trial/compose-project name when meta-n supplies no per-run label.
DEFAULT_RUN_LABEL = "oh-tb-bridge"

# Module-global handle to the live agent's captured state so main()'s top-level
# error path can recover partial token spend / inner-call count even when
# _run_trial raises before the AgentResult is captured. Set inside the injected
# agent's perform_task; read defensively.
_ACTIVE_CAPTURE: dict | None = None

# Module-level record of every command our executor sent into the container, with
# its (approximate) exit code and captured output. This is the OH-side analogue of
# terminal_bench's commands.txt — Terminus 2 writes that file itself, but our
# custom executor bypasses the path that populates it, so we capture commands here
# for attribution (command_history in the result JSON).
_COMMANDS_EXECUTED: list[dict] = []

# Module-global flag recording whether the in-process command-capture executor was
# WIRED (the custom ``_TBSessionExecutor`` installed by ``_TBTerminalTool.create``).
# This is the H3 producer signal for ``attribution_available``: it reflects whether
# capture was POSSIBLE on this run, NOT whether any command happened to be
# captured. Emitted in the success result payload; the error payload hard-sets it
# False (a degraded path is UNMEASURABLE). Reset at ``_run`` start.
_CAPTURE_WIRED: bool = False


# The litellm routing helpers (``_route_model`` / ``_KNOWN_LITELLM_PROVIDERS``), the
# token-budget kill-switch trio, and ``_sanitize_run_label`` are imported / wrapped
# from ``_runner_common`` at module top (single-sourced across the runners).


# ===========================================================================
# Request loading / normalisation (mirrors t2_runner._load_request)
# ===========================================================================
def _load_request(args: argparse.Namespace) -> dict:
    """Build the request dict from ``--request <json file>`` (the canonical meta-n
    path) and/or the discrete named flags. Values from a ``--request`` file are the
    base; any explicitly provided discrete flag overrides the corresponding key.
    """
    req: dict = {}

    # 1) Base layer: the JSON request file (canonical meta-n path).
    if args.request:
        req = json.loads(Path(args.request).read_text())

    # 2) Override / fill from discrete flags (the named-flag entry form).
    if args.task_id is not None:
        req["task_id"] = args.task_id
    if args.tasks_dir is not None:
        req["tasks_dir"] = args.tasks_dir
    if args.model_routing is not None:
        # --model-routing is the (possibly already-litellm-routed) model_name, e.g.
        # "google/gemma-4-31b-qat" or "openai/google/gemma-4-31b-qat".
        req["model_name"] = args.model_routing
    if args.base_url is not None:
        req["api_base"] = args.base_url
    if args.max_iterations is not None:
        req["max_iterations"] = args.max_iterations
    if args.timeout is not None:
        # --timeout is the agent wall timeout (seconds) for this single task.
        req["agent_timeout_sec"] = args.timeout
    if args.output_dir is not None:
        req["output_dir"] = args.output_dir
    if getattr(args, "run_label", None) is not None:
        req["run_label"] = args.run_label
    if getattr(args, "token_budget", None) is not None:
        req["token_budget"] = args.token_budget
    if getattr(args, "max_output_tokens", None) is not None:
        req["max_output_tokens"] = args.max_output_tokens

    # --instruction-suffix / --system-suffix-file points at a FILE whose contents
    # are injected as the OpenHands AgentContext.system_message_suffix (the
    # injection "additional_context"). Unlike Terminus 2 (no suffix slot, so it was
    # a prefix), OpenHands has a real system_message_suffix slot, so we use it.
    suffix_path = args.instruction_suffix or args.system_suffix_file
    if suffix_path is not None:
        p = Path(suffix_path)
        if p.exists():
            req["system_message_suffix"] = p.read_text()

    # --staged-files points at a JSON file: {"container/rel/path": "<contents>"}.
    if args.staged_files is not None:
        staged_path = Path(args.staged_files)
        if staged_path.exists():
            req["staged_files"] = json.loads(staged_path.read_text())

    # ---- Defaults for anything still unset --------------------------------
    req.setdefault("api_base", DEFAULT_API_BASE)
    req.setdefault("model_name", DEFAULT_MODEL)
    req.setdefault("temperature", 0.7)
    req.setdefault("max_output_tokens", DEFAULT_MAX_OUTPUT_TOKENS)
    req.setdefault("system_message_suffix", "")
    req.setdefault("staged_files", {})
    req.setdefault("agent_timeout_sec", 900)
    req.setdefault("test_timeout_sec", 180)
    req.setdefault("cmd_max_timeout_sec", 60.0)
    req.setdefault("token_budget", 0)
    req.setdefault("no_rebuild", False)

    # Clamp max_iterations to the safety ceiling (<= 8).
    its = int(req.get("max_iterations", MAX_ITERATIONS_CEILING)
              or MAX_ITERATIONS_CEILING)
    req["max_iterations"] = max(1, min(its, MAX_ITERATIONS_CEILING))

    # Force max_output_tokens high enough for the slow local reasoning model.
    req["max_output_tokens"] = max(
        int(req.get("max_output_tokens") or DEFAULT_MAX_OUTPUT_TOKENS),
        DEFAULT_MAX_OUTPUT_TOKENS,
    )

    return req


# ===========================================================================
# Injected agent class -- built lazily so the SDK imports it subclasses live in
# function scope, never at module top level (the critical invariant).
# ===========================================================================
def _build_agent_class(req: dict):
    """Construct and return ``InjectedOpenHandsTBAgent`` (a terminal_bench
    ``BaseAgent``) wired with this request's injection config: system-message-suffix
    injection, helper-file staging, the >=4096 max_output_tokens floor, litellm
    routing, and the ported token-budget kill switch. SDK imports are local to this
    function (the import-safety invariant).
    """
    # ---- terminal_bench imports (SDK scope only) -------------------------------
    from terminal_bench.agents.base_agent import AgentResult, BaseAgent
    from terminal_bench.agents.failure_mode import FailureMode
    from terminal_bench.terminal.models import TerminalCommand
    from terminal_bench.terminal.tmux_session import TmuxSession

    # ---- openhands imports (SDK scope only) ------------------------------------
    from openhands.sdk import LLM, Agent, AgentContext, Conversation
    from openhands.sdk.tool import ToolExecutor
    from openhands.sdk.tool.registry import register_tool
    from openhands.sdk.tool.spec import Tool
    from openhands.tools.terminal import TerminalTool
    from openhands.tools.terminal.definition import (
        TerminalAction,
        TerminalObservation,
    )

    system_message_suffix = req.get("system_message_suffix", "") or ""
    staged_files = req.get("staged_files", {}) or {}
    max_output_tokens = int(req.get("max_output_tokens", DEFAULT_MAX_OUTPUT_TOKENS))
    api_base = req.get("api_base", DEFAULT_API_BASE)
    temperature = float(req.get("temperature", 0.7))
    token_budget = int(req.get("token_budget", 0) or 0)
    cmd_max_timeout = float(req.get("cmd_max_timeout_sec", 60.0))

    # --- 1. custom ToolExecutor: one OH TerminalAction -> the TmuxSession --------
    class _TBSessionExecutor(ToolExecutor):
        """Routes one OH terminal action into the terminal_bench ``TmuxSession``
        (the live task container). Appends a sentinel ``echo`` to approximate the
        command's exit code, records the command for attribution, strips the
        sentinel from what the model sees, and returns a ``TerminalObservation``.
        Never crashes the run: an executor error is surfaced to the agent as an
        error observation."""

        def __init__(self, session: "TmuxSession"):
            self._session = session

        def __call__(self, action: "TerminalAction", conversation=None):
            command = (getattr(action, "command", "") or "").strip()
            if not command:
                return TerminalObservation.from_text("", command=command)
            # Sentinel-tagged exit-code probe so we can approximate $?.
            wrapped = f"{command}\necho __OH_RC__$?"
            try:
                self._session.send_command(
                    TerminalCommand(
                        command=wrapped,
                        block=True,
                        append_enter=True,
                        max_timeout_sec=cmd_max_timeout,
                    )
                )
                out = self._session.get_incremental_output()
            except Exception as exc:  # noqa: BLE001 - surface, do not crash the run
                msg = f"[executor error] {exc!r}"
                _COMMANDS_EXECUTED.append(
                    {"command": command, "rc": None, "output": msg[-2000:],
                     "error": True}
                )
                return TerminalObservation.from_text(
                    msg, is_error=True, command=command
                )

            rc = None
            for raw in out.splitlines():
                line = raw.strip()
                if line.startswith("__OH_RC__"):
                    tail = line[len("__OH_RC__"):].strip()
                    if tail.isdigit():
                        rc = int(tail)
            # Strip our sentinel echo lines from what the model sees.
            cleaned = "\n".join(
                ln for ln in out.splitlines() if "__OH_RC__" not in ln
            )
            _COMMANDS_EXECUTED.append(
                {"command": command, "rc": rc, "output": cleaned[-2000:],
                 "error": bool(rc)}
            )
            obs = TerminalObservation.from_text(
                cleaned, is_error=bool(rc), command=command
            )
            # Best-effort exit-code stamp (field name varies across SDK minors).
            with contextlib.suppress(Exception):
                obs.exit_code = rc
            return obs

    # --- 2. TerminalTool subclass that injects OUR executor ----------------------
    class _TBTerminalTool(TerminalTool):
        """``create`` ignores the SDK's own terminal backend and wires our session
        executor instead. The live session is handed in via the class attribute
        ``_tb_session`` (the registry only passes ``conv_state`` to ``create``)."""

        _tb_session: "TmuxSession | None" = None

        @classmethod
        def create(cls, conv_state, **params):
            session = cls._tb_session
            if session is None:
                raise RuntimeError("_TBTerminalTool._tb_session was not set")
            # H3 producer: our custom session executor is being wired, so in-process
            # command capture is POSSIBLE for this run — record it so the success
            # payload advertises attribution_available=True regardless of how many
            # commands the agent ultimately issues.
            global _CAPTURE_WIRED
            _CAPTURE_WIRED = True
            return TerminalTool.create(
                conv_state, executor=_TBSessionExecutor(session)
            )

    def _stage_helper_files(session) -> None:
        """Stage the injection's helper files into the container (shared guard).

        Delegates to the single-sourced, security-load-bearing
        ``_runner_common.stage_helper_files`` (path-traversal validation +
        ``/app/<rel>`` copy — keys are workspace-relative and already carry the
        ``helpers/`` prefix) over the request's ``staged_files`` map.
        Best-effort: a staging failure is logged to stderr but never aborts the run.
        """
        _stage_helper_files_common(session, staged_files, prefix="oh_tb")

    class InjectedOpenHandsTBAgent(BaseAgent):
        """OpenHands as a terminal_bench ``BaseAgent`` (option-c).

        At depth 1 (``system_message_suffix == ""`` and no staged files) this is the
        vanilla OH agent driving the container, except for the mandatory high
        max_output_tokens (which the local model requires regardless).

        ``captured`` is a class attribute the runner reads after the trial to recover
        token totals / inner-call count / failure_mode even when the agent aborts (a
        timeout / kill-switch path never returns a usable AgentResult through the
        harness's wait_for)."""

        # Per-run capture dict; reset in perform_task and published to the module
        # global so main()'s error path can recover partial token spend.
        captured: dict | None = None

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._model = kwargs.get("model_name", DEFAULT_MODEL)
            self._base_url = kwargs.get("api_base", api_base)
            self._max_output_tokens = int(
                kwargs.get("max_output_tokens", max_output_tokens)
            )
            self._temperature = float(kwargs.get("temperature", temperature))
            self._system_message_suffix = kwargs.get(
                "system_message_suffix", system_message_suffix
            )
            self._max_iterations = int(kwargs.get("max_iterations", 8))
            self._token_budget = int(kwargs.get("token_budget", token_budget) or 0)
            self._timestamped_markers: list = []

        @staticmethod
        def name() -> str:
            return "openhands-meta-n"

        def perform_task(
            self,
            instruction: str,
            session: "TmuxSession",
            logging_dir: "Path | None" = None,
        ) -> "AgentResult":
            global _ACTIVE_CAPTURE
            cap = {
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "inner_calls": 0,
                # R2-EA-1: prompt-cache reads + the agent's last message, shipped
                # to meta-n for parity with the REST oh_runner (agent_cached_tokens
                # / reasoning_summary on the OH-TB consumer side).
                "cache_read_tokens": 0,
                "last_message": "",
                "status": "",
                "error": None,
                "token_budget_hit": False,
            }
            InjectedOpenHandsTBAgent.captured = cap
            _ACTIVE_CAPTURE = cap

            failure_mode = FailureMode.NONE
            llm = None
            conv = None
            try:
                # 0) Stage any helper files into the container BEFORE the loop.
                _stage_helper_files(session)

                # 1) LLM — slow local reasoning model; high max_output_tokens floor;
                #    litellm routing; ported token-budget kill switch.
                model = _route_model(self._model, self._base_url)
                llm = LLM(
                    model=model,
                    base_url=self._base_url,
                    api_key=os.environ.get("OPENAI_API_KEY", "dummy"),
                    temperature=self._temperature,
                    max_output_tokens=max(self._max_output_tokens,
                                          DEFAULT_MAX_OUTPUT_TOKENS),
                )
                _install_token_budget_killswitch(llm, self._token_budget)

                # 2) Register our terminal tool wired to THIS task's session. The
                #    registry resolves the tool by name at Agent build time, so the
                #    session must be set on the class BEFORE constructing the Agent.
                _TBTerminalTool._tb_session = session
                register_tool("terminal", _TBTerminalTool)

                # 3) system_message_suffix injection (OH has a real suffix slot,
                #    unlike Terminus 2 where the injection is a prefix).
                ctx = (
                    AgentContext(system_message_suffix=self._system_message_suffix)
                    if self._system_message_suffix
                    else None
                )

                # 4) Agent: terminal tool ONLY (file_editor dropped — folds into
                #    shell heredocs in the container). No browser / sub-agents.
                agent = Agent(
                    llm=llm,
                    tools=[Tool(name="terminal")],
                    agent_context=ctx,
                )

                # 5) Conversation rooted in a host-local scratch workspace (required
                #    by LocalConversation but irrelevant — all real work happens in
                #    the container via our executor). Cap iterations; disable the
                #    Rich visualizer so the SDK writes nothing to stdout.
                ws = Path(logging_dir or ".") / "oh_workspace"
                ws.mkdir(parents=True, exist_ok=True)
                conv = Conversation(
                    agent,
                    workspace=str(ws),
                    max_iteration_per_run=min(
                        self._max_iterations, MAX_ITERATIONS_CEILING
                    ),
                    stuck_detection=True,
                    visualizer=None,
                )

                # 6) Drive — run() takes NO message; send first, then run. Catch ONLY
                #    the token-budget kill switch (a soft stop); everything else
                #    re-raises into the outer handler.
                conv.send_message(instruction)
                try:
                    conv.run()
                except BaseException as run_exc:  # noqa: BLE001
                    if not _is_token_budget_error(run_exc):
                        raise
                    cap["token_budget_hit"] = True
                    failure_mode = FailureMode.UNKNOWN  # soft stop -> token_budget

                # 7) Harvest execution status off conv.state (advisory telemetry).
                with contextlib.suppress(Exception):
                    cap["status"] = str(conv.state.execution_status.value)

            except Exception as exc:  # noqa: BLE001 - record, return partial result
                cap["error"] = f"{type(exc).__name__}: {exc}"
                failure_mode = FailureMode.UNKNOWN_AGENT_ERROR
                traceback.print_exc()
            finally:
                # Token totals + inner-call count from the LLM metrics. This is the
                # ONLY source that survives an agent abort (the harness's wait_for
                # may discard a partial AgentResult), so we always read it here.
                if llm is not None:
                    try:
                        usage = llm.metrics.accumulated_token_usage
                        cap["total_input_tokens"] = int(
                            getattr(usage, "prompt_tokens", 0) or 0
                        )
                        cap["total_output_tokens"] = int(
                            getattr(usage, "completion_tokens", 0) or 0
                        )
                        # R2-EA-1: prompt-cache reads (optional per the contract;
                        # meta-n reads it defensively) — parity with oh_runner.
                        cap["cache_read_tokens"] = int(
                            getattr(usage, "cache_read_tokens", 0) or 0
                        )
                    except Exception:  # noqa: BLE001
                        pass
                    try:
                        # Metrics.token_usages is a per-call list -> its length is
                        # the real inner LLM-call count (the OH analogue of T2's
                        # budgeted-LLM call counter).
                        cap["inner_calls"] = len(
                            getattr(llm.metrics, "token_usages", None) or []
                        )
                    except Exception:  # noqa: BLE001
                        pass
                # R2-EA-1: the agent's last message (bounded), harvested off the
                # conversation events before close — parity with oh_runner's
                # last_message. Best-effort; survives an abort that still left a
                # partial event stream.
                with contextlib.suppress(Exception):
                    if conv is not None:
                        cap["last_message"] = _last_agent_message_events(
                            list(conv.state.events)
                        )
                with contextlib.suppress(Exception):
                    if conv is not None:
                        conv.close()

            return AgentResult(
                total_input_tokens=cap["total_input_tokens"],
                total_output_tokens=cap["total_output_tokens"],
                failure_mode=failure_mode,
                timestamped_markers=self._timestamped_markers,
            )

    return InjectedOpenHandsTBAgent


# ===========================================================================
# Result-JSON helpers (mirror t2_runner)
# ===========================================================================
# ``_safe_read_text`` / ``_failure_mode_str`` / ``_write_result`` are imported
# from ``_runner_common`` at module top (single-sourced across the runners).


def _error_payload(task_id: str, failure_mode: str, error: str,
                   total_input_tokens: int = 0,
                   total_output_tokens: int = 0,
                   agent_calls: int = 0,
                   command_history: list | None = None,
                   last_message: str = "",
                   cache_read_tokens: int = 0) -> dict:
    """Build a valid result JSON for the error path (same schema t2_runner writes).

    Delegates to the single-sourced ``_runner_common.error_result_payload``.
    ``total_*_tokens`` / ``agent_calls`` carry any partial inner spend recovered
    from the captured agent state so a run that burned real tokens then aborted is
    priced correctly (not at $0) and reports a real inner-call count.
    ``command_history`` carries any commands the executor already ran so
    attribution survives the abort.

    R2-EA-1: ``last_message`` / ``cache_read_tokens`` carry any partial agent
    last-message + prompt-cache reads recovered from the captured state so the
    OH-TB consumer's ``reasoning_summary`` / ``agent_cached_tokens`` are populated
    on the error path too (parity with the success payload and the REST oh_runner).

    H3 producer: a degraded/error payload hard-sets ``attribution_available=False``
    — the run is UNMEASURABLE (the contract maps False → None / unmeasurable on the
    consumer side), so any partial command_history is NOT treated as a measured
    sample.
    """
    payload = _error_result_payload(
        task_id,
        failure_mode,
        error,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        agent_calls=agent_calls,
        command_history=command_history,
    )
    payload["attribution_available"] = False
    payload["last_message"] = str(last_message or "")
    payload["cache_read_tokens"] = int(cache_read_tokens or 0)
    return payload


def _command_history_strings() -> list[str]:
    """Flatten the module-level executed-command log into the ``command_history``
    string list the result schema (and meta-n's attribution) expects. Each entry is
    the literal shell command our executor sent into the container, in order."""
    return [str(c.get("command", "")) for c in _COMMANDS_EXECUTED
            if str(c.get("command", "")).strip()]


def _last_agent_message_events(events) -> str:
    """Join the ``TextContent.text`` of the last source=='agent' MessageEvent.

    R2-EA-1 parity with ``scripts/oh_runner.py:_last_agent_message`` so OH-over-TB
    ships the agent's final message as ``last_message`` (the consumer's
    ``reasoning_summary``) just like the REST OH runner does. The openhands import
    is FUNCTION-scoped (the module-top import-safety invariant); a missing SDK /
    unexpected event shape degrades to ``""`` rather than raising.
    """
    try:
        from openhands.sdk.event import MessageEvent
    except Exception:  # noqa: BLE001 - SDK absent / shape drift -> no summary
        return ""
    last = ""
    try:
        for e in events or []:
            if isinstance(e, MessageEvent) and getattr(e, "source", None) == "agent":
                msg = getattr(e, "llm_message", None)
                parts: list[str] = []
                for c in (getattr(msg, "content", None) or []):
                    text = getattr(c, "text", None)
                    if isinstance(text, str):
                        parts.append(text)
                joined = "".join(parts).strip()
                if joined:
                    last = joined
    except Exception:  # noqa: BLE001 - best-effort summary, never sink the run
        return last
    return last


# ===========================================================================
# The actual single-task run (drives terminal_bench's own machinery)
# ===========================================================================
def _run(req: dict, result_path: str) -> int:
    """Execute one task end-to-end via ``Harness._run_trial`` and write the result
    JSON. Returns a process exit code (0 on completion incl. scored failure).
    Mirrors ``t2_runner._run``."""
    task_id = req.get("task_id", "")
    started = time.time()

    # SDK imports -- function scope ONLY (the import-safety invariant).
    from terminal_bench.harness.harness import Harness
    from terminal_bench.handlers.trial_handler import TrialHandler

    tasks_dir = req["tasks_dir"]
    model_name = req["model_name"]
    api_base = req.get("api_base", DEFAULT_API_BASE)
    temperature = float(req.get("temperature", 0.7))
    max_iterations = int(req["max_iterations"])
    output_dir = req["output_dir"]
    agent_timeout_sec = float(req.get("agent_timeout_sec", 900))
    test_timeout_sec = float(req.get("test_timeout_sec", 180))
    no_rebuild = bool(req.get("no_rebuild", False))
    # keep_images=True preserves the per-task image on disk (containers are still
    # reaped by the harness's unconditional ``compose down``); only ``--rmi all``
    # is skipped. Set via META_N_TB_KEEP_IMAGES (see _external_tb._tb_keep_images).
    keep_images = bool(req.get("keep_images", False))

    task_input_path = Path(tasks_dir) / task_id
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Build the injected agent class wired with this request's injection config.
    Injected = _build_agent_class(req)
    Injected.captured = None
    # H3 producer: reset the capture-wired flag; it flips True inside
    # ``_TBTerminalTool.create`` once our custom session executor is installed.
    global _CAPTURE_WIRED
    _CAPTURE_WIRED = False

    # Register the class under a STABLE module name so AgentFactory can import it by
    # "module:class". When run as a script this module is "__main__"; register both
    # names so import_path resolution works regardless of how the harness imports it.
    import_module_name = __name__ if __name__ != "__main__" else "oh_tb_runner"
    sys.modules.setdefault("oh_tb_runner", sys.modules[__name__])
    setattr(sys.modules["oh_tb_runner"], "InjectedOpenHandsTBAgent", Injected)
    if __name__ in sys.modules:
        setattr(sys.modules[__name__], "InjectedOpenHandsTBAgent", Injected)
    # Ensure this script's dir is importable by the harness.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    agent_import_path = f"{import_module_name}:InjectedOpenHandsTBAgent"

    # Per-run-unique trial identity: meta-n passes the lease session as run_label so
    # the compose project (== trial_name) is GLOBALLY unique (avoids container-name
    # collisions under --parallel). Fall back to a uuid when no label is supplied.
    run_label = _sanitize_run_label(req.get("run_label") or DEFAULT_RUN_LABEL)
    run_id = run_label

    # Build the Harness for this single task. model_name is passed BOTH as
    # Harness(model_name=...) (metadata/lock only) AND inside agent_kwargs (the
    # harness does NOT forward model_name to the agent). dataset_path is REQUIRED
    # (Harness.__init__ eagerly builds a Dataset); point it at the local tasks dir
    # and scope to the one task via task_ids so no remote registry is touched.
    harness = Harness(
        output_path=output_path,
        run_id=run_id,
        agent_import_path=agent_import_path,
        model_name=model_name,
        dataset_path=Path(tasks_dir),
        agent_kwargs={
            "model_name": model_name,
            "api_base": api_base,
            "temperature": temperature,
            "max_iterations": max_iterations,
            # Injection-only kwargs (read by InjectedOpenHandsTBAgent.__init__):
            "system_message_suffix": req.get("system_message_suffix", ""),
            "staged_files": req.get("staged_files", {}),
            "max_output_tokens": req.get(
                "max_output_tokens", DEFAULT_MAX_OUTPUT_TOKENS
            ),
            "token_budget": req.get("token_budget", 0),
        },
        no_rebuild=no_rebuild,
        cleanup=not keep_images,  # keep_images preserves the image on disk (containers still reaped)
        task_ids=[task_id],
        n_concurrent_trials=1,
        # Enforce the wall + test timeouts at the harness level (the harness wraps
        # perform_task in asyncio.wait_for(global_agent_timeout_sec)).
        global_agent_timeout_sec=agent_timeout_sec,
        global_test_timeout_sec=test_timeout_sec,
    )

    trial = TrialHandler(
        trial_name=run_label,
        input_path=task_input_path,
        output_path=output_path / run_id,
    )

    # ---- Drive the single-task trial. spin_up_terminal's finally guarantees
    # ---- Terminal.stop() (container teardown) even if _run_trial raises. ----
    results = harness._run_trial(trial)

    # ---- Map TrialResults + captured agent state -> result JSON --------------
    cap = getattr(Injected, "captured", None) or {}

    is_resolved = bool(getattr(results, "is_resolved", False))
    reward = 1.0 if is_resolved else 0.0

    # Token totals: results.total_*_tokens are set from the AgentResult (and may be
    # 0 / None on an agent abort). Fall back to the captured agent state (read off
    # llm.metrics, which survives an abort).
    total_in = getattr(results, "total_input_tokens", None) or 0
    total_out = getattr(results, "total_output_tokens", None) or 0
    if not total_in and not total_out:
        total_in = int(cap.get("total_input_tokens", 0) or 0)
        total_out = int(cap.get("total_output_tokens", 0) or 0)

    # Real inner LLM-call count (the OH analogue of len(metrics.token_usages)).
    agent_calls = int(cap.get("inner_calls", 0) or 0)

    # failure_mode: the harness's authoritative tag normally; if our kill switch
    # fired (a SOFT stop the harness sees as FailureMode.UNKNOWN), rewrite it to the
    # soft 'token_budget' tag the shared parser maps to TerminatedBy.TOKEN_BUDGET.
    failure_mode = _failure_mode_str(getattr(results, "failure_mode", None))
    if cap.get("token_budget_hit"):
        failure_mode = "token_budget"

    # parser_results: TrialResults stores {test_name: UnitTestStatus}. Serialise to
    # {name: status_string}.
    parser_results: dict = {}
    raw_parser = getattr(results, "parser_results", None) or {}
    try:
        for k, v in raw_parser.items():
            parser_results[str(k)] = getattr(v, "value", str(v))
    except Exception:
        parser_results = {}

    # Artifact paths the harness already wrote.
    tp = trial.trial_paths
    transcript_path = str(tp.agent_logging_dir)
    post_agent_pane = _safe_read_text(tp.post_agent_pane_path)
    post_test_pane = _safe_read_text(tp.post_test_pane_path)

    # command_history: our custom executor bypasses terminal_bench's commands.txt
    # writer, so we capture commands ourselves (the module-level executed log) for
    # attribution. Fall back to the harness's commands.txt if for some reason our
    # log is empty (e.g. the agent shelled out via a path we did not intercept).
    command_history = _command_history_strings()
    if not command_history:
        cmds = _safe_read_text(tp.commands_path)
        if cmds:
            command_history = [ln for ln in cmds.splitlines() if ln.strip()]

    # steps: number of commands the agent actually ran (the closest robust proxy to
    # turn count for OH-over-TB; episode dirs are a Terminus-2-specific artifact).
    steps = len(_COMMANDS_EXECUTED)

    payload = {
        "ok": True,
        "status": "ok",
        "reward": reward,
        "score": reward,
        "is_resolved": is_resolved,
        "task_id": task_id,
        "total_input_tokens": int(total_in),
        "total_output_tokens": int(total_out),
        "agent_calls": int(agent_calls),
        # R2-EA-1: prompt-cache reads + the agent's last message (parity with the
        # REST oh_runner). The OH-TB backend reads these as agent_cached_tokens /
        # reasoning_summary; absent them they silently defaulted to 0 / "".
        "cache_read_tokens": int(cap.get("cache_read_tokens", 0) or 0),
        "last_message": str(cap.get("last_message", "") or ""),
        "failure_mode": failure_mode,
        "parser_results": parser_results,
        "timestamped_markers": [],
        "command_history": command_history,
        # H3 producer: was in-process command capture WIRED for this run? True iff
        # the custom _TBSessionExecutor was installed (NOT whether any command was
        # seen) — so an empty command_history with this True is a MEASURED-none,
        # while command_count==0 with this True flags a LOST stream (the
        # discriminator _external_tb/telemetry consume).
        "attribution_available": _CAPTURE_WIRED,
        "transcript_path": transcript_path,
        "post_agent_pane": post_agent_pane[-20000:],
        "post_test_pane": post_test_pane[-20000:],
        "wall_s": round(time.time() - started, 3),
        "steps": steps,
        "trial_started_at": getattr(results, "trial_started_at", None),
        "trial_ended_at": getattr(results, "trial_ended_at", None),
        "error": None,
    }
    _write_result(result_path, payload)
    return 0


# ===========================================================================
# Entry point (mirrors t2_runner.main)
# ===========================================================================
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="oh_tb_runner",
        description="Standalone OpenHands-on-terminal-bench subprocess bridge.",
    )
    # Canonical meta-n invocation: request + result JSON.
    parser.add_argument("--request", help="Path to the request JSON written by meta-n.")
    parser.add_argument("--result", "--result-json", dest="result",
                        help="Path to write the result JSON.")
    # Discrete-flag entry form (parity with t2_runner's named flags).
    parser.add_argument("--task-id", dest="task_id",
                        help="Terminal-bench task id (e.g. hello-world).")
    parser.add_argument("--tasks-dir", dest="tasks_dir",
                        help="Directory containing the task folders.")
    parser.add_argument("--instruction-suffix", dest="instruction_suffix",
                        help="Path to a file whose contents are injected as the "
                             "OpenHands AgentContext.system_message_suffix.")
    parser.add_argument("--system-suffix-file", dest="system_suffix_file",
                        help="Alias for --instruction-suffix (OH naming).")
    parser.add_argument("--model-routing", dest="model_routing",
                        help="model_name (litellm-routed), e.g. "
                             "google/gemma-4-31b-qat.")
    parser.add_argument("--base-url", dest="base_url",
                        help="OpenAI-compatible api_base, e.g. "
                             "http://127.0.0.1:1234/v1.")
    parser.add_argument("--max-iterations", dest="max_iterations", type=int,
                        help="Max OpenHands iterations (clamped to <= 8).")
    parser.add_argument("--max-output-tokens", dest="max_output_tokens", type=int,
                        help="Inner-call max_output_tokens (floored to >= 4096).")
    parser.add_argument("--token-budget", dest="token_budget", type=int,
                        help="Per-run cumulative inner-token ceiling (0 disables).")
    parser.add_argument("--timeout", dest="timeout", type=float,
                        help="Agent wall timeout in seconds for this task.")
    parser.add_argument("--staged-files", dest="staged_files",
                        help="Path to a JSON file mapping container-relative paths "
                             "to file contents to stage into the container.")
    parser.add_argument("--output-dir", dest="output_dir",
                        help="Directory for harness logs/artifacts.")
    parser.add_argument("--run-label", dest="run_label",
                        help="Per-run-unique compose project / trial name (meta-n "
                             "passes the lease session); a uuid is used when absent.")
    args = parser.parse_args(argv)

    # --result is mandatory: it is the file meta-n reads.
    if not args.result:
        print("[oh_tb_runner] FATAL: --result/--result-json is required.",
              file=sys.stderr)
        return 2

    result_path = args.result

    # Always set the inner LLM auth env (LM Studio accepts any key). We never
    # overwrite an OPENAI_API_KEY meta-n may have passed deliberately, but the
    # contract specifies a dummy so litellm's openai/ provider is satisfied.
    os.environ.setdefault("OPENAI_API_KEY", "dummy")

    task_id = ""
    try:
        req = _load_request(args)
        task_id = req.get("task_id", "")
        # Validate the minimal required fields up front so a missing field produces
        # a clean error JSON rather than a deep traceback.
        for field in ("task_id", "tasks_dir", "model_name", "output_dir"):
            if not req.get(field):
                raise ValueError(f"request is missing required field: {field!r}")
        return _run(req, result_path)
    except Exception as exc:  # noqa: BLE001 - top-level guard: ALWAYS emit JSON
        tb = traceback.format_exc()
        print(f"[oh_tb_runner] ERROR: {exc}\n{tb}", file=sys.stderr)
        failure_mode = _classify_runner_error(exc)
        # Recover any partial inner spend AND inner-call count from the captured
        # agent state so an abort that burned real tokens is neither priced at $0
        # nor reported with 0 inner calls; carry any commands already run.
        part_in = part_out = part_calls = part_cache = 0
        part_last = ""
        if _ACTIVE_CAPTURE is not None:
            part_in = int(_ACTIVE_CAPTURE.get("total_input_tokens", 0) or 0)
            part_out = int(_ACTIVE_CAPTURE.get("total_output_tokens", 0) or 0)
            part_calls = int(_ACTIVE_CAPTURE.get("inner_calls", 0) or 0)
            # R2-EA-1: carry partial cache reads + last message through the error
            # path so the OH-TB consumer fields are not lost on an abort.
            part_cache = int(_ACTIVE_CAPTURE.get("cache_read_tokens", 0) or 0)
            part_last = str(_ACTIVE_CAPTURE.get("last_message", "") or "")
            if _ACTIVE_CAPTURE.get("token_budget_hit"):
                failure_mode = "token_budget"
        try:
            _write_result(
                result_path,
                _error_payload(
                    task_id, failure_mode, tb, part_in, part_out, part_calls,
                    _command_history_strings(), part_last, part_cache,
                ),
            )
        except Exception as werr:  # pragma: no cover - cannot even write result
            print(f"[oh_tb_runner] FATAL: could not write result JSON: {werr}",
                  file=sys.stderr)
            return 1
        # Exit 0 on a caught error: meta-n has a valid JSON to parse.
        return 0


if __name__ == "__main__":
    sys.exit(main())
