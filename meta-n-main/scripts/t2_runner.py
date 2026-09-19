#!/usr/bin/env python
"""Standalone Terminus 2 subprocess bridge for meta-n's external-agents seam.

WHAT THIS IS
------------
This module is the *standalone runner* described in the locked
"CONTRACT - Terminus 2 Subprocess Bridge".  meta-n's own process must NEVER
import ``terminal_bench`` / ``openhands`` (their pins conflict with meta-n's),
so meta-n shells out to::

    <repo>/.venv_external_agents/bin/python -m t2_runner \
        --request <request.json> --result <result.json>

This file lives inside a meta-n directory (``scripts/``) for convenience, but it
is run ONLY under the external-agents venv interpreter
(``.venv_external_agents/bin/python``) where ``terminal_bench`` 0.2.18 and
``litellm`` are installed.  It imports the agent SDK, drives terminal_bench's OWN
single-task trial machinery (``Harness._run_trial`` -> ``spin_up_terminal`` ->
``TmuxSession`` -> verifier -> reward), and writes a result JSON that meta-n
parses into ``AgentRunResult``.

CRITICAL INVARIANT: ``import terminal_bench`` (and everything that pulls it in)
happens ONLY inside ``main()`` / helper functions, never at module top level, so
that nothing meta-n could conceivably import this module and drag the conflicting
SDK into meta-n's interpreter.  The module top level is deliberately limited to
the stdlib.

SAFETY / TEARDOWN
-----------------
* Terminus 2 executes INSIDE the Docker container that ``spin_up_terminal``
  builds; the host is never touched by agent commands.
* ``spin_up_terminal`` is a context manager whose ``finally`` block calls
  ``Terminal.stop()`` (container teardown) on BOTH normal and exceptional exit.
  We additionally wrap the whole run so that *any* exception still results in a
  valid result JSON (``status="error"``) and we still attempt teardown.
* Bounded by three independent stops: ``max_episodes`` (<= 8), the harness's
  in-process ``asyncio.wait_for(global_agent_timeout_sec)``, and meta-n's outer
  ``wait_for`` -> SIGKILL of this runner's process group.

RESULT JSON
-----------
Written to the ``--result`` path (schema per CONTRACT section 4).  On ANY caught
exception we still write a valid JSON with ``ok=false`` / ``status="error"`` /
``reward=0.0`` and exit 0, so meta-n always has a file to parse.  We exit nonzero
only if we cannot even write the result file.
"""

# ---------------------------------------------------------------------------
# MODULE TOP LEVEL: STDLIB ONLY.
# Never import terminal_bench / litellm here -- only inside functions, so this
# module is import-safe even from meta-n's interpreter (which lacks the SDK).
# ---------------------------------------------------------------------------
import argparse
import contextlib
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

# Shared, stdlib-only runner helpers (single-sourced across the three runners; see
# scripts/_runner_common.py — a top-level module on the child PYTHONPATH). Imported
# at module top so this module stays stdlib-only at module scope (terminal_bench /
# litellm are still imported lazily inside the functions below). Re-exported under
# the historical private names so existing references keep resolving unchanged.
from _runner_common import (  # noqa: E402  (top-level module on the child PYTHONPATH)
    TokenBudgetExceeded,
    classify_runner_error as _classify_runner_error,
    error_result_payload as _error_result_payload,
    failure_mode_str as _failure_mode_str,
    safe_read_text as _safe_read_text,
    sanitize_run_label as _sanitize_run_label_common,
    stage_helper_files as _stage_helper_files_common,
    write_result as _write_result,
)


def _sanitize_run_label(label) -> str:
    """T2 wrapper over the shared ``sanitize_run_label`` (pins the ``t2`` stem).

    Compose project names must match ``[a-z0-9][a-z0-9_-]*``. meta-n passes the
    already-sanitized lease session; this is a defensive pass plus a ``t2-{uuid}``
    fallback so two concurrent runs never share a project (which would collide on
    the globally unique ``container_name``).
    """
    return _sanitize_run_label_common(label, fallback="t2")


# ---------------------------------------------------------------------------
# Constants (the locked routing + endpoint facts from the CONTRACT).
# ---------------------------------------------------------------------------
DEFAULT_API_BASE = "http://127.0.0.1:1234/v1"
# The local LM Studio model is a SLOW reasoning model: without a high max_tokens
# it truncates at finish_reason="length" with empty content, which LiteLLM turns
# into OutputLengthExceededError. We force >= 4096 on every inner call.
DEFAULT_MAX_OUTPUT_TOKENS = 4096
# Hard cap on episodes regardless of what the request asks for (safety bound).
MAX_EPISODES_CEILING = 8
# Fallback trial/compose-project name when meta-n supplies no per-run label.
DEFAULT_RUN_LABEL = "t2-bridge"

# Module-global handle to the live budgeted LLM, so the top-level error path in
# main() can recover partial token spend even when _run_trial raises before the
# AgentResult is captured. Set by _BudgetedLiteLLM.__init__; read defensively.
_ACTIVE_BUDGETED_LLM = None

# Module-global in-process capture of every command Terminus 2 issued into the
# container, mirroring oh_tb_runner's ``_COMMANDS_EXECUTED``. terminal_bench's own
# ``commands.txt`` lives under the DockerRunGuard lease workdir, which meta-n
# reaps (concurrency.py ``shutil.rmtree``) before the runner result is read — so
# the default run otherwise yields a null command_history and attribution reads as
# 0. We capture commands here (via the ``session.send_keys`` wrapper installed by
# ``_install_command_capture``) so attribution survives lease reaping and any
# abort path. Reset at ``_run`` start; read by ``_command_history_strings``.
_COMMANDS_EXECUTED: list[str] = []

# Module-global flag recording whether the in-process command-capture wrapper was
# WIRED (the ``session.send_keys`` wrap installed by ``_install_command_capture``).
# This is the H3 producer signal for ``attribution_available``: it reflects whether
# capture was POSSIBLE on this run, NOT whether any command happened to be
# captured. Emitted in the success result payload; the error payload hard-sets it
# False (a degraded path is UNMEASURABLE). Reset at ``_run`` start.
_CAPTURE_WIRED: bool = False

# H15: pure control-key sends T2 emits between real commands via
# ``session.send_keys`` (e.g. a bare ``Enter`` to submit, ``Tab`` to complete,
# ``C-m`` for newline). These are control-plane keystrokes, not command events, so
# ``_command_history_strings`` drops a send whose every whitespace-split token is a
# control key — otherwise they inflate ``command_count`` (the measured-zero-vs-lost
# discriminator) and pollute attribution.
_CONTROL_ONLY = frozenset(
    {
        "Enter", "Tab", "Escape", "Space", "BSpace",
        "Up", "Down", "Left", "Right",
        "Home", "End", "PageUp", "PageDown",
    }
)
#: A ``C-<char>`` / ``M-<char>`` chord (Ctrl-/Meta-modified single key).
_CONTROL_CHORD_RE = re.compile(r"^[CM]-.$")


def _is_control_only(text: str) -> bool:
    """True iff EVERY whitespace-split token of ``text`` is a pure control send.

    A token is control-only iff it is a named control key (``_CONTROL_ONLY``) or a
    ``C-<char>`` / ``M-<char>`` chord. A string of only such tokens is a keystroke
    control event (e.g. ``"Enter"``, ``"C-c"``), not a command, and is dropped from
    the captured command history. A whitespace-only / empty string is treated as
    control-only (it is filtered upstream anyway).
    """
    toks = text.split()
    if not toks:
        return True
    return all(t in _CONTROL_ONLY or _CONTROL_CHORD_RE.match(t) for t in toks)


# ===========================================================================
# Request loading / normalisation
# ===========================================================================
def _load_request(args: argparse.Namespace) -> dict:
    """Build the request dict from either ``--request <json file>`` (the primary
    CONTRACT invocation) or the discrete named flags
    (``--task-id``/``--instruction-suffix``/``--model-routing``/...).

    The discrete-flag form exists so the runner literally satisfies the flag
    list named at the top of the task; the canonical path meta-n uses is
    ``--request``.  Values from a ``--request`` file are the base; any discrete
    flag that is explicitly provided overrides the corresponding key.
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
        # --model-routing is the litellm-routed model_name, e.g.
        # "openai/google/gemma-4-31b-qat".
        req["model_name"] = args.model_routing
    if args.base_url is not None:
        req["api_base"] = args.base_url
    if args.max_episodes is not None:
        req["max_episodes"] = args.max_episodes
    if args.timeout is not None:
        # --timeout is the agent wall timeout (seconds) for this single task.
        req["agent_timeout_sec"] = args.timeout
    if args.output_dir is not None:
        req["output_dir"] = args.output_dir
    if getattr(args, "run_label", None) is not None:
        req["run_label"] = args.run_label

    # --instruction-suffix points at a FILE whose contents are prepended to the
    # task instruction (the injection "additional_context"). Per Terminus 2's
    # prompt structure there is no suffix slot, so we fold it as a PREFIX.
    if args.instruction_suffix is not None:
        suffix_path = Path(args.instruction_suffix)
        if suffix_path.exists():
            req["additional_context"] = suffix_path.read_text()

    # --staged-files points at a JSON file: {"container/rel/path": "<contents>"}.
    if args.staged_files is not None:
        staged_path = Path(args.staged_files)
        if staged_path.exists():
            req["staged_files"] = json.loads(staged_path.read_text())

    # ---- Defaults for anything still unset --------------------------------
    req.setdefault("api_base", DEFAULT_API_BASE)
    req.setdefault("temperature", 0.7)
    req.setdefault("parser_name", "json")  # agent RESPONSE parser (json|xml)
    req.setdefault("max_output_tokens", DEFAULT_MAX_OUTPUT_TOKENS)
    req.setdefault("additional_context", "")
    req.setdefault("staged_files", {})
    req.setdefault("agent_timeout_sec", 900)
    req.setdefault("test_timeout_sec", 180)
    req.setdefault("no_rebuild", False)

    # Clamp max_episodes to the safety ceiling (<= 8).
    eps = int(req.get("max_episodes", MAX_EPISODES_CEILING) or MAX_EPISODES_CEILING)
    req["max_episodes"] = max(1, min(eps, MAX_EPISODES_CEILING))

    # Force max_output_tokens high enough for the slow local reasoning model.
    req["max_output_tokens"] = max(
        int(req.get("max_output_tokens") or DEFAULT_MAX_OUTPUT_TOKENS),
        DEFAULT_MAX_OUTPUT_TOKENS,
    )

    return req


# ===========================================================================
# Injection classes -- defined lazily inside _build_injected_class() so that the
# terminal_bench imports they subclass live in function scope, never at module
# top level.
# ===========================================================================
def _build_injected_classes(req: dict):
    """Construct and return ``InjectedTerminus2`` (a ``Terminus2`` subclass) wired
    with the request's injection context, max_tokens fold, and file staging.

    Returns the class object; the harness instantiates it via
    ``AgentFactory.get_agent(import_path=..., **agent_kwargs)``.  We stash the
    request on a module-global so the dynamically constructed class can read the
    staging map / additional_context even though the harness controls __init__.
    """
    # Imports are intentionally inside this function (SDK scope only).
    from terminal_bench.agents.terminus_2.terminus_2 import Terminus2
    from terminal_bench.agents.base_agent import AgentResult
    from terminal_bench.llms.lite_llm import LiteLLM

    additional_context = req.get("additional_context", "") or ""
    staged_files = req.get("staged_files", {}) or {}
    max_output_tokens = int(req.get("max_output_tokens", DEFAULT_MAX_OUTPUT_TOKENS))
    api_base = req.get("api_base", DEFAULT_API_BASE)
    temperature = float(req.get("temperature", 0.7))
    token_budget = int(req.get("token_budget", 0) or 0)

    class _BudgetedLiteLLM(LiteLLM):
        """LiteLLM wrapper that (a) forces ``max_tokens`` on every inner call,
        (b) ACCUMULATES per-call prompt/completion tokens so the token basis
        round-trips even on an agent abort (timeout / context / parse failure),
        and (c) enforces an optional cumulative ``token_budget`` kill switch.

        ``LiteLLM.call()`` returns a *str* and discards the response usage, so a
        run that burns real tokens then aborts would otherwise report 0 tokens
        (CostGuard then prices it at $0 — a budget leak for priced backbones).
        We register a litellm ``CustomLogger`` that reads ``response_obj.usage``
        on every successful completion into instance counters
        (``total_in_tokens`` / ``total_out_tokens``), which the runner falls back
        to when the harness/AgentResult totals are 0 (the abort paths). Before
        each call we raise :class:`TokenBudgetExceeded` once cumulative tokens
        exceed ``token_budget`` so over-budget fails soft (failure_mode=
        ``token_budget``) instead of running unbounded.
        """

        def __init__(self, *a, max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
                     token_budget=0, **k):
            super().__init__(*a, **k)
            self._max_out = max_output_tokens
            self._token_budget = int(token_budget or 0)
            self.total_in_tokens = 0
            self.total_out_tokens = 0
            # Real inner LLM-call counter (parity with OH's len(metrics.token_usages)).
            # Counted in ``call()`` per successful completion so the runner can emit
            # a real ``agent_calls`` instead of the prior 0/"unmeasured".
            self.total_calls = 0
            self._install_usage_callback()
            # Publish to the module global so main()'s top-level error path can
            # recover partial token spend if _run_trial raises before capture.
            global _ACTIVE_BUDGETED_LLM
            _ACTIVE_BUDGETED_LLM = self

        def _install_usage_callback(self) -> None:
            """Register a per-instance litellm CustomLogger to harvest usage."""
            try:
                import litellm
                from litellm.integrations.custom_logger import CustomLogger
            except Exception:  # noqa: BLE001 - usage accounting is best-effort
                return

            wrapper = self

            class _UsageCollector(CustomLogger):
                def log_success_event(self, kwargs, response_obj, start_time,
                                      end_time):
                    usage = getattr(response_obj, "usage", None)
                    if usage is None and isinstance(response_obj, dict):
                        usage = response_obj.get("usage")
                    if usage is None:
                        return
                    pt = getattr(usage, "prompt_tokens", None)
                    ct = getattr(usage, "completion_tokens", None)
                    if pt is None and isinstance(usage, dict):
                        pt = usage.get("prompt_tokens")
                        ct = usage.get("completion_tokens")
                    wrapper.total_in_tokens += int(pt or 0)
                    wrapper.total_out_tokens += int(ct or 0)

            with contextlib.suppress(Exception):
                litellm.callbacks.append(_UsageCollector())

        def call(self, *a, **k):
            # In-run token-budget kill switch: once cumulative tokens exceed the
            # budget, fail soft with a stable, NON-retried signal the runner maps
            # to failure_mode='token_budget' (plan §5.5).
            if self._token_budget > 0:
                spent = self.total_in_tokens + self.total_out_tokens
                if spent > self._token_budget:
                    raise TokenBudgetExceeded(
                        f"token budget {self._token_budget} exceeded "
                        f"(spent={spent})"
                    )
            # setdefault: respect an explicit max_tokens if a caller ever sets
            # one, otherwise pin our high floor.
            k.setdefault("max_tokens", self._max_out)
            result = super().call(*a, **k)
            # Count only COMPLETED inner LLM calls (after super().call returns):
            # a budget kill-switch raise above, or a transport exception inside,
            # never reaches here, so this is the real number of inner completions
            # — the T2 analogue of OH's len(metrics.token_usages).
            self.total_calls += 1
            return result

    class InjectedTerminus2(Terminus2):
        """Terminus 2 with (a) max_tokens fold, (b) instruction injection, and
        (c) helper-file staging into the container.

        At depth 1 (``additional_context == ""`` and no staged files) this is
        byte-identical to vanilla Terminus2 except for the mandatory high
        max_tokens -- which the local model requires regardless.

        ``captured_result`` is a class attribute the runner reads after the trial
        to recover the AgentResult totals / failure_mode even though the harness
        owns the agent instance.
        """

        captured_result = None  # type: ignore[var-annotated]  # set per-run
        # The budgeted LLM is stashed here so the runner can read its accumulated
        # token counters even on an agent abort (where AgentResult never returns).
        budgeted_llm = None  # type: ignore[var-annotated]

        def __init__(self, *a, **k):
            # The harness passes model_name / api_base / temperature / max_episodes
            # / parser_name via agent_kwargs. Strip our injection-only kwargs if
            # they were routed through agent_kwargs (defensive; we read from the
            # closure too).
            self._inj_additional_context = k.pop("additional_context", additional_context)
            self._inj_staged_files = k.pop("staged_files", staged_files)
            self._inj_max_output_tokens = k.pop("max_output_tokens", max_output_tokens)
            super().__init__(*a, **k)
            # Replace the inner LLM with the budgeted variant. We reuse the
            # model_name the base class stored and the api_base/temperature
            # captured at build time (the base class does not retain api_base).
            self._llm = _BudgetedLiteLLM(
                model_name=self._model_name,
                api_base=api_base,
                temperature=temperature,
                max_output_tokens=self._inj_max_output_tokens,
                token_budget=token_budget,
            )
            # Expose the budgeted LLM so the runner can recover partial token
            # spend on abort paths (where the harness/AgentResult totals are 0).
            InjectedTerminus2.budgeted_llm = self._llm

        def _stage_helper_files(self, session) -> None:
            """Stage the injection's helper files into the container (shared guard).

            Delegates to the single-sourced, security-load-bearing
            ``_runner_common.stage_helper_files`` (path-traversal validation +
            ``/app/<rel>`` copy — keys are workspace-relative and already carry
            the ``helpers/`` prefix). Best-effort: a staging failure is logged
            to stderr but never aborts the run (the agent may not need the helper)."""
            _stage_helper_files_common(
                session, self._inj_staged_files, prefix="t2"
            )

        def perform_task(self, instruction, session, logging_dir=None,
                         time_limit_seconds=None):
            # 1) Stage any helper files BEFORE the agent loop starts.
            self._stage_helper_files(session)

            # 1b) Wrap session.send_keys so every command T2 issues is captured
            #     in-process (survives DockerRunGuard lease reaping; F2).
            _install_command_capture(session)

            # 2) Injection fold: additional_context FIRST, raw task LAST (Terminus
            #    2 has no dedicated suffix slot, so the injection is a prefix). At
            #    depth 1 additional_context == "" -> composed == instruction.
            parts = [p for p in (self._inj_additional_context, instruction) if p]
            composed = "\n\n".join(parts)

            res: "AgentResult" = super().perform_task(
                composed, session, logging_dir, time_limit_seconds
            )
            # 3) Capture the AgentResult so the runner can read authoritative
            #    token totals / failure_mode after _run_trial returns.
            InjectedTerminus2.captured_result = res
            return res

    return InjectedTerminus2


# ===========================================================================
# In-process command capture (mirrors oh_tb_runner; F2)
# ===========================================================================
def _install_command_capture(session) -> None:
    """Wrap the live ``session.send_keys`` so every command T2 issues is captured.

    terminal_bench's Terminus 2 agent drives the container by calling
    ``session.send_keys(command.keystrokes, ...)`` (terminus_2.py); that single
    chokepoint also writes ``commands.txt`` — but ONLY under the DockerRunGuard
    lease workdir, which meta-n reaps before the runner's result is read. We
    therefore append each non-empty keystroke string to the module-global
    ``_COMMANDS_EXECUTED`` here, so attribution survives lease reaping.

    The wrapper captures the ORIGINAL bound method, guards re-entry with a
    ``_metan_wrapped`` attribute (so a re-entered ``perform_task`` does not double-
    wrap / double-capture), appends the keystrokes (``str``; joined if a list),
    then returns ``orig(keys, *a, **k)`` UNCHANGED — zero behavioral change to the
    agent or to terminal_bench's own ``commands.txt`` writer. Best-effort: a
    wrap failure is swallowed (capture is diagnostic, never load-bearing).
    """
    try:
        orig = session.send_keys
        if getattr(orig, "_metan_wrapped", False):
            return

        def _wrapped(keys, *a, **k):
            try:
                text = (
                    " ".join(str(x) for x in keys)
                    if isinstance(keys, (list, tuple))
                    else str(keys)
                )
                if text.strip():
                    _COMMANDS_EXECUTED.append(text)
            except Exception:  # noqa: BLE001 - capture must never sink a command
                pass
            return orig(keys, *a, **k)

        _wrapped._metan_wrapped = True  # type: ignore[attr-defined]
        session.send_keys = _wrapped  # type: ignore[method-assign]
        # H3 producer: capture is now WIRED for this run — record it so the
        # success payload advertises attribution_available=True (capture POSSIBLE)
        # regardless of how many commands are ultimately captured.
        global _CAPTURE_WIRED
        _CAPTURE_WIRED = True
    except Exception:  # noqa: BLE001 - capture is best-effort, never aborts a run
        pass


def _command_history_strings() -> list[str]:
    """Flatten the module-level executed-command log into the ``command_history``
    string list the result schema (and meta-n's attribution) expects.

    Each retained entry is a real command keystroke string T2 sent into the
    container, in order. H15: pure control-key sends (``Enter`` / ``Tab`` /
    ``C-m`` / …) are filtered out via :func:`_is_control_only`, so
    ``command_count`` is a count of command events — not inflated by the
    control-plane keystrokes T2 emits between commands."""
    return [
        c for c in _COMMANDS_EXECUTED
        if c and c.strip() and not _is_control_only(c)
    ]


# ===========================================================================
# Result-JSON helpers
# ===========================================================================
# ``_safe_read_text`` / ``_failure_mode_str`` / ``_write_result`` are imported
# from ``_runner_common`` at module top (single-sourced across the runners).


def _count_episode_dirs(agent_logging_dir) -> int:
    """Approximate step/turn count from the number of ``episode-*`` dirs Terminus
    2 writes under the agent logging dir.  NOTE: this is an approximation of turns
    and is NOT equal to the number of LLM calls."""
    try:
        d = Path(agent_logging_dir)
        if not d.exists():
            return 0
        return sum(1 for c in d.iterdir() if c.is_dir() and c.name.startswith("episode-"))
    except Exception:
        return 0


def _error_payload(task_id: str, failure_mode: str, error: str,
                   total_input_tokens: int = 0,
                   total_output_tokens: int = 0,
                   agent_calls: int = 0,
                   command_history: list | None = None) -> dict:
    """Build a valid result JSON for the error path (CONTRACT section 4).

    Delegates to the single-sourced ``_runner_common.error_result_payload`` so the
    error-schema producer cannot drift from what ``_external_tb._to_run_result``
    reads. ``total_*_tokens`` carry any partial inner-token spend recovered from
    the budgeted LLM's per-call accumulators, so a run that burned real tokens then
    aborted is priced correctly (not at $0). ``agent_calls`` likewise carries the
    partial inner-call count recovered from the budgeted LLM. ``command_history``
    carries any commands T2 already issued before the abort so attribution
    survives the error path (F2).

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
    return payload


# ===========================================================================
# The actual single-task run (drives terminal_bench's own machinery)
# ===========================================================================
def _run(req: dict, result_path: str) -> int:
    """Execute one task end-to-end via ``Harness._run_trial`` and write the result
    JSON.  Returns a process exit code (0 on completion incl. scored failure)."""
    task_id = req.get("task_id", "")
    started = time.time()

    # SDK imports -- function scope ONLY.
    from terminal_bench.harness.harness import Harness
    from terminal_bench.handlers.trial_handler import TrialHandler

    tasks_dir = req["tasks_dir"]
    model_name = req["model_name"]
    api_base = req.get("api_base", DEFAULT_API_BASE)
    temperature = float(req.get("temperature", 0.7))
    max_episodes = int(req["max_episodes"])
    parser_name = req.get("parser_name", "json")
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
    InjectedTerminus2 = _build_injected_classes(req)
    # Reset the per-process captured result (the class is freshly built each run,
    # but be explicit in case of re-entry).
    InjectedTerminus2.captured_result = None
    # Reset the in-process command capture for this run (re-entry safety; the
    # runner is normally one task per subprocess but be explicit). F2.
    _COMMANDS_EXECUTED.clear()
    # H3 producer: reset the capture-wired flag; it flips True inside
    # ``_install_command_capture`` once the send_keys wrapper is installed.
    global _CAPTURE_WIRED
    _CAPTURE_WIRED = False

    # Register the class under this module's name so AgentFactory can import it by
    # "module:class". When run as `python -m t2_runner`, this module is "__main__";
    # we register both names so import_path resolution works regardless.
    import_module_name = __name__ if __name__ != "__main__" else "t2_runner"
    sys.modules.setdefault("t2_runner", sys.modules[__name__])
    # Make sure the chosen class name resolves on whichever module object the
    # factory imports.
    setattr(sys.modules["t2_runner"], "InjectedTerminus2", InjectedTerminus2)
    if __name__ in sys.modules:
        setattr(sys.modules[__name__], "InjectedTerminus2", InjectedTerminus2)
    agent_import_path = f"{import_module_name}:InjectedTerminus2"

    # Per-run-unique trial identity: meta-n passes the lease session as
    # ``run_label`` so the compose project (== trial_name, which DockerComposeManager
    # passes verbatim to ``docker compose -p`` and which each task's
    # ``container_name: ${...CLIENT_CONTAINER_NAME}`` resolves to) is GLOBALLY unique.
    # The prior hardcoded "t2-bridge" collided under ``--parallel`` (Docker requires
    # container_name to be globally unique) and let one run's timeout sweep kill a
    # sibling's live container. Fall back to a uuid when no label is supplied.
    run_label = _sanitize_run_label(req.get("run_label"))
    run_id = run_label
    # Build the Harness for this single task. model_name is passed BOTH as
    # Harness(model_name=...) (used only for metadata/lock) AND inside
    # agent_kwargs, because the harness does NOT forward model_name to the agent.
    #
    # dataset_path is REQUIRED: Harness.__init__ eagerly builds a Dataset, and
    # without a path (or a name+version) DatasetConfig validation raises. We point
    # it at the local tasks dir (the parent of <task_id>/) and scope it to the one
    # task via task_ids, so no remote registry is touched.
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
            "max_episodes": max_episodes,
            "parser_name": parser_name,
            # Injection-only kwargs (read by InjectedTerminus2.__init__):
            "additional_context": req.get("additional_context", ""),
            "staged_files": req.get("staged_files", {}),
            "max_output_tokens": req.get("max_output_tokens", DEFAULT_MAX_OUTPUT_TOKENS),
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

    # The harness writes trial artifacts under output_path/run_id/...; TrialHandler
    # must use that same path so post-test/post-agent panes line up.
    trial = TrialHandler(
        trial_name=run_label,
        input_path=task_input_path,
        output_path=output_path / run_id,
    )

    # ---- Drive the single-task trial. spin_up_terminal's finally guarantees
    # ---- Terminal.stop() (container teardown) even if _run_trial raises. ----
    results = harness._run_trial(trial)

    # ---- Map TrialResults + captured AgentResult -> result JSON --------------
    captured = InjectedTerminus2.captured_result

    is_resolved = bool(getattr(results, "is_resolved", False))
    reward = 1.0 if is_resolved else 0.0

    # Token totals: results.total_*_tokens are set from the AgentResult on success
    # (and 0 on agent abort). Fall back to the captured AgentResult, then to the
    # budgeted LLM's per-call accumulators (the ONLY source that survives an agent
    # abort: timeout / context / parse failure never return an AgentResult, so
    # without this the run reports 0 tokens and CostGuard prices it at $0).
    total_in = getattr(results, "total_input_tokens", 0) or 0
    total_out = getattr(results, "total_output_tokens", 0) or 0
    if (not total_in and not total_out) and captured is not None:
        total_in = getattr(captured, "total_input_tokens", 0) or 0
        total_out = getattr(captured, "total_output_tokens", 0) or 0
    if not total_in and not total_out:
        budgeted = getattr(InjectedTerminus2, "budgeted_llm", None)
        if budgeted is not None:
            total_in = getattr(budgeted, "total_in_tokens", 0) or 0
            total_out = getattr(budgeted, "total_out_tokens", 0) or 0

    # Real inner LLM-call count (plan: T2 must report a real inner_calls like OH,
    # not 0/"unmeasured"). The budgeted LLM tallies one per COMPLETED inner call;
    # read it off the class-level handle (survives an agent abort, like tokens).
    agent_calls = 0
    budgeted = getattr(InjectedTerminus2, "budgeted_llm", None)
    if budgeted is not None:
        agent_calls = int(getattr(budgeted, "total_calls", 0) or 0)

    failure_mode = _failure_mode_str(getattr(results, "failure_mode", None))

    # parser_results: TrialResults stores {test_name: UnitTestStatus}. Serialise to
    # {name: status_string}.
    parser_results = {}
    raw_parser = getattr(results, "parser_results", None) or {}
    try:
        for k, v in raw_parser.items():
            parser_results[str(k)] = getattr(v, "value", str(v))
    except Exception:
        parser_results = {}

    # timestamped_markers from the captured AgentResult (list of [ts, text]).
    timestamped_markers = []
    if captured is not None:
        try:
            timestamped_markers = [list(m) for m in getattr(captured, "timestamped_markers", [])]
        except Exception:
            timestamped_markers = []

    # Artifact paths the harness already wrote.
    tp = trial.trial_paths
    transcript_path = str(tp.agent_logging_dir)
    post_agent_pane = _safe_read_text(tp.post_agent_pane_path)
    post_test_pane = _safe_read_text(tp.post_test_pane_path)
    # command_history: prefer the in-process capture (the send_keys wrapper),
    # which survives the DockerRunGuard lease reaping that deletes commands.txt.
    # Fall back to the harness's commands.txt only when our capture is empty (e.g.
    # a path we did not intercept). Mirrors oh_tb_runner's ordering. F2.
    command_history = _command_history_strings()
    if not command_history:
        cmds = _safe_read_text(tp.commands_path)
        if cmds:
            command_history = [ln for ln in cmds.splitlines() if ln.strip()]

    steps = _count_episode_dirs(tp.agent_logging_dir)

    payload = {
        "ok": True,
        "status": "ok",
        "reward": reward,
        "score": reward,
        "is_resolved": is_resolved,
        "task_id": task_id,
        "total_input_tokens": int(total_in),
        "total_output_tokens": int(total_out),
        # Real inner LLM-call count from the budgeted LiteLLM wrapper (one per
        # COMPLETED inner ``call()``). This is the T2 analogue of OH's
        # ``len(metrics.token_usages)`` — NOT the episode count (do not equate the
        # two: an episode can issue 0 or >1 LLM calls).
        "agent_calls": int(agent_calls),
        "failure_mode": failure_mode,
        "parser_results": parser_results,
        "timestamped_markers": timestamped_markers,
        "command_history": command_history,
        # H3 producer: was in-process command capture WIRED for this run? True iff
        # the send_keys wrapper was installed (NOT whether any command was seen) —
        # so an empty command_history with this True is a MEASURED-none, while
        # command_count==0 with this True flags a LOST stream (the discriminator
        # _external_tb/telemetry consume).
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
# Entry point
# ===========================================================================
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="t2_runner",
        description="Standalone Terminus 2 subprocess bridge for terminal_bench.",
    )
    # Canonical meta-n invocation (CONTRACT section 2): request + result JSON.
    parser.add_argument("--request", help="Path to the request JSON written by meta-n.")
    parser.add_argument("--result", "--result-json", dest="result",
                        help="Path to write the result JSON.")
    # Discrete-flag entry form (the named flags from the task header).
    parser.add_argument("--task-id", dest="task_id",
                        help="Terminal-bench task id (e.g. hello-world).")
    parser.add_argument("--tasks-dir", dest="tasks_dir",
                        help="Directory containing the task folders.")
    parser.add_argument("--instruction-suffix", dest="instruction_suffix",
                        help="Path to a file whose contents are prepended to the "
                             "task instruction (the injection context).")
    parser.add_argument("--model-routing", dest="model_routing",
                        help="litellm-routed model_name, e.g. "
                             "openai/google/gemma-4-31b-qat.")
    parser.add_argument("--base-url", dest="base_url",
                        help="OpenAI-compatible api_base, e.g. "
                             "http://127.0.0.1:1234/v1.")
    parser.add_argument("--max-episodes", dest="max_episodes", type=int,
                        help="Max Terminus 2 episodes (clamped to <= 8).")
    parser.add_argument("--timeout", dest="timeout", type=float,
                        help="Agent wall timeout in seconds for this task.")
    parser.add_argument("--staged-files", dest="staged_files",
                        help="Path to a JSON file mapping container-relative paths "
                             "to file contents to stage into the container.")
    parser.add_argument("--output-dir", dest="output_dir",
                        help="Directory for harness logs/artifacts.")
    parser.add_argument("--run-label", dest="run_label",
                        help="Per-run-unique compose project / trial name "
                             "(meta-n passes the lease session); a uuid is used "
                             "when absent. Prevents container-name collisions under "
                             "concurrent trials.")
    args = parser.parse_args(argv)

    # --result is mandatory: it is the file meta-n reads.
    if not args.result:
        # We cannot write a result JSON without a destination; this is the one
        # case where we exit nonzero.
        print("[t2_runner] FATAL: --result/--result-json is required.",
              file=sys.stderr)
        return 2

    result_path = args.result

    # Always set the inner LLM auth env (LM Studio accepts any key). We never
    # overwrite an OPENAI_API_KEY meta-n may have passed deliberately, but the
    # CONTRACT specifies a dummy so litellm's openai/ provider is satisfied.
    os.environ.setdefault("OPENAI_API_KEY", "dummy")

    task_id = ""
    try:
        req = _load_request(args)
        task_id = req.get("task_id", "")
        # Validate the minimal required fields up front so a missing field
        # produces a clean error JSON rather than a deep traceback.
        for field in ("task_id", "tasks_dir", "model_name", "output_dir"):
            if not req.get(field):
                raise ValueError(f"request is missing required field: {field!r}")
        return _run(req, result_path)
    except Exception as exc:  # noqa: BLE001 - top-level guard: ALWAYS emit JSON
        # Classify the failure mode best-effort. We default to a generic
        # env/agent error; meta-n maps unknown modes to AGENT_ERROR / ENV_ERROR.
        tb = traceback.format_exc()
        print(f"[t2_runner] ERROR: {exc}\n{tb}", file=sys.stderr)
        failure_mode = _classify_runner_error(exc)
        # Recover any partial inner-token spend AND inner-call count from the
        # budgeted LLM so an abort that burned real tokens is neither priced at $0
        # nor reported with 0 inner calls.
        part_in = part_out = part_calls = 0
        if _ACTIVE_BUDGETED_LLM is not None:
            part_in = getattr(_ACTIVE_BUDGETED_LLM, "total_in_tokens", 0) or 0
            part_out = getattr(_ACTIVE_BUDGETED_LLM, "total_out_tokens", 0) or 0
            part_calls = getattr(_ACTIVE_BUDGETED_LLM, "total_calls", 0) or 0
        try:
            _write_result(
                result_path,
                _error_payload(
                    task_id, failure_mode, tb, part_in, part_out, part_calls,
                    _command_history_strings(),
                ),
            )
        except Exception as werr:  # pragma: no cover - cannot even write result
            print(f"[t2_runner] FATAL: could not write result JSON: {werr}",
                  file=sys.stderr)
            return 1
        # Exit 0 on a caught error: meta-n has a valid JSON to parse.
        return 0


if __name__ == "__main__":
    sys.exit(main())
