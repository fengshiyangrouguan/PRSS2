#!/usr/bin/env python
"""Standalone OpenHands runner — the subprocess half of the OH subprocess bridge.

This script runs **only** under
``<repo>/.venv_external_agents/bin/python`` and is
*never imported by meta-n* (whose env has no ``openhands``). meta-n invokes it by
path (``OpenHandsBackend.run``), it imports the OpenHands SDK in-process, drives a
single :class:`LocalConversation`, and writes one result-JSON to ``--result-file``
that meta-n reads back and folds into an ``AgentRunResult``.

Contract: it ALWAYS writes a valid result JSON (with ``error`` set on any failure)
and never exits non-zero without first writing that JSON, so meta-n always has
something to parse. The result-JSON schema is the locked bridge contract:

    {
      "status": "finished",            # ConversationExecutionStatus.value (lowercase)
      "last_message": "...",           # last source=='agent' MessageEvent text
      "command_history": [{"command": "...", "output": "..."}],
      "prompt_tokens": 0,
      "completion_tokens": 0,
      "cache_read_tokens": 0,           # optional; meta-n reads defensively
      "accumulated_cost_usd": 0.0,      # native_usd basis (0.0 for local/dummy)
      "model": "google/gemma-4-31b-qat",
      "solution_file_contents": "...",  # contents of {workspace}/{solution-file}
      "error": null
    }

Run path (in-process, inside this subprocess):
  1. LLM (LM Studio local; high max_output_tokens for the slow reasoning model)
  2. minimal tools: terminal + file_editor ONLY (registrar runs first)
  3. AgentContext(system_message_suffix=...) injection slot
  4. Agent(llm, tools, agent_context)
  5. Conversation(agent, workspace=..., max_iteration_per_run<=8, stuck_detection=True)
  6. conv.send_message(instruction); conv.run()  (run() takes NO message)
  7. harvest off conv.state; 8. write result JSON atomically (temp + os.replace)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

# Shared, stdlib-only runner helpers (single-sourced across the three runners; see
# scripts/_runner_common.py). Imported at module top so the runners stay
# stdlib-only at module scope (the SDK is still imported lazily inside ``_run``).
# Re-exported under the historical private names so existing references — incl.
# the unit suite's ``oh_runner.TokenBudgetExceeded`` /
# ``oh_runner._install_token_budget_killswitch`` — keep resolving to the SAME
# objects (the token-budget tests rely on the class identity).
from _runner_common import (  # noqa: E402  (top-level module on the child PYTHONPATH)
    KNOWN_LITELLM_PROVIDERS as _KNOWN_LITELLM_PROVIDERS,
    TokenBudgetExceeded,
    accumulated_inner_tokens as _accumulated_inner_tokens,
    install_token_budget_killswitch as _install_token_budget_killswitch,
    is_token_budget_error as _is_token_budget_error,
    route_model as _route_model,
)

# Keep the SDK banner off stdout so nothing pollutes a parse of this process.
os.environ.setdefault("OPENHANDS_SUPPRESS_BANNER", "1")


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Write ``payload`` to ``path`` atomically (temp file + ``os.replace``)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False))
    os.replace(tmp, path)


def _read_text_or_empty(path: Path | None) -> str:
    """Return the contents of ``path``, or ``""`` if missing/unreadable."""
    if path is None:
        return ""
    try:
        return path.read_text()
    except OSError:
        return ""


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OpenHands subprocess-bridge runner")
    p.add_argument("--workspace", required=True, help="lease workspace dir")
    p.add_argument("--instruction-file", required=True, help="composed task text")
    p.add_argument("--system-suffix-file", default=None,
                   help="AgentContext.system_message_suffix contents")
    p.add_argument("--result-file", required=True, help="where to write result JSON")
    p.add_argument("--model", default="google/gemma-4-31b-qat")
    p.add_argument("--base-url", default="http://127.0.0.1:1234/v1")
    p.add_argument("--api-key", default="dummy",
                   help="provider key; '-' reads env OH_API_KEY to avoid argv leak")
    p.add_argument("--max-output-tokens", type=int, default=4096)
    p.add_argument("--max-iterations", type=int, default=8)
    p.add_argument("--max-budget-usd", type=float, default=0.0,
                   help="per-run USD ceiling handed to the SDK as a hard stop "
                        "(LLM.max_budget_per_task); 0/absent disables the cap")
    p.add_argument("--token-budget", type=int, default=0,
                   help="per-run cumulative INNER-token ceiling (prompt + "
                        "completion across all of this run's LLM calls). When "
                        ">0 a per-call kill switch raises before the call that "
                        "would run while already over budget, ending the run as "
                        "status='token_budget'. 0/absent disables the cap.")
    p.add_argument("--solution-file", default="solve.py",
                   help="workspace-relative authored target to round-trip")
    return p.parse_args(argv)


def _last_agent_message(events) -> str:
    """Join the ``TextContent.text`` of the last source=='agent' MessageEvent."""
    from openhands.sdk.event import MessageEvent

    last = ""
    for e in events:
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
    return last


#: ``ConversationErrorEvent.code`` the SDK stamps when ``conv.run()`` hits the
#: ``max_iteration_per_run`` ceiling *without* the agent finishing (it sets
#: ``execution_status = ERROR`` and emits this event rather than raising). This is
#: the ONE "error" status that is not a genuine failure: it is the iteration cap,
#: so meta-n must telemeter it as ``MAX_TURNS``, not ``AGENT_ERROR``.
_MAX_ITERATIONS_ERROR_CODE = "MaxIterationsReached"

#: ``ConversationErrorEvent.code`` the SDK stamps when our token-budget kill
#: switch raises inside ``conv.run()``. The SDK sets ``code=e.__class__.__name__``
#: (see ``LocalConversation.run``'s ``except`` block), so this is exactly the
#: kill-switch exception's class name. The runner recognizes it to rewrite the
#: overloaded ``status="error"`` into the SOFT ``status="token_budget"``.
_TOKEN_BUDGET_ERROR_CODE = TokenBudgetExceeded.__name__


def _last_error_event(events) -> tuple[str, str]:
    """Return ``(code, detail)`` of the last ``ConversationErrorEvent``, or ``("","")``.

    The SDK's ``LocalConversation.run()`` records a ``ConversationErrorEvent`` for
    both the max-iterations cap (``code="MaxIterationsReached"``) and genuine
    in-step exceptions (``code=<ExceptionClassName>``). Surfacing the code lets the
    runner distinguish "hit the iteration ceiling" (a benign ``MAX_TURNS`` stop)
    from a real failure, so the two are not conflated into ``agent_error``.
    """
    # ``ConversationErrorEvent`` is not re-exported from ``openhands.sdk.event``
    # top-level from ``openhands.sdk.event`` (as of 1.28.0–1.31.0); it lives in the ``conversation_error``
    # submodule. Import it defensively (falling back to top-level) so a future SDK
    # that re-exports it still works.
    try:
        from openhands.sdk.event.conversation_error import ConversationErrorEvent
    except Exception:  # noqa: BLE001 - tolerate an SDK that moves/exports it
        try:
            from openhands.sdk.event import ConversationErrorEvent  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001 - no error-event type: no error to read
            return "", ""

    code, detail = "", ""
    for e in events:
        if isinstance(e, ConversationErrorEvent):
            code = str(getattr(e, "code", "") or "")
            detail = str(getattr(e, "detail", "") or "")
    return code, detail


def _observation_index(events) -> dict[str, str]:
    """Map ``ObservationEvent.action_id`` -> rendered observation text."""
    from openhands.sdk.event import ObservationEvent

    out: dict[str, str] = {}
    for e in events:
        if isinstance(e, ObservationEvent):
            action_id = getattr(e, "action_id", None)
            if action_id is None:
                continue
            obs = getattr(e, "observation", None)
            text = ""
            if obs is not None:
                # Terminal/file observations expose a 'content' str; fall back
                # to str() of the whole observation object otherwise.
                content = getattr(obs, "content", None)
                text = content if isinstance(content, str) else str(obs)
            out[str(action_id)] = text
    return out


def _command_history(events) -> list[dict]:
    """Ordered agent actions paired with their observation output.

    Captures BOTH ``TerminalAction`` (shell commands) and ``FileEditorAction``
    (the file_editor tool the CO-Bench authoring task uses to write ``solve.py``).
    A pure-edit authoring run shells out nothing, so without the file-editor rows
    ``steps`` would under-report real agent work and attribution would mark the
    run 'unmeasurable' even though the agent acted. File-editor rows render a
    synthetic command ``file_editor <command> <path>`` so the command stream
    reflects the edits.
    """
    from openhands.sdk.event import ActionEvent
    from openhands.tools.terminal import TerminalAction
    try:
        from openhands.tools.file_editor import FileEditorAction
    except Exception:  # noqa: BLE001 - file_editor optional; degrade gracefully
        FileEditorAction = None  # type: ignore[assignment]

    obs_by_action = _observation_index(events)
    cmds: list[dict] = []
    for e in events:
        if not isinstance(e, ActionEvent):
            continue
        action = getattr(e, "action", None)
        command: str | None = None
        if isinstance(action, TerminalAction):
            command = str(getattr(action, "command", "") or "")
        elif FileEditorAction is not None and isinstance(action, FileEditorAction):
            sub = str(getattr(action, "command", "") or "")
            path = str(getattr(action, "path", "") or "")
            command = f"file_editor {sub} {path}".strip()
        if command is None:
            continue
        output = obs_by_action.get(str(getattr(e, "id", "")), "")
        cmds.append({"command": command, "output": str(output)})
    return cmds


#: litellm provider prefixes that already pin a routing target — when the model
#: id starts with one of these, no 'openai/' prefix is added. Kept as the UNION
#: of this runner's set and terminal_bench's ``_LITELLM_PROVIDER_PREFIXES`` so OH
#: and T2 route a given model id identically (audit reuse-simplify): ``cohere/``
#: is included here (it was T2-only) and ``text-completion-openai/`` stays
#: (it was OH-only).
def _run(args: argparse.Namespace) -> dict:
    """Drive one OpenHands conversation and harvest a result dict.

    Any exception propagates to ``main`` which folds it into a valid
    ``status="error"`` result JSON (best-effort solution read included).
    """
    from openhands.sdk import LLM, Agent, AgentContext, Conversation
    from openhands.tools.preset.default import register_default_tools
    from openhands.sdk.tool.spec import Tool
    from openhands.tools.terminal import TerminalTool
    from openhands.tools.file_editor import FileEditorTool

    workspace = Path(args.workspace)
    instruction = _read_text_or_empty(Path(args.instruction_file))
    suffix = _read_text_or_empty(
        Path(args.system_suffix_file) if args.system_suffix_file else None
    )

    # Resolve api key: '-' means read OH_API_KEY (avoids leaking it in ps/argv).
    api_key = args.api_key
    if api_key == "-":
        api_key = os.environ.get("OH_API_KEY", "dummy")

    # 1. LLM — high max_output_tokens for the SLOW local reasoning model.
    #
    # litellm routing: a custom OpenAI-compatible base_url (LM Studio) needs an
    # explicit provider prefix or litellm raises "LLM Provider NOT provided" (it
    # would otherwise read 'google/...' as the Google provider). When a base_url
    # is set and the model id carries no recognized litellm provider prefix,
    # prepend 'openai/' so the call routes to the OpenAI-compatible endpoint.
    # The contracted model id 'google/gemma-4-31b-qat' thus becomes
    # 'openai/google/gemma-4-31b-qat'. Local/dummy spend stays 0.0 (the model is
    # unmapped in litellm's price table -> native_usd basis, no PRICING entry).
    model = _route_model(args.model, args.base_url)
    llm = LLM(
        model=model,
        base_url=args.base_url,
        api_key=api_key,
        max_output_tokens=max(int(args.max_output_tokens), 4096),
    )

    # Per-run USD budget ceiling (plan §4.6). meta-n stamps
    # ``max_budget_per_task`` on the LLM's Metrics but does NOT rely on the SDK
    # enforcing it inside the conversation loop (recorded-but-unenforced as of the
    # verified 1.28.0 SDK; no accumulated_cost comparison) — the over-budget
    # contract is enforced by the spine's CostGuard pre-check + the iteration /
    # wall-clock caps. We still stamp the value so it round-trips and will become
    # a hard stop automatically if the SDK enforces it.
    budget = float(getattr(args, "max_budget_usd", 0.0) or 0.0)
    if budget > 0.0:
        try:
            llm.metrics.max_budget_per_task = budget
        except Exception:  # noqa: BLE001 - budget stamp must never abort the run
            pass

    # Per-run cumulative INNER-token kill switch (the OH↔T2 fairness fix). Unlike
    # ``max_budget_per_task`` above — which the SDK stores but meta-n does not rely on — this
    # installs an actual in-loop stop: a per-call hook that raises once cumulative
    # prompt+completion tokens exceed ``--token-budget`` (see
    # ``_install_token_budget_killswitch``). 0/absent disables it.
    token_budget = int(getattr(args, "token_budget", 0) or 0)
    _install_token_budget_killswitch(llm, token_budget)

    # 2. minimal tools: terminal + file editor ONLY (registrar MUST run first).
    register_default_tools(enable_browser=False)
    tools = [Tool(name=TerminalTool.name), Tool(name=FileEditorTool.name)]

    # 3. injection slot
    ctx = AgentContext(system_message_suffix=suffix) if suffix else None

    # 4. agent (no browser / no sub-agents — the tools list already excludes them)
    agent = Agent(llm=llm, tools=tools, agent_context=ctx)

    # 5. conversation rooted in the isolated scratch workspace; cap iterations <= 8.
    conv = Conversation(
        agent,
        workspace=str(workspace),
        max_iteration_per_run=min(int(args.max_iterations), 8),
        stuck_detection=True,
        # Disable the Rich conversation visualizer so the SDK writes nothing to
        # stdout — meta-n parses the result from --result-file (falling back to
        # the stdout payload main() emits when that write fails), so a clean
        # stdout keeps the fallback parse unambiguous.
        visualizer=None,
    )

    #: Set True when our token-budget kill switch fired during ``conv.run()`` so
    #: the harvest below rewrites the SDK's overloaded ERROR into the SOFT
    #: ``status="token_budget"``. The state is still fully harvestable: the SDK
    #: updates ``execution_status`` / events / metrics BEFORE re-raising
    #: ``ConversationRunError``.
    token_budget_hit = False
    try:
        # 6. drive — run() takes NO message; send first, then run synchronously.
        conv.send_message(instruction)
        try:
            conv.run()
        except BaseException as run_exc:  # noqa: BLE001 - catch the kill switch only
            # Re-raise everything EXCEPT our token-budget kill switch, which the
            # SDK wraps in ``ConversationRunError`` (with ``__cause__`` set to the
            # original ``TokenBudgetExceeded``). On that one signal we fall through
            # to harvest the partial state instead of failing the whole run.
            if not _is_token_budget_error(run_exc):
                raise
            token_budget_hit = True

        # 7. harvest off conv.state
        st = conv.state
        status = st.execution_status.value
        metrics = st.stats.get_combined_metrics()
        usage = getattr(metrics, "accumulated_token_usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        # cache_read_tokens is optional per the contract; meta-n reads it defensively.
        cache_read_tokens = int(getattr(usage, "cache_read_tokens", 0) or 0)
        cost = float(getattr(metrics, "accumulated_cost", 0.0) or 0.0)
        # Real LLM call count: Metrics.token_usages is a per-call list, so its
        # length is the number of inner LLM calls (not separately exposed as a
        # scalar). Fall back to len(metrics.costs) then 0.
        try:
            token_usages = getattr(metrics, "token_usages", None) or []
            agent_calls = len(token_usages)
            if not agent_calls:
                agent_calls = len(getattr(metrics, "costs", None) or [])
        except Exception:  # noqa: BLE001 - call-count read must never abort harvest
            agent_calls = 0
        events = list(st.events)
        cmds = _command_history(events)
        last = _last_agent_message(events)
        error_code, error_detail = _last_error_event(events)
    finally:
        try:
            conv.close()
        except Exception:  # noqa: BLE001 — close failures must not mask results
            pass

    # Disambiguate the SDK's overloaded ERROR status. ``conv.run()`` returns
    # *normally* (no exception) when it merely hit ``max_iteration_per_run`` — it
    # sets ``execution_status = ERROR`` and stamps a ``MaxIterationsReached``
    # ConversationErrorEvent. That is the iteration cap, NOT a failure, so report
    # it as ``max_iterations`` (→ ``MAX_TURNS``) instead of the raw ``error`` token
    # (→ ``AGENT_ERROR``). A genuine in-step exception re-raises out of
    # ``conv.run()`` and is folded into ``status="error"`` by ``main()``; any other
    # ERROR/STUCK status here is a real failure and keeps its native token.
    #
    # The token-budget kill switch is the second disambiguation: it ALSO surfaces
    # as ``execution_status = ERROR`` (with ``code="TokenBudgetExceeded"``), but it
    # is a SOFT envelope stop (like max-iterations), so we rewrite it to the
    # ``token_budget`` token (→ ``TerminatedBy.TOKEN_BUDGET``) and proceed to
    # scoring rather than reporting an agent error. ``token_budget_hit`` (set when
    # we caught the wrapped exception) is authoritative; the error-event code is a
    # belt-and-suspenders fallback.
    if token_budget_hit or error_code == _TOKEN_BUDGET_ERROR_CODE:
        status = "token_budget"
    elif (
        str(status) == "error"
        and error_code == _MAX_ITERATIONS_ERROR_CODE
    ):
        status = "max_iterations"

    solution = _read_text_or_empty(workspace / args.solution_file)

    return {
        "status": str(status),
        "last_message": last,
        "command_history": cmds,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cache_read_tokens": cache_read_tokens,
        "agent_calls": agent_calls,
        "accumulated_cost_usd": cost,
        "model": args.model,
        "solution_file_contents": solution,
        # ``error_code`` / ``error_detail`` are the SDK ConversationErrorEvent
        # signal (e.g. MaxIterationsReached). They are advisory telemetry only;
        # ``error`` stays ``None`` here because ``conv.run()`` did NOT raise (a
        # genuine exception path is handled in ``main()`` with ``error`` set).
        "error_code": error_code or None,
        "error_detail": error_detail or None,
        "error": None,
    }


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    result_file = Path(args.result_file)
    try:
        payload = _run(args)
    except BaseException as exc:  # noqa: BLE001 — ALWAYS write a valid JSON
        # Best-effort read of any solution the agent authored before failing,
        # so meta-n's authoritative FS read still has a redundant audit copy.
        solution = ""
        try:
            solution = _read_text_or_empty(Path(args.workspace) / args.solution_file)
        except Exception:  # noqa: BLE001
            solution = ""
        payload = {
            "status": "error",
            "last_message": "",
            "command_history": [],
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cache_read_tokens": 0,
            "agent_calls": 0,
            "accumulated_cost_usd": 0.0,
            "model": getattr(args, "model", ""),
            "solution_file_contents": solution,
            "error": f"{type(exc).__name__}: {exc}",
        }
        # Surface the traceback on stderr for post-hoc debugging (stdout stays clean).
        traceback.print_exc()
    try:
        _atomic_write_json(result_file, payload)
    except Exception:  # noqa: BLE001 — last-ditch: emit JSON to stdout so meta-n can fall back
        sys.stdout.write(json.dumps(payload, ensure_ascii=False))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
