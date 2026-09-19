#!/usr/bin/env python
"""Standalone builtin-script-on-terminal-bench subprocess bridge for meta-n.

WHAT THIS IS
------------
This is the FULL runner (promoted from ``builtin_tb_runner_spike.py``) that lets a
PRE-AUTHORED bash script drive a real **terminal_bench** task container and be
scored by terminal_bench's own verifier. It is the *same-container builtin
CONTROL* for the OpenHands / Terminus 2 A/B: builtin/OH/T2 all run the IDENTICAL
terminal-bench tasks scored by the IDENTICAL verifier (the native
``TerminalBenchExecutor`` CANNOT run the legacy ``original-tasks`` layout, so
``builtin`` must go through the SAME Harness the OH/T2 runners use).

It is the builtin analogue of ``scripts/t2_runner.py`` (Terminus 2) and
``scripts/oh_tb_runner.py`` (OpenHands): same ``--request`` / ``--result`` JSON
contract, same result-JSON schema, so meta-n's shared result parser
(``backends._external_tb._ExternalTBBackend._to_run_result``) consumes all three
unchanged.

The KEY difference from the OH / T2 runners: there is **NO LLM and NO agent loop
in this runner**. The bash script is authored meta-n-side by the native
``Layer1Solver`` (ONE outer LLM call, ``token_basis=outer``) and shipped here in
the ``--request`` JSON; this runner only writes it into the harness-provisioned
container and executes it, then runs the verifier. Authoring tokens are accounted
meta-n-side, so this runner reports ``total_input_tokens=0`` /
``total_output_tokens=0`` / ``agent_calls=0``.

It runs **only** under
``<repo>/.venv_external_agents/bin/python`` (the venv
that has ``terminal_bench``). meta-n NEVER imports this module — meta-n's own env
lacks the SDK and its pins conflict with meta-n's. meta-n shells out to::

    /Users/.../.venv_external_agents/bin/python scripts/builtin_tb_runner.py \
        --request <request.json> --result <result.json>

CRITICAL INVARIANT (mirrors t2_runner / oh_tb_runner): ``import terminal_bench``
(and everything that pulls it in) happens ONLY inside ``_run`` / helper functions,
NEVER at module top level, so that even if meta-n's interpreter imported this
module it would not drag the conflicting SDK into meta-n's process. The module top
level is deliberately limited to the stdlib + the stdlib-only
``scripts/_runner_common.py``.

DESIGN (builtin script AS a terminal_bench BaseAgent)
-----------------------------------------------------
1. ``BuiltinScriptTBAgent(BaseAgent)``: a TRIVIAL agent whose ``perform_task``
   writes the pre-authored bash script into the provisioned container (via the
   docker exec channel ``session.container.exec_run``, a base64 heredoc so the
   script body lands as ``/app/solution.sh`` INSIDE the container — the host is
   never touched) and EXECUTES it (capturing the output for the transcript). NO
   LLM, NO agent loop. The authored script arrives via ``agent_kwargs["script"]``.
2. ``_run`` builds ``Harness(... agent_import_path='builtin_tb_runner:Builtin...',
   task_ids=[task_id], cleanup=True, ...)`` and calls ``harness._run_trial`` —
   IDENTICAL provisioning to t2_runner / oh_tb_runner — then maps ``results``
   (``is_resolved`` -> reward) into the SAME result-JSON schema the other two
   runners write.

SAFETY / TEARDOWN / ROBUSTNESS
------------------------------
* The script executes INSIDE the Docker container ``spin_up_terminal`` builds; the
  host is never touched by the script (the agent only relays it into the
  container's exec channel). Helper-file staging is the one host touch and is
  path-validated by the shared ``_runner_common.stage_helper_files`` guard.
* ``spin_up_terminal``'s ``finally`` tears the container down (``cleanup=True``).
* Bounded by the harness's ``asyncio.wait_for(global_agent_timeout_sec)`` plus
  meta-n's outer SIGKILL of this runner's process group on a hard timeout.
* On ANY exception we STILL write a valid result JSON (``ok=false`` /
  ``status="error"`` / ``reward=0.0``) and exit 0, so meta-n always has a file to
  parse. We exit nonzero only if we cannot even write the result file.
"""

# ---------------------------------------------------------------------------
# MODULE TOP LEVEL: STDLIB ONLY.
# Never import terminal_bench here -- only inside functions, so this module is
# import-safe even from meta-n's interpreter (which lacks the SDK). This mirrors
# the t2_runner / oh_tb_runner invariant exactly.
# ---------------------------------------------------------------------------
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
import traceback
from pathlib import Path

# Shared, stdlib-only runner helpers (single-sourced across the runners; see
# scripts/_runner_common.py — a top-level module on the child PYTHONPATH). Imported
# at module top so this module stays stdlib-only at module scope (terminal_bench is
# still imported lazily inside ``_run`` and the agent class).
from _runner_common import (  # noqa: E402  (top-level module on the child PYTHONPATH)
    classify_runner_error as _classify_runner_error,
    error_result_payload as _error_result_payload,
    failure_mode_str as _failure_mode_str,
    safe_read_text as _safe_read_text,
    sanitize_run_label as _sanitize_run_label_common,
    stage_helper_files as _stage_helper_files_common,
    write_result as _write_result,
)


def _sanitize_run_label(label) -> str:
    """Builtin-TB wrapper over the shared ``sanitize_run_label`` (``builtin-tb`` stem).

    Compose project names must match ``[a-z0-9][a-z0-9_-]*``. meta-n passes the
    already-sanitized lease session; this is a defensive pass plus a
    ``builtin-tb-{uuid}`` fallback so two concurrent runs never share a project
    (which would collide on the globally unique ``container_name``).
    """
    return _sanitize_run_label_common(label, fallback="builtin-tb")


# ---------------------------------------------------------------------------
# Constants (mirror t2_runner / oh_tb_runner where shared).
# ---------------------------------------------------------------------------
#: Fallback trial/compose-project name when meta-n supplies no per-run label.
DEFAULT_RUN_LABEL = "builtin-tb-bridge"
#: Sentinel echoed after the script runs so the captured output unambiguously
#: proves the script body executed INSIDE the container (not on the host).
EXEC_SENTINEL = "__BUILTIN_TB_RAN__"

# Module-level record of what the agent observed in-container (read after the
# trial). The harness instantiates the agent itself, so a module global is the
# simplest cross-boundary channel (mirrors oh_tb_runner._COMMANDS_EXECUTED /
# t2_runner._ACTIVE_BUDGETED_LLM).
_CAPTURE: dict = {
    "wrote_script": False,
    "ran_script": False,
    "exec_rc": None,
    "sentinel_seen": False,
    "exec_output": "",
    "error": None,
}


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
    if args.output_dir is not None:
        req["output_dir"] = args.output_dir
    if getattr(args, "run_label", None) is not None:
        req["run_label"] = args.run_label
    if args.timeout is not None:
        # --timeout is the agent wall timeout (seconds) for this single task.
        req["agent_timeout_sec"] = args.timeout

    # --script-file points at a FILE whose contents are the pre-authored bash
    # script. (The canonical meta-n path passes the script inline in the request
    # JSON under ``bash_script``.)
    if getattr(args, "script_file", None) is not None:
        p = Path(args.script_file)
        if p.exists():
            req["bash_script"] = p.read_text()

    # --staged-files points at a JSON file: {"container/rel/path": "<contents>"}.
    if args.staged_files is not None:
        staged_path = Path(args.staged_files)
        if staged_path.exists():
            req["staged_files"] = json.loads(staged_path.read_text())

    # ---- Defaults for anything still unset --------------------------------
    # The pre-authored script the runner executes in the container. meta-n's
    # BuiltinTBBackend ships it under ``bash_script`` (the _RunPrep request key);
    # accept the ``script`` alias too so the discrete-flag entry form stays simple.
    if "bash_script" not in req and "script" in req:
        req["bash_script"] = req["script"]
    req.setdefault("bash_script", "")
    req.setdefault("staged_files", {})
    req.setdefault("agent_timeout_sec", 900)
    req.setdefault("test_timeout_sec", 180)
    req.setdefault("no_rebuild", False)

    return req


# ===========================================================================
# Injected agent class -- built lazily so the SDK imports it subclasses live in
# function scope, never at module top level (the critical invariant).
# ===========================================================================
def _build_agent_class(req: dict):
    """Construct and return ``BuiltinScriptTBAgent`` (a terminal_bench ``BaseAgent``)
    wired with this request's pre-authored script + staged helper files. SDK
    imports are local to this function (the import-safety invariant).
    """
    # ---- terminal_bench imports (SDK scope only) -------------------------------
    from terminal_bench.agents.base_agent import AgentResult, BaseAgent
    from terminal_bench.agents.failure_mode import FailureMode

    script_default = req.get("bash_script", "") or ""
    staged_files = req.get("staged_files", {}) or {}

    class BuiltinScriptTBAgent(BaseAgent):
        """Trivial terminal_bench agent: write a PRE-AUTHORED bash script into the
        provisioned container and execute it. NO LLM, NO agent loop.

        The script is authored meta-n-side by ``Layer1Solver`` (one outer LLM call)
        and arrives via ``agent_kwargs["script"]``. This agent only relays it into
        the SAME container the verifier inspects, so the builtin control runs the
        IDENTICAL env + verifier as the OH / T2 backends.
        """

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            # Accept (and ignore) the same kwargs the harness forwards to any agent
            # (model_name, api_base, ...) so instantiation never fails on an extra
            # kwarg. ``script`` is the one meaningful injection point; fall back to
            # the closure default (the request's script) if it was not forwarded.
            self._script = kwargs.get("script", script_default)
            self._staged_files = kwargs.get("staged_files", staged_files)

        @staticmethod
        def name() -> str:
            return "builtin-script-meta-n"

        def _stage_helper_files(self, session) -> None:
            """Stage the injection's helper files into the container (shared guard).

            Delegates to the single-sourced, security-load-bearing
            ``_runner_common.stage_helper_files`` (path-traversal validation +
            ``/app/<rel>`` copy — keys are workspace-relative and already carry
            the ``helpers/`` prefix). Best-effort: a staging failure is logged
            to stderr but never aborts the run (the script may not need the helper).
            """
            _stage_helper_files_common(
                session, self._staged_files, prefix="builtin_tb"
            )

        def perform_task(self, instruction, session, logging_dir=None):
            """Write the pre-authored script into the container and run it.

            Uses ``session.container.exec_run`` (the docker SDK exec channel the
            TmuxSession wraps) for a deterministic write+exec — the most unambiguous
            in-container execution path (the tmux pane is captured separately by the
            harness for the transcript). The host is never touched by the script.
            """
            failure_mode = FailureMode.NONE
            try:
                # 0) Stage any helper files into the container BEFORE the script.
                self._stage_helper_files(session)

                container = session.container  # the live docker container object

                # --- 1. Write the PRE-AUTHORED script into the container --------
                # base64 through `sh -c` so the script content lands as a file
                # INSIDE the container at /app/solution.sh regardless of quoting.
                b64 = base64.b64encode(
                    (self._script or "").encode("utf-8")
                ).decode("ascii")
                write_cmd = (
                    f"mkdir -p /app && printf %s {b64} | base64 -d "
                    f"> /app/solution.sh && chmod +x /app/solution.sh"
                )
                wres = container.exec_run(["sh", "-c", write_cmd])
                _CAPTURE["wrote_script"] = (getattr(wres, "exit_code", 1) == 0)

                # --- 2. EXECUTE the script INSIDE the container -----------------
                # Append a sentinel echo so the captured output unambiguously
                # proves the script body ran in-container.
                run_cmd = (
                    f"cd /app && bash /app/solution.sh; rc=$?; "
                    f"echo {EXEC_SENTINEL} rc=$rc"
                )
                rres = container.exec_run(["sh", "-c", run_cmd])
                run_out = (getattr(rres, "output", b"") or b"").decode(
                    "utf-8", errors="replace"
                )
                _CAPTURE["exec_rc"] = getattr(rres, "exit_code", None)
                _CAPTURE["ran_script"] = (getattr(rres, "exit_code", 1) == 0)
                _CAPTURE["sentinel_seen"] = EXEC_SENTINEL in run_out
                _CAPTURE["exec_output"] = run_out[-20000:]

            except Exception as exc:  # noqa: BLE001 - record, return a valid result
                _CAPTURE["error"] = f"{type(exc).__name__}: {exc}"
                failure_mode = FailureMode.UNKNOWN_AGENT_ERROR
                traceback.print_exc()

            # Authoring tokens are accounted meta-n-side; this runner reports 0.
            return AgentResult(
                total_input_tokens=0,
                total_output_tokens=0,
                failure_mode=failure_mode,
                timestamped_markers=[],
            )

    return BuiltinScriptTBAgent


# ===========================================================================
# Result-JSON helpers (mirror t2_runner / oh_tb_runner)
# ===========================================================================
# ``_safe_read_text`` / ``_failure_mode_str`` / ``_write_result`` are imported
# from ``_runner_common`` at module top (single-sourced across the runners).


def _error_payload(task_id: str, failure_mode: str, error: str) -> dict:
    """Build a valid result JSON for the error path (same schema the other runners
    write), delegating to the single-sourced ``_runner_common.error_result_payload``.
    Token fields stay 0 — authoring tokens are accounted meta-n-side, and no LLM
    runs in this runner.

    H3 producer: a degraded/error payload hard-sets ``attribution_available=False``
    — the run is UNMEASURABLE (the contract maps False → None / unmeasurable on the
    consumer side).
    """
    payload = _error_result_payload(task_id, failure_mode, error)
    payload["attribution_available"] = False
    return payload


# ===========================================================================
# The actual single-task run (drives terminal_bench's own machinery)
# ===========================================================================
def _run(req: dict, result_path: str) -> int:
    """Execute one task end-to-end via ``Harness._run_trial`` and write the result
    JSON. Returns a process exit code (0 on completion incl. scored failure).
    Mirrors ``t2_runner._run`` / ``oh_tb_runner._run``."""
    task_id = req.get("task_id", "")
    started = time.time()

    # SDK imports -- function scope ONLY (the import-safety invariant).
    from terminal_bench.harness.harness import Harness
    from terminal_bench.handlers.trial_handler import TrialHandler

    tasks_dir = req["tasks_dir"]
    output_dir = req["output_dir"]
    script = req.get("bash_script", "") or ""
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

    # Build the injected agent class wired with this request's script + staged files.
    Injected = _build_agent_class(req)

    # Register the class under a STABLE module name so AgentFactory can import it by
    # "module:class". When run as a script this module is "__main__"; register both
    # names so import_path resolution works regardless of how the harness imports it.
    import_module_name = __name__ if __name__ != "__main__" else "builtin_tb_runner"
    sys.modules.setdefault("builtin_tb_runner", sys.modules[__name__])
    setattr(sys.modules["builtin_tb_runner"], "BuiltinScriptTBAgent", Injected)
    if __name__ in sys.modules:
        setattr(sys.modules[__name__], "BuiltinScriptTBAgent", Injected)
    # Ensure this script's dir is importable by the harness.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    agent_import_path = f"{import_module_name}:BuiltinScriptTBAgent"

    # Per-run-unique trial identity: meta-n passes the lease session as run_label so
    # the compose project (== trial_name) is GLOBALLY unique (avoids container-name
    # collisions under --parallel). Fall back to a uuid when no label is supplied.
    run_label = _sanitize_run_label(req.get("run_label") or DEFAULT_RUN_LABEL)
    run_id = run_label

    # Build the Harness for this single task. dataset_path is REQUIRED
    # (Harness.__init__ eagerly builds a Dataset); point it at the local tasks dir
    # and scope to the one task via task_ids so no remote registry is touched. No
    # model_name matters (no LLM runs here); pass a metadata-only marker.
    harness = Harness(
        output_path=output_path,
        run_id=run_id,
        agent_import_path=agent_import_path,
        model_name="builtin-no-llm",  # metadata only; no LLM runs in this runner
        dataset_path=Path(tasks_dir),
        agent_kwargs={
            # The PRE-AUTHORED script the agent runs in-container (authored
            # meta-n-side by Layer1Solver). The agent also reads ``staged_files``.
            "script": script,
            "staged_files": req.get("staged_files", {}),
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

    # ---- Map TrialResults -> result JSON -------------------------------------
    cap = dict(_CAPTURE)

    is_resolved = bool(getattr(results, "is_resolved", False))
    reward = 1.0 if is_resolved else 0.0

    failure_mode = _failure_mode_str(getattr(results, "failure_mode", None))

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

    # command_history: the builtin control runs a single pre-authored script (not a
    # command stream), so the "command" is the script itself. Surface it (truncated)
    # so attribution has a non-empty stand-in; fall back to the harness commands.txt.
    command_history: list[str] = []
    if script.strip():
        command_history = [script[:4000]]
    if not command_history:
        cmds = _safe_read_text(tp.commands_path)
        if cmds:
            command_history = [ln for ln in cmds.splitlines() if ln.strip()]

    payload = {
        "ok": True,
        "status": "ok",
        "reward": reward,
        "score": reward,
        "is_resolved": is_resolved,
        "task_id": task_id,
        # Token fields 0: authoring tokens are accounted meta-n-side (the outer
        # Layer1Solver call), and NO LLM runs in this runner.
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "agent_calls": 0,
        "failure_mode": failure_mode,
        "parser_results": parser_results,
        "timestamped_markers": [],
        "command_history": command_history,
        # H3 producer: the pre-authored control script IS the command stream, so
        # attribution is MEASURABLE iff a non-empty script was shipped. An empty
        # script means there was nothing to attribute (UNMEASURABLE / None).
        "attribution_available": bool(script.strip()),
        "transcript_path": transcript_path,
        "post_agent_pane": post_agent_pane[-20000:],
        "post_test_pane": post_test_pane[-20000:],
        "wall_s": round(time.time() - started, 3),
        # One step: the single pre-authored script run (no agent turns).
        "steps": 1 if cap.get("ran_script") else 0,
        "trial_started_at": getattr(results, "trial_started_at", None),
        "trial_ended_at": getattr(results, "trial_ended_at", None),
        "error": None,
    }
    _write_result(result_path, payload)
    return 0


# ===========================================================================
# Entry point (mirrors t2_runner.main / oh_tb_runner.main)
# ===========================================================================
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="builtin_tb_runner",
        description="Standalone builtin-script-on-terminal-bench subprocess bridge.",
    )
    # Canonical meta-n invocation: request + result JSON.
    parser.add_argument("--request", help="Path to the request JSON written by meta-n.")
    parser.add_argument("--result", "--result-json", dest="result",
                        help="Path to write the result JSON.")
    # Discrete-flag entry form (parity with the sibling runners' named flags).
    parser.add_argument("--task-id", dest="task_id",
                        help="Terminal-bench task id (e.g. hello-world).")
    parser.add_argument("--tasks-dir", dest="tasks_dir",
                        help="Directory containing the task folders.")
    parser.add_argument("--script-file", dest="script_file",
                        help="Path to a file whose contents are the pre-authored "
                             "bash script to run in the container.")
    parser.add_argument("--staged-files", dest="staged_files",
                        help="Path to a JSON file mapping container-relative paths "
                             "to file contents to stage into the container.")
    parser.add_argument("--timeout", dest="timeout", type=float,
                        help="Agent wall timeout in seconds for this task.")
    parser.add_argument("--output-dir", dest="output_dir",
                        help="Directory for harness logs/artifacts.")
    parser.add_argument("--run-label", dest="run_label",
                        help="Per-run-unique compose project / trial name (meta-n "
                             "passes the lease session); a uuid is used when absent.")
    args = parser.parse_args(argv)

    # --result is mandatory: it is the file meta-n reads.
    if not args.result:
        print("[builtin_tb_runner] FATAL: --result/--result-json is required.",
              file=sys.stderr)
        return 2

    result_path = args.result

    # LM Studio / litellm contract dummy (harmless; no LLM runs in this runner).
    os.environ.setdefault("OPENAI_API_KEY", "dummy")

    task_id = ""
    try:
        req = _load_request(args)
        task_id = req.get("task_id", "")
        # Validate the minimal required fields up front so a missing field produces
        # a clean error JSON rather than a deep traceback.
        for field in ("task_id", "tasks_dir", "output_dir"):
            if not req.get(field):
                raise ValueError(f"request is missing required field: {field!r}")
        return _run(req, result_path)
    except Exception as exc:  # noqa: BLE001 - top-level guard: ALWAYS emit JSON
        tb = traceback.format_exc()
        print(f"[builtin_tb_runner] ERROR: {exc}\n{tb}", file=sys.stderr)
        failure_mode = _classify_runner_error(exc, include_token_budget=False)
        try:
            _write_result(
                result_path, _error_payload(task_id, failure_mode, tb)
            )
        except Exception as werr:  # pragma: no cover - cannot even write result
            print(f"[builtin_tb_runner] FATAL: could not write result JSON: {werr}",
                  file=sys.stderr)
            return 1
        # Exit 0 on a caught error: meta-n has a valid JSON to parse.
        return 0


if __name__ == "__main__":
    sys.exit(main())
