"""Shared, stdlib-only helpers for the external-agent subprocess runners.

The four subprocess runners (``oh_runner.py`` — OpenHands on a host workspace,
``t2_runner.py`` — Terminus 2 on terminal-bench, ``oh_tb_runner.py`` — OpenHands
on terminal-bench, and ``builtin_tb_runner.py`` — the native pre-authored bash
script in the terminal-bench container, no inner LLM) historically each carried
byte-identical copies of a handful of trust-boundary / accounting helpers: the
token-budget kill switch trio, the litellm model-routing prefix logic, the
compose-label sanitizer, the result-file writer, the error-path result payload,
and the host-side path-traversal guard that stages helper files into a container.
This module single-sources those invariants so a fix to the security-load-bearing
path-traversal guard (or any other) is made once (audit ``reuse-simplify``).

Importability contract
-----------------------
This module imports **only the standard library** — never ``openhands`` /
``terminal_bench`` / ``litellm``. The runners already prepend ``scripts/`` to the
child ``PYTHONPATH`` (the backends launch them as ``-m <runner>`` from that dir),
so ``import _runner_common`` resolves as a top-level module from the same
directory. Keeping it stdlib-only preserves the runners' critical invariant: no
heavy SDK is imported at module scope (the SDKs are imported lazily inside each
runner's ``_run``).

The runners re-export the names they need from this module at their own module
top level (e.g. ``from _runner_common import TokenBudgetExceeded``), so existing
``oh_runner.TokenBudgetExceeded`` / ``oh_runner._install_token_budget_killswitch``
references (including the ones the unit suite imports) keep resolving to the SAME
class/function objects defined here.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from uuid import uuid4

__all__ = [
    "TokenBudgetExceeded",
    "KNOWN_LITELLM_PROVIDERS",
    "route_model",
    "accumulated_inner_tokens",
    "install_token_budget_killswitch",
    "is_token_budget_error",
    "classify_runner_error",
    "sanitize_run_label",
    "safe_read_text",
    "write_result",
    "failure_mode_str",
    "error_result_payload",
    "stage_helper_files",
]


class TokenBudgetExceeded(Exception):
    """Raised by the per-call inner-token kill switch when cumulative inner tokens
    (prompt + completion) exceed the per-run ``token_budget``.

    A stable, NON-retried signal the runners map to a SOFT ``token_budget`` stop
    (→ ``TerminatedBy.TOKEN_BUDGET``) so an over-budget run fails soft (proceeds to
    scoring with partial credit possible) instead of running unbounded on the
    iteration/episode cap alone. NOT a subclass of any SDK-retried exception type,
    so an LLM ``@retry`` decorator will not swallow / re-attempt it.

    This single class is shared by all three runners (each re-exports it), so an
    ``isinstance`` / ``except`` against ``<runner>.TokenBudgetExceeded`` and against
    ``_runner_common.TokenBudgetExceeded`` is the same class object.
    """


# litellm provider prefixes that already pin a routing target — when the model id
# starts with one of these, no 'openai/' prefix is added. Kept as the UNION of
# oh_runner's and terminal_bench's sets so OH and T2 route a given model id
# identically.
KNOWN_LITELLM_PROVIDERS = (
    "openai/", "azure/", "anthropic/", "hosted_vllm/", "lm_studio/", "ollama/",
    "openrouter/", "vertex_ai/", "gemini/", "bedrock/", "groq/", "mistral/",
    "together_ai/", "fireworks_ai/", "deepseek/", "xai/", "text-completion-openai/",
    "cohere/",
)


def route_model(model: str, base_url: str | None) -> str:
    """Return a litellm-routable model id for a custom OpenAI-compatible endpoint.

    When ``base_url`` is set and ``model`` carries no recognized litellm provider
    prefix, prepend ``openai/`` so litellm routes the call to the OpenAI-compatible
    server (LM Studio) instead of guessing a provider from the leading path
    segment (``google/...`` -> Google). The contracted ``google/gemma-4-31b-qat``
    thus becomes ``openai/google/gemma-4-31b-qat``. No-op when there is no
    ``base_url`` or the model already has a provider prefix.
    """
    if not base_url:
        return model
    if model.startswith(KNOWN_LITELLM_PROVIDERS):
        return model
    return f"openai/{model}"


def accumulated_inner_tokens(llm) -> int:
    """Best-effort read of cumulative inner (prompt + completion) tokens off an
    OpenHands ``LLM``'s metrics. Returns 0 on any structural surprise so the
    kill-switch check can never abort a run by raising itself."""
    try:
        usage = llm.metrics.accumulated_token_usage
        pt = int(getattr(usage, "prompt_tokens", 0) or 0)
        ct = int(getattr(usage, "completion_tokens", 0) or 0)
        return pt + ct
    except Exception:  # noqa: BLE001 - accounting read must never abort the run
        return 0


def install_token_budget_killswitch(llm, token_budget: int) -> None:
    """Install a per-call cumulative inner-token kill switch on ``llm``.

    openhands-sdk has NO in-loop token/cost enforcement we rely on (it stores
    ``LLM.metrics.max_budget_per_task`` but never compares against it during
    ``conv.run()`` — verified against 1.28.0), so an OpenHands run otherwise stops only on
    ``max_iteration_per_run`` — the fairness asymmetry vs Terminus 2, which DOES
    enforce an inner cumulative-token budget. We close the gap by wrapping the
    LLM's own per-call telemetry hook ``LLM._telemetry.on_request`` — which
    ``LLM.completion`` invokes once at the start of EVERY inner completion (inside
    the retry-wrapped ``_one_attempt``). The wrapper checks the running
    ``metrics.accumulated_token_usage`` BEFORE the call and raises
    :class:`TokenBudgetExceeded` once cumulative tokens have already exceeded the
    budget. That exception is NOT in ``LLM_RETRY_EXCEPTIONS`` (so the retry
    decorator does not swallow/re-attempt it) and propagates out of
    ``LLM.completion`` → ``agent.step`` → ``conv.run()``, whose ``except`` block
    sets ``execution_status = ERROR`` and stamps a ``ConversationErrorEvent`` with
    ``code = "TokenBudgetExceeded"`` (the exception's class name). The runner reads
    that code and rewrites the overloaded ERROR into the soft token-budget stop.

    RESIDUAL OVERSHOOT: the check fires before a call once the budget is ALREADY
    exceeded, so the single inner call that first crosses the budget runs to
    completion (its tokens are counted but not pre-empted). The overshoot is
    therefore bounded by one inner call's prompt+completion tokens — identical in
    spirit to ``t2_runner``'s ``_BudgetedLiteLLM.call`` guard, which likewise
    checks ``spent > token_budget`` before the *next* call. A true mid-call abort
    is not available without forking the transport, which is out of
    scope; this is the closest enforceable mechanism.

    No-op when ``token_budget <= 0`` or the telemetry hook is unavailable.
    """
    if token_budget <= 0:
        return
    telemetry = getattr(llm, "_telemetry", None)
    if telemetry is None:
        return
    original_on_request = telemetry.on_request

    def _budgeted_on_request(*a, **k):
        spent = accumulated_inner_tokens(llm)
        if spent > token_budget:
            raise TokenBudgetExceeded(
                f"inner token budget {token_budget} exceeded (spent={spent})"
            )
        return original_on_request(*a, **k)

    # Bind on the instance; LLM is a pydantic model, so set via object.__setattr__
    # on the telemetry object (its on_request is a bound method we shadow).
    try:
        object.__setattr__(telemetry, "on_request", _budgeted_on_request)
    except Exception:  # noqa: BLE001 - if we cannot install it, the run is just
        # un-budgeted (the spine's USD/iteration/wall caps still bound it).
        pass


def is_token_budget_error(exc: BaseException) -> bool:
    """True iff ``exc`` is (or wraps) the token-budget kill switch.

    ``conv.run()`` wraps an in-loop exception in ``ConversationRunError`` with
    ``__cause__`` set to the original. We therefore match the exception itself,
    its ``__cause__`` chain, and (belt-and-suspenders) the class-name token in its
    message — so a future SDK that re-wraps differently still classifies the stop.
    """
    cur: BaseException | None = exc
    seen = 0
    while cur is not None and seen < 8:
        if isinstance(cur, TokenBudgetExceeded):
            return True
        if TokenBudgetExceeded.__name__ in f"{type(cur).__name__}: {cur}":
            return True
        cur = getattr(cur, "__cause__", None)
        seen += 1
    return False


def classify_runner_error(exc: BaseException, *, include_token_budget: bool = True) -> str:
    """Map a top-level runner exception to its stable ``failure_mode`` tag.

    Single-sources the ``except``-guard classifier each TB runner's ``main()``
    hand-rolled: ``token_budget`` when the inner-token kill switch fired (a
    :class:`TokenBudgetExceeded` or a ``"token budget"`` message), ``env_error`` on
    a docker/compose/image/container or missing-field/request substring, else the
    generic ``unknown_agent_error`` (meta-n maps unknown modes to AGENT_ERROR /
    ENV_ERROR). ``include_token_budget=False`` drops the token-budget branch for the
    LLM-free ``builtin_tb_runner``, which has no inner LLM to over-spend.
    """
    msg = str(exc).lower()
    if include_token_budget and (
        isinstance(exc, TokenBudgetExceeded) or "token budget" in msg
    ):
        return "token_budget"
    if "docker" in msg or "compose" in msg or "image" in msg or "container" in msg:
        return "env_error"
    if "missing required field" in msg or "request" in msg:
        return "env_error"
    return "unknown_agent_error"


def sanitize_run_label(label, *, fallback: str = "run") -> str:
    """Reduce a run label to a Docker-Compose-safe project name.

    Compose project names must match ``[a-z0-9][a-z0-9_-]*``. meta-n passes the
    already-sanitized lease session (``ext-{task_id}-{uuid}``); this is a
    defensive pass plus a per-call uuid fallback (``{fallback}-{uuid}``) so two
    concurrent runs never share a project (which would collide on the globally
    unique ``container_name``). ``fallback`` is the runner-specific stem
    (``"t2"`` / ``"oh-tb"``).
    """
    raw = str(label or "").strip().lower()
    cleaned = re.sub(r"[^a-z0-9_-]", "-", raw)
    cleaned = re.sub(r"-+", "-", cleaned).strip("-_")
    if not cleaned:
        return f"{fallback}-{uuid4().hex[:8]}"
    return cleaned


def safe_read_text(path) -> str:
    """Read a text file, returning "" on any error (used for transcripts/panes)."""
    try:
        if path is None:
            return ""
        p = Path(path)
        if p.exists() and p.is_file():
            return p.read_text(errors="replace")
    except Exception:
        pass
    return ""


def failure_mode_str(fm) -> str:
    """Normalise a FailureMode enum (or str/None) to its string value."""
    if fm is None:
        return "none"
    val = getattr(fm, "value", None)
    if val is not None:
        return str(val)
    return str(fm)


def write_result(result_path: str, payload: dict) -> None:
    """Atomically-ish write the result JSON. Writes to a temp file then renames,
    so meta-n never reads a half-written file."""
    p = Path(result_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str))
    os.replace(tmp, p)


def error_result_payload(
    task_id: str,
    failure_mode: str,
    error: str,
    *,
    total_input_tokens: int = 0,
    total_output_tokens: int = 0,
    agent_calls: int = 0,
    command_history: list | None = None,
) -> dict:
    """Build the canonical error-path result JSON shared by the 3 TB runners.

    This is the WRITE side of the schema ``_external_tb._to_run_result`` READS, so
    single-sourcing it here keeps producer and consumer from drifting on a schema
    change. The keys / order match what ``t2_runner`` / ``oh_tb_runner`` /
    ``builtin_tb_runner`` historically each hand-built.

    ``total_*_tokens`` / ``agent_calls`` carry any partial inner spend recovered
    from the captured agent state so a run that burned real tokens then aborted is
    priced correctly (not at $0) and reports a real inner-call count.
    ``command_history`` carries any commands the executor already ran so
    attribution survives the abort (the builtin runner has no inner LLM, so it
    passes the token kwargs as their ``0`` defaults).

    Args:
        task_id: The benchmark task id.
        failure_mode: The stable failure-mode tag for the abort.
        error: The error text (truncated to the last 4000 chars).
        total_input_tokens: Partial inner prompt tokens recovered before the abort.
        total_output_tokens: Partial inner completion tokens recovered.
        agent_calls: Partial inner LLM-call count recovered.
        command_history: Commands the executor already ran (``None`` → ``[]``).

    Returns:
        A JSON-ready result dict for the error path.
    """
    return {
        "ok": False,
        "status": "error",
        "reward": 0.0,
        "score": 0.0,
        "is_resolved": False,
        "task_id": task_id,
        "total_input_tokens": int(total_input_tokens or 0),
        "total_output_tokens": int(total_output_tokens or 0),
        "agent_calls": int(agent_calls or 0),
        "failure_mode": failure_mode,
        "parser_results": {},
        "timestamped_markers": [],
        "command_history": command_history or [],
        "transcript_path": "",
        "post_agent_pane": "",
        "post_test_pane": "",
        "wall_s": 0.0,
        "steps": 0,
        "trial_started_at": None,
        "trial_ended_at": None,
        "error": (error or "")[-4000:],
    }


def stage_helper_files(session, staged_files, *, prefix: str) -> None:
    """Write each staged file to a host temp dir, then copy it into the container
    under ``/app/<rel>`` via ``session.copy_to_container``.

    Keys are WORKSPACE-relative (the injection producer stages
    ``helpers/<name>.py`` / ``helpers/__init__.py`` / ``helpers/_lib_<name>.py`` /
    ``helpers/<name>.sh``; the CO-Bench host provider writes the same keys under
    its workspace root), and the TB base image WORKDIR is ``/app`` — so
    ``helpers/foo.py`` lands at ``/app/helpers/foo.py``, resolving the advertised
    ``./helpers/...`` invocations.

    SECURITY (load-bearing): ``rel`` originates from the (untrusted) Ω-generated
    injection's ``staged_files`` map — injected code is supposed to run only inside
    Docker, never touch the host. ``Path('/tmp/x') / '/etc/passwd'`` collapses to
    ``/etc/passwd`` and ``../..`` escapes tmp_root, letting a malicious/buggy key
    write ARBITRARY host files (this runner is on the host, OUTSIDE Docker). We
    reject absolute / ``..`` keys, then confirm the resolved write target stays
    under tmp_root before writing. Best-effort: a staging failure is logged to
    stderr but never aborts the run (the agent may not need the helper).

    Args:
        session: The live terminal-bench session exposing ``copy_to_container``.
        staged_files: Map of container-relative path → file contents (untrusted).
        prefix: Runner tag for the temp-dir prefix / log lines (``"t2"`` / ``"oh_tb"``).
    """
    if not staged_files:
        return
    import tempfile

    try:
        tmp_root = Path(tempfile.mkdtemp(prefix=f"{prefix}_staged_"))
        tmp_root_resolved = tmp_root.resolve()
        for rel, contents in staged_files.items():
            rel_path = Path(rel)
            if rel_path.is_absolute() or ".." in rel_path.parts:
                print(f"[{prefix}_runner] WARN skipping unsafe staged path: {rel!r}",
                      file=sys.stderr)
                continue
            host_file = (tmp_root / rel_path).resolve()
            if (
                host_file != tmp_root_resolved
                and tmp_root_resolved not in host_file.parents
            ):
                print(f"[{prefix}_runner] WARN staged path escapes tmp root: {rel!r}",
                      file=sys.stderr)
                continue
            host_file.parent.mkdir(parents=True, exist_ok=True)
            host_file.write_text(contents)
            # Container destination dir: /app/<rel parent> — keys are
            # workspace-relative and already carry their ``helpers/`` prefix, and
            # copy_to_container mkdir-p's the dir then drops the file at
            # <container_dir>/<basename> (rel_path is already validated as a
            # non-traversing relative path).
            container_dir = "/app"
            if rel_path.parent != Path("."):
                container_dir = f"/app/{rel_path.parent.as_posix()}"
            session.copy_to_container(
                paths=[host_file],
                container_dir=container_dir,
            )
    except Exception as exc:  # pragma: no cover - best effort
        print(f"[{prefix}_runner] WARN staging helper files failed: {exc}",
              file=sys.stderr)
