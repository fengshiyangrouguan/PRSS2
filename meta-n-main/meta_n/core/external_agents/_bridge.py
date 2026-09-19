"""Shared subprocess-bridge primitives for the external-agent backends.

Both the OpenHands and Terminus 2 backends drive their SDK in a *separate venv*
via a subprocess bridge, and historically each carried byte-identical copies of
the trust-boundary plumbing: the SIGKILL-the-process-group reaper, the
hard-timeout formula, and the curated env allowlist that keeps meta-n's secrets
out of the child. This module single-sources those invariants so there is one
copy of each (audit ``reuse-simplify``).

It imports **only the standard library** — no ``openhands`` / ``terminal_bench``
/ ``docker`` SDK — so the ``external_agents`` package stays importable with those
absent. The Docker-CLI subprocess teardown (``docker rm`` / ``network rm``) stays
in each backend because it is backend-shaped (label scoping, compose vs ``--rm``).
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
import signal
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    import asyncio

__all__ = [
    "DEFAULT_HARD_TIMEOUT_S",
    "SAFE_ENV_KEYS",
    "TEARDOWN_WAIT_S",
    "hard_timeout",
    "sigkill_group",
    "sanitize_compose_name",
    "scrubbed_child_env",
]

logger = logging.getLogger(__name__)

#: Default hard wall-clock ceiling (s) for a runner subprocess when no soft limit
#: is configured. Shared by both backends so the backstop is identical.
DEFAULT_HARD_TIMEOUT_S = 1800.0

#: Upper bound (s) on every post-SIGKILL reap ``await`` (``proc.wait()``) and
#: ``docker rm -f`` wait in the backends' teardown paths, so an un-reapable
#: child (uninterruptible sleep) or a wedged docker daemon cannot stall a
#: timeout/cancel/finally unwind indefinitely. Single-sourced here; the
#: backends alias it as ``_TEARDOWN_WAIT_S``.
TEARDOWN_WAIT_S: float = 70.0


def hard_timeout(soft: float | None, *, default: float = DEFAULT_HARD_TIMEOUT_S) -> float:
    """Hard wall-clock ceiling for a runner subprocess.

    ``default`` when there is no soft limit, else ``max(soft*1.25, soft+120)`` —
    enough slack past the soft budget for the runner (Docker build + agent +
    verifier) to wind down before the hard ``asyncio.wait_for`` fires. Identical
    formula for OH and T2 so the backstop is single-sourced.
    """
    if soft is None:
        return default
    return max(soft * 1.25, soft + 120.0)


def sigkill_group(proc: "asyncio.subprocess.Process | None") -> None:
    """SIGKILL a subprocess's whole process group (``start_new_session=True``).

    The runner is spawned in its own session/group, so
    ``os.killpg(os.getpgid(pid), SIGKILL)`` reaps the runner *and* any child it
    shelled out. A benign no-op when the process has already exited (its
    ``returncode`` is set), and never raises — pid/group lookup races and
    permission errors are suppressed.

    Guards against a pid/pgid-reuse race: if the child has already been reaped
    (``returncode is not None``) the OS may have recycled its pid onto an
    unrelated process, so we skip signalling rather than SIGKILL a stranger's
    group.
    """
    if proc is None:
        return
    pid = getattr(proc, "pid", None)
    if pid is None:
        return
    # Already reaped: skip to avoid pid/pgid reuse hitting an unrelated group.
    if getattr(proc, "returncode", None) is not None:
        return
    with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
        os.killpg(os.getpgid(pid), signal.SIGKILL)


def sanitize_compose_name(name: str, *, fallback: str = "0") -> str:
    """Reduce an arbitrary string to a Docker-Compose-safe project name.

    Compose project names must be ``[a-z0-9][a-z0-9_-]*``. This is the single
    canonical sanitizer for the whole external-agents path (it replaced four
    near-identical copies): lowercase, replace any char outside ``[a-z0-9_-]``
    with ``-``, collapse runs of ``-``, strip leading/trailing ``-``/``_``, and
    fall back to ``fallback`` (then prefix a digit if still not alnum-leading).
    Idempotent on an already-sanitized input.

    Args:
        name: The raw candidate name (typically ``ext-{task_id}-{uuid}``).
        fallback: Returned (sanitized) when the input reduces to empty.

    Returns:
        A compose-safe project name.
    """
    cleaned = re.sub(r"[^a-z0-9_-]", "-", str(name).strip().lower())
    cleaned = re.sub(r"-+", "-", cleaned).strip("-_")
    if not cleaned:
        cleaned = str(fallback).strip().lower() or "0"
    if not re.match(r"^[a-z0-9]", cleaned):
        cleaned = "0" + cleaned
    return cleaned


#: SAFETY: the curated allowlist of env vars a runner subprocess may inherit.
#: An external agent's terminal tool can shell out (on the bare host for the
#: CO-Bench path), so a single ``env`` / ``printenv`` would exfiltrate every
#: secret in meta-n's environment. Both backends therefore build the child env
#: from THIS allowlist only — never ``os.environ`` wholesale — so
#: OpenRouter/Azure/Anthropic keys never cross into the agent's reach. The
#: provider key the runner actually needs is injected separately by each backend.
SAFE_ENV_KEYS: tuple[str, ...] = (
    "PATH",
    "HOME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TMPDIR",
    "TZ",
    "TERM",
    "USER",
    "LOGNAME",
    "SHELL",
    "PYTHONPATH",  # needed if the runner imports from a path-rooted layout
    "PYTHONHASHSEED",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",  # litellm/httpx TLS verification roots
    "REQUESTS_CA_BUNDLE",
    "CURL_CA_BUNDLE",
    # Outbound-proxy config httpx/litellm honor — a proxied host otherwise
    # breaks the child runner's LLM egress. Egress CONFIG, not secrets; both
    # UPPER and lower case since the HTTP stack reads both spellings.
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
    # Docker-daemon connection config — the TB runner shells `docker`/`compose`,
    # so on a remote or rootless daemon it cannot reach the socket without these
    # and every TB trial fails env_error. Daemon CONFIG, not secrets.
    "DOCKER_HOST",
    "DOCKER_TLS_VERIFY",
    "DOCKER_CERT_PATH",
    "XDG_RUNTIME_DIR",
)


def scrubbed_child_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """Build a child-process env from the :data:`SAFE_ENV_KEYS` allowlist only.

    Copies only the allowlisted keys present in ``os.environ`` (never the whole
    environment), then overlays ``extra`` (the explicitly-needed credentials /
    flags each backend injects, e.g. ``OPENAI_API_KEY`` / ``OH_API_KEY``).

    Args:
        extra: Additional env entries to overlay onto the scrubbed base.

    Returns:
        A fresh dict safe to hand to ``create_subprocess_exec(env=...)``.
    """
    env = {k: os.environ[k] for k in SAFE_ENV_KEYS if k in os.environ}
    if extra:
        env.update({k: v for k, v in extra.items() if v is not None})
    return env
