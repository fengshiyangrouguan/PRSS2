"""Compose-layer helpers for the TerminalBench 2.0 integration.

Compose YAML templates, Docker name sanitizers, reward parsing, the
host-side verifier-dir anti-forgery purge, the ``_ExecResult`` container,
and the ``docker compose`` subprocess helpers. Split verbatim out of the
old single-file ``meta_n/integrations/terminal_bench.py`` module.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import shutil
from pathlib import Path

# Patch-target indirection: the compose helpers were module globals of the old
# single-file ``meta_n.integrations.terminal_bench`` module, and existing tests
# patch them ON THE PACKAGE (e.g. ``mock.patch("meta_n.integrations.
# terminal_bench._run_compose_command")`` / ``monkeypatch.setattr(tb, ...)``).
# Cross-function calls therefore resolve through the package namespace at call
# time (exactly like the old module-global lookup), never through a local
# binding a package-level patch could not see.
from meta_n.integrations import terminal_bench as _tb_pkg

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Compose YAML templates (mirroring Harbor's docker-compose-*.yaml)
# ---------------------------------------------------------------------------

_COMPOSE_BASE = """\
services:
  main:
    volumes:
      - ${HOST_LOGS_PATH}/verifier:/logs/verifier
      - ${HOST_LOGS_PATH}/agent:/logs/agent
    deploy:
      resources:
        limits:
          cpus: "${CPUS}"
          memory: "${MEMORY}"
"""

_COMPOSE_BUILD = """\
services:
  main:
    build:
      context: ${CONTEXT_DIR}
    image: ${IMAGE_NAME}
    pull_policy: build
    command: ["sh", "-c", "sleep infinity"]
"""

_COMPOSE_PREBUILT = """\
services:
  main:
    image: ${IMAGE_NAME}
    command: ["sh", "-c", "sleep infinity"]
"""

_COMPOSE_NO_NETWORK = """\
services:
  main:
    network_mode: none
"""


def _sanitize_compose_name(name: str) -> str:
    """Sanitize a string for use as a Docker Compose project name.

    Legacy NATIVE-path rule (lowercase, then digit-prefix, then substitute),
    byte-pinned by tests/test_terminal_bench.py::TestSanitize and load-bearing
    for ``task_id`` derivation in the legacy loader — task_id is a persisted
    run coordinate (traces, telemetry, summary.json). Intentionally NOT
    unified with the external-agents spine's
    ``meta_n.core.external_agents._bridge.sanitize_compose_name`` (which
    substitutes/collapses/strips in a different order, e.g. ``"-bad-start"``
    → ``"bad-start"`` vs ``"0-bad-start"`` here); merging the rules would
    change persisted identifiers on the native path. Namespaces stay disjoint
    anyway: native sessions are ``tb2-*``, spine sessions are ``ext-*``.
    """
    name = name.lower()
    if not re.match(r"^[a-z0-9]", name):
        name = "0" + name
    return re.sub(r"[^a-z0-9_-]", "-", name)


def _sanitize_image_name(name: str) -> str:
    """Sanitize a string for use as a Docker image name."""
    name = name.lower()
    if not re.match(r"^[a-z0-9]", name):
        name = "0" + name
    return re.sub(r"[^a-z0-9._-]", "-", name)


# ---------------------------------------------------------------------------
# Reward parsing
# ---------------------------------------------------------------------------

def _read_reward(verifier_dir: Path) -> float:
    """Read reward from verifier output. Returns 0.0 on any failure.

    A verifier that writes the literal string "NaN" or "inf" to
    reward.json/reward.txt would otherwise have those values pass
    through `float()` and poison downstream score aggregation. Clamp
    non-finite to 0.0 here for parity with the openevolve subprocess
    target (which clamps in `_run_openevolve_eval_in_process`) and the
    co_bench aggregator (which filters non-finite at average time).
    """
    def _coerce(raw) -> float:
        v = float(raw)
        return v if math.isfinite(v) else 0.0

    # Prefer reward.json (dict with "reward" key), fall back to reward.txt
    reward_json = verifier_dir / "reward.json"
    if reward_json.exists():
        try:
            content = reward_json.read_text().strip()
            if content:
                data = json.loads(content)
                # A conventional reward.json is an object with a "reward" key,
                # but some verifiers emit a bare scalar (e.g. ``1`` / ``0.5``).
                # Guard the ``.get`` so a non-dict value (scalar/list/null) does
                # not raise AttributeError and escape this function — that would
                # both violate the documented "returns 0.0 on any failure"
                # contract and skip the reward.txt fallback below. A bare scalar
                # is itself a valid reward; anything non-numeric raises in
                # ``_coerce`` and is caught, falling through to reward.txt.
                if isinstance(data, dict):
                    return _coerce(data.get("reward", 0))
                return _coerce(data)
        except (json.JSONDecodeError, ValueError, TypeError, AttributeError, OSError):
            logger.warning("Failed to parse reward.json")

    reward_txt = verifier_dir / "reward.txt"
    if reward_txt.exists():
        try:
            content = reward_txt.read_text().strip()
            if content:
                return _coerce(content)
        except (ValueError, TypeError, OSError):
            logger.warning("Failed to parse reward.txt")

    return 0.0


def _purge_verifier_dir(verifier_dir: Path) -> list[str]:
    """Remove everything in the bind-mounted verifier dir; return removed names.

    Anti-forgery sweep run on the HOST between the solve phase and the
    verifier phase: anything present in ``/logs/verifier`` at that point was
    written during the solve phase and can only be tampering (the honest flow
    writes rewards only from ``test.sh``, after this sweep). Directories that
    are not symlinks are removed recursively; symlinks are unlinked, never
    followed. Never raises — a purge failure must not abort the eval.
    """
    removed: list[str] = []
    if not verifier_dir.exists():
        return removed
    try:
        entries = list(verifier_dir.iterdir())
    except OSError:
        return removed
    for p in entries:
        try:
            if p.is_dir() and not p.is_symlink():
                shutil.rmtree(p, ignore_errors=True)
            else:
                p.unlink(missing_ok=True)
            removed.append(p.name)
        except OSError:
            continue
    return removed


# ---------------------------------------------------------------------------
# Simple container for compose exec results
# ---------------------------------------------------------------------------

class _ExecResult:
    """Result from a docker compose exec command."""

    __slots__ = ("stdout", "stderr", "return_code")

    def __init__(self, stdout: str = "", stderr: str = "", return_code: int = -1):
        self.stdout = stdout
        self.stderr = stderr
        self.return_code = return_code


# ---------------------------------------------------------------------------
# Compose subprocess helpers
# ---------------------------------------------------------------------------

async def _run_compose_command(
    compose_files: list[Path],
    project_name: str,
    project_dir: Path,
    command: list[str],
    env_overrides: dict[str, str] | None = None,
    timeout_sec: float | None = None,
    check: bool = True,
) -> _ExecResult:
    """Run a docker compose command and return the result."""
    full_command = [
        "docker", "compose",
        "--project-name", project_name,
        "--project-directory", str(project_dir.resolve()),
    ]
    for path in compose_files:
        full_command.extend(["-f", str(path.resolve())])
    full_command.extend(command)

    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)

    process = await asyncio.create_subprocess_exec(
        *full_command,
        env=env,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    try:
        if timeout_sec:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(), timeout=timeout_sec
            )
        else:
            stdout_bytes, stderr_bytes = await process.communicate()
    except asyncio.TimeoutError:
        process.terminate()
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(), timeout=5
            )
        except asyncio.TimeoutError:
            process.kill()
            stdout_bytes, stderr_bytes = await process.communicate()
        if check:
            raise RuntimeError(
                f"Docker compose command timed out after {timeout_sec}s: "
                f"{' '.join(command)}"
            )
        return _ExecResult(
            stdout="", stderr=f"Timeout after {timeout_sec}s", return_code=-1,
        )

    stdout = stdout_bytes.decode(errors="replace") if stdout_bytes else ""
    stderr = stderr_bytes.decode(errors="replace") if stderr_bytes else ""

    if check and process.returncode != 0:
        raise RuntimeError(
            f"Docker compose command failed (rc={process.returncode}): "
            f"{' '.join(command)}\nstdout: {stdout[:500]}\nstderr: {stderr[:500]}"
        )

    return _ExecResult(
        stdout=stdout, stderr=stderr, return_code=process.returncode or 0,
    )


async def _compose_exec(
    compose_files: list[Path],
    project_name: str,
    project_dir: Path,
    command: str,
    env_overrides: dict[str, str] | None = None,
    timeout_sec: float | None = None,
    user: str | None = None,
) -> _ExecResult:
    """Execute a command inside a running compose container."""
    exec_cmd = ["exec", "-T"]  # -T: disable pseudo-TTY (non-interactive)
    if user:
        exec_cmd.extend(["-u", user])
    exec_cmd.extend(["main", "bash", "-c", command])

    return await _tb_pkg._run_compose_command(
        compose_files=compose_files,
        project_name=project_name,
        project_dir=project_dir,
        command=exec_cmd,
        env_overrides=env_overrides,
        timeout_sec=timeout_sec,
        check=False,  # Don't raise on non-zero exit — we capture it
    )
