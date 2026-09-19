"""Base executor — runs scripts and captures execution traces."""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import tempfile
import time
from abc import ABC, abstractmethod
from pathlib import Path

from meta_n.core.meta_layer import TaskDescription, Trace

logger = logging.getLogger(__name__)

#: Post-kill drain bound: after the group SIGKILL, salvage whatever output is
#: already in the pipes but never wait past this for EOF — a child that
#: ``setsid``'d itself out of the group can hold the pipe write-ends open
#: indefinitely, and the timeout contract is "return within timeout + a small
#: constant" regardless of what the script spawned.
_KILL_GRACE_S = 2.0


def _kill_process_group(proc: asyncio.subprocess.Process) -> None:
    """SIGKILL the child's whole session group (``start_new_session=True``
    makes the child a group leader, so pgid == pid). Killing only the direct
    bash pid would orphan backgrounded children, which keep the stdout/stderr
    pipes open and run on unbounded on the host."""
    try:
        os.killpg(proc.pid, signal.SIGKILL)
        return
    except (ProcessLookupError, PermissionError, OSError):
        pass
    try:
        proc.kill()
    except (ProcessLookupError, OSError):
        pass


async def _drain_after_kill(comm_task: "asyncio.Task") -> tuple[bytes, bytes]:
    """Bounded post-kill drain of the in-flight ``communicate()`` task:
    salvage the output buffered before the kill, give up (cancelling the
    task) after ``_KILL_GRACE_S`` if surviving pipe holders never close."""
    try:
        return await asyncio.wait_for(comm_task, timeout=_KILL_GRACE_S)
    except asyncio.TimeoutError:
        return b"", b""


class BaseExecutor(ABC):
    """Abstract base for script execution."""

    @property
    def is_sandboxed(self) -> bool:
        """Whether code runs in a sandbox (e.g., Docker). Affects library validation."""
        return False

    @abstractmethod
    async def execute(self, script: str, task: TaskDescription, timeout: int = 30) -> Trace:
        """Run a script and return execution trace."""
        ...


class LocalExecutor(BaseExecutor):
    """Runs scripts via subprocess. For development/testing only."""

    # Warn once per process (class flag): tests construct LocalExecutor
    # dozens of times — per-instance warnings would spam.
    _warned_host_exec = False

    def __init__(self) -> None:
        if not LocalExecutor._warned_host_exec:
            LocalExecutor._warned_host_exec = True
            logger.warning(
                "LocalExecutor runs generated scripts directly on the HOST with no "
                "sandbox — development/testing only (benchmark runs use sandboxed "
                "executors)."
            )

    async def execute(self, script: str, task: TaskDescription, timeout: int = 30) -> Trace:
        """Run script locally via bash subprocess.

        Success requires exit_code == 0; the verification script runs only
        after a zero exit.
        """
        start = time.time()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
            f.write(script)
            script_path = f.name

        try:
            proc = await asyncio.create_subprocess_exec(
                "bash", script_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
            # asyncio.wait (NOT wait_for): on timeout the communicate() task is
            # left running, so the output it buffered before the kill survives
            # for the salvage drain instead of being discarded by cancellation.
            comm_task = asyncio.create_task(proc.communicate())
            done, _ = await asyncio.wait({comm_task}, timeout=timeout)
            if not done:
                _kill_process_group(proc)
                stdout_bytes, _ = await _drain_after_kill(comm_task)
                return Trace(
                    task_id=task.task_id,
                    script=script,
                    stdout=stdout_bytes.decode(errors="replace"),
                    stderr="Timeout exceeded",
                    exit_code=-1,
                    success=False,
                    error_summary=f"Script timed out after {timeout}s",
                    duration_s=time.time() - start,
                )
            stdout_bytes, stderr_bytes = comm_task.result()

            stdout = stdout_bytes.decode(errors="replace")
            stderr = stderr_bytes.decode(errors="replace")
            exit_code = proc.returncode or 0

            # Determine success via verification script if available
            success = exit_code == 0
            if success and task.verification_script:
                success = await self._verify(task.verification_script, timeout=10)

            return Trace(
                task_id=task.task_id,
                script=script,
                stdout=stdout,
                stderr=stderr,
                exit_code=exit_code,
                success=success,
                score=1.0 if success else 0.0,
                error_summary=stderr.strip()[:200] if not success else "",
                duration_s=time.time() - start,
            )
        finally:
            Path(script_path).unlink(missing_ok=True)

    async def _verify(self, verification_script: str, timeout: int = 10) -> bool:
        """Run verification script, return True if exits 0."""
        proc = await asyncio.create_subprocess_exec(
            "bash", "-c", verification_script,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        comm_task = asyncio.create_task(proc.communicate())
        done, _ = await asyncio.wait({comm_task}, timeout=timeout)
        if not done:
            _kill_process_group(proc)
            await _drain_after_kill(comm_task)
            return False
        comm_task.result()
        return proc.returncode == 0
