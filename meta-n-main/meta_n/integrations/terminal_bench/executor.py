"""TerminalBenchExecutor — runs bash scripts in task Docker containers.

Split verbatim out of the old single-file
``meta_n/integrations/terminal_bench.py`` module.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import tempfile
import time
from pathlib import Path
from uuid import uuid4

from meta_n.core.base_executor import BaseExecutor
from meta_n.core.meta_layer import TaskDescription, Trace
from meta_n.integrations.terminal_bench.adapter import TerminalBenchAdapter
from meta_n.integrations.terminal_bench.compose import (
    _ExecResult,
    _sanitize_compose_name,
)

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
# Executor
# ---------------------------------------------------------------------------

class TerminalBenchExecutor(BaseExecutor):
    """Executor that runs bash scripts in task-specific Docker containers.

    Each execute() call:
    1. Ensures the Docker image is built (cached after first call)
    2. Starts a fresh container from the cached image
    3. Injects the script and runs it
    4. Runs the verifier (test.sh)
    5. Reads the reward from the bind-mounted host directory
    6. Tears down the container (image persists)

    Concurrency: Each execute() is fully independent — unique compose project,
    unique temp dir, shared cached image (read-only). The orchestrator controls
    parallelism via ``asyncio.Semaphore(config.parallel)``; set ``--parallel N``
    to run N containers simultaneously.
    """

    def __init__(self, adapter: TerminalBenchAdapter):
        self.adapter = adapter
        self.adapter._executor = self  # back-ref for adapter.evaluate()
        self._docker_checked = False

    @property
    def is_sandboxed(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Output extraction helpers
    # ------------------------------------------------------------------

    _ERROR_MARKERS = re.compile(
        r"(?:^Traceback \(most recent call last\):"
        r"|(?:[Ee]rror|Exception|FAILED|command not found)[\s:])",
        re.MULTILINE,
    )

    _TEST_MARKERS = re.compile(
        r"(?:^=+ .*=+$"               # pytest section headers
        r"|^-+ .* -+$"                 # pytest subsection dividers
        r"|^(?:FAILED|PASSED|ERROR)\b" # pytest status keywords
        r"|^\d+ (?:passed|failed)"     # pytest summary line
        r"|^Traceback )",              # Python traceback
        re.MULTILINE,
    )

    @staticmethod
    def _extract_error_summary(stderr: str, max_len: int = 300) -> str:
        """Extract the most diagnostic portion of stderr.

        Scans for well-known error markers (Traceback, Error:, FAILED, etc.)
        and returns from the first marker forward (head of the matched section).
        Falls back to the tail of stderr if no markers are found.
        """
        if not stderr:
            return ""
        match = TerminalBenchExecutor._ERROR_MARKERS.search(stderr)
        if match:
            # Take from the marker forward — the error headline is at the start
            relevant = stderr[match.start():]
            return relevant[:max_len].strip()
        # No marker: tail is more likely to have useful content than head
        return stderr[-max_len:].strip()

    @staticmethod
    def _extract_eval_feedback(raw_output: str, max_len: int = 3000) -> str:
        """Extract test results from verifier output, stripping preamble noise.

        TB2 verifiers run apt-get/pip/uv setup before pytest. This method
        finds the first test-related marker and returns from there to the end.
        Falls back to the tail of the output if no markers are found.
        """
        if not raw_output:
            return ""
        match = TerminalBenchExecutor._TEST_MARKERS.search(raw_output)
        if match:
            relevant = raw_output[match.start():]
            return relevant[-max_len:].strip()
        return raw_output[-max_len:].strip()

    async def _preflight(self) -> None:
        """Check that Docker daemon is reachable. Called once."""
        if self._docker_checked:
            return
        proc = await asyncio.create_subprocess_exec(
            "docker", "info",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(
                "Docker daemon is not running. "
                "Please start Docker and try again."
            )
        self._docker_checked = True

    async def execute(
        self, script: str, task: TaskDescription, timeout: int = 3600
    ) -> Trace:
        """Run a bash script in a Docker container and verify the result."""
        # Legacy task.yaml tasks run ONLY through the external spine (the runner
        # owns their env + verifier); the native compose flow has no
        # environment/ + task.toml to build from and would fail at image build.
        if task.metadata.get("layout") == "legacy_yaml":
            raise RuntimeError(
                f"Task {task.task_id} uses the legacy terminal-bench task.yaml "
                "layout, which the native TerminalBenchExecutor cannot run. Use "
                "the external-agent spine: --base-solver builtin, openhands, or "
                "terminus2."
            )
        start = time.time()
        task_id = task.task_id
        task_dir = Path(task.metadata["task_dir"])
        task_timeout = task.metadata.get("timeout_sec", 1800)
        verifier_timeout = task.metadata.get("verifier_timeout_sec", 900)

        # 0. Docker preflight (once)
        try:
            await self._preflight()
        except RuntimeError as e:
            return Trace(
                task_id=task_id, script=script,
                stderr=str(e), exit_code=-1, success=False, score=0.0,
                error_summary=str(e)[:200],
                duration_s=time.time() - start,
            )

        # 1. Ensure image is built and cached
        try:
            await self.adapter._ensure_image(task_id)
        except Exception as e:
            logger.error("Image build failed for %s: %s", task_id, e)
            return Trace(
                task_id=task_id,
                script=script,
                stderr=f"Docker image build failed: {e}",
                exit_code=-1,
                success=False,
                score=0.0,
                error_summary=f"Image build failed: {str(e)[:150]}",
                duration_s=time.time() - start,
            )

        # 2. Create unique session and host log directory.
        # Force mode 0o700 explicitly: host_logs gets bind-mounted into the
        # container and contains reward.json + agent traces. On a shared
        # host with a permissive umask the default mkdtemp could leak this.
        session = _sanitize_compose_name(f"tb2-eval-{task_id}-{uuid4().hex[:8]}")
        host_logs = Path(tempfile.mkdtemp(prefix="tb2_eval_"))
        os.chmod(host_logs, 0o700)
        (host_logs / "verifier").mkdir()
        (host_logs / "agent").mkdir()

        compose_files = self.adapter._get_compose_files(task_id, prebuilt=True)
        env = self.adapter._compose_env(task_id, host_logs_path=str(host_logs))
        project_dir = task_dir / "environment"

        exec_result = _ExecResult()
        verifier_result = _ExecResult()
        verifier_output = ""
        score = 0.0

        try:
            # 3. Start fresh container from cached image
            await _tb_pkg._run_compose_command(
                compose_files=compose_files,
                project_name=session,
                project_dir=project_dir,
                command=["up", "-d", "--wait"],
                env_overrides=env,
                timeout_sec=300,
            )

            # 4. Make ONLY the agent log dir writable (the solve script may
            # write traces to /logs/agent). The held-out verifier tests are NOT
            # staged and the reward dir /logs/verifier is deliberately left
            # locked until AFTER the solve script runs (step 6b) — otherwise the
            # solver could read the test set and overfit, or pre-write a forged
            # /logs/verifier/reward.json. This mirrors harbor's flow: the agent
            # phase sees no /tests and no writable verifier dir.
            await _tb_pkg._compose_exec(
                compose_files, session, project_dir,
                "chmod 777 /logs/agent",
                env_overrides=env,
                user="root",
                timeout_sec=60,
            )

            # 5. Inject script as file (avoids shell quoting issues)
            script_file = host_logs / "solve.sh"
            script_file.write_text(script)
            await _tb_pkg._run_compose_command(
                compose_files, session, project_dir,
                ["cp", str(script_file), "main:/tmp/solve.sh"],
                env_overrides=env,
                timeout_sec=120,
            )

            # 6. Execute the solution script (with no /tests present and the
            # reward dir still locked — no test-set leak, no forgeable reward).
            exec_result = await _tb_pkg._compose_exec(
                compose_files, session, project_dir,
                "bash /tmp/solve.sh",
                env_overrides=env,
                timeout_sec=task_timeout,
            )

            # 6a. HOST-side anti-forgery sweep: /logs/verifier was "locked" only
            # by file modes, which do not constrain a solve script running as
            # the image's default user when that user is root
            # (CAP_DAC_OVERRIDE). Anything present now was written during the
            # solve phase and can only be tampering, so purge it from the host
            # (the container cannot block host-side unlinks) before the
            # verifier is unlocked. Residual (out of scope): a background
            # process left by solve.sh, or root tampering with /tests after
            # staging — closing those requires verifying in a fresh container.
            removed = _tb_pkg._purge_verifier_dir(host_logs / "verifier")
            if removed:
                logger.warning(
                    "Task %s: purged %d pre-verifier file(s) from %s (possible "
                    "reward forgery by the solve phase): %s",
                    task_id, len(removed), host_logs / "verifier", removed,
                )

            # 6b. Only NOW, after the agent phase has finished, stage the
            # held-out verifier tests and make the reward dir writable.
            await _tb_pkg._compose_exec(
                compose_files, session, project_dir,
                "chmod 777 /logs/verifier",
                env_overrides=env,
                user="root",
                timeout_sec=60,
            )
            tests_dir = task_dir / "tests"
            if tests_dir.exists():
                await _tb_pkg._run_compose_command(
                    compose_files, session, project_dir,
                    ["cp", f"{tests_dir}/.", "main:/tests/"],
                    env_overrides=env,
                    timeout_sec=300,
                )
                # Ensure test.sh is executable
                await _tb_pkg._compose_exec(
                    compose_files, session, project_dir,
                    "chmod -R +x /tests/",
                    env_overrides=env,
                    user="root",
                    timeout_sec=60,
                )

            # 7. Run verification (output goes to bind-mounted file)
            verifier_result = await _tb_pkg._compose_exec(
                compose_files, session, project_dir,
                "/tests/test.sh > /logs/verifier/test-stdout.txt 2>&1; exit 0",
                env_overrides=env,
                timeout_sec=verifier_timeout,
            )

            # 8. Read reward and verifier output from bind-mounted host dir
            score = _tb_pkg._read_reward(host_logs / "verifier")
            verifier_stdout_path = host_logs / "verifier" / "test-stdout.txt"
            verifier_output = ""
            if verifier_stdout_path.exists():
                raw = verifier_stdout_path.read_text(errors="replace")
                verifier_output = self._extract_eval_feedback(raw)

        except Exception as e:
            logger.warning(
                "Execution error for %s (session=%s): %s",
                task_id, session, e,
            )
            if not exec_result.stderr:
                exec_result = _ExecResult(
                    stderr=f"Execution error: {e}", return_code=-1,
                )

        finally:
            # 9. Always tear down — NO --rmi, preserving the cached image.
            # --volumes reaps this run's anonymous/named volumes (the compose
            # project is unique per execute(), so none are shared across runs).
            try:
                await _tb_pkg._run_compose_command(
                    compose_files, session, project_dir,
                    ["down", "--remove-orphans", "--volumes"],
                    env_overrides=env,
                    check=False,
                    timeout_sec=120,
                )
            except Exception as e:
                logger.warning("Cleanup failed for session %s: %s", session, e)

            shutil.rmtree(host_logs, ignore_errors=True)

        # 10. Build Trace
        duration = time.time() - start

        # eval_feedback: prefer file-based verifier output (always populated),
        # fall back to compose exec stderr (e.g. on timeout)
        eval_feedback = verifier_output if verifier_output else verifier_result.stderr

        return Trace(
            task_id=task_id,
            script=script,
            stdout=exec_result.stdout[:10_000],
            stderr=exec_result.stderr[:5_000],
            exit_code=exec_result.return_code,
            success=score > 0.0,
            score=score,
            error_summary=(
                self._extract_error_summary(exec_result.stderr)
                if score == 0.0 and exec_result.stderr
                else ""
            ),
            eval_feedback=eval_feedback,
            duration_s=duration,
        )
