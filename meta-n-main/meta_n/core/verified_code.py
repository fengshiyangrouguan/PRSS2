"""Forensic improvement #2 — VERIFIED ``code_library`` channel (build-only).

The code channel is the ONLY one that can add capability the base model lacks
(the FEAL proof: a correct ``solve_feal_key5`` helper lifts a hard-0 task to 1.0),
yet today Ω helpers are generic shells the solver re-derives inline (0/6 adoption),
so the channel merely ties best-of-N. This module supplies the VERIFY-THEN-INJECT
gate: after Ω produces a ``code_library`` helper, EXECUTE it in a ``--network none``
sandbox against a held-out check and KEEP it in the injection ONLY if it passes —
otherwise DROP it so dead/wrong helpers never reach a candidate.

DOCKER-ONLY INVARIANT: the ONLY place an Ω-generated helper actually RUNS is inside
the ``--network none --read-only`` container driven by
:meth:`SandboxedHeldoutVerifier.verify` (lifted verbatim from
``scripts/experiments/metan_feal_gap_test.py::_helper_recovers_in_container``).
Untrusted Ω code NEVER executes on the host.

HONEST SCOPE: build-only / UNMEASURED. There is no live value-oracle bed in this
build except FEAL, so for ordinary benchmarks the orchestrator falls back to the
capability-preserving :class:`StubHeldoutVerifier` (KEEP + flag UNVERIFIED), which
is the clean seam where a real :class:`SandboxedHeldoutVerifier` would plug in.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Callable, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class VerifyResult:
    """Outcome of a single held-out helper check.

    ``ran_in_sandbox`` is load-bearing provenance: ``True`` means the helper was
    actually executed inside the ``--network none`` container against a real
    held-out predicate; ``False`` means a capability-preserving STUB kept it
    without a held-out harness (so the kept-helper decision is honest about
    whether it was verified).
    """

    passed: bool
    evidence: str = ""
    ran_in_sandbox: bool = False


@runtime_checkable
class HeldoutVerifier(Protocol):
    """Held-out verifier seam consumed by the orchestrator's verify-gate."""

    def verify(
        self,
        name: str,
        source: str,
        task_id: str,
        context_sources: list[str],
    ) -> VerifyResult:
        ...


class StubHeldoutVerifier:
    """Default fallback for benches with no held-out value-oracle.

    KEEPS every helper (capability-preserving — never silently strips capability
    where we cannot verify) but flags it ``ran_in_sandbox=False`` / UNVERIFIED.
    This is the explicit seam: a family with a real oracle overrides
    ``BenchmarkAdapter.make_heldout_verifier`` to return a
    :class:`SandboxedHeldoutVerifier` instead, and the drop-dead-helper benefit
    materializes there.
    """

    def verify(
        self,
        name: str,
        source: str,
        task_id: str,
        context_sources: list[str],
    ) -> VerifyResult:
        logger.info(
            "verified_code: no held-out harness for helper '%s' on task '%s' — "
            "KEEPING UNVERIFIED (stub seam)",
            name,
            task_id,
        )
        return VerifyResult(
            passed=True,
            evidence="no held-out harness (stub) — kept UNVERIFIED",
            ran_in_sandbox=False,
        )


def _docker_run_json(argv: list[str], timeout_s: float, container: str) -> dict:
    """Run ``argv`` (a ``docker run`` invocation) and parse its last JSON stdout line.

    Mirrors ``metan_feal_gap_test._helper_recovers_in_container``'s one-JSON-line
    protocol. Never raises: a timeout / non-JSON / crash all map to a structured
    ``{"passed": False, ...}`` dict so the verify-gate fails CLOSED (drop the
    unverifiable helper) rather than crashing the run.
    """
    try:
        proc = subprocess.run(
            argv, capture_output=True, text=True, timeout=timeout_s
        )
    except subprocess.TimeoutExpired:
        try:
            subprocess.run(
                ["docker", "rm", "-f", container],
                capture_output=True,
                timeout=30,
            )
        except Exception:  # bounded teardown — a wedged daemon must not block forever
            pass
        return {"status": "timeout", "passed": False}
    except Exception as exc:  # pragma: no cover - docker-absent host
        return {"status": "exec_error", "passed": False, "error": str(exc)}
    last = ""
    for line in (proc.stdout or "").splitlines():
        if line.strip():
            last = line.strip()
    try:
        result = json.loads(last)
        if not isinstance(result, dict):
            # Valid JSON but not an object (bare true / 42 / [..]): it has no
            # .setdefault, so route it to the same fail-closed branch as non-JSON
            # rather than letting an AttributeError escape the "Never raises" contract.
            raise ValueError("last stdout line is not a JSON object")
        result.setdefault("status", "ok")
        return result
    except (json.JSONDecodeError, ValueError):
        return {
            "status": "no_json",
            "passed": False,
            "raw_stdout": (proc.stdout or "")[-2000:],
            "raw_stderr": (proc.stderr or "")[-2000:],
        }


class SandboxedHeldoutVerifier:
    """Execute an Ω helper inside a ``--network none`` container vs a held-out check.

    The container layout mirrors the FEAL omega-loop sandbox: the candidate file
    is ``<instruments> + <context helpers> + <the helper under test>`` (so the
    helper can call lower-layer helpers / instruments by bare name), the
    benchmark-supplied ``runner_src`` is the HUMAN value-oracle (the only code that
    reads ground truth, and it reads it ONLY inside the container — never on host,
    never shown to Ω), and ``success_predicate`` maps the runner's emitted JSON to
    pass/fail.

    ``run_container`` is an injectable seam: it defaults to the real
    :func:`_docker_run_json`, but unit tests pass a fake that records the argv
    (asserting ``--network none`` / ``--read-only``) and returns canned JSON WITHOUT
    Docker.
    """

    def __init__(
        self,
        *,
        image: str,
        runner_src: str,
        mounts: Optional[list[tuple[str, str]]] = None,
        instruments_src: str = "",
        success_predicate: Optional[Callable[[dict], bool]] = None,
        runner_args: Optional[list[str]] = None,
        wall_clock_s: float = 360.0,
        memory: str = "1g",
        cpus: str = "2",
        pids_limit: str = "256",
        run_container: Optional[Callable[[list[str], float, str], dict]] = None,
    ):
        self.image = image
        self.runner_src = runner_src
        self.mounts = list(mounts or [])  # (host_path, container_path), bound :ro
        self.instruments_src = instruments_src
        self.success_predicate = success_predicate or (lambda r: bool(r.get("passed")))
        self.runner_args = list(runner_args or [])
        self.wall_clock_s = wall_clock_s
        self.memory = memory
        self.cpus = cpus
        self.pids_limit = pids_limit
        self._run_container = run_container or _docker_run_json

    def verify(
        self,
        name: str,
        source: str,
        task_id: str,
        context_sources: list[str],
    ) -> VerifyResult:
        tmpdir = tempfile.mkdtemp(prefix="metan_verify_")
        ran_in_sandbox = False
        try:
            candidate = "\n\n".join(
                [self.instruments_src]
                + [s for s in (context_sources or []) if s]
                + [source or ""]
            ).strip()
            (Path(tmpdir) / "_candidate.py").write_text(candidate)
            (Path(tmpdir) / "_runner.py").write_text(self.runner_src)
            container = f"metan-verify-{os.getpid()}-{uuid.uuid4().hex[:8]}"
            argv = [
                "docker", "run", "--rm", "--name", container,
                "--network", "none",                          # HARD isolation (Docker-only invariant)
                "--read-only", "--tmpfs", "/tmp:rw,exec,size=64m",
                "--memory", self.memory, "--cpus", self.cpus,
                "--pids-limit", self.pids_limit,
                "-v", f"{tmpdir}/_candidate.py:/app/_candidate.py:ro",
                "-v", f"{tmpdir}/_runner.py:/app/_runner.py:ro",
            ]
            for host_path, cont_path in self.mounts:
                argv += ["-v", f"{host_path}:{cont_path}:ro"]
            argv += [
                "-w", "/app", self.image,
                "python", "/app/_runner.py", name, *self.runner_args,
            ]
            ran_in_sandbox = True  # the container invocation has now started
            result = self._run_container(argv, self.wall_clock_s, container)
            # An "exec_error" status means the `docker run` process itself never
            # executed (docker-absent host / spawn failure caught inside
            # _docker_run_json) — the helper never ran in the sandbox, so the
            # provenance bit must be honest (same principle as the audit-#52
            # host-side-crash fix). "timeout"/"no_json"/"ok" keep True: the
            # container was at least invoked.
            ran_in_sandbox = result.get("status") != "exec_error"
            passed = bool(self.success_predicate(result))
            return VerifyResult(
                passed=passed,
                evidence=json.dumps(result)[:2000],
                ran_in_sandbox=ran_in_sandbox,
            )
        except Exception as exc:  # never let a verifier crash the run
            logger.warning(
                "verified_code: sandbox verify crashed for helper '%s': %s",
                name, exc,
            )
            return VerifyResult(
                passed=False,
                evidence=f"verifier crash: {exc}",
                ran_in_sandbox=ran_in_sandbox,
            )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
