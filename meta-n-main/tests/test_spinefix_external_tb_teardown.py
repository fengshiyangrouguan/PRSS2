"""Regression tests for SPINE findings F6/F7 in ``_external_tb.py``.

Both findings name the SAME async resource leak in
:meth:`_ExternalTBBackend._run_bridge`: ``_force_compose_down(run_label)`` — the
ONLY reaper of a SIGKILLed runner's orphan trial compose container / network —
was called on the hard-timeout branch and the ``CancelledError`` branch but NOT
on the generic ``except Exception`` branch and NOT in the ``finally``, and the
cancel-branch call was UNSHIELDED (a second cancel mid-teardown could abort it).

The fix hoists a shielded, de-duplicated belt-and-suspenders sweep into the
``finally`` (fires only when the runner was launched, did NOT complete cleanly,
and was not already swept), and wraps the cancel-branch sweep in ``asyncio.shield``.

Every test here is LLM-free / offline: no LM Studio, no Docker daemon, no network.
``asyncio.create_subprocess_exec`` is stubbed to a fake process, ``_sigkill_group``
is a no-op, and ``_force_compose_down`` is replaced by a recorder — so no real
``docker`` / runner binary is ever launched.
"""

from __future__ import annotations

import asyncio
import re
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from meta_n.core.external_agents.backend import AgentRunContext, Prompt
from meta_n.core.external_agents.backends import _external_tb
from meta_n.core.external_agents.backends._external_tb import (
    _ExternalTBBackend,
    _RunPrep,
)
from meta_n.core.external_agents.terminated import TerminatedBy


# --------------------------------------------------------------------------- #
# Stubs                                                                        #
# --------------------------------------------------------------------------- #
class _RecordingTB(_ExternalTBBackend):
    """Concrete backend that records every teardown / SIGKILL instead of doing it.

    ``_force_compose_down`` is overridden to append to ``compose_down_calls``
    (rather than shelling out to ``docker``); ``_sigkill_group`` is a no-op that
    counts calls (rather than ``os.killpg`` on a fake pid).
    """

    name = "rec-tb"
    _RUNNER_MODULE = "stub.runner"
    _TRIAL_COMPOSE_PROJECT = "rec-bridge"

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self.compose_down_calls: list[str | None] = []
        self.sigkill_calls = 0

    def _resolve_terminated(self, failure_mode: object) -> TerminatedBy:
        return TerminatedBy.UNKNOWN

    def _extra_request_fields(self, ctx, max_episodes):  # noqa: D401
        return {}

    # Instance overrides of the base static/class helpers so the bridge control
    # flow runs without touching real processes or docker.
    def _sigkill_group(self, proc) -> None:  # type: ignore[override]
        self.sigkill_calls += 1

    async def _force_compose_down(self, run_label=None) -> None:  # type: ignore[override]
        self.compose_down_calls.append(run_label)


def _make_backend() -> _RecordingTB:
    return _RecordingTB(
        model="openai/dummy",
        api_base="http://localhost:0",
        venv_python="/nonexistent/python",
        runner_dir="/nonexistent",
        tasks_dir="/nonexistent",
    )


class _FakeRunnerProc:
    """Stand-in for the launched runner ``asyncio.subprocess.Process``.

    ``mode`` picks how ``communicate()`` behaves:
      * ``"ok"``      → returns cleanly (the happy path; sets returncode 0),
      * ``"raise"``   → raises a non-timeout/non-cancel error (``except Exception``),
      * ``"cancel"``  → raises ``asyncio.CancelledError`` (cancel branch),
      * ``"timeout"`` → sleeps past the hard timeout (hard-timeout branch).
    """

    def __init__(self, *, mode: str) -> None:
        self.mode = mode
        self.returncode: int | None = None
        self.pid = 424242

    async def communicate(self):
        if self.mode == "ok":
            self.returncode = 0
            return b"", b""
        if self.mode == "raise":
            raise RuntimeError("boom during communicate")
        if self.mode == "cancel":
            raise asyncio.CancelledError()
        if self.mode == "timeout":
            await asyncio.sleep(5)  # outlives the tiny hard timeout below
            self.returncode = 0
            return b"", b""
        raise AssertionError(f"unknown mode {self.mode!r}")  # pragma: no cover

    async def wait(self):
        if self.returncode is None:
            self.returncode = -9
        return self.returncode


def _patch_create(monkeypatch, proc: _FakeRunnerProc) -> None:
    async def fake_create(*args, **kwargs):
        return proc

    monkeypatch.setattr(
        _external_tb.asyncio, "create_subprocess_exec", fake_create
    )


def _make_ctx(tmp_path: Path) -> AgentRunContext:
    ws = SimpleNamespace(
        task_id="task-1", staged_files={}, run_label="rl-123", agent_pid=None
    )
    return AgentRunContext(
        instruction="do the thing",
        prompt=Prompt(system_suffix="", prefix=""),
        workspace=ws,
        time_limit_s=1.0,
        max_turns=0,
        token_budget=0,
        max_budget_usd=0.0,
        logging_dir=tmp_path,
    )


def _drive(backend: _RecordingTB, ctx: AgentRunContext):
    """Run one ``_run_bridge`` to completion under a fresh event loop."""
    return asyncio.run(
        backend._run_bridge(ctx, None, None, time.time(), _RunPrep())
    )


# --------------------------------------------------------------------------- #
# F6/F7 — the generic ``except Exception`` path must still sweep (fail pre-fix) #
# --------------------------------------------------------------------------- #
def test_generic_exception_path_force_compose_down(monkeypatch, tmp_path):
    """A non-timeout/non-cancel fault must still sweep the killed runner's compose.

    Pre-fix the generic ``except Exception`` branch returned a degraded result
    WITHOUT any ``_force_compose_down`` (the ``finally`` had none either), leaking
    the SIGKILLed runner's orphan trial container/network — so this asserted count
    was 0 and the test failed. Post-fix the ``finally`` catch-all sweeps exactly once.
    """
    backend = _make_backend()
    _patch_create(monkeypatch, _FakeRunnerProc(mode="raise"))
    ctx = _make_ctx(tmp_path)

    result = _drive(backend, ctx)

    # Never raised — degraded result returned (never-raise contract preserved).
    assert result.failure_mode == "env_error"
    # The leak is closed: exactly one belt-and-suspenders sweep, not zero.
    assert len(backend.compose_down_calls) == 1


# --------------------------------------------------------------------------- #
# Happy path stays byte-for-byte: a clean run is NEVER swept                    #
# --------------------------------------------------------------------------- #
def test_happy_path_no_force_compose_down(monkeypatch, tmp_path):
    """A cleanly-completed runner (which tore its own containers down) is untouched."""
    backend = _make_backend()
    _patch_create(monkeypatch, _FakeRunnerProc(mode="ok"))
    ctx = _make_ctx(tmp_path)
    # Give the runner a genuine success result so this is a true happy path.
    (tmp_path / backend._RESULT_FILENAME).write_text(
        '{"ok": true, "is_resolved": true, "reward": 1.0}'
    )

    result = _drive(backend, ctx)

    assert result.native_resolved is True
    # No sweep on the happy path — the runner already cleaned up after itself.
    assert backend.compose_down_calls == []


# --------------------------------------------------------------------------- #
# De-dup: the timeout branch sweeps inline; the ``finally`` must NOT repeat it  #
# --------------------------------------------------------------------------- #
def test_timeout_path_sweeps_exactly_once(monkeypatch, tmp_path):
    """The hard-timeout branch sweeps once; the ``finally`` de-dups via ``torn_down``."""
    backend = _make_backend()
    # Force the hard wall to fire well before the fake runner's 5s sleep.
    monkeypatch.setattr(backend, "_hard_timeout", lambda soft: 0.02)
    _patch_create(monkeypatch, _FakeRunnerProc(mode="timeout"))
    ctx = _make_ctx(tmp_path)

    result = _drive(backend, ctx)

    # No result file was written, so the zeroed hard-timeout result is returned.
    assert result.failure_mode == "agent_timeout"
    # Exactly one sweep — the inline timeout sweep, NOT doubled by the finally.
    assert len(backend.compose_down_calls) == 1


# --------------------------------------------------------------------------- #
# Cancel branch: still sweeps, and CancelledError still propagates              #
# --------------------------------------------------------------------------- #
def test_cancel_path_sweeps_and_propagates(monkeypatch, tmp_path):
    """Cooperative cancellation sweeps exactly once and re-raises CancelledError."""
    backend = _make_backend()
    _patch_create(monkeypatch, _FakeRunnerProc(mode="cancel"))
    ctx = _make_ctx(tmp_path)

    with pytest.raises(asyncio.CancelledError):
        _drive(backend, ctx)

    # One sweep from the cancel branch; the finally is de-duped by torn_down.
    assert len(backend.compose_down_calls) == 1


# --------------------------------------------------------------------------- #
# Source-level guards for the shield + reap-count invariants                    #
# --------------------------------------------------------------------------- #
def test_cancel_and_finally_sweeps_are_shielded():
    """Both the cancel-branch and finally sweeps wrap the force-down in shield.

    The shield is what keeps a SECOND cancel from aborting the force-down
    mid-teardown; it is hard to observe behaviorally, so lock it in via source.
    Pre-fix there was exactly one (unshielded) ``await self._force_compose_down``
    in the cancel branch and none shielded.
    """
    src = Path(_external_tb.__file__).read_text()
    shielded = re.findall(
        r"asyncio\.shield\(self\._force_compose_down\(run_label\)\)", src
    )
    # One in the cancel branch, one in the finally catch-all.
    assert len(shielded) == 2, f"expected 2 shielded sweeps, found {len(shielded)}"
    # The de-dup flag is present.
    assert "torn_down" in src


def test_proc_wait_reaps_still_bounded_and_unchanged():
    """The three bounded ``proc.wait()`` reaps are untouched by this fix."""
    src = Path(_external_tb.__file__).read_text()
    assert "await proc.wait()" not in src  # no bare (unbounded) reaps
    bounded = re.findall(
        r"asyncio\.wait_for\(\s*proc\.wait\(\),\s*timeout=_TEARDOWN_WAIT_S", src
    )
    assert len(bounded) == 3, f"expected 3 bounded reaps, found {len(bounded)}"
