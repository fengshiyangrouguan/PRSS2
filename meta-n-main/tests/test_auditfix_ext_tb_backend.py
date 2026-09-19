"""Regression tests for audit fixes 54, 55, 69 in ``_external_tb.py``.

All tests are LLM-free / offline: no LM Studio, no Docker daemon, no network.
The docker-teardown tests stub ``asyncio.create_subprocess_exec`` /
``asyncio.wait_for`` so no real ``docker`` binary is ever launched.

Each test fails on the pre-fix code and passes after the fix:

* 54 — every post-SIGKILL ``proc.wait()`` reap is bounded by
  ``asyncio.wait_for(..., timeout=_TEARDOWN_WAIT_S)`` so a child stuck in
  uninterruptible sleep cannot hang the timeout/cancel/finally unwind. Pre-fix
  the reaps were bare ``await proc.wait()`` and ``_TEARDOWN_WAIT_S`` did not exist.
* 55 — ``_force_compose_down`` kills a hung docker CLI child on a ``wait_for``
  timeout instead of leaking it. Pre-fix the child was never killed.
* 69 — the synthesized-degraded result dicts declare
  ``attribution_available=False`` (unmeasurable), so ``_to_run_result`` does not
  default the missing key to ``True`` ("attribution-wired"). Pre-fix the key was
  absent and defaulted to ``True``.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from meta_n.core.external_agents.backends import _external_tb
from meta_n.core.external_agents.backends._external_tb import (
    _TEARDOWN_WAIT_S,
    _ExternalTBBackend,
)
from meta_n.core.external_agents.terminated import TerminatedBy


# --------------------------------------------------------------------------- #
# Stubs                                                                        #
# --------------------------------------------------------------------------- #
class _StubTB(_ExternalTBBackend):
    """Minimal concrete backend so the base result helpers are exercisable."""

    name = "stub-tb"
    _RUNNER_MODULE = "stub.runner"
    _TRIAL_COMPOSE_PROJECT = "stub-bridge"

    def _resolve_terminated(self, failure_mode: object) -> TerminatedBy:
        return TerminatedBy.UNKNOWN

    def _extra_request_fields(self, ctx, max_episodes):  # noqa: D401
        return {}


def _make_backend() -> _StubTB:
    return _StubTB(
        model="openai/dummy",
        api_base="http://localhost:0",
        venv_python="/nonexistent/python",
        runner_dir="/nonexistent",
        tasks_dir="/nonexistent",
    )


class _FakeProc:
    """A stand-in for ``asyncio.subprocess.Process`` that records ``kill()``."""

    def __init__(self, out: bytes = b"") -> None:
        self._out = out
        self.returncode: int | None = None
        self.killed = False
        self.pid = 4321

    async def communicate(self):
        return self._out, b""

    async def wait(self):
        self.returncode = 0
        return 0

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9


# --------------------------------------------------------------------------- #
# Finding 54 — bounded post-SIGKILL reaps                                      #
# --------------------------------------------------------------------------- #
def test_teardown_wait_constant_is_finite_positive():
    # The constant did not exist pre-fix; importing it would AttributeError.
    assert isinstance(_TEARDOWN_WAIT_S, (int, float))
    assert 0 < _TEARDOWN_WAIT_S < float("inf")


def test_no_unbounded_proc_wait_reap():
    src = Path(_external_tb.__file__).read_text()
    # Pre-fix: three bare ``await proc.wait()`` reaps.
    assert "await proc.wait()" not in src
    # Post-fix: all three reaps go through the bounded wait_for wrapper.
    bounded = re.findall(
        r"asyncio\.wait_for\(\s*proc\.wait\(\),\s*timeout=_TEARDOWN_WAIT_S",
        src,
    )
    assert len(bounded) == 3, f"expected 3 bounded reaps, found {len(bounded)}"


# --------------------------------------------------------------------------- #
# Finding 55 — kill a hung docker CLI child on wait_for timeout                #
# --------------------------------------------------------------------------- #
def _install_docker_fakes(monkeypatch, *, ps_out: bytes, net_out: bytes,
                          timeout_on: set[int]):
    """Stub docker subprocess creation + wait_for; return the created procs."""
    created: dict[str, _FakeProc] = {}

    async def fake_create(*args, **kwargs):
        # args[0] == "docker"; dispatch by the sub-command.
        if args[1] == "ps":
            proc = _FakeProc(out=ps_out)
            created["ps"] = proc
        elif args[1] == "rm":
            proc = _FakeProc()
            created["rm"] = proc
        elif args[1] == "network" and args[2] == "ls":
            proc = _FakeProc(out=net_out)
            created["netls"] = proc
        elif args[1] == "network" and args[2] == "rm":
            proc = _FakeProc()
            created["netrm"] = proc
        else:  # pragma: no cover - defensive
            proc = _FakeProc()
        return proc

    state = {"n": 0}

    async def fake_wait_for(coro, timeout):
        state["n"] += 1
        if state["n"] in timeout_on:
            coro.close()  # avoid 'coroutine never awaited'
            raise asyncio.TimeoutError()
        return await coro

    monkeypatch.setattr(_external_tb.asyncio, "create_subprocess_exec", fake_create)
    monkeypatch.setattr(_external_tb.asyncio, "wait_for", fake_wait_for)
    return created


def test_force_compose_down_kills_hung_docker_ps(monkeypatch):
    created = _install_docker_fakes(
        monkeypatch, ps_out=b"", net_out=b"", timeout_on={1}
    )
    # Must not raise even though `docker ps` wedged.
    asyncio.run(_ExternalTBBackend._force_compose_down("lbl"))
    assert created["ps"].killed is True


def test_force_compose_down_kills_hung_docker_rm(monkeypatch):
    # ps returns one container id so `docker rm` is launched; rm (call #2) wedges.
    created = _install_docker_fakes(
        monkeypatch, ps_out=b"cid123\n", net_out=b"", timeout_on={2}
    )
    asyncio.run(_ExternalTBBackend._force_compose_down("lbl"))
    assert created["ps"].killed is False  # ps completed cleanly
    assert created["rm"].killed is True  # wedged rm was killed


def test_force_compose_down_kills_hung_network_rm(monkeypatch):
    # ps returns no ids (rm skipped); netls succeeds (#2), `network rm` (#3) wedges.
    created = _install_docker_fakes(
        monkeypatch, ps_out=b"", net_out=b"net1\n", timeout_on={3}
    )
    asyncio.run(_ExternalTBBackend._force_compose_down("lbl"))
    assert "rm" not in created
    assert created["netls"].killed is False
    assert created["netrm"].killed is True


# --------------------------------------------------------------------------- #
# Finding 69 — synthesized-degraded dicts are unmeasurable (False)            #
# --------------------------------------------------------------------------- #
def test_read_result_parsed_nonobject_marks_attribution_false(tmp_path):
    backend = _make_backend()
    res = tmp_path / "result.json"
    res.write_text("null")  # valid JSON, non-object body -> parse_error
    data, parsed_ok = backend._read_result_parsed(res, b"boom-traceback")
    assert parsed_ok is False
    assert data["failure_mode"] == "parse_error"
    assert data["attribution_available"] is False


def test_read_result_parsed_missing_file_marks_attribution_false(tmp_path):
    backend = _make_backend()
    res = tmp_path / "does_not_exist.json"
    data, parsed_ok = backend._read_result_parsed(res, b"")
    assert parsed_ok is False
    assert data["failure_mode"] == "env_error"
    assert data["attribution_available"] is False


@pytest.mark.parametrize("body,fm", [("null", "parse_error"), (None, "env_error")])
def test_to_run_result_degraded_attribution_false(tmp_path, body, fm):
    backend = _make_backend()
    res = tmp_path / "result.json"
    if body is not None:
        res.write_text(body)
    data, _ = backend._read_result_parsed(res, b"")
    assert data["failure_mode"] == fm
    ctx = SimpleNamespace(logging_dir=tmp_path)
    result = backend._to_run_result(data, ctx, 0.0)
    # Pre-fix the missing key defaulted to True ("attribution-wired"); the honest
    # value for a no-output degraded row is False (unmeasurable).
    assert result.attribution_available is False
