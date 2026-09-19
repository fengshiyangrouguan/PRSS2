"""CO-Bench spine env provider — agent_pid stamp + host-FS staging guard.

Pure unit coverage for the no-Docker CO-Bench external-agent env (plan §5.9),
exercising two fixes that are off the legacy ``--base-solver builtin`` path but
live on the spine ``openhands + co_bench`` path:

* **#6 — the in-process hard_kill stamp lands.** ``_COBenchEnv.workspace_handle``
  must be the env object itself (parity with TB's
  ``_TBTerminus2Env.workspace_handle = self``) so the backend's
  ``setattr(ctx.workspace, "agent_pid", pid)`` (openhands.py) writes onto the
  slot the lease's synchronous ``hard_kill`` reads. The handle must still be
  ``str``/``os.fspath``-able to the host workspace dir so the backend's
  path-stringifying callsites keep the same value the old ``str`` handle gave.
  Negative control: a bare ``str`` handle silently drops the stamp under the
  backend's ``contextlib.suppress`` — proving the str variant was the dead path.

* **#4 — host-FS staging rejects path traversal.** ``COBenchEnvProvider`` writes
  to the BARE HOST (no container), so an absolute or ``..``-traversing staged
  key must be skipped-with-warning (never escape the workspace root), matching
  ``OpenHandsBackend.stage_files`` and the container-staging writers.

Touches no Docker / no ``openhands`` install — only ``meta_n`` + stdlib.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
from pathlib import Path

import pytest

from meta_n.integrations.co_bench import (
    COBenchAdapter,
    COBenchEnvProvider,
    _COBenchEnv,
)


# ---------------------------------------------------------------------------
# #6 — the agent_pid stamp lands on the env via workspace_handle = self
# ---------------------------------------------------------------------------
def test_cobench_env_handle_is_self(tmp_path):
    """``workspace_handle`` is the env object itself (the stamp target)."""
    env = _COBenchEnv(tmp_path / "workspace")
    assert env.workspace_handle is env


def test_cobench_env_stringifies_to_workspace_root(tmp_path):
    """str()/os.fspath()/Path() of the handle still resolve the host path.

    The backend stringifies ``ctx.workspace`` (e.g. ``str(getattr(ctx.workspace,
    "workspace_handle", env))`` and ``Path(str(...))``) to address the host
    workspace; that value must be byte-identical to the old ``str(workspace_root)``
    handle so nothing downstream changes.
    """
    ws = tmp_path / "workspace"
    env = _COBenchEnv(ws)
    assert str(env) == str(ws)
    assert os.fspath(env) == str(ws)
    assert Path(env) == ws
    # The handle (== env) stringifies the same way the old str handle did.
    handle = getattr(env, "workspace_handle", env)
    assert str(handle) == str(ws)


def test_agent_pid_stamp_lands_through_self_handle(tmp_path):
    """Mimic solver.py + openhands.py: the pid stamp reaches env.agent_pid.

    solver.py sets ``ctx.workspace = getattr(env, "workspace_handle", env)`` and
    the OpenHands backend does ``setattr(ctx.workspace, "agent_pid", pid)`` under
    ``contextlib.suppress(Exception)``. With ``workspace_handle = self`` the
    setattr must succeed and land on the slot the lease ``hard_kill`` reads.
    """
    env = _COBenchEnv(tmp_path / "workspace")
    assert env.agent_pid is None

    # Exactly the spine chain (solver.py:382 then openhands.py:347):
    ctx_workspace = getattr(env, "workspace_handle", env)
    with contextlib.suppress(Exception):
        setattr(ctx_workspace, "agent_pid", 4242)

    assert env.agent_pid == 4242  # stamp landed -> hard_kill is now armed


def test_str_handle_silently_drops_stamp_negative_control(tmp_path):
    """A bare ``str`` workspace_handle is the DEAD path (stamp swallowed).

    Proves the bug the fix closes: stamping ``agent_pid`` onto an immutable
    ``str`` handle raises ``AttributeError``, which the backend's
    ``contextlib.suppress`` swallows, leaving the real env's ``agent_pid`` None
    and ``hard_kill`` a no-op.
    """

    class _StrHandleEnv:
        def __init__(self, root: Path):
            self.workspace_root = root
            self.workspace_handle = str(root)  # the OLD, broken handle
            self.agent_pid = None

    env = _StrHandleEnv(tmp_path / "workspace")
    ctx_workspace = getattr(env, "workspace_handle", env)  # a plain str
    with contextlib.suppress(Exception):
        setattr(ctx_workspace, "agent_pid", 4242)  # AttributeError on str -> swallowed

    assert env.agent_pid is None  # stamp lost -> hard_kill would no-op


def test_hard_kill_fires_when_pid_stamped(tmp_path, monkeypatch):
    """End-to-end of the stamp: provision installs hard_kill; a stamped pid reaps.

    Provisions the real ``COBenchEnvProvider`` (no Docker), stamps a pid through
    the self-handle exactly as the backend would, then invokes the lease's
    synchronous ``hard_kill`` and asserts it routes the stamped pid into
    ``_kill_process_tree`` (monkeypatched so no real process is signalled).
    """
    from meta_n.core.external_agents.concurrency import EnvLease
    import meta_n.integrations.co_bench as cobench_mod

    killed: list[int] = []
    monkeypatch.setattr(
        cobench_mod, "_kill_process_tree", lambda pid: killed.append(pid)
    )

    provider = COBenchEnvProvider(COBenchAdapter.__new__(COBenchAdapter))
    lease = EnvLease(workdir=tmp_path, session="ext-cobench-test", task_id="t0")

    from meta_n.core.meta_layer import TaskDescription

    task = TaskDescription(task_id="t0", description="author solve.py", metadata={})

    async def _drive():
        async with provider.provision(task, lease) as env:
            # Backend stamp path (solver.py:382 + openhands.py:347):
            handle = getattr(env, "workspace_handle", env)
            with contextlib.suppress(Exception):
                setattr(handle, "agent_pid", 9931)
            assert env.agent_pid == 9931
            # The lease's installed synchronous hard_kill must reap the pid.
            assert lease.hard_kill is not None
            lease.hard_kill()

    asyncio.run(_drive())
    assert killed == [9931]


def test_hard_kill_noop_when_no_pid(tmp_path, monkeypatch):
    """No stamped pid (e.g. REST path) -> hard_kill is a safe no-op."""
    from meta_n.core.external_agents.concurrency import EnvLease
    import meta_n.integrations.co_bench as cobench_mod
    from meta_n.core.meta_layer import TaskDescription

    killed: list[int] = []
    monkeypatch.setattr(
        cobench_mod, "_kill_process_tree", lambda pid: killed.append(pid)
    )

    provider = COBenchEnvProvider(COBenchAdapter.__new__(COBenchAdapter))
    lease = EnvLease(workdir=tmp_path, session="ext-cobench-test", task_id="t0")
    task = TaskDescription(task_id="t0", description="x", metadata={})

    async def _drive():
        async with provider.provision(task, lease) as env:
            assert env.agent_pid is None
            lease.hard_kill()  # must not raise

    asyncio.run(_drive())
    assert killed == []


# ---------------------------------------------------------------------------
# #4 — host-FS staging rejects absolute / `..` traversal keys
# ---------------------------------------------------------------------------
def _stage(provider: COBenchEnvProvider, env: _COBenchEnv, files: dict[str, str]):
    asyncio.run(provider.stage_files(env, files))


def test_stage_files_writes_relative_key(tmp_path):
    """A normal workspace-relative helper is written under the workspace root."""
    provider = COBenchEnvProvider(COBenchAdapter.__new__(COBenchAdapter))
    root = tmp_path / "workspace"
    root.mkdir()
    env = _COBenchEnv(root)
    _stage(provider, env, {"helpers/x.py": "ok"})
    assert (root / "helpers" / "x.py").read_text() == "ok"


def test_stage_files_skips_absolute_and_traversal(tmp_path, caplog):
    """Absolute and ``..``-traversing keys are skipped-with-warning, never written.

    Mirrors ``OpenHandsBackend.stage_files``: the only legit keys are
    workspace-relative helper files, so a key that would escape the workspace
    root must be dropped (defense-in-depth on the bare host).
    """
    provider = COBenchEnvProvider(COBenchAdapter.__new__(COBenchAdapter))
    root = tmp_path / "workspace"
    root.mkdir()
    env = _COBenchEnv(root)

    # An absolute escape target and a `..` escape target outside the workspace.
    abs_target = tmp_path / "etc_evil"
    parent_escape = tmp_path / "escape"

    with caplog.at_level("WARNING"):
        _stage(
            provider,
            env,
            {
                "helpers/ok.py": "good",
                str(abs_target): "bad-abs",
                "../escape": "bad-rel",
                "sub/../../escape2": "bad-rel2",
            },
        )

    # Only the safe relative key landed; nothing escaped the workspace root.
    assert (root / "helpers" / "ok.py").read_text() == "good"
    assert not abs_target.exists()
    assert not parent_escape.exists()
    assert not (tmp_path / "escape2").exists()
    # The two/three unsafe keys were each warned about and skipped.
    assert "skipping unsafe staged path" in caplog.text


def test_stage_files_empty_is_noop(tmp_path):
    """The gen0 vanilla-agent baseline stages nothing — a valid no-op."""
    provider = COBenchEnvProvider(COBenchAdapter.__new__(COBenchAdapter))
    root = tmp_path / "workspace"
    root.mkdir()
    env = _COBenchEnv(root)
    _stage(provider, env, {})
    assert list(root.iterdir()) == []
