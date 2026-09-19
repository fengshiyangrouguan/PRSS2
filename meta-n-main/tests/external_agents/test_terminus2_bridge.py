"""Terminus2 subprocess-bridge robustness + env-scrub + teardown-hook guards.

Install-free: drives ``Terminus2Backend._read_result`` / ``_read_result_parsed``
and the env-scrub on synthetic inputs; never imports ``terminal_bench``.
"""

from __future__ import annotations

import asyncio
import json
import os

import pytest

from meta_n.core.external_agents.backend import AgentRunContext, AgentRunResult, Prompt
from meta_n.core.external_agents.backends.terminus2 import Terminus2Backend
from meta_n.core.external_agents.terminated import TerminatedBy

from .conftest import make_task


def _backend() -> Terminus2Backend:
    return Terminus2Backend(
        model="openai/google/gemma-4-31b-qat",
        api_base="http://127.0.0.1:1234/v1",
        venv_python="/nonexistent/python",
        runner_dir="/tmp",
        tasks_dir="/tmp/tasks",
    )


class _WS:
    def __init__(self, *, task_id="t", staged_files=None, run_label=""):
        self.task_id = task_id
        self.staged_files = staged_files if staged_files is not None else {}
        self.run_label = run_label
        self.agent_pid = None


def _ctx(tmp_path, ws=None):
    return AgentRunContext(
        instruction="do it",
        prompt=Prompt(),
        workspace=ws if ws is not None else _WS(),
        time_limit_s=60.0,
        max_turns=8,
        token_budget=0,
        max_budget_usd=2.0,
        logging_dir=tmp_path,
    )


# --- #3: non-dict result JSON must NOT raise out of the backend --------------


@pytest.mark.parametrize("body", ["null", "[1, 2, 3]", '"a string"', "42", "true"])
def test_read_result_non_object_degrades_to_parse_error(tmp_path, body):
    """Valid JSON whose top level is non-dict becomes a degraded parse_error dict
    (mirrors OH's isinstance(dict) guard) instead of returning the raw value."""
    b = _backend()
    res = tmp_path / "t2_result.json"
    res.write_text(body)
    data, parsed_ok = b._read_result_parsed(res, b"")
    assert isinstance(data, dict)
    assert parsed_ok is False
    assert data["failure_mode"] == "parse_error"
    assert data["ok"] is False


def test_to_run_result_on_non_object_does_not_raise(tmp_path):
    """The full _read_result -> _to_run_result chain folds a non-object body into
    a degraded AgentRunResult (PARSE_ERROR), never an AttributeError."""
    b = _backend()
    res = tmp_path / "t2_result.json"
    res.write_text("[1, 2, 3]")
    data = b._read_result(res, b"")
    r = b._to_run_result(data, _ctx(tmp_path), 0.0)
    assert isinstance(r, AgentRunResult)
    assert r.terminated_by is TerminatedBy.PARSE_ERROR
    assert r.failure_mode == "parse_error"
    assert r.agent_tokens == 0


def test_read_result_object_round_trips(tmp_path):
    b = _backend()
    res = tmp_path / "t2_result.json"
    res.write_text(json.dumps({"ok": True, "is_resolved": True, "reward": 1.0}))
    data, parsed_ok = b._read_result_parsed(res, b"")
    assert parsed_ok is True
    assert data["is_resolved"] is True


def test_read_result_missing_file_is_env_error(tmp_path):
    b = _backend()
    data, parsed_ok = b._read_result_parsed(tmp_path / "nope.json", b"boom stderr")
    assert parsed_ok is False
    assert data["failure_mode"] == "env_error"
    assert "boom stderr" in data["error"]


# --- #minor: T2 child env is scrubbed (no secret leak) ----------------------


def test_t2_child_env_excludes_secrets(monkeypatch, tmp_path):
    """The runner child env must build from the scrubbed allowlist, NOT
    os.environ wholesale — parity with OpenHandsBackend."""
    captured = {}

    async def _fake_exec(*cmd, env=None, **kwargs):
        captured["env"] = env

        class _P:
            pid = 4242
            returncode = 0

            async def communicate(self):
                return (b"", b"")

            async def wait(self):
                return 0

        return _P()

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-leak")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ant-leak")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_exec)

    b = _backend()
    # Make the pre-spawn existence checks pass by pointing at real-ish paths is
    # not needed: run() writes the request then spawns; the workdir is tmp_path.
    ctx = _ctx(tmp_path / "logs")
    asyncio.run(b.run(ctx, None, None))

    env = captured["env"]
    assert env is not None
    assert env.get("OPENAI_API_KEY") == "dummy"  # the one explicitly-injected key
    assert "OPENROUTER_API_KEY" not in env
    assert "ANTHROPIC_API_KEY" not in env
    # PYTHONPATH is set so -m t2_runner resolves.
    assert b._runner_dir in env.get("PYTHONPATH", "")


# --- #minor: hard-timeout override must NOT clobber a genuinely RESOLVED run -


def _run_with_hard_timeout(monkeypatch, tmp_path, result_body: str | None):
    """Drive run() forcing the meta-n hard wall-clock timeout to fire, with an
    optional pre-written runner result file, and return the AgentRunResult."""
    logs = tmp_path / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    if result_body is not None:
        (logs / "t2_result.json").write_text(result_body)

    async def _fake_exec(*cmd, env=None, **kwargs):
        class _P:
            pid = 777
            returncode = None

            async def communicate(self):
                await asyncio.sleep(10)  # outlast the (tiny) hard timeout
                return (b"", b"")

            async def wait(self):
                self.returncode = -9
                return -9

        return _P()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_exec)
    b = _backend()
    # Force the hard timeout to fire almost immediately.
    monkeypatch.setattr(b, "_hard_timeout", lambda soft: 0.05)
    monkeypatch.setattr(b, "_sigkill_group", staticmethod(lambda proc: None))

    async def _noop_down(run_label=None):
        return None

    monkeypatch.setattr(b, "_force_compose_down", _noop_down)
    ctx = _ctx(logs)
    return asyncio.run(b.run(ctx, None, None))


def test_timeout_keeps_a_resolved_run_completed(monkeypatch, tmp_path):
    """A run that genuinely PASSED before the hard wall keeps COMPLETED /
    native_resolved=True — never overwritten with TIMEOUT (would be a
    contradictory success=True + terminated=TIMEOUT telemetry row)."""
    body = json.dumps({"ok": True, "is_resolved": True, "reward": 1.0,
                       "total_input_tokens": 5, "total_output_tokens": 3})
    r = _run_with_hard_timeout(monkeypatch, tmp_path, body)
    assert r.native_resolved is True
    assert r.terminated_by is TerminatedBy.COMPLETED
    assert r.failure_mode is None


def test_timeout_overrides_a_clean_unresolved_run(monkeypatch, tmp_path):
    """A clean-tagged UNRESOLVED run that ran past the hard wall is stamped
    TIMEOUT / agent_timeout."""
    body = json.dumps({"ok": True, "is_resolved": False, "reward": 0.0,
                       "failure_mode": "unset"})
    r = _run_with_hard_timeout(monkeypatch, tmp_path, body)
    assert r.terminated_by is TerminatedBy.TIMEOUT
    assert r.failure_mode == "agent_timeout"


def test_timeout_with_unparseable_result_is_agent_timeout(monkeypatch, tmp_path):
    """On the timeout path a present-but-unparseable result file is classified
    as the meta-n wall-clock timeout, not the synthesized env/parse error."""
    r = _run_with_hard_timeout(monkeypatch, tmp_path, "{ this is not valid json")
    assert r.terminated_by is TerminatedBy.TIMEOUT
    assert r.failure_mode == "agent_timeout"


def test_t2_run_stamps_and_clears_agent_pid(monkeypatch, tmp_path):
    """run() stamps the runner pid onto the workspace handle (for the lease
    hard_kill) and clears it in finally (so a reused pid is never reaped)."""
    seen_pids = []

    async def _fake_exec(*cmd, env=None, **kwargs):
        class _P:
            pid = 31337
            returncode = 0

            async def communicate(self):
                # Capture the stamped pid mid-run (before finally clears it).
                seen_pids.append(getattr(ws, "agent_pid", "MISSING"))
                return (b"", b"")

            async def wait(self):
                return 0

        return _P()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_exec)
    ws = _WS()
    b = _backend()
    ctx = _ctx(tmp_path / "logs", ws=ws)
    asyncio.run(b.run(ctx, None, None))
    assert seen_pids == [31337]          # stamped during the run
    assert ws.agent_pid is None          # cleared in finally


# --- #2: TBTerminus2EnvProvider installs lease teardown + hard_kill ----------


def _provider():
    from meta_n.integrations.terminal_bench import TBTerminus2EnvProvider

    # The provider only reads task metadata; a bare object adapter is enough.
    return TBTerminus2EnvProvider(adapter=object())


def _lease(tmp_path):
    from meta_n.core.external_agents.env import EnvLease

    wd = tmp_path / "lease"
    wd.mkdir(parents=True, exist_ok=True)
    return EnvLease(workdir=wd, session="ext-mytask-abc123", task_id="mytask")


def test_t2_provider_installs_lease_hooks(tmp_path):
    """provision must populate lease.teardown AND lease.hard_kill so the
    DockerRunGuard sweep can reap a wedged runner/container (audit blocker #2:
    previously both were None and the sweep skipped the lease)."""

    async def _drive():
        provider = _provider()
        lease = _lease(tmp_path)
        task = make_task(task_id="mytask", task_name="mytask")
        async with provider.provision(task, lease) as env:
            # Both backstop hooks are installed while the lease is held.
            assert lease.hard_kill is not None
            assert lease.teardown is not None
            # The env exposes the agent_pid slot the backend stamps.
            assert hasattr(env, "agent_pid")
            assert env.agent_pid is None
            return lease, env

    asyncio.run(_drive())


def test_t2_provider_hard_kill_sigkills_stamped_pid(monkeypatch, tmp_path):
    """The installed hard_kill SIGKILLs the runner pid the backend stamped onto
    the env handle; a no-op when no pid is recorded."""
    killed = {}
    monkeypatch.setattr(os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(os, "killpg", lambda pgid, sig: killed.update(pgid=pgid))

    async def _drive():
        provider = _provider()
        lease = _lease(tmp_path)
        task = make_task(task_id="mytask", task_name="mytask")
        async with provider.provision(task, lease) as env:
            # No pid recorded yet -> hard_kill is a no-op.
            lease.hard_kill()
            assert "pgid" not in killed
            # Stamp a pid (as Terminus2Backend.run would) and reap it.
            env.agent_pid = 5150
            lease.hard_kill()
            assert killed.get("pgid") == 5150

    asyncio.run(_drive())


def test_t2_provider_teardown_calls_force_compose_down(monkeypatch, tmp_path):
    """The installed teardown delegates to the backend's label-scoped
    _force_compose_down(run_label) so a wedged runner's container is reaped."""
    from meta_n.core.external_agents.backends.terminus2 import Terminus2Backend

    seen = {}

    async def _fake_down(run_label=None):
        seen["run_label"] = run_label

    monkeypatch.setattr(Terminus2Backend, "_force_compose_down", staticmethod(_fake_down))

    async def _drive():
        provider = _provider()
        lease = _lease(tmp_path)
        task = make_task(task_id="mytask", task_name="mytask")
        async with provider.provision(task, lease) as env:  # noqa: F841
            await lease.teardown()

    asyncio.run(_drive())
    # The run_label threaded into teardown is the lease session.
    assert seen["run_label"] == "ext-mytask-abc123"
