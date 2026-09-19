"""OpenHands Docker-sandbox wiring — argv / base_url / env / mount construction.

Unit-level coverage for the opt-in ``sandbox=True`` path on
:class:`OpenHandsBackend`: it must wrap the runner in a ``docker run`` invocation
that bind-mounts the workspace + run dir, rewrites the loopback LLM ``base_url``
to ``host.docker.internal``, forwards ONLY the provider key (no secret leak), and
maps the runner flags to the container-side paths. These assertions are pure
(no Docker, no LLM, no ``openhands`` install) so they run in meta-n's env and gate
the wiring without the heavy e2e run (which is proven separately on local Gemma).

The default (``sandbox=False``) path must be byte-identical to before: it spawns
the venv python directly, with no ``docker`` token anywhere in the argv.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from meta_n.core.external_agents.backend import AgentRunContext, Prompt
from meta_n.core.external_agents.backends import openhands as oh_mod
from meta_n.core.external_agents.backends.openhands import (
    OpenHandsBackend,
    _HOST_GATEWAY_ALIAS,
    _SANDBOX_WORKSPACE,
    _SANDBOX_RUNDIR,
    _SANDBOX_RUNNER,
    _TEARDOWN_WAIT_S,
)


def _ctx(tmp_path: Path) -> AgentRunContext:
    return AgentRunContext(
        instruction="solve the task",
        prompt=Prompt(system_suffix="", prefix=""),
        workspace=str(tmp_path / "workspace"),
        time_limit_s=600.0,
        max_turns=6,
        token_budget=100_000,
        max_budget_usd=0.0,
        logging_dir=tmp_path / "log",
    )


def _build(tmp_path: Path, *, sandbox: bool):
    """Drive only the invocation builder (no spawn) and return its 3-tuple."""
    backend = OpenHandsBackend(
        model="google/gemma-4-31b-qat",
        api_base="http://127.0.0.1:1234/v1",
        api_key="secret-provider-key",
        sandbox=sandbox,
        solution_file="solve.py",
        local_default=True,
    )
    ctx = _ctx(tmp_path)
    run_dir = tmp_path / "log"
    run_dir.mkdir(parents=True, exist_ok=True)
    ws = str(ctx.workspace)
    instr_f = run_dir / "instruction.txt"
    suffix_f = run_dir / "system_suffix.txt"
    res_f = run_dir / "oh_result.json"
    argv, env, name = backend._build_invocation(
        ctx, ws, run_dir, instr_f, suffix_f, res_f
    )
    return backend, ctx, run_dir, ws, res_f, argv, env, name


# ---------------------------------------------------------------------------
# Host path (default) — unchanged: venv python, no docker.
# ---------------------------------------------------------------------------
def test_host_path_spawns_venv_python_no_docker(tmp_path):
    backend, _ctx_, _rd, ws, res_f, argv, env, name = _build(tmp_path, sandbox=False)
    assert name == ""  # no container on the host path
    assert argv[0] == backend._py  # venv interpreter
    assert argv[1] == backend._runner
    assert "docker" not in " ".join(argv)
    # Host path passes real (host) filesystem paths through to the runner.
    assert "--workspace" in argv and ws in argv
    assert str(res_f) in argv
    # Provider key reaches the runner via OH_API_KEY (never in argv).
    assert env["OH_API_KEY"] == "secret-provider-key"
    assert "secret-provider-key" not in argv


# ---------------------------------------------------------------------------
# Sandbox path — docker run with mounts, host-gateway, rewritten base_url.
# ---------------------------------------------------------------------------
def test_sandbox_path_wraps_runner_in_docker_run(tmp_path):
    backend, _ctx_, run_dir, ws, _res, argv, env, name = _build(tmp_path, sandbox=True)
    assert argv[0] == backend.docker_bin
    assert argv[1] == "run"
    assert "--rm" in argv
    # Unique container name returned for teardown, and present in the argv.
    assert name and name in argv
    # host.docker.internal is wired via --add-host.
    assert "--add-host" in argv
    assert f"{_HOST_GATEWAY_ALIAS}:host-gateway" in argv
    # The sandbox image and the in-container runner invocation are present.
    assert backend.sandbox_image in argv
    assert _SANDBOX_RUNNER in argv


def test_sandbox_bind_mounts_workspace_rundir_and_runner(tmp_path):
    backend, _ctx_, run_dir, ws, _res, argv, _env, _name = _build(tmp_path, sandbox=True)
    joined = argv
    # Workspace mounted rw at /workspace; run dir at /run; runner ro.
    assert f"{ws}:{_SANDBOX_WORKSPACE}" in joined
    assert f"{run_dir}:{_SANDBOX_RUNDIR}" in joined
    assert f"{backend._runner}:{_SANDBOX_RUNNER}:ro" in joined


def test_sandbox_rewrites_loopback_base_url(tmp_path):
    _b, _c, _rd, _ws, _res, argv, _env, _name = _build(tmp_path, sandbox=True)
    # The runner's --base-url must point at the host gateway, never loopback.
    i = argv.index("--base-url")
    base_url = argv[i + 1]
    assert _HOST_GATEWAY_ALIAS in base_url
    assert "127.0.0.1" not in base_url
    assert base_url == f"http://{_HOST_GATEWAY_ALIAS}:1234/v1"


def test_sandbox_runner_flags_use_container_paths(tmp_path):
    _b, _c, _rd, ws, res_f, argv, _env, _name = _build(tmp_path, sandbox=True)
    # The runner's --workspace is the CONTAINER path, not the host path.
    i = argv.index("--workspace")
    assert argv[i + 1] == _SANDBOX_WORKSPACE
    assert ws not in argv  # host workspace path must not leak into runner flags
    # Result/instruction/suffix files are addressed under /run inside the container.
    ri = argv.index("--result-file")
    assert argv[ri + 1] == f"{_SANDBOX_RUNDIR}/{res_f.name}"
    assert str(res_f) not in argv


def test_sandbox_forwards_only_provider_key_no_secret_leak(tmp_path):
    _b, _c, _rd, _ws, _res, argv, env, _name = _build(tmp_path, sandbox=True)
    # OH_API_KEY is forwarded by NAME (-e OH_API_KEY), value carried in env only —
    # never as an argv literal (no key in `ps`).
    assert "-e" in argv and "OH_API_KEY" in argv
    assert env["OH_API_KEY"] == "secret-provider-key"
    assert "secret-provider-key" not in argv
    # No wholesale env passthrough: the docker argv carries no --env-file and only
    # the two explicit -e names we set.
    assert "--env-file" not in argv
    e_values = [argv[i + 1] for i, tok in enumerate(argv) if tok == "-e"]
    assert set(e_values) <= {"OH_API_KEY", "OPENHANDS_SUPPRESS_BANNER=1"}


def test_sandbox_unique_container_name_each_call(tmp_path):
    """Two builds yield distinct container names (no collision under concurrency)."""
    names = set()
    for _ in range(3):
        _b, _c, _rd, _ws, _res, _argv, _env, name = _build(tmp_path, sandbox=True)
        names.add(name)
    assert len(names) == 3


# ---------------------------------------------------------------------------
# Teardown is bounded (#8) — _reap proc.wait() and _docker_rm rm.wait() can
# never hang the timeout/cancel unwind on a wedged child / wedged docker daemon.
# ---------------------------------------------------------------------------
class _HungProc:
    """Fake subprocess whose ``wait()`` never returns (simulates a wedged child).

    ``pid``/``returncode`` are arranged so ``sigkill_group`` (called by
    ``_reap`` via ``_killpg``) short-circuits without touching the OS: ``pid``
    is ``None`` → the killpg helper returns immediately, so the only thing that
    could block is the ``wait()`` the fix now bounds.
    """

    pid = None  # sigkill_group returns early on pid is None (no OS call)
    returncode = None
    killed = False

    async def wait(self):
        await asyncio.Event().wait()  # never set → hangs forever if unbounded

    def kill(self):
        self.killed = True


def _backend(**kw) -> OpenHandsBackend:
    return OpenHandsBackend(
        model="google/gemma-4-31b-qat",
        api_base="http://127.0.0.1:1234/v1",
        api_key="k",
        local_default=True,
        **kw,
    )


def test_docker_rm_self_bounds_on_wedged_daemon(monkeypatch):
    """``_docker_rm`` returns within its bound even if ``rm.wait()`` hangs forever.

    Monkeypatch ``create_subprocess_exec`` so the fake ``docker rm -f`` proc's
    ``wait()`` never returns; with the bound shrunk to a tiny value the call must
    self-terminate (no ``TimeoutError`` escapes — teardown is never-raise) and
    SIGKILL the hung CLI client so it is not left as a zombie.
    """
    backend = _backend()
    hung = _HungProc()

    async def _fake_exec(*_a, **_kw):
        return hung

    monkeypatch.setattr(oh_mod.asyncio, "create_subprocess_exec", _fake_exec)
    monkeypatch.setattr(oh_mod, "_TEARDOWN_WAIT_S", 0.05)

    async def _drive():
        # If _docker_rm did NOT self-bound, this outer wait_for would itself trip
        # (proving the hang). The outer ceiling is well above the inner bound.
        await asyncio.wait_for(backend._docker_rm("wedged-container"), timeout=5.0)

    asyncio.run(_drive())  # returns => self-bounded, no TimeoutError raised out
    assert hung.killed is True  # hung docker-CLI client was reaped, not leaked


def test_reap_self_bounds_on_wedged_child(monkeypatch):
    """``_reap`` returns even when the spawned proc's ``wait()`` hangs forever.

    Drives ``_reap`` with no container (host path) and a child whose ``wait()``
    never returns; the bounded ``proc.wait()`` must let ``_reap`` return.
    """
    backend = _backend()
    hung = _HungProc()
    monkeypatch.setattr(oh_mod, "_TEARDOWN_WAIT_S", 0.05)

    async def _drive():
        await asyncio.wait_for(backend._reap(hung, ""), timeout=5.0)

    asyncio.run(_drive())  # returns => the proc.wait() is bounded


def test_reap_bounds_both_waits_with_container(monkeypatch):
    """``_reap`` with a container bounds BOTH proc.wait() and the docker rm wait().

    The child proc's ``wait()`` hangs AND the ``docker rm -f`` proc's ``wait()``
    hangs; ``_reap`` must still return (each await independently bounded).
    """
    backend = _backend()
    child = _HungProc()
    rm_proc = _HungProc()

    async def _fake_exec(*_a, **_kw):
        return rm_proc  # the `docker rm -f` subprocess, also hung

    monkeypatch.setattr(oh_mod.asyncio, "create_subprocess_exec", _fake_exec)
    monkeypatch.setattr(oh_mod, "_TEARDOWN_WAIT_S", 0.05)

    async def _drive():
        await asyncio.wait_for(backend._reap(child, "wedged-container"), timeout=5.0)

    asyncio.run(_drive())  # returns => both waits bounded
    assert rm_proc.killed is True  # the hung rm client was reaped


def test_reap_host_path_does_not_call_docker_rm(monkeypatch):
    """Host path (no container_name) never spawns ``docker rm`` — byte-identical.

    The default (non-sandbox) teardown must stay exactly as before: no container
    name => ``_docker_rm`` is not invoked at all.
    """
    backend = _backend()
    spawned = []

    async def _fake_exec(*a, **_kw):
        spawned.append(a)
        return _HungProc()

    monkeypatch.setattr(oh_mod.asyncio, "create_subprocess_exec", _fake_exec)
    monkeypatch.setattr(oh_mod, "_TEARDOWN_WAIT_S", 0.05)

    # A child that exits immediately (returncode set) so wait() returns at once.
    class _DoneProc(_HungProc):
        returncode = 0

        async def wait(self):
            return 0

    asyncio.run(backend._reap(_DoneProc(), ""))
    assert spawned == []  # no docker rm spawned on the host path


def test_teardown_bound_matches_run_finally_constant():
    """The teardown bound is single-sourced: the module constant is a real float."""
    assert isinstance(_TEARDOWN_WAIT_S, float)
    assert _TEARDOWN_WAIT_S == 70.0
