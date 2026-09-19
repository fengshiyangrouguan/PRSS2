"""OpenHands-on-terminal-bench dispatch + subprocess-bridge backend mapping.

Covers the new ``--base-solver openhands`` path on the terminal_bench adapter:

* the adapter's ``make_agent_backend`` / ``make_env_provider`` / ``make_scorer``
  dispatch ``openhands`` to :class:`OpenHandsTBBackend` and the SHARED
  ``TBExternalEnvProvider`` / ``TBExternalScorer`` (the same shims Terminus 2
  uses), while ``terminus2`` stays on :class:`Terminus2Backend` and ``builtin`` /
  any unknown kind is rejected;
* :class:`OpenHandsTBBackend` maps a runner result dict (the shared t2-style
  schema) into an ``AgentRunResult`` — resolved → COMPLETED, ran-but-wrong →
  UNKNOWN, a non-object / missing result file → a degraded PARSE_ERROR /
  ENV_ERROR (never an AttributeError out of the never-raise ``run``);
* the request JSON the backend ships carries the runner's actual injection key
  (``system_message_suffix``, NOT ``additional_context``) and a clamped
  ``max_iterations``, and the child env is scrubbed of meta-n secrets;
* the shared env provider installs a label-scoped teardown hook that works for
  the OH backend (its ``_force_compose_down`` shares the Terminus 2 signature).

Install-free: drives synthetic dicts / monkeypatched ``create_subprocess_exec``;
never imports ``openhands`` / ``terminal_bench``.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from meta_n.core.external_agents.backend import AgentRunContext, AgentRunResult, Prompt
from meta_n.core.external_agents.backends.openhands_tb import OpenHandsTBBackend
from meta_n.core.external_agents.backends.terminus2 import Terminus2Backend
from meta_n.core.external_agents.terminated import TerminatedBy
from meta_n.integrations.terminal_bench import (
    TBExternalEnvProvider,
    TBExternalScorer,
    TBTerminus2EnvProvider,
    TBTerminus2Scorer,
    TerminalBenchAdapter,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _backend() -> OpenHandsTBBackend:
    return OpenHandsTBBackend(
        model="openai/google/gemma-4-31b-qat",
        api_base="http://127.0.0.1:1234/v1",
        venv_python="/nonexistent/python",
        runner_dir="/tmp",
        tasks_dir="/tmp/tasks",
    )


class _WS:
    def __init__(self, *, task_id="hello-world", staged_files=None, run_label=""):
        self.task_id = task_id
        self.staged_files = staged_files if staged_files is not None else {}
        self.run_label = run_label
        self.agent_pid = None


def _ctx(tmp_path, ws=None):
    return AgentRunContext(
        instruction="solve it",
        prompt=Prompt(),
        workspace=ws if ws is not None else _WS(),
        time_limit_s=60.0,
        max_turns=8,
        token_budget=0,
        max_budget_usd=0.5,
        logging_dir=tmp_path,
    )


_INNER_KW = dict(
    model="google/gemma-4-31b-qat",
    api_base="http://127.0.0.1:1234/v1",
    api_key="dummy",
    provider_env_var=None,
)


# ---------------------------------------------------------------------------
# (A) Adapter dispatch — both external kinds + rejection of non-external kinds
# ---------------------------------------------------------------------------


def test_make_agent_backend_dispatches_openhands():
    """``openhands`` builds the OpenHands-on-TB bridge with litellm-routed model."""
    ad = TerminalBenchAdapter()
    b = ad.make_agent_backend("openhands", **_INNER_KW)
    assert isinstance(b, OpenHandsTBBackend)
    assert b.name == "openhands"
    assert b.outer_token_mode is False
    # The un-prefixed pricing key gets an ``openai/`` litellm prefix for routing.
    assert b._model == "openai/google/gemma-4-31b-qat"
    assert b._api_base == "http://127.0.0.1:1234/v1"


def test_make_agent_backend_dispatches_terminus2_unchanged():
    """``terminus2`` still builds the Terminus 2 backend (regression guard)."""
    ad = TerminalBenchAdapter()
    b = ad.make_agent_backend("terminus2", **_INNER_KW)
    assert isinstance(b, Terminus2Backend)


@pytest.mark.parametrize("kind", ["bogus", "", "Openhands"])
def test_make_agent_backend_rejects_unknown_kinds(kind):
    """Unknown / empty / wrong-case kinds raise (unsupported by the factory).

    ``builtin`` is NO LONGER in this list — terminal_bench now runs it as a
    same-container control (see ``test_make_agent_backend_dispatches_builtin``).
    """
    ad = TerminalBenchAdapter()
    with pytest.raises(ValueError):
        ad.make_agent_backend(kind, **_INNER_KW)


def test_make_agent_backend_builtin_requires_solver():
    """``builtin`` without the authoring ``solver`` raises (the spine passes it)."""
    ad = TerminalBenchAdapter()
    with pytest.raises(ValueError):
        ad.make_agent_backend("builtin", **_INNER_KW)


def test_make_agent_backend_dispatches_builtin():
    """``builtin`` (with an authoring solver) builds the same-container control."""
    from meta_n.core.external_agents.backends.builtin_tb import BuiltinTBBackend

    ad = TerminalBenchAdapter()
    b = ad.make_agent_backend(
        "builtin", solver=object(), llm_client=object(),
        solver_language="bash", **_INNER_KW,
    )
    assert isinstance(b, BuiltinTBBackend)
    assert b.name == "builtin"
    # The authoring call rides the OUTER ledger (parity with the CO-Bench control).
    assert b.outer_token_mode is True


@pytest.mark.parametrize("kind", ["openhands", "terminus2", "builtin"])
def test_env_provider_and_scorer_shared_across_kinds(kind):
    """All three kinds resolve to the SHARED env provider + scorer.

    ``builtin`` reuses them too — its same-container control runs the IDENTICAL
    verifier, so the scorer (reading ``native_resolved`` / ``native_score``) works
    unchanged.
    """
    ad = TerminalBenchAdapter()
    ep = ad.make_env_provider(kind)
    sc = ad.make_scorer(kind)
    assert isinstance(ep, TBExternalEnvProvider)
    assert isinstance(sc, TBExternalScorer)
    # The shared names are aliases of the canonical Terminus 2 classes.
    assert isinstance(ep, TBTerminus2EnvProvider)
    assert isinstance(sc, TBTerminus2Scorer)


def test_shared_aliases_are_identity():
    """``TBExternalEnvProvider`` / ``TBExternalScorer`` are the SAME classes the
    Terminus 2 path already used (so scoring is reused verbatim)."""
    assert TBExternalEnvProvider is TBTerminus2EnvProvider
    assert TBExternalScorer is TBTerminus2Scorer


@pytest.mark.parametrize("kind", ["bogus", ""])
def test_env_provider_and_scorer_reject_unknown(kind):
    """Unknown kinds raise; ``builtin`` is NO LONGER rejected (it is supported)."""
    ad = TerminalBenchAdapter()
    with pytest.raises(ValueError):
        ad.make_env_provider(kind)
    with pytest.raises(ValueError):
        ad.make_scorer(kind)


# ---------------------------------------------------------------------------
# (B) Result mapping — resolved / ran-but-wrong / failure tags
# ---------------------------------------------------------------------------


def test_to_run_result_resolved_is_completed(tmp_path):
    """A clean resolved run → COMPLETED, native_resolved=True, reward mirrored."""
    b = _backend()
    data = {
        "ok": True,
        "is_resolved": True,
        "reward": 1.0,
        "total_input_tokens": 12,
        "total_output_tokens": 7,
        "agent_calls": 3,
    }
    r = b._to_run_result(data, _ctx(tmp_path), 0.0)
    assert isinstance(r, AgentRunResult)
    assert r.terminated_by is TerminatedBy.COMPLETED
    assert r.failure_mode is None
    assert r.native_resolved is True
    assert r.native_score == 1.0
    assert r.agent_tokens == 19
    assert r.agent_prompt_tokens == 12
    assert r.agent_completion_tokens == 7
    assert r.agent_calls == 3


def test_to_run_result_ran_but_wrong_is_unknown(tmp_path):
    """A clean-tagged UNRESOLVED run (tests simply did not pass) → UNKNOWN /
    score 0, NOT an agent error."""
    b = _backend()
    data = {"ok": True, "is_resolved": False, "reward": 0.0, "failure_mode": "none"}
    r = b._to_run_result(data, _ctx(tmp_path), 0.0)
    assert r.terminated_by is TerminatedBy.UNKNOWN
    assert r.failure_mode is None
    assert r.native_resolved is False
    assert r.native_score == 0.0


def test_to_run_result_token_budget_failure_folds(tmp_path):
    """A native ``token_budget`` tag folds to the shared TOKEN_BUDGET taxonomy."""
    b = _backend()
    data = {
        "ok": True,
        "is_resolved": False,
        "reward": 0.0,
        "failure_mode": "token_budget",
        "total_input_tokens": 100,
        "total_output_tokens": 50,
    }
    r = b._to_run_result(data, _ctx(tmp_path), 0.0)
    assert r.terminated_by is TerminatedBy.TOKEN_BUDGET
    assert r.failure_mode == "token_budget"
    # Partial inner spend survives an abort so CostGuard does not price at $0.
    assert r.agent_tokens == 150


def test_native_resolved_drives_scorer(tmp_path):
    """The SHARED scorer derives success from ``native_resolved`` off an OH run."""
    b = _backend()
    data = {"ok": True, "is_resolved": True, "reward": 1.0}
    run = b._to_run_result(data, _ctx(tmp_path), 0.0)
    scorer = TBExternalScorer.__new__(TBExternalScorer)  # no adapter needed
    res = asyncio.run(scorer.score(task=None, env=None, solution="", run=run))
    assert res.success is True
    assert res.score == 1.0
    assert res.valid is True


# ---------------------------------------------------------------------------
# (C) Degraded result-file handling — never raises out of run()
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("body", ["null", "[1, 2, 3]", '"a string"', "42", "true"])
def test_read_result_non_object_degrades_to_parse_error(tmp_path, body):
    b = _backend()
    res = tmp_path / "oh_tb_result.json"
    res.write_text(body)
    data, parsed_ok = b._read_result_parsed(res, b"")
    assert isinstance(data, dict)
    assert parsed_ok is False
    assert data["failure_mode"] == "parse_error"
    assert data["ok"] is False


def test_to_run_result_on_non_object_does_not_raise(tmp_path):
    b = _backend()
    res = tmp_path / "oh_tb_result.json"
    res.write_text("[1, 2, 3]")
    data = b._read_result(res, b"")
    r = b._to_run_result(data, _ctx(tmp_path), 0.0)
    assert isinstance(r, AgentRunResult)
    assert r.terminated_by is TerminatedBy.PARSE_ERROR
    assert r.failure_mode == "parse_error"
    assert r.agent_tokens == 0


def test_read_result_missing_file_is_env_error(tmp_path):
    b = _backend()
    data, parsed_ok = b._read_result_parsed(tmp_path / "nope.json", b"boom stderr")
    assert parsed_ok is False
    assert data["failure_mode"] == "env_error"
    assert "boom stderr" in data["error"]


# ---------------------------------------------------------------------------
# (D) Request JSON shape — runner-key contract + env scrub
# ---------------------------------------------------------------------------


def test_run_writes_request_with_runner_keys(monkeypatch, tmp_path):
    """The request JSON the backend ships uses the runner's ACTUAL injection key
    (``system_message_suffix``, NOT ``additional_context``) and a clamped
    ``max_iterations`` — the contract ``oh_tb_runner`` parses."""
    captured = {}

    async def _fake_exec(*cmd, env=None, **kwargs):
        captured["env"] = env
        captured["cmd"] = list(cmd)

        class _P:
            pid = 4242
            returncode = 0

            async def communicate(self):
                return (b"", b"")

            async def wait(self):
                return 0

        return _P()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_exec)
    # Inject a non-empty system suffix so the composed context is observable.
    ws = _WS(task_id="fix-permissions", run_label="ext-fix-permissions-abc123")
    ctx = AgentRunContext(
        instruction="solve it",
        prompt=Prompt(system_suffix="EXTRA SYSTEM GUIDANCE"),
        workspace=ws,
        time_limit_s=60.0,
        max_turns=8,
        token_budget=0,
        max_budget_usd=0.5,
        logging_dir=tmp_path / "logs",
    )
    b = _backend()
    asyncio.run(b.run(ctx, None, None))

    req = json.loads((tmp_path / "logs" / "oh_tb_request.json").read_text())
    # The runner reads ``system_message_suffix``; ``additional_context`` is NOT it.
    assert req["system_message_suffix"] == "EXTRA SYSTEM GUIDANCE"
    assert "additional_context" not in req
    assert req["task_id"] == "fix-permissions"
    # max_iterations is present (the runner reads ``req["max_iterations"]``).
    assert "max_iterations" in req
    assert 1 <= req["max_iterations"] <= 8
    # The launch resolves the runner module.
    assert "-m" in captured["cmd"]
    assert "oh_tb_runner" in captured["cmd"]


def test_instruction_is_advisory_not_forwarded_to_tb_runner(monkeypatch, tmp_path):
    """R2-EA-2 (audit) — ``AgentRunContext.instruction`` is ADVISORY on the TB
    subprocess backends: ``oh_tb_runner`` loads the canonical instruction from
    the on-disk ``task.yaml``, so a distinctive instruction string set on the
    context must NOT reach the request JSON the backend ships. Only ``prompt.*``
    is forwarded. The dataclass docstring documents the advisory contract so a
    future meta-layer that REWROTE the instruction is not silently void."""
    async def _fake_exec(*cmd, env=None, **kwargs):
        class _P:
            pid = 4242
            returncode = 0

            async def communicate(self):
                return (b"", b"")

            async def wait(self):
                return 0

        return _P()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_exec)
    sentinel = "REWRITTEN-INSTRUCTION-SENTINEL-9q42"
    ws = _WS(task_id="fix-permissions", run_label="ext-fix-permissions-abc123")
    ctx = AgentRunContext(
        instruction=sentinel,
        prompt=Prompt(system_suffix="EXTRA SYSTEM GUIDANCE"),
        workspace=ws,
        time_limit_s=60.0,
        max_turns=8,
        token_budget=0,
        max_budget_usd=0.5,
        logging_dir=tmp_path / "logs",
    )
    b = _backend()
    asyncio.run(b.run(ctx, None, None))

    req = json.loads((tmp_path / "logs" / "oh_tb_request.json").read_text())
    # The advisory instruction is NOT plumbed into the request (the runner reads
    # task.yaml); only the injection surface (system_message_suffix) is.
    assert sentinel not in json.dumps(req)
    assert req["system_message_suffix"] == "EXTRA SYSTEM GUIDANCE"
    # The dataclass documents the advisory / not-honored-by-TB contract.
    doc = (AgentRunContext.__doc__ or "").lower()
    assert "advisory" in doc
    assert "task.yaml" in doc


def test_child_env_excludes_secrets(monkeypatch, tmp_path):
    """The runner child env builds from the scrubbed allowlist (parity with T2)."""
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
    asyncio.run(b.run(_ctx(tmp_path / "logs"), None, None))
    env = captured["env"]
    assert env is not None
    assert env.get("OPENAI_API_KEY") == "dummy"
    assert "OPENROUTER_API_KEY" not in env
    assert "ANTHROPIC_API_KEY" not in env
    assert b._runner_dir in env.get("PYTHONPATH", "")


# ---------------------------------------------------------------------------
# (E) Hard-timeout override semantics (mirror the Terminus 2 contract)
# ---------------------------------------------------------------------------


def _run_with_hard_timeout(monkeypatch, tmp_path, result_body):
    """Drive run() forcing the meta-n hard wall-clock timeout to fire, with an
    optional pre-written runner result file; return the AgentRunResult."""
    logs = tmp_path / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    if result_body is not None:
        (logs / "oh_tb_result.json").write_text(result_body)

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
    monkeypatch.setattr(b, "_hard_timeout", lambda soft: 0.05)
    monkeypatch.setattr(b, "_sigkill_group", staticmethod(lambda proc: None))

    async def _noop_down(run_label=None):
        return None

    monkeypatch.setattr(b, "_force_compose_down", _noop_down)
    return asyncio.run(b.run(_ctx(logs), None, None))


def test_timeout_keeps_a_resolved_run_completed(monkeypatch, tmp_path):
    """A run that genuinely PASSED before the hard wall keeps COMPLETED."""
    body = json.dumps(
        {"ok": True, "is_resolved": True, "reward": 1.0,
         "total_input_tokens": 5, "total_output_tokens": 3}
    )
    r = _run_with_hard_timeout(monkeypatch, tmp_path, body)
    assert r.native_resolved is True
    assert r.terminated_by is TerminatedBy.COMPLETED
    assert r.failure_mode is None


def test_timeout_overrides_a_clean_unresolved_run(monkeypatch, tmp_path):
    """A clean-tagged UNRESOLVED run past the hard wall → TIMEOUT/agent_timeout."""
    body = json.dumps(
        {"ok": True, "is_resolved": False, "reward": 0.0, "failure_mode": "unset"}
    )
    r = _run_with_hard_timeout(monkeypatch, tmp_path, body)
    assert r.terminated_by is TerminatedBy.TIMEOUT
    assert r.failure_mode == "agent_timeout"


def test_timeout_with_no_result_is_agent_timeout(monkeypatch, tmp_path):
    """No runner result on the timeout path → the meta-n wall-clock timeout."""
    r = _run_with_hard_timeout(monkeypatch, tmp_path, None)
    assert r.terminated_by is TerminatedBy.TIMEOUT
    assert r.failure_mode == "agent_timeout"


# ---------------------------------------------------------------------------
# (F) Shared env provider installs an OH-compatible teardown hook
# ---------------------------------------------------------------------------


def test_shared_env_provider_teardown_uses_label_scoped_sweep(tmp_path):
    """The shared provider installs a lease teardown + hard_kill; the teardown
    delegates to a label-scoped ``_force_compose_down`` (the OH backend exposes
    the SAME signature, so the shared provider works for the OH path too)."""
    from meta_n.core.external_agents.env import EnvLease

    provider = TBExternalEnvProvider(adapter=object())
    wd = tmp_path / "lease"
    wd.mkdir(parents=True, exist_ok=True)
    lease = EnvLease(workdir=wd, session="ext-hello-world-abc123", task_id="hello-world")

    class _Task:
        task_id = "hello-world"
        metadata = {"task_name": "hello-world"}

    async def _drive():
        async with provider.provision(_Task(), lease) as env:
            assert env.task_id == "hello-world"
            assert env.run_label == "ext-hello-world-abc123"
            # The provider installed BOTH lease hooks (the cancelled-path backstop).
            assert callable(lease.hard_kill)
            assert callable(lease.teardown)
            # hard_kill is a no-op when no runner pid is stamped (never raises).
            lease.hard_kill()

    asyncio.run(_drive())
    # OH and T2 backends share the label-scoped teardown contract.
    assert hasattr(OpenHandsTBBackend, "_force_compose_down")
    assert hasattr(Terminus2Backend, "_force_compose_down")
