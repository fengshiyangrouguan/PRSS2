"""R5 regression tests for the terminal-bench integration + TB spine bridge.

Covers:
  * per-task verifier-timeout threading: the task's declared
    ``max_test_timeout_sec`` rides the env handle into the bridge request's
    ``test_timeout_sec`` (the runners forward it as the harness's
    ``global_test_timeout_sec``); the 180s bridge default applies only when
    the task declares none;
  * the legacy ``task.yaml`` layout guard: a caller that declares a NATIVE
    base solver fails at load time (pre-spend — before any solver LLM call),
    the native executor refuses the layout before any Docker work, and the
    spine-routed kinds keep loading the same tasks;
  * the litellm provider-prefix routing contract for provider-headed pricing
    keys (``anthropic/claude-...`` is deliberately left unrouted at this seam
    and must arrive pre-prefixed — see the ``_LITELLM_PROVIDER_PREFIXES``
    caveat in adapter.py).

No Docker / SDK / LLM required.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from unittest.mock import AsyncMock, patch

from meta_n.core.external_agents.backend import AgentRunContext, Prompt
from meta_n.core.external_agents.backends.terminus2 import Terminus2Backend
from meta_n.core.meta_layer import TaskDescription
from meta_n.integrations.terminal_bench import (
    TerminalBenchAdapter,
    TerminalBenchExecutor,
    _litellm_route_model,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _WS:
    """Minimal workspace handle mirroring _TBTerminus2Env's contract."""

    def __init__(self, task: TaskDescription | None = None):
        self.task_id = "hello-world"
        self.staged_files: dict[str, str] = {}
        self.run_label = ""
        self.agent_pid = None
        self.task = task


def _backend() -> Terminus2Backend:
    return Terminus2Backend(
        model="openai/google/gemma-4-31b-qat",
        api_base="http://127.0.0.1:1234/v1",
        venv_python="/nonexistent/python",
        runner_dir="/tmp",
        tasks_dir="/tmp/tasks",
    )


def _ctx(logging_dir, ws: _WS) -> AgentRunContext:
    return AgentRunContext(
        instruction="do it",
        prompt=Prompt(),
        workspace=ws,
        time_limit_s=60.0,
        max_turns=8,
        token_budget=0,
        max_budget_usd=2.0,
        logging_dir=logging_dir,
    )


def _drive_run_and_read_request(tmp_path, monkeypatch, ws: _WS) -> dict:
    """Run the bridge with a fake subprocess and return the request JSON."""

    async def _fake_exec(*cmd, env=None, **kwargs):
        class _P:
            pid = 111
            returncode = 0

            async def communicate(self):
                return (b"", b"")

            async def wait(self):
                return 0

        return _P()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_exec)
    b = _backend()
    logs = tmp_path / "logs"
    asyncio.run(b.run(_ctx(logs, ws), None, None))
    return json.loads((logs / "t2_request.json").read_text())


def _write_legacy_task(root, name="hello-world", extra_yaml=""):
    d = root / name
    d.mkdir()
    (d / "task.yaml").write_text("instruction: say hello\n" + extra_yaml)
    (d / "tests").mkdir()
    (d / "Dockerfile").write_text("FROM ubuntu:22.04\n")
    return d


# ---------------------------------------------------------------------------
# 1. Per-task verifier-timeout threading (spine bridge)
# ---------------------------------------------------------------------------

class TestVerifierTimeoutThreading:
    def test_declared_max_test_timeout_reaches_request(self, tmp_path, monkeypatch):
        task = TaskDescription(
            task_id="hello-world",
            description="x",
            metadata={"max_test_timeout_sec": 600},
        )
        req = _drive_run_and_read_request(tmp_path, monkeypatch, _WS(task=task))
        assert req["test_timeout_sec"] == 600.0

    def test_undeclared_falls_back_to_bridge_default(self, tmp_path, monkeypatch):
        task = TaskDescription(
            task_id="hello-world",
            description="x",
            metadata={"max_test_timeout_sec": None},
        )
        req = _drive_run_and_read_request(tmp_path, monkeypatch, _WS(task=task))
        assert req["test_timeout_sec"] == 180.0

    def test_no_task_on_handle_falls_back(self, tmp_path, monkeypatch):
        req = _drive_run_and_read_request(tmp_path, monkeypatch, _WS(task=None))
        assert req["test_timeout_sec"] == 180.0

    def test_resolve_helper_rejects_garbage_and_nonpositive(self, tmp_path):
        b = _backend()
        for bad in ("nope", -5, 0, "", None):
            task = TaskDescription(
                task_id="t", description="x",
                metadata={"max_test_timeout_sec": bad},
            )
            ctx = _ctx(tmp_path, _WS(task=task))
            assert b._resolve_test_timeout_sec(ctx) == 180.0

    def test_resolve_helper_accepts_numeric_string(self, tmp_path):
        b = _backend()
        task = TaskDescription(
            task_id="t", description="x",
            metadata={"max_test_timeout_sec": "3600"},
        )
        ctx = _ctx(tmp_path, _WS(task=task))
        assert b._resolve_test_timeout_sec(ctx) == 3600.0

    def test_legacy_loader_threads_declared_value(self, tmp_path):
        _write_legacy_task(tmp_path, "declared", "max_test_timeout_sec: 600\n")
        _write_legacy_task(tmp_path, "undeclared")
        a = TerminalBenchAdapter(task_cache_dir=str(tmp_path))
        try:
            tasks = {t.metadata["task_name"]: t for t in a.load_tasks()}
            assert tasks["declared"].metadata["max_test_timeout_sec"] == 600
            assert tasks["undeclared"].metadata["max_test_timeout_sec"] is None
            # The native-path fallback key is untouched by the threading.
            assert tasks["declared"].metadata["verifier_timeout_sec"] == 600
            assert tasks["undeclared"].metadata["verifier_timeout_sec"] == 900
        finally:
            a.cleanup()


# ---------------------------------------------------------------------------
# 2. Legacy task.yaml layout guard (native path fails pre-spend)
# ---------------------------------------------------------------------------

class TestLegacyLayoutGuard:
    def test_native_declared_base_solver_raises_at_load(self, tmp_path):
        _write_legacy_task(tmp_path)
        a = TerminalBenchAdapter(task_cache_dir=str(tmp_path), base_solver=None)
        try:
            with pytest.raises(RuntimeError, match="--base-solver"):
                a.load_tasks()
        finally:
            a.cleanup()

    @pytest.mark.parametrize("kind", ["terminus2", "openhands", "builtin"])
    def test_spine_declared_base_solver_loads(self, tmp_path, kind):
        _write_legacy_task(tmp_path)
        a = TerminalBenchAdapter(task_cache_dir=str(tmp_path), base_solver=kind)
        try:
            tasks = a.load_tasks()
            assert len(tasks) == 1
            assert tasks[0].metadata["layout"] == "legacy_yaml"
        finally:
            a.cleanup()

    def test_undeclared_base_solver_keeps_loading(self, tmp_path):
        # Callers that do not declare the route (back-compat) still load.
        _write_legacy_task(tmp_path)
        a = TerminalBenchAdapter(task_cache_dir=str(tmp_path))
        try:
            assert len(a.load_tasks()) == 1
        finally:
            a.cleanup()

    def test_native_declared_harbor_layout_unaffected(self, tmp_path):
        # The guard fires only for the legacy layout; harbor task.toml dirs
        # keep loading under a declared native base solver.
        d = tmp_path / "harbor-task"
        d.mkdir()
        (d / "task.toml").write_text("[environment]\ncpus = 1\n")
        (d / "instruction.md").write_text("x")
        (d / "environment").mkdir()
        (d / "environment" / "Dockerfile").write_text("FROM x\n")
        (d / "tests").mkdir()
        (d / "tests" / "test.sh").write_text("echo ok\n")
        a = TerminalBenchAdapter(task_cache_dir=str(tmp_path), base_solver=None)
        try:
            assert len(a.load_tasks()) == 1
        finally:
            a.cleanup()

    def test_executor_refuses_legacy_task_before_docker(self, tmp_path):
        task_dir = _write_legacy_task(tmp_path)
        a = TerminalBenchAdapter(task_cache_dir=str(tmp_path))
        executor = TerminalBenchExecutor(a)
        task = TaskDescription(
            task_id="hello-world",
            description="say hello",
            metadata={
                "layout": "legacy_yaml",
                "task_dir": str(task_dir),
            },
        )
        compose_calls: list = []

        async def _record_compose(*args, **kwargs):
            compose_calls.append(args)

        try:
            with patch(
                "meta_n.integrations.terminal_bench._run_compose_command",
                side_effect=_record_compose,
            ), patch.object(
                executor, "_preflight", new_callable=AsyncMock,
            ) as preflight, patch.object(
                a, "_ensure_image", new_callable=AsyncMock,
            ) as ensure_image:
                with pytest.raises(RuntimeError, match="external-agent spine"):
                    asyncio.run(executor.execute("echo hello", task))
            preflight.assert_not_awaited()
            ensure_image.assert_not_awaited()
            assert compose_calls == []
        finally:
            a.cleanup()


# ---------------------------------------------------------------------------
# 3. litellm routing contract for provider-headed pricing keys
# ---------------------------------------------------------------------------

class TestLitellmRoutingContract:
    def test_provider_headed_pricing_key_left_unrouted(self):
        # Pinned contract (see _LITELLM_PROVIDER_PREFIXES caveat): a pricing
        # key whose head IS a litellm provider is passed through unrouted, so
        # at the TB bridge seam (OPENAI_API_KEY-only child env) such ids must
        # arrive pre-prefixed (openai/... or openrouter/...).
        assert (
            _litellm_route_model("anthropic/claude-sonnet-4-20250514")
            == "anthropic/claude-sonnet-4-20250514"
        )

    def test_non_provider_pricing_key_gets_openai_prefix(self):
        assert (
            _litellm_route_model("google/gemma-4-31b-qat")
            == "openai/google/gemma-4-31b-qat"
        )

    def test_already_routed_id_is_idempotent(self):
        assert (
            _litellm_route_model("openai/google/gemma-4-31b-qat")
            == "openai/google/gemma-4-31b-qat"
        )
