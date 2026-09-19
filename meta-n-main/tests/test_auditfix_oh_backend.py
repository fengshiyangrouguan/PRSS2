"""Regression tests for the OpenHands-backend audit fixes (findings 9, 36, 37, 38).

All tests are LLM-free / offline: no LM Studio, no Docker, no network. The
finding-9 test spawns a tiny *local* python script (``sys.executable``) as a
stand-in runner that writes to stderr and exits WITHOUT producing the result
JSON — exercising the parse_error path purely with the standard library.

Each test FAILS on the pre-fix code and PASSES after the fix.
"""

from __future__ import annotations

import asyncio
import importlib
from pathlib import Path

import pytest

from meta_n.core.external_agents.backend import AgentRunContext, Prompt
from meta_n.core.external_agents.backends import openhands as oh_mod
from meta_n.core.external_agents.backends.openhands import OpenHandsBackend
from meta_n.core.external_agents.terminated import TerminatedBy


# --------------------------------------------------------------------------- #
# Finding 9 — runner stderr is surfaced on a degraded (parse_error) row.
# --------------------------------------------------------------------------- #
_FAKE_RUNNER = (
    "import sys\n"
    "sys.stderr.write('OH_RUNNER_TRACEBACK_MARKER: boom\\n')\n"
    "sys.exit(1)\n"  # exits without writing oh_result.json -> parse_error path
)


def _make_ctx(tmp_path: Path, ws: Path) -> AgentRunContext:
    return AgentRunContext(
        instruction="solve the task",
        prompt=Prompt(system_suffix="", prefix=""),
        workspace=str(ws),
        time_limit_s=30.0,
        max_turns=1,
        token_budget=0,
        max_budget_usd=0.0,
        logging_dir=tmp_path / "runlog",
    )


def test_finding9_runner_stderr_surfaced_on_parse_error(tmp_path):
    runner = tmp_path / "fake_runner.py"
    runner.write_text(_FAKE_RUNNER)
    ws = tmp_path / "ws"
    ws.mkdir()

    backend = OpenHandsBackend(
        model="dummy/model",
        venv_python=__import__("sys").executable,  # a real interpreter, offline
        runner_script=str(runner),
        local_default=True,
    )
    ctx = _make_ctx(tmp_path, ws)
    res = asyncio.run(backend.run(ctx, None, None))

    # No result JSON was written -> degraded parse_error row.
    assert res.failure_mode == "parse_error"
    assert res.terminated_by == TerminatedBy.PARSE_ERROR
    # The fix: the runner's stderr (its traceback) is retained, not dropped.
    assert "OH_RUNNER_TRACEBACK_MARKER" in res.stderr_tail


def test_finding9_clean_path_keeps_empty_stderr(tmp_path):
    """Happy-path parity: a runner that writes a valid completed result JSON must
    still yield stderr_tail == "" (no behavior change on success)."""
    runner = tmp_path / "ok_runner.py"
    # Parse --result-file out of argv and write a minimal completed result.
    runner.write_text(
        "import sys, json\n"
        "args = sys.argv\n"
        "rf = args[args.index('--result-file') + 1]\n"
        "sys.stderr.write('noise that must not leak on success\\n')\n"
        "open(rf, 'w').write(json.dumps({'status': 'finished', "
        "'last_message': 'done', 'command_history': []}))\n"
    )
    ws = tmp_path / "ws"
    ws.mkdir()
    backend = OpenHandsBackend(
        model="dummy/model",
        venv_python=__import__("sys").executable,
        runner_script=str(runner),
        local_default=True,
    )
    res = asyncio.run(backend.run(_make_ctx(tmp_path, ws), None, None))
    assert res.failure_mode is None
    assert res.stderr_tail == ""


# --------------------------------------------------------------------------- #
# Finding 36 — dead stage_files method removed.
# --------------------------------------------------------------------------- #
def test_finding36_stage_files_method_removed():
    assert not hasattr(OpenHandsBackend, "stage_files")


# --------------------------------------------------------------------------- #
# Finding 37 — dead _SAFE_ENV_KEYS import/re-export removed.
# --------------------------------------------------------------------------- #
def test_finding37_safe_env_keys_not_reexported():
    mod = importlib.reload(oh_mod)
    assert not hasattr(mod, "_SAFE_ENV_KEYS")
    assert "_SAFE_ENV_KEYS" not in mod.__all__


# --------------------------------------------------------------------------- #
# Finding 38 — write-only attributes (_mode/poll_interval_s/venv_bin) removed,
# while the parameters' live uses (mode validation, venv_bin -> self._py) remain.
# --------------------------------------------------------------------------- #
def test_finding38_write_only_attrs_removed():
    b = OpenHandsBackend(model="dummy/model", local_default=True)
    assert not hasattr(b, "_mode")
    assert not hasattr(b, "poll_interval_s")
    assert not hasattr(b, "venv_bin")


def test_finding38_mode_validation_still_live():
    with pytest.raises(ValueError):
        OpenHandsBackend(model="dummy/model", mode="bogus", local_default=True)


def test_finding38_venv_bin_param_still_resolves_py(tmp_path):
    b = OpenHandsBackend(
        model="dummy/model", venv_bin=str(tmp_path / "bin"), local_default=True
    )
    assert b._py == str(tmp_path / "bin" / "python")
