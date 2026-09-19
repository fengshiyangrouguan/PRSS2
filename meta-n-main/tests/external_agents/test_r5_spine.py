"""R5 spine fixes — TB helper-staging landing, telemetry index robustness,
OH stdout-fallback recovery, OH attribution capture-wiredness.

Install-free (plan §12.1): no SDK, no Docker, no LLM. The staging tests drive
the REAL ``_runner_common.stage_helper_files`` with a recording fake session
(the command list the code builds); the OH tests drive the REAL
``OpenHandsBackend.run`` over the subprocess bridge with a stand-in runner
(the ``test_oh_terminated.py`` harness pattern).
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

# scripts/ holds the shared runner base; add it so we can import it directly
# (same pattern as test_runner_common.py).
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import _runner_common  # noqa: E402  (after sys.path insert)

from meta_n.core.external_agents.backend import AgentRunContext, Prompt  # noqa: E402
from meta_n.core.external_agents.backends.openhands import OpenHandsBackend  # noqa: E402
from meta_n.core.external_agents.injection import InjectionMapper  # noqa: E402
from meta_n.core.external_agents.telemetry import AgentTelemetry  # noqa: E402
from meta_n.core.external_agents.telemetry.attribution import (  # noqa: E402
    attribute_utilities,
)
from meta_n.core.external_agents.telemetry.writer import iter_jsonl_objects  # noqa: E402
from meta_n.core.external_agents.terminated import TerminatedBy  # noqa: E402
from meta_n.core.meta_layer import SandboxMarker  # noqa: E402

from .conftest import make_injected, make_task  # noqa: E402


# ---------------------------------------------------------------------------
# 1. stage_helper_files — injection keys land at /app/helpers, not nested
# ---------------------------------------------------------------------------


class _FakeSession:
    """Records ``copy_to_container`` calls (the command list the code builds)."""

    def __init__(self):
        self.calls: list[dict] = []

    def copy_to_container(self, *, paths, container_dir):
        self.calls.append({"paths": list(paths), "container_dir": container_dir})


def _landings(sess: _FakeSession) -> set[str]:
    """Container paths where each staged file lands (dir + tar-arcname basename)."""
    return {
        f"{c['container_dir']}/{Path(p).name}"
        for c in sess.calls
        for p in c["paths"]
    }


def test_injection_shaped_keys_land_at_app_helpers():
    sess = _FakeSession()
    _runner_common.stage_helper_files(
        sess,
        {
            "helpers/foo.py": "def foo():\n    return 1\n",
            "helpers/__init__.py": "from .foo import foo\n",
            "helpers/_lib_foo.py": "print('lib')\n",
            "helpers/bar.sh": "bar() { :; }\n",
        },
        prefix="t2",
    )
    assert _landings(sess) == {
        "/app/helpers/foo.py",
        "/app/helpers/__init__.py",
        "/app/helpers/_lib_foo.py",
        "/app/helpers/bar.sh",
    }


def test_real_injection_plan_keys_land_at_app_helpers():
    """End-to-end producer→consumer: real InjectionMapper keys land as advertised."""
    source = (
        "def greet(name):\n"
        '    """Return a greeting for name.\n\n'
        "    Second line of the docstring.\n"
        '    """\n'
        "    return f'hi {name}'\n"
    )
    plan = InjectionMapper(
        [make_injected(code_library={"greet": source})], "python", SandboxMarker()
    ).build(make_task())
    assert set(plan.staged_files) == {"helpers/greet.py", "helpers/__init__.py"}

    sess = _FakeSession()
    _runner_common.stage_helper_files(sess, plan.staged_files, prefix="t2")
    # `from helpers import greet` at WORKDIR /app requires exactly these paths.
    assert _landings(sess) == {
        "/app/helpers/greet.py",
        "/app/helpers/__init__.py",
    }


def test_traversal_guards_still_reject_unsafe_keys():
    sess = _FakeSession()
    _runner_common.stage_helper_files(
        sess,
        {"/etc/passwd": "pwned", "../../x": "escape", "helpers/../../y": "escape"},
        prefix="oh_tb",
    )
    assert sess.calls == []


# ---------------------------------------------------------------------------
# 2. telemetry readers — valid-JSON non-object lines must not raise
# ---------------------------------------------------------------------------


def test_iter_jsonl_objects_skips_non_object_top_levels(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text('[1, 2, 3]\n"hello"\nnull\n42\n{"a": 1}\n')
    assert list(iter_jsonl_objects(p)) == [{"a": 1}]


def test_load_run_id_index_survives_malformed_lines(tmp_path):
    tel_dir = tmp_path / "run" / "telemetry"
    tel_dir.mkdir(parents=True)
    good = {"extra": {"record": {"run_id": "good", "terminated_by": "timeout"}}}
    (tel_dir / "agent_runs.jsonl").write_text(
        "\n".join(
            [
                "[1, 2, 3]",                                # list top level
                '"hello"',                                  # scalar top level
                "null",                                     # null top level
                '{"extra": {"record": null}}',              # null-record envelope
                '{"extra": {"record": [1]}}',               # non-dict record
                '{"extra": "bogus", "run_id": "toplvl"}',   # non-dict extra
                json.dumps(good),                           # the one good record
            ]
        )
        + "\n"
    )
    tel = AgentTelemetry(tmp_path / "run")  # must not raise
    assert "good" in tel._seen_run_ids  # noqa: SLF001
    assert "good" in tel._degraded_run_ids  # noqa: SLF001 (timeout is degraded)
    # Non-dict extra falls back to the top level (forward compatibility).
    assert "toplvl" in tel._seen_run_ids  # noqa: SLF001


# ---------------------------------------------------------------------------
# 3+4. OpenHandsBackend — stdout fallback + attribution wiredness
# (real backend.run() over a stand-in runner; no openhands install)
# ---------------------------------------------------------------------------

_RUNNER_TEMPLATE = """
import argparse, json, sys
from pathlib import Path

p = argparse.ArgumentParser()
for flag in (
    "--workspace", "--instruction-file", "--system-suffix-file",
    "--result-file", "--model", "--base-url", "--api-key",
    "--max-output-tokens", "--max-iterations", "--max-budget-usd",
    "--token-budget", "--solution-file",
):
    p.add_argument(flag)
a = p.parse_args()
{body}
"""


def _write_runner(tmp_path: Path, body: str) -> Path:
    runner = tmp_path / "fake_runner.py"
    runner.write_text(_RUNNER_TEMPLATE.format(body=textwrap.dedent(body)))
    return runner


def _make_ctx(tmp_path: Path) -> AgentRunContext:
    ws = tmp_path / "workspace"
    ws.mkdir(parents=True, exist_ok=True)
    return AgentRunContext(
        instruction="solve the task",
        prompt=Prompt(system_suffix="", prefix=""),
        workspace=str(ws),
        time_limit_s=30.0,
        max_turns=8,
        token_budget=10_000,
        max_budget_usd=0.0,
        logging_dir=tmp_path / "log",
    )


async def _run_backend(tmp_path: Path, body: str):
    backend = OpenHandsBackend(
        model="google/gemma-4-31b-qat",
        venv_python=sys.executable,  # host python; the stand-in needs no SDK
        runner_script=str(_write_runner(tmp_path, body)),
        local_default=True,
    )
    return await backend.run(_make_ctx(tmp_path), tel=None, rec=None)


_STDOUT_ONLY_PAYLOAD = {
    "status": "finished",
    "last_message": "done",
    "command_history": [{"command": "echo hi", "output": "hi"}],
    "prompt_tokens": 1234,
    "completion_tokens": 567,
    "cache_read_tokens": 0,
    "agent_calls": 3,
    "accumulated_cost_usd": 0.42,
    "error": None,
}


async def test_stdout_fallback_recovers_status_and_tokens(tmp_path):
    """Result file unwritable → runner emits payload on stdout → backend recovers."""
    body = f"""
        payload = json.loads({json.dumps(json.dumps(_STDOUT_ONLY_PAYLOAD))})
        sys.stdout.write(json.dumps(payload))
        sys.exit(1)
    """
    res = await _run_backend(tmp_path, body)
    assert res.terminated_by is TerminatedBy.COMPLETED
    assert res.failure_mode is None
    assert res.agent_prompt_tokens == 1234
    assert res.agent_completion_tokens == 567
    assert res.agent_tokens == 1801
    assert res.agent_calls == 3
    assert res.cost_usd == 0.42
    assert res.command_history == ["echo hi"]


async def test_no_file_and_no_stdout_payload_stays_parse_error(tmp_path):
    """Neither result file nor a parseable stdout payload → prior behavior."""
    body = """
        sys.stdout.write("not json")
        sys.exit(1)
    """
    res = await _run_backend(tmp_path, body)
    assert res.terminated_by is TerminatedBy.PARSE_ERROR
    assert res.failure_mode == "parse_error"
    assert res.agent_tokens == 0
    assert res.attribution_available is False


async def test_result_file_still_wins_over_stdout(tmp_path):
    """When the result file exists it stays authoritative; stdout is ignored."""
    file_payload = dict(_STDOUT_ONLY_PAYLOAD, prompt_tokens=10, completion_tokens=5)
    body = f"""
        payload = json.loads({json.dumps(json.dumps(file_payload))})
        Path(a.result_file).write_text(json.dumps(payload))
        sys.stdout.write("stray stdout noise")
        sys.exit(0)
    """
    res = await _run_backend(tmp_path, body)
    assert res.agent_tokens == 15
    assert res.terminated_by is TerminatedBy.COMPLETED


async def test_wired_zero_command_run_is_measured_zero(tmp_path):
    """Harvest ran (command_history present, error unset) with zero commands →
    attribution_available=True, so attribution yields the measured [] not None."""
    payload = {
        "status": "finished",
        "last_message": "answered in a message, no shell",
        "command_history": [],
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "accumulated_cost_usd": 0.0,
        "error": None,
    }
    body = f"""
        payload = json.loads({json.dumps(json.dumps(payload))})
        Path(a.result_file).write_text(json.dumps(payload))
        sys.exit(0)
    """
    res = await _run_backend(tmp_path, body)
    assert res.attribution_available is True
    assert res.command_history == []
    called, counts = attribute_utilities(
        res.command_history, ["greet"], res.attribution_available
    )
    assert called == []  # measured: none used — NOT None (unmeasurable)
    assert counts == {}


async def test_exception_payload_is_not_wired(tmp_path):
    """main()'s exception payload carries command_history=[] but the harvest
    never ran — key presence must not mark the run measurable."""
    payload = {
        "status": "error",
        "last_message": "",
        "command_history": [],
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cache_read_tokens": 0,
        "agent_calls": 0,
        "accumulated_cost_usd": 0.0,
        "model": "google/gemma-4-31b-qat",
        "solution_file_contents": "",
        "error": "RuntimeError: agent blew up mid-step",
    }
    body = f"""
        payload = json.loads({json.dumps(json.dumps(payload))})
        Path(a.result_file).write_text(json.dumps(payload))
        sys.exit(0)
    """
    res = await _run_backend(tmp_path, body)
    assert res.attribution_available is False
    called, _ = attribute_utilities(
        res.command_history, ["greet"], res.attribution_available
    )
    assert called is None  # unmeasurable — harvest never ran


async def test_wired_run_with_commands_stays_available(tmp_path):
    payload = {
        "status": "finished",
        "last_message": "ok",
        "command_history": [{"command": "ls", "output": ""}],
        "prompt_tokens": 1,
        "completion_tokens": 1,
        "accumulated_cost_usd": 0.0,
        "error": None,
    }
    body = f"""
        payload = json.loads({json.dumps(json.dumps(payload))})
        Path(a.result_file).write_text(json.dumps(payload))
        sys.exit(0)
    """
    res = await _run_backend(tmp_path, body)
    assert res.attribution_available is True
    assert res.command_history == ["ls"]
