"""_runner_common — shared venv-runner helpers (security guard + accounting).

``scripts/_runner_common.py`` is the single-sourced base for all FOUR subprocess
runners. Its module docstring labels ``stage_helper_files`` the
"security-load-bearing path-traversal guard": ``rel`` originates from the
UNTRUSTED Ω-generated ``staged_files`` map, and the runner writes to the HOST
(outside Docker), so an absolute / ``..`` key must be rejected before it can write
an arbitrary host file. That guard — and ``route_model`` / ``sanitize_run_label``
/ ``failure_mode_str`` / ``error_result_payload`` — were previously uncovered.

Install-free + stdlib-only: ``_runner_common`` imports no SDK; we add ``scripts/``
to ``sys.path`` (the same pattern the bridge uses to launch ``-m <runner>``) and
import it directly. Mirrors ``test_oh_token_budget.py``'s sys.path insert.
"""

from __future__ import annotations

import sys
from pathlib import Path

# scripts/ holds the shared runner base; add it so we can import it directly.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import _runner_common  # noqa: E402  (after sys.path insert)


# --- stage_helper_files: the path-traversal guard --------------------------


class _FakeSession:
    """Records ``copy_to_container`` calls so the guard's accept/skip is asserted."""

    def __init__(self):
        self.calls: list[dict] = []

    def copy_to_container(self, *, paths, container_dir):
        self.calls.append({"paths": list(paths), "container_dir": container_dir})


def test_stage_helper_files_rejects_absolute_key(tmp_path):
    sess = _FakeSession()
    _runner_common.stage_helper_files(
        sess, {"/etc/passwd": "pwned"}, prefix="t2"
    )
    # An absolute key is SKIPPED — no copy_to_container call, no host write.
    assert sess.calls == []


def test_stage_helper_files_rejects_parent_traversal(tmp_path):
    sess = _FakeSession()
    _runner_common.stage_helper_files(
        sess, {"../../x": "escape"}, prefix="t2"
    )
    assert sess.calls == []


def test_stage_helper_files_accepts_safe_nested_key(tmp_path):
    sess = _FakeSession()
    # Real injection-plan key shape: the producer stages workspace-relative
    # ``helpers/<name>.py`` keys, which must land at /app/helpers (WORKDIR /app).
    _runner_common.stage_helper_files(
        sess, {"helpers/util.py": "print('hi')"}, prefix="oh_tb"
    )
    assert len(sess.calls) == 1
    call = sess.calls[0]
    assert call["container_dir"] == "/app/helpers"
    staged = call["paths"][0]
    assert Path(staged).read_text() == "print('hi')"
    assert Path(staged).name == "util.py"


def test_stage_helper_files_top_level_key_lands_in_workdir(tmp_path):
    sess = _FakeSession()
    _runner_common.stage_helper_files(sess, {"util.py": "x = 1"}, prefix="t2")
    assert len(sess.calls) == 1
    assert sess.calls[0]["container_dir"] == "/app"


def test_stage_helper_files_empty_map_is_noop():
    sess = _FakeSession()
    _runner_common.stage_helper_files(sess, {}, prefix="t2")
    assert sess.calls == []


# --- route_model: litellm provider prefixing -------------------------------


def test_route_model_no_base_url_unchanged():
    assert _runner_common.route_model("google/gemma-4-31b-qat", None) == \
        "google/gemma-4-31b-qat"
    assert _runner_common.route_model("gpt-5.2", "") == "gpt-5.2"


def test_route_model_bare_model_with_base_url_gets_openai_prefix():
    out = _runner_common.route_model("google/gemma-4-31b-qat",
                                     "http://127.0.0.1:1234/v1")
    assert out == "openai/google/gemma-4-31b-qat"


def test_route_model_already_prefixed_unchanged():
    base = "http://127.0.0.1:1234/v1"
    assert _runner_common.route_model("openai/gpt-5.2", base) == "openai/gpt-5.2"
    assert _runner_common.route_model("anthropic/claude-sonnet-4", base) == \
        "anthropic/claude-sonnet-4"
    assert _runner_common.route_model("openrouter/x/y", base) == "openrouter/x/y"


# --- sanitize_run_label: compose-safe project name -------------------------


def test_sanitize_run_label_collapses_and_strips():
    # Unsafe chars -> hyphens; adjacent dashes collapse; leading/trailing -_ strip.
    assert _runner_common.sanitize_run_label("Ext-A..B-1") == "ext-a-b-1"
    assert _runner_common.sanitize_run_label("ext-mytask---x-1") == "ext-mytask-x-1"
    assert _runner_common.sanitize_run_label("ext_Task_1") == "ext_task_1"


def test_sanitize_run_label_empty_uses_fallback_uuid():
    out = _runner_common.sanitize_run_label("___", fallback="t2")
    assert out.startswith("t2-")
    assert len(out) > len("t2-")
    # Two empty labels never collide (per-call uuid).
    other = _runner_common.sanitize_run_label("", fallback="t2")
    assert out != other


# --- failure_mode_str: enum / str / None normalization ---------------------


def test_failure_mode_str_none_is_none_string():
    assert _runner_common.failure_mode_str(None) == "none"


def test_failure_mode_str_enum_uses_value():
    class _FM:
        value = "agent_timeout"

    assert _runner_common.failure_mode_str(_FM()) == "agent_timeout"


def test_failure_mode_str_bare_string_passthrough():
    assert _runner_common.failure_mode_str("token_budget") == "token_budget"


# --- error_result_payload: the single-sourced error schema -----------------


def test_error_result_payload_schema_and_defaults():
    p = _runner_common.error_result_payload("t-1", "env_error", "boom")
    assert p["ok"] is False
    assert p["status"] == "error"
    assert p["task_id"] == "t-1"
    assert p["failure_mode"] == "env_error"
    assert p["reward"] == 0.0 and p["score"] == 0.0 and p["is_resolved"] is False
    # Token/call fields default to 0; command_history defaults to [].
    assert p["total_input_tokens"] == 0
    assert p["total_output_tokens"] == 0
    assert p["agent_calls"] == 0
    assert p["command_history"] == []
    # The full key set the TB runners write (and _external_tb reads).
    expected_keys = {
        "ok", "status", "reward", "score", "is_resolved", "task_id",
        "total_input_tokens", "total_output_tokens", "agent_calls",
        "failure_mode", "parser_results", "timestamped_markers",
        "command_history", "transcript_path", "post_agent_pane",
        "post_test_pane", "wall_s", "steps", "trial_started_at",
        "trial_ended_at", "error",
    }
    assert set(p.keys()) == expected_keys


def test_error_result_payload_carries_partial_spend_and_history():
    p = _runner_common.error_result_payload(
        "t-2", "token_budget", "x" * 5000,
        total_input_tokens=100, total_output_tokens=50, agent_calls=3,
        command_history=["ls", "cat f"],
    )
    assert p["total_input_tokens"] == 100
    assert p["total_output_tokens"] == 50
    assert p["agent_calls"] == 3
    assert p["command_history"] == ["ls", "cat f"]
    # error text is truncated to the last 4000 chars.
    assert len(p["error"]) == 4000


# --- classify_runner_error: the single-sourced top-level except-guard ------


def test_classify_runner_error_token_budget_by_type():
    err = _runner_common.classify_runner_error(
        _runner_common.TokenBudgetExceeded("budget 100 exceeded")
    )
    assert err == "token_budget"


def test_classify_runner_error_token_budget_by_message():
    assert _runner_common.classify_runner_error(
        RuntimeError("inner token budget 100 exceeded")
    ) == "token_budget"


def test_classify_runner_error_env_error_substrings():
    for m in ("docker daemon down", "compose failed", "no such image",
              "container exited", "request is missing required field: 'task_id'"):
        assert _runner_common.classify_runner_error(RuntimeError(m)) == "env_error"


def test_classify_runner_error_default_unknown():
    assert _runner_common.classify_runner_error(RuntimeError("boom")) == \
        "unknown_agent_error"


def test_classify_runner_error_no_token_budget_branch():
    # The LLM-free builtin runner never emits token_budget: the type is ignored
    # and a bare "token budget" message falls through to the default.
    assert _runner_common.classify_runner_error(
        _runner_common.TokenBudgetExceeded("budget 100 exceeded"),
        include_token_budget=False,
    ) == "unknown_agent_error"
    assert _runner_common.classify_runner_error(
        RuntimeError("token budget 100 exceeded"), include_token_budget=False
    ) == "unknown_agent_error"
