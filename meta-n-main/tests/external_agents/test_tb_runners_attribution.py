"""H3 producer (attribution_available) + H15 control-key filter — TB runners.

These exercise the PRODUCER side of the H3 attribution contract directly on the
three terminal-bench subprocess runners:

* each runner's ERROR payload hard-sets ``attribution_available=False`` (a degraded
  path is UNMEASURABLE — the consumer maps False → None);
* the t2 runner's ``_CAPTURE_WIRED`` flag flips True when the ``send_keys`` capture
  wrapper is installed (the success payload advertises capture-POSSIBLE, NOT
  capture-happened); and
* the t2 runner's ``_command_history_strings`` filters pure control-key sends
  (``Enter`` / ``Tab`` / ``C-m`` …) so ``command_count`` is a count of command
  events, not inflated by control-plane keystrokes (H15).

Install-free + stdlib-only: the runner modules keep their SDK imports lazy (inside
``_run`` / ``_build_*``), so the module top level is stdlib-only and importable
here by adding ``scripts/`` to ``sys.path`` — the same pattern the subprocess
bridge uses to launch ``-m <runner>`` and the same pattern
``test_oh_token_budget`` / ``test_runner_common`` use.
"""

from __future__ import annotations

import sys
from pathlib import Path

# scripts/ holds the runners; add it so we can import the install-free top level.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import builtin_tb_runner  # noqa: E402  (after sys.path insert)
import oh_tb_runner  # noqa: E402
import t2_runner  # noqa: E402


# --- H15: t2 control-key keystroke filter ----------------------------------


def test_t2_is_control_only_classifies_keystrokes():
    # Named control keys and C-/M- chords are control-only.
    for keystroke in ("Enter", "Tab", "Escape", "C-m", "C-c", "M-.", "Escape Enter"):
        assert t2_runner._is_control_only(keystroke), keystroke
    # Real commands (or a command mixed with a trailing Enter token) are NOT.
    for command in ("ls -la", "python3 helpers/greet.py", "git status Enter"):
        assert not t2_runner._is_control_only(command), command


def test_t2_command_history_filters_control_only_sends():
    t2_runner._COMMANDS_EXECUTED.clear()
    t2_runner._COMMANDS_EXECUTED.extend(
        ["ls -la", "Enter", "C-m", "git commit", "Tab", "   "]
    )
    out = t2_runner._command_history_strings()
    # Only the two real commands survive; bare keystroke sends + blanks are dropped.
    assert out == ["ls -la", "git commit"]
    t2_runner._COMMANDS_EXECUTED.clear()


# --- H3: t2 capture-wired flag reflects the send_keys wrap ------------------


class _FakeSession:
    def __init__(self):
        self.sent: list = []

    def send_keys(self, keys, *a, **k):
        self.sent.append(keys)


def test_t2_install_command_capture_sets_capture_wired():
    t2_runner._CAPTURE_WIRED = False
    t2_runner._COMMANDS_EXECUTED.clear()
    sess = _FakeSession()
    t2_runner._install_command_capture(sess)
    # Wiring the wrapper flips the producer flag (capture is now POSSIBLE).
    assert t2_runner._CAPTURE_WIRED is True
    # The wrap is transparent: the command is BOTH captured and relayed unchanged.
    sess.send_keys("echo hi")
    assert sess.sent == ["echo hi"]
    assert "echo hi" in t2_runner._COMMANDS_EXECUTED
    t2_runner._COMMANDS_EXECUTED.clear()
    t2_runner._CAPTURE_WIRED = False


# --- H3: every runner's error payload is UNMEASURABLE (attribution False) ----


def test_t2_error_payload_attribution_false():
    p = t2_runner._error_payload(
        "t-1", "env_error", "boom", command_history=["ls"]
    )
    # A degraded path is UNMEASURABLE even though partial commands were captured.
    assert p["attribution_available"] is False


def test_oh_tb_error_payload_attribution_false():
    p = oh_tb_runner._error_payload(
        "t-1", "env_error", "boom", command_history=["ls"]
    )
    assert p["attribution_available"] is False


def test_builtin_tb_error_payload_attribution_false():
    p = builtin_tb_runner._error_payload("t-1", "env_error", "boom")
    assert p["attribution_available"] is False


# --- R2-EA-1: OH-TB ships last_message + cache_read_tokens (parity oh_runner) --


def test_oh_tb_error_payload_emits_last_message_and_cache_reads():
    # The OH-TB consumer (openhands_tb._extra_result_fields) reads ``last_message``
    # -> reasoning_summary and ``cache_read_tokens`` -> agent_cached_tokens. The
    # runner never shipped either key, so they silently defaulted to ""/0 on every
    # OH-over-TB run. The error payload now carries the partial recovered values.
    p = oh_tb_runner._error_payload(
        "t-1", "token_budget", "boom",
        total_input_tokens=10, total_output_tokens=5,
        last_message="partial answer", cache_read_tokens=42,
    )
    assert p["last_message"] == "partial answer"
    assert p["cache_read_tokens"] == 42


def test_oh_tb_error_payload_defaults_last_message_and_cache_reads():
    # Absent partial state, the keys are present with clean defaults (not missing),
    # so the consumer's defensive ``.get(...)`` reads a real measured value.
    p = oh_tb_runner._error_payload("t-1", "env_error", "boom")
    assert p["last_message"] == ""
    assert p["cache_read_tokens"] == 0


def test_oh_tb_last_agent_message_events_parses_last_agent_text(monkeypatch):
    # Verify the helper joins the LAST source=='agent' message text. Inject a stub
    # ``openhands.sdk.event.MessageEvent`` (the SDK is absent in meta-n's env) so
    # the function-scoped import resolves to our shape, mirroring oh_runner's
    # _last_agent_message contract.
    import sys
    import types

    class _MessageEvent:
        def __init__(self, source, texts):
            self.source = source
            self.llm_message = types.SimpleNamespace(
                content=[types.SimpleNamespace(text=t) for t in texts]
            )

    pkg = types.ModuleType("openhands")
    sdk = types.ModuleType("openhands.sdk")
    event_mod = types.ModuleType("openhands.sdk.event")
    event_mod.MessageEvent = _MessageEvent
    pkg.sdk = sdk
    sdk.event = event_mod
    monkeypatch.setitem(sys.modules, "openhands", pkg)
    monkeypatch.setitem(sys.modules, "openhands.sdk", sdk)
    monkeypatch.setitem(sys.modules, "openhands.sdk.event", event_mod)

    events = [
        _MessageEvent("user", ["hi"]),
        _MessageEvent("agent", ["first ", "agent msg"]),
        _MessageEvent("agent", ["FINAL ", "answer"]),
        _MessageEvent("user", ["thanks"]),
    ]
    assert oh_tb_runner._last_agent_message_events(events) == "FINAL answer"


def test_oh_tb_last_agent_message_events_degrades_to_empty():
    # Never raises: a falsy event stream, or events that are not MessageEvents
    # (incl. the SDK-absent ImportError path), return "" rather than blowing up —
    # the module-top import-safety / best-effort-summary contract.
    assert oh_tb_runner._last_agent_message_events([]) == ""
    assert oh_tb_runner._last_agent_message_events(None) == ""
    assert oh_tb_runner._last_agent_message_events([object(), object()]) == ""
