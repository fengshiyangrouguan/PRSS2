"""Shared subprocess-bridge primitives (``external_agents._bridge``).

Install-free: drives the single-sourced ``hard_timeout`` / ``sigkill_group`` /
``sanitize_compose_name`` / ``scrubbed_child_env`` helpers and asserts both
backends delegate to them. Imports only ``meta_n`` + stdlib (never
``openhands`` / ``terminal_bench``).
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from meta_n.core.external_agents import _bridge


# --- hard_timeout: single-sourced formula -----------------------------------


def test_hard_timeout_default_when_no_soft():
    assert _bridge.hard_timeout(None) == _bridge.DEFAULT_HARD_TIMEOUT_S
    assert _bridge.hard_timeout(None, default=42.0) == 42.0


def test_hard_timeout_slack_formula():
    # max(soft*1.25, soft+120)
    assert _bridge.hard_timeout(1000.0) == pytest.approx(1250.0)  # *1.25 wins
    assert _bridge.hard_timeout(100.0) == pytest.approx(220.0)    # +120 wins


def test_both_backends_share_the_hard_timeout_formula():
    from meta_n.core.external_agents.backends.openhands import OpenHandsBackend
    from meta_n.core.external_agents.backends.terminus2 import Terminus2Backend

    oh = OpenHandsBackend.__new__(OpenHandsBackend)
    t2 = Terminus2Backend.__new__(Terminus2Backend)
    for soft in (None, 100.0, 1000.0):
        assert oh._hard_timeout(soft) == _bridge.hard_timeout(soft)
        assert t2._hard_timeout(soft) == _bridge.hard_timeout(soft)


# --- sigkill_group: returncode/pid-reuse guard ------------------------------


def test_sigkill_group_noop_when_already_reaped(monkeypatch):
    """A process whose returncode is set is skipped (pid/pgid-reuse guard)."""
    called = {"n": 0}

    def _boom(*a, **k):  # pragma: no cover - must NOT be called
        called["n"] += 1
        raise AssertionError("killpg must not fire on a reaped process")

    monkeypatch.setattr(os, "killpg", _boom)
    monkeypatch.setattr(os, "getpgid", lambda pid: pid)
    # returncode is set -> the child was already reaped -> skip signalling.
    _bridge.sigkill_group(SimpleNamespace(pid=4242, returncode=0))
    assert called["n"] == 0


def test_sigkill_group_signals_live_process(monkeypatch):
    sent = {}
    monkeypatch.setattr(os, "getpgid", lambda pid: 999)
    monkeypatch.setattr(os, "killpg", lambda pgid, sig: sent.update(pgid=pgid, sig=sig))
    _bridge.sigkill_group(SimpleNamespace(pid=4242, returncode=None))
    assert sent["pgid"] == 999  # signalled the captured group


def test_sigkill_group_never_raises_on_none_or_lookup(monkeypatch):
    _bridge.sigkill_group(None)  # no proc
    _bridge.sigkill_group(SimpleNamespace(pid=None, returncode=None))  # no pid

    def _raise(pid):
        raise ProcessLookupError

    monkeypatch.setattr(os, "getpgid", _raise)
    # ProcessLookupError is suppressed (process already gone).
    _bridge.sigkill_group(SimpleNamespace(pid=1, returncode=None))


# --- sanitize_compose_name: one canonical sanitizer -------------------------


def test_sanitize_collapses_and_strips():
    assert _bridge.sanitize_compose_name("ext-Foo_Bar-99") == "ext-foo_bar-99"
    assert _bridge.sanitize_compose_name("../evil/!!") == "evil"
    assert _bridge.sanitize_compose_name("a--b---c") == "a-b-c"  # collapse runs


def test_sanitize_fallback_and_leading_digit():
    assert _bridge.sanitize_compose_name("", fallback="t2-bridge") == "t2-bridge"
    out = _bridge.sanitize_compose_name("!!!")  # reduces to empty -> fallback "0"
    assert out and out[0].isalnum()


# --- scrubbed_child_env: secrets never cross the boundary --------------------


def test_scrubbed_child_env_excludes_secrets(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-secret")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "az-secret")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ant-secret")
    monkeypatch.setenv("PATH", "/usr/bin")  # allowlisted
    env = _bridge.scrubbed_child_env({"OPENAI_API_KEY": "explicit"})
    # The explicitly-injected key crosses; no meta-n secret leaks.
    assert env["OPENAI_API_KEY"] == "explicit"
    assert env.get("PATH") == "/usr/bin"
    for secret in ("OPENROUTER_API_KEY", "AZURE_OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        assert secret not in env


def test_scrubbed_child_env_drops_none_extra():
    env = _bridge.scrubbed_child_env({"X": None, "Y": "v"})
    assert "X" not in env
    assert env["Y"] == "v"
