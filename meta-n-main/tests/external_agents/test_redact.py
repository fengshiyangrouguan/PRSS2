"""redact() — secret masking (§12.1)."""

from __future__ import annotations

import pytest

from meta_n.core.external_agents.telemetry import (
    _REDACTED,
    redact,
)

_REAL_KEY_BODY = "A" * 40  # >= 20 chars to match the sk-/ghp_ shapes


@pytest.mark.parametrize(
    "secret",
    [
        f"OPENAI_API_KEY=sk-{_REAL_KEY_BODY}",
        f"OPENROUTER_API_KEY=or-{_REAL_KEY_BODY}",
        f"ANTHROPIC_API_KEY=sk-ant-{_REAL_KEY_BODY}",
        f"AZURE_OPENAI_API_KEY={_REAL_KEY_BODY}",
        f"GH_TOKEN=ghp_{_REAL_KEY_BODY}",
        f"GITHUB_TOKEN=ghp_{_REAL_KEY_BODY}",
    ],
)
def test_named_assignment_keys_are_masked(secret):
    out = redact(secret)
    assert _REDACTED in out
    # The secret body is gone.
    assert _REAL_KEY_BODY not in out
    # The KEY name is retained (only the value is masked).
    key = secret.split("=", 1)[0]
    assert key in out


def test_bare_sk_key_masked():
    out = redact(f"using sk-{_REAL_KEY_BODY} now")
    assert _REDACTED in out
    assert f"sk-{_REAL_KEY_BODY}" not in out


def test_bare_openrouter_key_masked():
    # An OpenRouter ``or-…`` value echoed WITHOUT its assignment-key prefix must
    # still be masked by the bare-token pattern (not only the KEY=value form).
    out = redact(f"using or-{_REAL_KEY_BODY} now")
    assert _REDACTED in out
    assert f"or-{_REAL_KEY_BODY}" not in out


def test_word_starting_with_or_dash_not_overmasked():
    # The bare ``or-`` pattern requires a >=20-char token, so an ordinary
    # hyphenated word like ``or-else`` is left untouched.
    text = "do this or-else nothing happens"
    assert redact(text) == text


def test_sk_proj_key_masked():
    out = redact(f"key sk-proj-{_REAL_KEY_BODY} here")
    assert _REDACTED in out
    assert "sk-proj-" + _REAL_KEY_BODY not in out


def test_bearer_header_masked():
    out = redact("Authorization: Bearer abcDEF123456token")
    assert _REDACTED in out
    assert "abcDEF123456token" not in out


def test_ghp_token_masked():
    out = redact(f"token ghp_{_REAL_KEY_BODY} committed")
    assert _REDACTED in out
    assert f"ghp_{_REAL_KEY_BODY}" not in out


def test_non_secret_text_untouched():
    text = "the agent ran wc -l file.txt and printed 42 lines successfully"
    assert redact(text) == text


def test_redact_empty_input():
    assert redact("") == ""
    assert redact(None) == ""


def test_redact_is_idempotent():
    text = f"OPENAI_API_KEY=sk-{_REAL_KEY_BODY}"
    once = redact(text)
    twice = redact(once)
    assert once == twice
