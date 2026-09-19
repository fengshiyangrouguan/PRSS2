"""Regression tests for audit fixes in ``meta_n/core/llm_client.py``.

Offline / LLM-free: these exercise pure helper functions and module-level
import surface only — no network, no LM Studio, no Docker.

Covered findings:
  * 6  — reasoning-model detection must see through a provider-prefixed id
         (``openai/gpt-5.2``, ``openai/o3-mini``).
  * 32 — ``BudgetExceededError`` must no longer be imported (dead import) into
         the ``llm_client`` module namespace.

  * 51 — empty-content escalation recursion must run OUTSIDE the parent retry
         ``try``, so a transient escalation error cannot re-arm the parent loop
         and re-issue (re-bill) the already-successful base request.
  * 65 — the escalation path must write an io_logger record for the FIRST
         (empty, token-burning) round-trip before re-issuing, so that wasted
         call is not silently dropped from the I/O audit JSONL.
"""

import asyncio
from json import JSONDecodeError
from types import SimpleNamespace

import pytest

import meta_n.core.llm_client as llm_client
from meta_n.core.llm_client import (
    LLMClient,
    LLMConfig,
    _supports_custom_temperature,
    _uses_max_completion_tokens,
)


class TestFinding6ProviderPrefixedReasoningDetection:
    """Provider-prefixed (OpenRouter / LiteLLM) ids must be normalized to the
    bare model id before family detection. On the un-fixed code these all
    return the wrong answer because the leading ``"<provider>/"`` segment
    defeats every ``startswith`` test."""

    def test_openrouter_gpt5_uses_max_completion_tokens(self):
        # Pre-fix: returns False (string starts with "openai/", not "gpt-5.").
        assert _uses_max_completion_tokens("openai/gpt-5.2") is True

    def test_openrouter_gpt5_rejects_custom_temperature(self):
        # Pre-fix: returns True -> a custom temperature is sent -> HTTP 400.
        assert _supports_custom_temperature("openai/gpt-5.2") is False

    def test_openrouter_o3_series_uses_max_completion_tokens(self):
        assert _uses_max_completion_tokens("openai/o3-mini") is True

    def test_bare_azure_deployment_name_unchanged(self):
        # No "/" segment -> normalization is a no-op; behavior preserved.
        assert _uses_max_completion_tokens("gpt-5.2") is True
        assert _uses_max_completion_tokens("o3-mini") is True

    def test_non_reasoning_provider_prefixed_still_false(self):
        # Normalization must NOT flip ordinary chat models into reasoning ones.
        assert _uses_max_completion_tokens("openai/gpt-4o") is False
        assert _uses_max_completion_tokens("anthropic/claude-sonnet-4") is False
        assert _supports_custom_temperature("openai/gpt-4o") is True

    def test_chat_variant_exclusion_still_applies_when_prefixed(self):
        # gpt-5.x-chat-* are non-reasoning; the "-chat" exclusion must survive
        # the provider-prefix normalization.
        assert _uses_max_completion_tokens("openai/gpt-5-chat-latest") is False


class TestFinding32DeadImportRemoved:
    """``BudgetExceededError`` was imported but never used in llm_client.py
    (it is raised inside CostTracker and imported directly from its defining
    module everywhere else). The dead import must be gone from the module
    namespace, while ``CostTracker`` (the actually-used name) stays."""

    def test_budget_exceeded_error_not_in_module_namespace(self):
        # Pre-fix: the `from ... import BudgetExceededError, CostTracker` line
        # binds this name on the module -> hasattr is True -> test fails.
        assert not hasattr(llm_client, "BudgetExceededError")

    def test_cost_tracker_still_imported(self):
        assert hasattr(llm_client, "CostTracker")


# --------------------------------------------------------------------------- #
# Findings 51 / 65 — empty-content escalation (offline; the SDK ``_client`` is
# fully replaced with a scripted fake, so no network / LM Studio / Docker).
# --------------------------------------------------------------------------- #


def _usage(pt, ct):
    return SimpleNamespace(
        prompt_tokens=pt,
        completion_tokens=ct,
        total_tokens=pt + ct,
        prompt_tokens_details=None,
    )


def _resp(content, finish_reason, pt=5, ct=10):
    choice = SimpleNamespace(
        message=SimpleNamespace(content=content),
        finish_reason=finish_reason,
    )
    return SimpleNamespace(choices=[choice], usage=_usage(pt, ct))


class _RecordingIOLogger:
    """Minimal stand-in for LLMIOLogger — captures each ``log(**kwargs)``."""

    def __init__(self):
        self.records = []

    def log(self, **kwargs):
        self.records.append(kwargs)


def _make_client(**overrides):
    cfg_kwargs = dict(
        api_key="test-key",
        model="anthropic/claude-sonnet-4",  # non-reasoning -> uses max_tokens
        max_tokens=10,                       # base cap
        empty_content_retry_max_tokens=20,   # escalation cap (> base -> reachable)
        daily_budget_usd=0.0,                # no cost tracker / pricing lookup
        max_retries=0,
        retry_base_delay=0.0,
        retry_max_delay=0.0,
    )
    cfg_kwargs.update(overrides)
    client = LLMClient(LLMConfig(**cfg_kwargs))
    return client


def _install_create(client, create):
    """Replace the whole SDK client chain with a scripted async ``create``.

    The wrapper only ever touches ``self._client.chat.completions.create``."""
    client._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )


class TestFinding65EmptyEscalationIsAudited:
    """The first (empty, length-truncated, token-burning) round-trip must be
    written to the io_logger JSONL before the escalation re-issue. Pre-fix the
    escalation branch ``return``ed before the only io_logger.log, so the wasted
    first call silently never appeared in the audit trail."""

    def test_first_empty_call_is_logged_before_escalation(self):
        calls = []

        async def create(**kwargs):
            calls.append(kwargs)
            cap = kwargs.get("max_tokens") or kwargs.get("max_completion_tokens")
            if cap == 10:                     # base cap -> empty + length
                return _resp("", "length")
            return _resp("DONE", "stop")      # escalation cap -> real content

        client = _make_client()
        _install_create(client, create)
        rec = _RecordingIOLogger()
        client.io_logger = rec

        text, pt, ct, tt = asyncio.run(
            client.complete_with_breakdown([{"role": "user", "content": "hi"}])
        )

        # Escalation succeeded and produced the real content.
        assert text == "DONE"
        # Both round-trips' tokens folded into the returned breakdown.
        assert (pt, ct, tt) == (10, 20, 30)
        # BOTH round-trips are in the audit trail (pre-fix: only the second).
        assert len(rec.records) == 2
        first, second = rec.records
        assert first["response"] == ""
        assert first["extra"]["empty_content_escalated"] is True
        assert second["response"] == "DONE"

    def test_suppress_io_log_skips_the_empty_record(self):
        async def create(**kwargs):
            cap = kwargs.get("max_tokens") or kwargs.get("max_completion_tokens")
            if cap == 10:
                return _resp("", "length")
            return _resp("DONE", "stop")

        client = _make_client()
        _install_create(client, create)
        rec = _RecordingIOLogger()
        client.io_logger = rec

        text, *_ = asyncio.run(
            client.complete_with_breakdown(
                [{"role": "user", "content": "hi"}], _suppress_io_log=True
            )
        )
        assert text == "DONE"
        # The new empty-call log is gated on ``not _suppress_io_log`` too.
        assert rec.records == []


class TestFinding51EscalationErrorDoesNotRebillBase:
    """A transient escalation error must NOT re-arm the parent retry loop and
    re-issue the (already-successful) base request. The base request must be
    issued exactly once even when the escalation re-issue keeps failing."""

    def test_escalation_error_issues_base_request_once(self):
        calls = []

        async def create(**kwargs):
            calls.append(kwargs)
            cap = kwargs.get("max_tokens") or kwargs.get("max_completion_tokens")
            if cap == 10:                     # base cap -> empty + length
                return _resp("", "length")
            # Escalation cap -> persistent retryable (non-throttle) failure.
            raise JSONDecodeError("boom", "doc", 0)

        # max_retries=1: pre-fix the parent ``except`` would catch the escalation
        # error and re-issue the base request a SECOND time (re-billing). With
        # retry_base_delay=0 the non-throttle retry sleeps ~0s, so this is fast.
        client = _make_client(max_retries=1)
        _install_create(client, create)

        with pytest.raises(JSONDecodeError):
            asyncio.run(
                client.complete_with_breakdown([{"role": "user", "content": "hi"}])
            )

        base_caps = [
            c for c in calls
            if (c.get("max_tokens") or c.get("max_completion_tokens")) == 10
        ]
        # Exactly one base round-trip (pre-fix: 2 -> double-billed).
        assert len(base_caps) == 1
        # Only the base call returned a response, so usage was booked once;
        # the failing escalation attempts raise before the usage-booking code.
        assert client.cumulative_usage["calls"] == 1
