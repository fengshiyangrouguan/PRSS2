"""Tests for LLM client."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from openai import AuthenticationError, NotFoundError

from meta_n.core.llm_client import LLMClient, LLMConfig


class TestLLMConfig:
    def test_defaults(self):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            config = LLMConfig()
        assert config.base_url == "https://openrouter.ai/api/v1"
        assert config.api_key == "test-key"
        assert config.model == "anthropic/claude-sonnet-4-20250514"
        assert config.temperature == 0.7
        assert config.max_tokens == 16384
        # BLOCKER 2: empty-content escalation cap default (one free escalation
        # above the 16384 default for reasoning/QAT models that burn the budget).
        assert config.empty_content_retry_max_tokens == 32768

    def test_custom_config(self):
        config = LLMConfig(
            base_url="http://localhost:1234/v1",
            api_key="local-key",
            model="qwen3.6",
            temperature=0.0,
        )
        assert config.base_url == "http://localhost:1234/v1"
        assert config.model == "qwen3.6"

    def test_env_fallback(self):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "from-env"}):
            config = LLMConfig()
        assert config.api_key == "from-env"

    def test_no_env_key(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OPENROUTER_API_KEY", None)
            config = LLMConfig()
        assert config.api_key == ""


class TestReservationAndTimeoutSentinel:
    """Item #3: None-sentinel for cost_reservation_usd / request_timeout so an
    EXPLICIT constructor value (including an explicit 0.0 / 360.0) is never
    clobbered by META_N_COST_RESERVATION_USD / META_N_LLM_REQUEST_TIMEOUT,
    while staying byte-identical for the live driver env + the unset case.
    """

    # (A) LIVE-ENV NEUTRALITY — the CO-Bench builtin improvement driver exports
    # exactly META_N_COST_RESERVATION_USD=0 and META_N_LLM_REQUEST_TIMEOUT=1200
    # (scripts/experiments/metan_cobench_builtin_improvement.py). The resumed
    # control invocation MUST resolve these identically to before the fix.
    def test_live_driver_env_resolves_identically(self):
        with patch.dict(
            os.environ,
            {
                "META_N_COST_RESERVATION_USD": "0",
                "META_N_LLM_REQUEST_TIMEOUT": "1200",
                "OPENROUTER_API_KEY": "k",
            },
            clear=True,
        ):
            config = LLMConfig()
        assert config.cost_reservation_usd == 0.0
        assert config.request_timeout == 1200.0

    # (B) UNSET NEUTRALITY — no env vars: the documented defaults still apply.
    def test_unset_env_coalesces_to_defaults(self):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "k"}, clear=True):
            config = LLMConfig()
        assert config.cost_reservation_usd == 0.0
        assert config.request_timeout == 360.0
        # And both are always plain floats after __post_init__ (never None).
        assert isinstance(config.cost_reservation_usd, float)
        assert isinstance(config.request_timeout, float)

    # (C) BUG-FIX REGRESSION — an EXPLICIT 0.0 / 360.0 is no longer clobbered by
    # a non-zero env var (the never-clobber invariant the old ``== 0.0`` /
    # ``== 360.0`` gates violated).
    def test_explicit_zero_reservation_not_clobbered_by_env(self):
        with patch.dict(
            os.environ,
            {"META_N_COST_RESERVATION_USD": "0.25", "OPENROUTER_API_KEY": "k"},
            clear=True,
        ):
            config = LLMConfig(cost_reservation_usd=0.0)
        assert config.cost_reservation_usd == 0.0

    def test_explicit_default_timeout_not_clobbered_by_env(self):
        with patch.dict(
            os.environ,
            {"META_N_LLM_REQUEST_TIMEOUT": "900", "OPENROUTER_API_KEY": "k"},
            clear=True,
        ):
            config = LLMConfig(request_timeout=360.0)
        assert config.request_timeout == 360.0

    # (P0) Passing request_timeout=None explicitly (the main.py --request-timeout
    # default) is byte-identical to omitting it: both hit the env-fallback then
    # the 360.0 coalesce. This is the wiring guarantee for the new CLI flag.
    def test_explicit_none_timeout_identical_to_omitted(self):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "k"}, clear=True):
            omitted = LLMConfig()
            explicit_none = LLMConfig(request_timeout=None)
        assert omitted.request_timeout == explicit_none.request_timeout == 360.0

    def test_none_timeout_honors_env_fallback(self):
        with patch.dict(
            os.environ,
            {"META_N_LLM_REQUEST_TIMEOUT": "1200", "OPENROUTER_API_KEY": "k"},
            clear=True,
        ):
            config = LLMConfig(request_timeout=None)
        assert config.request_timeout == 1200.0

    # (D) ENV-OVERRIDES-DEFAULT still works when NO explicit arg is given.
    def test_env_overrides_default_reservation_when_unset(self):
        with patch.dict(
            os.environ,
            {"META_N_COST_RESERVATION_USD": "0.5", "OPENROUTER_API_KEY": "k"},
            clear=True,
        ):
            config = LLMConfig()
        assert config.cost_reservation_usd == 0.5

    def test_env_overrides_default_timeout_when_unset(self):
        with patch.dict(
            os.environ,
            {"META_N_LLM_REQUEST_TIMEOUT": "900", "OPENROUTER_API_KEY": "k"},
            clear=True,
        ):
            config = LLMConfig()
        assert config.request_timeout == 900.0

    # An explicit NON-default constructor value also survives (the original
    # intent of the gate) — explicit beats env in both directions.
    def test_explicit_nondefault_reservation_survives_env(self):
        with patch.dict(
            os.environ,
            {"META_N_COST_RESERVATION_USD": "0.5", "OPENROUTER_API_KEY": "k"},
            clear=True,
        ):
            config = LLMConfig(cost_reservation_usd=0.1)
        assert config.cost_reservation_usd == 0.1

    # The coalesced float is what LLMClient passes down: a cost tracker built
    # from an explicit-0.0 reservation under a nonzero env keeps 0.0.
    def test_cost_tracker_receives_explicit_zero_reservation(self):
        with patch.dict(
            os.environ,
            {"META_N_COST_RESERVATION_USD": "0.25", "OPENROUTER_API_KEY": "k"},
            clear=True,
        ):
            config = LLMConfig(
                cost_reservation_usd=0.0, daily_budget_usd=10.0, model="gpt-5.2"
            )
            client = LLMClient(config)
        assert client.cost_tracker is not None
        assert client.cost_tracker.reservation_usd == 0.0


class TestLLMClient:
    @pytest.fixture
    def mock_client(self):
        config = LLMConfig(api_key="test-key")
        client = LLMClient(config)
        return client

    @pytest.mark.asyncio
    async def test_complete(self, mock_client):
        # Mock the OpenAI client's response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello, world!"
        mock_response.usage = MagicMock()
        mock_response.usage.total_tokens = 25

        mock_client._client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        text, tokens = await mock_client.complete(
            messages=[{"role": "user", "content": "Say hello"}]
        )

        assert text == "Hello, world!"
        assert tokens == 25

    @pytest.mark.asyncio
    async def test_complete_no_usage(self, mock_client):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "response"
        mock_response.usage = None

        mock_client._client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        text, tokens = await mock_client.complete(
            messages=[{"role": "user", "content": "test"}]
        )

        assert text == "response"
        assert tokens == 0

    @pytest.mark.asyncio
    async def test_complete_empty_content(self, mock_client):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = None
        mock_response.usage = MagicMock()
        mock_response.usage.total_tokens = 10

        mock_client._client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        text, tokens = await mock_client.complete(
            messages=[{"role": "user", "content": "test"}]
        )

        assert text == ""
        assert tokens == 10

    @pytest.mark.asyncio
    async def test_custom_temperature_and_tokens(self, mock_client):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "ok"
        mock_response.usage = MagicMock()
        mock_response.usage.total_tokens = 5

        mock_create = AsyncMock(return_value=mock_response)
        mock_client._client.chat.completions.create = mock_create

        await mock_client.complete(
            messages=[{"role": "user", "content": "test"}],
            temperature=0.0,
            max_tokens=100,
        )

        call_kwargs = mock_create.call_args[1]
        assert call_kwargs["temperature"] == 0.0
        assert call_kwargs["max_tokens"] == 100


class TestEmptyContentEscalation:
    """BLOCKER 2 — empty-content escalation on the builtin TB spine path.

    A reasoning / QAT model (gemma-4-31b-qat) can spend its WHOLE ``max_tokens``
    budget on hidden reasoning and return ``finish_reason=='length'`` with EMPTY
    ``message.content``. Today that authored an empty bash script -> a spurious
    recipient failure in the headroom screen. The fix re-issues the SAME request
    ONCE at ``empty_content_retry_max_tokens`` so the model has room to finish
    reasoning AND emit content.
    """

    @staticmethod
    def _resp(content, *, finish_reason, total_tokens=16384) -> MagicMock:
        r = MagicMock()
        r.choices = [MagicMock()]
        r.choices[0].message.content = content
        r.choices[0].finish_reason = finish_reason
        r.usage = MagicMock()
        r.usage.prompt_tokens = 100
        r.usage.completion_tokens = total_tokens
        r.usage.total_tokens = total_tokens + 100
        r.usage.prompt_tokens_details = None
        return r

    @pytest.mark.asyncio
    async def test_empty_length_escalates_once_at_higher_cap(self):
        """First call: empty + finish_reason='length' at cap 16384. Expect ONE
        re-issue at the escalation cap (32768) whose non-empty content is returned."""
        config = LLMConfig(api_key="test-key", max_tokens=16384,
                           empty_content_retry_max_tokens=32768)
        client = LLMClient(config)
        empty = self._resp("", finish_reason="length", total_tokens=16384)
        recovered = self._resp("#!/bin/bash\necho ok", finish_reason="stop",
                               total_tokens=2000)
        mock_create = AsyncMock(side_effect=[empty, recovered])
        client._client.chat.completions.create = mock_create

        text, tokens = await client.complete(
            messages=[{"role": "user", "content": "author a script"}]
        )

        # Two create() calls: the empty one, then the escalated re-issue.
        assert mock_create.call_count == 2
        assert mock_create.call_args_list[0][1]["max_tokens"] == 16384
        assert mock_create.call_args_list[1][1]["max_tokens"] == 32768
        # The recovered (non-empty) content is what the caller receives.
        assert text == "#!/bin/bash\necho ok"

    @pytest.mark.asyncio
    async def test_no_escalation_when_disabled(self):
        """empty_content_retry_max_tokens=0 disables the escalation (byte-identical
        legacy behaviour: a single call returning '')."""
        config = LLMConfig(api_key="test-key", max_tokens=16384,
                           empty_content_retry_max_tokens=0)
        client = LLMClient(config)
        empty = self._resp("", finish_reason="length", total_tokens=16384)
        mock_create = AsyncMock(return_value=empty)
        client._client.chat.completions.create = mock_create

        text, _ = await client.complete(messages=[{"role": "user", "content": "x"}])
        assert mock_create.call_count == 1
        assert text == ""

    @pytest.mark.asyncio
    async def test_no_escalation_when_content_nonempty(self):
        """A non-empty completion (even if finish_reason=='length') keeps its
        partial content — escalation is scoped to the EMPTY+length case only."""
        config = LLMConfig(api_key="test-key", max_tokens=16384,
                           empty_content_retry_max_tokens=32768)
        client = LLMClient(config)
        truncated = self._resp("partial output", finish_reason="length")
        mock_create = AsyncMock(return_value=truncated)
        client._client.chat.completions.create = mock_create

        text, _ = await client.complete(messages=[{"role": "user", "content": "x"}])
        assert mock_create.call_count == 1
        assert text == "partial output"

    @pytest.mark.asyncio
    async def test_escalation_bounded_to_single_retry(self):
        """If the escalated re-issue is ALSO empty+length, the call does NOT loop
        forever — it returns '' after exactly one escalation (2 create() calls)."""
        config = LLMConfig(api_key="test-key", max_tokens=16384,
                           empty_content_retry_max_tokens=32768)
        client = LLMClient(config)
        empty1 = self._resp("", finish_reason="length", total_tokens=16384)
        empty2 = self._resp("", finish_reason="length", total_tokens=32768)
        mock_create = AsyncMock(side_effect=[empty1, empty2])
        client._client.chat.completions.create = mock_create

        text, _ = await client.complete(messages=[{"role": "user", "content": "x"}])
        assert mock_create.call_count == 2
        assert text == ""

    @pytest.mark.asyncio
    async def test_escalation_breakdown_sums_both_calls(self):
        """L2.2 (audit) — the breakdown tuple returned on the empty-content
        escalation must SUM the wasted first (empty) call AND the recovered
        re-issue, so ``LLMUsageTracker.record`` (inner_tokens in summary.json)
        does not undercount the escalation. The first empty call burned
        completion=16384 (tt=16484); the recovered call completion=2000
        (tt=2100); the returned breakdown must report the totals."""
        config = LLMConfig(api_key="test-key", max_tokens=16384,
                           empty_content_retry_max_tokens=32768)
        client = LLMClient(config)
        empty = self._resp("", finish_reason="length", total_tokens=16384)
        recovered = self._resp("#!/bin/bash\necho ok", finish_reason="stop",
                               total_tokens=2000)
        mock_create = AsyncMock(side_effect=[empty, recovered])
        client._client.chat.completions.create = mock_create

        text, pt, ct, tt = await client.complete_with_breakdown(
            messages=[{"role": "user", "content": "author a script"}]
        )
        assert text == "#!/bin/bash\necho ok"
        # prompt 100+100, completion 16384+2000, total 16484+2100.
        assert (pt, ct, tt) == (200, 18384, 18584)

    @pytest.mark.asyncio
    async def test_no_escalation_breakdown_is_single_call(self):
        """Unaffected path unchanged: with no escalation (non-empty first
        response) the breakdown tuple reports ONLY that one call's tokens —
        the summing fix never double-counts the default path."""
        config = LLMConfig(api_key="test-key", max_tokens=16384,
                           empty_content_retry_max_tokens=32768)
        client = LLMClient(config)
        ok = self._resp("#!/bin/bash\necho ok", finish_reason="stop",
                        total_tokens=2000)
        mock_create = AsyncMock(return_value=ok)
        client._client.chat.completions.create = mock_create

        text, pt, ct, tt = await client.complete_with_breakdown(
            messages=[{"role": "user", "content": "x"}]
        )
        assert mock_create.call_count == 1
        assert text == "#!/bin/bash\necho ok"
        assert (pt, ct, tt) == (100, 2000, 2100)


class TestLLMClientRetry:
    """Tests for retry behavior on transient errors."""

    def _make_client(self, max_retries: int = 2) -> LLMClient:
        config = LLMConfig(api_key="test-key", max_retries=max_retries, retry_base_delay=0.01)
        return LLMClient(config)

    def _make_success_response(self) -> MagicMock:
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = "ok"
        resp.usage = MagicMock()
        resp.usage.total_tokens = 10
        return resp

    def _make_not_found_error(self) -> NotFoundError:
        return NotFoundError(
            message="model not available",
            response=httpx.Response(404, request=httpx.Request("POST", "https://test")),
            body={"error": {"message": "model not available"}},
        )

    @pytest.mark.asyncio
    async def test_retries_on_not_found(self):
        client = self._make_client(max_retries=2)
        error = self._make_not_found_error()
        success = self._make_success_response()

        # Fail twice, then succeed
        client._client.chat.completions.create = AsyncMock(
            side_effect=[error, error, success]
        )

        text, tokens = await client.complete([{"role": "user", "content": "hi"}])
        assert text == "ok"
        assert tokens == 10
        assert client._client.chat.completions.create.call_count == 3

    @pytest.mark.asyncio
    async def test_raises_after_max_retries(self):
        client = self._make_client(max_retries=2)
        error = self._make_not_found_error()

        # Fail every time
        client._client.chat.completions.create = AsyncMock(side_effect=error)

        with pytest.raises(NotFoundError):
            await client.complete([{"role": "user", "content": "hi"}])
        # 1 initial + 2 retries = 3 attempts
        assert client._client.chat.completions.create.call_count == 3

    @pytest.mark.asyncio
    async def test_no_retry_on_auth_error(self):
        client = self._make_client(max_retries=2)
        error = AuthenticationError(
            message="invalid key",
            response=httpx.Response(401, request=httpx.Request("POST", "https://test")),
            body={"error": {"message": "invalid key"}},
        )

        client._client.chat.completions.create = AsyncMock(side_effect=error)

        with pytest.raises(AuthenticationError):
            await client.complete([{"role": "user", "content": "hi"}])
        # Auth errors are not retried — only 1 attempt
        assert client._client.chat.completions.create.call_count == 1

    @pytest.mark.asyncio
    async def test_succeeds_on_first_try(self):
        client = self._make_client(max_retries=2)
        success = self._make_success_response()
        client._client.chat.completions.create = AsyncMock(return_value=success)

        text, tokens = await client.complete([{"role": "user", "content": "hi"}])
        assert text == "ok"
        assert client._client.chat.completions.create.call_count == 1
