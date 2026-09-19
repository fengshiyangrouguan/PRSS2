"""Tests for the llm() helper injection into solve() namespaces."""

import asyncio
import threading
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from meta_n.core.llm_helpers import (
    LLMUsageTracker,
    make_llm_func,
    make_llm_func_from_config,
)


def _mock_llm_client(
    response: str = "hello",
    tokens: int = 50,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
):
    """Create a mock LLMClient that supports both complete contracts.

    ``llm_helpers`` calls ``complete_with_breakdown`` which returns a 4-tuple
    ``(text, prompt, completion, total)``. Tests that need to assert call
    args read ``client.complete_with_breakdown.call_args``. We also mock
    ``complete`` for any callers still on the legacy 2-tuple path.
    """
    if prompt_tokens is None:
        # Default split: 70% prompt, 30% completion (rounded). Arbitrary —
        # tests that need exact values pass them explicitly.
        prompt_tokens = (tokens * 7) // 10
    if completion_tokens is None:
        completion_tokens = tokens - prompt_tokens
    client = MagicMock()
    client.complete = AsyncMock(return_value=(response, tokens))
    client.complete_with_breakdown = AsyncMock(
        return_value=(response, prompt_tokens, completion_tokens, tokens)
    )
    return client


class TestLLMUsageTracker:
    def test_initial_state(self):
        tracker = LLMUsageTracker()
        assert tracker.total_tokens == 0
        assert tracker.call_count == 0

    def test_record(self):
        tracker = LLMUsageTracker()
        tracker.record(100)
        assert tracker.total_tokens == 100
        assert tracker.call_count == 1

    def test_accumulates(self):
        tracker = LLMUsageTracker()
        tracker.record(100)
        tracker.record(200)
        tracker.record(50)
        assert tracker.total_tokens == 350
        assert tracker.call_count == 3

    def test_thread_safety(self):
        tracker = LLMUsageTracker()
        errors = []

        def record_many():
            try:
                for _ in range(100):
                    tracker.record(1)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record_many) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert tracker.total_tokens == 1000
        assert tracker.call_count == 1000


class TestMakeLlmFunc:
    def test_returns_text(self):
        client = _mock_llm_client("Allergy", 30)
        tracker = LLMUsageTracker()
        llm, _ = make_llm_func(client, tracker)

        # Run in a thread (simulates asyncio.to_thread)
        result = asyncio.run(asyncio.to_thread(llm, "classify this"))

        assert result == "Allergy"
        assert tracker.total_tokens == 30
        assert tracker.call_count == 1

    def test_default_params(self):
        client = _mock_llm_client()
        llm, _ = make_llm_func(client)

        asyncio.run(asyncio.to_thread(llm, "test prompt"))

        call_args = client.complete_with_breakdown.call_args
        assert call_args[1]["temperature"] == 0.3
        assert call_args[1]["max_tokens"] == 512

    def test_custom_params(self):
        client = _mock_llm_client()
        llm, _ = make_llm_func(client)

        asyncio.run(asyncio.to_thread(llm, "test", temperature=0.8, max_tokens=1024))

        call_args = client.complete_with_breakdown.call_args
        assert call_args[1]["temperature"] == 0.8
        assert call_args[1]["max_tokens"] == 1024

    def test_exception_propagates(self):
        client = MagicMock()
        client.complete_with_breakdown = AsyncMock(side_effect=RuntimeError("API error"))
        llm, _ = make_llm_func(client)

        with pytest.raises(RuntimeError, match="API error"):
            asyncio.run(asyncio.to_thread(llm, "test"))

    def test_no_tracker(self):
        client = _mock_llm_client("ok", 10)
        llm, _ = make_llm_func(client, tracker=None)

        result = asyncio.run(asyncio.to_thread(llm, "test"))
        assert result == "ok"  # works without tracker

    def test_multiple_calls_accumulate(self):
        client = _mock_llm_client("resp", 25)
        tracker = LLMUsageTracker()
        llm, _ = make_llm_func(client, tracker)

        def run_calls():
            llm("prompt 1")
            llm("prompt 2")
            llm("prompt 3")

        asyncio.run(asyncio.to_thread(run_calls))

        assert tracker.call_count == 3
        assert tracker.total_tokens == 75


class TestMakeLlmFuncFromConfig:
    def test_lazy_init(self):
        """LLMClient should NOT be created until first call."""
        config = {"base_url": "http://test", "api_key": "key", "model": "m",
                  "temperature": 0.7, "max_tokens": 1024, "max_retries": 1}

        with patch("meta_n.core.llm_client.LLMClient") as mock_cls:
            llm, llm_batch = make_llm_func_from_config(config)
            assert callable(llm)
            assert callable(llm_batch)
            # LLMClient constructor should NOT have been called yet
            mock_cls.assert_not_called()

    def test_creates_client_on_first_call(self):
        config = {"base_url": "http://test", "api_key": "key", "model": "m",
                  "temperature": 0.7, "max_tokens": 1024, "max_retries": 1}

        with patch("meta_n.core.llm_client.AsyncOpenAI"):
            # llm_helpers.make_llm_func_from_config calls
            # LLMClient.complete_with_breakdown — patch the new contract.
            with patch("meta_n.core.llm_client.LLMClient.complete_with_breakdown",
                       new_callable=AsyncMock, return_value=("result", 14, 6, 20)):
                tracker = LLMUsageTracker()
                llm, _ = make_llm_func_from_config(config, tracker)
                result = llm("test prompt")
                assert result == "result"
                assert tracker.total_tokens == 20
                assert tracker.prompt_tokens == 14
                assert tracker.completion_tokens == 6


class TestLlmBatch:
    def test_batch_returns_ordered_results(self):
        call_count = 0

        async def mock_breakdown(messages, temperature=None, max_tokens=None, **kwargs):
            nonlocal call_count
            call_count += 1
            prompt = messages[0]["content"]
            # 7 prompt + 3 completion = 10 total
            return f"response_to_{prompt}", 7, 3, 10

        client = MagicMock()
        client.complete_with_breakdown = mock_breakdown
        tracker = LLMUsageTracker()
        _, llm_batch = make_llm_func(client, tracker)

        def run():
            return llm_batch(["p1", "p2", "p3"])

        results = asyncio.run(asyncio.to_thread(run))

        assert results == ["response_to_p1", "response_to_p2", "response_to_p3"]
        assert tracker.call_count == 3
        assert tracker.total_tokens == 30
        assert tracker.prompt_tokens == 21
        assert tracker.completion_tokens == 9

    def test_batch_empty_list(self):
        client = _mock_llm_client()
        _, llm_batch = make_llm_func(client)

        def run():
            return llm_batch([])

        results = asyncio.run(asyncio.to_thread(run))
        assert results == []

    def test_batch_single_item(self):
        # 11 prompt + 4 completion = 15 total
        client = _mock_llm_client("answer", 15, prompt_tokens=11, completion_tokens=4)
        tracker = LLMUsageTracker()
        _, llm_batch = make_llm_func(client, tracker)

        def run():
            return llm_batch(["one prompt"])

        results = asyncio.run(asyncio.to_thread(run))
        assert results == ["answer"]
        assert tracker.call_count == 1
        assert tracker.total_tokens == 15
        assert tracker.prompt_tokens == 11
        assert tracker.completion_tokens == 4

    def test_batch_respects_max_concurrent(self):
        """Verify semaphore limits concurrency."""
        max_active = 0
        current_active = 0
        lock = threading.Lock()

        async def mock_breakdown(messages, temperature=None, max_tokens=None, **kwargs):
            nonlocal max_active, current_active
            with lock:
                current_active += 1
                max_active = max(max_active, current_active)
            await asyncio.sleep(0.05)
            with lock:
                current_active -= 1
            return "resp", 4, 1, 5

        client = MagicMock()
        client.complete_with_breakdown = mock_breakdown
        _, llm_batch = make_llm_func(client)

        def run():
            return llm_batch(["p"] * 20, max_concurrent=3)

        asyncio.run(asyncio.to_thread(run))
        assert max_active <= 3, f"Expected max 3 concurrent, got {max_active}"

    def test_batch_exception_propagates(self):
        async def mock_breakdown(messages, temperature=None, max_tokens=None, **kwargs):
            prompt = messages[0]["content"]
            if "bad" in prompt:
                raise RuntimeError("API error")
            return "ok", 7, 3, 10

        client = MagicMock()
        client.complete_with_breakdown = mock_breakdown
        _, llm_batch = make_llm_func(client)

        def run():
            return llm_batch(["good", "bad", "good"])

        with pytest.raises(RuntimeError, match="API error"):
            asyncio.run(asyncio.to_thread(run))
