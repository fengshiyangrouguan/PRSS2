"""§6b F073 — per-factory persistent event loop for the inner llm()/llm_batch().

httpx pooled connections are loop-affine: the old per-call ``asyncio.run``
closed its loop after every call, so a keep-alive connection reused by the
NEXT call faulted (``RuntimeError: Event loop is closed``) — masked by the
retry loop at ~2s + one wasted request per call on the production subprocess
path. ``_LoopRunner`` gives each factory ONE background daemon-loop thread;
these tests pin loop identity (same loop across calls, per-factory isolation),
the keep-alive regression with ``max_retries=0``, thread-safety of the sync
closures, and BaseException (BudgetExceededError) propagation through the
cross-thread future.
"""

import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from meta_n.core.llm_helpers import (
    LLMUsageTracker,
    make_llm_func,
    make_llm_func_from_config,
)
from meta_n.utils.cost_tracker import BudgetExceededError


class _StubClient:
    """LLMClient stand-in that records the running loop of every call."""

    def __init__(self, text: str = "ok"):
        self.text = text
        self.loops: list[asyncio.AbstractEventLoop] = []
        self.config = SimpleNamespace(model="stub-model")

    async def complete_with_breakdown(
        self, messages, temperature=None, max_tokens=None, **kwargs,
    ):
        self.loops.append(asyncio.get_running_loop())
        return self.text, 10, 5, 15


class TestLoopIdentity:
    def test_sequential_llm_calls_share_one_loop(self):
        client = _StubClient()
        llm, _ = make_llm_func(client)

        assert llm("a") == "ok"
        assert llm("b") == "ok"

        assert len(client.loops) == 2
        assert client.loops[0] is client.loops[1]

    def test_llm_and_llm_batch_share_the_factory_loop(self):
        client = _StubClient()
        llm, llm_batch = make_llm_func(client)

        llm("a")
        assert llm_batch(["b", "c"]) == ["ok", "ok"]

        assert len(client.loops) == 3
        assert all(loop is client.loops[0] for loop in client.loops)

    def test_two_factories_do_not_share_loops(self):
        client_a, client_b = _StubClient(), _StubClient()
        llm_a, _ = make_llm_func(client_a)
        llm_b, _ = make_llm_func(client_b)

        llm_a("a")
        llm_b("b")

        assert client_a.loops[0] is not client_b.loops[0]


class _KeepAliveHandler(BaseHTTPRequestHandler):
    """Minimal OpenAI chat-completions endpoint over HTTP/1.1 keep-alive."""

    protocol_version = "HTTP/1.1"  # keep-alive on — the F073 trigger

    _BODY = json.dumps({
        "id": "chatcmpl-x", "object": "chat.completion", "created": 1,
        "model": "test-model",
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": "ok"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2,
                  "total_tokens": 12},
    }).encode()

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        self.rfile.read(n)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(self._BODY)))
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        self.wfile.write(self._BODY)

    def log_message(self, *args):  # keep pytest output clean
        pass


def test_cross_loop_keepalive_regression():
    """The repro-turned-test: with max_retries=0 (no masking), one cached
    LLMClient must survive sequential llm() calls plus an llm_batch() over a
    keep-alive connection. On the per-call asyncio.run implementation every
    other call raised APIConnectionError (Event loop is closed)."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), _KeepAliveHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        llm, llm_batch = make_llm_func_from_config({
            "base_url": f"http://127.0.0.1:{port}/v1",
            "api_key": "test-key",
            "model": "test-model",
            "max_retries": 0,
            "retry_base_delay": 0.01,
            "request_timeout": 10.0,
        })
        for i in range(4):
            assert llm(f"prompt {i}", max_tokens=64) == "ok"
        assert llm_batch(["a", "b", "c"], max_tokens=64) == ["ok", "ok", "ok"]
    finally:
        server.shutdown()
        server.server_close()


def test_llm_callable_from_multiple_threads():
    client = _StubClient()
    tracker = LLMUsageTracker()
    llm, _ = make_llm_func(client, tracker=tracker)

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda i: llm(f"p{i}"), range(8)))

    assert results == ["ok"] * 8
    assert tracker.call_count == 8
    # All 8 cross-thread submissions landed on the single factory loop.
    assert all(loop is client.loops[0] for loop in client.loops)


def test_budget_exceeded_propagates_through_runner():
    """BudgetExceededError is a BaseException subclass; Future.result() must
    re-raise it out of the sync closure (the orchestrator's checkpoint-save
    path depends on it bypassing `except Exception`)."""

    class _BudgetStub(_StubClient):
        async def complete_with_breakdown(self, *args, **kwargs):
            raise BudgetExceededError("cap hit")

    llm, _ = make_llm_func(_BudgetStub())
    with pytest.raises(BudgetExceededError, match="cap hit"):
        llm("x")
