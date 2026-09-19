"""Backend-request accounting tests. Zero network, zero cost.

The money fuse must count HTTP requests about to be sent -- not semantic calls.
`complete_with_breakdown` recurses for the empty-content escalation under
`_suppress_io_log=True`, which made 983 billed requests invisible in the
2026-09-17 pilot. These cases pin the counting semantics.

    A  normal                 logical=1  backend=1
    B  empty -> success       logical=1  backend=2
    C  empty -> empty         logical=1  backend=2
    D  cap=1 + empty          backend attempted 1; the escalation is refused
                              BEFORE the send, so wire requests stay 1
"""

import asyncio
import os
import sys
import tempfile
from types import SimpleNamespace

sys.path.insert(0, ".")
os.environ.setdefault("LLM_BACKEND", "mock")


def _make_client(script, ledger, cap):
    from meta_n.core.llm_client import LLMClient, LLMConfig

    os.environ["META_N_REQUEST_LEDGER"] = ledger
    os.environ["META_N_MAX_BACKEND_REQUESTS"] = str(cap)

    c = LLMClient(LLMConfig(base_url="http://stub.invalid/v1", api_key="k",
                            model="deepseek-flash", daily_budget_usd=0,
                            max_retries=0, retry_base_delay=0.01,
                            empty_content_retry_max_tokens=32768))
    c._rpbe_backend = None              # force the real send path

    state = {"wire": 0, "script": list(script)}

    def create(**kw):
        kind = state["script"].pop(0) if state["script"] else "ok"

        async def _go():
            state["wire"] += 1          # only counted when actually sent
            if kind == "error":
                from openai import APIConnectionError
                raise APIConnectionError(request=None)
            content = "" if kind == "empty" else "OK"
            finish = "length" if kind == "empty" else "stop"
            usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5,
                                    total_tokens=15, prompt_tokens_details=None)
            return SimpleNamespace(usage=usage, choices=[SimpleNamespace(
                finish_reason=finish,
                message=SimpleNamespace(content=content))])
        return _go()

    c._client = SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=create)))
    return c, state


def case(name, script, cap, expect_wire, expect_raises=False):
    with tempfile.TemporaryDirectory() as td:
        ledger = os.path.join(td, "req.jsonl")
        c, st = _make_client(script, ledger, cap)
        raised = None
        try:
            text, *_ = asyncio.run(c.complete_with_breakdown(
                [{"role": "user", "content": "x"}], max_tokens=64))
        except Exception as e:                                  # noqa: BLE001
            raised = type(e).__name__
        from meta_n.rpbe.accounting import snapshot
        snap = snapshot()
        ok = (st["wire"] == expect_wire
              and bool(raised) == expect_raises)
        print("%s %-26s wire_requests=%d (expect %d)  backend_logged=%d  "
              "raised=%s" % ("OK " if ok else "** ", name, st["wire"],
                             expect_wire, snap["backend_requests"],
                             raised or "-"))
        return ok


def main():
    print("BACKEND-REQUEST ACCOUNTING -- LLMClient, stubbed transport")
    print("=" * 78)
    results = [
        case("A normal", ["ok"], 0, 1),
        case("B empty -> success", ["empty", "ok"], 0, 2),
        case("C empty -> empty", ["empty", "empty"], 0, 2),
        # D is the point of the whole exercise: the cap must stop the
        # escalation BEFORE it hits the wire, so real spend stays at 1.
        case("D cap=1 + empty -> refused", ["empty", "ok"], 1, 1,
             expect_raises=True),
    ]
    print("-" * 78)
    with tempfile.TemporaryDirectory() as td:
        ledger = os.path.join(td, "req.jsonl")
        os.environ["META_N_REQUEST_LEDGER"] = ledger
        os.environ["META_N_MAX_BACKEND_REQUESTS"] = "0"
        _make_client(["empty", "ok", "ok"], ledger, 0)
        from meta_n.rpbe.accounting import format_accounting
        c, _ = _make_client(["empty", "ok", "ok"], ledger, 0)
        asyncio.run(c.complete_with_breakdown(
            [{"role": "user", "content": "x"}], max_tokens=64))
        print(format_accounting())
    print()
    print("VERDICT:", "ALL OK" if all(results) else "MISMATCH")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
