"""Regression tests: inner-LLM JSONL records are stamped with the writer's pid.

Every ``io_logger.log(...)`` in the shared ``_make_llm_funcs`` factory passes
``extra={"pid": os.getpid()}`` so that co_bench's
``_usage_from_log_since(pid=...)`` can exclude records written by concurrent
sibling subprocesses that share the same inner-log file. The stamp is additive
(under the already-optional ``extra`` key); all pre-existing record fields are
unchanged.
"""

from __future__ import annotations

import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

from meta_n.core.llm_helpers import (
    LLMUsageTracker,
    make_llm_func,
    make_llm_func_from_config,
)
from meta_n.integrations._subprocess_utils import _usage_from_log_since


def _mock_llm_client(response: str = "hello", pt: int = 14, ct: int = 6, tt: int = 20):
    client = MagicMock()
    client.complete_with_breakdown = AsyncMock(return_value=(response, pt, ct, tt))
    return client


def _read_records(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_llm_log_record_carries_pid(tmp_path):
    log_path = tmp_path / "inner.jsonl"
    llm, _ = make_llm_func(_mock_llm_client(), LLMUsageTracker(), log_path=log_path)

    assert llm("hi") == "hello"

    records = _read_records(log_path)
    assert len(records) == 1
    rec = records[0]
    assert rec["extra"]["pid"] == os.getpid()
    # Pre-existing record fields are unchanged by the additive stamp.
    assert rec["messages"] == [{"role": "user", "content": "hi"}]
    assert rec["response"] == "hello"
    assert rec["prompt_tokens"] == 14
    assert rec["completion_tokens"] == 6
    assert rec["total_tokens"] == 20


def test_llm_batch_log_records_carry_pid(tmp_path):
    log_path = tmp_path / "inner.jsonl"
    _, llm_batch = make_llm_func(_mock_llm_client(), None, log_path=log_path)

    assert llm_batch(["a", "b"]) == ["hello", "hello"]

    records = _read_records(log_path)
    assert len(records) == 2
    assert all(rec["extra"]["pid"] == os.getpid() for rec in records)


def test_from_config_log_record_carries_pid(tmp_path):
    """The subprocess factory shares _make_llm_funcs — same stamp applies."""
    log_path = tmp_path / "inner.jsonl"
    config = {"base_url": "http://test", "api_key": "key", "model": "m",
              "temperature": 0.7, "max_tokens": 1024, "max_retries": 1}

    with patch("meta_n.core.llm_client.AsyncOpenAI"), \
         patch("meta_n.core.llm_client.LLMClient.complete_with_breakdown",
               new_callable=AsyncMock, return_value=("result", 3, 2, 5)):
        llm, _ = make_llm_func_from_config(config, log_path=log_path)
        assert llm("p") == "result"

    records = _read_records(log_path)
    assert len(records) == 1
    assert records[0]["extra"]["pid"] == os.getpid()
    assert records[0]["model"] == "m"


def test_pid_stamp_enables_sibling_exclusion(tmp_path):
    """End-to-end: _usage_from_log_since(pid=...) keeps own-pid records and
    drops records stamped with a different pid."""
    log_path = tmp_path / "inner.jsonl"
    llm, _ = make_llm_func(_mock_llm_client(), None, log_path=log_path)
    llm("hi")

    own = _usage_from_log_since(str(log_path), 0, pid=os.getpid())
    assert own == {"total": 20, "prompt": 14, "completion": 6, "calls": 1}

    sibling = _usage_from_log_since(str(log_path), 0, pid=os.getpid() + 1)
    assert sibling == {"total": 0, "prompt": 0, "completion": 0, "calls": 0}
