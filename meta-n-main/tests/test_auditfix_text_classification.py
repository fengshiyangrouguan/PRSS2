"""Offline regression tests for audit findings 60, 61, 71 in
meta_n/integrations/text_classification.py.

Each test fails on the pre-fix code and passes after the fix. No network,
no real LLM, no Docker.
"""

from __future__ import annotations

import json

import pytest

import meta_n.integrations.text_classification as tc


# ---------------------------------------------------------------------------
# Finding 60 — non-atomic LawBench download bricks the cache on a partial write
# ---------------------------------------------------------------------------


def test_finding60_partial_download_leaves_no_corrupt_cache(tmp_path, monkeypatch):
    """A download that drops mid-transfer must not leave a partial 3-3.json.

    Pre-fix: urlretrieve writes directly into the final raw_path, so the
    partial file survives the exception and the exists() guard then SKIPS
    re-download forever (json.load crashes every subsequent run).
    Post-fix: the bytes land in a temp file that is unlinked on failure, so
    no corrupt cache file is left behind.
    """

    def fake_urlretrieve(url, filename, *args, **kwargs):
        # Simulate a partial transfer: write some bytes, then the network drops.
        with open(filename, "w") as f:
            f.write('{"question": "partial')  # truncated / invalid JSON
        raise OSError("connection reset by peer")

    monkeypatch.setattr(tc.urllib.request, "urlretrieve", fake_urlretrieve)

    with pytest.raises(OSError):
        tc._load_lawbench_charge(tmp_path)

    raw_path = tmp_path / "lawbench_charge" / "3-3.json"
    assert not raw_path.exists(), (
        "a failed/partial download must not leave a cache file that bricks "
        "every subsequent run"
    )


def test_finding60_corrupt_download_is_not_promoted(tmp_path, monkeypatch):
    """Even a 'successful' urlretrieve that yields unparseable bytes must not
    be promoted into the cache path (validate-before-rename)."""

    def fake_urlretrieve(url, filename, *args, **kwargs):
        with open(filename, "w") as f:
            f.write("this is not json")  # downloads fine, but corrupt

    monkeypatch.setattr(tc.urllib.request, "urlretrieve", fake_urlretrieve)

    with pytest.raises(json.JSONDecodeError):
        tc._load_lawbench_charge(tmp_path)

    raw_path = tmp_path / "lawbench_charge" / "3-3.json"
    assert not raw_path.exists()


# ---------------------------------------------------------------------------
# Finding 61 — solver-crash path in the subprocess discards partial usage
# ---------------------------------------------------------------------------


def test_finding61_crash_preserves_partial_usage(monkeypatch):
    """solve() that makes a real llm() call then raises must still report the
    usage already incurred (pre-fix returned _empty_usage()).
    """
    import meta_n.core.llm_helpers as helpers

    # Neutralise os.setsid so invoking the subprocess target in-process does
    # not detach the running test session.
    monkeypatch.delattr(tc.os, "setsid", raising=False)

    def fake_factory(config, tracker, log_path=None):
        def llm(*args, **kwargs):
            tracker.record(30, prompt_tokens=20, completion_tokens=10)
            return "resp"

        def llm_batch(prompts, *args, **kwargs):
            outs = []
            for _ in prompts:
                tracker.record(30, prompt_tokens=20, completion_tokens=10)
                outs.append("resp")
            return outs

        return llm, llm_batch

    monkeypatch.setattr(helpers, "make_llm_func_from_config", fake_factory)

    solution = (
        "def solve(cases, labels, few_shot):\n"
        "    llm('diagnose this')\n"
        "    raise ValueError('boom after a real llm call')\n"
    )

    class _FakeQueue:
        def __init__(self):
            self.items = []

        def put(self, item):
            self.items.append(item)

    q = _FakeQueue()
    tc._run_solve_in_process(
        solution, {"case_0": "x"}, ["a", "b"], [], {"model": "fake"}, q,
    )

    assert len(q.items) == 1
    status, payload, usage = q.items[0]
    assert status == "error"
    assert "boom" in payload
    # The one llm() call really happened — its usage must survive the crash.
    assert usage["calls"] == 1
    assert usage["total"] == 30
    assert usage["prompt"] == 20
    assert usage["completion"] == 10


# ---------------------------------------------------------------------------
# Finding 71 — timeout path silently zeroes inner-LLM usage already spent
# ---------------------------------------------------------------------------


def test_finding71_timeout_reconstructs_partial_usage(tmp_path):
    """A solver killed for exceeding the wall-clock timeout must report the
    inner-LLM usage it logged before the kill (pre-fix returned _empty_usage()).

    The solve() body writes two inner-LLM JSONL records to the inner-log (as
    LLMIOLogger would) and then hangs past the timeout, so the parent must
    reconstruct usage from those records.
    """
    log_path = str(tmp_path / "inner.jsonl")
    rec1 = json.dumps(
        {"source": "inner_llm", "total_tokens": 30,
         "prompt_tokens": 20, "completion_tokens": 10}
    )
    rec2 = json.dumps(
        {"source": "inner_llm", "total_tokens": 15,
         "prompt_tokens": 9, "completion_tokens": 6}
    )
    solution = (
        "import time\n"
        "def solve(cases, labels, few_shot):\n"
        f"    with open({log_path!r}, 'a') as f:\n"
        f"        f.write({rec1!r} + '\\n')\n"
        f"        f.write({rec2!r} + '\\n')\n"
        "        f.flush()\n"
        "    time.sleep(30)\n"
        "    return {}\n"
    )

    status, payload, usage = tc._run_solve_with_timeout(
        solution, {"case_0": "x"}, ["a"], [], None, 1,
        inner_log_path=log_path,
    )

    assert status == "error"
    assert "Timeout" in payload
    # Usage reconstructed from the inner-log the child appended before the kill.
    assert usage["calls"] == 2
    assert usage["total"] == 45
    assert usage["prompt"] == 29
    assert usage["completion"] == 16
