"""Regression test for re-audit finding #4.

Parallel (eval_workers>1) solver-timeout path must preserve the partial
inner-LLM usage reconstructed by ``_run_solve_with_timeout`` (the #71 fix),
mirroring the sequential path. Before the fix, ``_run_chunk`` raised a plain
``RuntimeError`` that discarded the reconstructed ``usage`` tuple element, so a
timed-out chunk's inner_tokens/inner_calls were lost from the error EvalResult.

Offline: ``_run_solve_with_timeout`` is monkeypatched, so no subprocess, LLM,
Docker, or Omega-generated code runs on the host.
"""

from meta_n.integrations import text_classification as tc
from meta_n.integrations.text_classification import TextClassificationAdapter


def _make_adapter(eval_workers: int = 2) -> TextClassificationAdapter:
    adapter = TextClassificationAdapter(
        dataset_name="symptom2disease",
        eval_workers=eval_workers,
        subprocess_isolate=True,  # route through real _run_chunk
        llm_client=None,
        inner_log_path="/tmp/does-not-matter.jsonl",
    )
    # Avoid touching disk: preload the data _get_data() would return.
    adapter._data = {
        "labels": ["A", "B"],
        "train": [{"text": "t", "label": "A"} for _ in range(5)],
        "val": [],
        "test": [],
    }
    return adapter


def _examples(n: int) -> list[dict]:
    return [{"text": f"case text {i}", "label": "A"} for i in range(n)]


def test_parallel_timeout_preserves_partial_inner_usage(monkeypatch):
    """A chunk that times out must fold its reconstructed usage into the
    error EvalResult (fails on pre-fix code where usage was discarded)."""
    partial_usage = {"total": 100, "prompt": 60, "completion": 40, "calls": 3}

    def fake_timeout(*_args, **_kwargs):
        # Emulate _run_solve_with_timeout on a kill: error status + the
        # partial usage reconstructed from the inner-log.
        return ("error", "Timeout (configured=300s, elapsed=301.0s)", partial_usage)

    monkeypatch.setattr(tc, "_run_solve_with_timeout", fake_timeout)

    adapter = _make_adapter(eval_workers=2)
    examples = _examples(4)  # > eval_workers -> parallel path

    result = adapter._run_solve_code("code", examples, metric="accuracy")

    assert result.success is False
    assert result.score == 0.0
    # The failed chunks' partial inner-LLM usage must survive into EvalResult.
    # Both chunks (4 examples / 2 workers) hit the mocked timeout, and the
    # aggregate must cover EVERY chunk that spent tokens — not just the first
    # failing one (the pre-F132 undercount asserted 1x here).
    assert result.inner_tokens == 200
    assert result.inner_prompt_tokens == 120
    assert result.inner_completion_tokens == 80
    assert result.inner_calls == 6


def test_chunk_error_carries_usage():
    """_run_chunk raises _ChunkError carrying the reconstructed usage."""
    import pytest

    def fake_timeout(*_args, **_kwargs):
        return ("error", "boom", {"total": 7, "prompt": 4, "completion": 3, "calls": 1})

    orig = tc._run_solve_with_timeout
    tc._run_solve_with_timeout = fake_timeout
    try:
        with pytest.raises(tc._ChunkError) as exc:
            tc._run_chunk("code", {"case_0": "x"}, ["A"], [], None, 300, None)
    finally:
        tc._run_solve_with_timeout = orig

    assert exc.value.usage == {"total": 7, "prompt": 4, "completion": 3, "calls": 1}
    # Subclass of RuntimeError so existing except-RuntimeError callers still work.
    assert isinstance(exc.value, RuntimeError)
