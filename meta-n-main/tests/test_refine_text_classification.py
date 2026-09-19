"""Refinement regression tests for meta_n/integrations/text_classification.py.

Covers:
* F116 — _fold_inner_usage renders the exact prompt-visible ``[inner-llm]``
  suffix (byte-identical to the three pre-consolidation copies) and appends
  nothing when no inner call happened.
* F130 — max_test=0 must yield an EMPTY test split (None-sentinel slicing),
  coherent with the ``_t0`` cache key; the default (None) path is unchanged.
* F132 — the parallel path folds inner-LLM usage from EVERY chunk (successes
  after the first failure and additional failed chunks included), while the
  reported error stays the submission-order-first failure.

Offline: no LLM, no Docker, no network, no subprocess (parallel-path tests
monkeypatch _run_solve_with_timeout).
"""

from __future__ import annotations

import json
import sys
import types

import meta_n.integrations.text_classification as tc
from meta_n.integrations.benchmark import EvalResult
from meta_n.integrations.text_classification import TextClassificationAdapter

# ---------------------------------------------------------------------------
# F116 — exact fold-helper suffix
# ---------------------------------------------------------------------------


def test_fold_inner_usage_exact_suffix():
    r = EvalResult(success=True, score=1.0, feedback="Accuracy: 1/1 = 1.0000")
    out = tc._fold_inner_usage(
        r, {"total": 45, "prompt": 29, "completion": 16, "calls": 2}
    )
    assert out is r  # mutates and returns the same result
    # Byte-identical to the pre-consolidation literal (prompt-visible surface).
    assert r.feedback == (
        "Accuracy: 1/1 = 1.0000\n[inner-llm] 2 calls, 45 tokens (in=29, out=16)"
    )
    assert (r.inner_tokens, r.inner_prompt_tokens,
            r.inner_completion_tokens, r.inner_calls) == (45, 29, 16, 2)


def test_fold_inner_usage_zero_calls_appends_nothing():
    r = EvalResult(success=True, score=1.0, feedback="fb")
    tc._fold_inner_usage(r, {"total": 0, "prompt": 0, "completion": 0, "calls": 0})
    assert r.feedback == "fb"
    assert (r.inner_tokens, r.inner_prompt_tokens,
            r.inner_completion_tokens, r.inner_calls) == (0, 0, 0, 0)


# ---------------------------------------------------------------------------
# F130 — max_test=0 coheres with the _t0 cache key
# ---------------------------------------------------------------------------


def _seed_lawbench(tmp_path, n: int = 260) -> None:
    """Pre-seed a fake 3-3.json so the exists() guard skips the download."""
    cache_dir = tmp_path / "lawbench_charge"
    cache_dir.mkdir(parents=True)
    raw = [
        {"question": f"q{i}", "answer": f"罪名:c{i % 7}", "instruction": ""}
        for i in range(n)
    ]
    (cache_dir / "3-3.json").write_text(json.dumps(raw, ensure_ascii=False))


def test_lawbench_max_test_zero_yields_empty_split(tmp_path):
    _seed_lawbench(tmp_path)
    data = tc._load_lawbench_charge(tmp_path, max_val=50, max_test=0)
    assert data["test"] == []


def test_lawbench_max_test_none_yields_full_remainder(tmp_path):
    _seed_lawbench(tmp_path, n=260)
    data = tc._load_lawbench_charge(tmp_path, max_val=50, max_test=None)
    assert len(data["test"]) == 260 - 200 - 50  # everything after train+val


def test_symptom2disease_t0_cache_key_coheres(tmp_path, monkeypatch):
    """max_test=0 writes a ``_t0`` cache whose content really IS an empty test
    split (pre-fix the falsy-0 slice cached the FULL split under _t0)."""
    fake = types.ModuleType("datasets")

    def load_dataset(name):
        return {
            "train": [{"input_text": f"t{i}", "output_text": "d"} for i in range(10)],
            "test": [{"input_text": f"x{i}", "output_text": "d"} for i in range(5)],
        }

    fake.load_dataset = load_dataset
    monkeypatch.setitem(sys.modules, "datasets", fake)

    data = tc._load_symptom2disease(tmp_path, max_val=3, max_test=0)
    assert data["test"] == []
    cache = tmp_path / "symptom2disease" / "cached_s42_v3_t0.json"
    assert cache.exists()
    assert json.loads(cache.read_text())["test"] == []


# ---------------------------------------------------------------------------
# F132 — parallel path accounts usage from EVERY chunk
# ---------------------------------------------------------------------------


def _make_adapter(eval_workers: int) -> TextClassificationAdapter:
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


def test_parallel_counts_usage_from_all_chunks(monkeypatch):
    """3 chunks: fail(A) / success(B) / fail(C) — usage must be A+B+C and the
    feedback must name the FIRST (submission-order) failure. Pre-F132 the
    aggregate stopped at the first failing future (A only)."""
    usage_a = {"total": 10, "prompt": 6, "completion": 4, "calls": 1}
    usage_b = {"total": 20, "prompt": 12, "completion": 8, "calls": 2}
    usage_c = {"total": 40, "prompt": 24, "completion": 16, "calls": 4}

    def fake_run(solution, chunk_cases, labels, few_shot, cfg, timeout,
                 inner_log_path=None, balanced_json_fallback=False):
        if "case_0" in chunk_cases:  # chunk 0 -> fails with usage A
            return ("error", "boom0", usage_a)
        if "case_2" in chunk_cases:  # chunk 1 -> succeeds with usage B
            return ("ok", {k: labels[0] for k in chunk_cases}, usage_b)
        return ("error", "boom2", usage_c)  # chunk 2 -> fails with usage C

    monkeypatch.setattr(tc, "_run_solve_with_timeout", fake_run)

    adapter = _make_adapter(eval_workers=3)
    examples = [{"text": f"t{i}", "label": "A"} for i in range(6)]  # 3 chunks of 2

    result = adapter._run_solve_code("code", examples, metric="accuracy")

    assert result.success is False
    assert result.score == 0.0
    assert "boom0" in result.feedback  # submission-order-first error reported
    assert "boom2" not in result.feedback
    assert result.inner_tokens == 70
    assert result.inner_prompt_tokens == 42
    assert result.inner_completion_tokens == 28
    assert result.inner_calls == 7


def test_parallel_success_path_unchanged(monkeypatch):
    """All chunks succeed -> merged predictions scored, usage summed, and the
    [inner-llm] suffix rendered exactly once (via _fold_inner_usage)."""
    usage = {"total": 5, "prompt": 3, "completion": 2, "calls": 1}

    def fake_run(solution, chunk_cases, labels, few_shot, cfg, timeout,
                 inner_log_path=None, balanced_json_fallback=False):
        return ("ok", {k: "A" for k in chunk_cases}, usage)

    monkeypatch.setattr(tc, "_run_solve_with_timeout", fake_run)

    adapter = _make_adapter(eval_workers=2)
    examples = [{"text": f"t{i}", "label": "A"} for i in range(4)]  # 2 chunks

    result = adapter._run_solve_code("code", examples, metric="accuracy")

    assert result.success is True
    assert result.score == 1.0
    assert result.inner_tokens == 10
    assert result.inner_calls == 2
    assert result.feedback.endswith("\n[inner-llm] 2 calls, 10 tokens (in=6, out=4)")
