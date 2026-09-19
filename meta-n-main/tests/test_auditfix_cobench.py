"""Regression tests for audit fixes in meta_n/integrations/co_bench.py.

Covers findings 10, 11, 23, 40, 58. Every test is LLM-free / offline: no
LM Studio, no Docker, no network. The two evaluator-driven tests
(findings 11 and 58) monkeypatch the module-level ``_run_with_timeout`` so NO
subprocess is spawned and no model/Omega-authored code is executed on the host —
the score for the single fake instance is canned directly.

Each test FAILS on the pre-fix code and PASSES after the fix.
"""

import textwrap
from pathlib import Path

import pytest

import meta_n.integrations.co_bench as co_bench
from meta_n.integrations.co_bench import (
    _CREW_TASK_NAME,
    COBenchAdapter,
    _TaskEvaluator,
    _empty_usage,
)


def _write_task(root: Path, task_name: str, config_src: str, data_files=()):
    """Create ``<root>/<task_name>/config.py`` (+ empty data files)."""
    task_dir = root / task_name
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "config.py").write_text(textwrap.dedent(config_src))
    for fname in data_files:
        (task_dir / fname).write_text("")  # load_data is config-controlled
    return task_dir


# ---------------------------------------------------------------------------
# Finding 23 — _average_score must count instance-level failures (error/timeout
# strings, non-finite) as 0, while still excluding the explicit None "skip"
# sentinel. Pure unit test on _average_score (no subprocess).
# ---------------------------------------------------------------------------

_MINIMAL_CONFIG = """
    def load_data(path):
        return []

    def eval_func(**kwargs):
        return 0.0
"""


def _minimal_evaluator(tmp_path: Path) -> _TaskEvaluator:
    _write_task(tmp_path, "Tmin", _MINIMAL_CONFIG)
    return _TaskEvaluator("Tmin", tmp_path, timeout=2, instance_workers=1)


def test_average_score_counts_error_string_as_zero(tmp_path):
    ev = _minimal_evaluator(tmp_path)
    # One feasible instance (1.0) and one timeout (recorded as an error STRING).
    # Pre-fix: the string is filtered out, mean = 1.0/1 = 1.0 (feasible-subset
    # inflation). Post-fix: it counts as 0.0, mean = (1.0 + 0.0)/2 = 0.5.
    avg = ev._average_score({"f.txt": ([1.0, "Timeout (10s)"], None)})
    assert avg == pytest.approx(0.5)


def test_average_score_multi_file_error_pulls_down(tmp_path):
    ev = _minimal_evaluator(tmp_path)
    # a: (2.0 + 0.0)/2 = 1.0 post-fix (pre-fix would be 2.0); b: 1.0.
    # Post-fix overall = (1.0 + 1.0)/2 = 1.0; pre-fix = (2.0 + 1.0)/2 = 1.5.
    avg = ev._average_score(
        {"a.txt": ([2.0, "boom"], None), "b.txt": ([1.0], None)}
    )
    assert avg == pytest.approx(1.0)


def test_average_score_preserves_none_skip_sentinel(tmp_path):
    ev = _minimal_evaluator(tmp_path)
    # None is the explicit norm_score "skip this instance" sentinel — it must be
    # EXCLUDED from the denominator (not counted as 0). mean = 1.0/1 = 1.0.
    avg = ev._average_score({"f.txt": ([1.0, None], None)})
    assert avg == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Finding 11 — an in-dev_map file whose load_data fails (-> ([], err)) must
# survive the dev/test filter and count as 0 in the denominator, instead of
# being silently dropped (which inflates the average).
# ---------------------------------------------------------------------------

_FINDING11_CONFIG = """
    def load_data(path):
        if path.endswith("bad.txt"):
            raise ValueError("boom load")
        return [{"x": 1}]

    def eval_func(**kwargs):
        return 1.0

    def get_dev():
        return {"good.txt": [0], "bad.txt": [0]}
"""


def test_loaddata_failure_in_devmap_counts_as_zero(tmp_path, monkeypatch):
    _write_task(tmp_path, "T11", _FINDING11_CONFIG, data_files=["good.txt", "bad.txt"])
    ev = _TaskEvaluator("T11", tmp_path, timeout=2, instance_workers=1)

    # No subprocess: canned "ok" score for the single good.txt instance.
    def _fake_run(*_args, **_kwargs):
        return ("ok", 1.0, _empty_usage())

    monkeypatch.setattr(co_bench, "_run_with_timeout", _fake_run)

    res = ev.evaluate("def solve(**kw):\n    return {}\n")
    # good.txt -> 1.0, bad.txt (load_data failed) -> 0, denominator = 2.
    # Pre-fix: bad.txt dropped from the filter, denominator = 1 -> 1.0.
    assert res["dev_score"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Finding 58 — when norm_score raises, the eval must FAIL CLOSED (score 0)
# rather than averaging the still-raw, un-normalized objective magnitudes.
# ---------------------------------------------------------------------------

_FINDING58_CONFIG = """
    def load_data(path):
        return [{"x": 1}]

    def eval_func(**kwargs):
        return 700.0

    def norm_score(results):
        raise ValueError("norm boom")

    def get_dev():
        return {"f.txt": [0]}
"""


def test_norm_score_failure_fails_closed(tmp_path, monkeypatch):
    _write_task(tmp_path, "T58", _FINDING58_CONFIG, data_files=["f.txt"])
    ev = _TaskEvaluator("T58", tmp_path, timeout=2, instance_workers=1)

    # Canned RAW objective value (700.0) — what eval_func would return before
    # normalization. No subprocess executed.
    def _fake_run(*_args, **_kwargs):
        return ("ok", 700.0, _empty_usage())

    monkeypatch.setattr(co_bench, "_run_with_timeout", _fake_run)

    res = ev.evaluate("def solve(**kw):\n    return {}\n")
    # Pre-fix: norm_score's exception is swallowed and the raw 700.0 becomes the
    # dev score (inflated, and direction-inverted for a minimization task).
    # Post-fix: every entry is marked errored -> counts as 0 -> dev_score 0.0.
    assert res["dev_score"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Finding 10 — the crew held-out verifier must gate ONLY a crew-ONLY task
# selection. A multi-task selection that merely CONTAINS crew must return None
# (else every non-crew helper is fail-closed-DROPped against the crew oracle).
# ---------------------------------------------------------------------------


def test_crew_only_selection_returns_verifier():
    from meta_n.core.verified_code import SandboxedHeldoutVerifier

    adapter = COBenchAdapter(data_dir="/tmp/co_bench", task_names=[_CREW_TASK_NAME])
    assert isinstance(adapter.make_heldout_verifier(), SandboxedHeldoutVerifier)


def test_multitask_selection_containing_crew_returns_none():
    # Pre-fix: `_CREW_TASK_NAME not in task_names` is False, so the crew verifier
    # is (wrongly) returned and would gate non-crew helpers against the crew
    # oracle. Post-fix: a non-crew-only selection returns None.
    adapter = COBenchAdapter(
        data_dir="/tmp/co_bench",
        task_names=[_CREW_TASK_NAME, "Set covering"],
    )
    assert adapter.make_heldout_verifier() is None


def test_non_crew_single_task_returns_none():
    adapter = COBenchAdapter(data_dir="/tmp/co_bench", task_names=["Set covering"])
    assert adapter.make_heldout_verifier() is None


# ---------------------------------------------------------------------------
# Finding 40 — _load_task_data must not store the dead "description" key.
# ---------------------------------------------------------------------------

_FINDING40_CONFIG = """
    DESCRIPTION = "Fake CO task description ABC"

    def solve(x):
        y = x + 1
        return {"y": y}

    def load_data(path):
        return [{"x": 1}]

    def eval_func(**kwargs):
        return 1.0
"""


def test_load_task_data_has_no_dead_description_key(tmp_path):
    _write_task(tmp_path, "T40", _FINDING40_CONFIG)
    adapter = COBenchAdapter(data_dir=tmp_path, task_names=["T40"])
    data = adapter._load_task_data("T40")
    assert data is not None
    assert "description" not in data  # dead key removed
    assert set(data.keys()) == {"problem_description", "solve_template"}
    # The description text is still folded into problem_description.
    assert "Fake CO task description ABC" in data["problem_description"]
    assert "def solve(" in data["solve_template"]
