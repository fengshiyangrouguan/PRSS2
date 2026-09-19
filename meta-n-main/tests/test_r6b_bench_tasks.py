"""F128 regression — ``--bench-tasks`` matching semantics stay documented.

Matching is per-benchmark: EXACT name for co_bench / terminal_bench /
swe_bench_verified (swe_bench fails hard on zero matches); SUBSTRING for
arc_agi_2 and the OpenEvolve family. These tests pin the documented SUBSTRING
selection (byte-identical to before F128) and the logging-only over-match
warning that surfaces when one pattern selects several tasks.
"""

import json
import logging
from pathlib import Path

import pytest

from meta_n.integrations.arc_agi import ARCAGI2Adapter
from meta_n.integrations.openevolve import AlphaEvolveMathAdapter

_ARC_TASK = {
    "train": [{"input": [[0]], "output": [[1]]}],
    "test": [{"input": [[0]], "output": [[1]]}],
}

_OE_INITIAL = (
    "# EVOLVE-BLOCK-START\n"
    "def pack():\n"
    "    return 0\n"
    "# EVOLVE-BLOCK-END\n"
)


@pytest.fixture
def arc_dir(tmp_path):
    split = tmp_path / "arc" / "data" / "evaluation"
    split.mkdir(parents=True)
    for stem in ("abc123", "abc456", "zzz789"):
        (split / f"{stem}.json").write_text(json.dumps(_ARC_TASK), encoding="utf-8")
    return tmp_path / "arc"


@pytest.fixture
def oe_dir(tmp_path):
    # The exact-name-that-still-over-matches hazard: "circle_packing" is a
    # substring of "circle_packing_v2".
    for name in ("circle_packing", "circle_packing_v2"):
        d = tmp_path / "oe" / name
        d.mkdir(parents=True)
        (d / "evaluator.py").write_text(
            "def evaluate(p):\n    return {'combined_score': 0.0}\n",
            encoding="utf-8",
        )
        (d / "initial_program.py").write_text(_OE_INITIAL, encoding="utf-8")
    return tmp_path / "oe"


def _over_match_records(caplog):
    return [r for r in caplog.records if "substring-matched" in r.getMessage()]


def test_arc_substring_over_match_warns_and_selection_unchanged(arc_dir, caplog):
    adapter = ARCAGI2Adapter(data_dir=str(arc_dir), task_ids=["abc"])
    with caplog.at_level(logging.WARNING, logger="meta_n.integrations.arc_agi"):
        tasks = adapter.load_tasks()
    # Documented SUBSTRING semantics: BOTH abc tasks load (selection unchanged).
    assert sorted(t.task_id for t in tasks) == ["abc123", "abc456"]
    warnings = _over_match_records(caplog)
    assert len(warnings) == 1
    msg = warnings[0].getMessage()
    assert "'abc'" in msg and "2 ARC tasks" in msg


def test_arc_exact_stem_single_match_no_warning(arc_dir, caplog):
    adapter = ARCAGI2Adapter(data_dir=str(arc_dir), task_ids=["abc123"])
    with caplog.at_level(logging.WARNING, logger="meta_n.integrations.arc_agi"):
        tasks = adapter.load_tasks()
    assert [t.task_id for t in tasks] == ["abc123"]
    assert not _over_match_records(caplog)


def test_openevolve_substring_over_match_warns(oe_dir, caplog):
    adapter = AlphaEvolveMathAdapter(
        data_dir=str(oe_dir), problem_names=["circle_packing"]
    )
    with caplog.at_level(logging.WARNING, logger="meta_n.integrations.openevolve"):
        problems = adapter._discover_problems()
    assert sorted(p["name"] for p in problems) == [
        "circle_packing", "circle_packing_v2",
    ]
    warnings = _over_match_records(caplog)
    assert len(warnings) == 1
    msg = warnings[0].getMessage()
    assert "'circle_packing'" in msg and "2 OpenEvolve problems" in msg


def test_openevolve_unique_match_no_warning(oe_dir, caplog):
    adapter = AlphaEvolveMathAdapter(
        data_dir=str(oe_dir), problem_names=["circle_packing_v2"]
    )
    with caplog.at_level(logging.WARNING, logger="meta_n.integrations.openevolve"):
        problems = adapter._discover_problems()
    assert [p["name"] for p in problems] == ["circle_packing_v2"]
    assert not _over_match_records(caplog)


def test_no_filter_no_warning(arc_dir, oe_dir, caplog):
    # Default path (no --bench-tasks): everything loads, zero warnings.
    with caplog.at_level(logging.WARNING):
        arc_tasks = ARCAGI2Adapter(data_dir=str(arc_dir)).load_tasks()
        problems = AlphaEvolveMathAdapter(data_dir=str(oe_dir))._discover_problems()
    assert len(arc_tasks) == 3
    assert len(problems) == 2
    assert not _over_match_records(caplog)


def test_help_text_states_per_benchmark_semantics():
    # Cheap doc-drift guard: the --bench-tasks help must name both matching
    # families. Read the source instead of invoking argparse.
    import meta_n

    src = (Path(meta_n.__file__).resolve().parent / "main.py").read_text(
        encoding="utf-8"
    )
    idx = src.index('"--bench-tasks"')
    help_region = src[idx:idx + 800]
    assert "EXACT" in help_region
    assert "SUBSTRING" in help_region
