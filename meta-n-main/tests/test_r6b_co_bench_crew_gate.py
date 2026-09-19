"""F120 regression — crew-solo precondition of the CO-Bench held-out gate.

``_CREW_HELDOUT_RUNNER`` deliberately diverges from ``_TaskEvaluator``:
(a) an empty dev list means "no dev instances" in the runner but defaults to
``[0]`` in the evaluator, and (b) the runner lists ``*.txt`` files only while
the evaluator lists every non-.py/__pycache__/.DS_Store entry. Both are
unreachable under the crew-solo gate (see the runner's header comment). These
tests pin the gate edge and the two crew-data facts that keep the divergences
unreachable, so any drift fails loudly BEFORE the gate silently diverges.
"""

import hashlib
import importlib.util
import os
from pathlib import Path

import pytest

from meta_n.integrations.co_bench import (
    _CREW_HELDOUT_RUNNER,
    _CREW_TASK_NAME,
    COBenchAdapter,
)

_CREW_DIR = Path(__file__).resolve().parents[1] / "data" / "co_bench" / _CREW_TASK_NAME


def test_multi_task_selection_returns_none():
    # The runner gates ONLY a crew-solo selection: a multi-task selection that
    # merely CONTAINS crew must fall back to None (⇒ the keep-all Stub), or the
    # crew oracle would fail-closed-DROP every non-crew helper.
    adapter = COBenchAdapter(
        data_dir="/tmp/co_bench",
        task_names=[_CREW_TASK_NAME, "Set covering"],
    )
    assert adapter.make_heldout_verifier() is None


def test_default_task_selection_returns_none():
    # task_names=None expands to the full CO_BENCH_TASKS list — not crew-solo.
    adapter = COBenchAdapter(data_dir="/tmp/co_bench", task_names=None)
    assert adapter.make_heldout_verifier() is None


@pytest.mark.skipif(
    not (_CREW_DIR / "config.py").exists(),
    reason="crew dataset not vendored in this checkout",
)
def test_crew_data_contract_upholds_unreachability():
    spec = importlib.util.spec_from_file_location(
        "crew_cfg_f120", str(_CREW_DIR / "config.py")
    )
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)

    evaluator_listing = sorted(
        f for f in os.listdir(_CREW_DIR)
        if not (f.endswith(".py") or f == "__pycache__" or f == ".DS_Store")
    )
    runner_listing = sorted(
        f for f in os.listdir(_CREW_DIR) if f.endswith(".txt")
    )

    # Divergence (b) unreachable: the evaluator-style listing and the runner's
    # *.txt-only listing see exactly the same crew data files.
    assert runner_listing, "crew dir has no .txt data files"
    assert evaluator_listing == runner_listing

    # Divergence (a) unreachable: get_dev() gives every crew data file a
    # NON-EMPTY dev index list, so the runner's empty-list-means-no-dev
    # convention (vs the evaluator's [0]-default) can never fire for crew.
    dev_map = cfg.get_dev()
    for fname in runner_listing:
        assert dev_map.get(fname), f"{fname}: empty/missing dev index list"


def test_runner_bytes_untouched_by_this_change():
    # The documented divergences remain IN the runner: F120 documents AROUND
    # the comparability-pinned string, it does not change gate behavior. Any
    # edit to the runner string invalidates byte-comparability with prior
    # --verified-code gate runs and requires re-baselining.
    assert "or [])" in _CREW_HELDOUT_RUNNER          # empty-dev ⇒ no-dev
    assert 'endswith(".txt")' in _CREW_HELDOUT_RUNNER  # *.txt-only listing
    # Frozen byte pin: the exact runner string prior --verified-code gate runs
    # executed. ANY byte change trips this (the substring asserts above are
    # readable documentation, not the guard) and requires re-baselining.
    assert (
        hashlib.sha256(_CREW_HELDOUT_RUNNER.encode()).hexdigest()
        == "63d34509440310860c8bd5fee8fd6f8972a6467d77b4f3148fd14176d357b47c"
    )
