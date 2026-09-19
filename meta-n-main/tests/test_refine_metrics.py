"""Refinement regression tests for meta_n/analysis/metrics.py.

Covers:
- F141: compute_failure_patterns no longer loads (nor depends on) the
  convergence history.
- F148: the shared contiguous injected-code loader keeps while-loop
  (stop-at-first-gap) semantics.
- F149: the by-iteration best-candidate grouping helper pins the strict-``>``
  tie semantics and the ``generation`` legacy fallback.
- F158: baseline-mode detection + honest skip labels in run_all().
- F162: analysis consumes the public ``classify_error`` (same function object
  as the ``OmegaEngine._classify_error`` staticmethod alias).

All offline / LLM-free / no Docker / no network.
"""

import json

from meta_n.analysis import metrics as metrics_mod
from meta_n.analysis.metrics import (
    ExperimentAnalyzer,
    _best_candidate_by_iteration,
    _iteration_of,
    _load_contiguous_injected_codes,
)
from meta_n.core.meta_layer import classify_error
from meta_n.core.omega import OmegaEngine


def _write_evolutionary_fixture(root):
    """Minimal evolutionary run dir (mirrors tests/test_metrics.py)."""
    summary = {
        "archive_size": 2,
        "total_iterations": 2,
        "total_tokens": 2000,
        "best_mean_score": 0.7,
        "best_candidate_id": "gen1_pgen0_seed_k0",
        "per_task_best_scores": {"t1": 0.9, "t2": 0.5},
        "convergence_history": [0.4, 0.7],
    }
    (root / "summary.json").write_text(json.dumps(summary))
    (root / "config.json").write_text(json.dumps({"model": "test"}))

    archive = root / "archive"
    archive.mkdir()
    index = {
        "size": 2,
        "best_mean_score": 0.7,
        "best_candidate_id": "gen1_pgen0_seed_k0",
        "per_task_best": {
            "t1": {"score": 0.9, "candidate_id": "gen1_pgen0_seed_k0"},
            "t2": {"score": 0.5, "candidate_id": "gen0_seed"},
        },
        "candidates": [
            {"candidate_id": "gen0_seed", "iteration": 0, "depth": 1,
             "mean_score": 0.4, "pass_at_1": 0.5,
             "per_task_scores": {"t1": 0.8, "t2": 0.0}, "total_tokens": 1000},
            {"candidate_id": "gen1_pgen0_seed_k0", "iteration": 1, "depth": 2,
             "mean_score": 0.7, "pass_at_1": 1.0,
             "per_task_scores": {"t1": 0.9, "t2": 0.5}, "total_tokens": 1000},
        ],
    }
    (archive / "index.json").write_text(json.dumps(index))

    seed_dir = archive / "gen0_seed"
    (seed_dir / "traces").mkdir(parents=True)
    (seed_dir / "summary.json").write_text(json.dumps(index["candidates"][0]))
    (seed_dir / "traces" / "t1.json").write_text(json.dumps(
        {"task_id": "t1", "depth": 1, "script": "x=1", "success": True,
         "score": 0.8, "error_summary": "", "stderr": "", "stdout": ""}
    ))
    (seed_dir / "traces" / "t2.json").write_text(json.dumps(
        {"task_id": "t2", "depth": 1, "script": "x=1", "success": False,
         "score": 0.0, "error_summary": "ModuleNotFoundError: scipy",
         "stderr": "No module named scipy", "stdout": ""}
    ))

    best_dir = archive / "gen1_pgen0_seed_k0"
    (best_dir / "traces").mkdir(parents=True)
    (best_dir / "summary.json").write_text(json.dumps(index["candidates"][1]))
    for tid, score in [("t1", 0.9), ("t2", 0.5)]:
        (best_dir / "traces" / f"{tid}.json").write_text(json.dumps(
            {"task_id": tid, "depth": 2, "script": "x=2", "success": True,
             "score": score, "error_summary": "", "stderr": "", "stdout": ""}
        ))
    (best_dir / "injected_code_d2.json").write_text(json.dumps(
        {"pre_process": "additional_context = 'no scipy'",
         "rationale": "Removed dependency on scipy", "source_depth": 2}
    ))

    lineage = root / "lineage"
    lineage.mkdir()
    (lineage / "best_chain.json").write_text(json.dumps(
        {"best_candidate_id": "gen1_pgen0_seed_k0", "depth": 2}
    ))
    return root


# --------------------------------------------------------------------------- #
# F141 — failure patterns do not depend on the convergence history
# --------------------------------------------------------------------------- #

def test_failure_patterns_does_not_require_convergence_history(tmp_path):
    _write_evolutionary_fixture(tmp_path)
    # Strip BOTH convergence sources from summary.json; the failure-rate
    # trajectory is built from the archive index, never from the history.
    summary = json.loads((tmp_path / "summary.json").read_text())
    summary.pop("convergence_history", None)
    summary.pop("mean_scores", None)
    (tmp_path / "summary.json").write_text(json.dumps(summary))

    analyzer = ExperimentAnalyzer(tmp_path, use_embeddings=False)
    result = analyzer.compute_failure_patterns()
    assert result["failure_rate_trajectory"] == [0.5, 0.0]
    assert result["seed_errors"]["total_failures"] == 1


def test_load_convergence_history_still_used_by_convergence(tmp_path):
    """_load_convergence_history itself stays (compute_convergence uses it)."""
    _write_evolutionary_fixture(tmp_path)
    analyzer = ExperimentAnalyzer(tmp_path, use_embeddings=False)
    assert analyzer.compute_convergence()["convergence_history"] == [0.4, 0.7]


# --------------------------------------------------------------------------- #
# F148 — contiguous loader semantics
# --------------------------------------------------------------------------- #

def test_contiguous_loader_stops_at_gap(tmp_path):
    """d2 present, d3 MISSING, d4 present → exactly one code (while-loop
    contiguity, not glob tolerance)."""
    (tmp_path / "injected_code_d2.json").write_text(json.dumps(
        {"pre_process": "x = 2", "source_depth": 2}
    ))
    (tmp_path / "injected_code_d4.json").write_text(json.dumps(
        {"pre_process": "x = 4", "source_depth": 4}
    ))
    codes = _load_contiguous_injected_codes(
        lambda d: tmp_path / f"injected_code_d{d}.json"
    )
    assert len(codes) == 1
    assert codes[0].source_depth == 2


def test_contiguous_loader_augment_mutates_before_construction(tmp_path):
    (tmp_path / "injected_code_d2.json").write_text(json.dumps(
        {"pre_process": "x = 2", "source_depth": 2}
    ))

    def augment(depth, data):
        data["raw_omega_response"] = f"resp d{depth}"

    codes = _load_contiguous_injected_codes(
        lambda d: tmp_path / f"injected_code_d{d}.json", augment=augment
    )
    assert codes[0].raw_omega_response == "resp d2"


def test_migrated_loaders_match_fixture(tmp_path):
    _write_evolutionary_fixture(tmp_path)
    analyzer = ExperimentAnalyzer(tmp_path, use_embeddings=False)
    codes = analyzer._load_injected_codes_for_candidate("gen1_pgen0_seed_k0")
    assert len(codes) == 1
    assert codes[0].rationale == "Removed dependency on scipy"
    # Dangling candidate id → empty (no dir, no d2 file).
    assert analyzer._load_injected_codes_for_candidate("nonexistent") == []


# --------------------------------------------------------------------------- #
# F149 — by-iteration grouping helper
# --------------------------------------------------------------------------- #

def test_best_by_iteration_tie_keeps_first():
    a = {"candidate_id": "a", "iteration": 1, "mean_score": 0.5}
    b = {"candidate_id": "b", "iteration": 1, "mean_score": 0.5}
    by_iter = _best_candidate_by_iteration([a, b])
    assert by_iter[1]["candidate_id"] == "a"  # strict > keeps first-seen


def test_best_by_iteration_higher_score_wins():
    a = {"candidate_id": "a", "iteration": 1, "mean_score": 0.5}
    b = {"candidate_id": "b", "iteration": 1, "mean_score": 0.6}
    assert _best_candidate_by_iteration([a, b])[1]["candidate_id"] == "b"


def test_iteration_falls_back_to_generation():
    assert _iteration_of({"iteration": 2, "generation": 3}) == 2
    assert _iteration_of({"generation": 3}) == 3
    assert _iteration_of({}) == 0


# --------------------------------------------------------------------------- #
# F158 — baseline mode detection + honest skip labels
# --------------------------------------------------------------------------- #

def test_detect_mode_baseline(tmp_path):
    (tmp_path / "summary.json").write_text(json.dumps(
        {"archive_semantics": "population", "convergence_history": [0.1, 0.2]}
    ))
    analyzer = ExperimentAnalyzer(tmp_path, use_embeddings=False)
    assert analyzer.mode == "baseline"


def test_baseline_run_all_does_not_raise_and_labels_skips(tmp_path):
    (tmp_path / "summary.json").write_text(json.dumps(
        {"archive_semantics": "population", "convergence_history": [0.1, 0.2]}
    ))
    analyzer = ExperimentAnalyzer(tmp_path, use_embeddings=False)
    results = analyzer.run_all()
    assert results["mode"] == "baseline"
    assert "baseline" in results["archive_diversity"]["skipped"]
    assert "linear" not in results["archive_diversity"]["skipped"]
    assert results["failure_patterns"]["skipped"] == "no seed traces"


def test_linear_skip_label_byte_identical(tmp_path):
    """Linear runs keep the exact pre-fix skip string (metrics.json bytes)."""
    (tmp_path / "depth_2").mkdir()
    (tmp_path / "summary.json").write_text(json.dumps({"mean_scores": {"1": 0.5}}))
    analyzer = ExperimentAnalyzer(tmp_path, use_embeddings=False)
    assert analyzer.compute_archive_diversity() == {
        "skipped": "linear mode — no archive"
    }


# --------------------------------------------------------------------------- #
# F162 — public taxonomy function consumed directly
# --------------------------------------------------------------------------- #

def test_public_aliases_are_same_objects(tmp_path):
    # OmegaEngine._classify_error is staticmethod(classify_error): identical
    # function object, so migrating metrics to the public name is byte-safe.
    assert OmegaEngine._classify_error is classify_error
    # metrics.py imports the public function at module scope.
    assert metrics_mod.classify_error is classify_error
    # F162 remainder: the public categorize_task alias is the same object.
    assert OmegaEngine.categorize_task is OmegaEngine._categorize_task

    # And the migrated call path yields the pre-fix distribution.
    _write_evolutionary_fixture(tmp_path)
    analyzer = ExperimentAnalyzer(tmp_path, use_embeddings=False)
    result = analyzer.compute_failure_patterns()
    assert "Dependency error" in result["seed_errors"]["distribution"]
