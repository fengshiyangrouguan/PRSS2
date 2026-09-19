"""Refinement regression tests for meta_n/analysis/emergent_roles.py.

Covers:
- F144: _load_injected_codes falls back to the evolutionary archive layout
  (archive/<best>/injected_code_d{N}.json) when no linear depth_N dirs exist,
  so analyze() on an evolutionary run no longer silently returns empty.
- F148: the linear path (shared contiguous loader + omega_response.txt merge)
  is behavior-identical to the pre-consolidation loop.

All offline / LLM-free / no Docker / no network. Embeddings are forced off
(sentence-transformers absent → the analyzer degrades gracefully already; we
pin that path explicitly so the tests are deterministic either way).
"""

import json

import pytest

from meta_n.analysis.emergent_roles import EmergentRoleAnalyzer


@pytest.fixture(autouse=True)
def _no_embedding_model(monkeypatch):
    """Force the no-embeddings path (deterministic regardless of extras)."""
    def _raise(self):
        raise ImportError("embeddings disabled in tests")
    monkeypatch.setattr(EmergentRoleAnalyzer, "_get_embedding_model", _raise)


def _write_linear_fixture(root):
    d2 = root / "depth_2"
    d2.mkdir()
    (d2 / "injected_code.json").write_text(json.dumps(
        {"pre_process": None,
         "rationale": "Adding error handling to catch script failures.",
         "source_depth": 2}
    ))
    (d2 / "omega_response.txt").write_text("Full omega response for depth 2...")
    d3 = root / "depth_3"
    d3.mkdir()
    (d3 / "injected_code.json").write_text(json.dumps(
        {"pre_process": "additional_context = 'classify the task'",
         "rationale": "Classify task type and select strategy.",
         "source_depth": 3}
    ))
    return root


def _write_evolutionary_fixture(root, best_id="gen1_pgen0_seed_k0",
                                with_lineage=True):
    """Archive layout mirroring the evolutionary_experiment metrics fixture."""
    (root / "summary.json").write_text(json.dumps(
        {"best_candidate_id": best_id, "best_mean_score": 0.7,
         "convergence_history": [0.4, 0.7]}
    ))
    archive = root / "archive"
    archive.mkdir()
    (archive / "index.json").write_text(json.dumps(
        {"size": 1, "best_candidate_id": best_id, "candidates": []}
    ))
    cand = archive / best_id
    cand.mkdir()
    (cand / "injected_code_d2.json").write_text(json.dumps(
        {"pre_process": "additional_context = 'no scipy'",
         "rationale": "Removed dependency on scipy", "source_depth": 2}
    ))
    (cand / "injected_code_d3.json").write_text(json.dumps(
        {"pre_process": "additional_context = 'retry on error'",
         "rationale": "Retry failed commands with error handling.",
         "source_depth": 3}
    ))
    if with_lineage:
        lineage = root / "lineage"
        lineage.mkdir()
        (lineage / "best_chain.json").write_text(json.dumps(
            {"best_candidate_id": best_id, "depth": 3}
        ))
    return root


# --------------------------------------------------------------------------- #
# F144 — evolutionary archive layout
# --------------------------------------------------------------------------- #

def test_analyze_reads_evolutionary_archive_layout(tmp_path):
    _write_evolutionary_fixture(tmp_path)
    analyzer = EmergentRoleAnalyzer(str(tmp_path))
    result = analyzer.analyze()
    # Pre-fix: silently empty (loader was blind to the archive layout).
    assert len(result.layer_profiles) == 2
    assert result.layer_profiles[0].depth == 2
    assert result.layer_profiles[1].depth == 3


def test_analyze_evolutionary_missing_lineage_falls_back_to_summary(tmp_path):
    _write_evolutionary_fixture(tmp_path, with_lineage=False)
    analyzer = EmergentRoleAnalyzer(str(tmp_path))
    result = analyzer.analyze()
    assert len(result.layer_profiles) == 2
    assert result.layer_profiles[0].depth == 2


def test_analyze_evolutionary_dangling_best_id_returns_empty(tmp_path):
    _write_evolutionary_fixture(tmp_path)
    # Point both resolvers at a candidate dir that does not exist.
    (tmp_path / "lineage" / "best_chain.json").write_text(json.dumps(
        {"best_candidate_id": "no_such_candidate"}
    ))
    analyzer = EmergentRoleAnalyzer(str(tmp_path))
    result = analyzer.analyze()
    assert result.layer_profiles == []


def test_analyze_evolutionary_no_best_id_returns_empty(tmp_path):
    _write_evolutionary_fixture(tmp_path, with_lineage=False)
    (tmp_path / "summary.json").write_text(json.dumps({"best_mean_score": 0.7}))
    analyzer = EmergentRoleAnalyzer(str(tmp_path))
    assert analyzer.analyze().layer_profiles == []


def test_empty_dir_still_returns_empty(tmp_path):
    # Neither depth_2 nor archive/index.json → unchanged empty result.
    analyzer = EmergentRoleAnalyzer(str(tmp_path))
    result = analyzer.analyze()
    assert result.layer_profiles == []
    assert result.mean_differentiation_score == 0.0


# --------------------------------------------------------------------------- #
# F148 — linear layout unchanged through the shared loader
# --------------------------------------------------------------------------- #

def test_analyze_linear_layout_unchanged(tmp_path):
    _write_linear_fixture(tmp_path)
    analyzer = EmergentRoleAnalyzer(str(tmp_path))
    result = analyzer.analyze()
    assert len(result.layer_profiles) == 2
    assert [p.depth for p in result.layer_profiles] == [2, 3]
    # All 7 improvement types scored, exactly as before the consolidation.
    assert len(result.layer_profiles[0].improvement_type_scores) == 7


def test_linear_loader_merges_omega_response(tmp_path):
    _write_linear_fixture(tmp_path)
    analyzer = EmergentRoleAnalyzer(str(tmp_path))
    codes = analyzer._load_injected_codes()
    assert codes[0].raw_omega_response == "Full omega response for depth 2..."
    assert codes[1].raw_omega_response == ""  # depth_3 ships no response file


def test_linear_layout_wins_over_archive_when_both_present(tmp_path):
    """A dir with BOTH layouts keeps the pre-fix (linear-first) behavior."""
    _write_linear_fixture(tmp_path)
    _write_evolutionary_fixture(tmp_path)
    analyzer = EmergentRoleAnalyzer(str(tmp_path))
    codes = analyzer._load_injected_codes()
    assert codes[0].rationale == "Adding error handling to catch script failures."
