"""Tests for the post-hoc metrics analysis module."""

import json
import tempfile
from pathlib import Path

import pytest

from meta_n.analysis.metrics import ExperimentAnalyzer


# --- Fixtures ---

@pytest.fixture
def linear_experiment(tmp_path):
    """Create a minimal linear experiment directory."""
    # summary.json
    summary = {
        "final_depth": 3,
        "total_tokens": 5000,
        "converged": True,
        "pass_at_1": {"1": 0.5, "2": 0.75, "3": 0.7},
        "mean_scores": {"1": 0.4, "2": 0.65, "3": 0.6},
        "tokens_per_depth": {"1": 1000, "2": 2000, "3": 2000},
    }
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    (tmp_path / "config.json").write_text(json.dumps({"model": "test"}))

    # Depth 1
    d1 = tmp_path / "depth_1"
    d1.mkdir()
    d1_traces = d1 / "traces"
    d1_traces.mkdir()
    for tid, score, success in [("t1", 0.8, True), ("t2", 0.0, False)]:
        trace = {"task_id": tid, "depth": 1, "script": "x=1", "success": success,
                 "score": score, "error_summary": "" if success else "timed out",
                 "stderr": "", "stdout": ""}
        (d1_traces / f"{tid}.json").write_text(json.dumps(trace))
    (d1 / "summary.json").write_text(json.dumps({"depth": 1, "pass_at_1": 0.5, "mean_score": 0.4}))

    # Depth 2
    d2 = tmp_path / "depth_2"
    d2.mkdir()
    d2_traces = d2 / "traces"
    d2_traces.mkdir()
    for tid, score, success in [("t1", 0.9, True), ("t2", 0.4, True)]:
        trace = {"task_id": tid, "depth": 2, "script": "x=2", "success": success,
                 "score": score, "error_summary": "", "stderr": "", "stdout": ""}
        (d2_traces / f"{tid}.json").write_text(json.dumps(trace))
    ic = {"pre_process": "additional_context = 'hint'",
          "rationale": "Fixed timeout issue", "source_depth": 2}
    (d2 / "injected_code.json").write_text(json.dumps(ic))
    (d2 / "omega_response.txt").write_text("rationale\nhint code")
    (d2 / "pre_process.py").write_text("additional_context = 'hint'")
    (d2 / "summary.json").write_text(json.dumps({"depth": 2, "pass_at_1": 0.75, "mean_score": 0.65}))

    # Depth 3
    d3 = tmp_path / "depth_3"
    d3.mkdir()
    d3_traces = d3 / "traces"
    d3_traces.mkdir()
    for tid, score, success in [("t1", 0.7, True), ("t2", 0.5, True)]:
        trace = {"task_id": tid, "depth": 3, "script": "x=3", "success": success,
                 "score": score, "error_summary": "", "stderr": "", "stdout": ""}
        (d3_traces / f"{tid}.json").write_text(json.dumps(trace))
    ic3 = {"pre_process": "if 'scheduling' in task.task_id:\n    additional_context = 'strategy'",
           "rationale": "Categorize tasks by type and apply strategy selection",
           "source_depth": 3}
    (d3 / "injected_code.json").write_text(json.dumps(ic3))
    (d3 / "pre_process.py").write_text(ic3["pre_process"])
    (d3 / "summary.json").write_text(json.dumps({"depth": 3, "pass_at_1": 0.7, "mean_score": 0.6}))

    return tmp_path


@pytest.fixture
def evolutionary_experiment(tmp_path):
    """Create a minimal evolutionary experiment directory."""
    summary = {
        "archive_size": 3,
        "total_iterations": 2,
        "total_tokens": 3000,
        "best_mean_score": 0.7,
        "best_candidate_id": "gen1_pgen0_seed_k0",
        "per_task_best_scores": {"t1": 0.9, "t2": 0.5},
        "convergence_history": [0.4, 0.7, 0.7],
    }
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    (tmp_path / "config.json").write_text(json.dumps({"model": "test"}))
    (tmp_path / "convergence.json").write_text(json.dumps([0.4, 0.7, 0.7]))

    archive = tmp_path / "archive"
    archive.mkdir()

    # Index
    index = {
        "size": 3,
        "best_mean_score": 0.7,
        "best_candidate_id": "gen1_pgen0_seed_k0",
        "per_task_best": {
            "t1": {"score": 0.9, "candidate_id": "gen1_pgen0_seed_k0"},
            "t2": {"score": 0.5, "candidate_id": "gen0_seed"},
        },
        "candidates": [
            {"candidate_id": "gen0_seed", "iteration": 0, "depth": 1,
             "mean_score": 0.4, "pass_at_1": 0.5, "per_task_scores": {"t1": 0.8, "t2": 0.0},
             "num_children": 1, "temperature_used": 0.7, "total_tokens": 1000},
            {"candidate_id": "gen1_pgen0_seed_k0", "iteration": 1, "depth": 2,
             "mean_score": 0.7, "pass_at_1": 1.0, "per_task_scores": {"t1": 0.9, "t2": 0.5},
             "num_children": 0, "temperature_used": 0.5, "total_tokens": 1000},
            {"candidate_id": "gen1_pgen0_seed_k1", "iteration": 1, "depth": 2,
             "mean_score": 0.55, "pass_at_1": 0.5, "per_task_scores": {"t1": 0.6, "t2": 0.5},
             "num_children": 0, "temperature_used": 0.7, "total_tokens": 1000},
        ],
    }
    (archive / "index.json").write_text(json.dumps(index))

    # Seed candidate
    seed_dir = archive / "gen0_seed"
    seed_dir.mkdir()
    (seed_dir / "summary.json").write_text(json.dumps(index["candidates"][0]))
    traces_dir = seed_dir / "traces"
    traces_dir.mkdir()
    (traces_dir / "t1.json").write_text(json.dumps(
        {"task_id": "t1", "depth": 1, "script": "x=1", "success": True,
         "score": 0.8, "error_summary": "", "stderr": "", "stdout": ""}
    ))
    (traces_dir / "t2.json").write_text(json.dumps(
        {"task_id": "t2", "depth": 1, "script": "x=1", "success": False,
         "score": 0.0, "error_summary": "ModuleNotFoundError: scipy",
         "stderr": "No module named scipy", "stdout": ""}
    ))

    # Best candidate
    best_dir = archive / "gen1_pgen0_seed_k0"
    best_dir.mkdir()
    (best_dir / "summary.json").write_text(json.dumps(index["candidates"][1]))
    traces_dir = best_dir / "traces"
    traces_dir.mkdir()
    (traces_dir / "t1.json").write_text(json.dumps(
        {"task_id": "t1", "depth": 2, "script": "x=2", "success": True,
         "score": 0.9, "error_summary": "", "stderr": "", "stdout": ""}
    ))
    (traces_dir / "t2.json").write_text(json.dumps(
        {"task_id": "t2", "depth": 2, "script": "x=2", "success": True,
         "score": 0.5, "error_summary": "", "stderr": "", "stdout": ""}
    ))
    ic = {"pre_process": "additional_context = 'no scipy'",
          "rationale": "Removed dependency on scipy", "source_depth": 2}
    (best_dir / "injected_code_d2.json").write_text(json.dumps(ic))

    # Lineage
    lineage_dir = tmp_path / "lineage"
    lineage_dir.mkdir()
    (lineage_dir / "best_chain.json").write_text(json.dumps({
        "best_candidate_id": "gen1_pgen0_seed_k0",
        "depth": 2,
        "mean_score": 0.7,
        "injected_codes": [ic],
        "ancestry": ["gen0_seed", "gen1_pgen0_seed_k0"],
    }))

    return tmp_path


# --- Tests ---

class TestModeDetection:
    def test_linear_mode(self, linear_experiment):
        analyzer = ExperimentAnalyzer(linear_experiment, use_embeddings=False)
        assert analyzer.mode == "linear"

    def test_evolutionary_mode(self, evolutionary_experiment):
        analyzer = ExperimentAnalyzer(evolutionary_experiment, use_embeddings=False)
        assert analyzer.mode == "evolutionary"

    def test_unknown_mode(self, tmp_path):
        (tmp_path / "summary.json").write_text("{}")
        with pytest.raises(FileNotFoundError):
            ExperimentAnalyzer(tmp_path)


class TestConvergence:
    def test_linear_convergence(self, linear_experiment):
        analyzer = ExperimentAnalyzer(linear_experiment, use_embeddings=False)
        result = analyzer.compute_convergence()
        assert result["convergence_history"] == [0.4, 0.65, 0.6]
        assert result["marginal_improvement"][1] == pytest.approx(0.25)
        assert result["final_score"] == 0.6
        assert result["total_steps"] == 3

    def test_evolutionary_convergence(self, evolutionary_experiment):
        analyzer = ExperimentAnalyzer(evolutionary_experiment, use_embeddings=False)
        result = analyzer.compute_convergence()
        assert result["convergence_history"] == [0.4, 0.7, 0.7]
        assert result["marginal_improvement"][1] == pytest.approx(0.3)
        assert result["final_score"] == 0.7

    def test_time_to_thresholds(self, evolutionary_experiment):
        analyzer = ExperimentAnalyzer(evolutionary_experiment, use_embeddings=False)
        result = analyzer.compute_convergence()
        assert result["time_to_thresholds"]["0.5"] == 1
        assert result["time_to_thresholds"]["0.7"] == 1
        assert result["time_to_thresholds"]["0.8"] is None


class TestEmergentAbstraction:
    def test_linear_abstraction(self, linear_experiment):
        analyzer = ExperimentAnalyzer(linear_experiment, use_embeddings=False)
        result = analyzer.compute_emergent_abstraction()
        chain = result["best_chain"]
        assert len(chain["abstraction_gradient"]) == 2  # depth 2 and 3
        assert len(chain["improvement_type_profiles"]) == 2
        assert chain["improvement_type_profiles"][0]["depth"] == 2
        assert chain["improvement_type_profiles"][1]["depth"] == 3

    def test_code_complexity(self, linear_experiment):
        analyzer = ExperimentAnalyzer(linear_experiment, use_embeddings=False)
        result = analyzer.compute_emergent_abstraction()
        gradient = result["best_chain"]["code_complexity_gradient"]
        # Depth 3 has if/branch, should have more nodes
        assert gradient[1]["num_branches"] >= gradient[0]["num_branches"]

    def test_no_codes_returns_skipped(self, tmp_path):
        (tmp_path / "depth_2").mkdir()
        (tmp_path / "summary.json").write_text(json.dumps({"mean_scores": {"1": 0.5}}))
        analyzer = ExperimentAnalyzer(tmp_path, use_embeddings=False)
        result = analyzer.compute_emergent_abstraction()
        assert "skipped" in result


class TestArchiveDiversity:
    def test_evolutionary_diversity(self, evolutionary_experiment):
        analyzer = ExperimentAnalyzer(evolutionary_experiment, use_embeddings=False)
        result = analyzer.compute_archive_diversity()
        assert result["oracle_mean"] == pytest.approx(0.7)
        assert result["contributing_chains"] == 2
        assert result["total_candidates"] == 3

    def test_linear_skipped(self, linear_experiment):
        analyzer = ExperimentAnalyzer(linear_experiment, use_embeddings=False)
        result = analyzer.compute_archive_diversity()
        assert "skipped" in result


class TestFailurePatterns:
    def test_seed_vs_best(self, evolutionary_experiment):
        analyzer = ExperimentAnalyzer(evolutionary_experiment, use_embeddings=False)
        result = analyzer.compute_failure_patterns()
        assert result["seed_errors"]["total_failures"] == 1
        assert "Dependency error" in result["seed_errors"]["distribution"]
        assert result["best_candidate_errors"]["total_failures"] == 0

    def test_error_reduction(self, evolutionary_experiment):
        analyzer = ExperimentAnalyzer(evolutionary_experiment, use_embeddings=False)
        result = analyzer.compute_failure_patterns()
        assert result["error_reduction"]["Dependency error"] == 1

    def test_linear_failure_patterns(self, linear_experiment):
        analyzer = ExperimentAnalyzer(linear_experiment, use_embeddings=False)
        result = analyzer.compute_failure_patterns()
        assert result["seed_errors"]["total_failures"] == 1


class TestRobustness:
    def test_no_regressions(self, evolutionary_experiment):
        analyzer = ExperimentAnalyzer(evolutionary_experiment, use_embeddings=False)
        result = analyzer.compute_robustness()
        # t1: 0.8 → 0.9 (improved), t2: 0.0 → 0.5 (improved)
        assert result["num_regressions"] == 0

    def test_best_depth_no_regression(self, linear_experiment):
        analyzer = ExperimentAnalyzer(linear_experiment, use_embeddings=False)
        result = analyzer.compute_robustness()
        # Best depth is 2 (mean=0.65): t1=0.9 (up from 0.8), t2=0.4 (up from 0.0)
        # No regressions when comparing seed to BEST depth
        assert result["num_regressions"] == 0

    def test_score_variance(self, evolutionary_experiment):
        analyzer = ExperimentAnalyzer(evolutionary_experiment, use_embeddings=False)
        result = analyzer.compute_robustness()
        assert "score_variance_by_depth" in result
        assert 2 in result["score_variance_by_depth"]
        assert result["score_variance_by_depth"][2]["n_candidates"] == 2

    def test_dev_test_gap_missing(self, evolutionary_experiment):
        analyzer = ExperimentAnalyzer(evolutionary_experiment, use_embeddings=False)
        result = analyzer.compute_robustness()
        assert result["dev_test_gap"]["skipped"] == "no test_results.json"


class TestRunAll:
    def test_linear_run_all(self, linear_experiment):
        analyzer = ExperimentAnalyzer(linear_experiment, use_embeddings=False)
        results = analyzer.run_all()
        assert results["mode"] == "linear"
        assert "convergence" in results
        assert "emergent_abstraction" in results
        assert "archive_diversity" in results
        assert "failure_patterns" in results
        assert "robustness" in results

    def test_evolutionary_run_all(self, evolutionary_experiment):
        analyzer = ExperimentAnalyzer(evolutionary_experiment, use_embeddings=False)
        results = analyzer.run_all()
        assert results["mode"] == "evolutionary"
        assert "convergence" in results


class TestSaveResults:
    def test_save_creates_json(self, evolutionary_experiment):
        analyzer = ExperimentAnalyzer(evolutionary_experiment, use_embeddings=False)
        results = analyzer.run_all()
        out = analyzer.save_results(results)
        assert (out / "metrics.json").exists()
        loaded = json.loads((out / "metrics.json").read_text())
        assert "convergence" in loaded


class TestASTComplexity:
    def test_simple_code(self):
        result = ExperimentAnalyzer._compute_ast_complexity("x = 1")
        assert result["num_nodes"] > 0
        assert result["num_branches"] == 0

    def test_branching_code(self):
        code = "if x > 0:\n    y = 1\nelse:\n    y = 2"
        result = ExperimentAnalyzer._compute_ast_complexity(code)
        assert result["num_branches"] == 1

    def test_empty_code(self):
        result = ExperimentAnalyzer._compute_ast_complexity("")
        assert result["num_nodes"] == 0

    def test_syntax_error(self):
        result = ExperimentAnalyzer._compute_ast_complexity("def f(:")
        assert result["num_nodes"] == 0


class TestAbstractionScore:
    def test_specific_code(self):
        from meta_n.core.meta_layer import InjectedCode
        ic = InjectedCode(pre_process="import scipy", rationale="Fix scipy timeout error")
        score = ExperimentAnalyzer._compute_abstraction_score(ic)
        assert score < 0.5  # more specific

    def test_generic_code(self):
        from meta_n.core.meta_layer import InjectedCode
        ic = InjectedCode(
            pre_process="if 'scheduling' in task.task_id: strategy = 'classify'",
            rationale="Categorize tasks by category and apply structural strategy framework",
        )
        score = ExperimentAnalyzer._compute_abstraction_score(ic)
        assert score > 0.4  # more generic


# --- T3.7: code_library channel folded into metrics (was prompt-blind) ---


class TestCodeLibraryChannelT37:
    """A pure-helper layer must no longer be reported zero-complexity /
    default-abstraction — the helper VALUES are folded into the AST + text inputs.
    The default/off path (no helpers) stays byte-identical to the prompt-only logic.
    """

    def test_abstraction_reads_helper_values(self):
        from meta_n.core.meta_layer import InjectedCode
        ic = InjectedCode(
            source_depth=2,
            code_library={
                "r": "def r(task):\n    if isinstance(task, str):\n        return classify(task)"
            },
        )
        # Previously 0.5 (empty text → neutral); now the generic helper body scores 1.0.
        assert ExperimentAnalyzer._compute_abstraction_score(ic) == 1.0

    def test_classify_reads_helper_values(self):
        from meta_n.core.meta_layer import InjectedCode
        ic = InjectedCode(
            source_depth=2,
            code_library={
                "h": "def h():\n    try:\n        recover()\n    except Exception:\n        handle()"
            },
        )
        scores = ExperimentAnalyzer._classify_improvement_types(ic)
        assert scores["error_handling"] > 0.0

    def test_classify_reads_bash_helper_values(self):
        from meta_n.core.meta_layer import InjectedCode
        ic = InjectedCode(
            source_depth=2,
            code_library_bash={"g": "g() { detect && route; }"},
        )
        scores = ExperimentAnalyzer._classify_improvement_types(ic)
        assert scores["strategy_selection"] > 0.0

    def test_default_off_path_no_helpers_unchanged(self):
        from meta_n.core.meta_layer import InjectedCode
        # No code_library → identical to the prompt-only path (cf. test_specific_code).
        ic = InjectedCode(pre_process="import scipy", rationale="Fix scipy timeout error")
        assert ExperimentAnalyzer._compute_abstraction_score(ic) < 0.5

    def test_ast_complexity_includes_helper_values(self, linear_experiment):
        # Overwrite depth_2 with a PURE-HELPER layer (no pre_process).
        ic = {
            "source_depth": 2,
            "code_library": {"h": "def h(x):\n    if x:\n        return x\n    return 0"},
        }
        (linear_experiment / "depth_2" / "injected_code.json").write_text(json.dumps(ic))
        analyzer = ExperimentAnalyzer(linear_experiment, use_embeddings=False)
        result = analyzer.compute_emergent_abstraction()
        grad = result["best_chain"]["code_complexity_gradient"]
        # depth-2 layer (index 0) is pure-helper → was zero; now non-zero (T3.7).
        assert grad[0]["num_nodes"] > 0
        assert grad[0]["num_functions"] >= 1
