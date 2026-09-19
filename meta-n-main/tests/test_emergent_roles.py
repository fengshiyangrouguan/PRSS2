"""Tests for emergent role analysis (Experiment 6)."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from meta_n.analysis.emergent_roles import (
    EmergentRoleAnalyzer,
    ImprovementType,
    LayerRoleProfile,
    RoleAnalysisResult,
)
from meta_n.core.meta_layer import InjectedCode, TaskDescription


# --- Fixtures ---


@pytest.fixture
def sample_tasks():
    return [
        TaskDescription(task_id="t1", description="Create a file called hello.txt"),
        TaskDescription(task_id="t2", description="Compute 6 * 7 and write result"),
        TaskDescription(task_id="t3", description="List all files in /tmp"),
    ]


@pytest.fixture
def error_handling_code():
    """Code that clearly does error handling."""
    return InjectedCode(
        pre_process=None,
        rationale="Adding error handling to catch script failures and ensure set -e is present.",
        source_depth=2,
    )


@pytest.fixture
def strategy_selection_code():
    """Code that classifies tasks and selects strategies."""
    return InjectedCode(
        pre_process="""
if 'file' in task.description.lower():
    additional_context = 'Use touch or echo > for file creation.'
elif 'compute' in task.description.lower():
    additional_context = 'Use echo $((...)) for arithmetic.'
else:
    additional_context = 'Keep it simple.'
""",
        rationale="Classify task type and select appropriate strategy. Detect patterns in task descriptions to provide targeted guidance.",
        source_depth=4,
    )


@pytest.fixture
def decomposition_code():
    """Code that decomposes tasks."""
    return InjectedCode(
        pre_process="""
steps = task.description.split('and')
additional_context = 'Break this down into steps:\\n'
for i, step in enumerate(steps, 1):
    additional_context += f'Step {i}: {step.strip()}\\n'
""",
        rationale="Decompose complex tasks into sub-tasks by splitting on conjunctions.",
        source_depth=5,
    )


@pytest.fixture
def experiment_dir(error_handling_code, strategy_selection_code):
    """Create a fake experiment directory with saved artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        exp_dir = Path(tmpdir)

        # Config
        tasks_path = exp_dir / "tasks.json"
        tasks_path.write_text(json.dumps([
            {"task_id": "t1", "description": "Create hello.txt"},
            {"task_id": "t2", "description": "Compute 6 * 7"},
        ]))
        config = {"tasks_file": str(tasks_path)}
        (exp_dir / "config.json").write_text(json.dumps(config))

        # Depth 2
        d2 = exp_dir / "depth_2"
        d2.mkdir()
        (d2 / "injected_code.json").write_text(
            error_handling_code.model_dump_json(exclude={"raw_omega_response"})
        )
        (d2 / "omega_response.txt").write_text("Full omega response for depth 2...")

        # Depth 3
        d3 = exp_dir / "depth_3"
        d3.mkdir()
        (d3 / "injected_code.json").write_text(
            strategy_selection_code.model_dump_json(exclude={"raw_omega_response"})
        )

        yield str(exp_dir)


# --- Classification tests ---


class TestImprovementClassification:
    def test_error_handling_detection(self, error_handling_code):
        analyzer = EmergentRoleAnalyzer("/tmp/fake")
        scores = analyzer._classify_improvement_type(error_handling_code)

        assert scores["error_handling"] > 0.3
        # Should be top or near-top type
        top = max(scores, key=scores.get)
        assert top in ("error_handling", "control_flow")

    def test_strategy_selection_detection(self, strategy_selection_code):
        analyzer = EmergentRoleAnalyzer("/tmp/fake")
        scores = analyzer._classify_improvement_type(strategy_selection_code)

        assert scores["strategy_selection"] > 0.2
        assert scores["prompt_mod"] > 0.2

    def test_decomposition_detection(self, decomposition_code):
        analyzer = EmergentRoleAnalyzer("/tmp/fake")
        scores = analyzer._classify_improvement_type(decomposition_code)

        assert scores["decomposition"] > 0.2

    def test_empty_code(self):
        analyzer = EmergentRoleAnalyzer("/tmp/fake")
        code = InjectedCode(rationale="", source_depth=2)
        scores = analyzer._classify_improvement_type(code)
        # All scores should be 0
        assert all(v == 0.0 for v in scores.values())

    def test_top_types_threshold(self):
        analyzer = EmergentRoleAnalyzer("/tmp/fake")
        scores = {"utility": 0.8, "prompt_mod": 0.5, "retry": 0.1, "error_handling": 0.0,
                  "control_flow": 0.0, "decomposition": 0.0, "strategy_selection": 0.0}
        types = analyzer._top_types(scores, threshold=0.3)
        assert ImprovementType.UTILITY in types
        assert ImprovementType.PROMPT_MOD in types
        assert ImprovementType.RETRY not in types


# --- Abstraction scoring tests ---


class TestAbstractionScoring:
    def test_generic_code_scores_high(self, strategy_selection_code, sample_tasks):
        analyzer = EmergentRoleAnalyzer("/tmp/fake")
        score = analyzer._compute_abstraction_score(strategy_selection_code, sample_tasks)
        # Code uses task.description.lower(), classify, etc. — generic patterns
        assert score > 0.4

    def test_empty_code_is_neutral(self, sample_tasks):
        analyzer = EmergentRoleAnalyzer("/tmp/fake")
        code = InjectedCode(source_depth=2)
        score = analyzer._compute_abstraction_score(code, sample_tasks)
        assert score == 0.5

    def test_task_specific_scores_low(self):
        analyzer = EmergentRoleAnalyzer("/tmp/fake")
        tasks = [TaskDescription(task_id="t1", description="Create hello world file")]
        code = InjectedCode(
            pre_process="additional_context = 'Create hello world file exactly as described'",
            rationale="Specifically handle the hello world file task",
            source_depth=2,
        )
        score = analyzer._compute_abstraction_score(code, tasks)
        # Contains literal task description phrases → more specific
        assert score < 0.7


# --- Injection targeting tests ---


class TestInjectionTargeting:
    def test_concrete_at_low_depth(self):
        analyzer = EmergentRoleAnalyzer("/tmp/fake")
        profiles = [
            LayerRoleProfile(
                depth=2,
                improvement_types=[ImprovementType.ERROR_HANDLING],
            ),
        ]
        targeting = analyzer._compute_injection_targeting(profiles)
        # Error handling at depth 2 = correct targeting
        assert targeting[2] > 0.5

    def test_abstract_at_high_depth(self):
        analyzer = EmergentRoleAnalyzer("/tmp/fake")
        profiles = [
            LayerRoleProfile(depth=2, improvement_types=[ImprovementType.ERROR_HANDLING]),
            LayerRoleProfile(depth=3, improvement_types=[ImprovementType.ERROR_HANDLING]),
            LayerRoleProfile(depth=4, improvement_types=[ImprovementType.ERROR_HANDLING]),
            LayerRoleProfile(depth=5, improvement_types=[ImprovementType.STRATEGY_SELECTION]),
        ]
        targeting = analyzer._compute_injection_targeting(profiles)
        # Strategy selection at high depth = good
        assert targeting[5] > 0.5


# --- Role differentiation tests ---


class TestRoleDifferentiation:
    def test_different_embeddings_high_distance(self):
        analyzer = EmergentRoleAnalyzer("/tmp/fake")
        p1 = LayerRoleProfile(
            depth=2,
            combined_embedding=np.array([1.0, 0.0, 0.0]),
        )
        p2 = LayerRoleProfile(
            depth=3,
            combined_embedding=np.array([0.0, 1.0, 0.0]),
        )
        dists = analyzer._compute_role_differentiation([p1, p2])
        assert "2-3" in dists
        assert dists["2-3"] > 0.9  # orthogonal vectors → high distance

    def test_identical_embeddings_zero_distance(self):
        analyzer = EmergentRoleAnalyzer("/tmp/fake")
        emb = np.array([1.0, 1.0, 1.0])
        p1 = LayerRoleProfile(depth=2, combined_embedding=emb)
        p2 = LayerRoleProfile(depth=3, combined_embedding=emb)
        dists = analyzer._compute_role_differentiation([p1, p2])
        assert dists["2-3"] < 0.01

    def test_missing_embeddings_skipped(self):
        analyzer = EmergentRoleAnalyzer("/tmp/fake")
        p1 = LayerRoleProfile(depth=2, combined_embedding=None)
        p2 = LayerRoleProfile(depth=3, combined_embedding=np.array([1.0]))
        dists = analyzer._compute_role_differentiation([p1, p2])
        assert len(dists) == 0

    def test_single_layer_no_distances(self):
        analyzer = EmergentRoleAnalyzer("/tmp/fake")
        p1 = LayerRoleProfile(depth=2, combined_embedding=np.array([1.0]))
        dists = analyzer._compute_role_differentiation([p1])
        assert len(dists) == 0


# --- I/O tests ---


class TestAnalyzerIO:
    def test_load_injected_codes(self, experiment_dir):
        analyzer = EmergentRoleAnalyzer(experiment_dir)
        codes = analyzer._load_injected_codes()
        assert len(codes) == 2
        assert codes[0].source_depth == 2
        assert codes[1].source_depth == 4  # strategy_selection_code has source_depth=4
        # Raw omega response loaded from file
        assert codes[0].raw_omega_response == "Full omega response for depth 2..."

    def test_load_tasks(self, experiment_dir):
        analyzer = EmergentRoleAnalyzer(experiment_dir)
        tasks = analyzer._load_tasks()
        assert len(tasks) == 2

    def test_full_analysis_pipeline(self, experiment_dir):
        """End-to-end: analyze a fake experiment directory."""
        analyzer = EmergentRoleAnalyzer(experiment_dir)
        result = analyzer.analyze()

        assert len(result.layer_profiles) == 2
        assert result.abstraction_gradient is not None
        assert len(result.abstraction_gradient) == 2

        # First layer (error handling) should be classified
        p1 = result.layer_profiles[0]
        assert len(p1.improvement_type_scores) == 7  # all types scored
        assert p1.depth == 2

    def test_save_and_load_results(self, experiment_dir):
        analyzer = EmergentRoleAnalyzer(experiment_dir)
        result = analyzer.analyze()

        output_dir = str(Path(experiment_dir) / "analysis")
        analyzer.save_results(result, output_dir)

        # Verify files created
        assert (Path(output_dir) / "role_analysis.json").exists()

        # Verify JSON is valid
        with open(Path(output_dir) / "role_analysis.json") as f:
            data = json.load(f)
        assert "layer_profiles" in data
        assert "mean_differentiation_score" in data

    def test_empty_experiment_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = EmergentRoleAnalyzer(tmpdir)
            result = analyzer.analyze()
            assert result.layer_profiles == []
            assert result.mean_differentiation_score == 0.0


# --- Serialization tests ---


class TestSerialization:
    def test_layer_profile_to_dict(self):
        profile = LayerRoleProfile(
            depth=3,
            improvement_types=[ImprovementType.UTILITY, ImprovementType.PROMPT_MOD],
            improvement_type_scores={"utility": 0.8, "prompt_mod": 0.5},
            abstraction_score=0.7,
        )
        d = profile.to_dict()
        assert d["depth"] == 3
        assert d["improvement_types"] == ["utility", "prompt_mod"]
        assert d["has_embeddings"] is False

    def test_result_to_dict(self):
        result = RoleAnalysisResult(
            experiment_dir="/tmp/test",
            mean_differentiation_score=0.42,
            abstraction_gradient=[0.3, 0.7],
        )
        d = result.to_dict()
        assert d["mean_differentiation_score"] == 0.42
        assert json.dumps(d)  # should be JSON-serializable


# --- T3.6: code_library channel folded into analysis (was prompt-blind) ---


class TestCodeLibraryChannelT36:
    """A pure-helper layer (only ``code_library``, no pre_process/rationale) must
    no longer score zero — the helper VALUES are folded into text + AST + embeddings.
    The default/off path (no helpers) stays byte-identical to the prompt-only logic.
    """

    def test_pure_helper_layer_scores_utility_was_zero(self):
        analyzer = EmergentRoleAnalyzer("/tmp/fake")
        helper = (
            "def retry_run(cmd):\n"
            "    for _ in range(3):\n"
            "        try:\n"
            "            return run(cmd)\n"
            "        except Exception:\n"
            "            continue"
        )
        with_helper = InjectedCode(source_depth=2, code_library={"retry_run": helper})
        scores = analyzer._classify_improvement_type(with_helper)
        # Helper body has a def + loop + try/except → utility/error/retry now fire.
        assert scores["utility"] > 0.0
        assert scores["error_handling"] > 0.0
        assert scores["retry"] > 0.0

    def test_bash_helper_values_folded_into_text(self):
        analyzer = EmergentRoleAnalyzer("/tmp/fake")
        code = InjectedCode(
            source_depth=2,
            code_library_bash={"recover": "recover() { echo retry fallback; }"},
        )
        scores = analyzer._classify_improvement_type(code)
        # bash helper text contributes keyword hits (retry/fallback).
        assert scores["retry"] > 0.0

    def test_default_off_path_no_helpers_unchanged(self):
        # No code_library → behaviour identical to the prompt-only path: an empty
        # InjectedCode still scores all-zero (matches test_empty_code).
        analyzer = EmergentRoleAnalyzer("/tmp/fake")
        empty = InjectedCode(rationale="", source_depth=2)
        scores = analyzer._classify_improvement_type(empty)
        assert all(v == 0.0 for v in scores.values())

    def test_abstraction_reads_helper_values(self, sample_tasks):
        analyzer = EmergentRoleAnalyzer("/tmp/fake")
        code = InjectedCode(
            source_depth=2,
            code_library={
                "route": "def route(task):\n    if isinstance(task, str):\n        return classify(task)"
            },
        )
        score = analyzer._compute_abstraction_score(code, sample_tasks)
        # Previously 0.5 (empty all_code → neutral); now the generic helper body scores.
        assert score != 0.5

    def test_abstraction_default_off_path_neutral(self, sample_tasks):
        analyzer = EmergentRoleAnalyzer("/tmp/fake")
        score = analyzer._compute_abstraction_score(InjectedCode(source_depth=2), sample_tasks)
        assert score == 0.5  # no pre_process/rationale/helpers → unchanged neutral

    def test_embeddings_include_helper_values(self, monkeypatch):
        analyzer = EmergentRoleAnalyzer("/tmp/fake")
        captured: list[str] = []

        class _FakeModel:
            def encode(self, text):
                captured.append(text)
                return np.ones(4)

        monkeypatch.setattr(analyzer, "_get_embedding_model", lambda: _FakeModel())
        code = InjectedCode(
            source_depth=2,
            rationale="r",
            code_library={"h": "def h():\n    return 42  # SENTINEL_HELPER"},
        )
        analyzer._compute_embeddings([LayerRoleProfile(depth=2)], [code])
        # The code embedding input must contain the helper source (T3.6).
        assert any("SENTINEL_HELPER" in t for t in captured)
