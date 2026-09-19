"""§6b F145 — role_analysis.json schema v2: ``_exp6``-suffixed metric keys.

The Exp-6 heuristics in ``emergent_roles`` and the primary-pipeline heuristics
in ``metrics`` are deliberately different algorithms whose numeric outputs are
NOT comparable. Schema v2 renames the Exp-6 serialized keys with an ``_exp6``
suffix (values unchanged) and stamps ``role_analysis_schema_version``, so the
two artifacts can no longer be read as sharing a metric namespace. These tests
pin the rename, the version stamp, the name-only guarantee, and the intentional
heuristic divergence itself.
"""

import json

from meta_n.analysis.emergent_roles import (
    ROLE_ANALYSIS_SCHEMA_VERSION,
    EmergentRoleAnalyzer,
    ImprovementType,
    LayerRoleProfile,
    RoleAnalysisResult,
)
from meta_n.analysis.metrics import ExperimentAnalyzer
from meta_n.core.meta_layer import InjectedCode, TaskDescription


def _profile() -> LayerRoleProfile:
    return LayerRoleProfile(
        depth=3,
        improvement_types=[ImprovementType.UTILITY, ImprovementType.PROMPT_MOD],
        improvement_type_scores={"utility": 0.8, "prompt_mod": 0.5},
        abstraction_score=0.7,
        referenced_depths=[2],
    )


def _result() -> RoleAnalysisResult:
    return RoleAnalysisResult(
        experiment_dir="/tmp/test",
        layer_profiles=[_profile()],
        mean_differentiation_score=0.42,
        abstraction_gradient=[0.3, 0.7],
        injection_targeting={3: 1.0},
    )


class TestExp6SerializedKeys:
    def test_exp6_serialized_keys_renamed_and_versioned(self):
        pd = _profile().to_dict()
        assert "abstraction_score_exp6" in pd
        assert "improvement_type_scores_exp6" in pd
        assert "abstraction_score" not in pd
        assert "improvement_type_scores" not in pd
        # Label list keeps its name (metrics.json's counterpart is "top_types").
        assert pd["improvement_types"] == ["utility", "prompt_mod"]

        rd = _result().to_dict()
        assert rd["role_analysis_schema_version"] == ROLE_ANALYSIS_SCHEMA_VERSION == 2
        # Version key first, so a reader sees the schema before the payload.
        assert next(iter(rd)) == "role_analysis_schema_version"
        assert "abstraction_gradient_exp6" in rd
        assert "abstraction_gradient" not in rd

    def test_saved_role_analysis_uses_exp6_keys(self, tmp_path):
        analyzer = EmergentRoleAnalyzer(str(tmp_path))
        analyzer.save_results(_result(), str(tmp_path / "analysis"))

        with open(tmp_path / "analysis" / "role_analysis.json") as f:
            data = json.load(f)
        assert data["role_analysis_schema_version"] == 2
        assert data["abstraction_gradient_exp6"] == [0.3, 0.7]
        assert "abstraction_gradient" not in data
        assert "abstraction_score" not in data["layer_profiles"][0]

    def test_rename_is_name_only(self):
        """v1 → v2 changed key NAMES only — every value equals its attribute."""
        profile, result = _profile(), _result()
        pd = profile.to_dict()
        assert pd["abstraction_score_exp6"] == profile.abstraction_score
        assert pd["improvement_type_scores_exp6"] == profile.improvement_type_scores

        rd = result.to_dict()
        assert rd["abstraction_gradient_exp6"] == result.abstraction_gradient


class TestHeuristicDivergenceContract:
    def test_heuristics_deliberately_diverge_from_metrics(self):
        """Pin the INTENTIONAL divergence between the two heuristic copies.

        The same injection scores 0.0 (all-specific patterns) under metrics'
        abstraction heuristic and 0.5 (no pattern hits at all) under Exp-6's;
        the type-score dicts have 5 vs 7 keys. This is why the serialized keys
        are suffix-separated — if a future change silently unifies the
        algorithms (making the ``_exp6`` suffix a lie), this test trips.
        """
        ic = InjectedCode(
            pre_process="import scipy\n# timeout error handling", source_depth=2
        )
        tasks = [
            TaskDescription(
                task_id="dummy_task_001",
                description="Sort the array of integers ascending please",
            )
        ]
        exp6 = EmergentRoleAnalyzer("/tmp/fake")

        metrics_score = ExperimentAnalyzer._compute_abstraction_score(ic)
        exp6_score = exp6._compute_abstraction_score(ic, tasks)
        assert metrics_score == 0.0
        assert exp6_score == 0.5

        metrics_types = ExperimentAnalyzer._classify_improvement_types(ic)
        exp6_types = exp6._classify_improvement_type(ic)
        assert len(metrics_types) == 5
        assert len(exp6_types) == 7
        assert set(exp6_types) == {t.value for t in ImprovementType}
