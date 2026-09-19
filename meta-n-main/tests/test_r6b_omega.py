"""§6b regression tests for meta_n/core/omega.py (F020).

F020 — scale-aware G1 buckets and near-zero thresholds:
- ``_scale_epsilon``: returns exactly ``base`` on the unit scale ([0,1]) and
  ``base * observed_range`` on non-unit scales.
- ``_format_delta_lines`` / ``_format_score_comparison``: improved/regressed
  classification uses the scale-aware epsilon; unit-scale output stays
  byte-identical (literal pinned at implementation time from the
  pre-F020 implementation).
- ``_novelty_directive``: plateau threshold is 1% of the observed spread on
  non-unit scales; unit behavior unchanged.
- G1 headline: the fixed [0,1] bucket line renders VERBATIM on the unit
  scale and is replaced by a scale-free min/median/max spread on non-unit
  scales (same treatment as the established G2 ``unit_scale`` fallback).
"""

import math
from unittest.mock import MagicMock

from meta_n.core.llm_client import LLMClient, LLMConfig
from meta_n.core.meta_layer import TaskDescription, Trace
from meta_n.core.omega import OmegaEngine


def _engine() -> OmegaEngine:
    return OmegaEngine(LLMClient(LLMConfig(api_key="test")))


def _t(tid: str, score: float, *, success: bool = True) -> Trace:
    return Trace(
        task_id=tid, success=success, score=score, depth=2, script="echo solve",
    )


# ---------------------------------------------------------------------------
# _scale_epsilon
# ---------------------------------------------------------------------------


class TestScaleEpsilon:
    def test_unit_scale_returns_base_exactly(self):
        """Identity, not approx: the unit path returns the ``base`` literal
        untouched (no arithmetic) — this is the byte-identity guarantee."""
        eps = OmegaEngine._scale_epsilon({"a": 0.0, "b": 0.5}, {"c": 1.0})
        assert eps == 0.01

    def test_empty_and_none_maps_return_base(self):
        assert OmegaEngine._scale_epsilon({}, None) == 0.01
        assert OmegaEngine._scale_epsilon() == 0.01

    def test_non_finite_values_ignored(self):
        eps = OmegaEngine._scale_epsilon(
            {"a": 0.2, "nan": math.nan, "inf": math.inf, "b": 0.9}
        )
        assert eps == 0.01

    def test_non_unit_scale_is_base_fraction_of_range(self):
        eps = OmegaEngine._scale_epsilon({"a": -12.4, "b": 45.0})
        assert eps == 0.01 * (45.0 - (-12.4))

    def test_archive_best_can_push_scale_non_unit(self):
        """Scale is a benchmark property: an otherwise-unit current map goes
        non-unit when any other map (e.g. archive best) is out of [0,1]."""
        eps = OmegaEngine._scale_epsilon({"a": 0.2, "b": 0.9}, {"a": 50.0})
        assert eps == 0.01 * (50.0 - 0.2)

    def test_custom_base(self):
        assert OmegaEngine._scale_epsilon({"a": 0.5}, base=0.02) == 0.02


# ---------------------------------------------------------------------------
# _format_delta_lines / _format_score_comparison
# ---------------------------------------------------------------------------


class TestDeltaLinesScaleAware:
    def test_non_unit_small_move_is_unchanged_large_move_improves(self):
        """Spread ~200 → eps ~2: a +0.5 move is noise (unchanged), +5 is a
        real improvement."""
        current = {"a": 100.0, "b": 300.0, "c": 150.0}
        previous = {"a": 99.5, "b": 295.0, "c": 150.0}
        lines = "\n".join(
            _engine()._format_delta_lines(current, previous, None)
        )
        assert "Tasks that IMPROVED (1):" in lines
        assert "b: 295.000 → 300.000" in lines
        assert "a:" not in lines  # +0.5 on a 200-spread scale → unchanged
        assert "Tasks unchanged: 2" in lines

    def test_non_unit_small_regression_is_unchanged(self):
        current = {"a": 99.5, "b": 300.0}
        previous = {"a": 100.0, "b": 300.0}
        lines = "\n".join(
            _engine()._format_delta_lines(current, previous, None)
        )
        assert "REGRESSED" not in lines
        assert "Tasks unchanged: 2" in lines

    def test_unit_scale_comparison_byte_identical(self):
        """Belt-and-braces alongside TestDeltaSectionByteIdentity: the
        EXPECTED literal was captured from the pre-F020 implementation and
        must never change on unit-scale inputs."""
        traces = [
            _t("imp", 0.90),
            _t("reg", 0.20, success=False),
            _t("unc", 0.50),
            _t("edge", 0.505),  # +0.005 → inside ±0.01 → unchanged
        ]
        prev = {"imp": 0.5, "reg": 0.8, "unc": 0.5, "edge": 0.5}
        archive = {"reg": 0.85, "imp": 0.95}
        out = _engine()._format_score_comparison(traces, prev, archive)
        assert out == (
            "\n## Baseline Comparison\n"
            "\n"
            "Tasks that IMPROVED (1):\n"
            "  imp: 0.500 → 0.900 (+0.400)\n"
            "Tasks that REGRESSED (1):\n"
            "  reg: 0.800 → 0.200 (-0.600) [archive best: 0.850]\n"
            "Tasks unchanged: 2\n"
            "Net effect: -0.049 mean score change"
        )


# ---------------------------------------------------------------------------
# _novelty_directive
# ---------------------------------------------------------------------------


class TestNoveltyDirectiveScaleAware:
    def test_non_unit_fires_on_relative_plateau(self):
        """+1.0 mean delta on a ~200 spread (eps ~2) is a plateau."""
        prev = {"a": 99.0, "b": 299.0}
        cur = {"a": 100.0, "b": 300.0}
        assert "PLATEAU RISK" in _engine()._novelty_directive(cur, prev, depth=3)

    def test_non_unit_silent_on_relative_gain(self):
        """+50 mean delta on a ~250 spread is a real gain — no directive."""
        prev = {"a": 50.0, "b": 250.0}
        cur = {"a": 100.0, "b": 300.0}
        assert _engine()._novelty_directive(cur, prev, depth=3) == ""

    def test_unit_scale_behavior_unchanged(self):
        prev = {"a": 0.50, "b": 0.60}
        gain = {"a": 0.52, "b": 0.62}   # +0.02 > 0.01 → silent
        flat = {"a": 0.505, "b": 0.605}  # +0.005 ≤ 0.01 → fires
        eng = _engine()
        assert eng._novelty_directive(gain, prev, depth=3) == ""
        assert "PLATEAU RISK" in eng._novelty_directive(flat, prev, depth=3)


# ---------------------------------------------------------------------------
# G1 objective headline — scale-aware spread line
# ---------------------------------------------------------------------------


_SR_SCORES = {"a": -12.4, "b": -3.1, "c": 0.8, "d": 45.0}


class TestG1SpreadLine:
    def test_unit_scale_bucket_line_verbatim(self):
        """The unit-scale G1 headline is byte-identical to the pre-F020
        implementation (literal pinned at implementation time)."""
        cur = {"imp": 0.90, "reg": 0.20, "unc": 0.50, "edge": 0.505}
        archive = {"reg": 0.85, "imp": 0.95}
        summary, headroom = _engine()._format_objective_and_headroom(
            [], cur, archive, 0.75,
        )
        assert summary == (
            "Current mean score: 0.526  ← PRIMARY OBJECTIVE: raise the MEAN "
            "CONTINUOUS SCORE (not just the pass count)\n"
            "Score spread (4 tasks): [0.0-0.3]x1  [0.3-0.7]x2  [0.7-1.0]x1\n"
            "pass@1 (binary success rate): 75.0%\n"
        )
        # The observed/unit_scale hoist must not change the G2 table either.
        assert headroom == (
            "## Opportunity — per-task headroom to best-known "
            "(largest gap first)\n"
            "Target the largest-gap tasks. Tasks with a small/zero gap are "
            "near their ceiling — do NOT regress them.\n"
            "\n"
            "| task | now | best-known | gap |\n"
            "|------|-----|-----------|-----|\n"
            "| reg | 0.200 | 0.850 | +0.650 |\n"
            "| imp | 0.900 | 0.950 | +0.050 |\n"
            "| unc | 0.500 | 0.500 | +0.000 |\n"
            "| edge | 0.505 | 0.505 | +0.000 |\n"
        )

    def test_non_unit_scores_render_scale_free_spread(self):
        summary, _ = _engine()._format_objective_and_headroom(
            [], _SR_SCORES, None, 0.5,
        )
        assert (
            "Score spread (4 tasks): min=-12.400  median=-1.150  max=45.000\n"
            in summary
        )
        assert "[0.3-0.7]" not in summary

    def test_build_prompt_non_unit_has_no_bucket_text(self):
        eng = OmegaEngine(llm_client=MagicMock())
        traces = [
            _t("a", -12.4, success=False),
            _t("b", -3.1, success=False),
            _t("c", 0.8),
            _t("d", 45.0),
        ]
        tasks = [
            TaskDescription(task_id=k, description=f"task {k}", metadata={})
            for k in _SR_SCORES
        ]
        prompt = eng._build_prompt(
            traces, [], tasks, 2,
            current_scores=_SR_SCORES,
            solver_language="python",
        )
        assert "[0.3-0.7]" not in prompt
        assert "min=-12.400" in prompt

    def test_build_prompt_unit_scale_keeps_bucket_text(self):
        eng = OmegaEngine(llm_client=MagicMock())
        cur = {"a": 0.1, "b": 0.5, "c": 0.9}
        traces = [_t(k, v, success=v >= 0.5) for k, v in cur.items()]
        tasks = [
            TaskDescription(task_id=k, description=f"task {k}", metadata={})
            for k in cur
        ]
        prompt = eng._build_prompt(
            traces, [], tasks, 2,
            current_scores=cur,
            solver_language="python",
        )
        assert "[0.0-0.3]x1  [0.3-0.7]x1  [0.7-1.0]x1" in prompt
