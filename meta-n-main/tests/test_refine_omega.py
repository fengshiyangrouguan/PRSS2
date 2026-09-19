"""Refinement regression tests for meta_n/core/omega.py (audit cluster C2).

Covers:
- F002: signed (negative) scores render in trace formatting; zero stays hidden
  (pins default-path byte-identity on [0,1] benchmarks).
- F015: the ``_format_delta_lines`` consolidation is byte-identical — the
  EXPECTED strings below were captured from the PRE-refactor implementation
  and must never change.
- F025: ``_select_representative_traces`` has no dead parameter.
- F014 (omega side): ``OmegaEngine._strip_library_prefix`` is the canonical
  ``meta_layer._strip_library_prefix_for_scan`` helper.
"""

import inspect

import pytest

from meta_n.core.llm_client import LLMClient, LLMConfig
from meta_n.core.meta_layer import (
    InjectedCode,
    Trace,
    _strip_library_prefix_for_scan,
)
from meta_n.core.omega import OmegaEngine


def _engine() -> OmegaEngine:
    return OmegaEngine(LLMClient(LLMConfig(api_key="test")))


def _t(
    task_id: str,
    score: float,
    *,
    success: bool = True,
    depth: int = 2,
    script: str = "echo solve",
    error_summary: str = "",
) -> Trace:
    return Trace(
        task_id=task_id,
        success=success,
        score=score,
        depth=depth,
        script=script,
        error_summary=error_summary,
    )


# ---------------------------------------------------------------------------
# F002 — negative scores must render; zero stays hidden (no-signal sentinel)
# ---------------------------------------------------------------------------


class TestNegativeScoreRendering:
    def test_negative_score_renders_in_raw_traces(self):
        out = _engine()._format_raw_traces([_t("sr_task", -0.7)])
        assert " score=-0.700" in out

    def test_negative_score_renders_in_inspiration(self):
        out = _engine()._format_inspiration([_t("sr_task", -0.7)])
        assert " score=-0.700" in out

    def test_zero_score_stays_hidden(self):
        """score=0.0 renders with NO score marker — pins the default-path
        ([0,1] benchmarks) byte-identity of the F002 fix."""
        engine = _engine()
        success_out = engine._format_raw_traces([_t("zero_pass", 0.0)])
        assert "[SUCCESS]" in success_out
        assert "score=" not in success_out
        failure_out = engine._format_raw_traces(
            [_t("zero_fail", 0.0, success=False, error_summary="boom")]
        )
        assert "[FAILURE" in failure_out
        assert "score=" not in failure_out
        inspiration_out = engine._format_inspiration([_t("zero_insp", 0.0)])
        assert "score=" not in inspiration_out


# ---------------------------------------------------------------------------
# F015 — delta-section rendering byte-identity (characterization)
# ---------------------------------------------------------------------------


def _case_mix():
    """Improved + regressed (one with archive-best suffix) + unchanged."""
    traces = [
        _t("imp_big", 0.90),
        _t("imp_small", 0.60),
        _t("reg_arch", 0.20, success=False, error_summary="execution timed out"),
        _t("reg_plain", 0.40, success=False,
           error_summary="constraint violated: capacity"),
        _t("unc_exact", 0.50),
        _t("unc_close", 0.30),
    ]
    prev = {
        "imp_big": 0.5,
        "imp_small": 0.55,
        "reg_arch": 0.8,
        "reg_plain": 0.5,
        "unc_exact": 0.5,
        "unc_close": 0.305,
    }
    archive = {"reg_arch": 0.85}
    stack = [InjectedCode(
        rationale="Guard the packing solver against timeouts.", source_depth=2,
    )]
    return traces, prev, archive, stack


def _case_truncation():
    """7 improved + 7 regressed: Section C truncates to 5 rows + '... and 2
    more'; the comparison path lists all rows."""
    traces, prev = [], {}
    for k in range(1, 8):
        tid = f"imp_{k}"
        traces.append(_t(tid, round(0.1 + 0.02 * k, 3)))
        prev[tid] = 0.1
    for k in range(1, 8):
        tid = f"reg_{k}"
        traces.append(_t(tid, round(0.9 - 0.02 * k, 3), success=False,
                         error_summary="execution timed out"))
        prev[tid] = 0.9
    archive = {"reg_2": 0.95, "reg_5": 0.93}
    return traces, prev, archive, []


def _case_no_shared():
    """previous_scores non-empty but disjoint from current tasks: comparison
    path emits NO net-effect line; Section C emits the N/A branch."""
    traces = [
        _t("new_pass", 0.70),
        _t("new_zero", 0.0, success=False,
           error_summary="IndexError: list index out of range"),
    ]
    prev = {"phantom_old": 0.5}
    return traces, prev, None, []


def _case_no_unchanged():
    """No unchanged tasks: line absent on the comparison path, 'Tasks
    unchanged: 0' on Section C."""
    traces = [
        _t("up", 0.9),
        _t("down", 0.1, success=False, error_summary="execution timed out"),
    ]
    prev = {"up": 0.5, "down": 0.6}
    return traces, prev, None, []


CASES = {
    "mix": _case_mix,
    "truncation": _case_truncation,
    "no_shared": _case_no_shared,
    "no_unchanged": _case_no_unchanged,
}

# Captured from the pre-F015 implementation (HEAD before the refactor);
# these strings are the byte-identity contract — do NOT regenerate them.
EXPECTED: dict[str, dict[str, str]] = {
    'mix': {
        'comparison': '\n## Baseline Comparison\n\nTasks that IMPROVED (2):\n  imp_big: 0.500 → 0.900 (+0.400)\n  imp_small: 0.550 → 0.600 (+0.050)\nTasks that REGRESSED (2):\n  reg_arch: 0.800 → 0.200 (-0.600) [archive best: 0.850]\n  reg_plain: 0.500 → 0.400 (-0.100)\nTasks unchanged: 2\nNet effect: -0.043 mean score change',
        'summary': '## Task Category Performance\n\n- **other** (6 tasks): 4/6 pass, mean=0.48, key issue: Timeout\n\n## Failure Pattern Distribution\n\n- **Timeout**: 1 tasks (50%) — e.g., "execution timed out"\n- **Constraint violation**: 1 tasks (50%) — e.g., "constraint violated: capacity"\n\n## Previous Layer Effectiveness\n\nPrevious layer rationale: "Guard the packing solver against timeouts...."\n\nTasks that IMPROVED (2):\n  imp_big: 0.500 → 0.900 (+0.400)\n  imp_small: 0.550 → 0.600 (+0.050)\nTasks that REGRESSED (2):\n  reg_arch: 0.800 → 0.200 (-0.600) [archive best: 0.850]\n  reg_plain: 0.500 → 0.400 (-0.100)\nTasks unchanged: 2\nNet effect: -0.043 mean score change (over 6 shared tasks)\n\n## Representative Traces (2 of 6)\n\n--- Task: reg_arch [FAILURE score=0.200 Timeout] ---\nScript:\n```bash\necho solve\n```\nError: execution timed out\n\n--- Task: imp_big [SUCCESS score=0.900] ---\nScript:\n```bash\necho solve\n```\n',
    },
    'no_shared': {
        'comparison': '\n## Baseline Comparison\n\nTasks that IMPROVED (1):\n  new_pass: 0.000 → 0.700 (+0.700)\nTasks unchanged: 1',
        'summary': '## Task Category Performance\n\n- **other** (2 tasks): 1/2 pass, mean=0.35, key issue: Indexing error\n\n## Failure Pattern Distribution\n\n- **Indexing error**: 1 tasks (100%) — e.g., "IndexError: list index out of range"\n\n## Previous Layer Effectiveness\n\nTasks that IMPROVED (1):\n  new_pass: 0.000 → 0.700 (+0.700)\nTasks unchanged: 1\nNet effect: N/A (no shared tasks between layers)\n\n## Representative Traces (2 of 2)\n\n--- Task: new_zero [FAILURE Indexing error] ---\nScript:\n```bash\necho solve\n```\nError: IndexError: list index out of range\n\n--- Task: new_pass [SUCCESS score=0.700] ---\nScript:\n```bash\necho solve\n```\n',
    },
    'no_unchanged': {
        'comparison': '\n## Baseline Comparison\n\nTasks that IMPROVED (1):\n  up: 0.500 → 0.900 (+0.400)\nTasks that REGRESSED (1):\n  down: 0.600 → 0.100 (-0.500)\nNet effect: -0.050 mean score change',
        'summary': '## Task Category Performance\n\n- **other** (2 tasks): 1/2 pass, mean=0.50, key issue: Timeout\n\n## Failure Pattern Distribution\n\n- **Timeout**: 1 tasks (100%) — e.g., "execution timed out"\n\n## Previous Layer Effectiveness\n\nTasks that IMPROVED (1):\n  up: 0.500 → 0.900 (+0.400)\nTasks that REGRESSED (1):\n  down: 0.600 → 0.100 (-0.500)\nTasks unchanged: 0\nNet effect: -0.050 mean score change (over 2 shared tasks)\n\n## Representative Traces (2 of 2)\n\n--- Task: down [FAILURE score=0.100 Timeout] ---\nScript:\n```bash\necho solve\n```\nError: execution timed out\n\n--- Task: up [SUCCESS score=0.900] ---\nScript:\n```bash\necho solve\n```\n',
    },
    'truncation': {
        'comparison': '\n## Baseline Comparison\n\nTasks that IMPROVED (7):\n  imp_7: 0.100 → 0.240 (+0.140)\n  imp_6: 0.100 → 0.220 (+0.120)\n  imp_5: 0.100 → 0.200 (+0.100)\n  imp_4: 0.100 → 0.180 (+0.080)\n  imp_3: 0.100 → 0.160 (+0.060)\n  imp_2: 0.100 → 0.140 (+0.040)\n  imp_1: 0.100 → 0.120 (+0.020)\nTasks that REGRESSED (7):\n  reg_7: 0.900 → 0.760 (-0.140)\n  reg_6: 0.900 → 0.780 (-0.120)\n  reg_5: 0.900 → 0.800 (-0.100) [archive best: 0.930]\n  reg_4: 0.900 → 0.820 (-0.080)\n  reg_3: 0.900 → 0.840 (-0.060)\n  reg_2: 0.900 → 0.860 (-0.040) [archive best: 0.950]\n  reg_1: 0.900 → 0.880 (-0.020)\nNet effect: +0.000 mean score change',
        'summary': '## Task Category Performance\n\n- **other** (14 tasks): 7/14 pass, mean=0.50, key issue: Timeout\n\n## Failure Pattern Distribution\n\n- **Timeout**: 7 tasks (100%) — e.g., "execution timed out"\n\n## Previous Layer Effectiveness\n\nTasks that IMPROVED (7):\n  imp_7: 0.100 → 0.240 (+0.140)\n  imp_6: 0.100 → 0.220 (+0.120)\n  imp_5: 0.100 → 0.200 (+0.100)\n  imp_4: 0.100 → 0.180 (+0.080)\n  imp_3: 0.100 → 0.160 (+0.060)\n  ... and 2 more\nTasks that REGRESSED (7):\n  reg_7: 0.900 → 0.760 (-0.140)\n  reg_6: 0.900 → 0.780 (-0.120)\n  reg_5: 0.900 → 0.800 (-0.100) [archive best: 0.930]\n  reg_4: 0.900 → 0.820 (-0.080)\n  reg_3: 0.900 → 0.840 (-0.060)\n  ... and 2 more\nTasks unchanged: 0\nNet effect: +0.000 mean score change (over 14 shared tasks)\n\n## Representative Traces (3 of 14)\n\n--- Task: reg_1 [FAILURE score=0.880 Timeout] ---\nScript:\n```bash\necho solve\n```\nError: execution timed out\n\n--- Task: reg_7 [FAILURE score=0.760 Timeout] ---\nScript:\n```bash\necho solve\n```\nError: execution timed out\n\n--- Task: imp_7 [SUCCESS score=0.240] ---\nScript:\n```bash\necho solve\n```\n',
    },
}


def _render(case: str) -> tuple[str, str]:
    traces, prev, archive, stack = CASES[case]()
    engine = _engine()
    cmp_out = engine._format_score_comparison(traces, prev, archive)
    sum_out = engine._summarize_traces(
        traces, prev, stack, archive_best_scores=archive,
    )
    return cmp_out, sum_out


class TestDeltaSectionByteIdentity:
    @pytest.mark.parametrize("case", sorted(CASES))
    def test_format_score_comparison_byte_identical(self, case):
        cmp_out, _ = _render(case)
        assert cmp_out == EXPECTED[case]["comparison"]

    @pytest.mark.parametrize("case", sorted(CASES))
    def test_summarize_traces_byte_identical(self, case):
        _, sum_out = _render(case)
        assert sum_out == EXPECTED[case]["summary"]


# ---------------------------------------------------------------------------
# F025 — dead parameter removed from _select_representative_traces
# ---------------------------------------------------------------------------


def test_select_representative_signature_has_no_dead_param():
    sig = inspect.signature(OmegaEngine._select_representative_traces)
    params = [p for p in sig.parameters if p != "self"]
    assert params == ["current_traces", "previous_scores", "max_count", "focus_task"]


# ---------------------------------------------------------------------------
# F014 (omega side) — single source of truth for the library-prefix strip
# ---------------------------------------------------------------------------


def test_strip_library_prefix_is_canonical_meta_layer_helper():
    assert OmegaEngine._strip_library_prefix is _strip_library_prefix_for_scan
    marked = "# helpers\n# --- end injected code library ---\n\necho hi"
    assert OmegaEngine._strip_library_prefix(marked) == "echo hi"
    assert OmegaEngine._strip_library_prefix("echo hi") == "echo hi"
