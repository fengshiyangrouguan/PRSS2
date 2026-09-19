"""Telemetry-gated injection tuning tests (Step 5, 4.3b): per-benchmark
code_library_is_live flag + the prepend gate in both native solver paths."""

from unittest.mock import MagicMock

import pytest

from meta_n.core.agentic_solver import AgenticSolver
from meta_n.core.archive import Candidate
from meta_n.core.evolutionary_orchestrator import (
    EvolutionaryConfig,
    EvolutionaryOrchestrator,
)
from meta_n.integrations.arc_agi import ARCAGI2Adapter
from meta_n.integrations.benchmark import BenchmarkAdapter
from meta_n.integrations.co_bench import COBenchAdapter
from meta_n.integrations.openevolve import OpenEvolveBaseAdapter


class _DummyAdapter(BenchmarkAdapter):
    @property
    def name(self):
        return "dummy"

    def load_tasks(self, limit=None):
        return []

    async def evaluate(self, task, solution):
        raise NotImplementedError


def _orch(adapter):
    orch = EvolutionaryOrchestrator(
        llm_client=MagicMock(), executor=MagicMock(), omega=MagicMock(),
        config=EvolutionaryConfig(max_depth=3, parallel=1, gate_tasks=0),
        solver_language="python",
    )
    orch.adapter = adapter
    return orch


# --------------------------------------------------------------------------- #
# the per-benchmark flag (stateless descriptor)
# --------------------------------------------------------------------------- #

def test_flag_default_true():
    assert _DummyAdapter().code_library_is_live() is True


def test_co_bench_and_swe_demote():
    assert COBenchAdapter.code_library_is_live(None) is False
    try:
        from meta_n.integrations.swe_bench import SWEBenchVerifiedAdapter
    except Exception as e:
        pytest.skip(f"swe_bench import unavailable: {e}")
    assert SWEBenchVerifiedAdapter.code_library_is_live(None) is False


def test_live_families_keep_default():
    # ARC / AlphaEvolve / SR (openevolve) keep helpers — call-rate is high there.
    assert ARCAGI2Adapter.code_library_is_live(None) is True
    assert OpenEvolveBaseAdapter.code_library_is_live(None) is True


# --------------------------------------------------------------------------- #
# the prepend gate (both native paths)
# --------------------------------------------------------------------------- #

def test_agentic_solver_demotes_when_not_live(make_injected):
    ic = make_injected(code_library={"helper": "def helper():\n    return 1"})
    live = AgenticSolver(MagicMock(), MagicMock(), injected_codes=[ic], code_library_is_live=True)
    dead = AgenticSolver(MagicMock(), MagicMock(), injected_codes=[ic], code_library_is_live=False)
    assert "helper" in live.merged_py
    assert dead.merged_py == {}
    assert dead.merged_bash == live.merged_bash  # bash helpers unaffected


def test_native_metalayer_gate_demotes(make_injected):
    class _Dead:
        def code_library_is_live(self):
            return False

    orch = _orch(_Dead())
    cand = Candidate(candidate_id="c", depth=2,
                     injected_codes=[make_injected(code_library={"helper": "def helper():\n    return 1"})])
    solver = orch._build_solver_from_candidate(cand)
    assert solver.merged_code_library == {}   # demoted dead helpers


def test_native_metalayer_keeps_when_live(make_injected):
    class _Live:
        def code_library_is_live(self):
            return True

    orch = _orch(_Live())
    cand = Candidate(candidate_id="c", depth=2,
                     injected_codes=[make_injected(code_library={"helper": "def helper():\n    return 1"})])
    solver = orch._build_solver_from_candidate(cand)
    assert "helper" in solver.merged_code_library   # default/live path unchanged


def test_no_adapter_defaults_live(make_injected):
    orch = _orch(None)  # no adapter → default True (legacy behavior unchanged)
    assert orch._code_library_is_live() is True
    cand = Candidate(candidate_id="c", depth=2,
                     injected_codes=[make_injected(code_library={"helper": "def helper():\n    return 1"})])
    assert "helper" in orch._build_solver_from_candidate(cand).merged_code_library
