"""P1c: --seed-code-library — inject a known-good helper into gen0 before Ω.

Loader validates each source through utils/safety (fail-fast). When set, the
gen0 seed becomes a depth-2 candidate whose solver routes through the candidate
builder (a MetaLayer staging the seeded library). Default None ⇒ byte-identical
(gen0 keeps injected_codes=[] and routes the bare Layer1Solver).
"""
import json

import pytest
from unittest.mock import MagicMock

from meta_n.core.archive import Candidate
from meta_n.core.evolutionary_orchestrator import (
    EvolutionaryConfig,
    EvolutionaryOrchestrator,
)
from meta_n.core.meta_layer import InjectedCode, MetaLayer
from meta_n.core.solver import Layer1Solver
from meta_n.main import load_seed_code_library

GOOD_HELPER = "def solve_crew_scheduling(N, K, time_limit, tasks, arcs):\n    return {}\n"


# --- Loader ----------------------------------------------------------------

def test_loader_parses_mapping(tmp_path):
    p = tmp_path / "crew.json"
    p.write_text(json.dumps({"solve_crew_scheduling": GOOD_HELPER}))
    loaded = load_seed_code_library(str(p))
    assert loaded == {"solve_crew_scheduling": GOOD_HELPER}


def test_loader_none_returns_none():
    assert load_seed_code_library(None) is None


def test_loader_rejects_unsafe_source(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"evil": "import os\nos.system('rm -rf /')\n"}))
    with pytest.raises(ValueError):
        load_seed_code_library(str(p))


def test_loader_rejects_non_object(tmp_path):
    p = tmp_path / "list.json"
    p.write_text(json.dumps(["not", "a", "mapping"]))
    with pytest.raises(ValueError):
        load_seed_code_library(str(p))


# --- Config + gen0 routing -------------------------------------------------

def _orch(**cfg) -> EvolutionaryOrchestrator:
    d = dict(max_depth=20, parallel=1, patience=8, gate_tasks=0,
             beam_width=1, beam_candidates=1)
    d.update(cfg)
    return EvolutionaryOrchestrator(
        llm_client=MagicMock(), executor=MagicMock(), omega=MagicMock(),
        config=EvolutionaryConfig(**d), solver_language="python",
    )


def test_config_default_none():
    assert EvolutionaryConfig().seed_code_library is None


def test_seeded_candidate_routes_through_metalayer():
    seeded = {"solve_crew_scheduling": GOOD_HELPER}
    orch = _orch(seed_code_library=seeded)
    # Build the gen0 seed exactly as run() does when seeding is on.
    seed = Candidate(
        candidate_id="gen0_seed", iteration=0, depth=2,
        injected_codes=[InjectedCode(code_library=dict(seeded), source_depth=0)],
    )
    assert seed.injected_codes[0].code_library == seeded
    solver = orch._build_solver_from_candidate(seed)
    assert isinstance(solver, MetaLayer)
    # The seeded helper is staged on the outermost layer (None adapter ⇒ live).
    assert "solve_crew_scheduling" in solver.merged_code_library


def test_empty_seed_routes_bare_layer1solver():
    orch = _orch()  # default: seed_code_library None
    seed = Candidate(candidate_id="gen0_seed", iteration=0, depth=1, injected_codes=[])
    solver = orch._build_solver_from_candidate(seed)
    # No injected codes ⇒ the builder returns the bare Layer1Solver (HEAD path).
    assert isinstance(solver, Layer1Solver)


# --- R3-A: agentic branch mirrors native demote (seed survives) ------------

def _demoting_orch(use_agentic, seeded):
    adapter = MagicMock(); adapter.code_library_is_live.return_value = False
    executor = MagicMock(); executor.adapter = adapter
    cfg = dict(max_depth=20, parallel=1, patience=8, gate_tasks=0,
               beam_width=1, beam_candidates=1, use_agentic=use_agentic,
               seed_code_library=dict(seeded))  # force_code_library_live stays default False
    return EvolutionaryOrchestrator(llm_client=MagicMock(), executor=executor,
               omega=MagicMock(), config=EvolutionaryConfig(**cfg), solver_language="python")


def test_agentic_seed_survives_demoting_family():
    seeded = {"solve_crew_scheduling": GOOD_HELPER}
    orch = _demoting_orch(use_agentic=True, seeded=seeded)
    seed = Candidate(candidate_id="gen0_seed", iteration=0, depth=2,
        injected_codes=[InjectedCode(code_library=dict(seeded), source_depth=0)])
    solver = orch._build_solver_from_candidate(seed)
    # AgenticSolver on a demoting family must keep the exempt source_depth==0 seed
    assert solver.merged_py == seeded          # PRE-FIX: {} (bug)


def test_agentic_seed_family_still_zeroes_nonseed_helper():
    seeded = {"solve_crew_scheduling": GOOD_HELPER}
    orch = _demoting_orch(use_agentic=True, seeded=seeded)
    seed = Candidate(candidate_id="c", iteration=0, depth=3, injected_codes=[
        InjectedCode(code_library=dict(seeded), source_depth=0),
        InjectedCode(code_library={"omega_h": "def omega_h():\n    return 1\n"}, source_depth=2)])
    solver = orch._build_solver_from_candidate(seed)
    assert "solve_crew_scheduling" in solver.merged_py   # seed rides through
    assert "omega_h" not in solver.merged_py             # non-seed still demoted


def test_native_agentic_seed_parity_on_demoting_family():
    # The exempt source_depth==0 seed must survive the demote on BOTH the native
    # MetaLayer path and the agentic path (R3-A closes the parity gap).
    seeded = {"solve_crew_scheduling": GOOD_HELPER}
    seed = Candidate(candidate_id="gen0_seed", iteration=0, depth=2,
        injected_codes=[InjectedCode(code_library=dict(seeded), source_depth=0)])
    native = _demoting_orch(use_agentic=False, seeded=seeded)._build_solver_from_candidate(seed)
    agentic = _demoting_orch(use_agentic=True, seeded=seeded)._build_solver_from_candidate(seed)
    assert "solve_crew_scheduling" in native.merged_code_library
    assert "solve_crew_scheduling" in agentic.merged_py
