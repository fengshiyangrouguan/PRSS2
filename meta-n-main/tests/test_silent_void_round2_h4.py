"""Round-2 silent-void audit — GROUP H4-cross-seams (R2-CS-1/2/3).

All three fixes are scoped to NON-default flags (verified_code / within-layer
refine / repropagation / deploy_verified_code) or the always-on S0.2 adoption
telemetry on the LIVE-helper path. The DEFAULT Omega / CO-Bench / [0,1]-scale
path stays byte-identical (covered by the Stage-2/3 goldens); these tests prove
the NEW behavior and that the unaffected path is unchanged.

R2-CS-1 · evolutionary_orchestrator.py:2374/2387 (refine) + 2649/2709 (reprop)
    Under ``verified_code``, a refined / re-propagated trace that re-derived a
    verified helper inline (advertised but never called) must NOT bank a
    per-task-best win the breed path would have barred. Default OFF ⇒ unbarred.

R2-CS-2 · meta_layer.py:populate_adoption_fields
    ``utilities_available`` must equal what the solver was ACTUALLY shown:
    Python helpers failing ``validate_library_function(for_advertising=True)``
    are excluded (they were never advertised/staged).

R2-CS-3 · meta_layer.py:_maybe_deploy_verified_helper
    The deploy wrapper must target the first STAGEABLE verified helper (mirrors
    ``prepend_python_library``'s validate gate), not ``sorted(library)[0]`` blindly
    — wrapping an un-stageable name would call an undefined helper (NameError).
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from meta_n.core.archive import Candidate
from meta_n.core.evolutionary_orchestrator import (
    EvolutionaryConfig,
    EvolutionaryOrchestrator,
    EvolutionaryResult,
)
from meta_n.core.meta_layer import (
    InjectedCode,
    MetaLayer,
    TaskDescription,
    Trace,
    populate_adoption_fields,
)

GOOD_SRC = "def good_helper(x):\n    return x + 1\n"


# --------------------------------------------------------------------------- #
# Shared fixtures / fakes                                                      #
# --------------------------------------------------------------------------- #
def _tasks(names=("t1",)):
    return [TaskDescription(task_id=n, description=f"Solve {n}") for n in names]


def _orch(**cfg) -> EvolutionaryOrchestrator:
    d = dict(max_depth=20, parallel=1, patience=8, gate_tasks=0,
             beam_width=1, beam_candidates=1)
    d.update(cfg)
    return EvolutionaryOrchestrator(
        llm_client=MagicMock(), executor=MagicMock(), omega=MagicMock(),
        config=EvolutionaryConfig(**d), solver_language="bash",
    )


def _passing_verifier():
    """Verifier whose every helper PASSES, ran genuinely in-sandbox."""
    v = MagicMock()
    v.verify = lambda name, source, task_id, ctx: SimpleNamespace(
        passed=True, ran_in_sandbox=True, evidence="ok"
    )
    return v


def _nonadopting_cand(cid, score, *, depth=2):
    """Candidate whose trace ADVERTISED good_helper but re-derived it inline."""
    tr = Trace(
        task_id="t1", depth=depth, script="def solve(**k):\n    return 1\n",
        success=True, score=score,
        utilities_available=["good_helper"], utilities_called=[],
    )
    return Candidate(
        candidate_id=cid, parent_id=None, iteration=0, depth=depth,
        injected_codes=[], traces=[tr], pass_at_1=1.0,
        mean_score=score, per_task_scores={"t1": score},
    )


def _seed(orch, score=0.5):
    seed = Candidate(
        candidate_id="seed", parent_id=None, iteration=0, depth=1,
        injected_codes=[],
        traces=[Trace(task_id="t1", depth=1, script="x", success=True, score=score)],
        pass_at_1=1.0, mean_score=score, per_task_scores={"t1": score},
    )
    orch.archive.add(seed)


def _wire_eval(orch, evaluated: Candidate):
    """Stub the heavy I/O so the refine/reprop add-guard is exercised in isolation."""
    async def fake_eval(cand, solver, tasks, precomputed=None):
        return evaluated
    orch._evaluate_candidate = fake_eval
    orch._build_solver_from_candidate = lambda c: MagicMock()
    orch._save_candidate_incremental = lambda *a, **k: None
    orch._make_heldout_verifier = _passing_verifier


# --------------------------------------------------------------------------- #
# R2-CS-1 — refine path bars the non-adopting verified win                     #
# --------------------------------------------------------------------------- #
def _run_refine(orch):
    orch.omega = MagicMock()
    orch.omega.refine = AsyncMock(
        return_value=(InjectedCode(code_library={"good_helper": GOOD_SRC}), 10)
    )
    _wire_eval(orch, _nonadopting_cand("refined", 0.9))
    _seed(orch, 0.5)
    return asyncio.run(orch._attempt_within_layer_refine(
        parent=Candidate(candidate_id="parent", parent_id=None, iteration=0,
                          depth=1, injected_codes=[]),
        buggy_child=_nonadopting_cand("buggy", 0.1),
        buggy_injection=InjectedCode(pre_process="echo hi"),
        gate_traces={"t1": Trace(task_id="t1", depth=2, script="x",
                                 success=False, score=0.1)},
        tasks=_tasks(), iteration=1, child_depth=2, temperature=0.5,
        previous_scores=None, archive_best_scores=None,
        result=EvolutionaryResult(), out_dir=Path("/tmp"),
    ))


def test_r2cs1_refine_bars_nonadopting_under_verified_code():
    orch = _orch(verified_code=True)
    _run_refine(orch)
    # The refined 0.9 advertised-but-uncalled trace is BARRED → seed 0.5 holds.
    assert orch.archive.per_task_best_scores()["t1"] == pytest.approx(0.5)
    assert "refined" in orch.archive._by_id          # still archived (monotonic)
    assert orch.archive.get("refined").mean_score == pytest.approx(0.9)


def test_r2cs1_refine_unbarred_when_flag_off():
    # Flag OFF ⇒ the exact original add(); the same trace WINS (penalty is the
    # only thing that bars it — proving the fix is load-bearing, not vacuous).
    orch = _orch(verified_code=False)
    _run_refine(orch)
    assert orch.archive.per_task_best_scores()["t1"] == pytest.approx(0.9)


# --------------------------------------------------------------------------- #
# R2-CS-1 — re-propagation path bars the non-adopting verified win             #
# --------------------------------------------------------------------------- #
def _run_reprop(orch):
    orch.omega = MagicMock()
    orch.omega.generate = AsyncMock(
        return_value=(InjectedCode(code_library={"good_helper": GOOD_SRC}), 10)
    )
    _wire_eval(orch, _nonadopting_cand("reprop", 0.9, depth=3))
    _seed(orch, 0.5)
    child = Candidate(
        candidate_id="child", parent_id=None, iteration=0, depth=3,
        injected_codes=[InjectedCode(pre_process="a"), InjectedCode(pre_process="b")],
        traces=[Trace(task_id="t1", depth=3, script="x", success=False, score=0.2)],
        pass_at_1=0.0, mean_score=0.2, per_task_scores={"t1": 0.2},
    )
    return asyncio.run(orch._attempt_repropagation(
        child=child, tasks=_tasks(), iteration=1, temperature=0.5,
        archive_best_scores=None, result=EvolutionaryResult(), out_dir=Path("/tmp"),
    ))


def test_r2cs1_reprop_bars_nonadopting_under_verified_code():
    orch = _orch(verified_code=True)
    _run_reprop(orch)
    assert orch.archive.per_task_best_scores()["t1"] == pytest.approx(0.5)  # barred
    assert "reprop" in orch.archive._by_id


def test_r2cs1_reprop_unbarred_when_flag_off():
    orch = _orch(verified_code=False)
    _run_reprop(orch)
    assert orch.archive.per_task_best_scores()["t1"] == pytest.approx(0.9)  # wins


# --------------------------------------------------------------------------- #
# R2-CS-2 — utilities_available == what the solver was actually shown          #
# --------------------------------------------------------------------------- #
BROKEN_SRC = "def broken_helper(x):\n    return x\nundefined_xyz_name\n"  # smoke-fails


def test_r2cs2_validate_skipped_helper_excluded_from_utilities_available():
    tr = Trace(task_id="t1", script="def solve(**k):\n    return good_helper(1)\n",
               success=True, score=1.0)
    populate_adoption_fields(
        tr, command_count=1,
        merged_code_library={"good_helper": GOOD_SRC, "broken_helper": BROKEN_SRC},
        executor=None,
    )
    # broken_helper was never advertised/staged → excluded from utilities_available.
    assert tr.utilities_available == ["good_helper"]
    assert "broken_helper" not in tr.utilities_available
    assert tr.utilities_called == ["good_helper"]


def test_r2cs2_valid_only_path_unchanged():
    # Unaffected path: a fully-valid library is advertised verbatim (no filtering).
    tr = Trace(task_id="t1", script="def solve(**k):\n    return 1\n",
               success=True, score=1.0)
    populate_adoption_fields(
        tr, command_count=1, merged_code_library={"good_helper": GOOD_SRC},
        executor=None,
    )
    assert tr.utilities_available == ["good_helper"]
    assert tr.utilities_called == []  # measured-zero (advertised, not called)


def test_r2cs2_empty_library_stays_unmeasurable():
    # Demoted / CO-Bench DEFAULT path: empty library ⇒ utilities_called None.
    tr = Trace(task_id="t1", script="x", success=True, score=1.0)
    populate_adoption_fields(tr, command_count=1, merged_code_library={}, executor=None)
    assert tr.utilities_called is None
    assert tr.utilities_available == []  # default, untouched


# --------------------------------------------------------------------------- #
# R2-CS-3 — deploy wrapper targets the first STAGEABLE verified helper         #
# --------------------------------------------------------------------------- #
class _CannedSolver:
    def __init__(self, script: str):
        self.script = script

    async def solve(self, task, additional_context: str = ""):
        return self.script, "canned", 7


class _CapturingExecutor:
    def __init__(self):
        self.exec_scripts: list[str] = []

    async def execute(self, script: str, task, timeout: int = 30) -> Trace:
        self.exec_scripts.append(script)
        return Trace(task_id=task.task_id, script=script, success=True, score=1.0)


@pytest.fixture
def task():
    return TaskDescription(task_id="t1", description="solve it")


# 'aaa_broken' sorts FIRST but fails validate (smoke); 'zzz_good' is stageable.
AAA_BROKEN = "def aaa_broken(x):\n    return x\nundefined_xyz_name\n"
ZZZ_GOOD = "def zzz_good(N, tasks):\n    return {'assignment': []}\n"


def _deploy_layer(library, solver, executor):
    return MetaLayer(
        depth=2, injected_code=InjectedCode(), inner_solver=solver,
        executor=executor, merged_code_library=library,
        deploy_verified_code=True, solver_language="python",
    )


@pytest.mark.asyncio
async def test_r2cs3_skips_unstageable_helper(task):
    ex = _CapturingExecutor()
    layer = _deploy_layer(
        {"aaa_broken": AAA_BROKEN, "zzz_good": ZZZ_GOOD},
        _CannedSolver(""),  # empty authored solve ⇒ non-adoption ⇒ deploy fallback
        ex,
    )
    await layer.execute(task)
    body = ex.exec_scripts[0]
    # The wrapper targets the first STAGEABLE helper, not sorted()[0]=aaa_broken.
    assert "return zzz_good(" in body
    assert "aaa_broken(" not in body  # the un-stageable name is never called


@pytest.mark.asyncio
async def test_r2cs3_all_unstageable_returns_script_unchanged(task):
    ex = _CapturingExecutor()
    authored = "def solve(**kw):\n    return {'assignment': []}\n"  # inline re-derive
    layer = _deploy_layer({"aaa_broken": AAA_BROKEN}, _CannedSolver(authored), ex)
    await layer.execute(task)
    # No stageable verified helper ⇒ wrapper not deployed; equals the authored
    # script run through the (helper-skipping) prepend — never a broken NameError.
    assert ex.exec_scripts[0] == layer._prepend_library(authored)
    assert "def solve(**kw):" in ex.exec_scripts[0]


# 'solve_crew_scheduling' is an ENTRY POINT that calls the '_crew_pack' utility;
# despite sorting AFTER it, the wrapper must target the entry point, not the dep.
CREW_PACK = "def _crew_pack(N, K, time_limit, tasks, arcs):\n    return [(0, 0)]\n"
SOLVE_CREW = (
    "def solve_crew_scheduling(N, K, time_limit, tasks, arcs):\n"
    "    packed = _crew_pack(N, K, time_limit, tasks, arcs)\n"
    "    return {'assignment': packed}\n"
)


@pytest.mark.asyncio
async def test_deploy_prefers_entry_point_over_dependency_utility(task):
    ex = _CapturingExecutor()
    layer = _deploy_layer(
        {"_crew_pack": CREW_PACK, "solve_crew_scheduling": SOLVE_CREW},
        _CannedSolver(""),  # empty authored solve ⇒ non-adoption ⇒ deploy fallback
        ex,
    )
    await layer.execute(task)
    body = ex.exec_scripts[0]
    # The call-graph pick prefers the entry point (called by nobody) over the
    # '_crew_pack' utility that sorts first but is a dependency of it.
    assert "return solve_crew_scheduling(" in body
    assert "return _crew_pack(" not in body


# Two independent helpers (no cross-call) ⇒ every name is an entry point ⇒
# sorted-first pick is preserved byte-identically to the legacy stageable[0].
AAA_SOLVE = "def aaa_solve(N, tasks):\n    return {'assignment': []}\n"
BBB_SOLVE = "def bbb_solve(N, tasks):\n    return {'assignment': []}\n"


@pytest.mark.asyncio
async def test_deploy_independent_helpers_keep_sorted_first(task):
    ex = _CapturingExecutor()
    layer = _deploy_layer(
        {"aaa_solve": AAA_SOLVE, "bbb_solve": BBB_SOLVE},
        _CannedSolver(""),  # empty authored solve ⇒ non-adoption ⇒ deploy fallback
        ex,
    )
    await layer.execute(task)
    body = ex.exec_scripts[0]
    assert "return aaa_solve(" in body
    assert "return bbb_solve(" not in body
