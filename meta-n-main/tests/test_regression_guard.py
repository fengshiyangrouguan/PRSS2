"""Forensic improvement #1 — REGRESSION GUARD + median-of-R FOCUS-DENOISE.

Flag-gated behind ``--regression-guard`` (config.regression_guard), default OFF.
Default-OFF byte-identity is covered by ``tests/golden/test_stage23_golden.py``;
this module proves the flag-ON behavior is load-bearing and correct, plus the
on-disk re-score that reproduces the +0.104 the asymmetric estimator washed out.

Three gates (mirroring the plan):
  (i)   SYNTHETIC regression guard — per-task-best := max(Ω, base floor); never
        accepts a per-task regression; monotone-max; the archived seed candidate
        is never mutated. OFF proves the guard is load-bearing.
  (ii)  SYNTHETIC focus-denoise — the consolidate FOCUS pick selects the genuinely
        low-headroom task on a median-of-R denoised score, not the degenerate
        one-draw-0.0 task. OFF is order-only round-robin.
  (iii) ON-DISK re-score — over the committed treatment/control result files the
        guard's net effect (max over the two arms) is +0.104 vs best-of-N.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from meta_n.core.archive import Archive, Candidate
from meta_n.core.evolutionary_orchestrator import (
    EvolutionaryConfig,
    EvolutionaryOrchestrator,
    EvolutionaryResult,
)
from meta_n.core.meta_layer import TaskDescription, Trace


# --- helpers ---------------------------------------------------------------

def _trace(tid: str, score: float) -> Trace:
    return Trace(
        task_id=tid, depth=1, script=f"def solve(**k): pass  # {tid}",
        success=True, score=score,
    )


def _cand(cid: str, scores: dict[str, float], depth: int = 1,
          parent_id: str | None = None) -> Candidate:
    traces = [_trace(t, s) for t, s in scores.items()]
    return Candidate(
        candidate_id=cid, parent_id=parent_id, iteration=0, depth=depth,
        injected_codes=[], traces=traces, pass_at_1=1.0,
        mean_score=sum(scores.values()) / len(scores),
        per_task_scores=dict(scores),
    )


def _tasks(names):
    return [TaskDescription(task_id=n, description=f"Solve {n}") for n in names]


def _orch(**cfg) -> EvolutionaryOrchestrator:
    d = dict(max_depth=20, parallel=1, patience=8, gate_tasks=0,
             beam_width=1, beam_candidates=1)
    d.update(cfg)
    return EvolutionaryOrchestrator(
        llm_client=MagicMock(), executor=MagicMock(), omega=MagicMock(),
        config=EvolutionaryConfig(**d), solver_language="bash",
    )


# --- (0) config / archive defaults are OFF ---------------------------------

def test_config_defaults_off():
    cfg = EvolutionaryConfig()
    assert cfg.regression_guard is False
    assert cfg.regression_guard_repeats == 3


def test_archive_default_has_no_floor():
    arch = Archive()  # no kwarg → guard OFF
    assert arch._regression_guard is False
    assert arch._base_floor == {}
    # set_base_floor is simply never called when OFF; add() clamp short-circuits.
    arch.add(_cand("gen0_seed", {"crew": 0.0}))
    arch.add(_cand("gen1", {"crew": 0.1}, depth=2))
    # Today's regression: the Ω 0.1 IS the per-task-best (no floor to refuse it).
    assert arch.per_task_best_scores()["crew"] == pytest.approx(0.1)


# --- (i) synthetic regression guard ----------------------------------------

def test_regression_guard_refuses_subfloor_omega_win_promoted():
    arch = Archive(regression_guard=True)
    seed = _cand("gen0_seed", {"crew": 0.0, "equitable": 0.48})
    arch.add(seed)
    # Base floor from the seed-only resamples: crew recovers to 0.658, equitable
    # is genuinely ~0.48. (best-of-R floor)
    arch.set_base_floor(
        {"crew": 0.658, "equitable": 0.48},
        {"crew": _trace("crew", 0.658), "equitable": _trace("equitable", 0.48)},
    )
    # Ω candidate: catastrophic regression on crew (0.1), genuine win on equitable.
    omega = _cand("gen4", {"crew": 0.1, "equitable": 1.0}, depth=5)
    arch.add(omega)

    ptb = arch.per_task_best_scores()
    # Never ship the per-task regression: base floor wins on crew.
    assert ptb["crew"] == pytest.approx(0.658)
    # The genuine Ω win is still promoted.
    assert ptb["equitable"] == pytest.approx(1.0)

    # Monotone-max: re-adding an even-worse sub-floor candidate never lowers it.
    arch.add(_cand("gen5", {"crew": 0.05}, depth=6))
    assert arch.per_task_best_scores()["crew"] == pytest.approx(0.658)

    # Monotonic invariant: the stored seed candidate/trace is NOT mutated by the
    # floor (the floor updates an INDEX entry with a fresh resampled trace).
    assert arch.get("gen0_seed").per_task_scores["crew"] == pytest.approx(0.0)


def test_regression_guard_off_is_load_bearing():
    # Same candidates, guard OFF → today's regression (Ω 0.1 becomes the best).
    arch = Archive(regression_guard=False)
    arch.add(_cand("gen0_seed", {"crew": 0.0, "equitable": 0.48}))
    arch.add(_cand("gen4", {"crew": 0.1, "equitable": 1.0}, depth=5))
    ptb = arch.per_task_best_scores()
    assert ptb["crew"] == pytest.approx(0.1)  # the regression the guard fixes
    assert ptb["equitable"] == pytest.approx(1.0)


def test_floor_does_not_demote_a_real_win_above_it():
    # A floor BELOW an existing real win must not pull the best down.
    arch = Archive(regression_guard=True)
    arch.add(_cand("gen0_seed", {"t": 0.3}))
    arch.add(_cand("gen2", {"t": 0.9}, depth=3))
    arch.set_base_floor({"t": 0.5}, {"t": _trace("t", 0.5)})
    assert arch.per_task_best_scores()["t"] == pytest.approx(0.9)


def test_all_crash_floor_does_not_suppress_valid_negative_omega():
    # Negative-capable scale, all-R base resamples crash on a task -> floor
    # (success=False, 0.0). A later VALID negative Ω fit must be promoted, not
    # clamped by the raw 0.0 floor (dual-channel clamp).
    arch = Archive(regression_guard=True, score_ceiling=None)
    crash = Trace(task_id="sr", depth=1, script="s", success=False, score=0.0)
    arch.set_base_floor({"sr": 0.0}, {"sr": crash})
    omega = Candidate(candidate_id="gen1", iteration=1, depth=2,
                      traces=[Trace(task_id="sr", depth=2, script="s",
                                    success=True, score=-0.30)],
                      mean_score=-0.30, per_task_scores={"sr": -0.30})
    arch.add(omega)
    assert arch.per_task_best_sources()["sr"] == "gen1"
    assert arch.best_score_for_task("sr") == pytest.approx(-0.30)
    # A genuine sub-floor regression is STILL refused (guard intact):
    arch.add(Candidate(candidate_id="gen2", iteration=2, depth=2,
                       traces=[Trace(task_id="sr", depth=2, script="s",
                                     success=True, score=-0.90)],
                       mean_score=-0.90, per_task_scores={"sr": -0.90}))
    assert arch.best_score_for_task("sr") == pytest.approx(-0.30)  # not lowered


# --- (ii) synthetic focus-denoise ------------------------------------------

@pytest.mark.asyncio
async def test_establish_base_floor_best_and_median():
    """End-to-end gen0 path: R seed resamples → best-of-R floor + median-of-R
    focus, with the resamples NOT added to the archive (monotonic)."""
    orch = _orch(regression_guard=True, regression_guard_repeats=4, consolidate=True)
    tasks = _tasks(["crew", "lowtask", "other"])

    # Seed = draw 0 (crew degenerate 0.0 — the seed-42 pathology).
    seed = _cand("gen0_seed", {"crew": 0.0, "lowtask": 0.19, "other": 0.95})
    orch.archive.add(seed)

    # Draws 1..3 from the resamples (crew is actually healthy).
    draws = iter([
        {"crew": 0.658, "lowtask": 0.18, "other": 0.90},
        {"crew": 0.62, "lowtask": 0.20, "other": 0.92},
        {"crew": 0.61, "lowtask": 0.22, "other": 0.88},
    ])

    async def fake_eval(cand, solver, tasks, precomputed=None, *, crn_repeat_offset=0):
        scores = next(draws)
        cand.traces = [_trace(t, s) for t, s in scores.items()]
        cand.per_task_scores = dict(scores)
        cand.mean_score = sum(scores.values()) / len(scores)
        return cand

    orch._evaluate_candidate = AsyncMock(side_effect=fake_eval)

    await orch._establish_base_floor(seed, MagicMock(), tasks, EvolutionaryResult())

    # Floor = best-of-R (crew recovers to 0.658 despite the degenerate seed draw).
    ptb = orch.archive.per_task_best_scores()
    assert ptb["crew"] == pytest.approx(0.658)
    assert ptb["lowtask"] == pytest.approx(0.22)
    assert ptb["other"] == pytest.approx(0.95)

    # Focus = median-of-R (crew ~0.62, NOT the 0.0 outlier; lowtask ~0.20).
    assert orch._base_focus_scores["crew"] == pytest.approx(0.62)
    assert orch._base_focus_scores["lowtask"] == pytest.approx(0.20)
    assert orch._base_focus_scores["other"] == pytest.approx(0.92)

    # Monotonic: resample candidates are NOT in the archive; seed unchanged.
    assert "gen0_seed_rg1" not in orch.archive._by_id
    assert orch.archive.get("gen0_seed").per_task_scores["crew"] == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_establish_base_floor_dual_channel_valid_over_crash():
    """R1-D_selection-1: on a negative-capable scale (symbolic_regression) the base
    floor must pick a VALID-but-poor fit (success=True, -0.7) over a CRASH
    (success=False, floored to 0.0), mirroring archive.py's (success, score)
    per-task-best ranking — so a genuine negative Ω gain (-0.5 > -0.7) is later
    promoted instead of being clamped by an inverted 0.0 floor."""
    orch = _orch(regression_guard=True, regression_guard_repeats=2, consolidate=True)
    tasks = _tasks(["sr"])

    # Draw 0 (the seed's own eval): a VALID negative fit.
    seed = Candidate(
        candidate_id="gen0_seed", parent_id=None, iteration=0, depth=1,
        injected_codes=[], pass_at_1=1.0, mean_score=-0.7,
        per_task_scores={"sr": -0.7},
        traces=[Trace(task_id="sr", depth=1, script="def solve(**k): pass",
                      success=True, score=-0.7)],
    )
    orch.archive.add(seed)

    # Draw 1 (the resample): a CRASH — score floored to 0.0, success=False.
    async def fake_eval(cand, solver, tasks, precomputed=None, *, crn_repeat_offset=0):
        cand.traces = [Trace(task_id="sr", depth=1, script="def solve(**k): pass",
                             success=False, score=0.0)]
        cand.per_task_scores = {"sr": 0.0}
        cand.mean_score = 0.0
        return cand
    orch._evaluate_candidate = AsyncMock(side_effect=fake_eval)

    # Guard the SELECTION KEY, not just the outcome: the pre-fix score-only max
    # inverts and picks the crash's 0.0; the dual-channel key picks the valid -0.7.
    seed_tr = orch.archive.get("gen0_seed").traces[0]
    crash_tr = Trace(task_id="sr", depth=1, script="s", success=False, score=0.0)
    draws = [(-0.7, seed_tr), (0.0, crash_tr)]
    assert max(draws, key=lambda st: st[0])[0] == 0.0            # pre-fix bug
    assert max(draws, key=lambda st: (bool(st[1].success), st[0]))[0] == -0.7  # fix

    await orch._establish_base_floor(seed, MagicMock(), tasks, EvolutionaryResult())

    # Floor is the valid negative fit, not the crash's 0.0.
    assert orch.archive.base_floor_snapshot()["sr"][0] == pytest.approx(-0.7)
    assert orch.archive.per_task_best_scores()["sr"] == pytest.approx(-0.7)
    assert orch.archive.per_task_best_traces()["sr"].success is True

    # A genuine negative Ω gain (-0.5 > the -0.7 floor) is promoted, not clamped.
    orch.archive.add(_cand("gen4", {"sr": -0.5}, depth=5))
    assert orch.archive.per_task_best_scores()["sr"] == pytest.approx(-0.5)


def test_focus_pick_denoised_headroom_on_vs_roundrobin_off():
    tasks = _tasks(["crew", "lowtask", "other"])

    # ON: median-of-R denoised scores → crew is healthy (0.62), lowtask is the
    # true highest-headroom (0.20). The degenerate 0.0 no longer drives focus.
    orch_on = _orch(regression_guard=True, consolidate=True)
    orch_on._base_focus_scores = {"crew": 0.62, "lowtask": 0.20, "other": 0.92}
    picked = orch_on._consolidation_targets(tasks, n=1, iteration=0)
    assert picked == ["lowtask"]
    # Iteration no longer rotates the pick (headroom is iteration-invariant).
    assert orch_on._consolidation_targets(tasks, n=1, iteration=5) == ["lowtask"]

    # OFF: pure round-robin (order-only, rotates by iteration) — unchanged HEAD.
    orch_off = _orch(regression_guard=False, consolidate=True)
    pool = sorted(t.task_id for t in tasks)  # ['crew','lowtask','other']
    assert orch_off._consolidation_targets(tasks, n=1, iteration=0) == [pool[0]]
    assert orch_off._consolidation_targets(tasks, n=1, iteration=1) == [pool[1]]


def test_focus_pick_falls_back_to_roundrobin_without_denoised_scores():
    # Guard ON but no denoised scores yet (e.g. before gen0 floor) → round-robin.
    orch = _orch(regression_guard=True, consolidate=True)
    assert orch._base_focus_scores == {}
    tasks = _tasks(["a", "b", "c"])
    pool = sorted(t.task_id for t in tasks)
    assert orch._consolidation_targets(tasks, n=1, iteration=0) == [pool[0]]
    assert orch._consolidation_targets(tasks, n=1, iteration=1) == [pool[1]]


# --- checkpoint round-trip (resume keeps the floor) -------------------------

def test_base_floor_serializes_for_checkpoint_roundtrip():
    """The floor must survive a pause/resume so the guard does not silently
    weaken. Simulate the checkpoint save/restore of _base_floor."""
    arch = Archive(regression_guard=True)
    arch.add(_cand("gen0_seed", {"crew": 0.0}))
    arch.set_base_floor({"crew": 0.658}, {"crew": _trace("crew", 0.658)})

    # Serialize exactly as _save_checkpoint does.
    serialized = {
        tid: {
            "score": e[0],
            "candidate_id": e[1],
            "trace": (e[2].model_dump(mode="json") if e[2] is not None else None),
        }
        for tid, e in arch._base_floor.items()
    }
    blob = json.dumps(serialized)  # must be JSON-serializable

    # Restore into a fresh (rebuilt) archive exactly as the resume branch does.
    restored = json.loads(blob)
    arch2 = Archive(regression_guard=True)
    arch2.add(_cand("gen0_seed", {"crew": 0.0}))
    floor = {tid: e["score"] for tid, e in restored.items()}
    floor_traces = {
        tid: (Trace.model_validate(e["trace"]) if e.get("trace") else None)
        for tid, e in restored.items()
    }
    arch2.set_base_floor(floor, floor_traces)
    assert arch2.per_task_best_scores()["crew"] == pytest.approx(0.658)
    # And a sub-floor Ω add is still refused after restore.
    arch2.add(_cand("gen3", {"crew": 0.1}, depth=4))
    assert arch2.per_task_best_scores()["crew"] == pytest.approx(0.658)


# --- (iii) on-disk re-score: reproduce +0.104 -------------------------------

_EXP_DIR = Path(__file__).resolve().parents[1] / "experiments" / "metan_e2_g9_v3_s42"


@pytest.mark.skipif(
    not (_EXP_DIR / "treatment_result.json").exists()
    or not (_EXP_DIR / "control_result.json").exists(),
    reason="on-disk g9_v3_s42 result fixtures not present",
)
def test_on_disk_rescore_reproduces_plus_0104():
    treatment = json.loads((_EXP_DIR / "treatment_result.json").read_text())
    control = json.loads((_EXP_DIR / "control_result.json").read_text())

    treat = {k: v["treatment_best"] for k, v in treatment["per_task"].items()}
    bestn = {k: v["best_of_n"] for k, v in control["per_task"].items()}
    shared = sorted(set(treat) & set(bestn))
    assert len(shared) == 6  # the 6 shared CO-Bench tasks

    def mean(d):
        return sum(d) / len(d)

    omega_mean = mean([treat[k] for k in shared])
    bestn_mean = mean([bestn[k] for k in shared])
    # The regression guard's deployable per-task-best := max(Ω, base/best-of-N).
    guard_mean = mean([max(treat[k], bestn[k]) for k in shared])

    # The washed headline: Ω barely beats best-of-N (one catastrophic regression
    # cancels the wins).
    assert omega_mean == pytest.approx(0.6846, abs=1e-3)
    assert bestn_mean == pytest.approx(0.6737, abs=1e-3)
    assert (omega_mean - bestn_mean) == pytest.approx(0.0109, abs=1e-3)
    # The guard recovers the +0.104 the asymmetric estimator washed out.
    assert guard_mean == pytest.approx(0.7776, abs=1e-3)
    assert (guard_mean - bestn_mean) == pytest.approx(0.1039, abs=1e-3)
