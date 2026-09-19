"""Refine §6b regression tests — orchestrator-loop cluster (plumbing stage).

Findings covered:
  * F034 — --focus-current-headroom: flag-gated re-ranking of the consolidate
    FOCUS pick at the CURRENT per-task best once a task genuinely improves past
    its best-of-R base floor (default OFF = frozen denoised-base ranking,
    byte-identical — the staleness pin in tests/test_regression_guard.py stays
    green).
  * F035 — --eval-repeats-gate-topup: flag-gated gate-trace top-up under
    eval_repeats>1 (gate trace = sample 0, R-1 fresh solves, median over the
    union; token accounting per the audit-#29 / H7 contracts; consolidation
    inherit-frozen traces exempt). Default OFF = the reuse short-circuit is
    byte-identical.
  * F033-half — the budget-halt and max-depth breaks no longer append an
    unpaired convergence_history entry; on EVERY exit path
    len(convergence_history) == len(oracle_history) + 1 == completed_gens + 1.
  * F063/F060 wiring — agentic_spend_budget / agentic_temperature reach the
    AgenticSolver construction site; spend-budget validation.

All tests are LLM-free / offline (no Docker, no network).
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from meta_n.core.archive import Candidate
from meta_n.core.evolutionary_orchestrator import (
    EvolutionaryConfig,
    EvolutionaryOrchestrator,
    EvolutionaryResult,
)
from meta_n.core.meta_layer import InjectedCode, TaskDescription, Trace

# --------------------------------------------------------------------------- #
# Helpers (patterns shared with test_refine_evolutionary_orchestrator.py)
# --------------------------------------------------------------------------- #


def _make_orch(**overrides) -> EvolutionaryOrchestrator:
    defaults = dict(
        max_depth=3,
        parallel=2,
        patience=2,
        gate_tasks=0,
        beam_width=1,
        beam_candidates=1,
    )
    defaults.update(overrides)
    return EvolutionaryOrchestrator(
        llm_client=MagicMock(),
        executor=MagicMock(),
        omega=MagicMock(),
        config=EvolutionaryConfig(**defaults),
        solver_language="bash",
    )


def _seed_run_mocks(orch, score: float = 0.5) -> None:
    orch.solver.solve = AsyncMock(return_value=("echo hi", "r", 10))
    orch.executor.execute = AsyncMock(return_value=Trace(
        task_id="t1", success=True, score=score, script="echo hi"))
    orch.omega.generate = AsyncMock(return_value=(InjectedCode(), 50))


def _tasks(names) -> list[TaskDescription]:
    return [TaskDescription(task_id=n, description=f"Solve {n}") for n in names]


def _trace(tid: str, score: float, **kw) -> Trace:
    defaults = dict(task_id=tid, success=True, score=score, script=f"echo {tid}")
    defaults.update(kw)
    return Trace(**defaults)


def _candidate(cid: str, scores: dict[str, float], **kwargs) -> Candidate:
    traces = [_trace(tid, s) for tid, s in scores.items()]
    mean = sum(scores.values()) / len(scores) if scores else 0.0
    defaults = dict(
        candidate_id=cid, iteration=0, depth=1, traces=traces,
        mean_score=mean, pass_at_1=1.0, per_task_scores=dict(scores),
    )
    defaults.update(kwargs)
    return Candidate(**defaults)


# --------------------------------------------------------------------------- #
# F034 — --focus-current-headroom
# --------------------------------------------------------------------------- #


class TestF034FocusCurrentHeadroom:
    """Flag-gated consolidate-focus re-ranking at the current per-task best."""

    _BASE = {"low": 0.2, "mid": 0.6, "hi": 0.9}

    def _guard_orch(self, flag: bool) -> EvolutionaryOrchestrator:
        orch = _make_orch(
            regression_guard=True, consolidate=True,
            focus_current_headroom=flag,
        )
        orch._base_focus_scores = dict(self._BASE)
        orch.archive.set_base_floor(
            dict(self._BASE), {tid: _trace(tid, s) for tid, s in self._BASE.items()},
        )
        return orch

    def test_focus_frozen_ranking_when_flag_off(self):
        # Default OFF: a ceiling-improved task is STILL picked every iteration
        # (pins the old, deliberately test-frozen behavior — mirrors the repro
        # and the tests/test_regression_guard.py staleness pin).
        orch = self._guard_orch(flag=False)
        orch.archive.add(_candidate("c_lift", {"low": 1.0}))
        tasks = _tasks(["low", "mid", "hi"])
        assert orch._consolidation_targets(tasks, n=1, iteration=1) == ["low"]
        assert orch._consolidation_targets(tasks, n=1, iteration=5) == ["low"]

    def test_focus_rotates_after_improvement_when_flag_on(self):
        # Flag ON: 'low' improved to the 1.0 ceiling ranks at its current best
        # and rotates out of focus — 'mid' (true lowest headroom) is picked,
        # iteration-invariantly (no staleness).
        orch = self._guard_orch(flag=True)
        orch.archive.add(_candidate("c_lift", {"low": 1.0}))
        tasks = _tasks(["low", "mid", "hi"])
        assert orch._consolidation_targets(tasks, n=1, iteration=1) == ["mid"]
        assert orch._consolidation_targets(tasks, n=1, iteration=5) == ["mid"]

    def test_focus_flag_on_gen0_ranking_matches_median_baseline(self):
        # Flag ON with floor == PTB (no genuine improvement yet): the ranking is
        # IDENTICAL to flag OFF — the median-of-R denoise contract holds and the
        # gen0 pick is unchanged even when the flag is enabled.
        tasks = _tasks(["low", "mid", "hi"])
        on = self._guard_orch(flag=True)
        off = self._guard_orch(flag=False)
        for n in (1, 2, 3):
            assert (
                on._consolidation_targets(tasks, n=n, iteration=1)
                == off._consolidation_targets(tasks, n=n, iteration=1)
            )

    def test_focus_flag_on_noop_without_regression_guard(self):
        # Flag ON but guard OFF (no denoised scores): the round-robin path is
        # byte-identical (same expectations as tests/test_consolidation.py).
        orch = _make_orch(consolidate=True, focus_current_headroom=True)
        assert orch._base_focus_scores == {}
        tasks = _tasks(["a", "b", "c"])
        pool = sorted(t.task_id for t in tasks)
        assert orch._consolidation_targets(tasks, n=1, iteration=0) == [pool[0]]
        assert orch._consolidation_targets(tasks, n=1, iteration=1) == [pool[1]]
        assert orch._consolidation_targets(tasks, n=1, iteration=2) == [pool[2]]

    def test_focus_flag_on_task_missing_from_floor_uses_ptb(self):
        # A task absent from floor AND base_focus but present in the archive
        # ranks at its current per-task best (max(0.0, cur) — the else branch).
        orch = self._guard_orch(flag=True)
        orch.archive.add(_candidate("c_zz", {"zz": 0.5}))
        eff = orch._effective_focus_scores(["low", "mid", "hi", "zz"])
        assert eff["zz"] == pytest.approx(0.5)
        # Floor-equal tasks keep their denoised base scores.
        assert eff["low"] == pytest.approx(0.2)
        assert eff["mid"] == pytest.approx(0.6)
        assert eff["hi"] == pytest.approx(0.9)

    # --- Y4-P_benchmarks-3: iteration-rotated round-robin within tie groups --

    _BINARY_BASE = {"a": 0.0, "b": 0.0, "c": 0.0, "z": 1.0}

    def _binary_guard_orch(self) -> EvolutionaryOrchestrator:
        orch = _make_orch(
            regression_guard=True, consolidate=True,
            focus_current_headroom=True,
        )
        orch._base_focus_scores = dict(self._BINARY_BASE)
        orch.archive.set_base_floor(
            dict(self._BINARY_BASE),
            {tid: _trace(tid, s) for tid, s in self._BINARY_BASE.items()},
        )
        return orch

    def test_focus_tie_group_rotates_by_iteration(self):
        # Binary scale: the unimproved tasks tie at exactly 0.0 and the pick
        # must round-robin WITHIN that tie group across iterations instead of
        # pinning the alphabetically-first member forever; the improved task
        # ("z", at the 1.0 ceiling) is never picked.
        orch = self._binary_guard_orch()
        tasks = _tasks(["a", "b", "c", "z"])
        picks = [
            orch._consolidation_targets(tasks, n=1, iteration=i)
            for i in range(6)
        ]
        assert picks == [["a"], ["b"], ["c"], ["a"], ["b"], ["c"]]

    def test_focus_distinct_scores_unchanged_by_tie_rotation(self):
        # Distinct scores form singleton tie groups (offset 0) — the ranked
        # prefix is byte-identical to the pre-rotation expectations for every
        # (n, iteration), for both flag states.
        tasks = _tasks(["low", "mid", "hi"])
        expected = {1: ["low"], 2: ["low", "mid"], 3: ["low", "mid", "hi"]}
        for flag in (False, True):
            orch = self._guard_orch(flag=flag)
            for n in (1, 2, 3):
                for it in range(7):
                    assert (
                        orch._consolidation_targets(tasks, n=n, iteration=it)
                        == expected[n]
                    )

    def test_focus_tie_group_shrinks_after_improvement(self):
        # Compose: a candidate lifting "b" to 1.0 removes it from the 0.0 tie
        # group (genuine improvement past its base floor), and the rotation
        # continues over the remaining {a, c}.
        orch = self._binary_guard_orch()
        orch.archive.add(_candidate("c_lift_b", {"b": 1.0}))
        tasks = _tasks(["a", "b", "c", "z"])
        picks = [
            orch._consolidation_targets(tasks, n=1, iteration=i)[0]
            for i in range(4)
        ]
        assert picks == ["a", "c", "a", "c"]


# --------------------------------------------------------------------------- #
# F035 — --eval-repeats-gate-topup
# --------------------------------------------------------------------------- #


class TestF035GateTopup:
    """Gate-trace top-up under eval_repeats>1 (flag-gated, default OFF)."""

    def _instrumented(self, orch, scores=None, tokens=5, inner_tokens=0):
        """Stub _eval_solve_once with a per-task call counter.

        ``scores``: optional {task_id: [score, score, ...]} queues; default 0.4.
        Returns the counts dict.
        """
        counts: dict[str, int] = {}
        queues = {tid: list(q) for tid, q in (scores or {}).items()}

        async def fake_solve_once(solver, task, candidate, *, seed=None):
            counts[task.task_id] = counts.get(task.task_id, 0) + 1
            q = queues.get(task.task_id)
            score = q.pop(0) if q else 0.4
            tr = _trace(task.task_id, score, inner_tokens=inner_tokens)
            return tr, tokens

        orch._eval_solve_once = fake_solve_once
        return counts

    async def test_gate_reuse_single_draw_when_topup_off(self):
        # Pins current behavior (the repro): with eval_repeats=3 and a gate-
        # precomputed task, solve counts are {a: 0, b: 3} and 'a' keeps the
        # single gate draw.
        orch = _make_orch(eval_repeats=3)
        counts = self._instrumented(orch)
        gate = _trace("a", 0.9)
        cand = Candidate(candidate_id="c_off", depth=1)
        cand = await orch._evaluate_candidate(
            cand, MagicMock(), _tasks(["a", "b"]), precomputed={"a": gate},
        )
        assert counts == {"b": 3}
        assert cand.per_task_scores["a"] == pytest.approx(0.9)

    async def test_gate_topup_runs_r_minus_1_extra_solves(self):
        orch = _make_orch(eval_repeats=3, eval_repeats_gate_topup=True)
        counts = self._instrumented(orch, scores={"a": [0.9, 0.9]})
        gate = _trace("a", 0.9)
        cand = Candidate(candidate_id="c_on", depth=1)
        cand = await orch._evaluate_candidate(
            cand, MagicMock(), _tasks(["a", "b"]), precomputed={"a": gate},
        )
        assert counts == {"a": 2, "b": 3}
        # Median of {gate 0.9, topups 0.9, 0.9} — still 0.9 here.
        assert cand.per_task_scores["a"] == pytest.approx(0.9)

    async def test_gate_topup_median_can_reject_lucky_gate_draw(self):
        # The denoising actually bites: gate 0.9, top-ups 0.4/0.5 → median 0.5.
        orch = _make_orch(eval_repeats=3, eval_repeats_gate_topup=True)
        self._instrumented(orch, scores={"a": [0.4, 0.5]})
        gate = _trace("a", 0.9)
        cand = Candidate(candidate_id="c_med", depth=1)
        cand = await orch._evaluate_candidate(
            cand, MagicMock(), _tasks(["a"]), precomputed={"a": gate},
        )
        assert cand.per_task_scores["a"] == pytest.approx(0.5)

    async def test_gate_topup_exempts_frozen_consolidation(self):
        # precomputed_frozen=True (the consolidation inherit map) → NEVER
        # re-solved even with the flag ON: byte-identical reuse.
        orch = _make_orch(eval_repeats=3, eval_repeats_gate_topup=True)
        counts = self._instrumented(orch)
        frozen = _trace("a", 0.7)
        cand = Candidate(candidate_id="c_frozen", depth=1)
        cand = await orch._evaluate_candidate(
            cand, MagicMock(), _tasks(["a", "b"]),
            precomputed={"a": frozen}, precomputed_frozen=True,
        )
        assert counts == {"b": 3}
        assert cand.per_task_scores["a"] == pytest.approx(0.7)

    async def test_gate_topup_noop_at_eval_repeats_1(self):
        # Flag ON but R=1 → identical to the reuse path (the reuse IS the draw).
        orch = _make_orch(eval_repeats=1, eval_repeats_gate_topup=True)
        counts = self._instrumented(orch)
        gate = _trace("a", 0.9)
        cand = Candidate(candidate_id="c_r1", depth=1)
        cand = await orch._evaluate_candidate(
            cand, MagicMock(), _tasks(["a", "b"]), precomputed={"a": gate},
        )
        assert counts == {"b": 1}
        assert cand.per_task_scores["a"] == pytest.approx(0.9)

    async def test_gate_topup_outer_tokens_exclude_gate_sample(self):
        # Sample 0 contributes 0 OUTER tokens (the gate already counted them —
        # audit #29); the candidate pays only the top-up outer spend.
        orch = _make_orch(eval_repeats=3, eval_repeats_gate_topup=True)
        self._instrumented(orch, tokens=5)
        gate = _trace("a", 0.9)
        cand = Candidate(candidate_id="c_tok", depth=1)
        cand = await orch._evaluate_candidate(
            cand, MagicMock(), _tasks(["a"]), precomputed={"a": gate},
        )
        assert cand.total_tokens == 10  # 2 top-ups x 5; gate sample = 0

    async def test_gate_topup_inner_tokens_sum_union(self):
        # H7: inner accounting covers the gate sample + every top-up.
        orch = _make_orch(eval_repeats=3, eval_repeats_gate_topup=True)
        self._instrumented(orch, inner_tokens=7)
        gate = _trace("a", 0.9, inner_tokens=3)
        cand = Candidate(candidate_id="c_inner", depth=1)
        cand = await orch._evaluate_candidate(
            cand, MagicMock(), _tasks(["a"]), precomputed={"a": gate},
        )
        assert cand.inner_tokens == 3 + 7 + 7

    async def test_gate_topup_fault_falls_back_to_partial_union(self):
        # Fault fallback: with R=3, the SECOND top-up raising must not discard
        # the measured samples — the task scores the median over {gate 0.9,
        # top-up 0.5} (upper median 0.9), the gate sample's inner tokens stay
        # accounted, and NO failed 0.0 trace is recorded for the task.
        orch = _make_orch(eval_repeats=3, eval_repeats_gate_topup=True)
        calls = {"n": 0}

        async def flaky_solve_once(solver, task, candidate, *, seed=None):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("transient solver fault")
            return _trace(task.task_id, 0.5, inner_tokens=7), 5

        orch._eval_solve_once = flaky_solve_once
        gate = _trace("a", 0.9, inner_tokens=50)
        cand = Candidate(candidate_id="c_fault", depth=1)
        cand = await orch._evaluate_candidate(
            cand, MagicMock(), _tasks(["a"]), precomputed={"a": gate},
        )
        assert calls["n"] == 2  # first top-up ran; second raised and stopped
        # Median over the partial union {0.5, 0.9} — the gate draw survives.
        assert cand.per_task_scores["a"] == pytest.approx(0.9)
        (tr,) = cand.traces
        assert tr.success and tr.score == pytest.approx(0.9)
        assert "solve raised" not in (tr.error_summary or "")
        # Inner tokens: gate sample (50) + the one top-up that ran (7).
        assert cand.inner_tokens == 50 + 7
        # Outer tokens: only the top-up that actually ran (gate counted 0).
        assert cand.total_tokens == 5

    async def test_gate_topup_stamps_repeat_index_on_spine(self):
        # H6 mirror of test_eval_repeats_stamps_repeat_index_on_spine_solver:
        # top-ups stamp repeat indices 1..R-1 (sample 0 is the gate draw) and
        # restore 0 afterwards.
        orch = _make_orch(eval_repeats=3, eval_repeats_gate_topup=True)
        orch._uses_external_spine = lambda: True
        assert orch._agent_telemetry is None  # no restamp side channel here

        seen: list = []

        class _FakeSpineSolver:
            async def execute(self, task):
                seen.append(getattr(self, "repeat_index", None))
                return _trace(task.task_id, 0.5), 0

        solver = _FakeSpineSolver()
        gate = _trace("t1", 0.9)
        cand = Candidate(candidate_id="c_spine_topup", depth=1)
        await orch._evaluate_candidate(
            cand, solver, _tasks(["t1"]), precomputed={"t1": gate},
        )
        assert seen == [1, 2]
        assert getattr(solver, "repeat_index") == 0


# --------------------------------------------------------------------------- #
# Y4-P_benchmarks-5 — base-floor resamples land on DISTINCT telemetry run_ids
# --------------------------------------------------------------------------- #


async def test_base_floor_resamples_write_distinct_run_ids(tmp_path):
    # Spine + regression_guard at eval_repeats=1 (the reachable combo): the
    # R-1 base-floor resamples re-run the SAME seed solver at the SAME
    # (generation, candidate_id, task_id, depth) coordinates, so each must be
    # stamped with its crn_repeat_offset as the telemetry repeat index — else
    # all R rows share one run_id and R-1 real runs are de-dup-dropped (or a
    # clean resample silently supersedes a degraded draw-0 row).
    from meta_n.core.external_agents.backend import AgentRunResult
    from meta_n.core.external_agents.telemetry import AgentTelemetry
    from meta_n.integrations.benchmark import EvalResult

    orch = _make_orch(
        regression_guard=True, regression_guard_repeats=3, eval_repeats=1,
    )
    orch._uses_external_spine = lambda: True
    orch._agent_telemetry = AgentTelemetry(tmp_path)

    seen: list[tuple[str, object]] = []

    class _FakeSpineSolver:
        backend = SimpleNamespace(name="terminus2", outer_token_mode=False)
        depth = 1
        generation = 0
        candidate_id = "gen0_seed"
        max_turns = 8

        def __init__(self, tel):
            self._tel = tel

        async def execute(self, task):
            # Synchronous prefix (mirrors _execute_with_plan): the record is
            # opened before any await, reading repeat_index off the solver.
            seen.append((task.task_id, getattr(self, "repeat_index", None)))
            rec = self._tel.start_record(task, self)
            self._tel.finish_record(
                rec, AgentRunResult(), EvalResult(success=True, score=0.5)
            )
            return _trace(task.task_id, 0.5), 0

    solver = _FakeSpineSolver(orch._agent_telemetry)
    tasks = _tasks(["t1", "t2"])
    seed = Candidate(candidate_id="gen0_seed", iteration=0, depth=1)
    seed = await orch._evaluate_candidate(seed, solver, tasks)  # draw 0
    orch.archive.add(seed)
    await orch._establish_base_floor(seed, solver, tasks, EvolutionaryResult())

    rows = []
    for line in (
        (tmp_path / "telemetry" / "agent_runs.jsonl").read_text().splitlines()
    ):
        if line.strip():
            obj = json.loads(line)
            rec = (obj.get("extra", {}) or {}).get("record")
            rows.append(rec if rec is not None else obj)
    assert len(rows) == 6  # 2 tasks x (draw 0 + 2 resamples); pre-fix: 2 rows
    run_ids_by_task: dict[str, set[str]] = {}
    for row in rows:
        run_ids_by_task.setdefault(row["task_id"], set()).add(row["run_id"])
    assert {t: len(ids) for t, ids in run_ids_by_task.items()} == {"t1": 3, "t2": 3}
    # Resample r observed repeat_index r (offset r*E with E=1); draw 0 is
    # never stamped (offset 0 keeps the historical run_id basis).
    per_task: dict[str, list] = {}
    for tid, idx in seen:
        per_task.setdefault(tid, []).append(idx)
    assert per_task == {"t1": [None, 1, 2], "t2": [None, 1, 2]}
    assert solver.repeat_index == 0  # finally-reset after the last resample


# --------------------------------------------------------------------------- #
# F033-half — history pairing on budget / max-depth exits
# --------------------------------------------------------------------------- #


def _assert_pairing_and_disk(orch, result, tmp_path):
    assert len(result.convergence_history) == len(result.oracle_history) + 1
    orch.save_results(result)
    conv = json.loads((tmp_path / "convergence.json").read_text())
    oracle = json.loads((tmp_path / "oracle_convergence.json").read_text())
    assert conv == result.convergence_history
    assert oracle == result.oracle_history


async def test_budget_halt_history_pairing(tmp_path):
    orch = _make_orch(output_dir=str(tmp_path), max_iterations=50, patience=3)
    _seed_run_mocks(orch)
    answers = iter([False])  # iteration 1 proceeds; iteration 2 top halts
    orch._cost_guard = SimpleNamespace(
        headroom_exhausted=lambda: next(answers, True)
    )
    result = await orch.run(_tasks(["t1"]))
    # 1 completed generation: seed entry + 1 iteration; NO break-path append.
    assert result.convergence_history == [0.5, 0.5]
    assert result.oracle_history == [0.5]
    _assert_pairing_and_disk(orch, result, tmp_path)


async def test_max_depth_halt_history_pairing(tmp_path):
    # max_depth=1: the seed (depth 1) is not extendable, so the pool exhausts
    # at the top of iteration 1 — the reachable-on-default-runs break path.
    orch = _make_orch(
        output_dir=str(tmp_path), max_depth=1, max_iterations=50, patience=3,
    )
    _seed_run_mocks(orch)
    result = await orch.run(_tasks(["t1"]))
    # 0 completed generations: seed entry only; NO break-path append.
    assert result.convergence_history == [0.5]
    assert result.oracle_history == []
    assert result.run_status == "completed"  # unchanged by F033
    _assert_pairing_and_disk(orch, result, tmp_path)


async def test_normal_exit_history_pairing_unchanged(tmp_path):
    # Patience exit (the untouched path): exact lists pinned as a regression
    # guard — the F033 deletion must not perturb normal-iteration appends.
    orch = _make_orch(output_dir=str(tmp_path), max_iterations=5, patience=1)
    _seed_run_mocks(orch)
    result = await orch.run(_tasks(["t1"]))
    assert result.convergence_history == [0.5, 0.5]
    assert result.oracle_history == [0.5]
    _assert_pairing_and_disk(orch, result, tmp_path)


# --------------------------------------------------------------------------- #
# F063/F060 — agentic spend-budget / temperature wiring
# --------------------------------------------------------------------------- #


def test_agentic_solver_gets_spend_budget_and_temperature():
    orch = _make_orch(
        use_agentic=True, agentic_spend_budget=50_000, agentic_temperature=0.2,
    )
    solver = orch._build_solver_from_candidate(
        Candidate(candidate_id="c_ag", depth=1)
    )
    assert solver.spend_budget == 50_000
    assert solver.temperature == 0.2


def test_agentic_temperature_default_is_pin():
    # The byte-identity anchor: the config default equals the historical
    # AgenticSolver constructor pin, and the built solver carries it.
    assert EvolutionaryConfig().agentic_temperature == 0.7
    assert EvolutionaryConfig().agentic_spend_budget is None
    orch = _make_orch(use_agentic=True)
    solver = orch._build_solver_from_candidate(
        Candidate(candidate_id="c_pin", depth=1)
    )
    assert solver.temperature == 0.7
    assert solver.spend_budget is None


def test_agentic_solver_threads_no_outer_context():
    # Y4-C_callability-5 (R3-E completion): the agentic branch of
    # _build_solver_from_candidate forwards config.no_outer_context so the E3
    # ablation reaches the AgenticSolver pre_process chain (behavioral at
    # depth>=3, i.e. >=2 injected blocks).
    orch = _make_orch(use_agentic=True, no_outer_context=True)
    cand = Candidate(
        candidate_id="c_noc", depth=2,
        injected_codes=[InjectedCode(pre_process="context = 'x'")],
    )
    assert orch._build_solver_from_candidate(cand).no_outer_context is True


def test_agentic_solver_no_outer_context_defaults_false():
    # OFF-path byte-identity guard: the default config leaves the ctor default.
    orch = _make_orch(use_agentic=True)
    cand = Candidate(
        candidate_id="c_noc_def", depth=2,
        injected_codes=[InjectedCode(pre_process="context = 'x'")],
    )
    assert orch._build_solver_from_candidate(cand).no_outer_context is False


@pytest.mark.parametrize("bad", [0, -5])
def test_agentic_spend_budget_must_be_positive_when_set(bad):
    with pytest.raises(ValueError, match="agentic_spend_budget"):
        _make_orch(agentic_spend_budget=bad)


def test_agentic_spend_budget_none_is_valid():
    orch = _make_orch(agentic_spend_budget=None)
    assert orch.config.agentic_spend_budget is None


# --------------------------------------------------------------------------- #
# F075 — balanced_json_fallback forwarding (plumbing half; module side is
# covered in tests/test_r6b_solver.py)
# --------------------------------------------------------------------------- #


def test_orchestrator_forwards_balanced_json_fallback():
    orch = EvolutionaryOrchestrator(
        llm_client=MagicMock(), executor=MagicMock(), omega=MagicMock(),
        config=EvolutionaryConfig(), solver_language="python",
        balanced_json_fallback=True,
    )
    assert orch.solver.balanced_json_fallback is True


def test_orchestrator_balanced_json_fallback_defaults_off():
    orch = _make_orch()
    assert orch.solver.balanced_json_fallback is False
