"""Tests for the archive-based evolutionary orchestrator."""

import json
import random
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from meta_n.core.archive import Archive, Candidate
from meta_n.core.evolutionary_orchestrator import (
    EvolutionaryConfig,
    EvolutionaryOrchestrator,
    EvolutionaryResult,
)
from meta_n.core.meta_layer import InjectedCode, TaskDescription, Trace


# --- Helpers ---

def make_trace(task_id: str, success: bool = True, score: float = 1.0) -> Trace:
    return Trace(
        task_id=task_id,
        depth=1,
        script=f"def solve(**kwargs): pass  # {task_id}",
        success=success,
        score=score,
        error_summary="" if success else "error",
    )


def make_candidate(
    cid: str, mean_score: float = 0.5, tasks: list[str] | None = None,
    parent_id: str | None = None, depth: int = 1,
    injected_codes: list[InjectedCode] | None = None,
) -> Candidate:
    tasks = tasks or ["task_a", "task_b"]
    traces = [make_trace(t, success=True, score=mean_score) for t in tasks]
    return Candidate(
        candidate_id=cid,
        parent_id=parent_id,
        iteration=0,
        depth=depth,
        injected_codes=injected_codes or [],
        traces=traces,
        pass_at_1=1.0,
        mean_score=mean_score,
        per_task_scores={t: mean_score for t in tasks},
    )


def make_tasks(names: list[str] | None = None) -> list[TaskDescription]:
    names = names or ["task_a", "task_b"]
    return [TaskDescription(task_id=n, description=f"Solve {n}") for n in names]


# --- Archive Tests ---

class TestArchive:
    def test_add_updates_best(self):
        archive = Archive()
        c1 = make_candidate("c1", mean_score=0.5)
        c2 = make_candidate("c2", mean_score=0.8)
        archive.add(c1)
        assert archive.best_mean_score == 0.5
        archive.add(c2)
        assert archive.best_mean_score == 0.8
        assert archive.best_candidate.candidate_id == "c2"

    def test_archive_grows_monotonically(self):
        archive = Archive()
        for i in range(5):
            archive.add(make_candidate(f"c{i}", mean_score=0.1 * i))
        assert len(archive) == 5

    def test_per_task_best_tracking(self):
        archive = Archive()
        c1 = make_candidate("c1", mean_score=0.3)
        c1.traces = [make_trace("task_a", score=0.3), make_trace("task_b", score=0.3)]
        c1.per_task_scores = {"task_a": 0.3, "task_b": 0.3}
        archive.add(c1)

        c2 = make_candidate("c2", mean_score=0.5)
        c2.traces = [make_trace("task_a", score=0.9), make_trace("task_b", score=0.1)]
        c2.per_task_scores = {"task_a": 0.9, "task_b": 0.1}
        archive.add(c2)

        best = archive.per_task_best_scores()
        assert best["task_a"] == 0.9  # from c2
        assert best["task_b"] == 0.3  # from c1

    def test_select_parents_returns_correct_count(self):
        archive = Archive()
        for i in range(5):
            archive.add(make_candidate(f"c{i}", mean_score=0.1 * (i + 1)))
        parents = archive.select_parents(3)
        assert len(parents) == 3

    def test_select_parents_favors_high_score(self):
        archive = Archive()
        low = make_candidate("low", mean_score=0.01)
        high = make_candidate("high", mean_score=0.99)
        archive.add(low)
        archive.add(high)

        rng = random.Random(42)
        counts = {"low": 0, "high": 0}
        for _ in range(1000):
            # Reset num_children to avoid accumulation affecting weights
            low.num_children = 0
            high.num_children = 0
            selected = archive.select_parents(1, rng=rng)
            counts[selected[0].candidate_id] += 1

        assert counts["high"] > counts["low"] * 2  # high should dominate

    def test_select_parents_novelty_bonus(self):
        archive = Archive(novelty_alpha=1.0)  # high novelty weight
        explored = make_candidate("explored", mean_score=0.5)
        explored.num_children = 100  # heavily explored
        fresh = make_candidate("fresh", mean_score=0.5)
        fresh.num_children = 0  # unexplored
        archive.add(explored)
        archive.add(fresh)

        rng = random.Random(42)
        counts = {"explored": 0, "fresh": 0}
        for _ in range(1000):
            explored.num_children = 100
            fresh.num_children = 0
            selected = archive.select_parents(1, rng=rng)
            counts[selected[0].candidate_id] += 1

        assert counts["fresh"] > counts["explored"]

    def test_select_parents_does_not_increment_num_children(self):
        """num_children is incremented by the orchestrator when a child is
        actually added to the archive, not at selection time."""
        archive = Archive()
        c = make_candidate("c1", mean_score=0.5)
        archive.add(c)
        assert c.num_children == 0
        archive.select_parents(3)
        assert c.num_children == 0  # unchanged — orchestrator increments later

    def test_select_parents_empty_archive(self):
        archive = Archive()
        assert archive.select_parents(3) == []

    def test_select_parents_handles_nan_score(self):
        """A candidate with NaN mean_score must not crash rng.choices.

        Regression: openevolve evaluator returned NaN combined_score, which
        propagated to candidate.mean_score and produced NaN weights, crashing
        rng.choices with "Total of weights must be finite".
        """
        archive = Archive()
        good = make_candidate("good", mean_score=0.5)
        bad = make_candidate("bad", mean_score=float("nan"))
        archive.add(good)
        archive.add(bad)

        rng = random.Random(42)
        # Must not raise
        for _ in range(20):
            selected = archive.select_parents(1, rng=rng)
            assert len(selected) == 1
            assert selected[0].candidate_id in {"good", "bad"}

    def test_select_parents_handles_inf_score(self):
        """A candidate with +inf or -inf mean_score must not crash."""
        archive = Archive()
        normal = make_candidate("normal", mean_score=0.5)
        pos_inf = make_candidate("pos_inf", mean_score=float("inf"))
        neg_inf = make_candidate("neg_inf", mean_score=float("-inf"))
        archive.add(normal)
        archive.add(pos_inf)
        archive.add(neg_inf)

        rng = random.Random(42)
        for _ in range(20):
            selected = archive.select_parents(1, rng=rng)
            assert len(selected) == 1

    def test_archive_add_is_idempotent(self):
        """Adding the same candidate_id twice is a no-op (warning logged).

        Regression: Archive.add was non-idempotent. If a crash landed between
        a per-candidate disk save and the checkpoint update, on resume the
        archive rebuild would contain the candidate, then the loop would
        re-generate the same id and call archive.add(...) again, producing
        a duplicate in self.candidates and corrupting per_task_best tracking.
        """
        archive = Archive()
        c1 = make_candidate("c_dup", mean_score=0.4)
        archive.add(c1)
        assert len(archive) == 1
        # Second add with same id: no-op
        c1_duplicate = make_candidate("c_dup", mean_score=0.9)
        archive.add(c1_duplicate)
        assert len(archive) == 1
        # Best score reflects the *first* add, not the duplicate
        assert archive.best_mean_score == 0.4
        # _by_id still points at the original
        assert archive.get("c_dup").mean_score == 0.4

    def test_best_score_for_task_returns_none_when_missing(self):
        """best_score_for_task returns None for tasks that have never been
        evaluated, or where every trace was non-finite. This lets callers
        distinguish 'no data' from a candidate that genuinely scored 0.0.
        """
        archive = Archive()
        # Empty archive
        assert archive.best_score_for_task("anything") is None

        # Add a candidate with a finite score for task_a only
        c = Candidate(candidate_id="c1", traces=[
            make_trace("task_a", success=True, score=0.4),
            make_trace("task_b", success=False, score=float("nan")),
        ], pass_at_1=0.5, mean_score=0.4)
        archive.add(c)
        assert archive.best_score_for_task("task_a") == 0.4
        assert archive.best_score_for_task("task_b") is None  # NaN-only
        assert archive.best_score_for_task("task_unseen") is None

    def test_per_task_best_ignores_nan_score(self):
        """A trace with NaN score must not corrupt per-task best tracking.

        Edge case: if NaN trace lands first for a task, naive `score > current[0]`
        would never replace it (since `finite > NaN` is False), permanently
        poisoning the per-task best.
        """
        archive = Archive()

        # First candidate has a NaN trace for task_x and finite for task_y
        c_nan_first = Candidate(candidate_id="c_nan", traces=[
            make_trace("task_x", success=False, score=float("nan")),
            make_trace("task_y", success=True, score=0.5),
        ], pass_at_1=0.5, mean_score=0.25)
        archive.add(c_nan_first)

        # Second candidate has a finite trace for task_x — must be picked up
        c_finite = Candidate(candidate_id="c_finite", traces=[
            make_trace("task_x", success=True, score=0.7),
        ], pass_at_1=1.0, mean_score=0.7)
        archive.add(c_finite)

        best = archive.per_task_best_scores()
        assert best["task_x"] == 0.7  # not NaN
        assert best["task_y"] == 0.5

    def test_select_parents_nan_candidate_rarely_selected(self):
        """NaN-scored candidates fall back to exploration_bonus only — they
        should be sampled less often than candidates with normal positive scores.
        """
        archive = Archive(novelty_alpha=0.3)
        good = make_candidate("good", mean_score=0.8)
        bad = make_candidate("bad", mean_score=float("nan"))
        archive.add(good)
        archive.add(bad)

        rng = random.Random(42)
        counts = {"good": 0, "bad": 0}
        for _ in range(1000):
            good.num_children = 0
            bad.num_children = 0
            selected = archive.select_parents(1, rng=rng)
            counts[selected[0].candidate_id] += 1
        # good has fitness 0.8 + bonus 0.3, bad has fitness 0 (NaN sanitized) + bonus 0.3
        # ratio ~ 1.1 / 0.3 = 3.67×; require at least 2× for robustness
        assert counts["good"] > counts["bad"] * 2

    def test_get_inspiration_for_failed_tasks(self):
        archive = Archive()
        tasks = make_tasks(["task_a", "task_b"])

        # c1 failed task_a but solved task_b
        c1 = Candidate(candidate_id="c1", traces=[
            make_trace("task_a", success=False, score=0.0),
            make_trace("task_b", success=True, score=0.9),
        ], per_task_scores={"task_a": 0.0, "task_b": 0.9}, mean_score=0.45)
        archive.add(c1)

        # c2 solved task_a but failed task_b
        c2 = Candidate(candidate_id="c2", traces=[
            make_trace("task_a", success=True, score=0.8),
            make_trace("task_b", success=False, score=0.0),
        ], per_task_scores={"task_a": 0.8, "task_b": 0.0}, mean_score=0.4)
        archive.add(c2)

        # Inspiration for c1: should include c2's task_a (c2 did better)
        inspiration = archive.get_inspiration_traces(c1, tasks)
        assert len(inspiration) == 1
        assert inspiration[0].task_id == "task_a"

    def test_get_inspiration_no_failures(self):
        archive = Archive()
        tasks = make_tasks()
        c = make_candidate("c1", mean_score=0.9)
        archive.add(c)
        inspiration = archive.get_inspiration_traces(c, tasks)
        assert inspiration == []

    def test_get_by_id(self):
        archive = Archive()
        c = make_candidate("c1")
        archive.add(c)
        assert archive.get("c1").candidate_id == "c1"

    def test_to_dict(self):
        archive = Archive()
        archive.add(make_candidate("c1", mean_score=0.5))
        d = archive.to_dict()
        assert d["size"] == 1
        assert d["best_mean_score"] == 0.5
        assert len(d["candidates"]) == 1
        assert d["candidates"][0]["candidate_id"] == "c1"


# --- Candidate Tests ---

class TestCandidate:
    def test_serialization_roundtrip(self):
        c = make_candidate("c1", mean_score=0.7, depth=2, injected_codes=[
            InjectedCode(pre_process="x = 1", source_depth=2),
        ])
        data = c.model_dump()
        c2 = Candidate(**data)
        assert c2.candidate_id == "c1"
        assert c2.mean_score == 0.7
        assert len(c2.injected_codes) == 1

    def test_default_values(self):
        c = Candidate(candidate_id="test")
        assert c.depth == 1
        assert c.injected_codes == []
        assert c.num_children == 0
        assert c.mean_score == 0.0


# --- EvolutionaryConfig Tests ---

class TestEvolutionaryConfig:
    def test_inherits_orchestrator_config(self):
        cfg = EvolutionaryConfig(
            epsilon=0.01, max_depth=5, parallel=3,
            beam_width=2, beam_candidates=3, patience=3,
        )
        assert cfg.epsilon == 0.01
        assert cfg.max_depth == 5
        assert cfg.parallel == 3
        assert cfg.beam_width == 2
        assert cfg.beam_candidates == 3
        assert cfg.patience == 3

    def test_default_temperatures(self):
        cfg = EvolutionaryConfig()
        assert cfg.temperatures == [0.5, 0.7, 0.9]


# --- EvolutionaryResult Tests ---

class TestEvolutionaryResult:
    def test_to_dict(self):
        result = EvolutionaryResult(
            archive_size=5,
            total_iterations=3,
            best_mean_score=0.75,
            best_candidate_id="gen2_p0_k1",
        )
        d = result.to_dict()
        assert d["archive_size"] == 5
        assert d["best_mean_score"] == 0.75


# --- EvolutionaryOrchestrator Tests ---

class TestEvolutionaryOrchestrator:
    def _make_orchestrator(self, **config_overrides):
        """Create orchestrator with mocked LLM."""
        llm_client = MagicMock()
        executor = MagicMock()
        omega = MagicMock()
        defaults = dict(
            max_depth=3,
            parallel=1,
            patience=1,
            gate_tasks=0,
            beam_width=1,
            beam_candidates=1,
        )
        defaults.update(config_overrides)
        config = EvolutionaryConfig(**defaults)
        orch = EvolutionaryOrchestrator(
            llm_client=llm_client,
            executor=executor,
            omega=omega,
            config=config,
            solver_language="bash",
        )
        return orch

    @pytest.mark.asyncio
    async def test_seed_only_empty_omega(self):
        """Omega returns empty injection → only seed in archive."""
        orch = self._make_orchestrator()

        # Mock solver.solve
        orch.solver.solve = AsyncMock(return_value=("echo hello", "reasoning", 100))

        # Mock executor
        orch.executor.execute = AsyncMock(return_value=Trace(
            task_id="t1", success=True, score=0.5, script="echo hello",
        ))

        # Omega returns empty
        orch.omega.generate = AsyncMock(return_value=(InjectedCode(), 50))

        tasks = [TaskDescription(task_id="t1", description="test")]
        result = await orch.run(tasks)

        assert result.archive_size == 1  # only seed
        assert result.best_mean_score == 0.5

    @pytest.mark.asyncio
    async def test_single_iteration_improvement(self):
        """Child improves over seed."""
        orch = self._make_orchestrator(patience=2)

        # Solver returns basic solution
        orch.solver.solve = AsyncMock(return_value=("echo hello", "reasoning", 100))

        call_count = 0

        async def mock_execute(script, task, timeout=30):
            nonlocal call_count
            call_count += 1
            # Seed gets 0.3, children get 0.8
            score = 0.3 if call_count <= 1 else 0.8
            return Trace(
                task_id=task.task_id, success=True, score=score, script=script,
            )

        orch.executor.execute = mock_execute

        # Omega returns non-empty injection first time, empty second time (to stop)
        orch.omega.generate = AsyncMock(side_effect=[
            (InjectedCode(pre_process="x=1", source_depth=2), 50),
            (InjectedCode(), 50),  # empty → no child added → patience ticks
            (InjectedCode(), 50),  # empty again → patience exhausted
        ])

        tasks = [TaskDescription(task_id="t1", description="test")]
        result = await orch.run(tasks)

        assert result.archive_size == 2  # seed + 1 child
        assert result.best_mean_score == 0.8

    @pytest.mark.asyncio
    async def test_max_depth_respected(self):
        """Parents at max_depth are not extended."""
        orch = self._make_orchestrator(max_depth=2, patience=1)

        orch.solver.solve = AsyncMock(return_value=("echo hi", "r", 100))
        orch.executor.execute = AsyncMock(return_value=Trace(
            task_id="t1", success=True, score=0.5, script="echo hi",
        ))

        # First Omega call succeeds (depth 2), then returns injection
        # but parent will be at depth 2 (max), so it should be skipped
        orch.omega.generate = AsyncMock(return_value=(
            InjectedCode(pre_process="x=1", source_depth=2), 50
        ))

        tasks = [TaskDescription(task_id="t1", description="test")]
        result = await orch.run(tasks)

        # Seed (depth 1) + child (depth 2). Next gen: child is at max_depth,
        # gets skipped, no candidates added, patience exhausted.
        assert result.archive_size == 2

    @pytest.mark.asyncio
    async def test_gate_check_filters_broken(self):
        """Gate-failing children are not added to archive."""
        orch = self._make_orchestrator(gate_tasks=1, patience=1)

        orch.solver.solve = AsyncMock(return_value=("echo hi", "r", 100))

        call_count = 0

        async def mock_execute(script, task, timeout=30):
            nonlocal call_count
            call_count += 1
            # Seed succeeds, all children fail
            success = call_count <= 1
            return Trace(
                task_id=task.task_id,
                success=success,
                score=1.0 if success else 0.0,
                script=script,
                error_summary="" if success else "broken",
            )

        orch.executor.execute = mock_execute

        orch.omega.generate = AsyncMock(return_value=(
            InjectedCode(pre_process="x=1", source_depth=2), 50
        ))

        tasks = [TaskDescription(task_id="t1", description="test")]
        result = await orch.run(tasks)

        assert result.archive_size == 1  # only seed — child failed gate

    @pytest.mark.asyncio
    async def test_multi_parent_multi_child(self):
        """B=2, K=2 produces up to 4 children per iteration."""
        orch = self._make_orchestrator(beam_width=2, beam_candidates=2, patience=1)

        orch.solver.solve = AsyncMock(return_value=("echo hi", "r", 100))
        orch.executor.execute = AsyncMock(return_value=Trace(
            task_id="t1", success=True, score=0.5, script="echo hi",
        ))

        # Return valid injections for all calls
        orch.omega.generate = AsyncMock(return_value=(
            InjectedCode(pre_process="x=1", source_depth=2), 50
        ))

        tasks = [TaskDescription(task_id="t1", description="test")]
        result = await orch.run(tasks)

        # Gen 0: seed (1 candidate)
        # Gen 1: B=2 parents selected (but only 1 exists, so seed selected twice),
        #   K=2 children each = up to 4 children. All have same score → patience ticks.
        # Archive: 1 seed + up to 4 children = 5
        assert result.archive_size >= 2  # at least seed + some children

    @pytest.mark.asyncio
    async def test_temperature_cycling(self):
        """K candidates get different temperatures."""
        orch = self._make_orchestrator(
            beam_candidates=3,
            patience=1,
            temperatures=[0.3, 0.6, 0.9],
        )

        orch.solver.solve = AsyncMock(return_value=("echo hi", "r", 100))
        orch.executor.execute = AsyncMock(return_value=Trace(
            task_id="t1", success=True, score=0.5, script="echo hi",
        ))

        recorded_temps = []

        async def mock_generate(traces, context_stack, tasks, depth, temperature=None, inspiration_traces=None, previous_scores=None, archive_best_scores=None, solver_language="python", no_code_library=False, **kwargs):
            recorded_temps.append(temperature)
            return InjectedCode(pre_process="x=1", source_depth=depth), 50

        orch.omega.generate = mock_generate

        tasks = [TaskDescription(task_id="t1", description="test")]
        await orch.run(tasks)

        # Should have used all 3 temperatures for the 3 candidates
        assert 0.3 in recorded_temps
        assert 0.6 in recorded_temps
        assert 0.9 in recorded_temps

    @pytest.mark.asyncio
    async def test_result_serialization(self):
        result = EvolutionaryResult(
            archive_size=3,
            total_iterations=2,
            total_tokens=1000,
            best_mean_score=0.75,
            best_candidate_id="gen1_p0_k0",
            per_task_best_scores={"t1": 0.8, "t2": 0.7},
            convergence_history=[0.5, 0.75],
        )
        d = result.to_dict()
        assert d["archive_size"] == 3
        assert d["convergence_history"] == [0.5, 0.75]

    def test_build_solver_from_candidate(self):
        """Solver chain reconstruction produces correct nesting."""
        orch = self._make_orchestrator()
        ic1 = InjectedCode(pre_process="additional_context = 'hint1'", source_depth=2)
        ic2 = InjectedCode(pre_process="additional_context = 'hint2'", source_depth=3)
        candidate = Candidate(
            candidate_id="test",
            depth=3,
            injected_codes=[ic1, ic2],
        )
        solver = orch._build_solver_from_candidate(candidate)

        # Should be MetaLayer wrapping MetaLayer wrapping Layer1Solver
        from meta_n.core.meta_layer import MetaLayer
        assert isinstance(solver, MetaLayer)
        assert solver.depth == 3
        assert isinstance(solver.inner_solver, MetaLayer)
        assert solver.inner_solver.depth == 2

    def test_builtin_base_solver_builds_native_layer1_solver(self):
        """base_solver='builtin' keeps the legacy native path: a depth-1
        candidate must build a plain Layer1Solver (NOT an ExternalAgentSolver
        and NOT a MetaLayer). Regression: 'builtin' is truthy, so the earlier
        `... or self.config.base_solver` checks mistakenly routed it through the
        execute()-only external contract, which Layer1Solver does not implement.
        """
        from meta_n.core.solver import Layer1Solver

        orch = self._make_orchestrator(base_solver="builtin")
        # The spine must NOT be initialized for 'builtin' (it is the native A/B
        # control, not an external agent).
        assert orch._run_guard is None
        assert orch._cost_guard is None
        assert orch._agent_telemetry is None

        seed = Candidate(candidate_id="gen0_seed", iteration=0, depth=1)
        solver = orch._build_solver_from_candidate(seed)
        assert isinstance(solver, Layer1Solver)
        assert solver is orch.solver

    @pytest.mark.asyncio
    async def test_builtin_base_solver_gen0_seed_uses_solve_not_execute(self):
        """A gen0 seed with base_solver='builtin' must dispatch via solve() +
        executor.execute(), never solver.execute(). Pins the blocker fix: the
        native Layer1Solver has no execute(), so a wrong (truthy 'builtin')
        route would AttributeError inside the evaluation gather and abort the
        run. Here a clean run with the expected seed score proves the depth-1
        native solve() path was taken.
        """
        orch = self._make_orchestrator(base_solver="builtin", patience=1)

        # Real Layer1Solver has no execute(); guarantee the test fails loudly if
        # the routing ever calls it again.
        assert not hasattr(orch.solver, "execute")
        orch.solver.solve = AsyncMock(return_value=("echo hello", "reasoning", 100))
        orch.executor.execute = AsyncMock(return_value=Trace(
            task_id="t1", success=True, score=0.5, script="echo hello",
        ))
        # Empty Omega → only the seed lands in the archive.
        orch.omega.generate = AsyncMock(return_value=(InjectedCode(), 50))

        tasks = [TaskDescription(task_id="t1", description="test")]
        result = await orch.run(tasks)

        assert result.archive_size == 1  # seed only
        assert result.best_mean_score == 0.5
        # The native depth-1 path was taken: solve() produced the script and the
        # executor ran it.
        orch.solver.solve.assert_awaited()
        orch.executor.execute.assert_awaited()

    def test_select_temperature(self):
        orch = self._make_orchestrator(temperatures=[0.3, 0.7, 0.9])
        assert orch._select_temperature(0) == 0.3
        assert orch._select_temperature(1) == 0.7
        assert orch._select_temperature(2) == 0.9
        assert orch._select_temperature(3) == 0.3  # wraps around

    @pytest.mark.asyncio
    async def test_evaluate_candidate_filters_nan_from_aggregates(self):
        """A NaN-scored trace must not propagate into candidate.mean_score
        or candidate.per_task_scores.

        Regression: bare `sum(t.score)/len(traces)` produced NaN, which
        downstream wrote into summary.json (json.dumps emits invalid `NaN`)
        and biased archive parent selection.
        """
        import math
        orch = self._make_orchestrator(parallel=2)

        async def mock_execute(script, task, timeout=30):
            # task_a → finite, task_b → NaN
            score = 0.6 if task.task_id == "task_a" else float("nan")
            return Trace(
                task_id=task.task_id, success=score >= 0.5,
                score=score, script=script,
            )

        orch.executor.execute = mock_execute
        orch.solver.solve = AsyncMock(return_value=("echo hi", "r", 100))

        cand = Candidate(candidate_id="c_nan_mix", depth=1)
        tasks = make_tasks(["task_a", "task_b"])
        await orch._evaluate_candidate(cand, orch.solver, tasks)

        # mean_score should be the mean of finite scores only (0.6),
        # not (0.6 + NaN)/2 = NaN.
        assert math.isfinite(cand.mean_score)
        assert cand.mean_score == 0.6
        # per_task_scores should exclude the NaN entry.
        assert "task_a" in cand.per_task_scores
        assert "task_b" not in cand.per_task_scores

    @pytest.mark.asyncio
    async def test_evaluate_candidate_all_nan_zeroes_mean(self):
        """If every trace has a non-finite score, mean_score defaults to 0.0
        rather than NaN."""
        import math
        orch = self._make_orchestrator(parallel=2)

        async def mock_execute(script, task, timeout=30):
            return Trace(
                task_id=task.task_id, success=False,
                score=float("nan"), script=script,
            )

        orch.executor.execute = mock_execute
        orch.solver.solve = AsyncMock(return_value=("echo hi", "r", 100))

        cand = Candidate(candidate_id="c_all_nan", depth=1)
        tasks = make_tasks(["task_a", "task_b"])
        await orch._evaluate_candidate(cand, orch.solver, tasks)

        assert math.isfinite(cand.mean_score)
        assert cand.mean_score == 0.0
        assert cand.per_task_scores == {}

    @pytest.mark.asyncio
    async def test_eval_repeats_stamps_repeat_index_on_spine_solver(self):
        """H6 companion: under eval_repeats>1 a spine solver is stamped with a
        DISTINCT repeat_index per sample so each of the R telemetry rows gets a
        distinct run_id (compute_run_id folds a non-zero repeat_index); without
        it the R rows share coordinates and R-1 are dropped by the run_id de-dup.
        The index is read in execute()'s synchronous prefix, then restored."""
        orch = self._make_orchestrator(eval_repeats=3)
        orch._uses_external_spine = lambda: True

        seen: list = []

        class _FakeSpineSolver:
            async def execute(self, task):
                # repeat_index is read here (mirrors start_record's sync read).
                seen.append(getattr(self, "repeat_index", None))
                return Trace(task_id=task.task_id, success=True, score=0.5), 0

        solver = _FakeSpineSolver()
        cand = Candidate(candidate_id="c_spine", depth=1)
        await orch._evaluate_candidate(cand, solver, make_tasks(["t1"]))

        assert seen == [0, 1, 2]                       # one distinct index / sample
        assert getattr(solver, "repeat_index") == 0    # restored to baseline after

    @pytest.mark.asyncio
    async def test_eval_repeats_1_does_not_stamp_repeat_index(self):
        """H6 default: at eval_repeats=1 the solver is NEVER stamped (the single
        sample reads the getattr default 0) — byte-identical run_id basis."""
        orch = self._make_orchestrator(eval_repeats=1)
        orch._uses_external_spine = lambda: True

        class _FakeSpineSolver:
            async def execute(self, task):
                return Trace(task_id=task.task_id, success=True, score=0.5), 0

        solver = _FakeSpineSolver()
        cand = Candidate(candidate_id="c_spine1", depth=1)
        await orch._evaluate_candidate(cand, solver, make_tasks(["t1"]))
        assert not hasattr(solver, "repeat_index")     # never stamped at R=1

    @pytest.mark.asyncio
    async def test_gate_repeats_stamps_repeat_index_on_spine_solver(self):
        """H6 companion (gate side): under gate_repeats>1 the R gate solves of a
        task get distinct repeat_index stamps (the gate phase separates gate rows
        from eval rows; the repeat index separates the R gate rows). Restored 0."""
        orch = self._make_orchestrator(gate_tasks=1, gate_repeats=3)
        orch._uses_external_spine = lambda: True

        seen: list = []

        class _FakeSpineSolver:
            async def execute(self, task):
                seen.append(getattr(self, "repeat_index", None))
                return Trace(task_id=task.task_id, success=True, score=0.5), 0

        solver = _FakeSpineSolver()
        # depth=2 → gate scale 1 so a gate task is actually sampled.
        cand = Candidate(candidate_id="c_gate", depth=2)
        await orch._gate_check(cand, None, solver, make_tasks(["a", "b", "c", "d"]))

        assert seen == [0, 1, 2]                        # R distinct gate samples
        assert getattr(solver, "repeat_index") == 0     # restored to baseline after


class TestForceCodeLibraryLive:
    """--force-code-library-live / EvolutionaryConfig.force_code_library_live:
    when True the adapter's code_library_is_live() is treated as True
    (un-demote), so Ω's Python helpers are staged + callable even on families
    (CO-Bench/SWE) that normally demote them. Default OFF = byte-identical."""

    def _make_orchestrator(self, demoting: bool = True, **config_overrides):
        llm_client = MagicMock()
        executor = MagicMock()
        omega = MagicMock()
        defaults = dict(
            max_depth=3, parallel=1, patience=1, gate_tasks=0,
            beam_width=1, beam_candidates=1,
        )
        defaults.update(config_overrides)
        config = EvolutionaryConfig(**defaults)
        orch = EvolutionaryOrchestrator(
            llm_client=llm_client, executor=executor, omega=omega,
            config=config, solver_language="python",
        )
        # Simulate a demoting adapter (CO-Bench: code_library_is_live() -> False).
        adapter = MagicMock()
        adapter.code_library_is_live.return_value = not demoting
        orch.adapter = adapter
        return orch

    def test_default_config_field_is_false(self):
        assert EvolutionaryConfig().force_code_library_live is False

    def test_default_honors_demoting_adapter(self):
        """Default OFF: a demoting adapter (CO-Bench) stays demoted."""
        orch = self._make_orchestrator(demoting=True)
        assert orch._code_library_is_live() is False

    def test_force_overrides_demoting_adapter(self):
        """force_code_library_live=True un-demotes even a demoting adapter."""
        orch = self._make_orchestrator(demoting=True, force_code_library_live=True)
        assert orch._code_library_is_live() is True

    def test_force_does_not_disturb_live_adapter(self):
        """A live adapter stays live with or without the override (idempotent)."""
        orch_off = self._make_orchestrator(demoting=False)
        orch_on = self._make_orchestrator(demoting=False, force_code_library_live=True)
        assert orch_off._code_library_is_live() is True
        assert orch_on._code_library_is_live() is True

    def test_default_clears_merged_py_on_demoting_adapter(self):
        """Default OFF: merged Python helpers are CLEARED on a demoting adapter,
        so the outermost MetaLayer carries no code_library."""
        orch = self._make_orchestrator(demoting=True)
        ic = InjectedCode(
            code_library={"helper": "def helper():\n    return 1"},
            source_depth=2,
        )
        candidate = Candidate(candidate_id="c_demote", depth=2, injected_codes=[ic])
        solver = orch._build_solver_from_candidate(candidate)
        from meta_n.core.meta_layer import MetaLayer
        assert isinstance(solver, MetaLayer)
        assert solver.merged_code_library == {}

    def test_force_keeps_merged_py_on_demoting_adapter(self):
        """force_code_library_live=True: merged Python helpers SURVIVE on the
        same demoting adapter and reach the outermost MetaLayer."""
        orch = self._make_orchestrator(demoting=True, force_code_library_live=True)
        ic = InjectedCode(
            code_library={"helper": "def helper():\n    return 1"},
            source_depth=2,
        )
        candidate = Candidate(candidate_id="c_force", depth=2, injected_codes=[ic])
        solver = orch._build_solver_from_candidate(candidate)
        from meta_n.core.meta_layer import MetaLayer
        assert isinstance(solver, MetaLayer)
        assert "helper" in solver.merged_code_library


# --------------------------------------------------------------------------- #
# F6: run_config dumped to config.json at run START (in addition to END)
# --------------------------------------------------------------------------- #

class TestRunConfigDump:
    def _make_orchestrator(self, output_dir):
        config = EvolutionaryConfig(
            output_dir=str(output_dir),
            max_depth=3, parallel=1, patience=1, gate_tasks=0,
            beam_width=1, beam_candidates=1,
        )
        return EvolutionaryOrchestrator(
            llm_client=MagicMock(),
            executor=MagicMock(),
            omega=MagicMock(),
            config=config,
            solver_language="bash",
        )

    def test_write_run_config_creates_config_json(self, tmp_path):
        orch = self._make_orchestrator(tmp_path)
        cfg = {"model": "m", "foster_adoption": True, "bench_tasks": None}
        orch._write_run_config(tmp_path, cfg, stage="start")
        path = tmp_path / "config.json"
        assert path.exists()
        assert json.loads(path.read_text()) == cfg

    def test_write_run_config_none_is_noop(self, tmp_path):
        orch = self._make_orchestrator(tmp_path)
        orch._write_run_config(tmp_path, None, stage="start")
        assert not (tmp_path / "config.json").exists()

    def test_start_write_preserves_end_only_fields(self, tmp_path):
        """A START re-write (e.g. resume) must not clobber the richer END dump's
        extra fields; it only refreshes the keys it shares."""
        orch = self._make_orchestrator(tmp_path)
        orch._write_run_config(tmp_path, {"a": 1, "end_only": 9}, stage="end")
        orch._write_run_config(tmp_path, {"a": 2}, stage="start")
        merged = json.loads((tmp_path / "config.json").read_text())
        assert merged["a"] == 2          # START refreshes the shared key
        assert merged["end_only"] == 9   # END-only field survives

    def test_end_write_overwrites_full_payload(self, tmp_path):
        """The END dump is authoritative — full overwrite, no merge."""
        orch = self._make_orchestrator(tmp_path)
        orch._write_run_config(tmp_path, {"a": 1, "stale": 7}, stage="start")
        orch._write_run_config(tmp_path, {"a": 2}, stage="end")
        out = json.loads((tmp_path / "config.json").read_text())
        assert out == {"a": 2}

    @pytest.mark.asyncio
    async def test_run_writes_config_json_at_start(self, tmp_path):
        orch = self._make_orchestrator(tmp_path)
        orch.solver.solve = AsyncMock(return_value=("echo hi", "r", 10))
        orch.executor.execute = AsyncMock(return_value=Trace(
            task_id="t1", success=True, score=0.5, script="echo hi"))
        orch.omega.generate = AsyncMock(return_value=(InjectedCode(), 5))
        cfg = {"model": "m", "bench_tasks": None, "foster_adoption": False}
        tasks = [TaskDescription(task_id="t1", description="x")]
        await orch.run(tasks, run_config=cfg)
        path = tmp_path / "config.json"
        assert path.exists()
        # run() performs ONLY the START dump (save_results writes the END dump);
        # so the on-disk content equals exactly what we passed in.
        assert json.loads(path.read_text()) == cfg

    @pytest.mark.asyncio
    async def test_run_without_run_config_writes_no_config_json(self, tmp_path):
        """Default callers (no run_config) leave config.json absent — START dump
        is a no-op, preserving prior behavior byte-identically."""
        orch = self._make_orchestrator(tmp_path)
        orch.solver.solve = AsyncMock(return_value=("echo hi", "r", 10))
        orch.executor.execute = AsyncMock(return_value=Trace(
            task_id="t1", success=True, score=0.5, script="echo hi"))
        orch.omega.generate = AsyncMock(return_value=(InjectedCode(), 5))
        tasks = [TaskDescription(task_id="t1", description="x")]
        await orch.run(tasks)
        assert not (tmp_path / "config.json").exists()
