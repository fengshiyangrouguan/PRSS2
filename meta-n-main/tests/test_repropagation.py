"""Stage 3 DOWNWARD RE-PROPAGATION — the non-behavioral substrate + the loop.

Covers the plan's Stage-3 gates:
  (a) flag-OFF ⇒ the post-archive.add path never calls re-propagation (byte
      identity additionally proven by tests/golden/test_stage23_golden.py).
  (b) flag-ON ⇒ a synthetic depth-3 chain with a buggy INTERMEDIATE injection is
      regenerated into a NEW candidate (old retained), and the
      error-correction-vs-novel classifier labels a localized fix ERROR-CORRECTION
      while a hand-built novel-helper diff is labelled NOVEL.
  (c) re-propagation NEVER touches gen0 / depth-1 (d_t >= 2), archive monotonic.

Plus the two non-behavioral substrate pieces:
  * run_pre_process per-layer emission capture (OPT-IN; 2-tuple preserved).
  * omega.generate / _build_prompt downstream-feedback param (byte-identical when
    empty/absent — the omega-prompt golden depends on it).

All Ω / solver interaction is stubbed (AsyncMock) — no live LLM, no Docker.
"""

from __future__ import annotations

import json

import pytest
from unittest.mock import AsyncMock, MagicMock

from meta_n.core.archive import Candidate
from meta_n.core.evolutionary_orchestrator import (
    EvolutionaryConfig,
    EvolutionaryOrchestrator,
    EvolutionaryResult,
)
from meta_n.core.meta_layer import (
    InjectedCode,
    TaskDescription,
    Trace,
    run_pre_process,
)
from meta_n.core.omega import OmegaEngine
from meta_n.core.self_repair import classify_repair


def make_tasks(names=None) -> list[TaskDescription]:
    names = names or ["task_a", "task_b"]
    return [TaskDescription(task_id=n, description=f"Solve {n}") for n in names]


# --------------------------------------------------------------------------- #
# (c) classifier — error-correction vs novel over a pre→post injection diff
# --------------------------------------------------------------------------- #

class TestClassifyRepair:
    def test_localized_edit_is_error_correction(self):
        pre = InjectedCode(
            pre_process="result = solve_inner(data)\nadditional_context = str(result)",
            rationale="emit the solved result as context",
            source_depth=2,
        )
        post = InjectedCode(
            pre_process=(
                "result = solve_inner(data)\n"
                "if result is None:\n    result = 0\n"
                "additional_context = str(result)"
            ),
            rationale="emit the solved result as context, guarding a None",
            source_depth=2,
        )
        assert classify_repair(
            pre, post,
            pre_failure_class="Runtime error", post_failure_class="",
        ) == "error_correction"

    def test_new_helper_is_novel(self):
        pre = InjectedCode(
            pre_process="additional_context = hint()",
            rationale="cheap heuristic", source_depth=2,
        )
        post = InjectedCode(
            pre_process="additional_context = dp_solve()",
            code_library={"dp_solve": "def dp_solve():\n    return 1"},
            rationale="exact dynamic program", source_depth=2,
        )
        assert classify_repair(pre, post) == "novel"

    def test_failure_class_shift_is_novel(self):
        pre = InjectedCode(pre_process="x = guard(a)", rationale="guard a", source_depth=2)
        post = InjectedCode(pre_process="x = guard(a)", rationale="guard a", source_depth=2)
        # Identical code + rationale, but the failure mode changed entirely.
        assert classify_repair(
            pre, post,
            pre_failure_class="Numeric instability",
            post_failure_class="Timeout",
        ) == "novel"

    def test_rationale_drift_is_novel(self):
        pre = InjectedCode(
            pre_process="additional_context = run(a)",
            rationale="greedy nearest neighbor sweep", source_depth=2,
        )
        post = InjectedCode(
            pre_process="additional_context = run(a)",
            rationale="exact branch and bound enumeration", source_depth=2,
        )
        assert classify_repair(pre, post) == "novel"

    def test_drastic_rewrite_is_novel(self):
        pre = InjectedCode(pre_process="a = 1", rationale="same idea", source_depth=2)
        post = InjectedCode(
            pre_process="zzzzzz = qqqqqq(wwwwww, eeeeee, rrrrrr, tttttt)",
            rationale="same idea", source_depth=2,
        )
        assert classify_repair(pre, post) == "novel"

    def test_resolved_repair_does_not_read_spurious_class_shift(self):
        # A clean localized fix leaves the post failure class "" → signal 2 must
        # not fire (else every successful repair would be mislabelled novel).
        pre = InjectedCode(
            pre_process="additional_context = render(x)",
            rationale="render x into context", source_depth=2,
        )
        post = InjectedCode(
            pre_process="additional_context = render(x or default)",
            rationale="render x into context with a default", source_depth=2,
        )
        assert classify_repair(
            pre, post, pre_failure_class="Indexing error", post_failure_class="",
        ) == "error_correction"


# --------------------------------------------------------------------------- #
# (a) run_pre_process — plain 2-tuple contract (the opt-in collect_emissions
#     mode was removed: it never grew a production consumer)
# --------------------------------------------------------------------------- #

class TestRunPreProcessEmissions:
    def _codes(self):
        shallow = InjectedCode(
            pre_process="additional_context = 'from-d2'", source_depth=2,
        )
        deep = InjectedCode(
            pre_process="additional_context = 'from-d3'", source_depth=3,
        )
        return [shallow, deep]  # shallowest → deepest

    def test_default_is_two_tuple_unchanged(self):
        ran, ctx = run_pre_process(self._codes(), make_tasks(["t"])[0])
        assert ran is True
        # deepest-first accumulation (d3 then d2).
        assert ctx == "from-d3\nfrom-d2"


# --------------------------------------------------------------------------- #
# (b) omega downstream-feedback param — byte-identical when empty/absent
# --------------------------------------------------------------------------- #

class TestOmegaDownstreamFeedback:
    def _engine(self):
        return OmegaEngine(llm_client=MagicMock())

    def _inputs(self):
        traces = [
            Trace(task_id="task_a", depth=2, script="echo a", success=False,
                  score=0.0, error_summary="boom", exit_code=1),
        ]
        context = [InjectedCode(pre_process="additional_context='below'",
                                rationale="below", source_depth=2)]
        return traces, context, make_tasks()

    def test_absent_vs_none_vs_empty_are_byte_identical(self):
        engine = self._engine()
        traces, context, tasks = self._inputs()
        base = engine._build_prompt(
            traces, context, tasks, depth=3,
            previous_scores={"task_a": 0.4, "task_b": 0.9},
            archive_best_scores={"task_a": 0.6, "task_b": 1.0},
        )
        as_none = engine._build_prompt(
            traces, context, tasks, depth=3,
            previous_scores={"task_a": 0.4, "task_b": 0.9},
            archive_best_scores={"task_a": 0.6, "task_b": 1.0},
            downstream_injections=None,
        )
        as_empty = engine._build_prompt(
            traces, context, tasks, depth=3,
            previous_scores={"task_a": 0.4, "task_b": 0.9},
            archive_best_scores={"task_a": 0.6, "task_b": 1.0},
            downstream_injections=[],
        )
        assert base == as_none == as_empty
        assert "Downstream layers built ON TOP" not in base

    def test_non_empty_renders_section(self):
        engine = self._engine()
        traces, context, tasks = self._inputs()
        above = [InjectedCode(pre_process="additional_context='above'",
                              rationale="the layer built on top", source_depth=3)]
        rendered = engine._build_prompt(
            traces, context, tasks, depth=2,
            downstream_injections=above,
        )
        assert "Downstream layers built ON TOP" in rendered
        assert "error-correction, not a new design" in rendered
        # The above-layer injection is rendered in-context.
        assert "the layer built on top" in rendered


# --------------------------------------------------------------------------- #
# _should_repropagate / _pick_repropagation_depth — the trigger
# --------------------------------------------------------------------------- #

class TestRepropagationTrigger:
    def _orch(self, tmp_path) -> EvolutionaryOrchestrator:
        config = EvolutionaryConfig(
            output_dir=str(tmp_path), max_depth=4, parallel=1, patience=1,
            gate_tasks=1, beam_width=1, beam_candidates=1, repropagation=True,
        )
        return EvolutionaryOrchestrator(
            llm_client=MagicMock(), executor=MagicMock(), omega=MagicMock(),
            config=config, solver_language="bash",
        )

    def _chain(self, child_mean: float):
        """A depth-3 chain: parent(d2) -> child(d3) with non-empty intermediate."""
        d2 = InjectedCode(pre_process="additional_context='d2'", source_depth=2)
        d3 = InjectedCode(pre_process="additional_context='d3'", source_depth=3)
        parent = Candidate(
            candidate_id="gen1_b0_k0", parent_id="gen0_seed", depth=2,
            injected_codes=[d2], mean_score=0.5,
            per_task_scores={"task_a": 0.5},
        )
        child = Candidate(
            candidate_id="gen2_b0_k0", parent_id="gen1_b0_k0", depth=3,
            injected_codes=[d2, d3], mean_score=child_mean,
            per_task_scores={"task_a": child_mean},
        )
        return parent, child

    def test_plateau_triggers(self, tmp_path):
        orch = self._orch(tmp_path)
        parent, child = self._chain(child_mean=0.5)  # delta 0 → plateau
        orch.archive.add(parent)
        assert orch._should_repropagate(child) is True

    def test_improvement_does_not_trigger(self, tmp_path):
        orch = self._orch(tmp_path)
        parent, child = self._chain(child_mean=0.7)  # +0.2 → genuine improvement
        orch.archive.add(parent)
        assert orch._should_repropagate(child) is False

    def test_no_parent_does_not_trigger(self, tmp_path):
        orch = self._orch(tmp_path)
        _, child = self._chain(child_mean=0.5)
        # parent not added to archive → unresolvable → never re-propagate.
        assert orch._should_repropagate(child) is False

    def test_pick_depth_is_deepest_intermediate(self, tmp_path):
        # depth-4 chain: intermediates are d2, d3 → deepest non-empty = d3.
        d2 = InjectedCode(pre_process="additional_context='d2'", source_depth=2)
        d3 = InjectedCode(pre_process="additional_context='d3'", source_depth=3)
        d4 = InjectedCode(pre_process="additional_context='d4'", source_depth=4)
        child = Candidate(candidate_id="c", depth=4, injected_codes=[d2, d3, d4])
        assert EvolutionaryOrchestrator._pick_repropagation_depth(child) == 3

    def test_pick_depth_skips_empty_intermediate(self, tmp_path):
        d2 = InjectedCode(pre_process="additional_context='d2'", source_depth=2)
        empty_d3 = InjectedCode(source_depth=3)  # empty
        d4 = InjectedCode(pre_process="additional_context='d4'", source_depth=4)
        child = Candidate(candidate_id="c", depth=4, injected_codes=[d2, empty_d3, d4])
        # d3 is empty → fall through to d2.
        assert EvolutionaryOrchestrator._pick_repropagation_depth(child) == 2

    def test_pick_depth_none_for_depth_two(self, tmp_path):
        # depth-2 has no INTERMEDIATE layer (range [2, 1] is empty) → never touch
        # gen0 / depth-1.
        d2 = InjectedCode(pre_process="additional_context='d2'", source_depth=2)
        child = Candidate(candidate_id="c", depth=2, injected_codes=[d2])
        assert EvolutionaryOrchestrator._pick_repropagation_depth(child) is None

    # R3-B: the plateau threshold is now scale-aware (OmegaEngine._scale_epsilon),
    # restoring parity with omega.py G5 on non-unit score scales.
    def _wide_chain(self, parent_mean, child_mean, child_task_a):
        """A depth-3 wide-scale chain (union range 200) with non-empty d2."""
        d2 = InjectedCode(pre_process="additional_context='d2'", source_depth=2)
        d3 = InjectedCode(pre_process="additional_context='d3'", source_depth=3)
        parent = Candidate(
            candidate_id="gen1_b0_k0", parent_id="gen0_seed", depth=2,
            injected_codes=[d2], mean_score=parent_mean,
            per_task_scores={"task_a": 0.0, "task_b": 200.0},
        )
        child = Candidate(
            candidate_id="gen2_b0_k0", parent_id="gen1_b0_k0", depth=3,
            injected_codes=[d2, d3], mean_score=child_mean,
            per_task_scores={"task_a": child_task_a, "task_b": 200.0},
        )
        return parent, child

    def test_wide_scale_relative_plateau_fires(self, tmp_path):
        orch = self._orch(tmp_path)
        # union range 200 → eps = 2.0; unit-delta +1.5 < 2.0 ⇒ relative plateau.
        # Old absolute-0.01 code: 1.5 > 0.01 ⇒ returned False (missed the plateau).
        parent, child = self._wide_chain(
            parent_mean=100.0, child_mean=101.5, child_task_a=1.5
        )
        orch.archive.add(parent)
        assert orch._should_repropagate(child) is True

    def test_wide_scale_relative_gain_does_not_fire(self, tmp_path):
        orch = self._orch(tmp_path)
        # union range 200 → eps = 2.0; unit-delta +3.0 > 2.0 ⇒ genuine gain.
        parent, child = self._wide_chain(
            parent_mean=100.0, child_mean=103.0, child_task_a=3.0
        )
        orch.archive.add(parent)
        assert orch._should_repropagate(child) is False

    def test_unit_scale_gain_still_byte_identical(self, tmp_path):
        # [0,1] scores ⇒ eps == 0.01 exactly ⇒ old absolute-threshold outcome.
        orch = self._orch(tmp_path)
        parent, child = self._chain(child_mean=0.7)  # +0.2 on unit scale
        orch.archive.add(parent)
        assert orch._should_repropagate(child) is False


# --------------------------------------------------------------------------- #
# (b) _attempt_repropagation — the kept-candidate construction + monotonic
# --------------------------------------------------------------------------- #

class TestAttemptRepropagation:
    def _orch(self, tmp_path) -> EvolutionaryOrchestrator:
        config = EvolutionaryConfig(
            output_dir=str(tmp_path), max_depth=3, parallel=1, patience=1,
            gate_tasks=1, beam_width=1, beam_candidates=1, repropagation=True,
        )
        return EvolutionaryOrchestrator(
            llm_client=MagicMock(), executor=MagicMock(), omega=MagicMock(),
            config=config, solver_language="bash",
        )

    @pytest.mark.asyncio
    async def test_regenerates_intermediate_and_classifies_error_correction(self, tmp_path):
        orch = self._orch(tmp_path)
        # depth-3 chain with a BUGGY intermediate (d2) injection; full-chain traces
        # are failing (the contamination the top layer couldn't overcome).
        buggy_d2 = InjectedCode(
            pre_process="result = solve_inner(data)\nadditional_context = str(result)",
            rationale="emit the solved result as context", source_depth=2,
        )
        top_d3 = InjectedCode(
            pre_process="additional_context = additional_context + ' refined'",
            rationale="post-process the d2 hint", source_depth=3,
        )
        child = Candidate(
            candidate_id="gen2_b0_k0", parent_id="gen1_b0_k0", iteration=2, depth=3,
            injected_codes=[buggy_d2, top_d3], mean_score=0.2,
            per_task_scores={"task_a": 0.2},
            traces=[Trace(task_id="task_a", depth=3, script="x", success=False,
                          score=0.2, error_summary="IndexError: list index out of range")],
        )
        orch.archive.add(child)

        # Ω regenerates the d2 layer as a LOCALIZED fix (same approach/rationale).
        fixed_d2 = InjectedCode(
            pre_process=(
                "result = solve_inner(data)\n"
                "if result is None:\n    result = 0\n"
                "additional_context = str(result)"
            ),
            rationale="emit the solved result as context, guarding a None",
            source_depth=2,
            raw_omega_prompt="REPROP PROMPT", raw_omega_response="REPROP RESPONSE",
        )
        orch.omega.generate = AsyncMock(return_value=(fixed_d2, 9))
        orch._build_solver_from_candidate = MagicMock(return_value=MagicMock())

        async def fake_eval(cand, solver, tasks, precomputed=None):
            cand.traces = [Trace(task_id="task_a", depth=3, script="ok",
                                 success=True, score=0.8)]
            cand.per_task_scores = {"task_a": 0.8}
            cand.mean_score = 0.8
            return cand
        orch._evaluate_candidate = AsyncMock(side_effect=fake_eval)

        result = EvolutionaryResult()
        reprop, spent = await orch._attempt_repropagation(
            child=child, tasks=make_tasks(["task_a"]), iteration=2,
            temperature=0.7, archive_best_scores=None, result=result,
            out_dir=tmp_path,
        )

        # A NEW candidate was kept (the old child retained → monotonic).
        assert reprop is not None
        assert reprop.candidate_id == "gen2_b0_k0_reprop_d2"
        assert reprop.parent_id == "gen2_b0_k0"      # parent = the revised child
        assert reprop.depth == 3
        assert orch.archive.get("gen2_b0_k0") is child
        assert orch.archive.get("gen2_b0_k0_reprop_d2") is reprop
        # The reprop offspring is an archived child edge of `child` (parent_id=child);
        # the live num_children must be bumped so it equals Archive.rebuild_from_disk's
        # parent_id-edge recompute — no fresh-vs-resume UCB divergence.
        assert child.num_children == 1

        # REPLACE (not augment) the d2 layer; the top d3 layer is untouched.
        assert reprop.injected_codes[0] is fixed_d2
        assert reprop.injected_codes[1] is top_d3
        assert len(reprop.injected_codes) == 2

        # Ω was driven with the layers BELOW d2 (none) + the above layer as
        # downstream feedback + the full-chain traces.
        kw = orch.omega.generate.await_args.kwargs
        assert kw["depth"] == 2
        assert kw["context_stack"] == []
        assert kw["downstream_injections"] == [top_d3]
        assert kw["traces"] == child.traces

        # downstream SelfRepairEvent provenance + classification.
        assert len(reprop.self_repair_events) == 1
        ev = reprop.self_repair_events[0]
        assert ev.granularity == "downstream"
        assert ev.target_depth == 2
        assert ev.parent_candidate_id == "gen2_b0_k0"
        assert ev.candidate_id == "gen2_b0_k0_reprop_d2"
        assert ev.mean_before == pytest.approx(0.2)
        assert ev.mean_after == pytest.approx(0.8)
        assert ev.per_task_delta["task_a"] == pytest.approx(0.6)
        # SR2: accepted = matched-denominator score-improvement verdict;
        # archived = the keep decision (re-propagation archives unconditionally);
        # mean_after_full mirrors the full reprop mean.
        assert ev.accepted is True
        assert ev.archived is True
        assert ev.mean_after_full == pytest.approx(0.8)
        assert ev.pre_code_hash != ev.post_code_hash
        assert ev.classification == "error_correction"

        # Sidecar persisted under the re-propagated candidate's archive dir.
        cand_dir = tmp_path / "archive" / "gen2_b0_k0_reprop_d2"
        assert (cand_dir / "repropagation_d2.json").exists()
        loaded = json.loads((cand_dir / "repropagation_d2.json").read_text())
        assert loaded["granularity"] == "downstream"
        assert loaded["classification"] == "error_correction"
        assert "raw_omega_prompt" not in loaded
        assert spent == reprop.total_tokens

    @pytest.mark.asyncio
    async def test_reprop_previous_scores_is_below_chain_ancestor_baseline(self, tmp_path):
        """R1-B_depth-4: for a d_t>=3 reprop the Ω regression baseline must be the
        below-chain ancestor at depth d_t-1 (its per_task_scores == the cumulative
        effect of layers d_t..top BEFORE the revision), NOT the child's own current
        scores — which would make Section C a degenerate +0.000 delta + an
        unconditional PLATEAU RISK misfire."""
        config = EvolutionaryConfig(
            output_dir=str(tmp_path), max_depth=4, parallel=1, patience=1,
            gate_tasks=1, beam_width=1, beam_candidates=1, repropagation=True,
        )
        orch = EvolutionaryOrchestrator(
            llm_client=MagicMock(), executor=MagicMock(), omega=MagicMock(),
            config=config, solver_language="bash",
        )
        d2 = InjectedCode(pre_process="additional_context='d2'", source_depth=2)
        d3 = InjectedCode(pre_process="additional_context='d3'", source_depth=3)
        d4 = InjectedCode(pre_process="additional_context='d4'", source_depth=4)

        # Lineage archived down to the depth-2 below-chain ancestor (d_t=3 for a
        # depth-4 chain → below=[d2] → below_depth = d_t-1 = 2). Its per_task_scores
        # are DISTINCT from the child's, so we can tell which baseline Ω received.
        seed = Candidate(candidate_id="gen0_seed", parent_id=None, depth=1,
                         per_task_scores={"task_a": 0.9})
        anc_d2 = Candidate(candidate_id="anc_d2", parent_id="gen0_seed", depth=2,
                           injected_codes=[d2], per_task_scores={"task_a": 0.5})
        parent_d3 = Candidate(candidate_id="parent_d3", parent_id="anc_d2", depth=3,
                              injected_codes=[d2, d3], per_task_scores={"task_a": 0.35})
        child = Candidate(
            candidate_id="gen3_b0_k0", parent_id="parent_d3", iteration=3, depth=4,
            injected_codes=[d2, d3, d4], mean_score=0.2,
            per_task_scores={"task_a": 0.2},
            traces=[Trace(task_id="task_a", depth=4, script="x", success=False,
                          score=0.2, error_summary="boom")],
        )
        for c in (seed, anc_d2, parent_d3, child):
            orch.archive.add(c)

        fixed_d3 = InjectedCode(
            pre_process="additional_context='d3 fixed'", source_depth=3,
            raw_omega_prompt="P", raw_omega_response="R",
        )
        orch.omega.generate = AsyncMock(return_value=(fixed_d3, 7))
        orch._build_solver_from_candidate = MagicMock(return_value=MagicMock())

        async def fake_eval(cand, solver, tasks, precomputed=None):
            cand.traces = [Trace(task_id="task_a", depth=4, script="ok",
                                 success=True, score=0.8)]
            cand.per_task_scores = {"task_a": 0.8}
            cand.mean_score = 0.8
            return cand
        orch._evaluate_candidate = AsyncMock(side_effect=fake_eval)

        result = EvolutionaryResult()
        reprop, _spent = await orch._attempt_repropagation(
            child=child, tasks=make_tasks(["task_a"]), iteration=3,
            temperature=0.7, archive_best_scores=None, result=result,
            out_dir=tmp_path,
        )
        assert reprop is not None
        kw = orch.omega.generate.await_args.kwargs
        # d_t=3 (deepest intermediate of the depth-4 chain) → baseline is the
        # depth-2 below-chain ancestor's scores, NOT the child's own {task_a:0.2}.
        assert kw["depth"] == 3
        assert kw["previous_scores"] == {"task_a": 0.5}
        assert kw["previous_scores"] != child.per_task_scores
        assert kw["current_scores"] == {"task_a": 0.2}  # child's authoritative current

    @pytest.mark.asyncio
    async def test_reprop_section_c_nondegenerate_with_ancestor_baseline(self):
        """Render-level companion: with previous_scores = the ancestor baseline
        (!= trace-derived current), Section C emits a real IMPROVED row + non-zero
        Net effect and the +0.000 PLATEAU RISK misfire is gone."""
        engine = OmegaEngine(llm_client=MagicMock())
        traces = [Trace(task_id="task_a", depth=3, script="ok", success=True,
                        score=0.8, reasoning="r")]
        below = [InjectedCode(pre_process="additional_context='d2'", source_depth=2)]
        tasks = make_tasks(["task_a"])

        real = engine._build_prompt(
            traces, below, tasks, depth=3,
            previous_scores={"task_a": 0.5},   # below-chain ancestor baseline
            current_scores={"task_a": 0.8},
            solver_language="bash",
        )
        assert "Tasks that IMPROVED" in real
        assert "0.500 → 0.800 (+0.300)" in real
        assert "Net effect: +0.300" in real
        assert "changed the mean score by only +0.000" not in real

        # Contrast: the pre-fix degenerate baseline (previous == current) renders a
        # +0.000 Net effect AND the unconditional PLATEAU RISK the fix removes.
        degenerate = engine._build_prompt(
            traces, below, tasks, depth=3,
            previous_scores={"task_a": 0.8},
            current_scores={"task_a": 0.8},
            solver_language="bash",
        )
        assert "Net effect: +0.000" in degenerate
        assert "changed the mean score by only +0.000" in degenerate

    @pytest.mark.asyncio
    async def test_novel_helper_regeneration_labelled_novel(self, tmp_path):
        orch = self._orch(tmp_path)
        buggy_d2 = InjectedCode(
            pre_process="additional_context = greedy(data)",
            rationale="greedy heuristic", source_depth=2,
        )
        top_d3 = InjectedCode(pre_process="additional_context += '!'",
                              rationale="top", source_depth=3)
        child = Candidate(
            candidate_id="c", parent_id="p", iteration=2, depth=3,
            injected_codes=[buggy_d2, top_d3], mean_score=0.2,
            per_task_scores={"task_a": 0.2},
            traces=[Trace(task_id="task_a", depth=3, script="x", success=False,
                          score=0.2, error_summary="boom")],
        )
        orch.archive.add(child)
        # Regeneration introduces a NEW helper + a different algorithm class.
        novel_d2 = InjectedCode(
            pre_process="additional_context = dp_exact(data)",
            code_library={"dp_exact": "def dp_exact(d):\n    return 0"},
            rationale="exact dynamic program", source_depth=2,
        )
        orch.omega.generate = AsyncMock(return_value=(novel_d2, 5))
        orch._build_solver_from_candidate = MagicMock(return_value=MagicMock())

        async def fake_eval(cand, solver, tasks, precomputed=None):
            cand.traces = [Trace(task_id="task_a", depth=3, script="ok",
                                 success=True, score=0.3)]
            cand.per_task_scores = {"task_a": 0.3}
            cand.mean_score = 0.3
            return cand
        orch._evaluate_candidate = AsyncMock(side_effect=fake_eval)

        result = EvolutionaryResult()
        reprop, _ = await orch._attempt_repropagation(
            child=child, tasks=make_tasks(["task_a"]), iteration=2,
            temperature=0.7, archive_best_scores=None, result=result,
            out_dir=tmp_path,
        )
        assert reprop is not None
        assert reprop.self_repair_events[0].classification == "novel"

    @pytest.mark.asyncio
    async def test_empty_regeneration_keeps_nothing(self, tmp_path):
        orch = self._orch(tmp_path)
        buggy_d2 = InjectedCode(pre_process="additional_context='x'", source_depth=2)
        top_d3 = InjectedCode(pre_process="additional_context+='y'", source_depth=3)
        child = Candidate(
            candidate_id="c", parent_id="p", iteration=2, depth=3,
            injected_codes=[buggy_d2, top_d3], mean_score=0.2,
            per_task_scores={"task_a": 0.2},
            traces=[Trace(task_id="task_a", success=False, score=0.2)],
        )
        orch.archive.add(child)
        orch.omega.generate = AsyncMock(return_value=(InjectedCode(), 4))  # empty
        orch._build_solver_from_candidate = MagicMock(return_value=MagicMock())
        orch._evaluate_candidate = AsyncMock()

        result = EvolutionaryResult()
        reprop, spent = await orch._attempt_repropagation(
            child=child, tasks=make_tasks(["task_a"]), iteration=2,
            temperature=0.7, archive_best_scores=None, result=result,
            out_dir=tmp_path,
        )
        assert reprop is None
        assert spent == 4
        assert result.total_tokens == 4
        orch._evaluate_candidate.assert_not_awaited()
        assert "c_reprop_d2" not in orch.archive._by_id


# --------------------------------------------------------------------------- #
# Breeding-loop hook — flag ON fires on a plateaued depth-3 child, OFF does not
# --------------------------------------------------------------------------- #

class TestRepropagationHookWiring:
    def _orch(
        self, tmp_path, *, repropagation: bool,
        consolidate: bool = False, within_task_recursion: bool = False,
        **extra,
    ) -> EvolutionaryOrchestrator:
        cfg = dict(
            output_dir=str(tmp_path), max_depth=3, max_iterations=3, patience=3,
            gate_tasks=1, beam_width=1, beam_candidates=1, parallel=1, seed=42,
            repropagation=repropagation, consolidate=consolidate,
            within_task_recursion=within_task_recursion,
        )
        cfg.update(extra)
        config = EvolutionaryConfig(**cfg)
        executor = MagicMock()
        # within_task_recursion reads adapter.score_scale()["hi"]; give a real
        # [0,1] scale so the saturation-release path never hits a MagicMock stub.
        executor.adapter.score_scale.return_value = {"lo": 0.0, "hi": 1.0}
        orch = EvolutionaryOrchestrator(
            llm_client=MagicMock(), executor=executor, omega=MagicMock(),
            config=config, solver_language="bash",
        )
        orch.solver.solve = AsyncMock(return_value=("echo hi", "stub", 10))
        orch.executor.execute = AsyncMock(side_effect=lambda script, task: Trace(
            task_id=task.task_id, depth=1, script="echo hi",
            success=True, score=0.5))
        orch.omega.generate = AsyncMock(return_value=(
            InjectedCode(pre_process="additional_context='m'",
                         rationale="m", source_depth=2), 7))
        return orch

    @pytest.mark.asyncio
    async def test_flag_off_never_calls_repropagation(self, tmp_path):
        orch = self._orch(tmp_path, repropagation=False)
        orch._attempt_repropagation = AsyncMock(return_value=(None, 0))
        await orch.run(make_tasks(["task_a", "task_b"]))
        orch._attempt_repropagation.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_flag_on_calls_repropagation_on_plateaued_depth3_child(self, tmp_path):
        orch = self._orch(tmp_path, repropagation=True)
        # Force the trigger so the test does not depend on gate-RNG scores; assert
        # the hook only ever fires with a depth-3 child.
        orch._should_repropagate = MagicMock(return_value=True)
        orch._attempt_repropagation = AsyncMock(return_value=(None, 0))
        await orch.run(make_tasks(["task_a", "task_b"]))
        assert orch._attempt_repropagation.await_count >= 1
        for call in orch._attempt_repropagation.await_args_list:
            assert call.kwargs["child"].depth == 3

    @pytest.mark.asyncio
    async def test_consolidate_suppresses_repropagation_via_focus_guard(self, tmp_path):
        """Documented design: --consolidate makes every bred candidate carry a
        non-None focus_task, and the reprop hook's ``and not focus_task`` guard
        suppresses repropagation there (reprop re-evaluates the full chain, which
        is incompatible with consolidate's inherit-frozen per-task map). Even with
        repropagation ON, within_task_recursion ON, and _should_repropagate forced
        True (isolating the focus_task guard as the sole suppressor), repropagation
        never fires — the two flags do NOT co-execute in a consolidate run."""
        orch = self._orch(
            tmp_path, repropagation=True, consolidate=True,
            within_task_recursion=True,
            # Run long enough (and breed every extendable parent) that a depth-3
            # child is bred — so the depth guard alone would NOT suppress reprop.
            max_iterations=6, beam_width=4, no_early_stop=True,
        )
        orch._should_repropagate = MagicMock(return_value=True)
        orch._attempt_repropagation = AsyncMock(return_value=(None, 0))
        await orch.run(make_tasks(["task_a", "task_b"]))
        # The focus_task guard suppresses reprop on every consolidate candidate.
        orch._attempt_repropagation.assert_not_awaited()
        # Not vacuous: the run reached depth 3, where the depth guard alone would
        # NOT have suppressed — so the focus_task guard is doing the work.
        assert any(c.depth == 3 for c in orch.archive.candidates)
