"""Stage 2 WITHIN-LAYER REFINE — OmegaEngine.refine + the gate-fail hook.

Covers the three plan gates:
  (a) flag-OFF ⇒ the gate-fail branch never calls refine (byte-identity is
      additionally proven by tests/golden/test_stage23_golden.py).
  (b) flag-ON ⇒ a gate-failed child with a deliberately-buggy injection is
      repaired into a NEW candidate that recovers the planted bug and logs a
      ``within_layer`` SelfRepairEvent (monotonic — the rejected child is never
      archived/mutated).
  (c) refine NEVER fires on gen0 / depth-1 / empty injections (empty injections
      ``continue`` before the gate, so neither the gate nor refine runs).

All Ω / solver interaction is stubbed (AsyncMock) — no live LLM, no Docker.
"""

from __future__ import annotations

import json

import pytest
from unittest.mock import AsyncMock, MagicMock

from meta_n.core.evolutionary_orchestrator import (
    EvolutionaryConfig,
    EvolutionaryOrchestrator,
    EvolutionaryResult,
)
from meta_n.core.archive import Candidate
from meta_n.core.meta_layer import InjectedCode, TaskDescription, Trace
from meta_n.core.omega import OmegaEngine


# --- Helpers ---------------------------------------------------------------

def make_tasks(names=None) -> list[TaskDescription]:
    names = names or ["task_a"]
    return [TaskDescription(task_id=n, description=f"Solve {n}") for n in names]


def _seed(cid="gen0_seed", score=0.8) -> Candidate:
    """A depth-1 archived parent (no injected_codes)."""
    return Candidate(
        candidate_id=cid,
        parent_id=None,
        iteration=0,
        depth=1,
        injected_codes=[],
        traces=[Trace(task_id="task_a", depth=1, script="echo x",
                      success=True, score=score)],
        mean_score=score,
        per_task_scores={"task_a": score},
    )


# --------------------------------------------------------------------------- #
# OmegaEngine.refine — reuses _build_prompt + a FIX-THE-BUG directive
# --------------------------------------------------------------------------- #

class TestOmegaRefine:
    def _engine(self, response: str):
        llm = MagicMock()
        llm.complete = AsyncMock(return_value=(response, 11))
        return OmegaEngine(llm_client=llm), llm

    def _buggy(self) -> InjectedCode:
        return InjectedCode(
            pre_process="value = items[idx]  # BUG: idx undefined",
            rationale="buggy approach",
            source_depth=2,
        )

    @pytest.mark.asyncio
    async def test_refine_parses_corrected_injection(self):
        resp = (
            "```rationale\nFixed the undefined index.\n```\n"
            "```pre_process\nadditional_context = 'fixed'\n```\n"
        )
        engine, llm = self._engine(resp)
        traces = [Trace(task_id="task_a", depth=2, script="echo a",
                        success=False, score=0.0, error_summary="NameError: idx")]
        injected, tokens = await engine.refine(
            prev_injection=self._buggy(),
            child_traces=traces,
            context_stack=[],
            tasks=make_tasks(),
            depth=2,
            mean_before=0.2,
        )
        assert tokens == 11
        assert injected.pre_process == "additional_context = 'fixed'"
        assert injected.rationale == "Fixed the undefined index."
        assert injected.source_depth == 2
        # The rendered refine prompt is paired onto the injection for provenance.
        assert injected.raw_omega_prompt
        llm.complete.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_refine_prompt_contains_directive_and_buggy_code(self):
        engine, _ = self._engine("```pre_process\nadditional_context='ok'\n```")
        buggy = self._buggy()
        traces = [Trace(task_id="task_a", depth=2, script="echo a",
                        success=False, score=0.0, error_summary="NameError: idx")]
        injected, _ = await engine.refine(
            prev_injection=buggy,
            child_traces=traces,
            context_stack=[],
            tasks=make_tasks(),
            depth=2,
            mean_before=0.2,
        )
        prompt = injected.raw_omega_prompt
        # The FIX-THE-BUG directive frames it as error-correction.
        assert "REFINE — FIX YOUR OWN BUGGY INJECTION" in prompt
        assert "FIX THE BUG" in prompt
        assert "do NOT change the approach" in prompt
        # The buggy injection is rendered in-context as the layer to repair.
        assert "BUG: idx undefined" in prompt
        # A known mean renders the concrete score.
        assert "scored only a mean of 0.200" in prompt

    @pytest.mark.asyncio
    async def test_refine_directive_gatefail_when_mean_unknown(self):
        engine, _ = self._engine("```pre_process\nadditional_context='ok'\n```")
        injected, _ = await engine.refine(
            prev_injection=self._buggy(),
            child_traces=[Trace(task_id="task_a", success=False, score=0.0)],
            context_stack=[],
            tasks=make_tasks(),
            depth=2,
            mean_before=None,
        )
        assert "FAILED THE QUALITY GATE" in injected.raw_omega_prompt

    def test_pre_process_hash_distinguishes_code(self):
        a = InjectedCode(pre_process="x = 1")
        b = InjectedCode(pre_process="x = 2")
        ha = EvolutionaryOrchestrator._pre_process_hash(a)
        hb = EvolutionaryOrchestrator._pre_process_hash(b)
        assert ha != hb
        # Same code ⇒ same hash (deterministic).
        assert ha == EvolutionaryOrchestrator._pre_process_hash(
            InjectedCode(pre_process="x = 1")
        )


# --------------------------------------------------------------------------- #
# _attempt_within_layer_refine — the kept-candidate construction (gate b)
# --------------------------------------------------------------------------- #

class TestAttemptWithinLayerRefine:
    def _orch(self, tmp_path) -> EvolutionaryOrchestrator:
        config = EvolutionaryConfig(
            output_dir=str(tmp_path),
            max_depth=3, parallel=1, patience=1, gate_tasks=1,
            beam_width=1, beam_candidates=1, within_layer_refine=True,
        )
        return EvolutionaryOrchestrator(
            llm_client=MagicMock(), executor=MagicMock(), omega=MagicMock(),
            config=config, solver_language="bash",
        )

    @pytest.mark.asyncio
    async def test_refine_recovers_bug_and_logs_event(self, tmp_path):
        orch = self._orch(tmp_path)
        parent = _seed()
        orch.archive.add(parent)

        buggy_injection = InjectedCode(
            pre_process="raise ValueError('planted bug')",
            rationale="buggy", source_depth=2,
        )
        buggy_child = Candidate(
            candidate_id="gen1_b0_k0", parent_id="gen0_seed",
            iteration=1, depth=2, injected_codes=[buggy_injection],
        )
        gate_traces = {"task_a": Trace(
            task_id="task_a", depth=2, script="x", success=False, score=0.1,
            error_summary="ValueError: planted bug")}

        fixed = InjectedCode(
            pre_process="additional_context = 'fixed'",
            rationale="fixed the bug", source_depth=2,
            raw_omega_prompt="REFINE PROMPT BODY",
            raw_omega_response="REFINE RESPONSE BODY",
        )
        orch.omega.refine = AsyncMock(return_value=(fixed, 7))
        orch._build_solver_from_candidate = MagicMock(return_value=MagicMock())
        # Refined injection now CLEARS the gate.
        orch._gate_check = AsyncMock(return_value=(True, 1, {
            "task_a": Trace(task_id="task_a", depth=2, script="ok",
                            success=True, score=0.8)}))

        async def fake_eval(cand, solver, tasks, precomputed=None):
            cand.traces = [Trace(task_id="task_a", depth=2, script="ok",
                                 success=True, score=0.8)]
            cand.per_task_scores = {"task_a": 0.8}
            cand.mean_score = 0.8
            return cand
        orch._evaluate_candidate = AsyncMock(side_effect=fake_eval)

        result = EvolutionaryResult()
        refined, spent = await orch._attempt_within_layer_refine(
            parent=parent, buggy_child=buggy_child,
            buggy_injection=buggy_injection, gate_traces=gate_traces,
            tasks=make_tasks(), iteration=1, child_depth=2, temperature=0.7,
            previous_scores=None, archive_best_scores=None,
            result=result, out_dir=tmp_path,
        )

        # A NEW candidate was kept.
        assert refined is not None
        assert refined.candidate_id == "gen1_b0_k0_refine"
        assert refined.parent_id == "gen0_seed"  # archived-parent lineage intact
        assert refined.depth == 2
        assert refined.injected_codes == [fixed]
        assert refined.mean_score == 0.8

        # Monotonic: the refined candidate is archived; the rejected child is NOT.
        assert orch.archive.get("gen1_b0_k0_refine") is refined
        assert "gen1_b0_k0" not in orch.archive._by_id

        # The Ω refine was driven with the buggy injection + a partial-signal mean.
        orch.omega.refine.assert_awaited_once()
        kw = orch.omega.refine.await_args.kwargs
        assert kw["prev_injection"] is buggy_injection
        assert kw["depth"] == 2
        assert kw["mean_before"] == pytest.approx(0.1)

        # within_layer SelfRepairEvent provenance.
        assert len(refined.self_repair_events) == 1
        ev = refined.self_repair_events[0]
        assert ev.granularity == "within_layer"
        assert ev.target_depth == 2
        assert ev.accepted is True
        assert ev.candidate_id == "gen1_b0_k0_refine"
        assert ev.parent_candidate_id == "gen1_b0_k0"
        assert ev.mean_before == pytest.approx(0.1)
        assert ev.mean_after == pytest.approx(0.8)
        assert ev.per_task_delta["task_a"] == pytest.approx(0.7)
        # The planted bug was actually changed (pre/post hashes differ).
        assert ev.pre_code_hash != ev.post_code_hash
        assert ev.raw_omega_prompt == "REFINE PROMPT BODY"

        # Sidecar persisted under the refined candidate's archive dir.
        cand_dir = tmp_path / "archive" / "gen1_b0_k0_refine"
        assert (cand_dir / "repropagation_d2.json").exists()
        loaded = json.loads((cand_dir / "repropagation_d2.json").read_text())
        assert loaded["granularity"] == "within_layer"
        assert "raw_omega_prompt" not in loaded  # excluded from JSON sidecar
        assert spent == refined.total_tokens

    @pytest.mark.asyncio
    async def test_event_scores_matched_denominator_subset(self, tmp_path):
        """SR1: mean_after is scored over the SAME keys as before_scores (the
        gate subset), NOT the full evaluated set, so accepted is a comparable
        verdict; archived (the keep bit) is separate and can diverge.

        The gate sees only ``task_a`` (subset), but the full eval also covers
        ``task_b``. The full mean (0.5) is up vs the gate's 0.1 — yet on the
        MATCHED gate task the refine actually REGRESSED (0.1→0.05), so the
        score-improvement verdict must be ``accepted=False`` while the candidate
        is still ``archived=True`` (it cleared the gate)."""
        orch = self._orch(tmp_path)
        parent = _seed()
        orch.archive.add(parent)

        buggy_injection = InjectedCode(
            pre_process="raise ValueError('planted bug')",
            rationale="buggy", source_depth=2,
        )
        buggy_child = Candidate(
            candidate_id="gen1_b0_k0", parent_id="gen0_seed",
            iteration=1, depth=2, injected_codes=[buggy_injection],
        )
        # Gate subset = {task_a: 0.1} only.
        gate_traces = {"task_a": Trace(
            task_id="task_a", depth=2, script="x", success=False, score=0.1,
            error_summary="ValueError: planted bug")}

        fixed = InjectedCode(
            pre_process="additional_context = 'fixed'",
            rationale="fixed the bug", source_depth=2,
        )
        orch.omega.refine = AsyncMock(return_value=(fixed, 7))
        orch._build_solver_from_candidate = MagicMock(return_value=MagicMock())
        orch._gate_check = AsyncMock(return_value=(True, 1, {
            "task_a": Trace(task_id="task_a", depth=2, script="ok",
                            success=True, score=0.05)}))

        async def fake_eval(cand, solver, tasks, precomputed=None):
            # Full set: task_a REGRESSED to 0.05, task_b high → full mean 0.5.
            cand.per_task_scores = {"task_a": 0.05, "task_b": 0.95}
            cand.traces = [
                Trace(task_id="task_a", depth=2, script="ok", success=True, score=0.05),
                Trace(task_id="task_b", depth=2, script="ok", success=True, score=0.95),
            ]
            cand.mean_score = 0.5
            return cand
        orch._evaluate_candidate = AsyncMock(side_effect=fake_eval)

        result = EvolutionaryResult()
        refined, _ = await orch._attempt_within_layer_refine(
            parent=parent, buggy_child=buggy_child,
            buggy_injection=buggy_injection, gate_traces=gate_traces,
            tasks=make_tasks(["task_a", "task_b"]), iteration=1, child_depth=2,
            temperature=0.7, previous_scores=None, archive_best_scores=None,
            result=result, out_dir=tmp_path,
        )
        assert refined is not None
        ev = refined.self_repair_events[0]
        # mean_after is the gate-SUBSET mean (task_a only = 0.05), NOT the full 0.5.
        assert ev.mean_before == pytest.approx(0.1)
        assert ev.mean_after == pytest.approx(0.05)
        assert ev.mean_after_full == pytest.approx(0.5)
        # Matched-denominator verdict: the gate task regressed → NOT accepted.
        assert ev.accepted is False
        # Keep decision is independent: it cleared the gate → archived.
        assert ev.archived is True
        # per_task_delta keyed on the gate subset only (no task_b leak).
        assert set(ev.per_task_delta) == {"task_a"}
        assert ev.per_task_delta["task_a"] == pytest.approx(-0.05)

    @pytest.mark.asyncio
    async def test_refine_still_gatefails_keeps_nothing(self, tmp_path):
        orch = self._orch(tmp_path)
        parent = _seed()
        orch.archive.add(parent)
        buggy = InjectedCode(pre_process="raise ValueError('x')", source_depth=2)
        buggy_child = Candidate(candidate_id="gen1_b0_k0", parent_id="gen0_seed",
                                iteration=1, depth=2, injected_codes=[buggy])
        gate_traces = {"task_a": Trace(task_id="task_a", success=False, score=0.1)}
        orch.omega.refine = AsyncMock(return_value=(
            InjectedCode(pre_process="still_broken = True", source_depth=2), 5))
        orch._build_solver_from_candidate = MagicMock(return_value=MagicMock())
        orch._gate_check = AsyncMock(return_value=(False, 1, {}))  # still fails
        orch._evaluate_candidate = AsyncMock()

        result = EvolutionaryResult()
        refined, spent = await orch._attempt_within_layer_refine(
            parent=parent, buggy_child=buggy_child, buggy_injection=buggy,
            gate_traces=gate_traces, tasks=make_tasks(), iteration=1,
            child_depth=2, temperature=0.7, previous_scores=None,
            archive_best_scores=None, result=result, out_dir=tmp_path,
        )
        assert refined is None
        assert spent == 5  # the Ω refine call only
        assert result.total_tokens == 5
        orch._evaluate_candidate.assert_not_awaited()
        assert "gen1_b0_k0_refine" not in orch.archive._by_id

    @pytest.mark.asyncio
    async def test_empty_refine_keeps_nothing(self, tmp_path):
        orch = self._orch(tmp_path)
        parent = _seed()
        orch.archive.add(parent)
        buggy = InjectedCode(pre_process="raise ValueError('x')", source_depth=2)
        buggy_child = Candidate(candidate_id="gen1_b0_k0", parent_id="gen0_seed",
                                iteration=1, depth=2, injected_codes=[buggy])
        gate_traces = {"task_a": Trace(task_id="task_a", success=False, score=0.1)}
        orch.omega.refine = AsyncMock(return_value=(InjectedCode(), 3))  # empty
        orch._gate_check = AsyncMock()

        result = EvolutionaryResult()
        refined, spent = await orch._attempt_within_layer_refine(
            parent=parent, buggy_child=buggy_child, buggy_injection=buggy,
            gate_traces=gate_traces, tasks=make_tasks(), iteration=1,
            child_depth=2, temperature=0.7, previous_scores=None,
            archive_best_scores=None, result=result, out_dir=tmp_path,
        )
        assert refined is None
        assert spent == 3
        orch._gate_check.assert_not_awaited()  # never even gets to building a candidate
        assert "gen1_b0_k0_refine" not in orch.archive._by_id

    @pytest.mark.asyncio
    async def test_regate_outer_spend_threaded_into_returned_tokens(self, tmp_path):
        """Lock-step contract: on a KEPT refine the re-gate's outer solve spend is
        threaded into the returned tokens_spent (not just result.total_tokens), so
        the caller's per-iteration iter_tokens stays reconciled with the
        authoritative outer counter. Wire a REAL cumulative_usage dict so
        _outer_cumulative_total() moves and the mocked gate bumps it."""
        orch = self._orch(tmp_path)
        orch.llm_client.cumulative_usage = {"total": 0}  # real dict -> non-zero delta
        parent = _seed()
        orch.archive.add(parent)
        buggy = InjectedCode(pre_process="raise ValueError('bug')", source_depth=2)
        buggy_child = Candidate(candidate_id="gen1_b0_k0", parent_id="gen0_seed",
                                iteration=1, depth=2, injected_codes=[buggy])
        gate_traces = {"task_a": Trace(task_id="task_a", depth=2, script="x",
                                       success=False, score=0.1)}
        fixed = InjectedCode(pre_process="additional_context='fixed'", source_depth=2)
        orch.omega.refine = AsyncMock(return_value=(fixed, 7))
        orch._build_solver_from_candidate = MagicMock(return_value=MagicMock())

        async def gate_bump(*a, **k):
            orch.llm_client.cumulative_usage["total"] += 100  # re-gate outer spend
            return (True, 1, {"task_a": Trace(task_id="task_a", depth=2, script="ok",
                                              success=True, score=0.8)})
        orch._gate_check = AsyncMock(side_effect=gate_bump)

        async def fake_eval(cand, solver, tasks, precomputed=None):
            cand.traces = [Trace(task_id="task_a", depth=2, script="ok",
                                 success=True, score=0.8)]
            cand.per_task_scores = {"task_a": 0.8}
            cand.mean_score = 0.8
            return cand
        orch._evaluate_candidate = AsyncMock(side_effect=fake_eval)

        result = EvolutionaryResult()
        refined, spent = await orch._attempt_within_layer_refine(
            parent=parent, buggy_child=buggy_child, buggy_injection=buggy,
            gate_traces=gate_traces, tasks=make_tasks(), iteration=1, child_depth=2,
            temperature=0.7, previous_scores=None, archive_best_scores=None,
            result=result, out_dir=tmp_path,
        )
        assert refined is not None
        # lock-step: returned tokens_spent == what was added to result.total_tokens.
        assert spent == result.total_tokens
        assert spent == refined.total_tokens + 100  # eval spend + re-gate outer spend

    @pytest.mark.asyncio
    async def test_regate_outer_spend_threaded_on_regate_fail(self, tmp_path):
        """Same lock-step contract on REJECTION: a still-gate-failing refine
        returns refine_tokens + the re-gate outer spend (matching what was added
        to result.total_tokens), and never evaluates."""
        orch = self._orch(tmp_path)
        orch.llm_client.cumulative_usage = {"total": 0}
        parent = _seed()
        orch.archive.add(parent)
        buggy = InjectedCode(pre_process="raise ValueError('bug')", source_depth=2)
        buggy_child = Candidate(candidate_id="gen1_b0_k0", parent_id="gen0_seed",
                                iteration=1, depth=2, injected_codes=[buggy])
        gate_traces = {"task_a": Trace(task_id="task_a", depth=2, script="x",
                                       success=False, score=0.1)}
        fixed = InjectedCode(pre_process="additional_context='fixed'", source_depth=2)
        orch.omega.refine = AsyncMock(return_value=(fixed, 7))
        orch._build_solver_from_candidate = MagicMock(return_value=MagicMock())

        async def gate_bump(*a, **k):
            orch.llm_client.cumulative_usage["total"] += 100
            return (False, 1, {})
        orch._gate_check = AsyncMock(side_effect=gate_bump)
        orch._evaluate_candidate = AsyncMock()

        result = EvolutionaryResult()
        refined, spent = await orch._attempt_within_layer_refine(
            parent=parent, buggy_child=buggy_child, buggy_injection=buggy,
            gate_traces=gate_traces, tasks=make_tasks(), iteration=1, child_depth=2,
            temperature=0.7, previous_scores=None, archive_best_scores=None,
            result=result, out_dir=tmp_path,
        )
        assert refined is None
        assert spent == result.total_tokens   # 7 (refine) + 100 (re-gate)
        assert spent == 7 + 100
        orch._evaluate_candidate.assert_not_awaited()


# --------------------------------------------------------------------------- #
# Breeding-loop hook — flag ON fires on gate-fail, flag OFF does not (gates a/c)
# --------------------------------------------------------------------------- #

class TestRefineHookWiring:
    def _orch(self, tmp_path, *, within_layer_refine: bool) -> EvolutionaryOrchestrator:
        config = EvolutionaryConfig(
            output_dir=str(tmp_path),
            max_depth=2, max_iterations=1, patience=1, gate_tasks=1,
            beam_width=1, beam_candidates=1, parallel=1, seed=42,
            within_layer_refine=within_layer_refine,
        )
        orch = EvolutionaryOrchestrator(
            llm_client=MagicMock(), executor=MagicMock(), omega=MagicMock(),
            config=config, solver_language="bash",
        )
        orch.solver.solve = AsyncMock(return_value=("echo hi", "stub", 10))
        orch.executor.execute = AsyncMock(side_effect=lambda script, task: Trace(
            task_id=task.task_id, depth=1, script="echo hi",
            success=True, score=0.5))
        orch.omega.generate = AsyncMock(return_value=(
            InjectedCode(pre_process="additional_context='buggy'",
                         rationale="buggy", source_depth=2), 7))
        # Force the bred child to FAIL the gate so the gate-fail branch is hit.
        orch._gate_check = AsyncMock(return_value=(False, 1, {
            "task_a": Trace(task_id="task_a", depth=2, script="x",
                            success=False, score=0.0, error_summary="bug")}))
        return orch

    @pytest.mark.asyncio
    async def test_flag_off_never_calls_refine(self, tmp_path):
        orch = self._orch(tmp_path, within_layer_refine=False)
        orch._attempt_within_layer_refine = AsyncMock(return_value=(None, 0))
        await orch.run(make_tasks(["task_a"]))
        orch._attempt_within_layer_refine.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_flag_on_calls_refine_on_gate_fail(self, tmp_path):
        orch = self._orch(tmp_path, within_layer_refine=True)
        orch._attempt_within_layer_refine = AsyncMock(return_value=(None, 0))
        await orch.run(make_tasks(["task_a"]))
        orch._attempt_within_layer_refine.assert_awaited_once()
        kw = orch._attempt_within_layer_refine.await_args.kwargs
        assert kw["buggy_injection"].pre_process == "additional_context='buggy'"
        assert kw["child_depth"] == 2
        assert kw["buggy_child"].candidate_id == "gen1_b0_k0"

    @pytest.mark.asyncio
    async def test_refine_success_branch_saves_without_scope_error(self, tmp_path):
        """When refine returns a kept candidate, the loop's success branch runs
        _save_running_summary + _save_checkpoint with the in-scope iteration
        vars (prev_best/prev_oracle/patience_counter) — exercise it end-to-end so
        a scoping regression there would crash run()."""
        orch = self._orch(tmp_path, within_layer_refine=True)
        kept = Candidate(
            candidate_id="gen1_b0_k0_refine", parent_id="gen0_seed",
            iteration=1, depth=2,
            injected_codes=[InjectedCode(pre_process="additional_context='ok'",
                                         source_depth=2)],
            traces=[Trace(task_id="task_a", depth=2, script="ok",
                          success=True, score=0.9)],
            mean_score=0.9, per_task_scores={"task_a": 0.9},
        )

        async def fake_refine(**kwargs):
            orch.archive.add(kept)  # mirror the real helper's archive.add
            return kept, 12
        orch._attempt_within_layer_refine = AsyncMock(side_effect=fake_refine)

        await orch.run(make_tasks(["task_a"]))  # must not raise
        orch._attempt_within_layer_refine.assert_awaited_once()
        assert "gen1_b0_k0_refine" in orch.archive._by_id

    @pytest.mark.asyncio
    async def test_empty_injection_skips_gate_and_refine(self, tmp_path):
        """An empty Ω injection ``continue``s BEFORE the gate, so neither the gate
        nor the refine fires — gen0-vanilla parity for the empty case."""
        orch = self._orch(tmp_path, within_layer_refine=True)
        orch.omega.generate = AsyncMock(return_value=(InjectedCode(), 7))  # empty
        orch._attempt_within_layer_refine = AsyncMock(return_value=(None, 0))
        await orch.run(make_tasks(["task_a"]))
        orch._attempt_within_layer_refine.assert_not_awaited()
        orch._gate_check.assert_not_awaited()
