"""Forensic improvement #3 — WITHIN-TASK RECURSION (``--within-task-recursion``).

Three fused mechanisms behind ONE flag (default OFF): (a) the Ω FOCUS-freeze
directive becomes a DEEPEN directive so a deeper layer RE-WORKS the SAME task its
parent worked; (b) the consolidate focus pick INHERITS the parent's fresh task so
the same task persists down the chain; (c) the archive gains a depth*headroom
selection term that favors extending a deep high-headroom chain.

Flag-OFF byte-identity is gated by ``tests/golden/test_stage23_golden.py`` (FOCUS
freeze verbatim + round-robin focus + zero depth term ⇒ archive/index.json +
summary.json + omega prompt identical). These tests cover:

  * config/archive defaults OFF;
  * flag-OFF parity at the seams (selection weights bit-identical to HEAD, the
    freeze directive string unchanged, gen0/depth-1 carry no directive);
  * flag-ON mechanism (DEEPEN directive, parent-fresh-task focus inheritance with
    saturation release + single-fresh guard, depth*headroom selection term);
  * the depth-probe (``analysis.depth_attribution``) verdict FLIP on a synthetic
    seed→d2→d3→d4 chain: router-stacking (A) FAIL → within-task recursion (B) PASS.

#3 is UNMEASURED build-only (no viable live bed); the depth-probe flip on the
synthetic chain is the only validation, not a score gain.
"""

from __future__ import annotations

import math

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from meta_n.analysis.depth_attribution import (
    deepest_leaf_id,
    split_candidate_mean,
    within_task_depth_profile,
    within_task_verdict,
)
from meta_n.core.archive import Archive, Candidate
from meta_n.core.evolutionary_orchestrator import (
    _WITHIN_TASK_DEPTH_BETA,
    EvolutionaryConfig,
    EvolutionaryOrchestrator,
)
from meta_n.core.meta_layer import InjectedCode, TaskDescription, Trace
from meta_n.core.omega import OmegaEngine


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _ic(d, *, pre=None, lib=None, tsm=None) -> InjectedCode:
    return InjectedCode(
        pre_process=pre, code_library=lib or {}, task_solution_map=tsm or {},
        source_depth=d,
    )


def _head_weights(cands, novelty_alpha=0.3) -> list[float]:
    """The HEAD ``_selection_weights`` formula (rank_norm + ucb), no depth term."""
    n = len(cands)
    finite = [
        c.mean_score if math.isfinite(c.mean_score) else float("-inf")
        for c in cands
    ]
    ln_n = math.log(n) if n > 1 else 0.0
    out = []
    for i, c in enumerate(cands):
        sb = sum(1 for m in finite if m < finite[i])
        rank_norm = sb / (n - 1) if n > 1 else 0.0
        ucb = (
            novelty_alpha * math.sqrt(ln_n / (1 + c.num_children))
            if ln_n > 0.0 else 0.0
        )
        w = rank_norm + ucb
        out.append(w if math.isfinite(w) and w > 1e-6 else 1e-6)
    return out


def _orch(within_task_recursion: bool, *, consolidate: bool = True):
    """A minimal orchestrator with MagicMock deps (no LLM/Docker), like the
    golden capture, just to exercise ``_within_task_focus``."""
    config = EvolutionaryConfig(
        max_depth=4,
        max_iterations=3,
        beam_width=1,
        beam_candidates=1,
        parallel=1,
        seed=42,
        consolidate=consolidate,
        within_task_recursion=within_task_recursion,
    )
    # _within_task_focus reads adapter.score_scale()["hi"] for the saturation
    # ceiling; give the mocked executor's adapter a real [0,1] scale so the
    # saturated-release path exercises the default unit-scale ceiling (1.0)
    # instead of a non-numeric MagicMock stub.
    executor = MagicMock()
    executor.adapter.score_scale.return_value = {"lo": 0.0, "hi": 1.0}
    return EvolutionaryOrchestrator(
        llm_client=MagicMock(),
        executor=executor,
        omega=MagicMock(),
        config=config,
        solver_language="bash",
    )


def _consolidate_parent(depth: int, fresh_task: str, fresh_score: float):
    """A consolidate-shaped parent: ONE task fresh-solved at its own depth, the
    rest inherited frozen at depth 1 (so ``split_candidate_mean`` reports exactly
    one fresh task — the consolidate focus)."""
    return Candidate(
        candidate_id="parent",
        parent_id="gp",
        depth=depth,
        per_task_scores={fresh_task: fresh_score, "t_other": 0.7},
        traces=[
            Trace(task_id=fresh_task, depth=depth, score=fresh_score),
            Trace(task_id="t_other", depth=1, score=0.7),
        ],
    )


# --------------------------------------------------------------------------- #
# defaults OFF
# --------------------------------------------------------------------------- #
def test_config_within_task_recursion_defaults_off():
    assert EvolutionaryConfig().within_task_recursion is False


def test_archive_depth_bonus_defaults_zero():
    assert Archive().within_task_depth_bonus == 0.0


def test_orchestrator_off_archive_has_zero_depth_bonus():
    orch = _orch(within_task_recursion=False)
    assert orch.archive.within_task_depth_bonus == 0.0


def test_orchestrator_on_archive_has_live_depth_bonus():
    orch = _orch(within_task_recursion=True)
    assert orch.archive.within_task_depth_bonus == _WITHIN_TASK_DEPTH_BETA


def test_depth_bonus_requires_consolidate_no_leak():
    # R1-B_depth-1: within_task_recursion is a MODIFIER scoped to consolidate
    # mode (config comment + argparse help). Without --consolidate the archive
    # depth term must NOT fire, else it silently biases parent selection toward
    # deep chains while DEEPEN + focus inheritance stay dormant.
    assert _orch(within_task_recursion=True, consolidate=False).archive.within_task_depth_bonus == 0.0
    assert _orch(within_task_recursion=True, consolidate=True).archive.within_task_depth_bonus == _WITHIN_TASK_DEPTH_BETA


# --------------------------------------------------------------------------- #
# (a) selection weights — flag-OFF bit-identical, flag-ON depth term
# --------------------------------------------------------------------------- #
def _weight_pool():
    shallow = Candidate(
        candidate_id="shallow", depth=2, mean_score=0.5, num_children=0,
        per_task_scores={"t_s": 0.5},
        traces=[Trace(task_id="t_s", depth=2, score=0.5)],
    )
    deep = Candidate(
        candidate_id="deep", depth=4, mean_score=0.5, num_children=0,
        per_task_scores={"t_d": 0.5},
        traces=[Trace(task_id="t_d", depth=4, score=0.5)],
    )
    return shallow, deep


def test_selection_weights_bonus_zero_is_head_identical():
    shallow, deep = _weight_pool()
    arc = Archive(novelty_alpha=0.3, within_task_depth_bonus=0.0)
    arc.add(shallow)
    arc.add(deep)
    pool = [shallow, deep]
    assert arc._selection_weights(pool) == _head_weights(pool, novelty_alpha=0.3)


def test_selection_weights_depth_term_favors_deep_highheadroom():
    shallow, deep = _weight_pool()
    # bonus OFF: identical means + identical children ⇒ identical weights.
    off = Archive(novelty_alpha=0.3, within_task_depth_bonus=0.0)
    off.add(shallow)
    off.add(deep)
    w_off = off._selection_weights([shallow, deep])
    assert w_off[0] == pytest.approx(w_off[1])

    # bonus ON: the deep high-headroom chain outranks the shallow one (depth_norm
    # 4/4 vs 2/4), while the shallow one is still lifted (both have headroom).
    on = Archive(novelty_alpha=0.3, within_task_depth_bonus=_WITHIN_TASK_DEPTH_BETA)
    on.add(shallow)
    on.add(deep)
    w_on = on._selection_weights([shallow, deep])
    assert w_on[1] > w_on[0]                 # deep > shallow
    assert w_on[1] > w_off[1]                # depth term strictly raised the deep weight


def test_fresh_headroom_signal_saturated_task_gets_no_bonus():
    """A chain whose fresh task is at the [0,1] ceiling has no headroom ⇒ 0.0."""
    saturated = Candidate(
        candidate_id="sat", depth=3, mean_score=1.0, num_children=0,
        per_task_scores={"t": 1.0},
        traces=[Trace(task_id="t", depth=3, score=1.0)],
    )
    arc = Archive(within_task_depth_bonus=_WITHIN_TASK_DEPTH_BETA)
    arc.add(saturated)
    assert arc._fresh_headroom_signal(saturated) == 0.0


# --------------------------------------------------------------------------- #
# (b) focus inheritance — _within_task_focus
# --------------------------------------------------------------------------- #
def test_within_task_focus_inherits_parent_fresh_task_when_on():
    orch = _orch(within_task_recursion=True)
    parent = _consolidate_parent(depth=2, fresh_task="t_focus", fresh_score=0.5)
    # cons_targets points at a DIFFERENT task so we can tell inherit from round-robin.
    assert orch._within_task_focus(parent, ["t_roundrobin"], k=0) == "t_focus"


def test_within_task_focus_roundrobin_when_off():
    orch = _orch(within_task_recursion=False)
    parent = _consolidate_parent(depth=2, fresh_task="t_focus", fresh_score=0.5)
    # OFF ⇒ the EXACT existing round-robin pick (cons_targets[k % len]).
    assert orch._within_task_focus(parent, ["t_roundrobin"], k=0) == "t_roundrobin"


def test_within_task_focus_off_empty_targets_is_none():
    orch = _orch(within_task_recursion=False, consolidate=False)
    parent = _consolidate_parent(depth=2, fresh_task="t_focus", fresh_score=0.5)
    assert orch._within_task_focus(parent, [], k=0) is None


def test_within_task_focus_falls_back_when_parent_not_single_fresh():
    """A gen0-seed-shaped parent (every task fresh at depth 1) has >1 fresh tasks
    ⇒ no single task to inherit ⇒ round-robin even when ON."""
    orch = _orch(within_task_recursion=True)
    seed = Candidate(
        candidate_id="gen0_seed", parent_id=None, depth=1,
        per_task_scores={"a": 0.4, "b": 0.5},
        traces=[Trace(task_id="a", depth=1, score=0.4),
                Trace(task_id="b", depth=1, score=0.5)],
    )
    assert orch._within_task_focus(seed, ["t_roundrobin"], k=0) == "t_roundrobin"


def test_within_task_focus_releases_saturated_task():
    """When the parent's single fresh task is already at the [0,1] ceiling there
    is no headroom to deepen ⇒ release back to round-robin (avoid starvation)."""
    orch = _orch(within_task_recursion=True)
    parent = _consolidate_parent(depth=2, fresh_task="t_done", fresh_score=1.0)
    assert orch._within_task_focus(parent, ["t_roundrobin"], k=0) == "t_roundrobin"


# --------------------------------------------------------------------------- #
# (b) inherited-frozen depth clamp — the depth-attribution correctness fix
# --------------------------------------------------------------------------- #
def _seed_ABC():
    """A gen0 seed that fresh-solves A, B, C all at depth 1 (real scripts so the
    inherit-frozen map retains them)."""
    return Candidate(
        candidate_id="gen0_seed", parent_id=None, depth=1,
        per_task_scores={"A": 0.3, "B": 0.5, "C": 0.5},
        traces=[Trace(task_id="A", depth=1, score=0.3, script="solve_a"),
                Trace(task_id="B", depth=1, score=0.5, script="solve_b"),
                Trace(task_id="C", depth=1, score=0.5, script="solve_c")],
    )


def _consolidation_child(orch, cid, focus, tasks, *, focus_score, parent_id):
    """A consolidate child: the inherit-frozen map for ``focus`` (every OTHER
    task at its per-task-best) + this child's OWN fresh focus solve at depth 2 —
    exactly how the breeding loop assembles a consolidation candidate."""
    frozen = orch._consolidation_precomputed(focus, tasks, child_depth=2)
    fresh = Trace(task_id=focus, depth=2, score=focus_score,
                  script=f"solve_{focus.lower()}_d2")
    traces = list(frozen.values()) + [fresh]
    return Candidate(
        candidate_id=cid, parent_id=parent_id, depth=2, traces=traces,
        per_task_scores={t.task_id: t.score for t in traces},
    )


def test_inherited_frozen_depth_clamped_so_split_recovers_focus_when_on():
    """An inherited same-depth sibling trace must NOT be bucketed as the child's
    own fresh solve. With the flag ON the frozen copy's depth is clamped strictly
    below the child depth, so ``split_candidate_mean`` reports exactly ONE fresh
    task (the child's real focus) and ``_within_task_focus`` inherits it."""
    orch = _orch(within_task_recursion=True)
    tasks = [TaskDescription(task_id=t, description=t) for t in ("A", "B", "C")]
    orch.archive.add(_seed_ABC())

    # k0 fresh-solves A at depth 2 and improves it → per-task-best[A] is now a
    # DEPTH-2 trace (the same-depth sibling that used to be mis-attributed).
    k0 = _consolidation_child(orch, "gen1_b0_k0", "A", tasks,
                              focus_score=0.6, parent_id="gen0_seed")
    orch.archive.add(k0)
    assert orch.archive.per_task_best_traces()["A"].depth == 2

    # Now assemble a child focusing B: its inherit-frozen map pulls A's depth-2
    # best, which the clamp lowers to depth 1 (strictly below child_depth=2).
    frozen = orch._consolidation_precomputed("B", tasks, child_depth=2)
    assert frozen["A"].depth == 1        # clamped, not 2 (the fix)

    k1 = _consolidation_child(orch, "gen2_b0_k0", "B", tasks,
                              focus_score=0.7, parent_id="gen1_b0_k0")
    # Exactly one fresh task (B) — A/C are inherited at depth 1 < 2.
    assert split_candidate_mean(k1).fresh_tasks == ["B"]
    # So the focus is INHERITED (B), not the round-robin target 'C'.
    assert orch._within_task_focus(k1, ["C"], k=0) == "B"


def test_inherited_frozen_depth_unchanged_when_off_byte_identity():
    """Companion OFF pin: with the flag OFF the frozen copy keeps its producing
    depth (2) — no clamp, frozen map byte-identical to HEAD."""
    orch = _orch(within_task_recursion=False)
    tasks = [TaskDescription(task_id=t, description=t) for t in ("A", "B", "C")]
    orch.archive.add(_seed_ABC())
    k0 = _consolidation_child(orch, "gen1_b0_k0", "A", tasks,
                              focus_score=0.6, parent_id="gen0_seed")
    orch.archive.add(k0)
    assert orch.archive.per_task_best_traces()["A"].depth == 2
    frozen = orch._consolidation_precomputed("B", tasks, child_depth=2)
    assert frozen["A"].depth == 2        # unchanged (OFF path)


# --------------------------------------------------------------------------- #
# (a) Ω directive — DEEPEN (ON) vs FOCUS-freeze (OFF)
# --------------------------------------------------------------------------- #
_FREEZE_BLOCK = (
    "\n## FOCUS — improve ONLY task 'task_a'\n"
    "Every OTHER task is FROZEN at its best-known solution and will "
    "NOT be re-solved; spend NO effort on them. Make the single "
    "highest-impact change for 'task_a' alone.\n"
)


def _render(focus_task, within_task_recursion):
    engine = OmegaEngine(llm_client=MagicMock())
    traces = [
        Trace(task_id="task_a", depth=2, script="echo a", success=False, score=0.0,
              reasoning="r-a", exit_code=1),
        Trace(task_id="task_b", depth=2, script="echo b", success=True, score=1.0,
              reasoning="r-b", exit_code=0),
    ]
    tasks = [
        TaskDescription(task_id="task_a", description="Solve task_a"),
        TaskDescription(task_id="task_b", description="Solve task_b"),
    ]
    return engine._build_prompt(
        traces, [], tasks, depth=3,
        previous_scores={"task_a": 0.4, "task_b": 0.9},
        archive_best_scores={"task_a": 0.6, "task_b": 1.0},
        solver_language="bash",
        focus_task=focus_task,
        within_task_recursion=within_task_recursion,
    )


def test_omega_freeze_directive_when_off_byte_string():
    prompt = _render("task_a", within_task_recursion=False)
    assert _FREEZE_BLOCK in prompt
    assert "## DEEPEN" not in prompt


def test_omega_deepen_directive_when_on():
    prompt = _render("task_a", within_task_recursion=True)
    assert "## DEEPEN — keep improving ONLY task 'task_a'" in prompt
    assert "improve 'task_a' FURTHER" in prompt
    assert "## FOCUS — improve ONLY task 'task_a'" not in prompt


def test_omega_no_directive_when_no_focus_even_with_flag_on():
    """gen0 / depth-1 (focus_task None) carry NO directive even when ON — neither
    FOCUS nor DEEPEN renders."""
    prompt = _render(None, within_task_recursion=True)
    assert "## DEEPEN" not in prompt
    assert "## FOCUS — improve ONLY" not in prompt


# --------------------------------------------------------------------------- #
# (c) depth-probe FLIP — router-stacking (A) FAIL → within-task recursion (B) PASS
# --------------------------------------------------------------------------- #
def _router_chain_A() -> Archive:
    """ROUTER-STACKING control: each depth FRESH-solves a DISJOINT task via a
    router layer (task_solution_map / task-identity-gated pre_process), every
    other task inherited frozen. Depth == breadth: every task solved ONCE."""
    arc = Archive()
    _gate = ('if "x" in task.task_id:\n    additional_context = "y"\n'
             'else:\n    additional_context = ""')
    arc.add(Candidate(
        candidate_id="s", depth=1, mean_score=0.5,
        per_task_scores={"t1": 0.5},
        traces=[Trace(task_id="t1", depth=1, score=0.5)],
    ))
    arc.add(Candidate(
        candidate_id="d2", parent_id="s", depth=2, mean_score=0.55,
        injected_codes=[_ic(2, tsm={"t2": "frozen"})],
        per_task_scores={"t1": 0.5, "t2": 0.6},
        traces=[Trace(task_id="t1", depth=1, score=0.5),
                Trace(task_id="t2", depth=2, score=0.6)],
    ))
    arc.add(Candidate(
        candidate_id="d3", parent_id="d2", depth=3, mean_score=0.6,
        injected_codes=[_ic(2, tsm={"t2": "frozen"}), _ic(3, pre=_gate)],
        per_task_scores={"t1": 0.5, "t2": 0.6, "t3": 0.7},
        traces=[Trace(task_id="t1", depth=1, score=0.5),
                Trace(task_id="t2", depth=2, score=0.6),
                Trace(task_id="t3", depth=3, score=0.7)],
    ))
    arc.add(Candidate(
        candidate_id="d4", parent_id="d3", depth=4, mean_score=0.65,
        injected_codes=[_ic(2, tsm={"t2": "frozen"}), _ic(3, pre=_gate),
                        _ic(4, tsm={"t4": "frozen"})],
        per_task_scores={"t1": 0.5, "t2": 0.6, "t3": 0.7, "t4": 0.8},
        traces=[Trace(task_id="t1", depth=1, score=0.5),
                Trace(task_id="t2", depth=2, score=0.6),
                Trace(task_id="t3", depth=3, score=0.7),
                Trace(task_id="t4", depth=4, score=0.8)],
    ))
    return arc


def _within_chain_B() -> Archive:
    """WITHIN-TASK-RECURSION: each depth re-FRESH-solves the SAME task 't1' with a
    generic (structure-conditioned) pre_process — depth compounds on one task."""
    arc = Archive()
    pre2 = 'n = task.metadata.get("n", 0)'
    pre3 = 'm = task.metadata.get("m", 0)'
    pre4 = 'k = task.metadata.get("k", 0)'
    arc.add(Candidate(
        candidate_id="s", depth=1, mean_score=0.3,
        per_task_scores={"t1": 0.3},
        traces=[Trace(task_id="t1", depth=1, score=0.3)],
    ))
    arc.add(Candidate(
        candidate_id="d2", parent_id="s", depth=2, mean_score=0.5,
        injected_codes=[_ic(2, pre=pre2)],
        per_task_scores={"t1": 0.5},
        traces=[Trace(task_id="t1", depth=2, score=0.5)],
    ))
    arc.add(Candidate(
        candidate_id="d3", parent_id="d2", depth=3, mean_score=0.7,
        injected_codes=[_ic(2, pre=pre2), _ic(3, pre=pre3)],
        per_task_scores={"t1": 0.7},
        traces=[Trace(task_id="t1", depth=3, score=0.7)],
    ))
    arc.add(Candidate(
        candidate_id="d4", parent_id="d3", depth=4, mean_score=0.9,
        injected_codes=[_ic(2, pre=pre2), _ic(3, pre=pre3), _ic(4, pre=pre4)],
        per_task_scores={"t1": 0.9},
        traces=[Trace(task_id="t1", depth=4, score=0.9)],
    ))
    return arc


def test_depth_probe_router_stacking_A_fails():
    arc = _router_chain_A()
    p = within_task_depth_profile(arc, deepest_leaf_id(arc))
    assert p.max_chain_depth == 4
    assert p.router_fraction == pytest.approx(1.0)   # every injected layer is a router
    assert p.max_multiplicity == 1                   # every task solved exactly once
    assert p.genuine_depth_tasks == []
    v = within_task_verdict(p)
    assert v["verdict"] == "FAIL"
    assert "ROUTER-STACKING" in v["reason"]


def test_depth_probe_within_task_recursion_B_passes():
    arc = _within_chain_B()
    p = within_task_depth_profile(arc, deepest_leaf_id(arc))
    assert p.max_chain_depth == 4
    assert p.router_fraction == 0.0                  # no router layers
    assert p.max_multiplicity == 4                   # t1 re-solved at depths 1,2,3,4
    assert p.genuine_depth_tasks == ["t1"]
    v = within_task_verdict(p)
    assert v["verdict"] == "PASS"


def test_depth_probe_flip_within_task_beats_router():
    """The headline #3 verifier: within-task recursion (B) LOWERS router_fraction
    and RAISES genuine within-task depth vs the consolidate router-stacking (A)."""
    pA = within_task_depth_profile(_router_chain_A(), "d4")
    pB = within_task_depth_profile(_within_chain_B(), "d4")
    # router_fraction DROPS (and crosses below the 0.5 FAIL threshold).
    assert pB.router_fraction < pA.router_fraction
    assert pB.router_fraction < 0.5 <= pA.router_fraction
    # genuine within-task depth RISES.
    assert pB.max_multiplicity > pA.max_multiplicity
    assert pB.max_multiplicity >= 2 and pA.max_multiplicity == 1
    assert len(pB.genuine_depth_tasks) > len(pA.genuine_depth_tasks)
    # verdict flips FAIL → PASS.
    assert within_task_verdict(pA)["verdict"] == "FAIL"
    assert within_task_verdict(pB)["verdict"] == "PASS"


def _disjoint_focus_chain() -> Archive:
    """DISJOINT PER-TASK FOCUS with a GENERIC pre_process: gen0 seed FRESH-solves
    ALL of {t1..t4} at depth 1, then each deeper layer focuses a DIFFERENT task
    (t2@2, t3@3, t4@4) while carrying the rest frozen. No routing (the pre_process
    conditions on task.metadata only), yet no task is re-solved by >= 2 DEEP layers
    — the seed baseline + one deep focus is disjoint per-task breadth, not depth."""
    arc = Archive()
    pre2 = 'n = task.metadata.get("size", 0)'
    pre3 = 'm = task.metadata.get("size", 0)'
    pre4 = 'k = task.metadata.get("size", 0)'
    arc.add(Candidate(
        candidate_id="s", parent_id=None, depth=1, mean_score=0.4,
        per_task_scores={"t1": 0.4, "t2": 0.4, "t3": 0.4, "t4": 0.4},
        traces=[Trace(task_id=t, depth=1, score=0.4) for t in ("t1", "t2", "t3", "t4")],
    ))
    arc.add(Candidate(
        candidate_id="d2", parent_id="s", depth=2, mean_score=0.5,
        injected_codes=[_ic(2, pre=pre2)],
        per_task_scores={"t1": 0.4, "t2": 0.6, "t3": 0.4, "t4": 0.4},
        traces=[Trace(task_id="t1", depth=1, score=0.4),
                Trace(task_id="t2", depth=2, score=0.6),
                Trace(task_id="t3", depth=1, score=0.4),
                Trace(task_id="t4", depth=1, score=0.4)],
    ))
    arc.add(Candidate(
        candidate_id="d3", parent_id="d2", depth=3, mean_score=0.6,
        injected_codes=[_ic(2, pre=pre2), _ic(3, pre=pre3)],
        per_task_scores={"t1": 0.4, "t2": 0.6, "t3": 0.7, "t4": 0.4},
        traces=[Trace(task_id="t1", depth=1, score=0.4),
                Trace(task_id="t2", depth=2, score=0.6),
                Trace(task_id="t3", depth=3, score=0.7),
                Trace(task_id="t4", depth=1, score=0.4)],
    ))
    arc.add(Candidate(
        candidate_id="d4", parent_id="d3", depth=4, mean_score=0.7,
        injected_codes=[_ic(2, pre=pre2), _ic(3, pre=pre3), _ic(4, pre=pre4)],
        per_task_scores={"t1": 0.4, "t2": 0.6, "t3": 0.7, "t4": 0.8},
        traces=[Trace(task_id="t1", depth=1, score=0.4),
                Trace(task_id="t2", depth=2, score=0.6),
                Trace(task_id="t3", depth=3, score=0.7),
                Trace(task_id="t4", depth=4, score=0.8)],
    ))
    return arc


def test_depth_probe_disjoint_focus_generic_preprocess_fails():
    """R1-B_depth-3: seed's universal depth-1 fresh baseline must NOT count toward
    genuine depth. A generic-pre_process chain where each deep layer focuses a
    DIFFERENT task (seed + one deep focus per task) is disjoint per-task breadth,
    not within-task depth — it PASSed on the seed-inflated max and must now FAIL."""
    arc = _disjoint_focus_chain()
    p = within_task_depth_profile(arc, deepest_leaf_id(arc))
    assert p.max_chain_depth == 4
    assert p.router_fraction == 0.0                  # generic pre_process, no routing
    assert p.max_multiplicity == 2                   # seed-INCLUSIVE raw (t2/t3/t4: seed + focus)
    assert p.max_deep_multiplicity == 1              # each deep task touched by ONE deep layer
    assert p.deep_multiplicity == {"t1": 0, "t2": 1, "t3": 1, "t4": 1}
    assert p.genuine_depth_tasks == []
    v = within_task_verdict(p)
    assert v["verdict"] == "FAIL"
    assert "disjoint" in v["reason"] and "router breadth" in v["reason"]
    assert v["max_deep_multiplicity"] == 1
    assert v["max_multiplicity"] == 2                # seed-inclusive field preserved for compat


def _deep_inherited_disjoint_focus_archive(orch) -> Archive:
    """R1-B_depth-2: a PLAIN-consolidate (flag OFF, no within-task clamp)
    disjoint-focus chain whose inherited-frozen traces retain DEEP producing-depths
    STRICTLY GREATER than the inheriting candidate's depth — the general form of the
    case the disjoint-focus test above pins at depth 1.

    seed@1 fresh-solves t1..t7. An OFF-chain a-candidate fresh-solves t2/t3/t4 at
    depth 6 (deep per-task-bests, above the whole b-chain, so they are strictly
    deeper than every inheritor). A late shallow re-breed from the seed —
    b2@2(focus t5), b3@3(focus t6), b4@4(focus t7), b5@5(focus t1) — assembles its
    inherit-frozen map via the REAL OFF-path ``_consolidation_precomputed`` /
    ``per_task_best_traces`` in eval order, so every b-candidate carries
    t2@6/t3@6/t4@6 UNCLAMPED. Under the old ``d >= depth`` rule those deep inherited
    traces were mis-bucketed as the b-candidate's OWN fresh solve (spurious PASS);
    the ``d == depth`` fix buckets them INHERITED."""
    tasks = [TaskDescription(task_id=t, description=t)
             for t in ("t1", "t2", "t3", "t4", "t5", "t6", "t7")]
    orch.archive.add(Candidate(
        candidate_id="gen0_seed", parent_id=None, depth=1,
        per_task_scores={t.task_id: 0.4 for t in tasks},
        traces=[Trace(task_id=t.task_id, depth=1, score=0.4, success=True,
                      script=f"solve_{t.task_id}") for t in tasks],
    ))
    # OFF-chain a-candidate: deep per-task-bests for t2/t3/t4 at depth 6.
    orch.archive.add(Candidate(
        candidate_id="a6", parent_id="gen0_seed", depth=6,
        per_task_scores={t: 0.6 for t in ("t2", "t3", "t4")},
        traces=[Trace(task_id=t, depth=6, score=0.6, success=True,
                      script=f"solve_{t}_d6") for t in ("t2", "t3", "t4")],
    ))

    def _cons_child(cid, focus, depth, parent_id):
        frozen = orch._consolidation_precomputed(focus, tasks, child_depth=depth)
        fresh = Trace(task_id=focus, depth=depth, score=0.6, success=True,
                      script=f"solve_{focus}_d{depth}")
        traces = list(frozen.values()) + [fresh]
        return Candidate(
            candidate_id=cid, parent_id=parent_id, depth=depth, traces=traces,
            per_task_scores={tr.task_id: tr.score for tr in traces},
        )

    for cid, focus, depth, parent_id in (
        ("b2", "t5", 2, "gen0_seed"), ("b3", "t6", 3, "b2"),
        ("b4", "t7", 4, "b3"), ("b5", "t1", 5, "b4"),
    ):
        orch.archive.add(_cons_child(cid, focus, depth, parent_id))
    return orch.archive


def test_depth_probe_plain_consolidate_deep_inherited_frozen_not_fresh():
    """R1-B_depth-2: plain-consolidate inherited-frozen traces at d > the inheriting
    candidate's depth are INHERITED, not fresh — no spurious within-task depth.

    Under the old ``d >= candidate.depth`` rule this archive returned a spurious PASS
    (t2/t3/t4 fresh at four deep chain depths); the ``d == candidate.depth`` fix
    buckets those deep inherited copies as inherited, so the chain correctly FAILs."""
    orch = _orch(within_task_recursion=False)
    arc = _deep_inherited_disjoint_focus_archive(orch)

    # (a) b2@2 focuses t5 only; t2/t3/t4 are inherited at depth 6 (> 2), NOT fresh.
    assert split_candidate_mean(arc.get("b2")).fresh_tasks == ["t5"]

    # (b) no task re-solved by >= 2 STACKED DEEP layers.
    p = within_task_depth_profile(arc, "b5")
    assert p.max_chain_depth == 5
    assert p.max_deep_multiplicity < 2
    assert p.genuine_depth_tasks == []

    # (c) the verdict FAILs (was a spurious PASS under the old d >= depth rule).
    assert within_task_verdict(p)["verdict"] == "FAIL"


_CORPUS_ARCHIVE = (
    Path(__file__).resolve().parents[1]
    / "experiments" / "metan_e2_g9_v3_s42" / "treatment" / "run" / "archive"
)


@pytest.mark.skipif(
    not _CORPUS_ARCHIVE.exists(), reason="on-disk archive corpus absent"
)
def test_deep_inherited_frozen_not_fresh_on_real_corpus():
    """R1-B_depth-2 on real data: gen8_b0_k0 (depth 3) inherits a crew_scheduling
    trace frozen at depth 5 (d > candidate depth); the ``d == depth`` fix keeps it
    OUT of fresh_tasks (the old ``d >= depth`` rule mis-listed it as freshly solved)."""
    from meta_n.analysis.depth_attribution import load_archive

    arc = load_archive(_CORPUS_ARCHIVE)
    assert "crew_scheduling" not in split_candidate_mean(arc.get("gen8_b0_k0")).fresh_tasks


def test_depth_probe_construction_is_monotonic():
    """The probe is read-only: building/profiling never mutates archived
    candidates or their traces (monotonic invariant)."""
    arc = _within_chain_B()
    before = {c.candidate_id: list(c.per_task_scores.items()) for c in arc.candidates}
    within_task_depth_profile(arc, "d4")
    after = {c.candidate_id: list(c.per_task_scores.items()) for c in arc.candidates}
    assert before == after
