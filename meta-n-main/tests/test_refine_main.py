"""Regression tests for the C9_main_cli refinement wave (meta_n/main.py).

Findings covered:
  * F031/F231 — --no-early-stop / --gate-margin / --temperatures wired to the CLI
    with defaults that exactly mirror EvolutionaryConfig.
  * F032 — startup-validation refusals return 1; main() exits 1 (vs budget's 2).
  * F048 — evolutionary_run_kwargs is the single source for EvolutionaryConfig
    AND the config.json provenance block; key order + default values frozen.
  * F054/F227 — --local-exec help says UNSUPPORTED (refusal is by design).
  * F197/F049 — the four OpenEvolve-family benchmarks share one registry-driven
    branch; construction inputs (class / default dir / filter kwarg) frozen.
  * F229 — evolutionary_noop_flag_notes surfaces path-inert flags (warn-only).
  * F230 — --use-agentic without --use-archive is refused (linear path retired).
  * F232 — --instance-workers help covers CO-Bench AND text-classification.

No LLM / Docker / network is touched.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import meta_n.main as main_mod
from meta_n.core.evolutionary_orchestrator import EvolutionaryConfig
from meta_n.main import (
    _OPENEVOLVE_FAMILY,
    async_main,
    evolutionary_noop_flag_notes,
    evolutionary_run_kwargs,
)


# ---------------------------------------------------------------------------
# Helpers (patterns shared with test_auditfix_main_cli / test_main_args)
# ---------------------------------------------------------------------------


def _parse(argv):
    with patch.object(sys, "argv", ["meta-n", *argv]):
        return main_mod.parse_args()


def _capture_parser() -> argparse.ArgumentParser:
    """Run ``parse_args()`` but intercept the final ``parser.parse_args()`` so
    we get the live ArgumentParser without touching sys.argv or exiting."""
    captured = {}
    real = argparse.ArgumentParser.parse_args

    def _spy(self, *a, **k):
        captured["parser"] = self
        return argparse.Namespace()

    argparse.ArgumentParser.parse_args = _spy
    try:
        main_mod.parse_args()
    finally:
        argparse.ArgumentParser.parse_args = real
    return captured["parser"]


def _help_for(parser: argparse.ArgumentParser, option: str) -> str:
    for action in parser._actions:
        if option in (action.option_strings or []):
            return action.help or ""
    raise AssertionError(f"option {option!r} not found on parser")


def _llm_ns(**overrides) -> SimpleNamespace:
    """Minimal namespace that survives async_main's LLM setup (no network)."""
    base = dict(
        model="m", base_url="u", api_key="k", max_tokens=1,
        request_timeout=None, empty_retry_max_tokens=0, exclude_providers=None,
        azure_endpoint=None, azure_api_version="v", daily_budget_usd=None,
        cost_ledger_dir=None, local_exec=False, benchmark=None, tasks=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _write_tasks(tmp_path) -> str:
    tasks_file = tmp_path / "tasks.json"
    tasks_file.write_text(json.dumps([{"task_id": "t1", "description": "d"}]))
    return str(tasks_file)


# ---------------------------------------------------------------------------
# F031/F231 — wire --no-early-stop / --gate-margin / --temperatures
# ---------------------------------------------------------------------------


def test_f031_defaults():
    ns = _parse([])
    assert ns.no_early_stop is False
    assert ns.gate_margin == 0.0
    assert ns.temperatures == [0.5, 0.7, 0.9]


def test_f031_parsing():
    assert _parse(["--gate-margin", "none"]).gate_margin is None
    assert _parse(["--gate-margin", "off"]).gate_margin is None
    assert _parse(["--gate-margin", "0.05"]).gate_margin == 0.05
    assert _parse(["--temperatures", "1.0"]).temperatures == [1.0]
    assert _parse(["--no-early-stop"]).no_early_stop is True


def test_f031_defaults_mirror_dataclass():
    # An all-defaults CLI parse builds an EvolutionaryConfig identical to a
    # bare EvolutionaryConfig() on the three newly-wired fields.
    ns = _parse([])
    built = EvolutionaryConfig(
        **{**evolutionary_run_kwargs(ns), "seed_code_library": None}
    )
    bare = EvolutionaryConfig()
    for f in ("no_early_stop", "gate_margin", "temperatures"):
        assert getattr(built, f) == getattr(bare, f), f


def test_f031_stale_namespace_does_not_attribute_error():
    # Wrapper scripts build Namespaces without the new attrs; getattr defaults
    # must keep evolutionary_run_kwargs from raising a NEW AttributeError.
    ns = _parse([])
    for attr in ("no_early_stop", "gate_margin", "temperatures"):
        delattr(ns, attr)
    kw = evolutionary_run_kwargs(ns)
    assert kw["no_early_stop"] is False
    assert kw["gate_margin"] == 0.0
    assert kw["temperatures"] == [0.5, 0.7, 0.9]


# ---------------------------------------------------------------------------
# F032 — startup-validation refusals exit nonzero
# ---------------------------------------------------------------------------


def test_f032_local_exec_returns_1():
    assert asyncio.run(async_main(SimpleNamespace(local_exec=True))) == 1


def test_f032_no_tasks_or_benchmark_returns_1(capsys, monkeypatch):
    monkeypatch.delenv("META_N_DAILY_BUDGET_USD", raising=False)
    # use_archive=True to pass the retirement guard (which precedes the
    # adapter dispatch) and reach the no-tasks/-benchmark check this targets.
    assert asyncio.run(async_main(_llm_ns(use_archive=True))) == 1
    assert "provide --tasks or --benchmark" in capsys.readouterr().out


def test_f032_resume_without_exp_name_returns_1(tmp_path, monkeypatch):
    monkeypatch.delenv("META_N_DAILY_BUDGET_USD", raising=False)
    ns = _llm_ns(
        tasks=_write_tasks(tmp_path), base_solver=None,
        use_archive=True, resume=True, exp_name=None,
    )
    assert asyncio.run(async_main(ns)) == 1


def test_f032_main_exits_1_on_validation_failure(monkeypatch):
    monkeypatch.setattr(main_mod, "parse_args", lambda: SimpleNamespace())

    async def _fake(args):
        return 1

    monkeypatch.setattr(main_mod, "async_main", _fake)
    with pytest.raises(SystemExit) as ei:
        main_mod.main()
    assert ei.value.code == 1


def test_f032_main_exits_0_on_success(monkeypatch):
    monkeypatch.setattr(main_mod, "parse_args", lambda: SimpleNamespace())

    async def _fake(args):
        return None  # success paths fall off the end

    monkeypatch.setattr(main_mod, "async_main", _fake)
    main_mod.main()  # no SystemExit


# ---------------------------------------------------------------------------
# F048 — evolutionary_run_kwargs consolidation (config.json byte-identity)
# ---------------------------------------------------------------------------

#: Frozen key order: the historical run_config.update literal order (minus
#: "orchestrator") with the F031 provenance keys appended. json.dump preserves
#: dict order, so this list IS the config.json key order for archive runs.
_FROZEN_EVO_KEYS = [
    "beam_width", "beam_candidates", "max_iterations", "patience",
    "gate_tasks", "gate_repeats", "eval_repeats", "paired_eval",
    "consolidate", "protect_floor", "use_inspiration", "no_code_library",
    "no_outer_context", "foster_adoption", "force_code_library_live",
    "verified_code", "deploy_verified_code", "seed_code_library",
    "regression_guard", "regression_guard_repeats", "novelty_alpha", "seed",
    "use_agentic", "agentic_max_turns", "agentic_token_budget",
    "agentic_error_hints", "agentic_preamble", "within_layer_refine",
    "repropagation", "within_task_recursion", "base_solver",
    "agentic_time_limit_s", "agentic_max_budget_usd", "max_docker",
    "scratch_root",
    # F031: new provenance keys, deliberately appended.
    "no_early_stop", "gate_margin", "temperatures",
    # Refine §6b (F006/F034/F035/F063/F060): new provenance keys, deliberately
    # appended (the F031 additive pattern — config.json gains keys only).
    "elite_rotation", "focus_current_headroom", "eval_repeats_gate_topup",
    "agentic_spend_budget", "agentic_temperature",
]


def test_f048_key_order_frozen():
    ns = _parse([])
    assert list(evolutionary_run_kwargs(ns).keys()) == _FROZEN_EVO_KEYS


def test_f048_default_values_match_argparse_defaults():
    ns = _parse([])
    kw = evolutionary_run_kwargs(ns)
    parser = _capture_parser()
    # config-field name -> value derived from a differently-named/negated flag
    special = {
        "use_inspiration": not parser.get_default("no_inspiration"),
        "agentic_time_limit_s": parser.get_default("agent_time_limit"),
        "agentic_max_budget_usd": parser.get_default("agent_max_budget"),
    }
    for key, value in kw.items():
        expected = special[key] if key in special else parser.get_default(key)
        assert value == expected, (key, value, expected)


def test_f048_config_matches_hand_built_explicit_kwargs():
    # The consolidated construction equals the pre-F048 explicit construction
    # field-by-field (all-defaults namespace; seed_code_library None both ways).
    ns = _parse([])
    built = EvolutionaryConfig(
        epsilon=ns.epsilon, max_depth=ns.max_depth, output_dir="./out",
        parallel=ns.parallel, max_retries=ns.max_retries,
        retry_threshold=ns.retry_threshold,
        **{**evolutionary_run_kwargs(ns), "seed_code_library": None},
    )
    legacy = EvolutionaryConfig(
        epsilon=ns.epsilon,
        max_depth=ns.max_depth,
        output_dir="./out",
        parallel=ns.parallel,
        beam_width=ns.beam_width,
        beam_candidates=ns.beam_candidates,
        max_iterations=ns.max_iterations,
        patience=ns.patience,
        gate_tasks=ns.gate_tasks,
        gate_repeats=ns.gate_repeats,
        eval_repeats=ns.eval_repeats,
        paired_eval=ns.paired_eval,
        consolidate=ns.consolidate,
        protect_floor=ns.protect_floor,
        use_inspiration=not ns.no_inspiration,
        no_code_library=ns.no_code_library,
        no_outer_context=ns.no_outer_context,
        foster_adoption=ns.foster_adoption,
        force_code_library_live=ns.force_code_library_live,
        verified_code=ns.verified_code,
        deploy_verified_code=ns.deploy_verified_code,
        seed_code_library=None,
        regression_guard=ns.regression_guard,
        regression_guard_repeats=ns.regression_guard_repeats,
        novelty_alpha=ns.novelty_alpha,
        seed=ns.seed,
        max_retries=ns.max_retries,
        retry_threshold=ns.retry_threshold,
        use_agentic=ns.use_agentic,
        agentic_max_turns=ns.agentic_max_turns,
        agentic_token_budget=ns.agentic_token_budget,
        agentic_error_hints=ns.agentic_error_hints,
        agentic_preamble=ns.agentic_preamble,
        within_layer_refine=ns.within_layer_refine,
        repropagation=ns.repropagation,
        within_task_recursion=ns.within_task_recursion,
        base_solver=ns.base_solver,
        agentic_time_limit_s=ns.agent_time_limit,
        agentic_max_budget_usd=ns.agent_max_budget,
        max_docker=ns.max_docker,
        scratch_root=ns.scratch_root,
    )
    for f in dataclasses.fields(EvolutionaryConfig):
        assert getattr(built, f.name) == getattr(legacy, f.name), f.name


def test_f048_seed_code_library_provenance_is_the_path():
    # The single deliberate dual form: provenance records the PATH.
    ns = _parse(["--seed-code-library", "/tmp/seed.json"])
    assert evolutionary_run_kwargs(ns)["seed_code_library"] == "/tmp/seed.json"


# ---------------------------------------------------------------------------
# F054/F227 — --local-exec help no longer advertises a working capability
# ---------------------------------------------------------------------------


def test_f054_local_exec_help_says_unsupported():
    assert "UNSUPPORTED" in _help_for(_capture_parser(), "--local-exec")


# ---------------------------------------------------------------------------
# F197/F049 — OpenEvolve-family registry
# ---------------------------------------------------------------------------

_EXPECTED_FAMILY = {
    "alphaevolve_math": (
        "meta_n.integrations.openevolve", "AlphaEvolveMathAdapter",
        "./data/openevolve/examples/alphaevolve_math_problems", "problem_names",
    ),
    "symbolic_regression": (
        "meta_n.integrations.openevolve", "SymbolicRegressionAdapter",
        "./data/openevolve/examples/symbolic_regression/problems", "problem_names",
    ),
    "algotune": (
        "meta_n.integrations.openevolve", "AlgoTuneAdapter",
        "./data/openevolve/examples/algotune", "task_names",
    ),
    "arc_agi_2": (
        "meta_n.integrations.arc_agi", "ARCAGI2Adapter",
        "./data/arc_agi_2", "task_ids",
    ),
}


def test_f197_registry_frozen():
    # Pins byte-identity of the construction inputs (class, default dir,
    # filter kwarg) vs the pre-consolidation elif branches.
    assert _OPENEVOLVE_FAMILY == _EXPECTED_FAMILY


def test_f197_registry_keys_subset_of_benchmark_choices():
    parser = _capture_parser()
    choices = None
    for action in parser._actions:
        if "--benchmark" in (action.option_strings or []):
            choices = action.choices
    assert choices is not None
    assert set(_OPENEVOLVE_FAMILY) <= set(choices)


@pytest.mark.parametrize("benchmark", sorted(_OPENEVOLVE_FAMILY))
def test_f197_adapter_constructs_with_registry_kwargs(benchmark, tmp_path):
    # Proves the module path, class name, and filter kwarg stay valid.
    mod, cls, _default_dir, filter_kw = _OPENEVOLVE_FAMILY[benchmark]
    adapter_cls = getattr(importlib.import_module(mod), cls)
    adapter = adapter_cls(data_dir=str(tmp_path), **{filter_kw: None})
    assert isinstance(adapter, adapter_cls)
    assert adapter.data_dir == Path(str(tmp_path))


# ---------------------------------------------------------------------------
# F229 — evolutionary_noop_flag_notes (warn-only, pure)
# ---------------------------------------------------------------------------


def _noop_ns(**overrides) -> SimpleNamespace:
    ns = _parse([])
    ns.use_archive = True
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


def test_f229_defaults_no_notes():
    assert evolutionary_noop_flag_notes(_noop_ns(), external_spine=False) == []
    assert evolutionary_noop_flag_notes(_noop_ns(), external_spine=True) == []


def test_f229_preamble_without_agentic_is_noted():
    notes = evolutionary_noop_flag_notes(
        _noop_ns(agentic_preamble=True, use_agentic=False), external_spine=False
    )
    assert len(notes) == 1 and "--agentic-preamble" in notes[0]


def test_f229_max_turns_consumed_by_spine_not_noted():
    # agentic_max_turns IS consumed by the external-spine solver build.
    notes = evolutionary_noop_flag_notes(
        _noop_ns(agentic_max_turns=9, use_agentic=False), external_spine=True
    )
    assert notes == []


def test_f229_max_docker_without_spine_is_noted():
    notes = evolutionary_noop_flag_notes(
        _noop_ns(max_docker=4), external_spine=False
    )
    assert len(notes) == 1 and "--max-docker" in notes[0]


def test_f229_agent_budget_with_spine_not_noted():
    notes = evolutionary_noop_flag_notes(
        _noop_ns(agent_max_budget=1.0), external_spine=True
    )
    assert notes == []


# ---------------------------------------------------------------------------
# R1-E_omega_context-2 — --no-outer-context inert off the native MetaLayer path
# ---------------------------------------------------------------------------


def test_f229_no_outer_context_agentic_deep_not_noted():
    # Y4-C_callability-5 wired no_outer_context into AgenticSolver
    # (thread_outer_context on _run_all_pre_process), so a deep agentic run
    # genuinely consumes the flag -> no note.
    notes = evolutionary_noop_flag_notes(
        _noop_ns(no_outer_context=True, use_agentic=True, max_depth=3),
        external_spine=False,
    )
    assert not any("--no-outer-context" in n for n in notes)


def test_f229_no_outer_context_agentic_max_depth_2_is_noted():
    # At max_depth<=2 a candidate carries a single injected layer — there is
    # no inter-layer outer_context to thread on the agentic path either.
    notes = evolutionary_noop_flag_notes(
        _noop_ns(no_outer_context=True, use_agentic=True, max_depth=2),
        external_spine=False,
    )
    assert len(notes) == 1 and "--no-outer-context" in notes[0]


def test_f229_no_outer_context_external_spine_is_noted():
    notes = evolutionary_noop_flag_notes(
        _noop_ns(no_outer_context=True), external_spine=True
    )
    assert any("--no-outer-context" in n for n in notes)


def test_f229_force_code_library_live_external_spine_is_noted():
    # R3-A_code_channel-2: the spine's InjectionMapper always stages every safe
    # Ω helper, so --force-code-library-live is inert there -> warn.
    assert any(
        "--force-code-library-live" in n
        for n in evolutionary_noop_flag_notes(
            _noop_ns(force_code_library_live=True), external_spine=True
        )
    )


def test_f229_force_code_library_live_native_not_noted():
    # Native/agentic paths DO honor the demotion gate -> no note.
    assert not any(
        "--force-code-library-live" in n
        for n in evolutionary_noop_flag_notes(
            _noop_ns(force_code_library_live=True), external_spine=False
        )
    )


def test_f229_force_code_library_live_default_no_note():
    assert not any(
        "--force-code-library-live" in n
        for n in evolutionary_noop_flag_notes(_noop_ns(), external_spine=True)
    )


def test_f229_no_outer_context_max_depth_2_is_noted():
    notes = evolutionary_noop_flag_notes(
        _noop_ns(no_outer_context=True, max_depth=2), external_spine=False
    )
    assert any("--no-outer-context" in n for n in notes)


def test_f229_no_outer_context_builtin_deep_not_noted():
    # max_depth>=3 on the builtin native path: the flag IS genuinely wired
    # (a non-outermost MetaLayer can exist) -> must NOT warn.
    notes = evolutionary_noop_flag_notes(
        _noop_ns(no_outer_context=True, max_depth=3), external_spine=False
    )
    assert not any("--no-outer-context" in n for n in notes)


def test_f229_no_outer_context_default_no_note():
    # Flag defaults False -> no --no-outer-context note even on the spine.
    notes = evolutionary_noop_flag_notes(_noop_ns(), external_spine=True)
    assert not any("--no-outer-context" in n for n in notes)


# ---------------------------------------------------------------------------
# Y4-S_stack-1 / Y4-S_stack-3 — gate-fed flags dead where the gate never runs
# ---------------------------------------------------------------------------


def test_f229_within_layer_refine_under_consolidate_is_noted():
    notes = evolutionary_noop_flag_notes(
        _noop_ns(within_layer_refine=True, consolidate=True), external_spine=False
    )
    assert any("--within-layer-refine" in n for n in notes)


def test_f229_within_layer_refine_gate_skipped_is_noted():
    notes = evolutionary_noop_flag_notes(
        _noop_ns(within_layer_refine=True, gate_tasks=0), external_spine=False
    )
    assert any("--within-layer-refine" in n for n in notes)


def test_f229_within_layer_refine_live_combo_not_noted():
    # Gated non-consolidate profile: the refine hook is genuinely reachable.
    notes = evolutionary_noop_flag_notes(
        _noop_ns(within_layer_refine=True, consolidate=False, gate_tasks=3),
        external_spine=False,
    )
    assert not any("--within-layer-refine" in n for n in notes)


def test_f229_gate_topup_under_consolidate_is_noted():
    # R>1 satisfies the old check, but consolidate still kills the gate.
    notes = evolutionary_noop_flag_notes(
        _noop_ns(eval_repeats_gate_topup=True, eval_repeats=3, consolidate=True),
        external_spine=False,
    )
    assert any("--eval-repeats-gate-topup" in n for n in notes)


def test_f229_gate_topup_live_combo_not_noted():
    notes = evolutionary_noop_flag_notes(
        _noop_ns(eval_repeats_gate_topup=True, eval_repeats=3, consolidate=False),
        external_spine=False,
    )
    assert not any("--eval-repeats-gate-topup" in n for n in notes)


# ---------------------------------------------------------------------------
# R1-D_selection-4 — archive-path modifier flags inert without their base flag
# ---------------------------------------------------------------------------


def test_f229_focus_current_headroom_without_base_is_noted():
    notes = evolutionary_noop_flag_notes(
        _noop_ns(focus_current_headroom=True), external_spine=False
    )
    assert len(notes) == 1 and "--focus-current-headroom" in notes[0]


def test_f229_focus_current_headroom_with_both_bases_not_noted():
    notes = evolutionary_noop_flag_notes(
        _noop_ns(focus_current_headroom=True, regression_guard=True, consolidate=True),
        external_spine=False,
    )
    assert notes == []


def test_f229_focus_current_headroom_partial_base_is_noted():
    # regression_guard alone is not enough (consolidate absent)
    notes = evolutionary_noop_flag_notes(
        _noop_ns(focus_current_headroom=True, regression_guard=True),
        external_spine=False,
    )
    assert len(notes) == 1 and "--focus-current-headroom" in notes[0]


def test_f229_within_task_recursion_without_consolidate_is_noted():
    # Same modifier class as --focus-current-headroom: every consumer (DEEPEN
    # directive, focus inheritance, archive depth term) is consolidate-gated.
    notes = evolutionary_noop_flag_notes(
        _noop_ns(within_task_recursion=True), external_spine=False
    )
    assert len(notes) == 1 and "--within-task-recursion" in notes[0]


def test_f229_within_task_recursion_with_consolidate_not_noted():
    notes = evolutionary_noop_flag_notes(
        _noop_ns(within_task_recursion=True, consolidate=True),
        external_spine=False,
    )
    assert notes == []


def test_f229_gate_topup_without_repeats_is_noted():
    notes = evolutionary_noop_flag_notes(
        _noop_ns(eval_repeats_gate_topup=True, eval_repeats=1), external_spine=False
    )
    assert len(notes) == 1 and "--eval-repeats-gate-topup" in notes[0]


def test_f229_gate_topup_with_repeats_not_noted():
    notes = evolutionary_noop_flag_notes(
        _noop_ns(eval_repeats_gate_topup=True, eval_repeats=3), external_spine=False
    )
    assert notes == []


def test_f229_regression_guard_repeats_without_guard_is_noted():
    notes = evolutionary_noop_flag_notes(
        _noop_ns(regression_guard_repeats=5, regression_guard=False),
        external_spine=False,
    )
    assert len(notes) == 1 and "--regression-guard-repeats" in notes[0]


def test_f229_regression_guard_repeats_with_guard_not_noted():
    notes = evolutionary_noop_flag_notes(
        _noop_ns(regression_guard_repeats=5, regression_guard=True),
        external_spine=False,
    )
    assert notes == []


def test_f229_elite_rotation_at_beam_width_1_is_noted():
    # Y4-P_benchmarks-8: elite_rotation is consulted only inside the n>=2
    # reserved-elite block of select_parents; the default --beam-width 1
    # (via _parse) never reaches it.
    notes = evolutionary_noop_flag_notes(
        _noop_ns(elite_rotation=True), external_spine=False
    )
    assert len(notes) == 1 and "--elite-rotation" in notes[0]


def test_f229_elite_rotation_at_beam_width_2_not_noted():
    # Pins the B=2 pool-edge reachability: at beam_width=2 the flag IS
    # consulted (it changes the reserved slot whenever the archive-best sits
    # outside the breedable pool), so a future over-tightening of the gate to
    # beam_width < 3 must fail loudly here.
    notes = evolutionary_noop_flag_notes(
        _noop_ns(elite_rotation=True, beam_width=2), external_spine=False
    )
    assert notes == []


def test_f229_elite_rotation_at_beam_width_3_not_noted():
    notes = evolutionary_noop_flag_notes(
        _noop_ns(elite_rotation=True, beam_width=3), external_spine=False
    )
    assert notes == []


def test_f229_elite_rotation_noted_on_external_spine_too():
    # select_parents runs in the orchestrator on BOTH native and external-spine
    # archive runs, so the beam-width note is spine-independent.
    notes = evolutionary_noop_flag_notes(
        _noop_ns(elite_rotation=True), external_spine=True
    )
    assert len(notes) == 1 and "--elite-rotation" in notes[0]


# ---------------------------------------------------------------------------
# Y4-S_stack-4 / Y4-C_callability-6 — consolidate moots the repropagation hook
# and the gate-family knobs (warn-only)
# ---------------------------------------------------------------------------


def test_f229_consolidate_repropagation_is_noted():
    notes = evolutionary_noop_flag_notes(
        _noop_ns(consolidate=True, repropagation=True), external_spine=False
    )
    assert len(notes) == 1 and "--repropagation" in notes[0]


def test_f229_repropagation_without_consolidate_not_noted():
    # Non-consolidate repropagation is live orchestration — no note.
    notes = evolutionary_noop_flag_notes(
        _noop_ns(consolidate=False, repropagation=True), external_spine=False
    )
    assert not any("--repropagation" in n for n in notes)


def test_f229_consolidate_alone_all_defaults_no_notes():
    # Preserves the notes-silent-by-default invariant: consolidate ON with
    # every gate knob at its default fires nothing.
    assert evolutionary_noop_flag_notes(
        _noop_ns(consolidate=True), external_spine=False
    ) == []


@pytest.mark.parametrize(
    "flag,overrides",
    [
        ("--gate-tasks", {"gate_tasks": 5}),
        ("--gate-repeats", {"gate_repeats": 3}),
        ("--gate-margin", {"gate_margin": 0.1}),
        ("--gate-margin", {"gate_margin": None}),  # 'none' ablation is non-default
        ("--protect-floor", {"protect_floor": 0.05}),
    ],
)
def test_f229_consolidate_gate_knob_each_noted(flag, overrides):
    notes = evolutionary_noop_flag_notes(
        _noop_ns(consolidate=True, **overrides), external_spine=False
    )
    assert len(notes) == 1 and flag in notes[0]
    assert "gate is skipped entirely" in notes[0]


def test_f229_consolidate_gate_knobs_combined_single_note():
    # One combined note naming every mooted gate flag (not one note per flag).
    notes = evolutionary_noop_flag_notes(
        _noop_ns(consolidate=True, gate_tasks=5, gate_margin=0.1,
                 protect_floor=0.05),
        external_spine=False,
    )
    assert len(notes) == 1
    for flag in ("--gate-tasks", "--gate-margin", "--protect-floor"):
        assert flag in notes[0], flag


def test_f229_consolidate_gate_tasks_zero_not_noted():
    # gate_tasks=0 is an explicit gate skip — trivially honored under
    # consolidate, not a misleading no-op.
    notes = evolutionary_noop_flag_notes(
        _noop_ns(consolidate=True, gate_tasks=0), external_spine=False
    )
    assert notes == []


def test_f229_gate_knobs_without_consolidate_not_noted():
    # Off consolidate the gate actually runs — the knobs are live, no note.
    notes = evolutionary_noop_flag_notes(
        _noop_ns(consolidate=False, gate_tasks=5, protect_floor=0.05),
        external_spine=False,
    )
    assert not any("gate is skipped" in n for n in notes)


# ---------------------------------------------------------------------------
# F230 — --use-agentic without --use-archive is refused (linear path retired)
# ---------------------------------------------------------------------------


def test_f230_use_agentic_without_archive_is_refused(tmp_path, capsys, monkeypatch):
    # The linear orchestrator (which downgraded --use-agentic to single-shot)
    # is retired: a no-archive run is refused at startup regardless of
    # --use-agentic.
    monkeypatch.delenv("META_N_DAILY_BUDGET_USD", raising=False)
    ns = _llm_ns(
        tasks=_write_tasks(tmp_path), base_solver=None, resume=False,
        use_agentic=True, use_archive=False,
        # Neutralize the default benchmark-features YAML so it cannot flip
        # use_archive on (which would route evolutionary and skip this guard).
        benchmark_config="none",
    )
    rc = asyncio.run(async_main(ns))
    assert rc == 1
    out = capsys.readouterr().out
    assert "--use-archive is required" in out
    assert "the linear orchestrator has been retired" in out


# ---------------------------------------------------------------------------
# F232 — --instance-workers help covers both consumers
# ---------------------------------------------------------------------------


def test_f232_instance_workers_help_mentions_both_consumers():
    help_text = _help_for(_capture_parser(), "--instance-workers")
    assert "CO-Bench" in help_text
    assert "classification" in help_text
