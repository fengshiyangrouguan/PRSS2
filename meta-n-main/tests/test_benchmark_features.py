"""Benchmark-features YAML config (metacognition stack ON by default).

The bundled ``configs/benchmark_features.yaml`` is loaded by ``meta_n.main`` by
default and turns on the metacognition stack (use_archive / consolidate /
regression_guard + companions, per-family foster_adoption, noisy-classification
eval_repeats) WITHOUT changing any code default. ``--benchmark-config none``
reproduces the bare pre-metacognition behavior.

Covered:
  * apply_benchmark_config precedence (defaults / per-bench / CLI-explicit /
    unknown benchmark / missing arg attr).
  * the bundled YAML parses and carries the expected blocks.
  * --benchmark-config resolution ('none' skips; 'auto' → bundled path; else path).
  * build_base_run_config records the additive provenance keys.
  * the defaults block honors the companion-requirement invariants.
  * shipped-config self-consistency: no block carries an inert (gate-fed)
    key, and no benchmark round-trip triggers an F229 no-op note.

No LLM / Docker / network is touched.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml

import meta_n.main as main_mod
from meta_n.main import (
    apply_benchmark_config,
    build_base_run_config,
    _benchmark_config_path,
    _user_set_dests,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
YAML_PATH = REPO_ROOT / "meta_n" / "configs" / "benchmark_features.yaml"


def _parse(argv):
    with patch.object(sys, "argv", ["meta-n", *argv]):
        return main_mod.parse_args()


def _bundled_cfg() -> dict:
    return yaml.safe_load(YAML_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# apply_benchmark_config — precedence
# ---------------------------------------------------------------------------


def test_defaults_applied_to_every_benchmark():
    args = SimpleNamespace(benchmark="terminal_bench", use_archive=False, consolidate=False)
    cfg = {"defaults": {"use_archive": True, "consolidate": True}, "terminal_bench": {}}
    applied = apply_benchmark_config(args, set(), cfg)
    assert args.use_archive is True
    assert args.consolidate is True
    assert applied == ["consolidate", "use_archive"]


def test_per_bench_block_overrides_defaults():
    args = SimpleNamespace(
        benchmark="symptom2disease", use_archive=False, eval_repeats=1
    )
    cfg = {
        "defaults": {"use_archive": True, "eval_repeats": 1},
        "symptom2disease": {"eval_repeats": 3},
    }
    apply_benchmark_config(args, set(), cfg)
    assert args.use_archive is True
    assert args.eval_repeats == 3  # per-bench beats defaults


def test_cli_explicit_wins_over_yaml():
    args = SimpleNamespace(benchmark="co_bench", use_archive=False, consolidate=False)
    cfg = {"defaults": {"use_archive": True, "consolidate": True}, "co_bench": {}}
    # user set --use-archive explicitly (dest in user_set) → YAML must not touch it
    applied = apply_benchmark_config(args, {"use_archive"}, cfg)
    assert args.use_archive is False  # untouched — CLI-explicit wins
    assert args.consolidate is True
    assert applied == ["consolidate"]


def test_unknown_benchmark_applies_only_defaults():
    args = SimpleNamespace(benchmark="does_not_exist", use_archive=False)
    cfg = {"defaults": {"use_archive": True}, "co_bench": {"foster_adoption": True}}
    applied = apply_benchmark_config(args, set(), cfg)
    assert args.use_archive is True
    assert applied == ["use_archive"]  # co_bench block never consulted
    assert not hasattr(args, "foster_adoption")


def test_missing_arg_attr_is_skipped_without_crash():
    # A YAML key that is not a real arg attr must be skipped (warned), never set.
    args = SimpleNamespace(benchmark="co_bench")  # no use_archive attr
    cfg = {"defaults": {"use_archive": True}}
    warned: list[str] = []
    applied = apply_benchmark_config(args, set(), cfg, warn=warned.append)
    assert applied == []
    assert not hasattr(args, "use_archive")
    assert warned == ["use_archive"]


def test_empty_cfg_applies_nothing():
    args = SimpleNamespace(benchmark="co_bench", use_archive=False)
    assert apply_benchmark_config(args, set(), {}) == []
    assert args.use_archive is False


# ---------------------------------------------------------------------------
# Bundled YAML shape
# ---------------------------------------------------------------------------


def test_bundled_yaml_parses_with_expected_defaults():
    cfg = _bundled_cfg()
    d = cfg["defaults"]
    for key in (
        "use_archive",
        "consolidate",
        "regression_guard",
        "regression_guard_repeats",
        "within_task_recursion",
        "focus_current_headroom",
        "symmetric_trace_sampling",
    ):
        assert key in d, key
    # protect_floor is a NUMERIC gate-veto threshold (float), not a bool — it must
    # NOT be set as a default (a bool True would be inert + disable a gate short-circuit).
    assert "protect_floor" not in d
    # within_layer_refine is consumed only in the quality-gate FAIL branch, and
    # consolidate (set in this block) gives every child a focus task, which skips
    # the gate — the pair is mutually inert (Y4-S_stack-1 / Y4-P_benchmarks-1 /
    # Y4-C_callability-1).
    assert "within_layer_refine" not in d
    # eval_repeats_gate_topup tops up reused GATE traces only; consolidate skips
    # the gate and the consolidation inherit map is precomputed-frozen, so the
    # key could never fire under this file — dead even at eval_repeats>1
    # (Y4-Y_mechanism-3 / Y4-S_stack-3 / Y4-P_benchmarks-2 / Y4-C_callability-3).
    assert "eval_repeats_gate_topup" not in d
    # elite_rotation is consulted only inside select_parents' n>=2 reserved-elite
    # block; at the shipped --beam-width 1 it would be dead and self-warning, so
    # it must stay unset (header documents the beam_width: 2+ custom profile).
    assert "elite_rotation" not in d
    # the on-by-default booleans are actually True
    for key in (
        "use_archive",
        "consolidate",
        "regression_guard",
        "within_task_recursion",
        "focus_current_headroom",
        "symmetric_trace_sampling",
    ):
        assert d[key] is True, key
    assert d["regression_guard_repeats"] == 3


def test_bundled_yaml_live_families_foster_adoption():
    cfg = _bundled_cfg()
    for fam in ("alphaevolve_math", "symbolic_regression", "algotune", "arc_agi_2"):
        assert cfg[fam].get("foster_adoption") is True, fam


def test_bundled_yaml_co_bench_does_not_force_code_channel():
    cfg = _bundled_cfg()
    # co_bench (and the other demoted-helper families) keep the code channel
    # demoted — no foster_adoption, no force_code_library_live forcing.
    assert cfg["co_bench"] == {}
    for key in ("force_code_library_live", "foster_adoption"):
        assert key not in cfg["co_bench"], key
        assert key not in cfg["swe_bench_verified"], key
        assert key not in cfg["terminal_bench"], key


def test_bundled_yaml_noisy_classification_eval_repeats():
    cfg = _bundled_cfg()
    for b in ("symptom2disease", "lawbench_charge"):
        assert cfg[b]["eval_repeats"] == 3, b
        # eval_repeats_gate_topup must NOT ride along here: these blocks inherit
        # consolidate from defaults, which skips the gate — the top-up would be
        # an inert key even at eval_repeats>1 (Y4-S_stack-3 / Y4-P_benchmarks-2).
        assert "eval_repeats_gate_topup" not in cfg[b], b


def test_bundled_yaml_does_not_set_ablation_or_probe_flags():
    # ablation flags + probes are intentionally NOT set on anywhere in the file.
    cfg = _bundled_cfg()
    forbidden = {
        "no_code_library",
        "no_inspiration",
        "no_outer_context",
        "seed_code_library",
        "paired_eval",
        "repropagation",
    }
    for block in cfg.values():
        assert set(block or {}).isdisjoint(forbidden), block


# ---------------------------------------------------------------------------
# Bundled YAML applied onto a real default namespace
# ---------------------------------------------------------------------------


def test_apply_bundled_defaults_onto_parsed_args():
    # co_bench's per-bench block is {} in the bundled YAML, so this exercises
    # pure defaults-onto-a-real-parsed-namespace.
    ns = _parse(["--benchmark", "co_bench"])
    applied = apply_benchmark_config(ns, set(), _bundled_cfg())
    assert ns.use_archive is True
    assert ns.consolidate is True
    assert ns.regression_guard is True
    assert ns.within_task_recursion is True
    assert ns.regression_guard_repeats == 3
    assert "use_archive" in applied
    # a family-only key is not applied outside the live-helper families
    assert ns.foster_adoption is False


def test_tasks_path_untouched_by_bundled_yaml():
    # Y4-P_benchmarks-6 / Y4-S_stack-5: the defaults block must NOT leak onto
    # the --tasks (benchmark=None) path — bare `meta-n --tasks my.json` keeps
    # the historical linear (pre-metacognition) baseline.
    ns = _parse(["--tasks", "my.json"])
    before = dict(vars(ns))
    applied = apply_benchmark_config(ns, set(), _bundled_cfg())
    assert applied == []
    assert ns.use_archive is False
    assert ns.consolidate is False
    assert ns.regression_guard is False
    assert dict(vars(ns)) == before  # nothing mutated at all


def test_no_benchmark_skips_yaml_entirely():
    # Y4-S_stack-5: the gate keys on a FALSY benchmark (None / ""), never on
    # cfg membership — a hand-built namespace without the attr also skips,
    # while an unknown benchmark STRING still receives the defaults block
    # (test_unknown_benchmark_applies_only_defaults pins that side).
    ns = SimpleNamespace(use_archive=False)  # no benchmark attr at all
    assert apply_benchmark_config(ns, set(), _bundled_cfg()) == []
    assert ns.use_archive is False
    ns2 = SimpleNamespace(benchmark="", use_archive=False)
    assert apply_benchmark_config(ns2, set(), _bundled_cfg()) == []
    ns3 = SimpleNamespace(benchmark="does_not_exist", use_archive=False)
    assert apply_benchmark_config(ns3, set(), {"defaults": {"use_archive": True}}) == [
        "use_archive"
    ]
    assert ns3.use_archive is True


def test_apply_bundled_live_family():
    ns = _parse(["--benchmark", "alphaevolve_math"])
    apply_benchmark_config(ns, set(), _bundled_cfg())
    assert ns.use_archive is True
    assert ns.foster_adoption is True


def test_apply_bundled_co_bench_keeps_code_channel_demoted():
    ns = _parse(["--benchmark", "co_bench"])
    apply_benchmark_config(ns, set(), _bundled_cfg())
    assert ns.use_archive is True
    assert ns.foster_adoption is False
    assert ns.force_code_library_live is False


def test_apply_bundled_classification_eval_repeats():
    ns = _parse(["--benchmark", "symptom2disease"])
    apply_benchmark_config(ns, set(), _bundled_cfg())
    assert ns.eval_repeats == 3
    assert ns.use_archive is True


# ---------------------------------------------------------------------------
# --benchmark-config resolution / arg default
# ---------------------------------------------------------------------------


def test_benchmark_config_arg_defaults_auto():
    assert _parse([]).benchmark_config == "auto"
    assert _parse(["--benchmark-config", "none"]).benchmark_config == "none"
    assert _parse(["--benchmark-config", "/x/y.yaml"]).benchmark_config == "/x/y.yaml"


def test_benchmark_config_none_resolves_to_skip():
    assert _benchmark_config_path("none") is None
    assert _benchmark_config_path("NONE") is None
    assert _benchmark_config_path(None) is None


def test_benchmark_config_auto_resolves_bundled_path():
    p = _benchmark_config_path("auto")
    assert p is not None
    assert p.name == "benchmark_features.yaml"
    assert p == YAML_PATH
    assert p.exists()


def test_benchmark_config_custom_path():
    assert _benchmark_config_path("/tmp/custom.yaml") == Path("/tmp/custom.yaml")


def test_none_choice_applies_no_overrides():
    # --benchmark-config none must leave args untouched (bare code defaults).
    ns = _parse([])
    before = dict(vars(ns))
    # async_main resolves 'none' → None and skips apply_benchmark_config entirely.
    assert _benchmark_config_path("none") is None
    # nothing mutated
    assert dict(vars(ns)) == before


def test_user_set_dests_detects_bare_and_joined_forms():
    parser = main_mod.build_parser()
    assert "eval_repeats" in _user_set_dests(parser, ["--eval-repeats", "3"])
    assert "eval_repeats" in _user_set_dests(parser, ["--eval-repeats=3"])
    assert "use_archive" in _user_set_dests(parser, ["--use-archive"])
    assert "use_archive" not in _user_set_dests(parser, ["--consolidate"])


# ---------------------------------------------------------------------------
# build_base_run_config provenance keys
# ---------------------------------------------------------------------------


def test_provenance_keys_recorded_in_base_run_config():
    ns = _parse([])
    ns.benchmark_config_resolved = str(YAML_PATH)
    ns.benchmark_config_applied = ["consolidate", "use_archive"]
    cfg = build_base_run_config(
        ns, benchmark_name="co_bench", solver_language="python",
        executor_name="X",
    )
    assert cfg["benchmark_config"] == str(YAML_PATH)
    assert cfg["benchmark_config_applied"] == ["consolidate", "use_archive"]


def test_provenance_keys_default_when_missing():
    # A hand-built namespace without the stash still yields the additive keys
    # (getattr defaults: 'none' / []), never an AttributeError.
    ns = _parse([])  # no benchmark_config_resolved / _applied attrs
    cfg = build_base_run_config(
        ns, benchmark_name="co_bench", solver_language="python",
        executor_name="X",
    )
    assert cfg["benchmark_config"] == "none"
    assert cfg["benchmark_config_applied"] == []


# ---------------------------------------------------------------------------
# Companion-requirement invariants baked into the defaults block
# ---------------------------------------------------------------------------


def test_defaults_block_companion_requirements():
    cfg = _bundled_cfg()
    d = cfg["defaults"]
    # within_task_recursion is a modifier on consolidate.
    if d.get("within_task_recursion"):
        assert d.get("consolidate") is True
    # focus_current_headroom requires --regression-guard (and --consolidate).
    if d.get("focus_current_headroom"):
        assert d.get("regression_guard") is True
        assert d.get("consolidate") is True
    # regression_guard requires the base-floor resamples > 0.
    if d.get("regression_guard"):
        assert int(d.get("regression_guard_repeats", 0)) > 0
    # elite_rotation is a modifier on beam_width >= 2 (R1-D_selection-2). Any
    # block that arms it must also raise beam_width in the SAME block, or the
    # key is dead-and-self-warning at the code-default B=1 (Y4 conflict
    # resolution: the shipped file leaves BOTH unset — see the header note).
    for name, block in cfg.items():
        block = block or {}
        if block.get("elite_rotation"):
            assert int(block.get("beam_width", 1) or 1) >= 2, name
    # eval_repeats stays a per-benchmark (classification) dial, never a default.
    assert "eval_repeats" not in d
    # Gate-fed keys cannot coexist with consolidate: consolidate gives every
    # child a focus task, which skips the quality gate — the gate-FAIL refine
    # hook and the gate-trace top-up both lose their only entry point
    # (Y4-P_benchmarks-1 / Y4-P_benchmarks-2).
    assert not (d.get("within_layer_refine") and d.get("consolidate"))
    assert not (d.get("eval_repeats_gate_topup") and d.get("consolidate"))
    # Any block that sets eval_repeats_gate_topup must also make it live:
    # eval_repeats > 1 in the SAME block, and consolidate effectively off
    # (the top-up denoises reused GATE traces only) — Y4-Y_mechanism-3.
    d_consolidate = bool(d.get("consolidate", False))
    for name, block in cfg.items():
        block = block or {}
        if block.get("eval_repeats_gate_topup"):
            assert int(block.get("eval_repeats", 1) or 1) > 1, name
            assert not block.get("consolidate", d_consolidate), name


# ---------------------------------------------------------------------------
# Shipped-config self-consistency (no inert keys, no self-warnings)
# ---------------------------------------------------------------------------


def test_bundled_yaml_yields_no_noop_notes_on_any_benchmark():
    # The shipped default config must never trigger its own F229 no-op warning
    # banner: every key it applies has to be live on the path it configures
    # (Y4-Y_mechanism-3 / Y4-P_benchmarks-2 — the whole-class regression test).
    # NO exceptions: a key that is inert under the stack this same file ships
    # (e.g. elite_rotation at the code-default --beam-width 1) is documented in
    # the header and left unset, never shipped armed-but-warning.
    cfg = _bundled_cfg()
    for bench in [k for k in cfg if k != "defaults"]:
        ns = _parse(["--benchmark", bench])
        apply_benchmark_config(ns, set(), cfg)
        for spine in (False, True):
            notes = main_mod.evolutionary_noop_flag_notes(ns, spine)
            assert notes == [], (bench, spine)


def test_bundled_yaml_gate_knobs_only_in_non_consolidate_blocks():
    # Y4-P_benchmarks-7: gate knobs act only inside _gate_check, and consolidate
    # mode skips the gate (G9, eo.py:1202) — a gate knob in any block whose
    # effective consolidate is on would be silently inert.
    cfg = _bundled_cfg()
    gate_knobs = {"protect_floor", "gate_margin", "gate_repeats", "gate_tasks"}
    d_consolidate = bool((cfg.get("defaults") or {}).get("consolidate", False))
    for name, block in cfg.items():
        block = block or {}
        knobs = gate_knobs & set(block)
        if knobs and block.get("consolidate", d_consolidate):
            raise AssertionError(
                f"{name} sets {sorted(knobs)} but consolidate is effectively on"
                f" — inert key(s)"
            )


def test_bundled_yaml_docker_families_floor_budget_dial():
    # Y4-C_callability-12: the Docker families keep the regression guard (seed
    # floor + archive clamp) but opt out of the 2 extra full-bench gen0
    # resample passes (3x500 / 3x113 container evals is not a default budget).
    cfg = _bundled_cfg()
    for fam in ("terminal_bench", "swe_bench_verified"):
        assert cfg[fam].get("regression_guard_repeats") == 1, fam
    # round-trip: per-bench dial beats defaults, guard itself stays ON
    ns = _parse(["--benchmark", "terminal_bench"])
    apply_benchmark_config(ns, set(), cfg)
    assert ns.regression_guard is True and ns.regression_guard_repeats == 1


# ---------------------------------------------------------------------------
# Y4-S_stack-7 — parser-armed type validation of custom-YAML values
# ---------------------------------------------------------------------------


def test_custom_yaml_garbage_int_fails_fast():
    ns = _parse(["--benchmark", "co_bench"])
    cfg = {"defaults": {"eval_repeats": "three"}}
    with pytest.raises(ValueError, match="eval_repeats"):
        apply_benchmark_config(ns, set(), cfg, parser=main_mod.build_parser())


def test_custom_yaml_numeric_string_coerced():
    # Mirror argparse CLI semantics: '3' -> 3 (int), not the str '3'.
    ns = _parse(["--benchmark", "co_bench"])
    cfg = {"defaults": {"eval_repeats": "3"}}
    apply_benchmark_config(ns, set(), cfg, parser=main_mod.build_parser())
    assert ns.eval_repeats == 3
    assert type(ns.eval_repeats) is int


def test_custom_yaml_string_bool_fails_fast():
    # Pins the semantic-inversion fix: the YAML string 'false' on a boolean
    # dest must raise, never be setattr'd (it reads as truthy downstream).
    ns = _parse(["--benchmark", "co_bench"])
    cfg = {"defaults": {"regression_guard": "false"}}
    with pytest.raises(ValueError, match="regression_guard"):
        apply_benchmark_config(ns, set(), cfg, parser=main_mod.build_parser())


def test_custom_yaml_bool_for_int_fails_fast():
    # bool is an int subclass — `regression_guard_repeats: true` must raise,
    # not silently become 1.
    ns = _parse(["--benchmark", "co_bench"])
    cfg = {"defaults": {"regression_guard_repeats": True}}
    with pytest.raises(ValueError, match="regression_guard_repeats"):
        apply_benchmark_config(ns, set(), cfg, parser=main_mod.build_parser())


def test_custom_yaml_gate_margin_none_string_coerced():
    # The custom _gate_margin converter: YAML 'none' behaves like the CLI form
    # (--gate-margin none → None) instead of landing as the string 'none'.
    ns = _parse(["--benchmark", "co_bench"])
    cfg = {"defaults": {"gate_margin": "none"}}
    applied = apply_benchmark_config(ns, set(), cfg, parser=main_mod.build_parser())
    assert ns.gate_margin is None
    assert applied == ["gate_margin"]


def test_custom_yaml_gate_margin_bool_fails_fast():
    # PyYAML parses on/off/yes/no as booleans, so `gate_margin: off` arrives
    # as False — which is not-None downstream (thresholding stays ENABLED at
    # margin 0), the inverse of the CLI's `--gate-margin off` → None. Must
    # refuse at startup, never setattr.
    ns = _parse(["--benchmark", "co_bench"])
    cfg = {"defaults": {"gate_margin": False}}
    with pytest.raises(ValueError, match="gate_margin"):
        apply_benchmark_config(ns, set(), cfg, parser=main_mod.build_parser())


def test_custom_yaml_gate_margin_float_applies():
    ns = _parse(["--benchmark", "co_bench"])
    cfg = {"defaults": {"gate_margin": 0.05}}
    applied = apply_benchmark_config(ns, set(), cfg, parser=main_mod.build_parser())
    assert applied == ["gate_margin"]
    assert ns.gate_margin == 0.05


def test_custom_yaml_elite_rotation_beam_width_pairing_applies():
    # The shipped header directs custom profiles to pair elite_rotation with
    # beam_width: 2+ in the SAME block — so beam_width must be YAML-settable,
    # land as an int, and leave the armed elite_rotation note-free.
    ns = _parse(["--benchmark", "co_bench"])
    cfg = {"defaults": {"use_archive": True, "elite_rotation": True,
                        "beam_width": 2}}
    warned: list[str] = []
    applied = apply_benchmark_config(
        ns, set(), cfg, warn=warned.append, parser=main_mod.build_parser()
    )
    assert applied == ["beam_width", "elite_rotation", "use_archive"]
    assert warned == []
    assert ns.beam_width == 2 and ns.elite_rotation is True
    assert main_mod.evolutionary_noop_flag_notes(ns, external_spine=False) == []


def test_custom_yaml_beam_width_bool_fails_fast():
    # int-dest validation covers the new key: `beam_width: true` must raise,
    # not silently become 1.
    ns = _parse(["--benchmark", "co_bench"])
    cfg = {"defaults": {"beam_width": True}}
    with pytest.raises(ValueError, match="beam_width"):
        apply_benchmark_config(ns, set(), cfg, parser=main_mod.build_parser())


def test_yaml_cannot_set_protected_dests():
    # Y4-Y_mechanism-4: with parser validation armed (the production call
    # site), a key outside _BENCHMARK_CONFIG_SETTABLE_DESTS — even a real args
    # attr like model/api_key — is warn+skipped, never setattr'd. YAML must
    # never steer run identity or credentials.
    ns = _parse(["--benchmark", "co_bench"])
    model_before = ns.model
    warned: list[str] = []
    cfg = {"defaults": {"model": "evil/model", "api_key": "stolen",
                        "consolidate": True}}
    applied = apply_benchmark_config(
        ns, set(), cfg, warn=warned.append, parser=main_mod.build_parser()
    )
    assert applied == ["consolidate"]
    assert ns.model == model_before
    assert ns.api_key is None
    assert any("model" in w and "not settable" in w for w in warned)
    assert any("api_key" in w and "not settable" in w for w in warned)


def test_parserless_call_keeps_legacy_lenient_behavior():
    # parser=None (hand-built-namespace / unit callers) stays byte-identical:
    # no allowlist, no type validation — raw setattr as before.
    ns = SimpleNamespace(benchmark="co_bench", model="m0", use_archive=False)
    applied = apply_benchmark_config(
        ns, set(), {"defaults": {"model": "m1", "use_archive": True}}
    )
    assert applied == ["model", "use_archive"]
    assert ns.model == "m1"


def test_bundled_yaml_validates_clean_with_parser():
    # Byte-identity pin: applying the bundled YAML WITH the parser must not
    # raise, and must produce the same applied list + attr values as the
    # lenient parserless application (the fast path passes values as-is) —
    # for EVERY benchmark choice (Y4-C_callability-8).
    benches = ["co_bench", "symptom2disease", "lawbench_charge",
               "terminal_bench", "swe_bench_verified", "alphaevolve_math",
               "symbolic_regression", "algotune", "arc_agi_2"]
    for argv in [["--benchmark", b] for b in benches]:
        plain, validated = _parse(argv), _parse(argv)
        applied_plain = apply_benchmark_config(plain, set(), _bundled_cfg())
        applied_validated = apply_benchmark_config(
            validated, set(), _bundled_cfg(), parser=main_mod.build_parser()
        )
        assert applied_plain == applied_validated, argv
        assert applied_plain, argv  # the bundled defaults actually land
        assert vars(plain) == vars(validated), argv


# ---------------------------------------------------------------------------
# Y4-P_benchmarks-8 — an explicitly armed-but-inert elite_rotation still warns
# ---------------------------------------------------------------------------


def test_cli_elite_rotation_at_default_beam_width_still_noted():
    # The shipped YAML no longer arms elite_rotation (dead at the code-default
    # --beam-width 1), but a user who passes it explicitly at B=1 must still
    # get the F229 note (the note machinery stays; only the self-warning
    # shipped default was removed).
    argv = ["--benchmark", "co_bench", "--model", "m", "--elite-rotation"]
    ns = _parse(argv)
    apply_benchmark_config(
        ns, _user_set_dests(main_mod.build_parser(), argv), _bundled_cfg()
    )
    notes = main_mod.evolutionary_noop_flag_notes(ns, external_spine=False)
    assert any("--elite-rotation" in n for n in notes)


# ---------------------------------------------------------------------------
# Y4-C_callability-9 — per-benchmark scope: --tasks runs keep bare defaults
# ---------------------------------------------------------------------------


def test_tasks_run_skips_bundled_config(tmp_path, monkeypatch):
    # --tasks (benchmark=None) + default 'auto' config must keep bare code
    # defaults: no YAML overrides, provenance 'none' (guard in async_main).
    monkeypatch.delenv("META_N_DAILY_BUDGET_USD", raising=False)
    tasks = tmp_path / "tasks.json"
    tasks.write_text('[{"task_id": "t1", "description": "d"}]')
    argv = ["--tasks", str(tasks), "--api-key", "k", "--resume"]
    monkeypatch.setattr(sys, "argv", ["meta-n"] + argv)
    args = main_mod.build_parser().parse_args(argv)
    # --resume without --exp-name returns 1 AFTER the config block
    # (main.py resume guard), so no LLM/network is reached.
    assert asyncio.run(main_mod.async_main(args)) == 1
    assert args.use_archive is False and args.consolidate is False
    assert args.benchmark_config_applied == []
    assert args.benchmark_config_resolved == "none"


# ---------------------------------------------------------------------------
# Y4-C_callability-10 — every YAML-settable dest's --help names the config
# ---------------------------------------------------------------------------


def test_yaml_settable_dests_help_mentions_bundled_config():
    # Drift guard: any future YAML key added without a help annotation (or any
    # annotation deleted) fails here, so --help can't go stale again.
    cfg = _bundled_cfg()
    keys: set[str] = set()
    for block in cfg.values():
        keys |= set(block or {})
    actions = {a.dest: a for a in main_mod.build_parser()._actions}
    assert keys, "bundled YAML sets no keys?"
    for key in sorted(keys):
        assert key in actions, f"YAML key {key!r} has no matching parser dest"
        assert "bundled benchmark config" in (actions[key].help or ""), key

# ---------------------------------------------------------------------------
# Y4-Y_mechanism-1 — 'auto' resolves package-relatively; missing file is a
# startup error
# ---------------------------------------------------------------------------


def test_auto_path_is_package_relative():
    # Exact guard against a parents[1]/site-packages regression: 'auto' must
    # resolve INSIDE the meta_n package (shipped as package data), never
    # relative to the install prefix.
    assert _benchmark_config_path("auto") == (
        Path(main_mod.__file__).resolve().parent
        / "configs"
        / "benchmark_features.yaml"
    )


def test_missing_benchmark_config_is_startup_error(tmp_path, capsys):
    # F032 convention: a missing config file (explicit typo OR broken 'auto'
    # install) refuses at startup with rc=1 — never a silent fork onto bare
    # code defaults.
    args = _parse(["--benchmark", "co_bench"])
    args.benchmark_config = str(tmp_path / "nope.yaml")
    with patch.object(sys, "argv", ["meta-n"]):
        rc = asyncio.run(main_mod.async_main(args))
    assert rc == 1
    out = capsys.readouterr().out
    assert "file not found" in out
    assert "nope.yaml" in out
    assert "--benchmark-config none" in out


# ---------------------------------------------------------------------------
# Y4-Y_mechanism-6 — structural shape guards (warn-and-degrade, never raise)
# ---------------------------------------------------------------------------


def test_structural_malformation_never_raises():
    # (a) top-level list: warn + apply nothing.
    ns = SimpleNamespace(benchmark="co_bench", use_archive=False)
    warned: list[str] = []
    assert apply_benchmark_config(ns, set(), ["a"], warn=warned.append) == []
    assert warned and "top-level" in warned[0]
    # (b) non-mapping defaults block: warn + treat as empty.
    ns = SimpleNamespace(benchmark="co_bench", use_archive=False)
    warned = []
    assert apply_benchmark_config(
        ns, set(), {"defaults": True}, warn=warned.append
    ) == []
    assert warned and "defaults" in warned[0]
    # (c) valid defaults survive a malformed per-bench block (mirrors the
    # unknown-benchmark semantics), and the block is warned about.
    ns = SimpleNamespace(benchmark="co_bench", use_archive=False)
    warned = []
    applied = apply_benchmark_config(
        ns,
        set(),
        {"defaults": {"use_archive": True}, "co_bench": ["x"]},
        warn=warned.append,
    )
    assert applied == ["use_archive"] and ns.use_archive is True
    assert any("co_bench" in w for w in warned)
    # (d) legitimate empty/None blocks stay warning-free (normalized by `or {}`).
    for cfg in ({"defaults": None}, {"co_bench": {}}, {"defaults": None, "co_bench": None}):
        ns = SimpleNamespace(benchmark="co_bench", use_archive=False)
        warned = []
        assert apply_benchmark_config(ns, set(), cfg, warn=warned.append) == []
        assert warned == [], cfg


# ---------------------------------------------------------------------------
# Y4-Y_mechanism-8 — provenance sentinels for found-but-degenerate configs
# ---------------------------------------------------------------------------


def _run_async_main_until_config(args):
    """Drive the REAL async_main through the benchmark-config block, aborting
    right after it (at _resolve_backend_and_model) via a sentinel exception —
    no LLM / adapter / network is ever reached."""

    class _Sentinel(Exception):
        pass

    with patch.object(sys, "argv", ["meta-n"]), patch.object(
        main_mod, "_resolve_backend_and_model", side_effect=_Sentinel
    ):
        with pytest.raises(_Sentinel):
            asyncio.run(main_mod.async_main(args))


def test_empty_config_file_records_path_and_zero_overrides(tmp_path, capsys):
    p = tmp_path / "empty.yaml"
    p.write_text("")
    args = _parse(["--benchmark", "co_bench", "--benchmark-config", str(p)])
    _run_async_main_until_config(args)
    assert args.benchmark_config_resolved == str(p)  # found+parsed, 0 overrides
    assert args.benchmark_config_applied == []
    assert "applied 0 overrides" in capsys.readouterr().out


def test_comments_only_config_file_records_path(tmp_path):
    p = tmp_path / "comments.yaml"
    p.write_text("---\n# nothing here\n")
    args = _parse(["--benchmark", "co_bench", "--benchmark-config", str(p)])
    _run_async_main_until_config(args)
    assert args.benchmark_config_resolved == str(p)
    assert args.benchmark_config_applied == []


def test_malformed_config_file_records_parse_error_sentinel(tmp_path, capsys):
    p = tmp_path / "broken.yaml"
    p.write_text("defaults: [unclosed\n  bad: {")
    args = _parse(["--benchmark", "co_bench", "--benchmark-config", str(p)])
    _run_async_main_until_config(args)
    assert args.benchmark_config_resolved == f"parse-error:{p}"
    assert args.benchmark_config_applied == []
    assert "failed to parse" in capsys.readouterr().out


def test_list_config_file_records_parse_error_and_never_crashes(tmp_path, capsys):
    # Crash regression: a top-level-list YAML previously raised AttributeError
    # at startup; now it warns, records the sentinel, and proceeds on bare
    # code defaults.
    p = tmp_path / "list.yaml"
    p.write_text("- a\n- b\n")
    args = _parse(["--benchmark", "co_bench", "--benchmark-config", str(p)])
    _run_async_main_until_config(args)
    assert args.benchmark_config_resolved == f"parse-error:{p}"
    assert args.benchmark_config_applied == []
    assert "did not parse to a mapping" in capsys.readouterr().out
    assert args.use_archive is False  # bare code defaults


def test_none_choice_records_none_sentinel():
    # Byte-identity guard: --benchmark-config none short-circuits before any
    # new code — resolved 'none', applied [], args untouched.
    args = _parse(["--benchmark", "co_bench", "--benchmark-config", "none"])
    _run_async_main_until_config(args)
    assert args.benchmark_config_resolved == "none"
    assert args.benchmark_config_applied == []
    assert args.use_archive is False


# ---------------------------------------------------------------------------
# Y4-S_stack-6 / Y4-C_callability-11 — --no-<flag> opt-out forms
# ---------------------------------------------------------------------------

#: The ten YAML-settable default-OFF booleans migrated to BooleanOptionalAction.
_BOOLOPT_DESTS = [
    "use_archive", "consolidate", "regression_guard", "within_task_recursion",
    "within_layer_refine", "elite_rotation", "focus_current_headroom",
    "symmetric_trace_sampling", "eval_repeats_gate_topup", "foster_adoption",
]


def test_no_flag_opts_out_of_yaml_default_end_to_end():
    # The stack-minus-one scenario: keep the whole default stack but opt out of
    # exactly one key via its --no- form (CLI-explicit beats YAML).
    argv = ["--benchmark", "co_bench", "--use-archive", "--no-consolidate"]
    args = main_mod.build_parser().parse_args(argv)
    user = _user_set_dests(main_mod.build_parser(), argv)
    applied = apply_benchmark_config(args, user, _bundled_cfg())
    assert args.consolidate is False
    assert "consolidate" in user and "consolidate" not in applied
    assert args.use_archive is True
    assert args.regression_guard is True  # rest of the stack intact


def test_every_yaml_boolean_flag_has_negative_form():
    # Guards future YAML keys against re-introducing the opt-out gap: every
    # bool-valued key across all blocks must have a --no- option string.
    cfg = _bundled_cfg()
    bool_keys: set[str] = set()
    for block in cfg.values():
        for k, v in (block or {}).items():
            if isinstance(v, bool):
                bool_keys.add(k)
    assert bool_keys
    actions = {a.dest: a for a in main_mod.build_parser()._actions}
    for key in sorted(bool_keys):
        opts = actions[key].option_strings
        assert any(o.startswith("--no-") for o in opts), (key, opts)


def test_boolopt_flags_default_false_when_absent():
    # Bare-path byte-identity: every migrated dest is exactly False (never
    # None — BooleanOptionalAction's implicit default) on a flagless parse.
    ns = _parse(["--benchmark", "co_bench"])
    for dest in _BOOLOPT_DESTS:
        assert getattr(ns, dest) is False, dest


def test_no_forms_are_precedence_protected_and_note_free():
    # --no-X marks the dest user-set (never YAML-overwritten), so the bundled
    # default is not re-applied on top of the explicit opt-out.
    argv = ["--benchmark", "co_bench", "--no-regression-guard"]
    user = _user_set_dests(main_mod.build_parser(), argv)
    assert "regression_guard" in user
    ns = main_mod.build_parser().parse_args(argv)
    applied = apply_benchmark_config(ns, user, _bundled_cfg())
    assert ns.regression_guard is False
    assert "regression_guard" not in applied


# ---------------------------------------------------------------------------
# Y4-Y_mechanism-7 — annotate_config_sourced (YAML-sourced flag attribution)
# ---------------------------------------------------------------------------


def _stamp_provenance(ns, applied, resolved=None):
    ns.benchmark_config_applied = applied
    ns.benchmark_config_resolved = str(YAML_PATH) if resolved is None else resolved
    return ns


def test_annotate_config_sourced_notes_carry_source():
    # A CUSTOM config that arms elite_rotation at the default beam width fires
    # the F229 note, and the note must be attributed to the config file. (The
    # bundled config yields zero notes by invariant — see
    # test_bundled_yaml_yields_no_noop_notes_on_any_benchmark.)
    ns = _parse(["--benchmark", "co_bench"])
    applied = apply_benchmark_config(
        ns, set(), {"defaults": {"use_archive": True, "elite_rotation": True}}
    )
    _stamp_provenance(ns, applied, resolved="/x/custom.yaml")
    notes = main_mod.evolutionary_noop_flag_notes(ns, external_spine=False)
    annotated = main_mod.annotate_config_sourced(notes, ns)
    assert len(annotated) == 1
    assert annotated[0].startswith("--elite-rotation")
    assert annotated[0].endswith("(set by benchmark-config: /x/custom.yaml)")


def test_annotate_config_note_names_config_source():
    # A custom YAML that arms within_task_recursion WITHOUT consolidate leaves
    # every consumer dormant; the F229 note must point at the file.
    ns = _parse(["--benchmark", "co_bench"])
    applied = apply_benchmark_config(
        ns, set(), {"defaults": {"use_archive": True, "within_task_recursion": True}}
    )
    _stamp_provenance(ns, applied, resolved="/x/custom.yaml")
    notes = main_mod.evolutionary_noop_flag_notes(ns, external_spine=False)
    out = main_mod.annotate_config_sourced(notes, ns)
    assert len(out) == 1
    assert out[0].startswith("--within-task-recursion")
    assert out[0].endswith("(set by benchmark-config: /x/custom.yaml)")


def test_annotate_byte_identity_without_applied_entries():
    # With applied=[] (and with the provenance attrs absent entirely) the
    # helper returns its input unchanged — bare-path console bytes identical.
    texts = ["--consolidate", "--eval-repeats", "--gate-margin"]
    ns = _parse(["--benchmark", "co_bench"])
    _stamp_provenance(ns, [])
    assert main_mod.annotate_config_sourced(texts, ns) == texts
    bare = SimpleNamespace()  # no provenance attrs at all
    assert main_mod.annotate_config_sourced(texts, bare) == texts


def test_annotate_joined_gate_note_carries_source():
    # Two YAML-set gate knobs under consolidate collapse into ONE joined note
    # ('--gate-tasks, --gate-repeats (...)') whose head token carries a
    # trailing comma; attribution must still land.
    ns = _parse(["--benchmark", "co_bench"])
    cfg = {"defaults": {"use_archive": True, "consolidate": True,
                        "gate_tasks": 5, "gate_repeats": 2}}
    applied = apply_benchmark_config(
        ns, set(), cfg, parser=main_mod.build_parser()
    )
    assert {"gate_tasks", "gate_repeats"} <= set(applied)
    _stamp_provenance(ns, applied, resolved="/x/custom.yaml")
    notes = main_mod.evolutionary_noop_flag_notes(ns, external_spine=False)
    annotated = main_mod.annotate_config_sourced(notes, ns)
    joined = [n for n in annotated if "--gate-tasks, --gate-repeats" in n]
    assert len(joined) == 1
    assert joined[0].endswith("(set by benchmark-config: /x/custom.yaml)")


def test_annotate_no_false_attribution_for_cli_explicit_flag():
    # An explicitly-passed CLI flag is excluded from benchmark_config_applied
    # by apply_benchmark_config (user_set wins), so its note text carries no
    # suffix even when the YAML also lists the key.
    argv = ["--benchmark", "co_bench", "--elite-rotation"]
    ns = main_mod.build_parser().parse_args(argv)
    user = _user_set_dests(main_mod.build_parser(), argv)
    applied = apply_benchmark_config(
        ns, user, {"defaults": {"use_archive": True, "elite_rotation": True}}
    )
    assert "elite_rotation" not in applied
    _stamp_provenance(ns, applied, resolved="/x/custom.yaml")
    notes = main_mod.evolutionary_noop_flag_notes(ns, external_spine=False)
    out = main_mod.annotate_config_sourced(notes, ns)
    assert len(out) == 1
    assert out[0].startswith("--elite-rotation")
    assert "set by benchmark-config" not in out[0]


# ---------------------------------------------------------------------------
# Y4-S_stack-4 / Y4-C_callability-6 — consolidate moots repropagation + gate
# knobs (end-to-end round trips through the bundled YAML)
# ---------------------------------------------------------------------------


def test_bundled_stack_surfaces_repropagation_mooted_by_consolidate():
    # Explicit --repropagation under the bundled default stack (consolidate ON
    # via YAML) must surface the never-fires note instead of failing silently.
    argv = ["--benchmark", "co_bench", "--use-archive", "--repropagation"]
    ns = main_mod.build_parser().parse_args(argv)
    apply_benchmark_config(ns, _user_set_dests(main_mod.build_parser(), argv),
                           _bundled_cfg())
    assert ns.consolidate is True and ns.repropagation is True
    notes = main_mod.evolutionary_noop_flag_notes(ns, external_spine=False)
    assert any("--repropagation" in n for n in notes)


def test_bundled_stack_surfaces_gate_knobs_mooted_by_consolidate():
    # Explicit gate knobs under the bundled default stack (consolidate ON via
    # YAML skips the gate entirely) must surface the gate-skipped note.
    argv = ["--benchmark", "co_bench", "--gate-tasks", "5",
            "--protect-floor", "0.05"]
    ns = main_mod.build_parser().parse_args(argv)
    apply_benchmark_config(ns, _user_set_dests(main_mod.build_parser(), argv),
                           _bundled_cfg())
    assert ns.consolidate is True
    notes = main_mod.evolutionary_noop_flag_notes(ns, external_spine=False)
    gate_notes = [n for n in notes if "gate is skipped entirely" in n]
    assert len(gate_notes) == 1
    assert "--gate-tasks" in gate_notes[0] and "--protect-floor" in gate_notes[0]


def test_cli_within_task_recursion_without_consolidate_is_noted():
    # `--use-archive --benchmark-config none --within-task-recursion`: with
    # consolidate off, every consumer of the modifier (DEEPEN directive, focus
    # inheritance, archive depth term) is dormant — the F229 note must fire.
    ns = _parse(["--benchmark", "co_bench", "--use-archive",
                 "--benchmark-config", "none", "--within-task-recursion"])
    notes = main_mod.evolutionary_noop_flag_notes(ns, external_spine=False)
    wtr = [n for n in notes if n.startswith("--within-task-recursion")]
    assert len(wtr) == 1 and "--consolidate" in wtr[0]


def test_bundled_stack_no_consolidate_surfaces_within_task_recursion():
    # --no-consolidate under the bundled stack leaves the YAML-armed
    # within_task_recursion dead; the note must appear and be attributed to
    # the config file (the value came from the YAML, not the CLI).
    argv = ["--benchmark", "co_bench", "--use-archive", "--no-consolidate"]
    ns = main_mod.build_parser().parse_args(argv)
    applied = apply_benchmark_config(
        ns, _user_set_dests(main_mod.build_parser(), argv), _bundled_cfg()
    )
    assert ns.within_task_recursion is True and ns.consolidate is False
    _stamp_provenance(ns, applied)
    notes = main_mod.evolutionary_noop_flag_notes(ns, external_spine=False)
    annotated = main_mod.annotate_config_sourced(notes, ns)
    wtr = [n for n in annotated if n.startswith("--within-task-recursion")]
    assert len(wtr) == 1
    assert wtr[0].endswith(f"(set by benchmark-config: {YAML_PATH})")


def test_mistyped_yaml_value_is_clean_startup_refusal(tmp_path, capsys):
    # Y4-Y_mechanism-4: at the production call site a wrong-typed YAML value is
    # an F032 startup refusal (rc=1, red message naming the key) — a clean
    # fail-at-startup instead of the historical mid-run TypeError.
    p = tmp_path / "bad.yaml"
    p.write_text('defaults:\n  use_archive: "false"\n')
    args = _parse(["--benchmark", "co_bench", "--benchmark-config", str(p)])
    with patch.object(sys, "argv", ["meta-n"]):
        rc = asyncio.run(main_mod.async_main(args))
    assert rc == 1
    out = capsys.readouterr().out
    assert "use_archive" in out and "boolean" in out
    assert "--benchmark-config none" in out
    # the wrong-typed value was never setattr'd
    assert args.use_archive is False
