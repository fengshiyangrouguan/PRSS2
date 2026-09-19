"""Refine §6b — main.py plumbing tests (flag wiring for the module-side wave).

Covered here:
  * F006 — --elite-rotation: CLI default OFF, provenance key appended (F031
    pattern), and the generation offset actually reaches
    Archive.select_parents.
  * F034 / F035 — --focus-current-headroom / --eval-repeats-gate-topup CLI
    wiring (defaults OFF; appended provenance keys).
  * F063 / F060 — --agentic-spend-budget / --agentic-temperature CLI wiring +
    the evolutionary_noop_flag_notes entries.
  * F075 — --classify-balanced-json-fallback: parse, evolutionary-orchestrator
    forwarding (the archive half is in
    tests/test_r6b_evolutionary_orchestrator.py), base-config provenance.
  * F156 — symmetric_trace_sampling base-config provenance (parse tests live in
    tests/test_main_args.py, mirroring omega_context_budget).
  * F068 — --empty-retry-max-tokens help names the min(cap, 4x) re-issue rule.

No LLM / Docker / network is touched.
"""

from __future__ import annotations

import argparse
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import meta_n.main as main_mod
from meta_n.core.evolutionary_orchestrator import (
    EvolutionaryConfig,
    EvolutionaryOrchestrator,
)
from meta_n.core.meta_layer import InjectedCode, TaskDescription, Trace
from meta_n.main import (
    build_base_run_config,
    evolutionary_noop_flag_notes,
    evolutionary_run_kwargs,
)

# ---------------------------------------------------------------------------
# Helpers (shared patterns with test_refine_main.py / test_main_args.py)
# ---------------------------------------------------------------------------


def _parse(argv):
    with patch.object(sys, "argv", ["meta-n", *argv]):
        return main_mod.parse_args()


def _capture_parser() -> argparse.ArgumentParser:
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


# ---------------------------------------------------------------------------
# F006 — --elite-rotation
# ---------------------------------------------------------------------------


async def test_config_flag_plumbs_generation_offset(tmp_path):
    """The 1-based generation index reaches Archive.select_parents when the
    flag is ON; the OFF path passes the byte-identical 0 offset."""

    async def _offsets(elite_rotation: bool) -> list[int]:
        config = EvolutionaryConfig(
            output_dir=str(tmp_path / ("on" if elite_rotation else "off")),
            max_depth=3, max_iterations=2, no_early_stop=True,
            gate_tasks=0, beam_width=2, beam_candidates=1, parallel=1,
            seed=42, elite_rotation=elite_rotation,
        )
        orch = EvolutionaryOrchestrator(
            llm_client=MagicMock(), executor=MagicMock(), omega=MagicMock(),
            config=config, solver_language="bash",
        )
        # Stubbed run as in tests/golden/capture_stage23_golden.py.
        orch.solver.solve = AsyncMock(return_value=("echo golden", "stub", 10))
        orch.executor.execute = AsyncMock(
            side_effect=lambda script, task: Trace(
                task_id=task.task_id, depth=1, script="echo golden",
                success=True, score=0.5,
            )
        )
        orch.omega.generate = AsyncMock(return_value=(
            InjectedCode(
                pre_process="additional_context = 'hint'",
                rationale="stub", source_depth=2,
            ),
            7,
        ))

        recorded: list[int] = []
        orig = orch.archive.select_parents

        def spy(n, rng=None, pool=None, elite_rotation=0):
            recorded.append(elite_rotation)
            return orig(n, rng=rng, pool=pool, elite_rotation=elite_rotation)

        orch.archive.select_parents = spy
        tasks = [
            TaskDescription(task_id="task_a", description="Solve task_a"),
            TaskDescription(task_id="task_b", description="Solve task_b"),
        ]
        await orch.run(tasks)
        return recorded

    assert await _offsets(True) == [1, 2]
    assert await _offsets(False) == [0, 0]


def test_cli_default_off_and_parse():
    assert _parse([]).elite_rotation is False
    assert _parse(["--elite-rotation"]).elite_rotation is True
    kw = evolutionary_run_kwargs(_parse([]))
    keys = list(kw.keys())
    # Appended AFTER the F031 tail (the frozen-order, append-only contract);
    # the full frozen order is pinned by test_refine_main.py::_FROZEN_EVO_KEYS.
    assert kw["elite_rotation"] is False
    assert keys.index("elite_rotation") > keys.index("temperatures")
    assert EvolutionaryConfig().elite_rotation is False


# ---------------------------------------------------------------------------
# F034 / F035 — CLI wiring (defaults OFF; appended provenance keys)
# ---------------------------------------------------------------------------


def test_focus_current_headroom_cli_wiring():
    assert _parse([]).focus_current_headroom is False
    assert _parse(["--focus-current-headroom"]).focus_current_headroom is True
    kw = evolutionary_run_kwargs(_parse(["--focus-current-headroom"]))
    assert kw["focus_current_headroom"] is True
    assert EvolutionaryConfig().focus_current_headroom is False


def test_eval_repeats_gate_topup_cli_wiring():
    assert _parse([]).eval_repeats_gate_topup is False
    assert _parse(["--eval-repeats-gate-topup"]).eval_repeats_gate_topup is True
    kw = evolutionary_run_kwargs(_parse(["--eval-repeats-gate-topup"]))
    assert kw["eval_repeats_gate_topup"] is True
    assert EvolutionaryConfig().eval_repeats_gate_topup is False


def test_new_bool_flags_survive_stale_namespaces():
    # getattr defaults keep wrapper-script namespaces (without the new attrs)
    # valid — same treatment as the F031 keys.
    ns = _parse([])
    for attr in (
        "elite_rotation", "focus_current_headroom", "eval_repeats_gate_topup",
        "agentic_spend_budget", "agentic_temperature",
    ):
        delattr(ns, attr)
    kw = evolutionary_run_kwargs(ns)
    assert kw["elite_rotation"] is False
    assert kw["focus_current_headroom"] is False
    assert kw["eval_repeats_gate_topup"] is False
    assert kw["agentic_spend_budget"] is None
    assert kw["agentic_temperature"] == 0.7


# ---------------------------------------------------------------------------
# F063 / F060 — --agentic-spend-budget / --agentic-temperature
# ---------------------------------------------------------------------------


def test_agentic_spend_budget_cli_wiring():
    assert _parse([]).agentic_spend_budget is None
    assert _parse(["--agentic-spend-budget", "50000"]).agentic_spend_budget == 50000
    assert evolutionary_run_kwargs(_parse([]))["agentic_spend_budget"] is None
    kw = evolutionary_run_kwargs(_parse(["--agentic-spend-budget", "50000"]))
    assert kw["agentic_spend_budget"] == 50000


def test_agentic_temperature_cli_wiring():
    assert _parse([]).agentic_temperature == 0.7
    assert _parse(["--agentic-temperature", "0.2"]).agentic_temperature == 0.2
    kw = evolutionary_run_kwargs(_parse(["--agentic-temperature", "0.2"]))
    assert kw["agentic_temperature"] == 0.2


def _noop_ns(**overrides):
    ns = _parse([])
    ns.use_archive = True
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


def test_spend_budget_without_agentic_is_noted():
    notes = evolutionary_noop_flag_notes(
        _noop_ns(agentic_spend_budget=50000, use_agentic=False),
        external_spine=False,
    )
    assert len(notes) == 1 and "--agentic-spend-budget" in notes[0]
    assert "CostGuard" in notes[0]


def test_spend_budget_with_agentic_not_noted():
    notes = evolutionary_noop_flag_notes(
        _noop_ns(agentic_spend_budget=50000, use_agentic=True),
        external_spine=False,
    )
    assert notes == []


def test_spend_budget_on_spine_run_is_noted():
    # The builtin AgenticSolver is the ONLY consumer of spend_budget, so on an
    # external-spine run (which has its own CostGuard) the flag is still a
    # silent no-op and the note must fire — same placement as the
    # agentic_temperature note.
    notes = evolutionary_noop_flag_notes(
        _noop_ns(agentic_spend_budget=50000, use_agentic=False),
        external_spine=True,
    )
    assert len(notes) == 1 and "--agentic-spend-budget" in notes[0]
    assert "CostGuard" in notes[0]


def test_agentic_temperature_without_agentic_is_noted():
    notes = evolutionary_noop_flag_notes(
        _noop_ns(agentic_temperature=0.2, use_agentic=False),
        external_spine=False,
    )
    assert len(notes) == 1 and "--agentic-temperature" in notes[0]


def test_agentic_temperature_default_produces_no_note():
    assert evolutionary_noop_flag_notes(_noop_ns(), external_spine=False) == []


# ---------------------------------------------------------------------------
# F075 — --classify-balanced-json-fallback
# ---------------------------------------------------------------------------


def test_classify_balanced_json_fallback_parse():
    assert _parse([]).classify_balanced_json_fallback is False
    assert _parse(
        ["--classify-balanced-json-fallback"]
    ).classify_balanced_json_fallback is True


def test_evolutionary_orchestrator_forwards_balanced_json_fallback():
    orch = EvolutionaryOrchestrator(
        llm_client=MagicMock(), executor=MagicMock(), omega=MagicMock(),
        config=EvolutionaryConfig(), solver_language="python",
        balanced_json_fallback=True,
    )
    assert orch.solver.balanced_json_fallback is True
    default = EvolutionaryOrchestrator(
        llm_client=MagicMock(), executor=MagicMock(), omega=MagicMock(),
        config=EvolutionaryConfig(), solver_language="python",
    )
    assert default.solver.balanced_json_fallback is False


# ---------------------------------------------------------------------------
# F075 / F156 — base-config provenance (both-paths keys, #62 treatment)
# ---------------------------------------------------------------------------


def test_base_run_config_records_both_paths_knobs():
    ns = _parse(["--classify-balanced-json-fallback", "--symmetric-trace-sampling"])
    cfg = build_base_run_config(
        ns, benchmark_name="symptom2disease", solver_language="python",
        executor_name="DummyExecutor",
    )
    assert cfg["classify_balanced_json_fallback"] is True
    assert cfg["symmetric_trace_sampling"] is True
    default = build_base_run_config(
        _parse([]), benchmark_name="symptom2disease", solver_language="python",
        executor_name="DummyExecutor",
    )
    assert default["classify_balanced_json_fallback"] is False
    assert default["symmetric_trace_sampling"] is False


def test_base_run_config_symmetric_trace_sampling_missing_attr_defaults_false():
    # R1-E: getattr-wrapped read tolerates a stale/partial Namespace that omits
    # symmetric_trace_sampling entirely (all other base-config keys present).
    ns = _parse([])
    delattr(ns, "symmetric_trace_sampling")
    cfg = build_base_run_config(
        ns, benchmark_name="symptom2disease", solver_language="python",
        executor_name="DummyExecutor",
    )
    assert cfg["symmetric_trace_sampling"] is False


def test_async_main_wiring_seams_ast_pinned():
    """AST pin of the two async_main wiring seams for the both-paths flags
    (pattern: test_auditfix_main_cli.py::
    test_finding_14_run_config_update_splats_evolutionary_kwargs).

    Parse/provenance/constructor tests all stay green if these call-site
    kwargs are deleted — either flag would then be recorded in config.json yet
    silently no-op end-to-end. Pin that:
      (a) the OmegaEngine(...) call carries
          symmetric_trace_sampling=args.symmetric_trace_sampling (F156);
      (b) the EvolutionaryOrchestrator constructor call carries
          balanced_json_fallback=args.classify_balanced_json_fallback (F075).
    """
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(main_mod))
    fn = next(
        n for n in tree.body
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "async_main"
    )

    def _is_args_attr(node, attr: str) -> bool:
        return (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "args"
            and node.attr == attr
        )

    def _is_args_getattr(node, attr: str) -> bool:
        # getattr(args, "<attr>", False) — the getattr-wrapped read (R1-E).
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) == 3
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == "args"
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value == attr
            and isinstance(node.args[2], ast.Constant)
            and node.args[2].value is False
        )

    def _is_args_ref(node, attr: str) -> bool:
        # R1-E: accept EITHER the bare args.<attr> attribute OR the
        # getattr(args, "<attr>", False) wrapped read (convention-consistency
        # with the other both-paths knobs; stale-namespace safe).
        return _is_args_attr(node, attr) or _is_args_getattr(node, attr)

    omega_ok = False
    orch_ok: dict[str, bool] = {}
    for node in ast.walk(fn):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
            continue
        kwargs = {k.arg: k.value for k in node.keywords if k.arg}
        if node.func.id == "OmegaEngine":
            omega_ok = _is_args_ref(
                kwargs.get("symmetric_trace_sampling"),
                "symmetric_trace_sampling",
            )
        elif node.func.id == "EvolutionaryOrchestrator":
            orch_ok[node.func.id] = _is_args_attr(
                kwargs.get("balanced_json_fallback"),
                "classify_balanced_json_fallback",
            )
    assert omega_ok, (
        "async_main's OmegaEngine(...) call no longer forwards "
        "symmetric_trace_sampling=args.symmetric_trace_sampling"
    )
    assert orch_ok == {
        "EvolutionaryOrchestrator": True,
    }, (
        f"async_main orchestrator constructor missing/lost "
        f"balanced_json_fallback=args.classify_balanced_json_fallback: {orch_ok}"
    )


def test_new_evo_keys_not_in_base_run_config():
    # The five §6b evolutionary keys are archive-only provenance
    # (evolutionary_run_kwargs), never base-config keys.
    cfg = build_base_run_config(
        _parse([]), benchmark_name="co_bench", solver_language="python",
        executor_name="DummyExecutor",
    )
    for key in (
        "elite_rotation", "focus_current_headroom", "eval_repeats_gate_topup",
        "agentic_spend_budget", "agentic_temperature",
    ):
        assert key not in cfg, key


# ---------------------------------------------------------------------------
# F068 — --empty-retry-max-tokens help names the relative cap
# ---------------------------------------------------------------------------


def test_empty_retry_help_names_min_4x_rule():
    help_text = _help_for(_capture_parser(), "--empty-retry-max-tokens")
    assert "min(this, 4x the call's requested max_tokens)" in help_text
