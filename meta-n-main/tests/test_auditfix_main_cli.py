"""Offline regression tests for audit fixes in meta_n/main.py.

One test per assigned finding. Each test is designed to FAIL against the
pre-fix source and PASS after the fix. No LLM / Docker / network / Omega is
touched — these only inspect the argparse parser, ``build_base_run_config``,
and the static source of the run_config provenance dict.

Findings covered:
  * #14 — --deploy-verified-code / --seed-code-library now recorded in the
          --use-archive run_config.update provenance dict.
  * #26 — --daily-budget-usd help text states the real 500 USD Azure default
          (was a stale 295).
  * #27 — build_base_run_config records the task-subset-determining ``seed``
          on BOTH orchestrator paths (was dropped on the linear path).
  * #62 — build_base_run_config records the result-affecting LLM/Omega flags
          omega_context_budget / exclude_providers / request_timeout /
          instance_workers on BOTH paths.
"""

import argparse

import meta_n.main as main_mod
from meta_n.main import build_base_run_config


def _capture_parser() -> argparse.ArgumentParser:
    """Run ``parse_args()`` but intercept the final ``parser.parse_args()`` so
    we get the live ArgumentParser (with real action help strings) without
    touching ``sys.argv`` or exiting."""
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


def _default_args() -> argparse.Namespace:
    """A real default Namespace (every flag at its argparse default)."""
    real = argparse.ArgumentParser.parse_args
    captured = {}

    def _spy(self, *a, **k):
        ns = real(self, [])  # parse an empty argv => all defaults
        captured["ns"] = ns
        return ns

    argparse.ArgumentParser.parse_args = _spy
    try:
        return main_mod.parse_args()
    finally:
        argparse.ArgumentParser.parse_args = real


def _help_for(parser: argparse.ArgumentParser, option: str) -> str:
    for action in parser._actions:
        if option in (action.option_strings or []):
            return action.help or ""
    raise AssertionError(f"option {option!r} not found on parser")


# --------------------------------------------------------------------------
# #26 — --daily-budget-usd help text default drift (295 -> 500)
# --------------------------------------------------------------------------
def test_finding_26_daily_budget_help_states_500_not_295():
    parser = _capture_parser()
    help_text = _help_for(parser, "--daily-budget-usd")
    # The code default in _resolve_budget is 500.0 for Azure; the help must
    # match it and must NOT advertise the stale 295 default.
    assert "500" in help_text, help_text
    assert "295" not in help_text, help_text


# --------------------------------------------------------------------------
# #27 — seed recorded by build_base_run_config (BOTH paths, incl. linear)
# --------------------------------------------------------------------------
def test_finding_27_base_run_config_records_seed():
    args = _default_args()
    args.seed = 1234
    cfg = build_base_run_config(
        args,
        benchmark_name="terminal_bench",
        solver_language="bash",
        executor_name="DummyExecutor",
    )
    assert "seed" in cfg, sorted(cfg)
    assert cfg["seed"] == 1234


# --------------------------------------------------------------------------
# #62 — result-affecting LLM/Omega flags recorded by build_base_run_config
# --------------------------------------------------------------------------
def test_finding_62_base_run_config_records_result_affecting_flags():
    args = _default_args()
    args.omega_context_budget = 32768
    args.exclude_providers = ["together"]
    args.request_timeout = 1200.0
    args.instance_workers = 4
    cfg = build_base_run_config(
        args,
        benchmark_name="co_bench",
        solver_language="python",
        executor_name="DummyExecutor",
    )
    for key, expected in (
        ("omega_context_budget", 32768),
        ("exclude_providers", ["together"]),
        ("request_timeout", 1200.0),
        ("instance_workers", 4),
    ):
        assert key in cfg, (key, sorted(cfg))
        assert cfg[key] == expected, (key, cfg[key])


# --------------------------------------------------------------------------
# #14 — deploy_verified_code / seed_code_library recorded in the
#       --use-archive run_config provenance (F048: the provenance dict is now
#       built by evolutionary_run_kwargs; the runtime check covers key
#       existence, and the AST check below pins the one wiring line that
#       splats the dict into run_config — together they close the loop the
#       old update({...}) literal pin covered).
# --------------------------------------------------------------------------
def test_finding_14_archive_run_config_records_deploy_and_seed_library():
    keys = set(main_mod.evolutionary_run_kwargs(_default_args()))
    # Both behavioral flags must have a provenance channel in config.json.
    assert "deploy_verified_code" in keys, sorted(keys)
    assert "seed_code_library" in keys, sorted(keys)


def test_finding_14_run_config_update_splats_evolutionary_kwargs():
    """async_main must splat evolutionary_run_kwargs into run_config.

    evolutionary_run_kwargs() carrying the right keys is worthless if the
    single wiring line ``run_config.update({..., **evo_kwargs})`` disappears —
    every archive-run provenance key would silently vanish from config.json
    while the suite stays green. Pin the splat in async_main's AST.
    """
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(main_mod))
    fn = next(
        n for n in tree.body
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "async_main"
    )
    for node in ast.walk(fn):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "update"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "run_config"
        ):
            for arg in node.args:
                if isinstance(arg, ast.Dict) and any(
                    k is None  # a None key in ast.Dict is a **splat
                    and isinstance(v, ast.Name)
                    and v.id == "evo_kwargs"
                    for k, v in zip(arg.keys, arg.values)
                ):
                    return
    raise AssertionError(
        "async_main no longer splats evolutionary_run_kwargs (evo_kwargs) "
        "into run_config — archive-run provenance would be silently lost"
    )
