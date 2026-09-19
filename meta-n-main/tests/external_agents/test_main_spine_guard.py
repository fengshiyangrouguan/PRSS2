"""Item #2: main.py external-only startup guards gate on the SAME external-spine
capability probe the orchestrator uses (``_uses_external_spine``), not a hardcoded
``base_solver in ('openhands','terminus2')`` check.

The three guards (require --use-archive, require --daily-budget-usd>0, the
budget-precheck headroom note) must:
  * still fire for the genuine external kinds (openhands / terminus2);
  * NEWLY fire for a spine-routed ``builtin`` on an advertising adapter
    (terminal_bench), which previously slipped past them;
  * stay NEUTRAL for a non-advertising adapter (CO-Bench / classify) and for the
    ``--tasks`` (adapter=None) path — i.e. behave byte-identically to the old
    hardcoded check there.

This pins the exact boolean expression main.py evaluates so a regression in the
guard predicate is caught without spinning up Docker / an LLM.
"""

from __future__ import annotations

import asyncio

import pytest

from types import SimpleNamespace

from meta_n.integrations.co_bench import COBenchAdapter
from meta_n.main import async_main, build_base_run_config


def _external_spine(base_solver: str | None, adapter: object) -> bool:
    """The exact predicate main.py AND the orchestrator share (F042:
    core/spine_routing.uses_external_spine) to gate the external-only guards."""
    from meta_n.core.spine_routing import uses_external_spine

    return uses_external_spine(base_solver, adapter)


def test_cobench_adapter_does_not_advertise_spine_builtin():
    adapter = COBenchAdapter()
    assert adapter.advertises_spine_builtin() is False


def test_builtin_cobench_is_not_external_spine():
    # The LIVE paused run's control invocation: builtin + co_bench. external_spine
    # must be False so ALL THREE guards are skipped, identical to the old
    # hardcoded ``base_solver in ('openhands','terminus2')`` check.
    adapter = COBenchAdapter()
    assert _external_spine("builtin", adapter) is False
    # None base_solver (legacy default) is likewise off the spine.
    assert _external_spine(None, adapter) is False


def test_external_kinds_are_external_spine():
    adapter = COBenchAdapter()
    assert _external_spine("openhands", adapter) is True
    assert _external_spine("terminus2", adapter) is True


def test_tasks_path_adapter_none_is_not_external_spine():
    # The ``--tasks`` branch leaves adapter=None; the getattr default lambda
    # makes the probe NameError-safe and False.
    assert _external_spine("builtin", None) is False
    assert _external_spine(None, None) is False
    # External kinds still fire even with adapter=None (they don't consult it).
    assert _external_spine("openhands", None) is True


def test_advertising_adapter_promotes_builtin_to_external_spine():
    # An adapter that advertises spine-builtin (parity with terminal_bench)
    # flips builtin onto the spine so the three guards now cover it.
    class _AdvertisingAdapter:
        def advertises_spine_builtin(self) -> bool:
            return True

    adapter = _AdvertisingAdapter()
    assert _external_spine("builtin", adapter) is True
    # ...but only via the literal True (strict ``is True``), mirroring the
    # orchestrator's strict check: a truthy non-True must not promote.
    class _TruthyAdapter:
        def advertises_spine_builtin(self):
            return "yes"  # truthy but not the literal True

    assert _external_spine("builtin", _TruthyAdapter()) is False


# ---------------------------------------------------------------------------
# The linear (no --use-archive) orchestrator has been retired: a no-archive run
# is refused at startup rather than silently dropping evolutionary-only flags.
# ---------------------------------------------------------------------------


def _startup_args(tmp_path, **overrides):
    """A namespace that reaches the no-archive startup guard, which fires right
    after budget resolution and BEFORE the LLM client / adapter are built."""
    tasks_file = tmp_path / "tasks.json"
    tasks_file.write_text('[{"task_id": "t1", "description": "d"}]')
    base = dict(
        model="m", base_url="u", api_key="k", max_tokens=1,
        request_timeout=None, empty_retry_max_tokens=0, exclude_providers=None,
        azure_endpoint=None, azure_api_version="v", daily_budget_usd=None,
        cost_ledger_dir=None, local_exec=False, benchmark=None,
        tasks=str(tasks_file), base_solver=None, resume=False,
        use_agentic=False, use_archive=False, benchmark_config="none",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_no_archive_run_is_refused(tmp_path, capsys, monkeypatch):
    # A run without --use-archive (even carrying evolutionary-only flags that
    # the retired linear path once silently dropped) is refused at startup.
    monkeypatch.delenv("META_N_DAILY_BUDGET_USD", raising=False)
    args = _startup_args(tmp_path, eval_repeats=2)
    assert asyncio.run(async_main(args)) == 1
    out = capsys.readouterr().out
    assert "--use-archive is required" in out
    assert "the linear orchestrator has been retired" in out


def test_no_archive_refused_before_llm_client_build(tmp_path, capsys, monkeypatch):
    # Regression: the guard must fire BEFORE LLMClient construction, so a
    # no-archive run without credentials gets the clean retirement message
    # rather than a misleading "Missing credentials" OpenAI error.
    monkeypatch.delenv("META_N_DAILY_BUDGET_USD", raising=False)
    args = _startup_args(tmp_path, api_key=None, base_url=None)
    assert asyncio.run(async_main(args)) == 1
    assert "--use-archive is required" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# T2.5: --local-exec was never wired into executor selection, so async_main
# refuses to start when it is passed (instead of silently no-op'ing). The guard
# is the very first statement, so it short-circuits before any LLM/benchmark
# setup; the default-off path falls through to that setup.
# ---------------------------------------------------------------------------


def test_local_exec_is_rejected_at_startup(capsys):
    # local_exec=True -> guard fires (F032: returns 1 = misconfiguration exit
    # code) before touching args.model et al.
    result = asyncio.run(async_main(SimpleNamespace(local_exec=True)))
    assert result == 1
    assert "--local-exec is not supported" in capsys.readouterr().out


def test_local_exec_off_does_not_short_circuit():
    # Default-off (local_exec absent/False): the guard does NOT fire, so async_main
    # proceeds past it and fails only when it reaches the (here-missing) LLM config
    # fields — proving the guard is inert on the default path.
    with pytest.raises(AttributeError):
        asyncio.run(async_main(SimpleNamespace(local_exec=False)))


def test_base_run_config_omits_ablation_keys():
    """H8: the COMMON run_config keeps bench_tasks (task filtering applies to
    both paths) but OMITS the evolutionary-only ablation provenance keys, so a
    linear run's config.json no longer records ablations it cannot apply."""
    args = SimpleNamespace(
        model="m", base_url="u", max_tokens=1, empty_retry_max_tokens=2,
        epsilon=0.0, max_depth=1,
        parallel=1, tasks="t.json", n_few_shot=0, max_val=0, max_retries=0,
        retry_threshold=0.0, bench_tasks=None,
        # audit #27/#62: build_base_run_config now records these provenance keys
        # (seed determines the task subset; the rest are result-affecting LLM/Ω
        # flags) — they are NOT the evolutionary-only ablation keys asserted below.
        seed=42, omega_context_budget=None, exclude_providers=None,
        request_timeout=None, instance_workers=0,
        # Refine §6b F156/F075: two more both-paths result-affecting knobs
        # recorded by build_base_run_config (#62 treatment).
        symmetric_trace_sampling=False, classify_balanced_json_fallback=False,
    )
    cfg = build_base_run_config(
        args, benchmark_name="b", solver_language="bash", executor_name="X",
    )
    assert "bench_tasks" in cfg
    for key in (
        "no_code_library", "no_outer_context",
        "foster_adoption", "force_code_library_live",
    ):
        assert key not in cfg
