"""P1b: CO-Bench held-out SANDBOX verifier (--verified-code gate).

The override is only ever reached under --verified-code; with the flag OFF it is
never called (byte-identical). These tests exercise the ON behavior with a fake
``run_container`` so no Docker/LLM is touched.
"""
import re
from pathlib import Path

from meta_n.core.verified_code import SandboxedHeldoutVerifier
from meta_n.integrations.co_bench import (
    _CREW_BEST_OF_N,
    _CREW_HELDOUT_RUNNER,
    _CREW_TASK_NAME,
    COBenchAdapter,
)

HELPER_NAME = "solve_crew_scheduling"
HELPER_SRC = "def solve_crew_scheduling(N, K, time_limit, tasks, arcs):\n    return {}\n"


def _crew_adapter():
    return COBenchAdapter(data_dir="/tmp/co_bench", task_names=[_CREW_TASK_NAME])


def test_crew_returns_sandboxed_verifier():
    v = _crew_adapter().make_heldout_verifier()
    assert isinstance(v, SandboxedHeldoutVerifier)
    assert v.image == "python:3.12-slim"
    # mounts point at the crew data dir under /app/crew.
    assert any(cont == "/app/crew" and _CREW_TASK_NAME in host for host, cont in v.mounts)
    assert v.runner_args[0] == "test"
    assert v.runner_args[1] == repr(_CREW_BEST_OF_N)


def test_non_crew_task_returns_none():
    adapter = COBenchAdapter(data_dir="/tmp/co_bench", task_names=["Assignment problem"])
    assert adapter.make_heldout_verifier() is None


def test_crew_mount_source_is_absolute_for_relative_data_dir():
    # R1-C_verify_gate-3: a RELATIVE data_dir must still yield an ABSOLUTE
    # Docker bind-mount SOURCE (a `-v` requirement). Pre-fix the host path was
    # the relative 'data/co_bench/Crew scheduling'; post-fix .resolve() makes it
    # absolute. Inspects v.mounts only — no Docker/LLM.
    adapter = COBenchAdapter(data_dir="./data/co_bench", task_names=[_CREW_TASK_NAME])
    v = adapter.make_heldout_verifier()
    host = next(h for h, cont in v.mounts if cont == "/app/crew")
    assert Path(host).is_absolute()
    assert _CREW_TASK_NAME in host


def _verify_with_fake(fake_result):
    v = _crew_adapter().make_heldout_verifier()
    v._run_container = lambda argv, timeout, container: fake_result
    return v.verify(HELPER_NAME, HELPER_SRC, "crew_scheduling", context_sources=[])


def test_exec_error_drops_fail_closed():
    res = _verify_with_fake({"status": "exec_error", "passed": False})
    assert res.passed is False
    # VerifyResult contract (verified_code.py: ran_in_sandbox docstring): True
    # means the helper actually executed inside the --network none container.
    # exec_error == the `docker run` process never spawned, so it must be False.
    assert res.ran_in_sandbox is False


def test_non_json_drops():
    res = _verify_with_fake({"status": "no_json", "passed": False})
    assert res.passed is False


def test_pass_dict_is_kept():
    res = _verify_with_fake({
        "status": "ok", "passed": True,
        "feasible_subset_mean": 0.74, "feasibility_rate": 0.8,
    })
    assert res.passed is True
    assert res.ran_in_sandbox is True


def test_network_none_and_read_only_in_argv():
    captured = {}

    def fake(argv, timeout, container):
        captured["argv"] = argv
        return {"passed": True}

    v = _crew_adapter().make_heldout_verifier()
    v._run_container = fake
    v.verify(HELPER_NAME, HELPER_SRC, "crew_scheduling", context_sources=[])
    argv = captured["argv"]
    assert "--network" in argv and argv[argv.index("--network") + 1] == "none"
    assert "--read-only" in argv
    assert HELPER_NAME in argv


# ---------------------------------------------------------------------------
# T1.1: the in-container PASS predicate must key on the FULL-SET mean (matched
# to the full-set best-of-N gate), NOT the zeros-dropped feasible-subset mean.
# The predicate runs only inside the --network none container; here we extract
# and evaluate the exact expression from the runner source.
# ---------------------------------------------------------------------------
def _runner_gate(out: dict, gate: float) -> bool:
    m = re.search(r'out\["passed"\] = \((.*?)\)', _CREW_HELDOUT_RUNNER, re.S)
    assert m is not None, "could not locate the PASS predicate in the runner"
    expr = " ".join(m.group(1).split())  # flatten the multi-line predicate
    return bool(eval(expr, {}, {"out": out, "GATE": gate}))  # noqa: S307


def test_runner_pass_predicate_keys_on_full_mean():
    # Source-level: the matched (full_mean) comparison is present and the old
    # mismatched (feasible_subset_mean) comparison is gone.
    assert 'out["full_mean"] > GATE' in _CREW_HELDOUT_RUNNER
    assert 'out["feasible_subset_mean"] > GATE' not in _CREW_HELDOUT_RUNNER


def test_gate_rejects_feasible_subset_only_win():
    # A feasible-subset-only win (subset mean 0.74 > 0.658) but full_mean <= GATE
    # must NOT pass — this is exactly the crew helper case the old gate wrongly
    # passed (audit T1.1).
    out = {"full_mean": 0.5911, "feasible_subset_mean": 0.7356,
           "feasibility_rate": 0.6}
    assert _runner_gate(out, _CREW_BEST_OF_N) is False


def test_gate_passes_matched_full_set_win():
    # A genuine full-set win (full_mean > GATE) with majority feasibility passes.
    out = {"full_mean": 0.70, "feasible_subset_mean": 0.74,
           "feasibility_rate": 0.8}
    assert _runner_gate(out, _CREW_BEST_OF_N) is True


def test_gate_requires_majority_feasibility():
    # Full-set win but feasibility below 50% still fails closed.
    out = {"full_mean": 0.70, "feasible_subset_mean": 0.95,
           "feasibility_rate": 0.4}
    assert _runner_gate(out, _CREW_BEST_OF_N) is False


# ---------------------------------------------------------------------------
# R3-C_verify_gate-1: the crew verify gate now DROPS any helper the name-based
# deploy wrapper could not bind — so verify agrees with the deploy path (every
# verified helper is name-deployable; deployed score == verified score).
# ---------------------------------------------------------------------------
def test_guard_rejects_non_forwardable_helper_before_sandbox():
    calls = []

    def fake(argv, timeout, container):
        calls.append(argv)
        return {"status": "ok", "passed": True}

    v = _crew_adapter().make_heldout_verifier()
    v._run_container = fake

    # A renamed helper (none of the crew deploy keys present) is NOT
    # name-forwardable: the name-deploy wrapper could not bind it, so verify must
    # DROP it — and short-circuit BEFORE the sandbox (the fake is never called).
    renamed = "def solve_crew(n, k, tl, t, a):\n    return {'crews': [[1]]}\n"
    res = v.verify("solve_crew", renamed, "crew_scheduling", context_sources=[])
    assert res.passed is False
    assert res.ran_in_sandbox is False
    assert calls == []  # guard short-circuited: sandbox never invoked

    # Companion: a canonical helper with a **kwargs catch-all IS forwardable, so
    # it reaches the sandbox and is kept (the fake returns passed=True).
    kw_src = (
        "def solve_crew_scheduling(N, K, time_limit, tasks, arcs, **kwargs):\n"
        "    return {}\n"
    )
    res2 = v.verify(
        "solve_crew_scheduling", kw_src, "crew_scheduling", context_sources=[]
    )
    assert res2.passed is True
    assert calls  # the sandbox WAS invoked for the forwardable helper
