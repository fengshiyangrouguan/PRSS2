"""Forensic improvement #2 — VERIFIED code_library + FORCED ADOPTION.

Flag-gated behind ``--verified-code`` (config.verified_code), default OFF.
Default-OFF byte-identity is covered by ``tests/golden/test_stage23_golden.py``
and ``tests/test_stage1_golden.py``; this module proves the flag-ON behavior:

  (1) DROP-FAILING / KEEP-PASSING — the verify-gate sandbox-executes each Ω helper
      against a held-out check and keeps ONLY the helpers that pass; pre_process /
      bash pass through; the omega object is NOT mutated (monotonic).
  (2) SANDBOXED — every container invocation carries ``--network none`` AND
      ``--read-only`` (Ω code runs ONLY in-container, never host).
  (3) STUB SEAM — with no adapter verifier, helpers are KEPT and flagged
      ``ran_in_sandbox=False`` / UNVERIFIED (capability-preserving, clean seam).
  (4) FORCED ADOPTION — the foster affordance (REQUIRED + wired skeleton) is
      present only when ON; the MetaLayer is built with foster_adoption forced ON.
  (5) INLINE-RE-DERIVATION PENALTY — a trace that advertised a verified helper but
      re-derived it inline is barred from per-task-best (composes with #1's guard).

HONEST SCOPE: #2 is build-only / UNMEASURED — no live value-oracle bed in this
build except FEAL, so the sandbox is exercised here via an injected ``run_container``
fake (Docker-free) plus a single REAL-Docker integration test (skipif no docker).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from meta_n.core.archive import Archive, Candidate
from meta_n.core.evolutionary_orchestrator import (
    EvolutionaryConfig,
    EvolutionaryOrchestrator,
)
from meta_n.core.meta_layer import (
    InjectedCode,
    TaskDescription,
    Trace,
    format_python_library_descriptions,
)
from meta_n.core.verified_code import (
    SandboxedHeldoutVerifier,
    StubHeldoutVerifier,
    VerifyResult,
    _docker_run_json,
)


# --- helpers ---------------------------------------------------------------

GOOD_SRC = "def good_helper(x):\n    \"\"\"Add one.\"\"\"\n    return x + 1"
BAD_SRC = "def bad_helper(x):\n    \"\"\"Wrong / explodes.\"\"\"\n    raise RuntimeError('boom')"


def _tasks(names=("t1",)):
    return [TaskDescription(task_id=n, description=f"Solve {n}") for n in names]


def _orch(**cfg) -> EvolutionaryOrchestrator:
    d = dict(max_depth=20, parallel=1, patience=8, gate_tasks=0,
             beam_width=1, beam_candidates=1)
    d.update(cfg)
    return EvolutionaryOrchestrator(
        llm_client=MagicMock(), executor=MagicMock(), omega=MagicMock(),
        config=EvolutionaryConfig(**d), solver_language="bash",
    )


def _fake_sandboxed(verdicts: dict, recorder: list) -> SandboxedHeldoutVerifier:
    """A SandboxedHeldoutVerifier whose container exec is faked: it records the
    argv (for the --network none / --read-only assertions) and returns canned
    per-helper JSON WITHOUT Docker."""

    def fake_run(argv, timeout_s, container):
        recorder.append(list(argv))
        idx = argv.index("/app/_runner.py")
        name = argv[idx + 1]
        return {"passed": bool(verdicts[name])}

    return SandboxedHeldoutVerifier(
        image="dummy-image",
        runner_src="print('{\"passed\": true}')",
        run_container=fake_run,
    )


def _cand(cid, scores, *, depth=1, parent_id=None, traces=None):
    traces = traces if traces is not None else [
        Trace(task_id=t, depth=depth, script="def solve(**k): pass",
              success=True, score=s)
        for t, s in scores.items()
    ]
    return Candidate(
        candidate_id=cid, parent_id=parent_id, iteration=0, depth=depth,
        injected_codes=[], traces=traces, pass_at_1=1.0,
        mean_score=sum(scores.values()) / len(scores),
        per_task_scores=dict(scores),
    )


# --- (0) defaults OFF -------------------------------------------------------

def test_config_default_off():
    assert EvolutionaryConfig().verified_code is False


def test_verify_gate_off_is_identity():
    """Flag OFF ⇒ the gate returns the SAME object (no sandbox, no copy)."""
    orch = _orch(verified_code=False)
    injected = InjectedCode(
        pre_process="echo hi",
        code_library={"good_helper": GOOD_SRC, "bad_helper": BAD_SRC},
    )
    # Even with a verifier that would fail bad_helper, OFF never consults it.
    orch._make_heldout_verifier = lambda: _fake_sandboxed(
        {"good_helper": True, "bad_helper": False}, []
    )
    import asyncio
    out, sandboxed = asyncio.run(
        orch._verify_and_filter_code_library(injected, [], _tasks())
    )
    assert out is injected  # identical object, byte-for-byte
    assert set(out.code_library) == {"good_helper", "bad_helper"}
    assert sandboxed == set()  # OFF ⇒ no sandbox-verified names


# --- (1)+(2) DROP-FAILING / KEEP-PASSING, sandboxed -------------------------

async def test_drop_failing_keep_passing_sandboxed():
    orch = _orch(verified_code=True)
    recorder: list = []
    orch._make_heldout_verifier = lambda: _fake_sandboxed(
        {"good_helper": True, "bad_helper": False}, recorder
    )
    injected = InjectedCode(
        pre_process="export X=1",
        code_library={"good_helper": GOOD_SRC, "bad_helper": BAD_SRC},
        code_library_bash={"bh": "echo hello"},
    )
    out, sandboxed = await orch._verify_and_filter_code_library(
        injected, [], _tasks()
    )

    # Failing helper dropped, passing kept.
    assert set(out.code_library) == {"good_helper"}
    assert "bad_helper" not in out.code_library
    # The kept helper genuinely ran in the sandbox (ran_in_sandbox=True).
    assert sandboxed == {"good_helper"}
    # pre_process + bash helpers pass through untouched.
    assert out.pre_process == "export X=1"
    assert out.code_library_bash == {"bh": "echo hello"}
    # Monotonic: the original omega object is NOT mutated (still has both).
    assert set(injected.code_library) == {"good_helper", "bad_helper"}
    assert out is not injected

    # Sandboxed: BOTH helpers were exec'd with --network none AND --read-only.
    assert len(recorder) == 2
    for argv in recorder:
        assert "--network" in argv
        assert argv[argv.index("--network") + 1] == "none"
        assert "--read-only" in argv
        # No host exec path: the only program is python inside the container image.
        assert argv[0] == "docker" and "run" in argv


async def test_all_helpers_pass_returns_equivalent_library():
    orch = _orch(verified_code=True)
    orch._make_heldout_verifier = lambda: _fake_sandboxed(
        {"good_helper": True}, []
    )
    injected = InjectedCode(code_library={"good_helper": GOOD_SRC})
    out, sandboxed = await orch._verify_and_filter_code_library(
        injected, [], _tasks()
    )
    # Nothing dropped ⇒ identical object returned.
    assert out is injected
    assert sandboxed == {"good_helper"}


# --- (1b) SIBLING CONTEXT + DEPENDENCY-AWARE KEEP ---------------------------

_SIB_SRC = "def _sib(x):\n    return x * 2"
_ENTRY_CALLS_SIB = "def solve_entry(x):\n    return _sib(x) + 1"
_DEAD_SRC = "def _dead(x):\n    return x - 1"
_ENTRY_NO_DEAD = "def solve_entry(x):\n    return x + 1"


class _RecordingVerifier:
    """Records the context each helper was verified WITH and applies per-name
    verdicts (a bool, or a predicate over the context list)."""

    def __init__(self, verdicts: dict):
        self.verdicts = verdicts
        self.calls: dict = {}

    def verify(self, name, source, task_id, context_sources):
        ctx = list(context_sources or [])
        self.calls[name] = ctx
        rule = self.verdicts[name]
        passed = rule(ctx) if callable(rule) else bool(rule)
        return VerifyResult(passed=passed, evidence=name, ran_in_sandbox=True)


async def test_entry_point_verified_with_sibling_context_and_dep_kept():
    """An entry-point helper that CALLS a sibling verifies only when the sibling
    is in its container context (deploy prepends ALL helpers). The sibling — which
    fails the single-entry-point oracle standalone — is kept back because the
    passing entry-point name-calls it (deploy would NameError otherwise). The
    dependency-kept sibling is NOT flagged sandbox-verified (never verified AS an
    entry point → the T3.2 penalty stays on genuinely-verified names only)."""
    orch = _orch(verified_code=True)
    verifier = _RecordingVerifier({
        "solve_entry": lambda ctx: any(_SIB_SRC in s for s in ctx),
        "_sib": False,  # fails as a standalone entry point
    })
    orch._make_heldout_verifier = lambda: verifier
    injected = InjectedCode(
        code_library={"solve_entry": _ENTRY_CALLS_SIB, "_sib": _SIB_SRC}
    )
    out, sandboxed = await orch._verify_and_filter_code_library(
        injected, [], _tasks()
    )
    # solve_entry's verify context carried _sib's source (matched to deploy).
    assert any(_SIB_SRC in s for s in verifier.calls["solve_entry"])
    # Both kept: entry passes with the sibling available; _sib retained because
    # the kept entry-point name-calls it.
    assert set(out.code_library) == {"solve_entry", "_sib"}
    # Only the genuinely-verified entry point is sandbox-verified.
    assert sandboxed == {"solve_entry"}


async def test_uncalled_failing_sibling_still_dropped():
    """NEGATIVE / drop-dead preserved: a helper that fails standalone AND is not
    name-called by any kept helper is still dropped."""
    orch = _orch(verified_code=True)
    verifier = _RecordingVerifier({
        "solve_entry": True,   # passes; does NOT call _dead
        "_dead": False,        # fails standalone, uncalled
    })
    orch._make_heldout_verifier = lambda: verifier
    injected = InjectedCode(
        code_library={"solve_entry": _ENTRY_NO_DEAD, "_dead": _DEAD_SRC}
    )
    out, sandboxed = await orch._verify_and_filter_code_library(
        injected, [], _tasks()
    )
    assert set(out.code_library) == {"solve_entry"}
    assert sandboxed == {"solve_entry"}


# --- (3) STUB SEAM ----------------------------------------------------------

def test_stub_verifier_keeps_and_flags_unverified():
    res = StubHeldoutVerifier().verify("good_helper", GOOD_SRC, "t1", [])
    assert res.passed is True
    assert res.ran_in_sandbox is False
    assert "stub" in res.evidence.lower() or "unverified" in res.evidence.lower()


async def test_verify_gate_falls_back_to_stub_keeps_all():
    """No adapter verifier ⇒ StubHeldoutVerifier ⇒ both helpers KEPT (capability-
    preserving), nothing dropped, identical object."""
    orch = _orch(verified_code=True)
    orch.adapter = None  # no held-out harness
    injected = InjectedCode(code_library={"good_helper": GOOD_SRC, "bad_helper": BAD_SRC})
    out, sandboxed = await orch._verify_and_filter_code_library(
        injected, [], _tasks()
    )
    assert out is injected
    assert set(out.code_library) == {"good_helper", "bad_helper"}
    # Stub kept both but verified NEITHER (ran_in_sandbox=False).
    assert sandboxed == set()


def test_make_heldout_verifier_prefers_adapter_else_stub():
    orch = _orch(verified_code=True)
    # adapter returns None ⇒ stub
    orch.adapter = SimpleNamespace(make_heldout_verifier=lambda: None)
    assert isinstance(orch._make_heldout_verifier(), StubHeldoutVerifier)
    # adapter returns a real verifier ⇒ used
    sentinel = StubHeldoutVerifier()
    orch.adapter = SimpleNamespace(make_heldout_verifier=lambda: sentinel)
    assert orch._make_heldout_verifier() is sentinel
    # adapter raises ⇒ stub fallback (never crashes the run)
    def _boom():
        raise ValueError("nope")
    orch.adapter = SimpleNamespace(make_heldout_verifier=_boom)
    assert isinstance(orch._make_heldout_verifier(), StubHeldoutVerifier)


# --- (4) FORCED ADOPTION ----------------------------------------------------

def test_forced_adoption_affordance_present_only_when_on():
    from meta_n.core.prompts import LIBRARY_INSTRUCTIONS_FOSTER

    lib = {"good_helper": GOOD_SRC}
    on = format_python_library_descriptions(lib, executor=None, foster_adoption=True)
    off = format_python_library_descriptions(lib, executor=None, foster_adoption=False)

    assert LIBRARY_INSTRUCTIONS_FOSTER.strip() in on
    assert "Wired solve() skeleton" in on
    assert "MUST call" in on
    # OFF: the foster affordance is absent (byte-identical to the legacy advert).
    assert LIBRARY_INSTRUCTIONS_FOSTER.strip() not in off
    assert "Wired solve() skeleton" not in off


def test_verified_code_forces_foster_on_metalayer():
    lib_ic = InjectedCode(code_library={"good_helper": GOOD_SRC})
    cand = Candidate(
        candidate_id="c1", parent_id="gen0_seed", iteration=0, depth=2,
        injected_codes=[lib_ic],
    )
    # verified_code ON forces the MetaLayer foster affordance ON...
    solver_on = _orch(verified_code=True)._build_solver_from_candidate(cand)
    assert solver_on.foster_adoption is True
    # ...and OFF (with foster_adoption also OFF) is byte-identical (False).
    solver_off = _orch(verified_code=False, foster_adoption=False)._build_solver_from_candidate(cand)
    assert solver_off.foster_adoption is False
    # foster_adoption alone still works (X or False == X).
    solver_foster = _orch(verified_code=False, foster_adoption=True)._build_solver_from_candidate(cand)
    assert solver_foster.foster_adoption is True


# --- (5) INLINE-RE-DERIVATION PENALTY ---------------------------------------

def test_nonadopting_verified_tasks():
    orch = _orch(verified_code=True)
    injected = InjectedCode(code_library={"good_helper": GOOD_SRC})

    def _cand_with(util_avail, util_called):
        tr = Trace(
            task_id="t1", depth=2, script="x", success=True, score=0.9,
            utilities_available=util_avail, utilities_called=util_called,
        )
        return _cand("g1", {"t1": 0.9}, depth=2, traces=[tr])

    # Advertised the verified helper but re-derived it inline (called == []) → barred.
    assert orch._nonadopting_verified_tasks(
        _cand_with(["good_helper"], []), injected
    ) == {"t1"}
    # Adopted (called the helper) → not barred.
    assert orch._nonadopting_verified_tasks(
        _cand_with(["good_helper"], ["good_helper"]), injected
    ) == set()
    # Unmeasurable (utilities_called is None) → never penalized.
    assert orch._nonadopting_verified_tasks(
        _cand_with(["good_helper"], None), injected
    ) == set()
    # No verified helpers at all → empty.
    assert orch._nonadopting_verified_tasks(
        _cand_with(["good_helper"], []), InjectedCode()
    ) == set()


def test_prior_gen_verified_helper_barred():
    """R1-C_verify_gate-7: the non-adoption penalty must cover a PRIOR-generation
    verified helper still staged in the merged chain, not just the newest layer —
    with override-aware aggregation (a later unverified redefinition clears the
    stale verification)."""
    orch = _orch(verified_code=True)
    L1 = InjectedCode(
        code_library={"prior_helper": GOOD_SRC},
        sandbox_verified_names=["prior_helper"],
    )
    L2 = InjectedCode(code_library={"new_helper": "def new_helper(x): return x"})

    def _child(util_avail, util_called, *, l2=L2):
        tr = Trace(
            task_id="t1", depth=3, script="x", success=True, score=0.9,
            utilities_available=util_avail, utilities_called=util_called,
        )
        return Candidate(
            candidate_id="child", parent_id="gen0_seed", iteration=0, depth=3,
            injected_codes=[L1, l2], traces=[tr], per_task_scores={"t1": 0.9},
            mean_score=0.9,
        )

    # prior_helper (verified in L1) was staged but re-derived inline (only
    # new_helper called) → barred. Pre-fix (newest-only verified={new_helper})
    # this was set() — prior-gen verification was invisible.
    child = _child(["new_helper", "prior_helper"], ["new_helper"])
    assert orch._nonadopting_verified_tasks(child, L2, {"new_helper"}) == {"t1"}

    # (i) BOTH verified helpers adopted → not barred.
    child_ok = _child(["new_helper", "prior_helper"], ["new_helper", "prior_helper"])
    assert orch._nonadopting_verified_tasks(child_ok, L2, {"new_helper"}) == set()

    # (ii) L2 REDEFINES prior_helper but does NOT verify it (omitted from L2's
    # sandbox set) → last-writer-wins clears the stale verification, so the trace
    # re-deriving it inline is NOT penalized.
    L2_override = InjectedCode(code_library={
        "new_helper": "def new_helper(x): return x",
        "prior_helper": "def prior_helper(x): return x  # unverified stub",
    })
    child_override = _child(
        ["new_helper", "prior_helper"], ["new_helper"], l2=L2_override,
    )
    assert orch._nonadopting_verified_tasks(
        child_override, L2_override, {"new_helper"}
    ) == set()

    # End-to-end: the prior-gen bar actually blocks per-task-best (mirrors
    # test_archive_bar_from_best_penalty).
    arch = Archive()
    arch.add(_cand("gen0_seed", {"t1": 0.5}))
    arch.add(_cand("child", {"t1": 0.9}, depth=3), bar_from_best={"t1"})
    assert arch.per_task_best_scores()["t1"] == pytest.approx(0.5)
    assert "child" in arch._by_id


def test_archive_bar_from_best_penalty():
    # Barred non-adopting trace cannot become per-task-best, but is still archived.
    arch = Archive()
    arch.add(_cand("gen0_seed", {"t1": 0.5}))
    arch.add(_cand("g1", {"t1": 0.9}, depth=2), bar_from_best={"t1"})
    assert arch.per_task_best_scores()["t1"] == pytest.approx(0.5)  # barred
    assert "g1" in arch._by_id  # monotonic: candidate still present
    assert arch.get("g1").mean_score == pytest.approx(0.9)  # mean intact

    # Without the bar, the same trace WOULD win — proving the penalty is load-bearing.
    arch2 = Archive()
    arch2.add(_cand("gen0_seed", {"t1": 0.5}))
    arch2.add(_cand("g1", {"t1": 0.9}, depth=2))
    assert arch2.per_task_best_scores()["t1"] == pytest.approx(0.9)


def test_archive_default_add_byte_identical_no_bar():
    # Default add() (no bar_from_best) is unchanged: the higher trace wins.
    arch = Archive()
    arch.add(_cand("gen0_seed", {"t1": 0.5}))
    arch.add(_cand("g1", {"t1": 0.9}, depth=2))
    assert arch.per_task_best_scores()["t1"] == pytest.approx(0.9)


# --- VerifyResult / docker seam ---------------------------------------------

def test_docker_run_json_no_json_fails_closed():
    """A non-JSON / empty stdout maps to passed=False (fail closed) — never raises."""
    out = _docker_run_json(["true"], 5.0, "noexist")
    assert isinstance(out, dict)
    assert out.get("passed") is False


def test_verify_result_shape():
    r = VerifyResult(passed=True, evidence="ok", ran_in_sandbox=True)
    assert (r.passed, r.evidence, r.ran_in_sandbox) == (True, "ok", True)


# --- REAL-Docker integration (skipif no docker / image) ---------------------

_PY_IMAGE_CANDIDATES = ("python:3.12-slim", "python:3.11-slim", "python:3-slim", "python:3")


def _docker_image_present(image: str) -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        proc = subprocess.run(
            ["docker", "image", "inspect", image],
            capture_output=True, timeout=15,
        )
        return proc.returncode == 0
    except Exception:
        return False


def _available_py_image():
    for image in _PY_IMAGE_CANDIDATES:
        if _docker_image_present(image):
            return image
    return None


_PY_IMAGE = _available_py_image()


_REAL_RUNNER = (
    "import json, importlib.util, sys\n"
    "name = sys.argv[1]\n"
    "ok = False\n"
    "try:\n"
    "    spec = importlib.util.spec_from_file_location('cand', '/app/_candidate.py')\n"
    "    m = importlib.util.module_from_spec(spec)\n"
    "    spec.loader.exec_module(m)\n"
    "    fn = getattr(m, name, None)\n"
    "    ok = callable(fn) and fn(2) == 3\n"  # held-out check: helper(2) must equal 3
    "except Exception:\n"
    "    ok = False\n"
    "print(json.dumps({'passed': bool(ok)}))\n"
)


@pytest.mark.skipif(
    _PY_IMAGE is None,
    reason="docker or a python:* slim image not available locally",
)
def test_sandboxed_verifier_real_container():
    """End-to-end: the REAL --network none container passes a correct helper and
    fails a wrong one, with NO host execution of the Ω code."""
    verifier = SandboxedHeldoutVerifier(
        image=_PY_IMAGE,
        runner_src=_REAL_RUNNER,
        runner_args=[],
        wall_clock_s=60.0,
    )
    good = verifier.verify("add_one", "def add_one(x):\n    return x + 1", "t1", [])
    assert good.passed is True
    assert good.ran_in_sandbox is True

    bad = verifier.verify("add_one", "def add_one(x):\n    return x + 99", "t1", [])
    assert bad.passed is False
    assert bad.ran_in_sandbox is True
