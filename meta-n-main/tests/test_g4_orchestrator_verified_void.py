"""G4 (silent-void audit) — orchestrator + verified_code plumbing fixes.

Covers four CONFIRMED audit findings. Each proves the NEW (fixed) behavior AND
that the DEFAULT / flag-OFF path is unchanged (the all-OFF byte-identity is also
guarded by tests/golden/test_stage23_golden.py).

  * T2.1 — --paired-eval/CRN is a structural no-op on a backend without
    per-request seed support (only azure). The archive path now WARNS and stamps
    ``paired_eval_effective`` into summary.json. Default OFF ⇒ no warning, no key.
  * T2.2 — --verified-code with no family value-oracle falls back to the keep-all
    StubHeldoutVerifier (inert DROP gate); _make_heldout_verifier now WARNS.
  * T3.2 — a STUB-KEPT (ran_in_sandbox=False) helper must NOT enter the
    non-adoption penalty set; a sandbox-verified one must.
  * T3.3/T3.4 — the demoting-family code_library zeroing exempts source_depth==0
    seeds and warns about anything it zeroes under --verified-code/--seed-code-library;
    default (no flags) is the legacy silent ``{}``.
"""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from meta_n.core.archive import Candidate
from meta_n.core.evolutionary_orchestrator import (
    EvolutionaryConfig,
    EvolutionaryOrchestrator,
    EvolutionaryResult,
)
from meta_n.core.meta_layer import InjectedCode, Trace, merge_code_libraries
from meta_n.core.verified_code import StubHeldoutVerifier


def _orch(*, llm_client=None, **cfg) -> EvolutionaryOrchestrator:
    d = dict(max_depth=20, parallel=1, patience=8, gate_tasks=0,
             beam_width=1, beam_candidates=1)
    d.update(cfg)
    return EvolutionaryOrchestrator(
        llm_client=llm_client or MagicMock(),
        executor=MagicMock(), omega=MagicMock(),
        config=EvolutionaryConfig(**d), solver_language="bash",
    )


def _llm(model: str, backend: str):
    return SimpleNamespace(config=SimpleNamespace(model=model, backend=backend))


# --- T2.1 — paired_eval no-op warning + summary stamping --------------------

def test_paired_eval_off_is_silent_and_unstamped(tmp_path, caplog):
    """Default OFF ⇒ no warning, effective stays False, summary.json has NO key."""
    with caplog.at_level(logging.WARNING):
        orch = _orch(paired_eval=False, llm_client=_llm("local-gemma", "openrouter"))
    assert orch._paired_eval_effective is False
    assert not any("paired-eval" in r.message.lower() for r in caplog.records)

    orch.save_results(EvolutionaryResult(), output_dir=str(tmp_path))
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert "paired_eval_effective" not in summary  # byte-identical legacy summary


def test_paired_eval_on_unsupported_backend_warns_and_stamps_false(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        orch = _orch(paired_eval=True, llm_client=_llm("local-gemma", "openrouter"))
    assert orch._paired_eval_effective is False
    msgs = " ".join(r.message for r in caplog.records)
    assert "NO-OP" in msgs and "paired-eval" in msgs.lower()
    # T3.5 caveat is carried in the same warning.
    assert "depth>1" in msgs

    orch.save_results(EvolutionaryResult(), output_dir=str(tmp_path))
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["paired_eval_effective"] is False


def test_paired_eval_on_azure_is_effective_no_warning(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        orch = _orch(paired_eval=True, llm_client=_llm("gpt-4.1", "azure"))
    assert orch._paired_eval_effective is True
    assert not any("NO-OP" in r.message for r in caplog.records)

    orch.save_results(EvolutionaryResult(), output_dir=str(tmp_path))
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["paired_eval_effective"] is True


# --- T2.2 — stub fallback warns under verified_code -------------------------

def test_make_heldout_verifier_warns_on_stub_fallback(caplog):
    orch = _orch(verified_code=True)
    orch.adapter = None  # no value-oracle ⇒ stub
    with caplog.at_level(logging.WARNING):
        v = orch._make_heldout_verifier()
    assert isinstance(v, StubHeldoutVerifier)
    assert any("INERT" in r.message and r.levelno == logging.WARNING
               for r in caplog.records)


# --- T3.2 — ran_in_sandbox gates the non-adoption penalty -------------------

def _cand_advertised_not_called(name: str) -> Candidate:
    tr = Trace(
        task_id="t1", depth=2, script="x", success=True, score=0.9,
        utilities_available=[name], utilities_called=[],
    )
    return Candidate(candidate_id="c1", parent_id="gen0_seed", iteration=0,
                     depth=2, traces=[tr], per_task_scores={"t1": 0.9},
                     mean_score=0.9)


def test_stub_kept_helper_not_barred_but_sandbox_verified_is():
    orch = _orch(verified_code=True)
    injected = InjectedCode(code_library={"good_helper": "def good_helper(x): return x"})
    cand = _cand_advertised_not_called("good_helper")

    # Stub-kept (ran_in_sandbox=False) ⇒ empty sandbox set ⇒ NOT barred.
    assert orch._nonadopting_verified_tasks(cand, injected, set()) == set()
    # Sandbox-verified ⇒ in the set ⇒ barred (re-derived inline).
    assert orch._nonadopting_verified_tasks(
        cand, injected, {"good_helper"}
    ) == {"t1"}
    # Legacy call (None) ⇒ old behavior (every code_library key treated verified).
    assert orch._nonadopting_verified_tasks(cand, injected) == {"t1"}


# --- T3.3 / T3.4 — demoting-family zeroing exemption + warning --------------

def _seed_ic(name="seed_helper"):
    return InjectedCode(code_library={name: f"def {name}(x): return x"}, source_depth=0)


def _omega_ic(name="omega_helper", depth=2):
    return InjectedCode(code_library={name: f"def {name}(x): return x"}, source_depth=depth)


def test_demote_default_no_flags_is_silent_empty(caplog):
    orch = _orch()  # no verified_code, no seed_code_library
    merged = {"omega_helper": "def omega_helper(x): return x"}
    with caplog.at_level(logging.WARNING):
        out = orch._demote_python_library(merged, [_omega_ic()])
    assert out == {}  # legacy byte-identical zeroing
    assert not caplog.records  # silent


def test_demote_exempts_source_depth0_seed_under_seed_flag(caplog):
    seeded = {"seed_helper": "def seed_helper(x): return x"}
    orch = _orch(seed_code_library=seeded)
    merged = dict(seeded)
    merged["omega_helper"] = "def omega_helper(x): return x"
    with caplog.at_level(logging.WARNING):
        out = orch._demote_python_library(merged, [_seed_ic(), _omega_ic()])
    # source_depth==0 seed survives; the depth-2 Ω helper is zeroed + warned.
    assert set(out) == {"seed_helper"}
    assert any("omega_helper" in r.message for r in caplog.records)


def test_demote_warns_on_verified_kept_helper_zeroed(caplog):
    orch = _orch(verified_code=True)
    merged = {"omega_helper": "def omega_helper(x): return x"}
    with caplog.at_level(logging.WARNING):
        out = orch._demote_python_library(merged, [_omega_ic()])
    # No source_depth==0 contribution ⇒ everything zeroed, but loudly.
    assert out == {}
    assert any("omega_helper" in r.message and "force-code-library-live" in r.message
               for r in caplog.records)


def test_demote_collision_keeps_seed_source_not_omega_override(caplog):
    seed_src = "def foo(x): return x  # seed"
    omega_src = "def foo(x): return x*2  # omega"
    orch = _orch(seed_code_library={"foo": seed_src})
    chain = [InjectedCode(code_library={"foo": seed_src}, source_depth=0),
             InjectedCode(code_library={"foo": omega_src}, source_depth=2)]
    merged_py, _ = merge_code_libraries(chain)
    assert merged_py["foo"] == omega_src   # merge: later overrides earlier
    with caplog.at_level(logging.WARNING):
        out = orch._demote_python_library(merged_py, chain)
    assert out == {"foo": seed_src}        # exemption keeps the SEED, not Omega
    assert any("collision" in r.message.lower() and "foo" in r.message
               for r in caplog.records)
