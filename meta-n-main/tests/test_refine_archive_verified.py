"""Refinement wave C3 (archive / verified_code / self_repair) regression gate.

Covers:
* F005 — ``SandboxedHeldoutVerifier.verify`` provenance: an ``exec_error``
  status (the ``docker run`` process never spawned) must report
  ``ran_in_sandbox=False`` per the VerifyResult contract; ``timeout`` /
  ``no_json`` keep ``True`` (the container was invoked).
* F011 — wire: ``GRANULARITIES`` / ``REPAIR_CLASSES`` pin the literals
  production actually writes into the persisted self-repair schema.
* F018 — ``Archive.rebuild_from_disk`` default ``novelty_alpha`` matches the
  fresh-archive / EvolutionaryConfig default (tri-way pin).
* F019 — rebuild restores per-candidate inner-LLM accounting +
  self-repair sidecars; legacy summaries default to zeros / empty.
* F175 — wire: the concrete verifiers structurally satisfy the
  ``HeldoutVerifier`` Protocol seam.

No Docker / LLM / SDK: verifier tests use a fake ``run_container``.
"""

from __future__ import annotations

import json

from unittest.mock import AsyncMock

from meta_n.core.archive import Archive, Candidate
from meta_n.core.evolutionary_orchestrator import (
    EvolutionaryConfig,
    EvolutionaryOrchestrator,
)
from meta_n.core.meta_layer import InjectedCode, Trace
from meta_n.core.omega import OmegaEngine
from meta_n.core.self_repair import (
    GRANULARITIES,
    REPAIR_CLASSES,
    SelfRepairEvent,
    classify_repair,
)
from meta_n.core.verified_code import (
    HeldoutVerifier,
    SandboxedHeldoutVerifier,
    StubHeldoutVerifier,
)


# ---------------------------------------------------------------------------
# F005 — exec_error provenance
# ---------------------------------------------------------------------------

def _verify_with_fake(fake_result):
    v = SandboxedHeldoutVerifier(
        image="python:3.12-slim",
        runner_src="print('{}')",
        run_container=lambda argv, timeout, container: fake_result,
    )
    return v.verify("helper", "def helper():\n    return 1\n", "t1", context_sources=[])


def test_exec_error_reports_ran_in_sandbox_false():
    # exec_error == the `docker run` process itself never executed, so the
    # helper never ran in the sandbox — the provenance bit must be honest.
    res = _verify_with_fake({"status": "exec_error", "passed": False})
    assert res.passed is False
    assert res.ran_in_sandbox is False


def test_timeout_and_no_json_keep_ran_in_sandbox_true():
    # timeout / no_json mean the container WAS invoked (executed or at least
    # started); only the spawn-failure case is demoted.
    for status in ("timeout", "no_json"):
        res = _verify_with_fake({"status": status, "passed": False})
        assert res.passed is False
        assert res.ran_in_sandbox is True


# ---------------------------------------------------------------------------
# F011 — wire GRANULARITIES / REPAIR_CLASSES to production literals
# ---------------------------------------------------------------------------

def test_granularities_pin_the_orchestrator_literals():
    # The orchestrator writes granularity="within_layer" (Stage-2 refine) and
    # "downstream" (Stage-3 re-propagation) as string literals; the persisted
    # schema's closed set must stay in lock-step.
    assert GRANULARITIES == {"within_layer", "downstream"}
    assert SelfRepairEvent().granularity in GRANULARITIES


def test_repair_classes_pin_classify_repair_codomain():
    assert REPAIR_CLASSES == {"error_correction", "novel"}
    # Same approach (identical rationale + pre_process) -> localized fix.
    pre = InjectedCode(
        pre_process="task.description += ' hint'",
        rationale="add a parsing hint for the grid tasks",
    )
    post = InjectedCode(
        pre_process="task.description += ' hint fixed'",
        rationale="add a parsing hint for the grid tasks",
    )
    verdict = classify_repair(pre, post)
    assert verdict == "error_correction"
    assert verdict in REPAIR_CLASSES
    # New code_library helper -> approach changed.
    post_novel = post.model_copy(
        update={"code_library": {"solve_fast": "def solve_fast():\n    pass\n"}}
    )
    verdict_novel = classify_repair(pre, post_novel)
    assert verdict_novel == "novel"
    assert verdict_novel in REPAIR_CLASSES


# ---------------------------------------------------------------------------
# F018 — rebuild_from_disk default novelty_alpha
# ---------------------------------------------------------------------------

def test_rebuild_default_novelty_alpha_matches_fresh_archive(tmp_path):
    # Tri-way pin: a future drift in ANY of the three defaults fails loudly.
    rebuilt = Archive.rebuild_from_disk(tmp_path / "empty")
    assert (
        rebuilt.novelty_alpha
        == Archive().novelty_alpha
        == EvolutionaryConfig().novelty_alpha
        == 0.3
    )


# ---------------------------------------------------------------------------
# F019 — inner-LLM accounting + self-repair provenance survive resume
# ---------------------------------------------------------------------------

def _orch(tmp_path) -> EvolutionaryOrchestrator:
    cfg = EvolutionaryConfig()
    cfg.output_dir = str(tmp_path)
    executor = AsyncMock()
    executor.adapter = None
    return EvolutionaryOrchestrator(
        llm_client=AsyncMock(),
        executor=executor,
        omega=OmegaEngine(None),
        config=cfg,
    )


def _write_summary(archive_dir, cid, extra=None):
    cand_dir = archive_dir / cid
    cand_dir.mkdir(parents=True)
    summary = {
        "candidate_id": cid,
        "parent_id": None,
        "iteration": 0,
        "depth": 1,
        "mean_score": 0.5,
        "pass_at_1": 1.0,
        "per_task_scores": {"t1": 0.5},
        "num_children": 0,
        "temperature_used": 0.7,
        "total_tokens": 10,
        "created_at": "2026-06-25T09:00:49.105752",
    }
    summary.update(extra or {})
    (cand_dir / "summary.json").write_text(json.dumps(summary))
    return cand_dir


def test_summary_inner_keys_gated_absent_when_zero(tmp_path):
    # Byte-identity gate for the writer: a candidate with NO inner-LLM usage
    # keeps the inner_* keys ABSENT (not zero) in summary.json.
    orch = _orch(tmp_path)
    cand = Candidate(
        candidate_id="gen0_seed", depth=1, mean_score=0.5,
        traces=[Trace(task_id="t1", depth=1, score=0.5, success=True)],
    )
    orch._save_candidate_incremental(cand, tmp_path)
    summary = json.loads(
        (tmp_path / "archive" / "gen0_seed" / "summary.json").read_text()
    )
    for key in ("inner_tokens", "inner_prompt_tokens",
                "inner_completion_tokens", "inner_calls"):
        assert key not in summary


def test_rebuild_restores_inner_llm_accounting(tmp_path):
    # A summary carrying the gated inner_* keys restores them onto the
    # rebuilt candidate (per-candidate cost integrity across resume).
    archive_dir = tmp_path / "archive"
    _write_summary(archive_dir, "c1", extra={
        "inner_tokens": 120,
        "inner_prompt_tokens": 80,
        "inner_completion_tokens": 40,
        "inner_calls": 3,
    })
    archive = Archive.rebuild_from_disk(archive_dir)
    cand = archive.get("c1")
    assert cand.inner_tokens == 120
    assert cand.inner_prompt_tokens == 80
    assert cand.inner_completion_tokens == 40
    assert cand.inner_calls == 3


def test_rebuild_restores_self_repair_events(tmp_path):
    archive_dir = tmp_path / "archive"
    cand_dir = _write_summary(archive_dir, "c1")
    event = SelfRepairEvent(
        candidate_id="c1", granularity="downstream", target_depth=2,
        mean_before=0.1, mean_after=0.4, accepted=True, archived=True,
        raw_omega_prompt="PROMPT", raw_omega_response="RESPONSE",
    )
    (cand_dir / "repropagation_d2.json").write_text(
        json.dumps(event.to_sidecar_json())
    )
    # A corrupt sidecar is skipped with a warning — never sinks the candidate.
    (cand_dir / "repropagation_d3.json").write_text("{not json")

    archive = Archive.rebuild_from_disk(archive_dir)
    cand = archive.get("c1")
    assert len(cand.self_repair_events) == 1
    ev = cand.self_repair_events[0]
    assert ev.candidate_id == "c1"
    assert ev.granularity == "downstream"
    assert ev.target_depth == 2
    assert ev.mean_before == 0.1
    assert ev.mean_after == 0.4
    assert ev.accepted is True
    assert ev.archived is True
    # Raw Ω transcripts are sidecar-.txt-only: they restore as "" (parity with
    # InjectedCode's raw Ω text exclusion).
    assert ev.raw_omega_prompt == ""
    assert ev.raw_omega_response == ""


def test_rebuild_legacy_summary_defaults(tmp_path):
    # Legacy summaries (no inner_* keys, no sidecars) rebuild exactly as
    # before: zero counters + empty provenance.
    archive_dir = tmp_path / "archive"
    _write_summary(archive_dir, "c_legacy")
    archive = Archive.rebuild_from_disk(archive_dir)
    cand = archive.get("c_legacy")
    assert cand.inner_tokens == 0
    assert cand.inner_prompt_tokens == 0
    assert cand.inner_completion_tokens == 0
    assert cand.inner_calls == 0
    assert cand.self_repair_events == []


# ---------------------------------------------------------------------------
# F175 — HeldoutVerifier Protocol wired to its concrete implementations
# ---------------------------------------------------------------------------

def test_concrete_verifiers_satisfy_the_protocol():
    # runtime_checkable makes this a real structural check of the seam the
    # orchestrator's verify-gate is documented against.
    assert isinstance(StubHeldoutVerifier(), HeldoutVerifier)
    assert isinstance(
        SandboxedHeldoutVerifier(image="i", runner_src="r"), HeldoutVerifier
    )
    from meta_n.integrations.co_bench import _CREW_TASK_NAME, COBenchAdapter

    verifier = COBenchAdapter(
        data_dir="/tmp/co_bench", task_names=[_CREW_TASK_NAME]
    ).make_heldout_verifier()
    assert isinstance(verifier, HeldoutVerifier)
