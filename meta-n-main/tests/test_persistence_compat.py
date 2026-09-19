"""Persistence / back-compat guardrails (P0.2).

These tests freeze the invariants that every later schema-touching step
(num_children resume, task_solution_map, oracle fields, stopping) must keep:
on-disk archives stay loadable, summary.json stays additive, and the telemetry
run_id basis is byte-stable. They are the safety net for Steps 1, 4, 6, 8.
"""

import json
from pathlib import Path

import pytest

from meta_n.core.archive import Archive
from meta_n.core.evolutionary_orchestrator import EvolutionaryResult
from meta_n.core.external_agents.telemetry import compute_run_id
from meta_n.core.meta_layer import InjectedCode

LEGACY = Path(__file__).parent / "fixtures" / "legacy_archive"

# The frozen public schema of summary.json (EvolutionaryResult.to_dict). The
# analysis tooling, experiment_results.html generator, and extract_paper_numbers
# all read these keys — they must stay present with stable types.
# `run_status`/`total_iterations` presence also DISCRIMINATES final from
# mid-run summaries (F038) — the mid-run shape is pinned in
# tests/test_r6b_run_persistence.py.
FROZEN_SUMMARY_KEYS = {
    "archive_size", "total_iterations", "total_tokens", "token_usage",
    "best_mean_score", "best_candidate_id", "oracle_mean_score",
    "test_mean_score", "chain_test_mean_score", "per_task_best_scores",
    "convergence_history",
    "run_status",  # N9a — additive, always present
}


def test_resume_restores_num_children(tmp_path, make_candidate, write_candidate_dir):
    """Master resume-parity test: a parent's child-count must survive a rebuild.

    Step 1 (2.2): ``rebuild_from_disk`` recomputes num_children from parent_id
    edges, self-healing the stale per-candidate summary value (written once at
    creation = 0 and never re-flushed).
    """
    archive_dir = tmp_path / "archive"
    archive = Archive()

    # Seed parent — its per-candidate summary is written at creation, when
    # num_children is still 0 (this is the on-disk reality of a real run).
    parent = make_candidate("gen0_seed", mean_score=0.5, iteration=0)
    archive.add(parent)
    write_candidate_dir(archive_dir, parent)  # stale summary: num_children == 0

    # Two children are spawned; the orchestrator bumps parent.num_children
    # in-memory and re-flushes the LIVE archive index (index.json), but never
    # re-writes the parent's per-candidate summary.
    for cid in ("gen1_b0_k0", "gen1_b0_k1"):
        child = make_candidate(cid, mean_score=0.55, parent_id="gen0_seed", iteration=1)
        archive.add(child)
        write_candidate_dir(archive_dir, child)
        parent.num_children += 1
    (archive_dir / "index.json").write_text(json.dumps(archive.to_dict()))
    assert parent.num_children == 2  # live, in-memory and in index.json

    rebuilt = Archive.rebuild_from_disk(archive_dir)
    assert rebuilt.get("gen0_seed").num_children == 2


def test_legacy_archive_loads_without_new_fields():
    """A frozen pre-v2 archive (no v2 fields) must still rebuild cleanly."""
    archive = Archive.rebuild_from_disk(LEGACY)
    assert len(archive) == 2
    seed = archive.get("gen0_seed")
    assert seed.depth == 1
    child = archive.get("gen1_b0_k0")
    assert child.parent_id == "gen0_seed"
    assert child.depth == 2
    # The injected code from the legacy layer round-trips through the model.
    assert len(child.injected_codes) == 1
    assert "robust_zscore" in child.injected_codes[0].code_library


def test_legacy_injected_code_validates():
    raw = json.loads((LEGACY / "gen1_b0_k0" / "injected_code_d2.json").read_text())
    ic = InjectedCode.model_validate(raw)
    assert ic.source_depth == 2
    assert ic.pre_process.startswith("# legacy layer")


def test_summary_headline_contract():
    """EvolutionaryResult.to_dict keeps exactly the frozen public keys/types."""
    r = EvolutionaryResult(
        archive_size=3, total_iterations=2, total_tokens=1000,
        best_mean_score=0.75, best_candidate_id="gen1_b0_k0",
        per_task_best_scores={"t1": 0.8, "t2": 0.7},
        oracle_mean_score=0.8, convergence_history=[0.5, 0.75],
    )
    d = r.to_dict()
    assert set(d.keys()) == FROZEN_SUMMARY_KEYS
    assert isinstance(d["token_usage"], dict)
    assert isinstance(d["per_task_best_scores"], dict)
    assert isinstance(d["convergence_history"], list)
    assert isinstance(d["best_candidate_id"], str)
    # Legacy runs (no external-agent rollup) must NOT carry the optional key.
    assert "agent_telemetry_rollup" not in d


def test_run_id_basis_is_stable():
    """The telemetry run_id basis is byte-stable; only non-eval phases differ."""
    assert compute_run_id(0, "gen0_seed", "task_a", 1) == "53d6684ae4016abd"
    assert compute_run_id(0, "gen0_seed", "task_a", 1, "eval") == "53d6684ae4016abd"
    assert compute_run_id(0, "gen0_seed", "task_a", 1, "gate") == "b75e82b0926cf865"
    assert compute_run_id(0, "gen0_seed", "task_a", 1, "gate") != \
        compute_run_id(0, "gen0_seed", "task_a", 1)
