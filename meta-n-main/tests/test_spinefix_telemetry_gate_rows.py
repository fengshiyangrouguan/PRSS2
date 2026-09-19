"""Regression tests for the gate-row cache self-bound (spinefix, finding #8).

Every gate-phase run is cached in the process-lifetime dict
``AgentTelemetry._gate_rows`` so a later gate-reuse can re-emit it as an
eval-phase clone (:meth:`restamp_reused_gate`). The ONLY eviction is that
restamp, which fires solely for gate-PASSING candidates reused at eval. A
gate-REJECTED candidate is never evaluated, so pre-fix its rows were never
popped and lived for the whole process — one dead entry per rejected task per
generation, unbounded growth.

The child loop is sequential (gate → reject/continue OR pass → eval+restamp
resolves a candidate before the next is gated), so ``_append_run`` bounds the
cache to the current candidate coordinate: the arrival of a gate row for a
DIFFERENT ``(generation, candidate_id, depth)`` frees any stale rows. These
tests are fully offline (filesystem only; no Docker / LLM / network).
"""

import json

import pytest

from meta_n.core.external_agents.telemetry import (
    AgentRunRecord,
    AgentTelemetry,
)


class _Solver:
    def __init__(self, candidate_id, generation=0, depth=1):
        self.candidate_id = candidate_id
        self.generation = generation
        self.depth = depth


class _Task:
    def __init__(self, task_id):
        self.task_id = task_id


def _gate_rec(run_id, candidate_id, task_id, *, generation=0, depth=1):
    return AgentRunRecord(
        run_id=run_id,
        generation=generation,
        candidate_id=candidate_id,
        task_id=task_id,
        depth=depth,
        agent="openhands",
        phase="gate",
        success=True,
    )


def _cached_candidates(tel):
    return {row.get("candidate_id") for row in tel._gate_rows.values()}


def test_rejected_candidate_gate_rows_are_freed(tmp_path):
    """A gate-REJECTED candidate's cached rows must NOT leak once the next
    candidate is gated. Pre-fix the cache kept growing; post-fix it is bounded
    to the current candidate coordinate."""
    tel = AgentTelemetry(tmp_path)

    # Candidate that will be REJECTED: several gate tasks cached, never reused.
    for task in ("t1", "t2", "t3"):
        tel._append_run(_gate_rec(f"g:{task}:rej", "C_reject", task))
    assert len(tel._gate_rows) == 3
    assert _cached_candidates(tel) == {"C_reject"}

    # Next candidate gets gated (sequential loop) -> C_reject is now fully
    # resolved (it was rejected, so restamp never popped its rows). The new
    # coordinate must evict the dead rows.
    tel._append_run(_gate_rec("g:t1:next", "C_next", "t1"))

    assert "C_reject" not in _cached_candidates(tel), (
        "gate-rejected candidate rows leaked past the next candidate's gate"
    )
    assert len(tel._gate_rows) == 1
    assert _cached_candidates(tel) == {"C_next"}


def test_many_rejected_candidates_do_not_accumulate(tmp_path):
    """Across many generations of rejected candidates the cache stays O(1) in
    the current candidate's gate-task count, not O(total rejected tasks)."""
    tel = AgentTelemetry(tmp_path)
    for gen in range(50):
        cand = f"C{gen}"
        for task in ("t1", "t2"):
            tel._append_run(
                _gate_rec(f"g{gen}:{task}", cand, task, generation=gen)
            )
        # Each candidate is rejected; only the freshest candidate's rows live.
        assert _cached_candidates(tel) == {cand}
        assert len(tel._gate_rows) == 2


def test_same_candidate_rows_accumulate_across_tasks(tmp_path):
    """Rows for the SAME candidate coordinate (more gate tasks / --gate-repeats)
    accumulate as before — the bound must not evict the current candidate."""
    tel = AgentTelemetry(tmp_path)
    # Two tasks plus a --gate-repeats r1 row, all for one candidate.
    tel._append_run(_gate_rec("g:t1:gate", "C", "t1"))
    tel._append_run(_gate_rec("g:t1:gate:r1", "C", "t1"))
    tel._append_run(_gate_rec("g:t2:gate", "C", "t2"))
    assert len(tel._gate_rows) == 3
    assert _cached_candidates(tel) == {"C"}


def test_restamp_reused_gate_still_pops_and_reemits(tmp_path):
    """The prior #21/#68 reused-gate restamp behavior is intact: a gate-PASSING
    candidate's row is popped and re-emitted as an eval-phase clone."""
    tel = AgentTelemetry(tmp_path)
    tel._append_run(_gate_rec("g:t1:gate", "C_pass", "t1"))
    tel._append_run(_gate_rec("g:t2:gate", "C_pass", "t2"))
    assert len(tel._gate_rows) == 2

    ok = tel.restamp_reused_gate(_Task("t1"), _Solver("C_pass"))
    assert ok is True
    # Only t1 is popped (restamp matches task_id); t2 survives for its own reuse.
    assert "g:t1:gate" not in tel._gate_rows
    assert "g:t2:gate" in tel._gate_rows

    # The physical run is re-emitted once as an eval-phase row on disk.
    runs_file = tel.telemetry_dir / AgentTelemetry.RUNS_FILE
    eval_rows = [
        rec
        for line in runs_file.read_text().splitlines()
        if line.strip()
        # The AgentRunRecord is nested at line["extra"]["record"] on disk.
        for rec in [json.loads(line).get("extra", {}).get("record", {})]
        if rec.get("run_id") == "g:t1:gate" and rec.get("phase") == "eval"
    ]
    assert len(eval_rows) == 1


def test_restamp_after_reject_eviction_returns_false(tmp_path):
    """Once a rejected candidate's rows are evicted by the next gate, a stray
    restamp for the old candidate is a safe no-op (returns False)."""
    tel = AgentTelemetry(tmp_path)
    tel._append_run(_gate_rec("g:t1:rej", "C_reject", "t1"))
    tel._append_run(_gate_rec("g:t1:next", "C_next", "t1"))

    assert tel.restamp_reused_gate(_Task("t1"), _Solver("C_reject")) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
