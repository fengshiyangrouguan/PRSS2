"""compute_run_id determinism + resume de-dup on read (§12.1)."""

from __future__ import annotations

import hashlib

from meta_n.core.external_agents.telemetry import (
    AgentRunRecord,
    AgentTelemetry,
    compute_run_id,
)

from .conftest import read_run_records


# --- determinism -----------------------------------------------------------


def test_run_id_is_deterministic_in_four_coordinates():
    a = compute_run_id(0, "cand-1", "task-7", 1)
    b = compute_run_id(0, "cand-1", "task-7", 1)
    assert a == b
    # 16 hex chars.
    assert len(a) == 16
    assert all(c in "0123456789abcdef" for c in a)


def test_run_id_matches_documented_formula():
    basis = "3:candX:taskY:2"
    expected = hashlib.sha1(basis.encode("utf-8")).hexdigest()[:16]
    assert compute_run_id(3, "candX", "taskY", 2) == expected


def test_run_id_changes_with_each_coordinate():
    base = compute_run_id(0, "c", "t", 1)
    assert compute_run_id(1, "c", "t", 1) != base
    assert compute_run_id(0, "c2", "t", 1) != base
    assert compute_run_id(0, "c", "t2", 1) != base
    assert compute_run_id(0, "c", "t", 2) != base


# --- execution-phase discriminator (item #1: gate vs eval de-dup) ----------


def test_run_id_phase_eval_is_byte_identical_to_default():
    # The default ('eval') basis MUST stay byte-identical so existing spine
    # run_ids and the golden formula test do not shift on resume.
    for coords in ((0, "c", "t", 2), (3, "candX", "taskY", 2), (1, "c2", "t9", 1)):
        assert compute_run_id(*coords, "eval") == compute_run_id(*coords)
        assert compute_run_id(*coords, None) == compute_run_id(*coords)


def test_run_id_gate_phase_is_distinct_from_eval():
    # A 'gate' execution must NOT collide with the full-eval execution at the
    # same (generation, candidate_id, task_id, depth) coordinates.
    coords = (0, "c", "t", 2)
    assert compute_run_id(*coords, "gate") != compute_run_id(*coords)
    assert compute_run_id(*coords, "gate") != compute_run_id(*coords, "eval")


def test_gate_and_eval_rows_both_written_same_coordinates(tmp_path):
    # The de-dup undercount this fixes: gate + eval at identical coordinates
    # must yield TWO rows (distinct ids), not one.
    out = tmp_path / "run"
    tel = AgentTelemetry(out)
    gate_id = compute_run_id(0, "cand", "task-1", 2, "gate")
    eval_id = compute_run_id(0, "cand", "task-1", 2, "eval")
    tel._append_run(_record(gate_id, task_id="task-1"))  # noqa: SLF001
    tel._append_run(_record(eval_id, task_id="task-1"))  # noqa: SLF001
    rows = read_run_records(out)
    assert {row["run_id"] for row in rows} == {gate_id, eval_id}
    assert len(rows) == 2


def test_gate_phase_resume_still_dedups_within_phase(tmp_path):
    # Resume idempotency is preserved WITHIN each phase: a re-run gate row
    # de-dups against the prior gate row (one gate row total).
    out = tmp_path / "run"
    gate_id = compute_run_id(0, "cand", "task-1", 2, "gate")
    tel1 = AgentTelemetry(out)
    tel1._append_run(_record(gate_id, task_id="task-1"))  # noqa: SLF001
    tel2 = AgentTelemetry(out)  # resumed process: index rebuilt from disk
    assert gate_id in tel2._seen_run_ids  # noqa: SLF001
    tel2._append_run(_record(gate_id, task_id="task-1"))  # noqa: SLF001 - dropped
    rows = read_run_records(out)
    assert len(rows) == 1


def test_start_record_threads_solver_execution_phase(tmp_path):
    # start_record reads ``solver.execution_phase`` (the orchestrator stamps
    # 'gate' on the spine solver) and folds it into the run_id, WITHOUT any
    # change to start_record's public signature (solver.py is untouched).
    class _Backend:
        name = "terminus2"
        outer_token_mode = False

    class _Solver:
        def __init__(self, phase=None):
            self.backend = _Backend()
            self.depth = 2
            self.generation = 0
            self.candidate_id = "c0"
            self.max_turns = 8
            if phase is not None:
                self.execution_phase = phase

    class _Task:
        task_id = "task-1"

    tel = AgentTelemetry(out := tmp_path / "run")
    # No execution_phase attr → defaults to 'eval' (byte-identical id).
    rec_default = tel.start_record(_Task(), _Solver())
    assert rec_default.run_id == compute_run_id(0, "c0", "task-1", 2)
    # execution_phase='gate' → distinct id.
    rec_gate = tel.start_record(_Task(), _Solver(phase="gate"))
    assert rec_gate.run_id == compute_run_id(0, "c0", "task-1", 2, "gate")
    assert rec_gate.run_id != rec_default.run_id
    # explicit 'eval' → byte-identical to default.
    rec_eval = tel.start_record(_Task(), _Solver(phase="eval"))
    assert rec_eval.run_id == rec_default.run_id


def test_repeat_index_distinct_but_default_byte_identical():
    # H6: the median-of-R / gate-repeat sample index gives each sample a DISTINCT
    # run_id (so R samples write R rows, not one), while the default repeat_index 0
    # keeps the basis byte-identical to the historical formula.
    coords = (0, "c", "t", 1)
    # repeat_index 0 (default) with eval phase == the 4-coord historical id.
    assert compute_run_id(*coords) == compute_run_id(*coords, "eval", 0)
    # A non-zero repeat index changes the id; successive indices differ.
    assert compute_run_id(*coords, "eval", 1) != compute_run_id(*coords)
    assert compute_run_id(*coords, "eval", 2) != compute_run_id(*coords, "eval", 1)
    # The gate phase at repeat 0 is also distinct from eval (item #1 still holds).
    assert compute_run_id(*coords, "gate", 0) != compute_run_id(*coords)
    # Phase and repeat compose into a unique id.
    gate_r1 = compute_run_id(*coords, "gate", 1)
    assert gate_r1 not in {
        compute_run_id(*coords),
        compute_run_id(*coords, "gate", 0),
        compute_run_id(*coords, "eval", 1),
    }


def test_start_record_threads_solver_repeat_index(tmp_path):
    # start_record reads ``solver.repeat_index`` (the orchestrator stamps it around
    # each extra median-of-R / gate sample) and folds it into the run_id + stamps
    # rec.phase, WITHOUT any change to the public signature.
    class _Backend:
        name = "terminus2"
        outer_token_mode = False

    class _Solver:
        def __init__(self, repeat_index=None, phase=None):
            self.backend = _Backend()
            self.depth = 2
            self.generation = 0
            self.candidate_id = "c0"
            self.max_turns = 8
            if repeat_index is not None:
                self.repeat_index = repeat_index
            if phase is not None:
                self.execution_phase = phase

    class _Task:
        task_id = "task-1"

    tel = AgentTelemetry(tmp_path / "run")
    # No repeat_index attr → defaults to 0 (byte-identical id) and phase 'eval'.
    rec_default = tel.start_record(_Task(), _Solver())
    assert rec_default.run_id == compute_run_id(0, "c0", "task-1", 2)
    assert rec_default.phase == "eval"
    # repeat_index=1 → distinct id matching the documented basis.
    rec_r1 = tel.start_record(_Task(), _Solver(repeat_index=1))
    assert rec_r1.run_id == compute_run_id(0, "c0", "task-1", 2, "eval", 1)
    assert rec_r1.run_id != rec_default.run_id
    # repeat_index=0 explicit → byte-identical to default.
    rec_r0 = tel.start_record(_Task(), _Solver(repeat_index=0))
    assert rec_r0.run_id == rec_default.run_id
    # gate phase + repeat_index compose (both discriminators fold in).
    rec_gate_r2 = tel.start_record(_Task(), _Solver(repeat_index=2, phase="gate"))
    assert rec_gate_r2.run_id == compute_run_id(0, "c0", "task-1", 2, "gate", 2)
    assert rec_gate_r2.phase == "gate"


def test_run_id_independent_of_session_uuid():
    # The session/container uuid (ext-{task}-{uuid}) is NOT an input — two runs
    # with different uuids but identical coordinates share one run_id.
    id1 = compute_run_id(0, "c", "t", 1)
    id2 = compute_run_id(0, "c", "t", 1)
    # (uuid would differ between the two runs; run_id does not.)
    assert id1 == id2


# --- resume de-dup on write ------------------------------------------------


def _record(run_id, task_id="t", score=1.0):
    return AgentRunRecord(run_id=run_id, task_id=task_id, agent="terminus2", score=score)


def test_resume_dedup_drops_duplicate_row(tmp_path):
    out = tmp_path / "run"
    tel = AgentTelemetry(out)
    rid = compute_run_id(0, "cand", "task-1", 1)

    tel._append_run(_record(rid))  # noqa: SLF001 - exercising the de-dup path
    tel._append_run(_record(rid))  # same run_id -> dropped on write

    rows = read_run_records(out)
    assert len(rows) == 1
    assert rows[0]["run_id"] == rid


def test_resume_dedup_survives_process_restart(tmp_path):
    out = tmp_path / "run"
    rid = compute_run_id(0, "cand", "task-1", 1)

    tel1 = AgentTelemetry(out)
    tel1._append_run(_record(rid))  # noqa: SLF001

    # New telemetry object (simulates a resumed process): the seen-id index is
    # rebuilt from disk so the re-executed task de-dups against the prior row.
    tel2 = AgentTelemetry(out)
    assert rid in tel2._seen_run_ids  # noqa: SLF001
    tel2._append_run(_record(rid))  # noqa: SLF001 - duplicate, dropped

    rows = read_run_records(out)
    assert len(rows) == 1


def test_distinct_run_ids_both_written(tmp_path):
    out = tmp_path / "run"
    tel = AgentTelemetry(out)
    r1 = compute_run_id(0, "cand", "task-1", 1)
    r2 = compute_run_id(0, "cand", "task-2", 1)
    tel._append_run(_record(r1, task_id="task-1"))  # noqa: SLF001
    tel._append_run(_record(r2, task_id="task-2"))  # noqa: SLF001
    rows = read_run_records(out)
    assert {row["run_id"] for row in rows} == {r1, r2}
