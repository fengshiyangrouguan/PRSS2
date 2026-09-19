"""T-R2.1 — gate-reuse re-stamp: a reused gate run counts once as eval.

Under the 1.6 gate-reuse optimization a gate-PASSING candidate's gate traces are
reused at eval WITHOUT re-solving, so each such task's ONLY physical telemetry
row is the ``phase="gate"`` one written during the gate check. ``fair_comparison``
defaults to eval-phase rows (so the legacy double-solve is not double-counted) and
drops ``phase="gate"`` — silently undercounting ``n_runs`` over a positively-
selected remainder.

``AgentTelemetry.restamp_reused_gate`` re-emits that cached gate row as an
eval-phase clone keyed on the SAME run_id, so both read-side de-dups
(``load_runs`` keep='last') supersede the gate row and the single physical run is
counted exactly once as eval — without duplicating any token/cost in the rollups.

Install-free: imports only ``meta_n`` + stdlib for the writer assertions; the read
side is pandas-gated (the ``analysis`` extra).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from meta_n.core.external_agents.backend import AgentRunResult
from meta_n.core.external_agents.telemetry import AgentTelemetry, compute_run_id
from meta_n.core.external_agents.terminated import TerminatedBy
from meta_n.integrations.benchmark import EvalResult

from .conftest import read_run_records


class _InnerBackend:
    name = "terminus2"
    outer_token_mode = False


class _Solver:
    def __init__(self, *, execution_phase="eval"):
        self.backend = _InnerBackend()
        self.depth = 1
        self.generation = 3
        self.candidate_id = "cand-7"
        self.max_turns = 8
        self.execution_phase = execution_phase


class _Task:
    task_id = "t-gate-reuse"


def _inner_run(tokens=1234):
    return AgentRunResult(
        agent_tokens=tokens,
        agent_prompt_tokens=tokens // 2,
        agent_completion_tokens=tokens - tokens // 2,
        agent_calls=4,
        cost_usd=0.5,
        cost_basis="priced_from_tokens",
        terminated_by=TerminatedBy.COMPLETED,
        attribution_available=False,
    )


def _write_gate_row(tel: AgentTelemetry, solver: _Solver, task: _Task):
    """Write one phase='gate' row exactly as the gate check would, return run_id."""
    prev = solver.execution_phase
    solver.execution_phase = "gate"
    try:
        rec = tel.start_record(task, solver)
    finally:
        solver.execution_phase = prev
    assert rec.phase == "gate"
    tel.finish_record(rec, _inner_run(), EvalResult(success=True, score=1.0, raw_score=1.0))
    return rec.run_id


def test_restamp_emits_eval_clone_with_same_run_id(tmp_path):
    tel = AgentTelemetry(output_dir=str(tmp_path))
    solver, task = _Solver(), _Task()

    gate_run_id = _write_gate_row(tel, solver, task)
    # Sanity: the gate run_id folds the 'gate' phase into the basis.
    assert gate_run_id == compute_run_id(3, "cand-7", "t-gate-reuse", 1, "gate")

    rows = read_run_records(Path(tmp_path))
    assert len(rows) == 1 and rows[0]["phase"] == "gate"
    gate_tokens = rows[0]["total_tokens"]

    # Reuse branch: re-stamp. (The orchestrator has restored execution_phase to
    # 'eval' by now; the lookup is independent of it — it recomputes the gate id.)
    assert tel.restamp_reused_gate(task, solver) is True

    rows = read_run_records(Path(tmp_path))
    # A SECOND physical line — the eval clone — sharing the gate run_id.
    assert len(rows) == 2
    assert {r["run_id"] for r in rows} == {gate_run_id}
    assert sorted(r["phase"] for r in rows) == ["eval", "gate"]
    # The eval clone preserves the gate run's token/cost/score (it IS that run).
    eval_row = next(r for r in rows if r["phase"] == "eval")
    assert eval_row["total_tokens"] == gate_tokens
    assert eval_row["score"] == 1.0
    assert eval_row["cost_usd"] == 0.5


def test_restamp_is_noop_without_a_cached_gate_row(tmp_path):
    # A non-gate ``precomputed`` source (e.g. consolidation reuse) or a cross-
    # process resume has no cached gate row -> returns False, writes nothing.
    tel = AgentTelemetry(output_dir=str(tmp_path))
    assert tel.restamp_reused_gate(_Task(), _Solver()) is False
    assert read_run_records(Path(tmp_path)) == []


def test_restamp_pops_cache_so_second_call_is_noop(tmp_path):
    tel = AgentTelemetry(output_dir=str(tmp_path))
    solver, task = _Solver(), _Task()
    _write_gate_row(tel, solver, task)
    assert tel.restamp_reused_gate(task, solver) is True
    # The cache is popped; a stray second call cannot keep appending eval clones.
    assert tel.restamp_reused_gate(task, solver) is False
    rows = read_run_records(Path(tmp_path))
    assert sorted(r["phase"] for r in rows) == ["eval", "gate"]


def test_restamped_run_counts_once_as_eval_on_read(tmp_path):
    pd = pytest.importorskip("pandas")  # analysis extra
    from meta_n.analysis.telemetry import fair_comparison, load_runs

    tel = AgentTelemetry(output_dir=str(tmp_path))
    solver, task = _Solver(), _Task()
    _write_gate_row(tel, solver, task)
    tel.restamp_reused_gate(task, solver)

    # load_runs de-dups keep='last' on the (gate) run_id -> the eval clone
    # supersedes the gate row: ONE surviving row, phase=eval.
    df = load_runs(tmp_path)
    assert len(df) == 1
    assert df.iloc[0]["phase"] == "eval"

    # fair_comparison (eval-only default) now counts the reused run exactly once,
    # instead of dropping it as a gate row (the undercount the fix repairs).
    out = fair_comparison(df)
    assert out.loc["terminus2", "n_runs"] == 1
    assert out.loc["terminus2", "inner_total_tokens"] == 1234

    # Without the fix the row would have stayed phase=gate and been dropped:
    gate_only = pd.DataFrame([{**df.iloc[0].to_dict(), "phase": "gate"}])
    assert fair_comparison(gate_only).empty