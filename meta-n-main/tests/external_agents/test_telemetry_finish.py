"""finish_record — display-only builtin cost + GRADE-axis reconciliation (§7).

Install-free: imports only ``meta_n`` + stdlib, no pandas, no Docker, no
subprocess. Exercises two refinements to ``AgentTelemetry.finish_record``:

1. An outer-basis (builtin) row reports ``run.cost_usd == 0.0`` (the daily cap is
   billed on the OUTER LLMClient ledger). On a PRICED backbone the row is now
   given a DISPLAY-ONLY cost derived from its own outer token counts, so a
   fair-comparison cost column is not silently $0 for the builtin control. An
   unpriced / unknown model leaves the row at $0.

2. Termination is reconciled against the authoritative scorer outcome on the
   GRADE axis: a scorer PASS coerces a non-degraded tag (incl. UNKNOWN) to
   COMPLETED; a scorer FAIL coerces COMPLETED to AGENT_ERROR; genuinely-degraded
   tags (TIMEOUT / ENV_ERROR / …) are left untouched.
"""

from __future__ import annotations

import logging
from pathlib import Path

from meta_n.core.external_agents.backend import AgentRunResult
from meta_n.core.external_agents.telemetry import AgentTelemetry
from meta_n.core.external_agents.terminated import TerminatedBy
from meta_n.integrations.benchmark import EvalResult

from .conftest import read_run_records


# --- minimal solver chain so start_record can resolve coordinates + model ----


class _Config:
    def __init__(self, model: str):
        self.model = model


class _LLMClient:
    def __init__(self, model: str):
        self.config = _Config(model)


class _Backend:
    """Outer-basis (builtin-like) backend exposing an llm_client for pricing."""

    name = "builtin"
    outer_token_mode = True

    def __init__(self, model: str):
        self.llm_client = _LLMClient(model)


class _InnerBackend:
    name = "terminus2"
    outer_token_mode = False


class _Solver:
    def __init__(self, backend):
        self.backend = backend
        self.depth = 1
        self.generation = 0
        self.candidate_id = "c0"
        self.max_turns = 8


class _Task:
    task_id = "t-cost"


def _outer_run(prompt=1000, completion=500, *, terminated=TerminatedBy.COMPLETED):
    """An outer-basis AgentRunResult with $0 native cost (the builtin shape)."""
    return AgentRunResult(
        agent_tokens=prompt + completion,
        agent_prompt_tokens=prompt,
        agent_completion_tokens=completion,
        agent_cached_tokens=0,
        agent_calls=1,
        cost_usd=0.0,
        cost_basis="priced_from_tokens",
        terminated_by=terminated,
        attribution_available=False,
    )


def _evalr(success=True, score=1.0):
    return EvalResult(success=success, score=score, raw_score=score)


# --- display-only builtin cost --------------------------------------------


def test_outer_basis_row_priced_from_own_tokens_on_priced_model(tmp_path):
    # gpt-5.2 is in the PRICING table (input 1.75 / output 14.00 per M).
    tel = AgentTelemetry(output_dir=str(tmp_path))
    rec = tel.start_record(_Task(), _Solver(_Backend("gpt-5.2")))
    assert rec.token_basis == "outer"
    tel.finish_record(rec, _outer_run(1000, 500), _evalr(), lease=None)

    # 1000 prompt @1.75/M + 500 completion @14.00/M = 0.00175 + 0.007 = 0.00875.
    expected = (1000 / 1_000_000) * 1.75 + (500 / 1_000_000) * 14.00
    assert rec.cost_usd == expected
    assert rec.cost_usd > 0.0
    # Persisted to the row, not just the in-memory record.
    rows = read_run_records(Path(tmp_path))
    assert len(rows) == 1
    assert rows[0]["cost_usd"] == expected


def test_outer_basis_row_stays_zero_on_unpriced_model(tmp_path):
    # An unknown model is absent from PRICING -> compute_cost_usd raises KeyError
    # -> the row is left at $0 (never sinks the write).
    tel = AgentTelemetry(output_dir=str(tmp_path))
    rec = tel.start_record(_Task(), _Solver(_Backend("totally-unknown-model-xyz")))
    tel.finish_record(rec, _outer_run(1000, 500), _evalr(), lease=None)
    assert rec.cost_usd == 0.0


def test_outer_basis_row_stays_zero_on_zero_priced_model(tmp_path):
    # The self-hosted $0 backbone is priced at 0.0 in PRICING -> cost stays $0.
    tel = AgentTelemetry(output_dir=str(tmp_path))
    rec = tel.start_record(_Task(), _Solver(_Backend("google/gemma-4-31b-qat")))
    tel.finish_record(rec, _outer_run(1000, 500), _evalr(), lease=None)
    assert rec.cost_usd == 0.0


def test_priced_model_is_not_serialized(tmp_path):
    # The resolved model rides as a NON-FIELD attr, never entering the JSON schema.
    tel = AgentTelemetry(output_dir=str(tmp_path))
    rec = tel.start_record(_Task(), _Solver(_Backend("gpt-5.2")))
    tel.finish_record(rec, _outer_run(), _evalr(), lease=None)
    rows = read_run_records(Path(tmp_path))
    assert "_priced_model" not in rows[0]
    assert "model" not in rows[0]


def test_inner_basis_native_cost_is_not_overwritten(tmp_path):
    # An inner-basis backend carrying a real native cost is untouched (the display
    # repricing only fires for an outer-basis row with cost_usd == 0.0).
    tel = AgentTelemetry(output_dir=str(tmp_path))
    rec = tel.start_record(_Task(), _Solver(_InnerBackend()))
    run = _outer_run()
    run.cost_usd = 0.42
    run.cost_basis = "native_usd"
    tel.finish_record(rec, run, _evalr(), lease=None)
    assert rec.cost_usd == 0.42


# --- GRADE-axis reconciliation --------------------------------------------


def test_scorer_pass_coerces_unknown_tag_to_completed(tmp_path):
    # A clean ran-but-wrong TB run is tagged UNKNOWN; a scorer PASS must coerce it
    # to COMPLETED (no internally-contradictory unknown+success row).
    tel = AgentTelemetry(output_dir=str(tmp_path))
    rec = tel.start_record(_Task(), _Solver(_InnerBackend()))
    run = _outer_run(terminated=TerminatedBy.UNKNOWN)
    tel.finish_record(rec, run, _evalr(success=True), lease=None)
    assert rec.terminated_by == TerminatedBy.COMPLETED.value
    assert rec.failure_mode is None


def test_scorer_fail_coerces_completed_to_agent_error(tmp_path):
    tel = AgentTelemetry(output_dir=str(tmp_path))
    rec = tel.start_record(_Task(), _Solver(_InnerBackend()))
    run = _outer_run(terminated=TerminatedBy.COMPLETED)
    tel.finish_record(rec, run, _evalr(success=False, score=0.0), lease=None)
    assert rec.terminated_by == TerminatedBy.AGENT_ERROR.value
    assert rec.failure_mode == "scored_fail"


def test_degraded_timeout_left_untouched_by_scorer(tmp_path):
    # A genuinely-degraded tag (TIMEOUT) describes HOW the run ended, not the
    # grade, so it is NOT reconciled even on a scorer fail.
    tel = AgentTelemetry(output_dir=str(tmp_path))
    rec = tel.start_record(_Task(), _Solver(_InnerBackend()))
    run = _outer_run(terminated=TerminatedBy.TIMEOUT)
    tel.finish_record(rec, run, _evalr(success=False, score=0.0), lease=None)
    assert rec.terminated_by == TerminatedBy.TIMEOUT.value


# --- F3/F4: command_count + lost-stream warning + pointer relpaths ----------


def _captured_run(commands, *, artifacts_path="", attribution=True):
    """An inner-basis run carrying a captured command stream (F3/F4)."""
    return AgentRunResult(
        agent_tokens=100,
        agent_prompt_tokens=60,
        agent_completion_tokens=40,
        agent_calls=1,
        cost_usd=0.0,
        cost_basis="priced_from_tokens",
        terminated_by=TerminatedBy.COMPLETED,
        command_history=list(commands),
        attribution_available=attribution,
        artifacts_path=artifacts_path,
    )


def test_command_count_equals_history_len(tmp_path):
    # command_count is the measured-zero-vs-lost discriminator: it equals the
    # captured command_history length and is serialized onto the row.
    tel = AgentTelemetry(output_dir=str(tmp_path))
    rec = tel.start_record(_Task(), _Solver(_InnerBackend()))
    rec.utilities_available = ["greet"]
    run = _captured_run(["ls", "python3 helpers/greet.py x", "echo done"])
    tel.finish_record(rec, run, _evalr(), lease=None)
    assert rec.command_count == 3
    assert rec.attribution_available is True
    rows = read_run_records(Path(tmp_path))
    assert rows[0]["command_count"] == 3
    assert rows[0]["attribution_available"] is True
    # The greet helper was called once -> measured, not lost.
    assert rows[0]["utilities_called"] == ["greet"]


def test_empty_history_is_measured_zero_not_lost(tmp_path):
    # No utilities staged + a captured (empty) stream: command_count 0 with
    # attribution_available True is a MEASURED-none, provably distinct from a lost
    # stream (which would carry no attribution at all).
    tel = AgentTelemetry(output_dir=str(tmp_path))
    rec = tel.start_record(_Task(), _Solver(_InnerBackend()))
    run = _captured_run([])
    tel.finish_record(rec, run, _evalr(), lease=None)
    assert rec.command_count == 0
    assert rec.attribution_available is True
    assert rec.utilities_called == []


def test_warns_when_utilities_staged_but_history_empty(tmp_path, caplog):
    # Utilities staged + capture POSSIBLE (attribution_available True) but NO
    # command stream came back -> attribution was EXPECTED but UNMEASURED.
    tel = AgentTelemetry(output_dir=str(tmp_path))
    rec = tel.start_record(_Task(), _Solver(_InnerBackend()))
    rec.utilities_available = ["greet", "count_lines"]
    run = _captured_run([], attribution=True)
    with caplog.at_level(logging.WARNING):
        tel.finish_record(rec, run, _evalr(), lease=None)
    assert any(
        "attribution expected but unmeasured" in r.message for r in caplog.records
    )
    assert rec.command_count == 0


def test_no_warning_when_utilities_staged_and_stream_present(tmp_path, caplog):
    tel = AgentTelemetry(output_dir=str(tmp_path))
    rec = tel.start_record(_Task(), _Solver(_InnerBackend()))
    rec.utilities_available = ["greet"]
    run = _captured_run(["python3 helpers/greet.py x"])
    with caplog.at_level(logging.WARNING):
        tel.finish_record(rec, run, _evalr(), lease=None)
    assert not any(
        "attribution expected but unmeasured" in r.message for r in caplog.records
    )


def test_agent_logs_ptr_relpath_assigned(tmp_path):
    # An artifacts_path UNDER output_dir is stored as a relpath pointer (not a blob,
    # not an absolute path). transcript_ptr mirrors it when otherwise unset.
    tel = AgentTelemetry(output_dir=str(tmp_path))
    rec = tel.start_record(_Task(), _Solver(_InnerBackend()))
    logs = tmp_path / "archive" / "c0" / "agent_logs"
    logs.mkdir(parents=True, exist_ok=True)
    run = _captured_run(["ls"], artifacts_path=str(logs))
    tel.finish_record(rec, run, _evalr(), lease=None)
    assert rec.agent_logs_ptr == "archive/c0/agent_logs"
    assert rec.transcript_ptr == "archive/c0/agent_logs"


def test_agent_logs_ptr_none_when_outside_output_dir(tmp_path):
    # An artifacts_path OUTSIDE output_dir is not relativizable -> pointer is
    # honestly None rather than a misleading absolute path.
    tel = AgentTelemetry(output_dir=str(tmp_path / "run"))
    (tmp_path / "run").mkdir(parents=True, exist_ok=True)
    rec = tel.start_record(_Task(), _Solver(_InnerBackend()))
    run = _captured_run(["ls"], artifacts_path="/some/other/place")
    tel.finish_record(rec, run, _evalr(), lease=None)
    assert rec.agent_logs_ptr is None


# --- H3: command_count as the measured-zero-vs-lost discriminator ------------


def test_command_count_measured_zero_vs_lost(tmp_path):
    # attribution_available True + a captured stream (['ls']) + utilities staged ->
    # command_count == 1: a REAL stream was captured, so a [] utilities_called
    # would be a MEASURED-none.
    tel = AgentTelemetry(output_dir=str(tmp_path / "a"))
    rec = tel.start_record(_Task(), _Solver(_InnerBackend()))
    rec.utilities_available = ["greet"]
    tel.finish_record(rec, _captured_run(["ls"]), _evalr(), lease=None)
    assert rec.command_count == 1
    assert rec.attribution_available is True
    assert rec.utilities_called == []  # measured-none (a real stream, none called)

    # attribution_available True + an EMPTY captured stream + utilities staged ->
    # command_count == 0: the LOST/ambiguous side (no stream came back though
    # capture was wired), NOT a measured zero.
    tel2 = AgentTelemetry(output_dir=str(tmp_path / "b"))
    rec2 = tel2.start_record(_Task(), _Solver(_InnerBackend()))
    rec2.utilities_available = ["greet"]
    tel2.finish_record(rec2, _captured_run([]), _evalr(), lease=None)
    assert rec2.command_count == 0
    assert rec2.attribution_available is True


# --- H12: a clean re-execution supersedes a prior degraded row ---------------


def test_degraded_then_clean_supersedes(tmp_path):
    # A degraded row (TIMEOUT) is written; a later NON-degraded re-execution of the
    # SAME run_id (a resume that re-ran a transiently-failed task) is APPENDED as a
    # second physical row instead of de-dupping, so the clean outcome is the LAST
    # row on disk (the keep='last' reader surfaces it).
    tel = AgentTelemetry(output_dir=str(tmp_path))
    rec_deg = tel.start_record(_Task(), _Solver(_InnerBackend()))
    tel.finish_timeout(_Task(), rec_deg, depth=1)  # degraded TIMEOUT row

    rec_clean = tel.start_record(_Task(), _Solver(_InnerBackend()))
    assert rec_clean.run_id == rec_deg.run_id  # same coordinates -> same run_id
    tel.finish_record(
        rec_clean, _captured_run(["ls"]), _evalr(success=True), lease=None
    )

    rows = read_run_records(Path(tmp_path))
    # BOTH physical rows are present; the clean COMPLETED row is LAST.
    assert len(rows) == 2
    assert rows[0]["terminated_by"] == TerminatedBy.TIMEOUT.value
    assert rows[-1]["terminated_by"] == TerminatedBy.COMPLETED.value
    assert rows[-1]["success"] is True


def test_clean_then_degraded_does_not_supersede(tmp_path):
    # The reverse must NOT supersede: a degraded re-execution of a run_id whose
    # on-disk row is CLEAN is dropped (a transient failure never overwrites a known
    # good outcome).
    tel = AgentTelemetry(output_dir=str(tmp_path))
    rec_clean = tel.start_record(_Task(), _Solver(_InnerBackend()))
    tel.finish_record(
        rec_clean, _captured_run(["ls"]), _evalr(success=True), lease=None
    )
    rec_deg = tel.start_record(_Task(), _Solver(_InnerBackend()))
    tel.finish_timeout(_Task(), rec_deg, depth=1)

    rows = read_run_records(Path(tmp_path))
    assert len(rows) == 1
    assert rows[0]["terminated_by"] == TerminatedBy.COMPLETED.value


def test_degraded_then_degraded_still_dedups(tmp_path):
    # A degraded row followed by another DEGRADED row for the same run_id still
    # de-dups (only a NON-degraded row supersedes).
    tel = AgentTelemetry(output_dir=str(tmp_path))
    rec1 = tel.start_record(_Task(), _Solver(_InnerBackend()))
    tel.finish_timeout(_Task(), rec1, depth=1)
    rec2 = tel.start_record(_Task(), _Solver(_InnerBackend()))
    tel.finish_error(_Task(), rec2, RuntimeError("boom"), depth=1)
    rows = read_run_records(Path(tmp_path))
    assert len(rows) == 1
    assert rows[0]["terminated_by"] == TerminatedBy.TIMEOUT.value
