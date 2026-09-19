"""analysis.telemetry — the ingestion / normalization layer (§7.12).

The post-hoc reporting path that builds the fair-comparison tables the A/B story
depends on. Only ``fair_comparison`` was covered (via hand-built DataFrames); the
LOADERS (``load_runs`` / ``pointer_resolve`` / ``_discover_paths``
/ ``_coerce_numeric``) were untested, so a silent parsing bug (a mis-joined
pointer, a missed telemetry file, a mis-typed numeric) would corrupt every
reported comparison with no test catching it.

These exercise ``meta_n.analysis.telemetry`` which hard-depends on ``pandas`` (the
``analysis`` extra); skipped wholesale when pandas is absent so the Phase-0
install-free gate stays green.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")  # analysis-extra dependency; skip if absent

from meta_n.analysis.telemetry import (  # noqa: E402
    _coerce_numeric,
    _discover_paths,
    fair_comparison,
    load_runs,
    pointer_resolve,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def _runs_dir(tmp_path: Path, rows: list[dict]) -> Path:
    """A single run dir with <run>/telemetry/agent_runs.jsonl."""
    run = tmp_path / "run_x"
    _write_jsonl(run / "telemetry" / "agent_runs.jsonl", rows)
    return run


# --- load_runs -------------------------------------------------------------


def test_load_runs_reads_rows_and_adds_run_dir(tmp_path):
    run = _runs_dir(
        tmp_path,
        [
            {"run_id": "a1", "agent": "builtin", "total_tokens": 100, "score": 0.5},
            {"run_id": "b2", "agent": "terminus2", "total_tokens": 200, "score": 1.0},
        ],
    )
    df = load_runs(run)
    assert len(df) == 2
    assert set(df["agent"]) == {"builtin", "terminus2"}
    # load_runs stamps run_dir + a namespaced global_run_id.
    assert (df["run_dir"] == str(run)).all()
    assert "global_run_id" in df.columns
    assert set(df["global_run_id"]) == {"run_x/a1", "run_x/b2"}


def test_load_runs_dedups_on_run_id_within_run(tmp_path):
    # A re-executed task on resume yields the SAME run_id -> the duplicate row is
    # dropped on read. keep="last" (H12 supersede): when two physical rows share a
    # run_id (a degraded row superseded by a later clean re-execution that the
    # writer appended), the LAST row wins so the clean outcome surfaces.
    run = _runs_dir(
        tmp_path,
        [
            {"run_id": "dup", "agent": "builtin", "score": 0.4},
            {"run_id": "dup", "agent": "builtin", "score": 0.9},  # later supersede
            {"run_id": "uniq", "agent": "builtin", "score": 0.1},
        ],
    )
    df = load_runs(run)
    assert len(df) == 2
    # keep="last": the later 0.9 row survives, the earlier 0.4 row is dropped.
    dup_rows = df[df["run_id"] == "dup"]
    assert len(dup_rows) == 1
    assert float(dup_rows.iloc[0]["score"]) == 0.9


def test_load_runs_missing_returns_empty(tmp_path):
    df = load_runs(tmp_path / "does_not_exist")
    assert df.empty


def test_load_runs_cross_run_concat(tmp_path):
    # A parent dir holding TWO run dirs -> rows from both, each namespaced.
    _write_jsonl(
        tmp_path / "runA" / "telemetry" / "agent_runs.jsonl",
        [{"run_id": "x", "agent": "builtin"}],
    )
    _write_jsonl(
        tmp_path / "runB" / "telemetry" / "agent_runs.jsonl",
        [{"run_id": "x", "agent": "openhands"}],
    )
    df = load_runs(tmp_path)
    # Same within-run run_id 'x' in both, but global_run_id namespaces them, so
    # BOTH survive the cross-run de-dup.
    assert len(df) == 2
    assert set(df["global_run_id"]) == {"runA/x", "runB/x"}


# --- _discover_paths -------------------------------------------------------


def test_discover_paths_finds_single_run(tmp_path):
    run = _runs_dir(tmp_path, [{"run_id": "a1", "agent": "builtin"}])
    pairs = _discover_paths(run, ("agent_runs.jsonl",))
    assert len(pairs) == 1
    found_run_dir, found_path = pairs[0]
    assert found_run_dir == run
    assert found_path == run / "telemetry" / "agent_runs.jsonl"


def test_discover_paths_direct_file(tmp_path):
    run = _runs_dir(tmp_path, [{"run_id": "a1", "agent": "builtin"}])
    direct = run / "telemetry" / "agent_runs.jsonl"
    pairs = _discover_paths(direct, ("agent_runs.jsonl",))
    assert len(pairs) == 1
    assert pairs[0][1] == direct


# --- _coerce_numeric -------------------------------------------------------


def test_coerce_numeric_typed_and_missing():
    df = pd.DataFrame([{"total_tokens": "100"}, {"total_tokens": 50}])
    out = _coerce_numeric(df, "total_tokens")
    assert list(out) == [100.0, 50.0]
    # A missing column degrades to all-zero (never raises).
    missing = _coerce_numeric(df, "no_such_col")
    assert list(missing) == [0.0, 0.0]


def test_coerce_numeric_bad_value_becomes_zero():
    df = pd.DataFrame([{"x": "not-a-number"}, {"x": "3"}])
    out = _coerce_numeric(df, "x")
    assert list(out) == [0.0, 3.0]


# --- pointer_resolve -------------------------------------------------------


def test_pointer_resolve_joins_onto_run_dir(tmp_path):
    run = _runs_dir(
        tmp_path,
        [
            {"run_id": "a1", "agent": "builtin",
             "transcript_ptr": "agent_logs/a1/transcript.txt"},
            {"run_id": "b2", "agent": "builtin", "transcript_ptr": None},
        ],
    )
    df = load_runs(run)
    resolved = pointer_resolve(df, "transcript_ptr")
    # The non-null pointer is joined onto its row's run_dir; the null stays None.
    by_run = dict(zip(df["run_id"], resolved))
    assert by_run["a1"] == str(run / "agent_logs/a1/transcript.txt")
    assert by_run["b2"] is None


def test_pointer_resolve_unknown_column_raises(tmp_path):
    run = _runs_dir(tmp_path, [{"run_id": "a1", "agent": "builtin"}])
    df = load_runs(run)
    with pytest.raises(KeyError):
        pointer_resolve(df, "not_a_pointer_col")


def test_trace_ptr_dropped_from_pointer_columns_and_schema():
    # T-R2.2: ``trace_ptr`` was declared/serialized but NEVER assigned (an
    # always-None column that could not distinguish "no blob" from "never
    # plumbed"). It is dropped from the analysis pointer set AND the writer schema.
    from meta_n.analysis.telemetry import _POINTER_COLUMNS
    from meta_n.core.external_agents.telemetry import AgentRunRecord

    assert "trace_ptr" not in _POINTER_COLUMNS
    assert _POINTER_COLUMNS == ("transcript_ptr", "agent_logs_ptr")
    # The two real pointers remain.
    rec = AgentRunRecord(run_id="r1")
    row = rec.to_row()
    assert "trace_ptr" not in row
    assert "transcript_ptr" in row and "agent_logs_ptr" in row
    assert not hasattr(rec, "trace_ptr")


# --- H6: fair_comparison defaults to eval-phase rows (gate excluded) --------


def _fair_row(agent, phase=None, **kw):
    base = dict(
        agent=agent,
        token_basis="inner",
        cost_basis="priced_from_tokens",
        total_tokens=0,
        inner_tokens=0,
        inner_calls=0,
        cost_usd=0.0,
        steps=0,
        score=0.0,
        success=False,
    )
    if phase is not None:
        base["phase"] = phase
    base.update(kw)
    return base


def test_fair_comparison_excludes_gate_phase():
    df = pd.DataFrame(
        [
            _fair_row("terminus2", phase="eval", total_tokens=100, inner_tokens=100,
                      score=1.0, success=True),
            _fair_row("terminus2", phase="gate", total_tokens=50, inner_tokens=50,
                      score=0.0, success=False),
        ]
    )
    out = fair_comparison(df)
    # Only the eval-phase row is counted by default (gate subset excluded).
    assert out.loc["terminus2", "n_runs"] == 1
    assert out.loc["terminus2", "inner_total_tokens"] == 100
    # Opt the gate rows back in explicitly.
    out_all = fair_comparison(df, phases=("eval", "gate"))
    assert out_all.loc["terminus2", "n_runs"] == 2
    assert out_all.loc["terminus2", "inner_total_tokens"] == 150


def test_fair_comparison_counts_rows_without_phase_field():
    # Back-compat: rows predating the phase field (no column) are all treated as
    # eval and fully counted — the gate default never silently drops them.
    df = pd.DataFrame(
        [
            _fair_row("terminus2", score=1.0, success=True),
            _fair_row("terminus2", score=0.0, success=False),
        ]
    )
    out = fair_comparison(df)
    assert out.loc["terminus2", "n_runs"] == 2


def test_fair_comparison_treats_nan_phase_as_eval():
    # A mixed frame where one row carries a phase and another has NaN (the row was
    # written before the field existed): the NaN row is treated as eval, the gate
    # row is excluded.
    df = pd.DataFrame(
        [
            _fair_row("terminus2", phase="eval", score=1.0, success=True),
            _fair_row("terminus2", score=0.5, success=True),  # no phase key -> NaN
            _fair_row("terminus2", phase="gate", score=0.0, success=False),
        ]
    )
    out = fair_comparison(df)
    assert out.loc["terminus2", "n_runs"] == 2  # eval + NaN(->eval), gate dropped
