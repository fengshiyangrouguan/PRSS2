"""Regression tests for the shared JSONL line-scan (cluster C10b, spec B4, F194).

``iter_jsonl_objects`` (meta_n/core/external_agents/telemetry.py) is the
canonical line-scan for ``agent_runs.jsonl`` readers: blank lines and bad-JSON
lines are skipped, OSError propagates so each caller keeps its own error
policy. Envelope unwrapping stays caller-specific — the parity tests below pin
each reader's unwrap semantics against hardcoded expectations so the migration
to the shared generator cannot have changed them.

All tests are LLM-free / offline (no SDK, no Docker, no network).
"""

import json
from unittest.mock import MagicMock

import pytest

from meta_n.core.evolutionary_orchestrator import (
    EvolutionaryConfig,
    EvolutionaryOrchestrator,
)
from meta_n.core.external_agents.telemetry import (
    AgentTelemetry,
    iter_jsonl_objects,
)

# --------------------------------------------------------------------------- #
# iter_jsonl_objects — unit tests
# --------------------------------------------------------------------------- #


def test_iter_jsonl_skips_blank_and_bad_lines(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text('{"a": 1}\n\n   \nnot json {\n{"b": 2}\n')
    assert list(iter_jsonl_objects(p)) == [{"a": 1}, {"b": 2}]


def test_iter_jsonl_empty_file_yields_nothing(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text("")
    assert list(iter_jsonl_objects(p)) == []


def test_iter_jsonl_propagates_oserror_on_missing_file(tmp_path):
    """OSError is the caller's to handle (partial-index vs empty-result)."""
    with pytest.raises(OSError):
        list(iter_jsonl_objects(tmp_path / "absent.jsonl"))


def test_iter_jsonl_accepts_str_path(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text('{"a": 1}\n')
    assert list(iter_jsonl_objects(str(p))) == [{"a": 1}]


# --------------------------------------------------------------------------- #
# Parity — AgentTelemetry._load_run_id_index (writer-side seen/degraded index)
# --------------------------------------------------------------------------- #


def _write_runs_jsonl(root, lines):
    tdir = root / "telemetry"
    tdir.mkdir(parents=True, exist_ok=True)
    (tdir / "agent_runs.jsonl").write_text(
        "\n".join(json.dumps(x) if not isinstance(x, str) else x for x in lines)
        + "\n"
    )


def test_load_run_id_index_parity(tmp_path):
    """Pins the pre-consolidation semantics on a mixed fixture: envelope rows,
    top-level fallback (seen only — the degraded marker deliberately reads the
    SAME record, no cross-envelope fallback), bare records, newest-wins on a
    duplicate run_id, bad + blank lines skipped."""
    _write_runs_jsonl(tmp_path, [
        # envelope, clean
        {"extra": {"record": {"run_id": "rid1", "terminated_by": "agent_done"}}},
        # envelope WITHOUT run_id but top-level run_id: counted as seen via the
        # forward-compat fallback; NOT eligible for the degraded marker.
        {"run_id": "rid2", "extra": {"record": {"terminated_by": "timeout"}}},
        # bare record, degraded ...
        {"run_id": "rid3", "terminated_by": "timeout"},
        # ... superseded by a newer clean row (newest-wins).
        {"run_id": "rid3", "terminated_by": "agent_done"},
        # envelope, degraded
        {"extra": {"record": {"run_id": "rid4", "terminated_by": "env_error"}}},
        "not json {",
        "",
    ])
    tel = AgentTelemetry(tmp_path)
    assert tel._seen_run_ids == {"rid1", "rid2", "rid3", "rid4"}
    assert tel._degraded_run_ids == {"rid4"}


def test_load_run_id_index_missing_file(tmp_path):
    tel = AgentTelemetry(tmp_path)  # no agent_runs.jsonl on disk
    assert tel._seen_run_ids == set()
    assert tel._degraded_run_ids == set()


# --------------------------------------------------------------------------- #
# Parity — EvolutionaryOrchestrator._read_agent_run_rows (permissive reader)
# --------------------------------------------------------------------------- #


def _make_orch(output_dir):
    return EvolutionaryOrchestrator(
        llm_client=MagicMock(),
        executor=MagicMock(),
        omega=MagicMock(),
        config=EvolutionaryConfig(output_dir=str(output_dir)),
    )


def test_read_agent_run_rows_parity(tmp_path):
    """Pins the permissive unwrap: envelope OR bare row, newest-wins de-dup on
    run_id, unkeyed records passed through after the keyed ones."""
    _write_runs_jsonl(tmp_path, [
        {"extra": {"record": {"run_id": "r1", "candidate_id": "c1", "v": 1}}},
        # newer row for the same run_id supersedes the first
        {"extra": {"record": {"run_id": "r1", "candidate_id": "c1", "v": 2}}},
        # bare (non-envelope) row with a top-level run_id
        {"run_id": "r2", "candidate_id": "c2", "v": 3},
        # record without any run_id — kept as unkeyed passthrough
        {"extra": {"record": {"candidate_id": "c9", "v": 4}}},
        "bad json line",
        "",
    ])
    orch = _make_orch(tmp_path)
    rows = orch._read_agent_run_rows()
    assert rows == [
        {"run_id": "r1", "candidate_id": "c1", "v": 2},
        {"run_id": "r2", "candidate_id": "c2", "v": 3},
        {"candidate_id": "c9", "v": 4},
    ]


def test_read_agent_run_rows_missing_file_returns_empty(tmp_path):
    orch = _make_orch(tmp_path)
    assert orch._read_agent_run_rows() == []


def test_read_agent_run_rows_cache_invalidates_on_append(tmp_path):
    """(size, mtime_ns) cache: an appended row is picked up on re-read."""
    _write_runs_jsonl(tmp_path, [{"run_id": "r1", "v": 1}])
    orch = _make_orch(tmp_path)
    assert len(orch._read_agent_run_rows()) == 1
    with open(tmp_path / "telemetry" / "agent_runs.jsonl", "a") as fh:
        fh.write(json.dumps({"run_id": "r2", "v": 2}) + "\n")
    assert len(orch._read_agent_run_rows()) == 2
