"""§6b telemetry-schema regressions (F157 + F081/F167, SCHEMA_VERSION 2).

Two changes land under the single SCHEMA_VERSION 1 -> 2 bump:

* F157 — ``AgentRunRecord.benchmark`` discriminator: the adapter's ``name`` is
  stamped onto the solver and persisted on every row so same-named agents (the
  CO-Bench and TB ``openhands`` backends; likewise the two ``builtin`` backends)
  stay separable, and ``fair_comparison`` groups by ``[agent, benchmark]`` ONLY
  when a frame carries >1 distinct non-empty benchmark. Legacy rows read as
  ``""`` and are never split on — every single-benchmark table is byte-identical.
* F081/F167 — the never-produced ``StepRecord``/``log_step``/``agent_steps.jsonl``
  channel is excised end-to-end (write side, exports, read side); per-run step
  counts remain first-class via ``AgentRunRecord.steps``.

Writer-side tests are stdlib-only; the ``fair_comparison`` read-side tests gate
on pandas per-test (the ``analysis`` extra), matching test_schema_basis.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import meta_n.core.external_agents as ea
import meta_n.core.external_agents.telemetry as ea_telemetry
from meta_n.core.external_agents.telemetry import (
    SCHEMA_VERSION,
    AgentRunRecord,
    AgentTelemetry,
)


# ---------------------------------------------------------------------------
# Local fakes (attribute bags read via getattr in start_record / solver init)
# ---------------------------------------------------------------------------


class _Task:
    task_id = "task_a"


class _Backend:
    name = "openhands"
    outer_token_mode = False


class _Solver:
    """Minimal solver double exposing the coordinates start_record reads."""

    def __init__(self, benchmark=None, phase="eval"):
        self.backend = _Backend()
        self.depth = 1
        self.generation = 0
        self.candidate_id = "cand"
        self.max_turns = 4
        self.execution_phase = phase
        if benchmark is not None:
            self.benchmark = benchmark


def _read_rows(output_dir: Path) -> list[dict]:
    """Un-nest the LLMIOLogger ``extra.record`` envelope from agent_runs.jsonl."""
    path = Path(output_dir) / "telemetry" / "agent_runs.jsonl"
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        rows.append((obj.get("extra") or {}).get("record") or obj)
    return rows


# ---------------------------------------------------------------------------
# F157 — writer side (stdlib-only)
# ---------------------------------------------------------------------------


def test_agent_run_record_benchmark_defaults_empty():
    rec = AgentRunRecord(run_id="x")
    assert rec.benchmark == ""
    row = rec.to_row()
    assert "benchmark" in row
    assert row["benchmark"] == ""


def test_start_record_stamps_benchmark_from_solver(tmp_path):
    tel = AgentTelemetry(tmp_path)
    rec = tel.start_record(_Task(), _Solver(benchmark="co_bench"))
    assert rec.benchmark == "co_bench"
    assert rec.to_row()["benchmark"] == "co_bench"


def test_start_record_benchmark_absent_solver_attr_defaults_empty(tmp_path):
    tel = AgentTelemetry(tmp_path)
    rec = tel.start_record(_Task(), _Solver(benchmark=None))
    assert rec.benchmark == ""


def test_solver_init_stamps_benchmark_from_adapter_name(tmp_path):
    from meta_n.core.external_agents.solver import ExternalAgentSolver

    class _Adapter:
        @property
        def name(self):
            return "terminal_bench"

    class _BrokenAdapter:
        @property
        def name(self):
            raise RuntimeError("broken test double")

    def _build(adapter):
        return ExternalAgentSolver(
            backend=_Backend(),
            env_provider=object(),
            scorer=object(),
            injected_codes=[],
            depth=1,
            run_guard=object(),
            telemetry=AgentTelemetry(tmp_path),
            cost_guard=None,
            adapter=adapter,
        )

    assert _build(_Adapter()).benchmark == "terminal_bench"
    # A raising ``name`` property must never sink solver construction.
    assert _build(_BrokenAdapter()).benchmark == ""


def test_restamp_reused_gate_clone_carries_benchmark(tmp_path):
    tel = AgentTelemetry(tmp_path)
    solver = _Solver(benchmark="co_bench", phase="gate")
    task = _Task()
    rec = tel.start_record(task, solver)
    assert rec.phase == "gate"
    tel._append_run(rec)

    assert tel.restamp_reused_gate(task, solver) is True
    rows = _read_rows(tmp_path)
    assert len(rows) == 2
    gate_row, eval_clone = rows
    assert gate_row["phase"] == "gate"
    assert eval_clone["phase"] == "eval"
    # The clone is the serialized gate row re-emitted, so it re-carries the
    # benchmark coordinate automatically.
    assert gate_row["benchmark"] == "co_bench"
    assert eval_clone["benchmark"] == "co_bench"
    assert eval_clone["run_id"] == gate_row["run_id"]


# ---------------------------------------------------------------------------
# F081/F167 — StepRecord/log_step channel excised (stdlib-only)
# ---------------------------------------------------------------------------


def test_steprecord_not_exported():
    assert not hasattr(ea, "StepRecord")
    assert "StepRecord" not in ea.__all__
    assert not hasattr(ea_telemetry, "StepRecord")
    assert "StepRecord" not in ea_telemetry.__all__


def test_log_step_removed(tmp_path):
    assert not hasattr(AgentTelemetry, "log_step")
    assert not hasattr(AgentTelemetry, "STEPS_FILE")
    tel = AgentTelemetry(tmp_path)
    assert not hasattr(tel, "_steps_log")


def test_load_steps_removed():
    pytest.importorskip("pandas")  # importing the analysis module needs pandas
    import meta_n.analysis.telemetry as analysis_telemetry

    assert not hasattr(analysis_telemetry, "load_steps")
    assert "load_steps" not in analysis_telemetry.__all__


def test_schema_json_v2_runs_only(tmp_path):
    assert SCHEMA_VERSION == 2
    tel = AgentTelemetry(tmp_path)
    schema = json.loads(
        (tel.telemetry_dir / "schema.json").read_text(encoding="utf-8")
    )
    assert schema["schema_version"] == 2
    assert schema["records"] == {
        "agent_runs.jsonl": "one AgentRunRecord per execute()"
    }
    assert "benchmark" in schema  # the F157 field doc
    # The steps channel is never created anymore (not even lazily).
    assert not (tel.telemetry_dir / "agent_steps.jsonl").exists()


def test_v1_schema_json_refreshed_to_v2_on_resume(tmp_path):
    # Write-once PER VERSION: a resumed pre-v2 run dir holds a stale v1 header
    # (still advertising the excised steps channel); the next AgentTelemetry
    # construction rewrites it to the current schema.
    teldir = tmp_path / "telemetry"
    teldir.mkdir(parents=True)
    (teldir / "schema.json").write_text(
        json.dumps({
            "schema_version": 1,
            "records": {
                "agent_runs.jsonl": "one AgentRunRecord per execute()",
                "agent_steps.jsonl": "one StepRecord per agent step",
            },
        }),
        encoding="utf-8",
    )
    AgentTelemetry(tmp_path)
    schema = json.loads((teldir / "schema.json").read_text(encoding="utf-8"))
    assert schema["schema_version"] == SCHEMA_VERSION == 2
    assert schema["records"] == {
        "agent_runs.jsonl": "one AgentRunRecord per execute()"
    }
    assert "benchmark" in schema  # the refreshed header carries the F157 doc


def test_v2_schema_json_not_rewritten(tmp_path):
    # An already-current file is left byte-identical (the custom marker
    # content proves no rewrite happened; mtime_ns unchanged too).
    teldir = tmp_path / "telemetry"
    teldir.mkdir(parents=True)
    path = teldir / "schema.json"
    path.write_text(
        json.dumps({"schema_version": 2, "marker": True}), encoding="utf-8"
    )
    before_bytes = path.read_bytes()
    before_mtime = path.stat().st_mtime_ns
    AgentTelemetry(tmp_path)
    assert path.read_bytes() == before_bytes
    assert path.stat().st_mtime_ns == before_mtime


def test_corrupt_schema_json_left_alone(tmp_path):
    # Silent-on-corrupt: an unreadable/invalid file is neither rewritten nor
    # allowed to sink construction (the no-crash contract).
    teldir = tmp_path / "telemetry"
    teldir.mkdir(parents=True)
    path = teldir / "schema.json"
    path.write_text("{ not json", encoding="utf-8")
    AgentTelemetry(tmp_path)
    assert path.read_text(encoding="utf-8") == "{ not json"


def test_run_rows_stamped_schema_version_2(tmp_path):
    tel = AgentTelemetry(tmp_path)
    rec = tel.start_record(_Task(), _Solver(benchmark="co_bench"))
    tel.finish_timeout(_Task(), rec, depth=1)
    (row,) = _read_rows(tmp_path)
    assert row["schema_version"] == 2
    assert row["benchmark"] == "co_bench"


# ---------------------------------------------------------------------------
# F157 — read side (pandas-gated, same convention as test_schema_basis.py)
# ---------------------------------------------------------------------------


def _fc_row(agent, benchmark=None, **kw):
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
    if benchmark is not None:
        base["benchmark"] = benchmark
    base.update(kw)
    return base


def test_fair_comparison_splits_on_multi_benchmark():
    pd = pytest.importorskip("pandas")
    from meta_n.analysis.telemetry import fair_comparison

    df = pd.DataFrame(
        [
            # CO-Bench OpenHandsBackend row (native-USD basis).
            _fc_row("openhands", benchmark="co_bench", cost_basis="native_usd",
                    total_tokens=1000, inner_tokens=1000, inner_calls=5,
                    cost_usd=0.10, score=1.0, success=True),
            # TB OpenHandsTBBackend row: SAME agent name, different benchmark.
            _fc_row("openhands", benchmark="terminal_bench",
                    total_tokens=2000, inner_tokens=2000, inner_calls=8,
                    cost_usd=0.05, score=0.0),
        ]
    )
    out = fair_comparison(df)
    assert isinstance(out.index, pd.MultiIndex)
    assert list(out.index.names) == ["agent", "benchmark"]
    assert len(out) == 2
    # Inner axes are NOT blended across the two benchmarks.
    assert out.loc[("openhands", "co_bench"), "inner_total_tokens"] == 1000
    assert out.loc[("openhands", "terminal_bench"), "inner_total_tokens"] == 2000
    assert out.loc[("openhands", "co_bench"), "inner_calls"] == 5
    assert out.loc[("openhands", "terminal_bench"), "inner_calls"] == 8
    assert out.loc[("openhands", "co_bench"), "mean_score"] == pytest.approx(1.0)
    assert out.loc[("openhands", "terminal_bench"), "mean_score"] == pytest.approx(0.0)
    # Per-row sentinel/notes still resolve the agent LEVEL under the MultiIndex.
    assert out.loc[("openhands", "co_bench"), "agent_calls"] == "unmeasured"
    assert out.loc[("openhands", "terminal_bench"), "agent_calls"] == "unmeasured"
    assert (out["notes"] == "").all()


def test_fair_comparison_single_benchmark_index_byte_identical():
    """Column-drop equivalence: with ONE non-empty benchmark the table equals
    (bit-for-bit, plain 'agent' index) the one computed from the same frame
    with no ``benchmark`` column at all — i.e. the column is inert unless >1
    distinct non-empty benchmark is present. This compares two NEW-code paths;
    the frozen-literal pin against the legacy (pre-F157) table is carried by
    tests/external_agents/test_schema_basis.py.
    """
    pd = pytest.importorskip("pandas")
    from meta_n.analysis.telemetry import fair_comparison

    df = pd.DataFrame(
        [
            _fc_row("openhands", benchmark="co_bench", cost_basis="native_usd",
                    total_tokens=1000, inner_tokens=1000, inner_calls=5,
                    score=1.0, success=True),
            _fc_row("terminus2", benchmark="co_bench",
                    total_tokens=2000, inner_tokens=2000, inner_calls=8),
        ]
    )
    pd.testing.assert_frame_equal(
        fair_comparison(df), fair_comparison(df.drop(columns=["benchmark"]))
    )


def test_fair_comparison_legacy_empty_benchmark_never_splits():
    pd = pytest.importorskip("pandas")
    from meta_n.analysis.telemetry import fair_comparison

    df = pd.DataFrame(
        [
            # Legacy (pre-v2) row: benchmark read as "" (unknown).
            _fc_row("openhands", benchmark="", inner_calls=5, score=1.0,
                    success=True),
            _fc_row("openhands", benchmark="co_bench", inner_calls=8),
        ]
    )
    out = fair_comparison(df)
    # ""+one non-empty value == a single knowable benchmark: never split.
    assert not isinstance(out.index, pd.MultiIndex)
    assert out.index.name == "agent"
    assert out.loc["openhands", "n_runs"] == 2


def test_fair_comparison_builtin_note_under_multiindex():
    pd = pytest.importorskip("pandas")
    from meta_n.analysis.telemetry import fair_comparison

    df = pd.DataFrame(
        [
            _fc_row("builtin", benchmark="co_bench", token_basis="outer",
                    total_tokens=500),
            _fc_row("builtin", benchmark="terminal_bench", token_basis="outer",
                    total_tokens=700),
            _fc_row("terminus2", benchmark="terminal_bench",
                    total_tokens=2000, inner_tokens=2000),
        ]
    )
    out = fair_comparison(df)
    assert isinstance(out.index, pd.MultiIndex)
    # The outer-basis note lands on BOTH builtin (agent, benchmark) rows and
    # only on those rows.
    assert "NOT comparable" in out.loc[("builtin", "co_bench"), "notes"]
    assert "NOT comparable" in out.loc[("builtin", "terminal_bench"), "notes"]
    assert out.loc[("terminus2", "terminal_bench"), "notes"] == ""
    # Basis masking is preserved per (agent, benchmark) row.
    assert out.loc[("builtin", "co_bench"), "outer_total_tokens"] == 500
    assert out.loc[("builtin", "co_bench"), "inner_total_tokens"] == 0
