"""S0.6 spine transcript de-reap — snapshot lease agent_logs before rmtree.

Install-free: imports only ``meta_n`` + stdlib, no Docker/SDK. Verifies the
NON-behavioral de-reap that fixes the FEAL null-transcript finding:

* a stubbed lease ``agent_logs`` dir is copied into
  ``<output_dir>/archive/<cand>/agent_logs/<run_id>/`` and ``rec.transcript_ptr``
  RESOLVES after the lease scratch dir is rmtree'd;
* a no-logs backend leaves the pointer honestly ``None``;
* the de-reap never raises (best-effort);
* driven end-to-end through ``ExternalAgentSolver.execute`` (a real lease that is
  rmtree'd on release), the archive copy survives the lease.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from meta_n.core.external_agents.backend import AgentBackend, AgentRunResult
from meta_n.core.external_agents.solver import ExternalAgentSolver
from meta_n.core.external_agents.telemetry import AgentRunRecord, AgentTelemetry
from meta_n.core.external_agents.terminated import TerminatedBy

from .conftest import FakeEnvProvider, FakeScorer, make_task, read_run_records


def _stub_lease_logs(scratch: Path, *, with_file: bool = True) -> Path:
    """Build a stubbed lease ``agent_logs`` dir under a scratch root."""
    logs = scratch / "ext_agent_stub" / "agent_logs"
    logs.mkdir(parents=True, exist_ok=True)
    if with_file:
        (logs / "transcript.jsonl").write_text('{"event": "step", "i": 0}\n')
        (logs / "events").mkdir(exist_ok=True)
        (logs / "events" / "0.json").write_text("{}")
    return logs


# ---------------------------------------------------------------------------
# Direct unit tests of AgentTelemetry.dereap_agent_logs (stubbed lease dir)
# ---------------------------------------------------------------------------


class TestDereapDirect:
    def test_copies_logs_and_pointer_resolves_after_release(self, tmp_path):
        output_dir = tmp_path / "run"
        tel = AgentTelemetry(output_dir, generation=0)
        scratch = tmp_path / "scratch"
        logs = _stub_lease_logs(scratch, with_file=True)

        rec = AgentRunRecord(run_id="rid1234", candidate_id="c0", task_id="t0")
        rel = tel.dereap_agent_logs(rec, logs, "c0", "rid1234")

        # Pointer was rewritten to the archive-relative copy.
        assert rel == "archive/c0/agent_logs/rid1234"
        assert rec.transcript_ptr == rel
        assert rec.agent_logs_ptr == rel

        # Simulate the lease release: the whole scratch tree is rmtree'd.
        shutil.rmtree(scratch, ignore_errors=True)
        assert not logs.exists()

        # The pointer STILL resolves (the durable archive copy survived).
        resolved = output_dir / rec.transcript_ptr
        assert resolved.is_dir()
        assert (resolved / "transcript.jsonl").read_text().startswith("{")
        assert (resolved / "events" / "0.json").exists()

    def test_no_logs_dir_leaves_pointer_honest_none(self, tmp_path):
        output_dir = tmp_path / "run"
        tel = AgentTelemetry(output_dir, generation=0)
        rec = AgentRunRecord(run_id="rid", candidate_id="c0")
        # Source dir does not exist (a backend that captured nothing).
        missing = tmp_path / "scratch" / "nope" / "agent_logs"
        rel = tel.dereap_agent_logs(rec, missing, "c0", "rid")
        assert rel is None
        assert rec.transcript_ptr is None
        assert rec.agent_logs_ptr is None
        assert not (output_dir / "archive").exists()

    def test_empty_logs_dir_leaves_pointer_honest_none(self, tmp_path):
        output_dir = tmp_path / "run"
        tel = AgentTelemetry(output_dir, generation=0)
        empty = _stub_lease_logs(tmp_path / "scratch", with_file=False)
        rec = AgentRunRecord(run_id="rid", candidate_id="c0")
        rel = tel.dereap_agent_logs(rec, empty, "c0", "rid")
        assert rel is None
        assert rec.transcript_ptr is None

    def test_never_raises_on_bad_input(self, tmp_path):
        output_dir = tmp_path / "run"
        tel = AgentTelemetry(output_dir, generation=0)
        rec = AgentRunRecord(run_id="rid")
        # None source, None ids — must not raise, must return None.
        assert tel.dereap_agent_logs(rec, None, None, None) is None
        assert rec.transcript_ptr is None

    def test_output_dir_override_is_honored(self, tmp_path):
        # Telemetry rooted elsewhere; the explicit output_dir override wins.
        tel = AgentTelemetry(tmp_path / "tel_root", generation=0)
        override = tmp_path / "run"
        override.mkdir()
        logs = _stub_lease_logs(tmp_path / "scratch", with_file=True)
        rec = AgentRunRecord(run_id="rid")
        rel = tel.dereap_agent_logs(rec, logs, "c0", "rid", output_dir=override)
        assert rel == "archive/c0/agent_logs/rid"
        assert (override / rel / "transcript.jsonl").exists()
        # Nothing landed under the telemetry root.
        assert not (tmp_path / "tel_root" / "archive").exists()


# ---------------------------------------------------------------------------
# End-to-end through ExternalAgentSolver.execute (a real, rmtree'd lease)
# ---------------------------------------------------------------------------


class _LogWritingBackend(AgentBackend):
    """A backend that writes a transcript into ``ctx.logging_dir`` like OH/T2."""

    name = "logwriter"
    outer_token_mode = False

    async def run(self, ctx, tel, rec) -> AgentRunResult:
        logdir = Path(ctx.logging_dir)
        logdir.mkdir(parents=True, exist_ok=True)
        (logdir / "transcript.jsonl").write_text('{"event": "done"}\n')
        return AgentRunResult(
            transcript="ok",
            reasoning_summary="did it",
            terminated_by=TerminatedBy.COMPLETED,
            attribution_available=False,
        )

    async def collect_metrics(self, env, run) -> AgentRunResult:
        return run


@pytest.mark.asyncio
async def test_execute_dereaps_into_archive(run_guard, telemetry):
    """Driving a real lease: the agent_logs survive into the archive copy."""
    solver = ExternalAgentSolver(
        backend=_LogWritingBackend(),
        env_provider=FakeEnvProvider(),
        scorer=FakeScorer(),
        injected_codes=[],
        depth=1,
        run_guard=run_guard,
        telemetry=telemetry,
        cost_guard=None,
        adapter=object(),
        run_ctx={"generation": 0, "candidate_id": "c0",
                 "output_dir": str(telemetry.output_dir)},
    )
    trace, tokens = await solver.execute(make_task(task_id="t-dereap"))
    assert trace.success is True

    # The lease scratch dir was rmtree'd on release, but the archive copy of the
    # agent_logs survives under output_dir.
    run_id = read_run_records(telemetry.output_dir)[0]["run_id"]
    archive_logs = (
        Path(telemetry.output_dir) / "archive" / "c0" / "agent_logs" / run_id
    )
    assert archive_logs.is_dir()
    assert (archive_logs / "transcript.jsonl").read_text().startswith("{")


@pytest.mark.asyncio
async def test_output_dir_falls_back_to_telemetry(run_guard, telemetry):
    """No output_dir in run_ctx -> the solver falls back to telemetry.output_dir."""
    solver = ExternalAgentSolver(
        backend=_LogWritingBackend(),
        env_provider=FakeEnvProvider(),
        scorer=FakeScorer(),
        injected_codes=[],
        depth=1,
        run_guard=run_guard,
        telemetry=telemetry,
        cost_guard=None,
        adapter=object(),
        run_ctx={"generation": 0, "candidate_id": "c0"},  # NO output_dir
    )
    assert solver.output_dir == telemetry.output_dir
    await solver.execute(make_task(task_id="t-fb"))
    run_id = read_run_records(telemetry.output_dir)[0]["run_id"]
    assert (
        Path(telemetry.output_dir) / "archive" / "c0" / "agent_logs" / run_id
    ).is_dir()
