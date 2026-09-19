"""Regression tests for the external-agents telemetry / solver audit fixes.

One test (or group) per CONFIRMED finding fixed in this change set:

* id 74 — ``redact()`` missed JSON/quoted-key secret assignments (``"HF_TOKEN":
  "hf_…"`` / ``"AZURE_OPENAI_API_KEY": "…"``) and had no bare ``hf_…`` fallback.
* id 21 — a CROSS-PROCESS resume dropped the clean re-execution of a degraded
  task (H12 supersede was process-local, never rebuilt from disk).
* id 68 — ``restamp_reused_gate`` re-stamped only the r0 gate row, dropping the
  other R-1 ``--gate-repeats`` samples from the eval-default view.
* id 8  — the de-reaped durable ``transcript_ptr``/``agent_logs_ptr`` never
  reached ``agent_runs.jsonl`` (dereap ran AFTER finish_record wrote the row,
  and finish_record clobbered the pointer).
* id 7  — OH/T2 real USD spend was dropped from the daily-cap ledger on the
  post-run scoring-fault degraded path (``cost_guard.record`` ran only AFTER
  ``scorer.score`` on the success path).

Install-free / offline: imports only ``meta_n`` + stdlib. No LM Studio, no
Docker, no network. Each test FAILS on the pre-fix code and PASSES after the fix.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from meta_n.core.external_agents.backend import (
    AgentBackend,
    AgentRunContext,
    AgentRunResult,
)
from meta_n.core.external_agents.budget import CostGuard
from meta_n.core.external_agents.concurrency import DockerRunGuard
from meta_n.core.external_agents.env import AgentEnvProvider, Scorer
from meta_n.core.external_agents.solver import ExternalAgentSolver
from meta_n.core.external_agents.telemetry import (
    _REDACTED,
    AgentTelemetry,
    redact,
)
from meta_n.core.external_agents.terminated import TerminatedBy
from meta_n.integrations.benchmark import EvalResult
from meta_n.utils.cost_tracker import CostTracker

_LONG = "A" * 30  # >= 20 chars to satisfy the bare-token shapes


# ---------------------------------------------------------------------------
# Shared lightweight stubs (no SDK, no Docker, no LLM)
# ---------------------------------------------------------------------------


class _InnerBackend:
    """Minimal inner-basis backend descriptor for direct telemetry tests."""

    name = "terminus2"
    outer_token_mode = False


class _Task:
    task_id = "t-auditfix"


class _Solver:
    def __init__(self, *, execution_phase="eval", repeat_index=0):
        self.backend = _InnerBackend()
        self.depth = 1
        self.generation = 2
        self.candidate_id = "cand-9"
        self.max_turns = 8
        self.execution_phase = execution_phase
        self.repeat_index = repeat_index


def _inner_run(commands=("ls",), *, artifacts_path="", tokens=100):
    return AgentRunResult(
        agent_tokens=tokens,
        agent_prompt_tokens=tokens * 6 // 10,
        agent_completion_tokens=tokens - tokens * 6 // 10,
        agent_calls=1,
        cost_usd=0.0,
        cost_basis="priced_from_tokens",
        terminated_by=TerminatedBy.COMPLETED,
        command_history=list(commands),
        attribution_available=False,
        artifacts_path=artifacts_path,
    )


def _evalr(success=True, score=1.0):
    return EvalResult(success=success, score=score, raw_score=score)


def _read_rows(output_dir: Path) -> list[dict]:
    """Un-nest the LLMIOLogger ``extra.record`` envelope from agent_runs.jsonl."""
    import json

    path = Path(output_dir) / "telemetry" / "agent_runs.jsonl"
    if not path.exists():
        return []
    out: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        rec = (obj.get("extra", {}) or {}).get("record")
        out.append(rec if rec is not None else obj)
    return out


# ---------------------------------------------------------------------------
# Finding 74 — redact() JSON/quoted-key secret gap
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "blob,secret",
    [
        (f'"HF_TOKEN": "hf_{_LONG}"', f"hf_{_LONG}"),
        (f"{{'HUGGINGFACE_TOKEN': 'hf_{_LONG}'}}", f"hf_{_LONG}"),
        (
            '{"AZURE_OPENAI_API_KEY": "abcdef0123456789abcdef0123456789"}',
            "abcdef0123456789abcdef0123456789",
        ),
        (f'"OPENAI_API_KEY":"sk-{_LONG}"', f"sk-{_LONG}"),
    ],
)
def test_redact_masks_json_quoted_key_secrets(blob, secret):
    # Pre-fix: pattern[0]'s separator group required whitespace or ``:``/``=``
    # immediately after the key, so a key followed by a closing quote (the JSON /
    # dict-repr form) never matched and the value was written verbatim.
    out = redact(blob)
    assert _REDACTED in out
    assert secret not in out
    # The key name itself is retained (only the value is masked).
    assert ("HF_TOKEN" in out or "HUGGINGFACE_TOKEN" in out
            or "AZURE_OPENAI_API_KEY" in out or "OPENAI_API_KEY" in out)


def test_redact_masks_bare_hf_token():
    # An ``hf_…`` value echoed WITHOUT its key prefix had no bare-value fallback
    # (sk-/or-/Bearer/gh[pousr]_ do not match it), so it escaped entirely.
    out = redact(f"using hf_{_LONG} now")
    assert _REDACTED in out
    assert f"hf_{_LONG}" not in out


def test_redact_named_assignment_and_idempotency_preserved():
    # The classic KEY=value form still masks, and redact stays idempotent.
    once = redact(f"HF_TOKEN=hf_{_LONG}")
    assert _REDACTED in once and f"hf_{_LONG}" not in once and "HF_TOKEN" in once
    assert redact(once) == once
    # Non-secret text is untouched.
    plain = "the agent ran wc -l file.txt and printed 42 lines"
    assert redact(plain) == plain


# ---------------------------------------------------------------------------
# Finding 21 — cross-process resume clean-supersede
# ---------------------------------------------------------------------------


def test_cross_process_resume_clean_supersedes_degraded(tmp_path):
    out = tmp_path / "run"

    # Process 1: a transient degraded TIMEOUT row is written for the task.
    tel1 = AgentTelemetry(output_dir=str(out))
    rec_deg = tel1.start_record(_Task(), _Solver())
    tel1.finish_timeout(_Task(), rec_deg, depth=1)

    # Process 2 (resume == a FRESH writer rebuilt from disk): the task re-executes
    # and now PASSES. The clean COMPLETED row must be appended (the keep='last'
    # reader then surfaces it), not de-dupped against the prior degraded row.
    tel2 = AgentTelemetry(output_dir=str(out))
    rec_clean = tel2.start_record(_Task(), _Solver())
    assert rec_clean.run_id == rec_deg.run_id  # same coordinates -> same run_id
    tel2.finish_record(rec_clean, _inner_run(), _evalr(success=True), lease=None)

    rows = _read_rows(out)
    # Pre-fix: only the scored-zero degraded row survives (clean re-exec dropped).
    assert len(rows) == 2
    assert rows[0]["terminated_by"] == TerminatedBy.TIMEOUT.value
    assert rows[-1]["terminated_by"] == TerminatedBy.COMPLETED.value
    assert rows[-1]["success"] is True


def test_cross_process_resume_does_not_resurrect_clean_row(tmp_path):
    # The reverse direction must still de-dup across processes: a degraded
    # re-execution of a run_id whose last on-disk row is CLEAN is dropped.
    out = tmp_path / "run"
    tel1 = AgentTelemetry(output_dir=str(out))
    rec_clean = tel1.start_record(_Task(), _Solver())
    tel1.finish_record(rec_clean, _inner_run(), _evalr(success=True), lease=None)

    tel2 = AgentTelemetry(output_dir=str(out))
    rec_deg = tel2.start_record(_Task(), _Solver())
    tel2.finish_timeout(_Task(), rec_deg, depth=1)

    rows = _read_rows(out)
    assert len(rows) == 1
    assert rows[0]["terminated_by"] == TerminatedBy.COMPLETED.value


# ---------------------------------------------------------------------------
# Finding 68 — restamp ALL gate-repeat samples
# ---------------------------------------------------------------------------


def test_restamp_reused_gate_restamps_all_repeat_samples(tmp_path):
    out = tmp_path / "run"
    tel = AgentTelemetry(output_dir=str(out))
    task = _Task()

    # Under --gate-repeats R=3 the gate phase writes R distinct rows per task
    # (run_ids '...:gate', '...:gate:r1', '...:gate:r2').
    n_repeats = 3
    for r in range(n_repeats):
        solver = _Solver(execution_phase="gate", repeat_index=r)
        rec = tel.start_record(task, solver)
        assert rec.phase == "gate"
        tel.finish_record(rec, _inner_run(), _evalr(success=True), lease=None)

    assert sum(1 for row in _read_rows(out) if row["phase"] == "gate") == n_repeats

    # Reuse branch: the gate-passing task is reused at eval without re-solving.
    eval_solver = _Solver(execution_phase="eval")
    assert tel.restamp_reused_gate(task, eval_solver) is True

    rows = _read_rows(out)
    eval_rows = [row for row in rows if row["phase"] == "eval"]
    # Pre-fix: only the r0 gate row was re-stamped -> 1 eval clone; the other R-1
    # physical gate solves (and their cost) were left phase='gate' and dropped by
    # the eval-default fair-comparison view. Post-fix: all R are re-stamped.
    assert len(eval_rows) == n_repeats


def test_restamp_reused_gate_noop_without_cache(tmp_path):
    tel = AgentTelemetry(output_dir=str(tmp_path / "run"))
    assert tel.restamp_reused_gate(_Task(), _Solver()) is False
    assert _read_rows(tmp_path / "run") == []


# ---------------------------------------------------------------------------
# execute()-driven harness for findings 8 + 7
# ---------------------------------------------------------------------------


class _LogWritingBackend(AgentBackend):
    """Inner-basis backend that writes a log file into ctx.logging_dir."""

    name = "terminus2"
    outer_token_mode = False

    def __init__(self, *, tokens: int = 1000):
        self._tokens = tokens

    async def run(self, ctx: AgentRunContext, tel, rec) -> AgentRunResult:
        logs = Path(ctx.logging_dir)
        logs.mkdir(parents=True, exist_ok=True)
        (logs / "events.jsonl").write_text('{"event": "step"}\n', encoding="utf-8")
        return AgentRunResult(
            transcript="ran",
            reasoning_summary="did it",
            agent_tokens=self._tokens,
            agent_prompt_tokens=self._tokens * 6 // 10,
            agent_completion_tokens=self._tokens - self._tokens * 6 // 10,
            agent_calls=2,
            cost_usd=0.0,
            cost_basis="priced_from_tokens",
            terminated_by=TerminatedBy.COMPLETED,
            attribution_available=False,
            # The lease scratch agent_logs dir — OUTSIDE output_dir (so the raw
            # pointer is unresolvable until de-reap copies it into the archive).
            artifacts_path=str(logs),
        )


class _EnvProvider(AgentEnvProvider):
    @asynccontextmanager
    async def provision(self, task, lease):
        root = lease.workdir / "workspace"
        root.mkdir(parents=True, exist_ok=True)

        class _Env:
            workspace_handle = str(root)

        yield _Env()

    async def stage_files(self, env, files):
        return None

    async def extract_solution(self, env, run) -> str:
        return "print('hi')"


class _OkScorer(Scorer):
    async def score(self, task, env, solution, run) -> EvalResult:
        return EvalResult(success=True, score=1.0, raw_score=1.0, feedback="ok")


class _RaisingScorer(Scorer):
    """A post-run scorer that raises AFTER the agent has already spent money."""

    async def score(self, task, env, solution, run) -> EvalResult:
        raise RuntimeError("scorer fault that slipped the inner guards")


def _build_solver(tmp_path, *, backend, scorer, cost_guard, telemetry):
    run_guard = DockerRunGuard(max_docker=1, scratch_root=str(tmp_path / "scratch"))
    return ExternalAgentSolver(
        backend=backend,
        env_provider=_EnvProvider(),
        scorer=scorer,
        injected_codes=[],
        depth=1,
        run_guard=run_guard,
        telemetry=telemetry,
        cost_guard=cost_guard,
        adapter=object(),
        time_limit_s=None,
        run_ctx={"output_dir": str(telemetry.output_dir), "candidate_id": "cand-8"},
    )


# ---------------------------------------------------------------------------
# Finding 8 — de-reaped pointer reaches agent_runs.jsonl
# ---------------------------------------------------------------------------


def test_dereaped_pointer_reaches_jsonl(tmp_path):
    out = tmp_path / "run"
    tel = AgentTelemetry(output_dir=str(out))
    solver = _build_solver(
        tmp_path,
        backend=_LogWritingBackend(),
        scorer=_OkScorer(),
        cost_guard=None,
        telemetry=tel,
    )

    class _TD:
        task_id = "t-dereap"
        description = "do the thing"
        metadata: dict = {}

    trace, _tokens = asyncio.run(solver.execute(_TD()))
    assert trace.success is True

    rows = _read_rows(out)
    assert len(rows) == 1
    row = rows[0]
    # Pre-fix: finish_record wrote the row BEFORE de-reap (and clobbered the
    # pointer back to the unresolvable scratch path), so both pointers were null.
    assert row["agent_logs_ptr"] is not None
    assert row["transcript_ptr"] is not None
    assert row["agent_logs_ptr"].startswith("archive/")
    # The durable archive copy actually exists at that relpath.
    assert (out / row["agent_logs_ptr"]).is_dir()
    assert (out / row["agent_logs_ptr"] / "events.jsonl").exists()


# ---------------------------------------------------------------------------
# Finding 7 — OH/T2 spend recorded even when post-run scoring faults
# ---------------------------------------------------------------------------


def test_spend_recorded_on_post_run_scoring_fault(tmp_path):
    out = tmp_path / "run"
    tel = AgentTelemetry(output_dir=str(out))
    tracker = CostTracker(
        ledger_dir=tmp_path / "costs", daily_cap_usd=100.0, reservation_usd=1.0
    )
    guard = CostGuard(tracker, model="gpt-5.2")
    solver = _build_solver(
        tmp_path,
        backend=_LogWritingBackend(tokens=5000),
        scorer=_RaisingScorer(),
        cost_guard=guard,
        telemetry=tel,
    )

    class _TD:
        task_id = "t-spend"
        description = "do the thing"
        metadata: dict = {}

    trace, _tokens = asyncio.run(solver.execute(_TD()))
    # The never-raise wrap turns the post-run scorer fault into a degraded trace.
    assert trace.success is False

    # Pre-fix: cost_guard.record ran only AFTER scorer.score on the success path,
    # so a post-run fault dropped the already-incurred OH/T2 spend from the ledger
    # (today_total_usd() == 0). Post-fix: spend is folded in right after
    # collect_metrics, so the daily cap sees it.
    assert tracker.today_total_usd() > 0.0
