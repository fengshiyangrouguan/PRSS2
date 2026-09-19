"""Checkpoint A — builtin-backend self-consistency round-trip (§12.1).

Builds the full spine harness (``ExternalAgentSolver`` + the real
:class:`BuiltinBackend`) around a synthetic, committed 2-task CO-Bench-shaped
fixture and a fake native solver whose calls flow through a fake outer
``LLMClient.cumulative_usage`` ledger — exactly the builtin token-accounting path
(``outer_token_mode=True``, plan §2.3 FIX / §7.11). No Docker, no CO-Bench data,
no subprocess: install-free.

NOTE on what this proves: the committed baseline (``cobench_2task_baseline.json``)
is hand-authored to EQUAL the synthetic INPUT fixture (``cobench_2task.json``), so
this asserts that ``BuiltinBackend`` passes its scores through UNCHANGED and
records the ``(outer_prompt, outer_completion, inner=0)`` tuple with
``token_basis="outer"`` — a self-consistency round-trip. It is NOT a parity check
against a real ``Layer1Solver``/legacy-path execution (the baseline is not a
recording of that path). The live co_bench+builtin parity is guarded separately by
the two tests in ``tests/test_evolutionary_orchestrator.py``. If the committed
baseline JSON is absent, the equality assertion is skipped — but the
``(outer_prompt, outer_completion, inner=0)`` *shape* is still asserted on
whatever the harness produces.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from meta_n.core.external_agents.backends.builtin import BuiltinBackend
from meta_n.core.external_agents.solver import ExternalAgentSolver
from meta_n.core.meta_layer import TaskDescription, Trace
from meta_n.integrations.benchmark import EvalResult

from contextlib import asynccontextmanager

from meta_n.core.external_agents import AgentTelemetry, DockerRunGuard
from meta_n.core.external_agents.budget import CostGuard
from meta_n.core.external_agents.env import AgentEnvProvider, Scorer
from meta_n.utils.cost_tracker import CostTracker

from .conftest import read_run_records

_FIXTURE_DIR = Path(__file__).parent / "fixtures"
_FIXTURE = _FIXTURE_DIR / "cobench_2task.json"
_BASELINE = _FIXTURE_DIR / "cobench_2task_baseline.json"


def _load_fixture() -> list[dict]:
    return json.loads(_FIXTURE.read_text())["tasks"]


# --- fakes that reproduce the builtin (outer-token) path -------------------


class _FakeLLMClient:
    """Outer client whose ``cumulative_usage`` ledger the builtin path reads."""

    def __init__(self):
        self.cumulative_usage = {
            "prompt": 0,
            "completion": 0,
            "total": 0,
            "cached": 0,
            "calls": 0,
        }


class _FakeNativeSolver:
    """A native solver that bumps the outer ledger and returns a scored Trace.

    Wrapped by ``BuiltinBackend`` via the ``execute()`` (agentic/chain) route, so
    no separate executor is needed. The ledger bump is what the builtin backend's
    ``_usage_delta`` recovers as the prompt/completion split.
    """

    def __init__(self, llm_client, spec_by_id: dict[str, dict]):
        self.llm_client = llm_client
        self.spec_by_id = spec_by_id

    async def execute(self, task) -> tuple[Trace, int]:
        spec = self.spec_by_id[task.task_id]
        p = int(spec["outer_prompt_tokens"])
        c = int(spec["outer_completion_tokens"])
        u = self.llm_client.cumulative_usage
        u["prompt"] += p
        u["completion"] += c
        u["total"] += p + c
        u["calls"] += 1
        trace = Trace(
            task_id=task.task_id,
            depth=1,
            script=spec["solution"],
            success=True,
            score=float(spec["score"]),
            duration_s=0.01,
        )
        return trace, p + c


class _SolveFileEnvProvider(AgentEnvProvider):
    """Provisions a workspace and stages the fixture's solve.py as the solution."""

    def __init__(self, spec_by_id: dict[str, dict]):
        self.spec_by_id = spec_by_id

    @asynccontextmanager
    async def provision(self, task, lease):
        root = lease.workdir / "workspace"
        root.mkdir(parents=True, exist_ok=True)

        class _Env:
            # The builtin backend recovers the task from ctx.workspace, so the
            # workspace_handle must carry it (it needs task.metadata etc.).
            workspace_handle = task

        yield _Env()

    async def stage_files(self, env, files):
        return None

    async def extract_solution(self, env, run) -> str:
        # The "authored" solve.py is the fixture solution.
        task_id = getattr(getattr(run, "native_handle", None), "task_id", "")
        spec = self.spec_by_id.get(task_id, {})
        return spec.get("solution", "")


class _FixtureScorer(Scorer):
    """Returns the recorded score/raw_score for the task (deterministic)."""

    def __init__(self, spec_by_id: dict[str, dict]):
        self.spec_by_id = spec_by_id

    async def score(self, task, env, solution, run) -> EvalResult:
        spec = self.spec_by_id[task.task_id]
        return EvalResult(
            success=True,
            score=float(spec["score"]),
            raw_score=float(spec["raw_score"]),
            feedback="fixture score",
        )


def _build_solver(spec_by_id, tmp_path) -> tuple[ExternalAgentSolver, AgentTelemetry]:
    llm = _FakeLLMClient()
    native = _FakeNativeSolver(llm, spec_by_id)
    backend = BuiltinBackend(
        solver=native,
        llm_client=llm,
        use_agentic=True,  # routes through solver.execute() — no executor needed
        depth=1,
    )
    telemetry = AgentTelemetry(tmp_path / "run")
    scratch = tmp_path / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    guard = DockerRunGuard(max_docker=1, scratch_root=str(scratch))
    tracker_dir = tmp_path / "costs"
    cost_guard = CostGuard(
        CostTracker(ledger_dir=tracker_dir, daily_cap_usd=1000.0), model="gpt-5.2"
    )
    solver = ExternalAgentSolver(
        backend=backend,
        env_provider=_SolveFileEnvProvider(spec_by_id),
        scorer=_FixtureScorer(spec_by_id),
        injected_codes=[],
        depth=1,
        run_guard=guard,
        telemetry=telemetry,
        cost_guard=cost_guard,
        adapter=object(),
    )
    return solver, telemetry


# --- tests -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_builtin_outer_token_mode_is_true():
    backend = BuiltinBackend(solver=object(), llm_client=None)
    assert backend.outer_token_mode is True
    assert backend.name == "builtin"


@pytest.mark.asyncio
async def test_builtin_parity_shape_and_scores(tmp_path):
    tasks = _load_fixture()
    spec_by_id = {t["task_id"]: t for t in tasks}
    solver, telemetry = _build_solver(spec_by_id, tmp_path)

    produced: dict[str, dict] = {}
    for t in tasks:
        task = TaskDescription(task_id=t["task_id"], description=t["description"])
        trace, outer_tokens = await solver.execute(task)

        # The builtin backend returns native OUTER tokens (outer_token_mode=True);
        # the spine returns that total as the int.
        assert trace.success is True
        assert trace.score == pytest.approx(t["score"])
        expected_total = t["outer_prompt_tokens"] + t["outer_completion_tokens"]
        assert outer_tokens == expected_total

        produced[t["task_id"]] = {
            "score": trace.score,
            "outer_tokens": outer_tokens,
        }

    # Read the telemetry rows and assert the (outer_prompt, outer_completion,
    # inner=0) shape on EVERY run — this holds regardless of the baseline file.
    rows = {r["task_id"]: r for r in read_run_records(telemetry.output_dir)}
    assert set(rows) == set(spec_by_id)
    for task_id, spec in spec_by_id.items():
        row = rows[task_id]
        # token_basis is 'outer' for the builtin backend (§2.3 FIX, §7.11).
        assert row["token_basis"] == "outer"
        # The (outer_prompt, outer_completion) split is recovered from the
        # outer LLMClient ledger delta and stamped on the row.
        assert row["prompt_tokens"] == spec["outer_prompt_tokens"]
        assert row["completion_tokens"] == spec["outer_completion_tokens"]
        # The INNER axis is zeroed on an outer-basis row: the spend is OUTER
        # authoring spend (already counted on prompt/completion/total), so the
        # record must NOT also stamp it onto inner_tokens/inner_calls (else any
        # consumer reading the inner axis without checking token_basis would
        # double-count the authoring spend as inner agent spend). This makes the
        # record self-consistent (token_basis='outer' with inner_* == 0) without
        # the downstream aggregator needing to mask it.
        assert row["inner_tokens"] == 0
        assert row["inner_calls"] == 0
        # The outer authoring totals are still the row's legitimate token quantity.
        assert row["total_tokens"] == (
            spec["outer_prompt_tokens"] + spec["outer_completion_tokens"]
        )
        # inner=0 (the third element of the documented tuple): for the builtin
        # backend the spine returns the spend as the OUTER int — it does NOT add
        # a second, inner contribution on top of the outer ledger. The outer int
        # the spine returned equals the outer prompt+completion total.
        assert produced[task_id]["outer_tokens"] == (
            spec["outer_prompt_tokens"] + spec["outer_completion_tokens"]
        )

    # --- baseline equality (skipped if the recorded baseline is absent) ---
    if not _BASELINE.exists():
        pytest.skip(
            f"recorded baseline {_BASELINE.name} absent; asserted "
            "(outer_prompt, outer_completion, inner=0) shape only"
        )

    baseline = json.loads(_BASELINE.read_text())["runs"]
    for task_id, spec in spec_by_id.items():
        base = baseline[task_id]
        row = rows[task_id]
        # Exact equality on scores.
        assert row["score"] == pytest.approx(base["score"])
        assert row["raw_score"] == pytest.approx(base["raw_score"])
        # Exact equality on the (outer_prompt, outer_completion, inner=0) tuple.
        assert (row["prompt_tokens"], row["completion_tokens"]) == (
            base["outer_prompt"],
            base["outer_completion"],
        )
        assert base["inner"] == 0
        assert row["token_basis"] == base["token_basis"] == "outer"
