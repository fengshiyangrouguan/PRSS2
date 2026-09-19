"""G3 silent-void fixes for MetaLayer (docs/metan_silent_void_audit.md).

T1.3 (dual-field trap): an ad-hoc single-layer MetaLayer that omits the
``merged_code_library`` kwarg must self-wire its OWN per-layer
``injected_code.code_library`` instead of silently dropping it. An explicit
kwarg (including ``{}``) still wins.

T3.1 (degrades-silently): MetaLayer.solve() must be symmetric with execute() —
advertise (fold lib_desc into the inner context) and stage (_prepend_library)
the library. Empty library stays byte-identical to the inner solver output.
"""

import pytest

from meta_n.core.base_executor import LocalExecutor
from meta_n.core.meta_layer import InjectedCode, MetaLayer, TaskDescription


class MockSolver:
    def __init__(self, script: str = "echo hello"):
        self.script = script
        self.last_additional_context = ""

    async def solve(self, task: TaskDescription, additional_context: str = "") -> tuple[str, str, int]:
        self.last_additional_context = additional_context
        return self.script, "mock reasoning", 50


@pytest.fixture
def executor():
    return LocalExecutor()


@pytest.fixture
def task():
    return TaskDescription(task_id="g3_001", description="Test task")


HELPER_SRC = "def my_helper(x):\n    return x + 1"


# --- T1.3: dual-field trap (self-wire from injected_code) ---


class TestT13DualFieldSelfWire:
    def test_omitted_kwarg_self_wires_from_injected_code(self, executor):
        ic = InjectedCode(code_library={"my_helper": HELPER_SRC})
        layer = MetaLayer(
            depth=2, injected_code=ic, inner_solver=MockSolver(), executor=executor,
        )
        # No kwarg ⇒ self-advertise the per-layer library.
        assert "my_helper" in layer.merged_code_library
        assert layer.merged_code_library["my_helper"] == HELPER_SRC

    def test_omitted_kwarg_self_wires_bash(self, executor):
        ic = InjectedCode(code_library_bash={"bfn": "bfn() { echo hi; }"})
        layer = MetaLayer(
            depth=2, injected_code=ic, inner_solver=MockSolver(), executor=executor,
        )
        assert "bfn" in layer.merged_code_library_bash

    def test_explicit_empty_kwarg_wins(self, executor):
        ic = InjectedCode(code_library={"my_helper": HELPER_SRC})
        layer = MetaLayer(
            depth=2, injected_code=ic, inner_solver=MockSolver(), executor=executor,
            merged_code_library={},
        )
        # Explicit {} ⇒ kwarg wins, NOT self-wired (orchestrator demote path).
        assert layer.merged_code_library == {}

    def test_explicit_nonempty_kwarg_wins(self, executor):
        ic = InjectedCode(code_library={"my_helper": HELPER_SRC})
        layer = MetaLayer(
            depth=2, injected_code=ic, inner_solver=MockSolver(), executor=executor,
            merged_code_library={"other": "def other(): pass"},
        )
        assert layer.merged_code_library == {"other": "def other(): pass"}
        assert "my_helper" not in layer.merged_code_library

    def test_no_library_anywhere_stays_empty(self, executor):
        layer = MetaLayer(
            depth=2, injected_code=InjectedCode(), inner_solver=MockSolver(), executor=executor,
        )
        assert layer.merged_code_library == {}
        assert layer.merged_code_library_bash == {}

    def test_explicit_none_means_empty_not_self_wired(self, executor):
        # REGRESSION (audit fix-round-1): the orchestrator passes an EXPLICIT
        # ``merged_code_library=None`` for every non-outermost layer to mean
        # "stage nothing here". That explicit None must reproduce HEAD's
        # ``None or {}`` == {} — it must NOT trigger the T1.3 self-wire (which
        # would let an intermediate-depth helper bypass demotion / double-prepend
        # on every depth>=2 candidate, a flagless default-path drift).
        ic = InjectedCode(
            code_library={"my_helper": HELPER_SRC},
            code_library_bash={"bfn": "bfn() { echo hi; }"},
        )
        layer = MetaLayer(
            depth=2, injected_code=ic, inner_solver=MockSolver(), executor=executor,
            merged_code_library=None, merged_code_library_bash=None,
        )
        assert layer.merged_code_library == {}
        assert layer.merged_code_library_bash == {}

    @pytest.mark.asyncio
    async def test_orchestrator_inner_layer_none_does_not_stage(self, executor, task):
        # The depth>=2 case the stage23 golden is blind to: an inner layer built
        # exactly as the orchestrator builds non-outermost layers (explicit None +
        # a non-empty per-layer library). solve() — invoked by the outer
        # execute() — must NOT advertise or stage the inner helper.
        inner_ic = InjectedCode(code_library={"inner_helper": "def inner_helper():\n    return 7"})
        inner = MetaLayer(
            depth=2, injected_code=inner_ic,
            inner_solver=MockSolver(script="def solve(**kw):\n    return 1"),
            executor=executor, merged_code_library=None,
        )
        assert inner.merged_code_library == {}
        script, _, _ = await inner.solve(task)
        assert "inner_helper" not in script        # not staged ⇒ no bypass / no double-prepend
        assert script == "def solve(**kw):\n    return 1"


# --- T3.1: solve() symmetric with execute() ---


class TestT31SolveLibraryChannel:
    @pytest.mark.asyncio
    async def test_solve_advertises_and_stages_with_library(self, executor, task):
        solver = MockSolver(script="def solve(**kw):\n    return my_helper(0)")
        ic = InjectedCode(code_library={"my_helper": HELPER_SRC})
        layer = MetaLayer(
            depth=2, injected_code=ic, inner_solver=solver, executor=executor,
            merged_code_library={"my_helper": HELPER_SRC},
        )
        script, reasoning, tokens = await layer.solve(task)
        # Advertised: helper name reached the inner solver's context.
        assert "my_helper" in solver.last_additional_context
        # Staged: helper def is prepended to the returned script.
        assert "my_helper" in script
        assert script.endswith(solver.script)
        assert reasoning == "mock reasoning"

    @pytest.mark.asyncio
    async def test_solve_empty_library_byte_identical(self, executor, task):
        solver = MockSolver(script="def solve(**kw):\n    return 1")
        layer = MetaLayer(
            depth=2, injected_code=InjectedCode(), inner_solver=solver, executor=executor,
            merged_code_library={},
        )
        script, _, _ = await layer.solve(task)
        # Empty library ⇒ no advertise, no stage ⇒ byte-identical inner output.
        assert script == solver.script
        assert solver.last_additional_context == ""

    @pytest.mark.asyncio
    async def test_solve_frozen_winner_short_circuits(self, executor, task):
        solver = MockSolver(script="def solve(**kw):\n    return 1")
        ic = InjectedCode(
            code_library={"my_helper": HELPER_SRC},
            task_solution_map={task.task_id: "FROZEN_SCRIPT"},
        )
        layer = MetaLayer(
            depth=2, injected_code=ic, inner_solver=solver, executor=executor,
            merged_code_library={"my_helper": HELPER_SRC},
        )
        script, reasoning, tokens = await layer.solve(task)
        # Frozen routing must still bypass the library channel entirely.
        assert script == "FROZEN_SCRIPT"
        assert tokens == 0
