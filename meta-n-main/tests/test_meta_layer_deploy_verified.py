"""P1a: --deploy-verified-code deterministic-adoption short-circuit.

The deploy fallback replaces a NON-ADOPTING authored solve() (empty / inline
re-derivation) with a deterministic wrapper that calls the verified helper. When
the flag is OFF the exec_script must be byte-identical to HEAD for every input.
"""
import pytest

from meta_n.core.meta_layer import (
    InjectedCode,
    MetaLayer,
    Trace,
    TaskDescription,
    _build_deploy_wrapper,
)

HELPER_NAME = "solve_crew_scheduling"
HELPER_SRC = (
    "def solve_crew_scheduling(N, K, time_limit, tasks, arcs):\n"
    "    return {'assignment': []}\n"
)


class CannedSolver:
    """Inner solver returning a fixed authored script."""

    def __init__(self, script: str):
        self.script = script

    async def solve(self, task, additional_context: str = ""):
        return self.script, "canned", 7


class CapturingExecutor:
    """Records the exec_script passed to execute()."""

    def __init__(self):
        self.exec_scripts = []

    async def execute(self, script: str, task: TaskDescription, timeout: int = 30) -> Trace:
        self.exec_scripts.append(script)
        return Trace(task_id=task.task_id, script=script, success=True, score=1.0)


@pytest.fixture
def task():
    return TaskDescription(task_id="crew_001", description="schedule the crew")


def _layer(solver, executor, *, deploy: bool):
    return MetaLayer(
        depth=2,
        injected_code=InjectedCode(),
        inner_solver=solver,
        executor=executor,
        merged_code_library={HELPER_NAME: HELPER_SRC},
        deploy_verified_code=deploy,
        solver_language="python",
    )


# --- ON behavior -----------------------------------------------------------

@pytest.mark.asyncio
async def test_empty_authored_solve_is_replaced(task):
    ex = CapturingExecutor()
    layer = _layer(CannedSolver(""), ex, deploy=True)
    await layer.execute(task)
    assert "def solve(**kw):" in ex.exec_scripts[0]
    assert HELPER_NAME in ex.exec_scripts[0]


@pytest.mark.asyncio
async def test_inline_rederivation_is_replaced(task):
    # Authored solve re-derives inline without calling the verified helper.
    authored = "def solve(**kw):\n    return {'assignment': []}\n"
    ex = CapturingExecutor()
    layer = _layer(CannedSolver(authored), ex, deploy=True)
    await layer.execute(task)
    body = ex.exec_scripts[0]
    # Wrapper now CALLS the helper (after the injected library prefix).
    assert f"return {HELPER_NAME}(" in body


@pytest.mark.asyncio
async def test_authored_call_kept(task):
    authored = (
        "def solve(**kw):\n"
        f"    return {HELPER_NAME}(**kw)\n"
    )
    ex_on = CapturingExecutor()
    layer_on = _layer(CannedSolver(authored), ex_on, deploy=True)
    await layer_on.execute(task)
    # Same as flag OFF — authored body preserved (the helper is already called).
    ex_off = CapturingExecutor()
    layer_off = _layer(CannedSolver(authored), ex_off, deploy=False)
    await layer_off.execute(task)
    assert ex_on.exec_scripts[0] == ex_off.exec_scripts[0]


# --- OFF byte-identity ------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "authored",
    ["", "def solve(**kw):\n    return {'assignment': []}\n",
     f"def solve(**kw):\n    return {HELPER_NAME}(**kw)\n"],
)
async def test_flag_off_byte_identical(task, authored):
    ex_off = CapturingExecutor()
    layer_off = _layer(CannedSolver(authored), ex_off, deploy=False)
    await layer_off.execute(task)
    # Reconstruct what _prepend_library would produce on the untouched script.
    expected = layer_off._prepend_library(authored)
    assert ex_off.exec_scripts[0] == expected


# --- Wrapper signature robustness ------------------------------------------

def test_wrapper_filters_stray_kwargs():
    src = "def helper(a, b):\n    return a + b\n"
    wrapper = _build_deploy_wrapper("helper", src)
    ns: dict = {}
    exec(src + wrapper, ns)
    # instance carries an extra key 'c' the helper does not accept.
    assert ns["solve"](a=1, b=2, c=99) == 3


def test_wrapper_var_keyword_forwards_all():
    src = "def helper(**kw):\n    return sorted(kw)\n"
    wrapper = _build_deploy_wrapper("helper", src)
    assert "helper(**kw)" in wrapper


def test_wrapper_unparseable_returns_none():
    assert _build_deploy_wrapper("missing", "def other():\n    pass\n") is None
    assert _build_deploy_wrapper("x", "def x(:\n bad") is None
