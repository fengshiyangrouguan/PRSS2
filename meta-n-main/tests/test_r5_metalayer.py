"""R5 — meta_layer/adoption language-correctness fixes.

1. ``_maybe_deploy_verified_helper`` must fire ONLY for a python-language layer:
   the deploy wrapper is Python ``def solve(**kw)`` source, so a bash layer must
   pass its authored script through untouched (pre-fix it shipped Python to the
   bash executor — guaranteed 'def: command not found').
2. ``populate_adoption_fields`` must union ``merged_code_library_bash`` names
   into ``utilities_available`` ONLY for a bash solver — the python
   advertise/stage path never shows or stages bash helpers, so counting them
   violates the shown==available invariant and corrupts the None-vs-[]
   discriminator.
3. ``classify_error``: the structured ``terminated_by`` branches (max_turns /
   env_error / timeout) must ALL outrank the 'max turns' text sniff, which is
   only a fallback when terminated_by carries no structured signal.
"""

from meta_n.core.meta_layer import (
    InjectedCode,
    MetaLayer,
    SandboxMarker,
    Trace,
    TaskDescription,
    classify_error,
    populate_adoption_fields,
)

HELPER_NAME = "count_lines"
HELPER_SRC = (
    "def count_lines(text):\n"
    "    return len(text.splitlines())\n"
)


class CannedSolver:
    def __init__(self, script: str):
        self.script = script

    async def solve(self, task, additional_context: str = ""):
        return self.script, "canned", 7


class CapturingExecutor:
    def __init__(self):
        self.exec_scripts = []

    async def execute(self, script: str, task: TaskDescription, timeout: int = 30) -> Trace:
        self.exec_scripts.append(script)
        return Trace(task_id=task.task_id, script=script, success=True, score=1.0)


def _layer(solver, executor, *, language: str, deploy: bool = True):
    return MetaLayer(
        depth=2,
        injected_code=InjectedCode(),
        inner_solver=solver,
        executor=executor,
        merged_code_library={HELPER_NAME: HELPER_SRC},
        deploy_verified_code=deploy,
        solver_language=language,
    )


# --- Fix 1: deploy gate on solver_language ----------------------------------

async def test_bash_layer_authored_script_passes_through_untouched():
    task = TaskDescription(task_id="t1", description="d")
    ex = CapturingExecutor()
    layer = _layer(CannedSolver("echo hello world"), ex, language="bash")
    await layer.execute(task)
    assert ex.exec_scripts[0] == layer._prepend_library("echo hello world")
    assert "def solve(**kw):" not in ex.exec_scripts[0]


def test_bash_layer_maybe_deploy_returns_identical_object():
    layer = _layer(CannedSolver(""), CapturingExecutor(), language="bash")
    script = "echo hello world"
    assert layer._maybe_deploy_verified_helper(script) is script


def test_openevolve_layer_maybe_deploy_returns_identical_object():
    layer = _layer(CannedSolver(""), CapturingExecutor(), language="openevolve")
    script = "def run_search():\n    pass\n"
    assert layer._maybe_deploy_verified_helper(script) is script


async def test_python_layer_wrapper_still_deploys():
    task = TaskDescription(task_id="t2", description="d")
    ex = CapturingExecutor()
    layer = _layer(CannedSolver(""), ex, language="python")
    await layer.execute(task)
    assert "def solve(**kw):" in ex.exec_scripts[0]
    assert f"return {HELPER_NAME}(" in ex.exec_scripts[0]


# --- Fix 2: language-aware bash-name union in populate_adoption_fields ------

def test_python_solver_bash_only_helpers_stay_unmeasurable():
    tr = Trace(task_id="t", script="print('greet')")
    populate_adoption_fields(
        tr,
        command_count=1,
        merged_code_library={},
        merged_code_library_bash={"greet": "greet() { echo hi; }"},
        executor=SandboxMarker(),
        solver_language="python",
    )
    assert tr.utilities_available == []
    assert tr.utilities_called is None
    assert tr.utilities_call_counts == {}


def test_python_solver_union_excludes_bash_names():
    clean = "def clean(x):\n    return x + 1"
    tr = Trace(
        task_id="t",
        script="# --- end injected code library ---\nclean(1)\ngreet\n",
    )
    populate_adoption_fields(
        tr,
        command_count=1,
        merged_code_library={"clean": clean},
        merged_code_library_bash={"greet": "greet() { echo hi; }"},
        executor=SandboxMarker(),
        solver_language="python",
    )
    assert tr.utilities_available == ["clean"]
    assert tr.utilities_called == ["clean"]


def test_bash_solver_bash_helpers_still_available_and_called():
    tr = Trace(
        task_id="t",
        script="# --- end injected code library ---\ngreet\n",
    )
    populate_adoption_fields(
        tr,
        command_count=1,
        merged_code_library={},
        merged_code_library_bash={"greet": "greet() { echo hi; }"},
        executor=SandboxMarker(),
        solver_language="bash",
    )
    assert tr.utilities_available == ["greet"]
    assert tr.utilities_called == ["greet"]


def test_bash_solver_union_keeps_both_libraries():
    clean = "def clean(x):\n    return x + 1"
    tr = Trace(
        task_id="t",
        script="# --- end injected code library ---\necho measured-zero\n",
    )
    populate_adoption_fields(
        tr,
        command_count=1,
        merged_code_library={"clean": clean},
        merged_code_library_bash={"greet": "greet() { echo hi; }"},
        executor=SandboxMarker(),
        solver_language="bash",
    )
    assert tr.utilities_available == ["clean", "greet"]
    assert tr.utilities_called == []


# --- Fix 3: structured terminated_by outranks 'max turns' text sniff --------

def _cls(**kw):
    return classify_error(Trace(task_id="t", success=False, **kw))


def test_env_error_tb_beats_max_turns_text():
    assert _cls(
        terminated_by="env_error",
        error_summary="docker compose failed; agent config: max turns=50",
    ) == "Environment fault"


def test_timeout_tb_beats_max_turns_text():
    assert _cls(
        terminated_by="timeout",
        error_summary="run killed after hitting max_turns limit",
    ) == "Timeout"


def test_pure_text_max_turns_with_empty_tb_is_turn_starvation():
    assert _cls(error_summary="hit max turns") == "Turn starvation"


def test_token_budget_tb_still_falls_through_to_max_turns_text():
    assert _cls(
        terminated_by="token_budget",
        error_summary="stopped near max turns",
    ) == "Turn starvation"


def test_structured_max_turns_still_beats_timeout_text():
    assert _cls(
        terminated_by="max_turns_unconfirmed_complete",
        error_summary="connection timeout",
    ) == "Turn starvation"
