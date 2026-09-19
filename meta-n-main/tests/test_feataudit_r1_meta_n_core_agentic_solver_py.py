"""Round-1 correctness regressions for meta_n/core/agentic_solver.py.

Covers two ROUND-1 fixes, both landed inside AgenticSolver:

* R1-A-1 — ``demoted_code_library`` constructor kwarg: default ``None`` keeps the
  legacy blanket-zero demote (byte-identical for every existing/external caller);
  a supplied already-demoted dict rides through so an exempted seed survives and
  its advertisement is restored via ``merged_py``.
* R1-A-3 — ``AgenticSolver.execute`` threads ``solver_language`` into
  ``populate_adoption_fields``, restoring parity with ``MetaLayer.execute`` so a
  sys/os-using LIVE Python helper advertised+staged to a BASH solver is counted
  in ``utilities_available`` / ``utilities_called`` instead of collapsing to the
  demoted-path ``None``. Python (for_advertising=True) stays byte-identical.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

from meta_n.core.agentic_solver import AgenticSolver
from meta_n.core.meta_layer import InjectedCode, TaskDescription, Trace


# --- R1-A-1: demoted_code_library constructor contract ---------------------

_SEED_SRC = "def seed_h(x): return x"
_OMEGA_SRC = "def omega_h(x): return x"


def _solver(injected_codes, *, code_library_is_live, demoted_code_library=None):
    return AgenticSolver(
        llm_client=AsyncMock(),
        executor=AsyncMock(),
        injected_codes=injected_codes,
        code_library_is_live=code_library_is_live,
        demoted_code_library=demoted_code_library,
    )


def test_demoted_none_is_legacy_blanket_zero():
    """Default None + not-live -> merged_py == {} (byte-identical legacy demote)."""
    ics = [InjectedCode(code_library={"omega_h": _OMEGA_SRC}, source_depth=2)]
    solver = _solver(ics, code_library_is_live=False, demoted_code_library=None)
    assert solver.merged_py == {}


def test_demoted_dict_rides_through_when_not_live():
    """A supplied already-demoted dict (exempted seed) survives into merged_py,
    restoring the seed's advertisement on a demoting family."""
    ics = [
        InjectedCode(code_library={"seed_h": _SEED_SRC}, source_depth=0),
        InjectedCode(code_library={"omega_h": _OMEGA_SRC}, source_depth=2),
    ]
    solver = _solver(
        ics, code_library_is_live=False, demoted_code_library={"seed_h": _SEED_SRC}
    )
    assert solver.merged_py == {"seed_h": _SEED_SRC}
    assert "omega_h" not in solver.merged_py


def test_live_family_ignores_demoted_dict():
    """code_library_is_live=True skips the demote branch entirely -> merged_py
    keeps the fully merged library regardless of demoted_code_library."""
    ics = [InjectedCode(code_library={"omega_h": _OMEGA_SRC}, source_depth=2)]
    solver = _solver(ics, code_library_is_live=True, demoted_code_library={"x": "y"})
    assert "omega_h" in solver.merged_py
    assert "x" not in solver.merged_py


# --- R1-A-3: solver_language threaded into populate_adoption_fields ---------

# A Python helper whose raw source trips the safety blocklist (import sys) — the
# shape a terminal/SWE ``_lib_`` helper takes. Barred by for_advertising=True
# (python) but advertised+staged to a bash solver via the CLI form.
_SYS_HELPER = "import sys\ndef sys_helper(x):\n    return sys.argv[1:]"


class _SandboxEcho:
    """Sandboxed executor stub: echoes the (already library-prepended) script
    back on the Trace so the adoption scan sees the staged CLI reference."""

    is_sandboxed = True

    async def execute(self, script, task, timeout=30):
        return Trace(
            task_id=task.task_id, script=script, score=0.8, success=True, exit_code=0
        )


def _bash_lang_solver(code_first_turn, *, solver_language):
    counter = {"llm": 0}
    responses = [code_first_turn, "<status>complete</status>"]

    async def mock_complete(messages, temperature=None, max_tokens=None, **kw):
        idx = min(counter["llm"], len(responses) - 1)
        counter["llm"] += 1
        return responses[idx], 100

    llm_client = AsyncMock()
    llm_client.complete = mock_complete
    return AgenticSolver(
        llm_client=llm_client,
        executor=_SandboxEcho(),
        injected_codes=[InjectedCode(code_library={"sys_helper": _SYS_HELPER})],
        solver_language=solver_language,
        code_library_is_live=True,
        max_turns=8,
    )


async def test_bash_solver_measures_sys_helper_via_execute():
    """bash + sandboxed + sys-using LIVE helper: the CLI-form call is advertised,
    staged, and MEASURED (PRE-FIX voided to available=[] / called=None)."""
    code = (
        '<code lang="bash">\n'
        "python3 ./helpers/_lib_sys_helper.py foo\n"
        "</code>\n<status>complete</status>"
    )
    solver = _bash_lang_solver(code, solver_language="bash")
    trace, _ = await solver.execute(TaskDescription(task_id="t", description="d"))
    assert trace.utilities_available == ["sys_helper"]
    assert trace.utilities_called == ["sys_helper"]


async def test_python_solver_control_is_byte_identical():
    """Language control: identical setup with solver_language='python' filters the
    sys-using helper (for_advertising=True) -> available=[] / called=None,
    unchanged both pre- and post-fix. Proves the fix is language-scoped."""
    code = (
        '<code lang="python">\n'
        "result = sys_helper(3)\n"
        "</code>\n<status>complete</status>"
    )
    solver = _bash_lang_solver(code, solver_language="python")
    trace, _ = await solver.execute(TaskDescription(task_id="t", description="d"))
    assert trace.utilities_available == []
    assert trace.utilities_called is None
