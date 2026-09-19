"""Re-audit regression tests for injection.py sibling-channel consistency.

Two findings from the #20 staging-filter fix (reaudit_findings.json ids 2 & 8),
both offline (no SDK / Docker / LLM / Ω): they exercise
``InjectionMapper.build`` on hand-built ``InjectedCode`` chains and assert that
the three sibling channels — *advertised* (prompt descriptions), *staged*
(``staged_files``), and the adoption-*denominator* (``utilities_available``) —
agree.

id=8 (medium): ``utilities_available`` was populated from the UNFILTERED merged
dicts while staging filtered to a safe subset, inflating the adoption
denominator with never-staged helpers.

id=2 (low): the staging gate (top-level-body only) dropped a helper that binds
its name inside a module-level compound statement (``try/except`` import
fallback), while the advertising gate still promoted it — an
advertised-but-unstaged helper whose ``from helpers import <name>`` raises
ImportError. The fix widens the gate to detect module-scope bindings anywhere
and advertises only from the staged subset (advertised ⊆ staged).
"""

from __future__ import annotations

from meta_n.core.external_agents.injection import _HELPERS_NOTE_PY, InjectionMapper
from meta_n.core.meta_layer import InjectedCode, SandboxMarker, TaskDescription


def _task() -> TaskDescription:
    return TaskDescription(task_id="t0", description="do the thing", metadata={})


def _injected(code_library=None, code_library_bash=None) -> InjectedCode:
    return InjectedCode(
        pre_process=None,
        code_library=dict(code_library or {}),
        code_library_bash=dict(code_library_bash or {}),
        source_depth=0,
    )


def _mapper(injected, language="python") -> InjectionMapper:
    return InjectionMapper(injected, language, SandboxMarker())


# --- id=8: utilities_available must equal the STAGED set, not merged ---------


def test_utilities_available_excludes_never_staged_helper():
    # 'good' has a valid top-level def; 'mismatch' is keyed 'mismatch' but its
    # body defines 'other' — the staging gate drops it (no module-scope binding
    # of 'mismatch'), so it is never written to helpers/ and the agent can never
    # call it. Pre-fix utilities_available counted it (['good','mismatch']),
    # inflating the adoption denominator; post-fix it must be excluded.
    good = "def good():\n    return 1\n"
    mismatch = "def other():\n    return 2\n"
    plan = _mapper([_injected(code_library={"good": good, "mismatch": mismatch})]).build(
        _task()
    )

    assert plan.utilities_available == ["good"]
    assert "mismatch" not in plan.utilities_available

    # Denominator contract (backend.py:90): every name counted is actually staged.
    for name in plan.utilities_available:
        assert f"helpers/{name}.py" in plan.staged_files
    assert "helpers/good.py" in plan.staged_files
    assert "helpers/mismatch.py" not in plan.staged_files


def test_utilities_available_bash_channel_excludes_never_staged_python_helper():
    # Sibling bash path: a mismatched python helper must be dropped from the
    # denominator too, while a valid bash helper is kept.
    good = "def good():\n    return 1\n"
    mismatch = "def other():\n    return 2\n"
    blit = "blit() {\n  echo hi\n}\n"
    plan = _mapper(
        [_injected(code_library={"good": good, "mismatch": mismatch},
                   code_library_bash={"blit": blit})],
        language="bash",
    ).build(_task())

    assert plan.utilities_available == ["blit", "good"]
    assert "mismatch" not in plan.utilities_available
    assert "helpers/mismatch.py" not in plan.staged_files


# --- id=2: compound-defined helper must be staged (regression from #20) ------


_TRY_FALLBACK_HELPER = (
    "try:\n"
    "    FAST = True\n"
    "    def compute(x):\n"
    '        """Fast path."""\n'
    "        return x * 2\n"
    "except Exception:\n"
    "    def compute(x):\n"
    '        """Slow path."""\n'
    "        return x + x\n"
)


def test_compound_defined_helper_is_staged():
    # 'compute' is bound at MODULE scope inside a top-level try/except (the common
    # import-fallback pattern), so `from .compute import compute` imports fine.
    # Pre-fix the top-level-only gate dropped it from staging; post-fix it must be
    # staged, re-exported, and counted.
    plan = _mapper([_injected(code_library={"compute": _TRY_FALLBACK_HELPER})]).build(
        _task()
    )

    assert "helpers/compute.py" in plan.staged_files
    assert "compute" in plan.utilities_available
    assert "from .compute import compute" in plan.staged_files["helpers/__init__.py"]
    # Advertised AND staged -> the prompt's import note is satisfiable.
    assert _HELPERS_NOTE_PY in plan.prompt.system_suffix


def test_advertised_subset_of_staged_nested_only_helper_not_advertised():
    # 'inner' is defined ONLY inside another function's body (a new scope), so it
    # is NOT importable as `from .inner import inner`. Pre-fix it was advertised
    # (ast.walk sees the nested FunctionDef) yet never staged -> ImportError at
    # `from helpers import inner`. Post-fix advertising is driven by the staged
    # subset, so a non-importable helper is neither advertised nor counted.
    nested = "def outer():\n    def inner():\n        return 1\n    return inner\n"
    plan = _mapper([_injected(code_library={"inner": nested})]).build(_task())

    assert "helpers/inner.py" not in plan.staged_files
    assert plan.utilities_available == []
    # advertised ⊆ staged: nothing staged -> nothing advertised.
    assert plan.prompt.system_suffix == ""


# --- gen0 parity (must stay byte-identical) ----------------------------------


def test_gen0_empty_injection_byte_identical():
    plan = _mapper([]).build(_task())
    assert plan.prompt.system_suffix == ""
    assert plan.prompt.prefix == ""
    assert plan.staged_files == {}
    assert plan.utilities_available == []
    assert plan.pre_process_ran is False
