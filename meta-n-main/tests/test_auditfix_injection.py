"""Regression tests for audit findings 20 & 67 (external_agents/injection.py).

Both findings concern :class:`meta_n.core.external_agents.injection.InjectionMapper`.

* Finding 20 — the spine staged + re-exported helper source with NO validation
  gate, so a single unparseable / name-mismatched helper would make
  ``helpers/__init__.py`` raise at ``import helpers`` (the unconditional
  ``from .<name> import <name>`` re-export), poisoning EVERY helper import and
  silently zeroing adoption.
* Finding 67 — the helpers affordance note ("…the signatures above are the
  contract") was appended whenever the RAW pre-validation merged dicts were
  non-empty, even when every helper failed advertising-validation and NO
  signatures were emitted, leaving a dangling-contract note.

Each test FAILS on the pre-fix code and PASSES after the fix. All are offline /
LLM-free (no LM Studio, no Docker, no network) — pure dataclass + static-AST
exercises of ``InjectionMapper.build``.
"""

from __future__ import annotations

import ast

from meta_n.core.external_agents.injection import InjectionMapper
from meta_n.core.meta_layer import InjectedCode, SandboxMarker, TaskDescription


def _task() -> TaskDescription:
    return TaskDescription(task_id="t0", description="do the thing", metadata={})


def _mapper(code_library, language="python") -> InjectionMapper:
    injected = [InjectedCode(code_library=dict(code_library), source_depth=0)]
    return InjectionMapper(injected, language, SandboxMarker())


_GOOD = "def good():\n    return 1\n"


# --- Finding 20: re-export must not be poisoned by one bad helper -----------


def test_syntax_broken_helper_does_not_poison_reexport():
    """A syntactically-broken helper must NOT be staged or re-exported, so the
    valid helpers' ``from helpers import <name>`` keeps working.

    Pre-fix: ``safe_py`` is filtered only by ``_unsafe_helper_name``, so the
    broken helper is written to ``helpers/bad.py`` and ``helpers/__init__.py``
    gets ``from .bad import bad`` — importing the package raises SyntaxError.
    """
    broken = "def bad(:\n    return 2\n"  # syntax error (unparseable)
    plan = _mapper({"good": _GOOD, "bad": broken}).build(_task())

    init = plan.staged_files["helpers/__init__.py"]

    # The broken helper is dropped entirely.
    assert "helpers/bad.py" not in plan.staged_files
    assert "from .bad import bad" not in init

    # The valid helper survives and is re-exported.
    assert plan.staged_files["helpers/good.py"] == _GOOD
    assert "from .good import good" in init

    # Strong invariant: every staged helper module parses, so executing
    # __init__.py's re-exports could never raise SyntaxError.
    for path, content in plan.staged_files.items():
        if path.startswith("helpers/") and path.endswith(".py"):
            ast.parse(content)  # raises SyntaxError if a broken helper leaked through


def test_name_mismatched_helper_dropped_from_staging():
    """A helper that parses but binds NO top-level symbol matching its key would
    make ``from .<name> import <name>`` raise ImportError — it must be dropped.

    This case is NOT caught by ``validate_library_function`` (the body is valid,
    safe Python); only an importability gate excludes it. Pre-fix it is staged
    and re-exported, breaking ``import helpers``.
    """
    mismatched = "BAR = 123\n"  # parses, binds BAR — never `foo`
    plan = _mapper({"good": _GOOD, "foo": mismatched}).build(_task())

    init = plan.staged_files["helpers/__init__.py"]
    assert "helpers/foo.py" not in plan.staged_files
    assert "from .foo import foo" not in init
    # Valid helper unaffected.
    assert "from .good import good" in init
    assert "helpers/good.py" in plan.staged_files


def test_bash_mode_skips_runnable_wrapper_for_broken_helper():
    """In bash mode the broken helper must not get a runnable ``_lib`` file
    either (it is dropped from the shared validated ``safe_py`` set)."""
    broken = "def bad(:\n    return 2\n"
    plan = _mapper({"good": _GOOD, "bad": broken}, language="bash").build(_task())
    assert "helpers/_lib_bad.py" not in plan.staged_files
    assert "helpers/_lib_good.py" in plan.staged_files
    assert "helpers/bad.py" not in plan.staged_files


def test_valid_helper_unchanged_by_the_gate():
    """The fix must not over-drop: a plain valid helper is still staged and
    re-exported exactly as before (no regression for the common path)."""
    plan = _mapper({"good": _GOOD}).build(_task())
    assert plan.staged_files["helpers/good.py"] == _GOOD
    init = plan.staged_files["helpers/__init__.py"]
    assert "from .good import good" in init
    assert '__all__ = ["good"]' in init


# --- Finding 67: no dangling-contract note when no signatures were emitted --


_FLAGGED = "def dngr(x):\n    import socket\n    return socket.gethostname()\n"


def test_note_not_appended_when_all_helpers_fail_advertising_validation():
    """A parseable-but-safety-flagged helper is NOT advertised (descriptions
    come back empty), so the helpers-note must NOT be appended.

    Pre-fix the note is gated on the raw ``self.merged_py`` (non-empty), so the
    system_suffix becomes the bare note referencing 'the signatures above' that
    were never emitted.
    """
    plan = _mapper({"dngr": _FLAGGED}).build(_task())
    suffix = plan.prompt.system_suffix
    # No signatures were produced -> no contract note, empty suffix.
    assert "signatures above are the contract" not in suffix
    assert "PREFER calling them" not in suffix
    assert suffix == ""


def test_note_still_appended_when_real_signatures_exist():
    """Positive control: a genuinely advertised helper still gets the note, so
    the Finding-67 fix did not over-suppress the affordance."""
    suffix = _mapper({"good": _GOOD}).build(_task()).prompt.system_suffix
    assert "PREFER calling them" in suffix
    assert "from helpers import" in suffix


# --- HARD INVARIANT: gen0-vanilla parity (empty injection) -----------------


def test_gen0_empty_injection_remains_empty_suffix_and_no_staged_files():
    """An empty injected_codes set must still yield an empty system_suffix and no
    staged files — byte-identical to the pre-fix vanilla baseline."""
    plan = InjectionMapper([], "python", SandboxMarker()).build(_task())
    assert plan.prompt.system_suffix == ""
    assert plan.prompt.prefix == ""
    assert plan.staged_files == {}
    assert plan.utilities_available == []
    assert plan.pre_process_ran is False
