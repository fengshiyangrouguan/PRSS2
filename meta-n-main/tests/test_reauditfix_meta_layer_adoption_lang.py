"""Re-audit fix #7 — populate_adoption_fields must mirror the advertise+stage
gate ACTUALLY used for the solver's language.

Regression: R2-CS-2 filtered Python helper names through
``validate_library_function(..., for_advertising=True)`` UNCONDITIONALLY. That is
correct for a PYTHON solver (format_python_library_descriptions also passes
for_advertising=True), but WRONG for a BASH solver under a sandboxed executor:
the bash advertise/stage gates (format_bash_library_descriptions:732 /
prepend_bash_library:868) deliberately do NOT pass for_advertising (audit-#66
by-design exception — bash ``_lib_`` helpers are advertised as the sys-using CLI
form ``python3 ./helpers/_lib_<name>.py`` and legitimately import sys). So a
sys/os-using Python helper IS advertised+staged to a bash solver, yet
``populate_adoption_fields`` dropped it from ``utilities_available``, deflating —
or, if ALL helpers were blocklisted, VOIDING (utilities_called=None) — the
live-helper bash adoption telemetry, conflating it with the CO-Bench demoted path.

These tests fail on the pre-fix code (bash + sys-helper collapses to
utilities_called=None / empty utilities_available) and pass after the fix.
"""

from meta_n.core.meta_layer import (
    SandboxMarker,
    Trace,
    populate_adoption_fields,
)

# A Python helper whose raw source trips the safety blocklist (import sys) — the
# exact shape a terminal/SWE-bench ``_lib_`` helper takes (argv parsing in its
# CLI tail). Under a SANDBOXED executor this is barred by for_advertising=True
# but PASSES the plain gate (parity with the bash advertise+stage path).
_SYS_HELPER = "import sys\ndef sys_helper(x):\n    return sys.argv[1:]"


def _fresh_trace(script: str) -> Trace:
    return Trace(task_id="t", script=script)


def test_bash_solver_keeps_sys_using_helper_available():
    """A sys-using helper survives utilities_available for a BASH solver.

    The bash solver invokes it via the CLI form; the script references the staged
    file, so it must count as both available AND called.
    """
    ex = SandboxMarker()  # is_sandboxed=True
    script = (
        "# --- end injected code library ---\n"
        "python3 ./helpers/_lib_sys_helper.py foo\n"
    )
    tr = _fresh_trace(script)
    populate_adoption_fields(
        tr,
        command_count=1,
        merged_code_library={"sys_helper": _SYS_HELPER},
        merged_code_library_bash={},
        executor=ex,
        solver_language="bash",
    )
    # Live helper: staged + advertised to the bash solver → must be available.
    assert tr.utilities_available == ["sys_helper"]
    # Not the CO-Bench demoted regime — measured, and the script calls it.
    assert tr.utilities_called == ["sys_helper"]


def test_bash_solver_native_inline_lib_form_counts_called():
    """R1-A-2: the native bash path stages+invokes the CLI form via the inline
    ``python3 /tmp/_lib_<name>.py`` path (code_library ``lib_path_fmt`` default),
    NOT the spine ``./helpers/`` path. That call-site must count as CALLED —
    fails pre-fix (file_pat only matched the ``helpers/`` prefix), passes post-fix.
    """
    ex = SandboxMarker()  # is_sandboxed=True
    script = (
        "# --- end injected code library ---\n"
        "python3 /tmp/_lib_sys_helper.py foo\n"
    )
    tr = _fresh_trace(script)
    populate_adoption_fields(
        tr,
        command_count=1,
        merged_code_library={"sys_helper": _SYS_HELPER},
        merged_code_library_bash={},
        executor=ex,
        solver_language="bash",
    )
    assert tr.utilities_available == ["sys_helper"]
    assert tr.utilities_called == ["sys_helper"]


def test_bash_solver_all_blocklisted_helpers_do_not_void_telemetry():
    """Pre-fix: an all-sys-helper bash trace collapsed to utilities_called=None,
    identical to the CO-Bench DEMOTED path. Post-fix it stays MEASURED."""
    ex = SandboxMarker()
    # Authored solve did not reference the helper → measured-zero (not None).
    tr = _fresh_trace("# --- end injected code library ---\necho hi\n")
    populate_adoption_fields(
        tr,
        command_count=1,
        merged_code_library={"sys_helper": _SYS_HELPER},
        merged_code_library_bash={},
        executor=ex,
        solver_language="bash",
    )
    assert tr.utilities_available == ["sys_helper"]
    # Measured-zero [], NOT the unmeasurable None of the demoted path.
    assert tr.utilities_called == []


def test_python_solver_filters_sys_using_helper():
    """For a PYTHON solver the same helper is (correctly) filtered out — parity
    with format_python_library_descriptions(for_advertising=True). It is never
    advertised, so it must not sit in utilities_available; with no other live
    helper the trace stays unmeasurable (None)."""
    ex = SandboxMarker()
    script = "# --- end injected code library ---\nsys_helper(1)\n"
    tr = _fresh_trace(script)
    populate_adoption_fields(
        tr,
        command_count=1,
        merged_code_library={"sys_helper": _SYS_HELPER},
        merged_code_library_bash={},
        executor=ex,
        solver_language="python",
    )
    assert tr.utilities_available == []
    assert tr.utilities_called is None


def test_python_default_language_is_advertising_gate():
    """Omitting solver_language defaults to the python (for_advertising=True)
    gate — backward-compatible with the existing native/agentic call sites that
    pass no solver_language."""
    ex = SandboxMarker()
    tr = _fresh_trace("# --- end injected code library ---\nsys_helper(1)\n")
    populate_adoption_fields(
        tr,
        command_count=1,
        merged_code_library={"sys_helper": _SYS_HELPER},
        merged_code_library_bash={},
        executor=ex,
    )
    assert tr.utilities_available == []
    assert tr.utilities_called is None


def test_gen0_empty_injection_byte_identical_for_bash():
    """Empty merged libraries → no helpers → utilities_called stays None for
    EITHER language (gen0 empty-injection byte-identity preserved)."""
    for lang in ("python", "bash"):
        tr = _fresh_trace("echo hi")
        populate_adoption_fields(
            tr,
            command_count=1,
            merged_code_library={},
            merged_code_library_bash={},
            executor=SandboxMarker(),
            solver_language=lang,
        )
        assert tr.utilities_available == []
        assert tr.utilities_called is None
        assert tr.command_count == 1


def test_clean_python_helper_available_for_both_languages():
    """A blocklist-CLEAN helper is available regardless of solver_language (the
    fix only changes the sandbox-blocklisted case)."""
    clean = "def clean(x):\n    return x + 1"
    for lang in ("python", "bash"):
        tr = _fresh_trace("# --- end injected code library ---\nclean(1)\n")
        populate_adoption_fields(
            tr,
            command_count=1,
            merged_code_library={"clean": clean},
            merged_code_library_bash={},
            executor=SandboxMarker(),
            solver_language=lang,
        )
        assert tr.utilities_available == ["clean"]
        assert tr.utilities_called == ["clean"]
