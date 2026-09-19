"""Code-library validation, advertising and staging utilities (split from meta_layer.py).

F257: relocated verbatim from ``meta_n.core.meta_layer`` — the functions here
feed the solver prompt (pinned by tests/golden/omega_prompt_golden.txt and the
wave-1 pinning tests), so no advertised string may change. ``meta_layer.py``
remains the import hub and re-exports every name defined here.
"""

from __future__ import annotations

import ast
import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from meta_n.core.base_executor import BaseExecutor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Standalone library utilities (used by both MetaLayer and AgenticSolver)
# ---------------------------------------------------------------------------


class SandboxMarker:
    """Lightweight stand-in for a sandboxed ``BaseExecutor``.

    The ``format_*_library_descriptions`` / ``validate_library_function``
    helpers gate host-side smoke tests and blocklist filtering on
    ``executor.is_sandboxed`` (see :func:`validate_library_function`). External
    agents run helper code inside their *own* sandbox, so passing a
    ``SandboxMarker`` makes those helpers treat the environment as sandboxed —
    descriptions are produced from a syntax check only, without running
    host-side validation that could spuriously drop otherwise-valid helpers.
    """

    is_sandboxed: bool = True


def validate_library_function(
    name: str, source: str, *, is_bash: bool = False,
    executor: BaseExecutor | None = None, for_advertising: bool = False,
) -> bool:
    """Validate a library function via static analysis and smoke test.

    Args:
        name: Library function name (must match defined function name).
        source: Function source code.
        is_bash: If True, skip all Python validation (it's bash code).
        executor: If sandboxed, skip safety blocklist but still check syntax.
        for_advertising: When True under a SANDBOXED executor, also apply the
            safety blocklist, so a parseable-but-flagged helper is not ADVERTISED
            to the solver (it may still be prepended and run inside the sandbox).
    """
    if is_bash:
        return True

    if executor and hasattr(executor, "is_sandboxed") and executor.is_sandboxed:
        try:
            ast.parse(source)
        except SyntaxError as e:
            logger.warning("Library '%s' has syntax error: %s — skipping", name, e)
            return False
        # (c) N8: in the sandbox we still PREPEND parseable helpers (the sandbox
        # contains them), but do NOT ADVERTISE one that trips the safety
        # blocklist — advertising tells the model to call it, and a flagged
        # helper should not be promoted even if it runs.
        if for_advertising:
            from meta_n.utils.safety import validate_code
            ok, reason = validate_code(source)
            if not ok:
                logger.debug(
                    "Library '%s' not advertised (sandboxed, flagged): %s", name, reason
                )
                return False
        return True

    # Full validation for non-sandboxed executors
    from meta_n.utils.safety import smoke_test_function, validate_code

    is_valid, error = validate_code(source)
    if not is_valid:
        logger.warning("Library '%s' failed validation: %s — skipping", name, error)
        return False

    passed, smoke_error = smoke_test_function(name, source)
    if not passed:
        logger.warning("Library '%s' failed smoke test: %s — skipping", name, smoke_error)
        return False

    return True


def _format_wired_skeleton_python(names: list[str]) -> str:
    """Build a wired ``solve()`` skeleton that already calls each helper by bare name.

    Mechanism-0 (foster_adoption=True only): showing a pre-wired skeleton makes
    "fill in around the calls" the path of least resistance, so the solver
    delegates the core logic to the helpers instead of re-deriving it inline.
    Benchmark-agnostic: the solver keeps its task-required ``solve()`` signature.
    """
    if not names:
        return ""
    calls = "\n".join(
        f"    _ = {n}(...)   # call the provided helper; do NOT re-implement it"
        for n in names
    )
    return (
        "\n## Wired solve() skeleton "
        "(build your solution AROUND these calls — keep them)\n"
        "```python\n"
        "def solve(*args, **kwargs):\n"
        "    # Keep the exact solve() signature required by the task above.\n"
        "    # Delegate the core work to the helpers below — fill in the arguments:\n"
        f"{calls}\n"
        "    ...\n"
        "    return ...  # return in the format the task requires\n"
        "```\n"
    )


def format_python_library_descriptions(
    code_library: dict[str, str], executor: BaseExecutor | None = None,
    *, foster_adoption: bool = False,
) -> str:
    """Format Python code_library function signatures for the solver prompt.

    When ``foster_adoption`` is True (mechanism-0), the advertised text REQUIRES
    the solver to call the helpers by name and includes a wired ``solve()``
    skeleton. Default False is byte-identical to the legacy output.
    """
    if not code_library:
        return ""

    from meta_n.core.prompts import LIBRARY_INSTRUCTIONS

    descriptions = []
    valid_names: list[str] = []
    for name, source in sorted(code_library.items()):
        if not validate_library_function(name, source, executor=executor, for_advertising=True):
            continue
        try:
            tree = ast.parse(source)
            for node in ast.walk(tree):
                # Async helpers are staged/extracted like sync ones (N8), so
                # advertise them with their real signature too.
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    sig_line = source.splitlines()[node.lineno - 1].strip()
                    docstring = ast.get_docstring(node)
                    desc = f"- {sig_line}"
                    if docstring:
                        lines = docstring.strip().splitlines()
                        desc += f'\n    """{lines[0]}'
                        for line in lines[1:]:
                            desc += f"\n    {line}"
                        desc += '\n    """'
                    descriptions.append(desc)
                    valid_names.append(name)
                    break
            else:
                descriptions.append(f"- {name}()")
                valid_names.append(name)
        except SyntaxError:
            continue

    if not descriptions:
        return ""

    if foster_adoption:
        from meta_n.core.prompts import LIBRARY_INSTRUCTIONS_FOSTER

        return (
            "\n## Available Helper Functions "
            "(REQUIRED — already implemented at runtime; you MUST call these, "
            "do NOT re-implement them)\n"
            + "\n".join(descriptions)
            + "\n"
            + _format_wired_skeleton_python(valid_names)
            + LIBRARY_INSTRUCTIONS_FOSTER
        )

    return (
        "\n## Available Helper Functions "
        "(already implemented and available at runtime — just call them)\n"
        + "\n".join(descriptions)
        + "\n"
        + LIBRARY_INSTRUCTIONS
    )


def format_bash_library_descriptions(
    code_library: dict[str, str],
    code_library_bash: dict[str, str],
    executor: BaseExecutor | None = None,
    *,
    lib_path_fmt: str = "/tmp/_lib_{name}.py",
    foster_adoption: bool = False,
) -> str:
    """Format library descriptions for bash solvers (both bash functions and Python scripts).

    Args:
        code_library: Python helpers (name -> source); described as
            ``python3 <lib_path_fmt> <args>`` runnable scripts.
        code_library_bash: Bash helpers (name -> source); described as directly
            callable functions.
        executor: Optional executor used to validate Python helpers.
        lib_path_fmt: Format string for the staged Python helper path, with a
            ``{name}`` placeholder. Defaults to the legacy native inline-script
            path ``/tmp/_lib_{name}.py`` (where ``prepend_bash_library`` writes the
            heredoc). The external-agent spine passes ``./helpers/_lib_{name}.py``
            because it stages real Python files at the workspace-relative
            ``helpers/`` path, so the description points at the file that actually
            exists.
    """
    if not code_library and not code_library_bash:
        return ""

    descriptions = []

    for name, source in sorted(code_library_bash.items()):
        preview = source.strip().split("\n")[0]
        descriptions.append(f"- {preview}  (bash function, call directly)")

    for name, source in sorted(code_library.items()):
        # NOTE (audit #66 — intentionally NOT gated on for_advertising): the bash
        # formatter deliberately does NOT pass for_advertising=True. A bash ``_lib_``
        # helper is advertised as a CLI form (``python3 ./helpers/_lib_<name>.py``)
        # and legitimately needs ``import sys`` for argv in its ``__main__`` tail —
        # which the host safety blocklist flags. Re-applying that blocklist here
        # (for_advertising=True under a sandboxed executor) would suppress EVERY
        # bash _lib_ helper (they all import sys), silently killing the entire bash
        # code-library channel. The python/bash advertising asymmetry is by design:
        # python agents call helpers as functions (sys irrelevant), bash agents run
        # them via the sys-using CLI form. See tests/experiments/test_tb2_adoption_smoke.
        if not validate_library_function(name, source, executor=executor):
            continue
        try:
            tree = ast.parse(source)
            for node in ast.walk(tree):
                # Include async helpers: they are staged by prepend_bash_library
                # and counted in utilities_available, so they must be shown too
                # (shown == staged == available).
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    doc = ast.get_docstring(node) or ""
                    first_line = doc.split("\n")[0] if doc else ""
                    lib_path = lib_path_fmt.format(name=name)
                    descriptions.append(
                        f"- python3 {lib_path} <args>  — {first_line}"
                    )
                    break
        except SyntaxError:
            continue

    if not descriptions:
        return ""

    if foster_adoption:
        from meta_n.core.prompts import LIBRARY_INSTRUCTIONS_BASH_FOSTER

        return (
            "\n## Available Helper Code "
            "(REQUIRED — already installed at runtime; you MUST call these)\n"
            + "\n".join(descriptions)
            + "\n"
            + LIBRARY_INSTRUCTIONS_BASH_FOSTER
        )

    from meta_n.core.prompts import LIBRARY_INSTRUCTIONS_BASH

    return (
        "\n## Available Helper Code "
        "(already installed at runtime)\n"
        + "\n".join(descriptions)
        + "\n"
        + LIBRARY_INSTRUCTIONS_BASH
    )


def build_python_lib_file(name: str, source: str) -> str:
    """Build the runnable ``/tmp/_lib_<name>.py`` **file content** (not a heredoc).

    Returns the helper's source plus the ``if __name__ == "__main__"`` CLI
    dispatch tail so the file can be invoked directly as
    ``python3 /tmp/_lib_<name>.py <args>`` (or staged verbatim as
    ``helpers/_lib_<name>.py``). This is the single source of truth for the
    wrapper body — :func:`wrap_python_as_file` wraps this content in a bash
    heredoc for the inline-script path, while the spine's
    :class:`~meta_n.core.external_agents.injection.InjectionMapper` stages this
    content directly as a real, importable/executable Python file.

    Args:
        name: The helper function name (also the callable dispatched in ``main``).
        source: The helper's Python source (defining ``<name>``).

    Returns:
        Runnable Python source for the ``_lib_<name>.py`` file.
    """
    return (
        f"{source}\n"
        f"\n"
        f'if __name__ == "__main__":\n'
        f"    import sys, json\n"
        f"    result = {name}(*sys.argv[1:])\n"
        f"    if result is not None:\n"
        f'        print(json.dumps(result) if not isinstance(result, str) else result)\n'
    )


def wrap_python_as_file(name: str, source: str) -> str:
    """Wrap a Python function as a bash heredoc that writes /tmp/_lib_<name>.py.

    The heredoc body is the runnable file content from
    :func:`build_python_lib_file`, so the inline-script path and the staged-file
    path emit byte-identical Python.
    """
    file_content = build_python_lib_file(name, source)
    return (
        f"cat << 'PYTHON_LIB_{name.upper()}_EOF' > /tmp/_lib_{name}.py\n"
        f"{file_content}"
        f"PYTHON_LIB_{name.upper()}_EOF"
    )


def prepend_python_library(
    script: str, code_library: dict[str, str], executor: BaseExecutor | None = None,
) -> str:
    """Prepend Python code library directly to script."""
    if not code_library:
        return script

    lib_parts = ["# --- injected code library ---"]
    valid_names: list[str] = []
    for name, source in sorted(code_library.items()):
        if not validate_library_function(name, source, executor=executor):
            continue
        lib_parts.append(f"# lib:{name}")
        lib_parts.append(source)
        valid_names.append(name)
    # (a) 4.5: the Ω prompt advertises helpers as BARE names, but models often
    # call them as ``solver_lib.<name>`` — a self-contradiction that NameErrors
    # at runtime. The helpers are prepended as bare defs, so rewrite the dotted
    # call form to bare in the solver body, and add a ``solver_lib`` namespace
    # shim (belt-and-suspenders for any dynamic ``getattr(solver_lib, ...)``).
    if valid_names:
        shim = (
            "import types as _types\nsolver_lib = _types.SimpleNamespace("
            + ", ".join(f"{n}={n}" for n in valid_names) + ")"
        )
        lib_parts.append(shim)
    lib_parts.append("# --- end injected code library ---\n")

    library_block = "\n\n".join(lib_parts)
    script = re.sub(r"\bsolver_lib\.(\w+)", r"\1", script)
    return f"{library_block}\n{script}"


def prepend_bash_library(
    script: str,
    code_library: dict[str, str],
    code_library_bash: dict[str, str],
    executor: BaseExecutor | None = None,
) -> str:
    """Prepend bash functions directly + write Python functions as /tmp files via heredoc."""
    if not code_library and not code_library_bash:
        return script

    parts = ["# --- injected code library ---"]

    for name, source in sorted(code_library_bash.items()):
        parts.append(f"# lib_bash:{name}")
        parts.append(source)

    for name, source in sorted(code_library.items()):
        if not validate_library_function(name, source, executor=executor):
            continue
        parts.append(f"# lib_py:{name} -> /tmp/_lib_{name}.py")
        parts.append(wrap_python_as_file(name, source))

    parts.append("# --- end injected code library ---\n")
    library_block = "\n\n".join(parts)
    return f"{library_block}\n{script}"
