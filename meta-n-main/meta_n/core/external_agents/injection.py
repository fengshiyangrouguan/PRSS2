"""Injection mapping — map both ``InjectedCode`` channels onto agent surfaces.

This is WAVE 2 of the external-agents integration (plan §3). :class:`InjectionMapper`
turns the list of ``InjectedCode`` blocks that Ω produced for a candidate chain into
an :class:`~meta_n.core.external_agents.backend.InjectionPlan` that the spine
(:class:`~meta_n.core.external_agents.solver.ExternalAgentSolver`) hands to a backend:

* The ``pre_process`` channel + library *descriptions* become the
  :class:`~meta_n.core.external_agents.backend.Prompt`'s ``system_suffix`` (OpenHands
  maps it to ``AgentContext.system_message_suffix``; Terminus 2 folds it into the
  leading instruction). The inter-layer ``additional_context`` becomes the ``prefix``.
* The ``code_library`` / ``code_library_bash`` channels become *staged files* under
  ``helpers/`` in the agent's workspace, with a Python re-export ``helpers/__init__.py``
  so the agent can ``from helpers import <name>``.

Reuse, not reimplementation
---------------------------
Every load-bearing transform is borrowed from :mod:`meta_n.core.meta_layer`:

* :func:`~meta_n.core.meta_layer.merge_code_libraries` — collapse the per-layer
  libraries into one dict each (later layers override earlier by name).
* :func:`~meta_n.core.meta_layer.run_pre_process` — run every layer's ``pre_process``
  block deepest-first, with ``validate_code`` rejection and the exec namespace seeded
  with ``task`` / ``additional_context`` / ``outer_context`` (the module-level helper
  shared with ``MetaLayer`` and ``AgenticSolver``).
* :func:`~meta_n.core.meta_layer.format_python_library_descriptions` /
  :func:`~meta_n.core.meta_layer.format_bash_library_descriptions` — render the helper
  signatures/docstrings for the prompt, routed by ``solver_language``.
* :func:`~meta_n.core.meta_layer.build_python_lib_file` — for *bash* callers, build
  the runnable ``_lib_<name>.py`` **file content** (helper source + a
  ``__main__`` CLI-dispatch tail) staged at ``helpers/_lib_<name>.py``, the path
  the bash descriptions reference. (Note: ``wrap_python_as_file`` is the *bash
  heredoc* wrapper used by the inline-script path; the spine stages a real Python
  file, not a heredoc.)

We deliberately do **not** reuse ``prepend_python_library`` / ``prepend_bash_library``:
those splice the library into a finished ``solve()`` script, a model the self-contained
agent bypasses. Helpers become files; the suffix tells the agent they exist (advisory
injection — see the "honest limitation" in plan §3).

Gen0 parity
-----------
An empty ``injected_codes`` yields an empty ``system_suffix``, no staged files, and
``pre_process_ran == False`` — i.e. a clean vanilla-agent baseline whose ``Prompt`` is
fully empty (asserted by the gen0 parity test, plan §12).

This module imports only :mod:`meta_n.core.meta_layer` utilities and the WAVE 1
``backend`` DTOs, so ``external_agents`` stays importable without ``openhands``,
``terminal_bench`` or ``docker`` installed (no external SDK touched here).
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from meta_n.core.meta_layer import (
    build_python_lib_file,
    format_bash_library_descriptions,
    format_python_library_descriptions,
    merge_code_libraries,
    run_pre_process,
)

from .backend import InjectionPlan, Prompt

if TYPE_CHECKING:  # pragma: no cover - typing only
    from meta_n.core.meta_layer import InjectedCode, SandboxMarker, TaskDescription


__all__ = ["InjectionMapper"]

#: Affordance appended to the system suffix whenever any helper is staged (G8).
#: Forensic log analysis (ab4) found agents NEVER called injected helpers
#: (``utilities_called: []``) under the old vague advisory note — no concrete
#: call syntax, no imperative. These give the agent the EXACT, language-correct
#: invocation and a "prefer these / they are tested" directive. Language-aware:
#: the staged-file model means the agent imports/runs files (NOT the inline
#: prepend the one-shot solver uses), so the call form differs by language.
_HELPERS_NOTE_PY = (
    "The helpers above are TESTED and staged at ./helpers/. PREFER calling them "
    "over re-implementing from scratch. Import directly: `from helpers import "
    "<name>` — the signatures above are the contract."
)
_HELPERS_NOTE_BASH = (
    "The helpers above are TESTED and staged at ./helpers/. PREFER calling them "
    "over re-implementing from scratch. Run a Python helper with "
    "`python3 ./helpers/_lib_<name>.py <args>` (result on stdout); source a bash "
    "helper from ./helpers/<name>.sh — the signatures above are the contract."
)


def _unsafe_helper_name(name: str) -> bool:
    """Whether a helper ``name`` would yield an unsafe ``helpers/<name>`` key.

    A legitimate name is a bare ``[A-Za-z0-9_]`` token (the only shape the Ω
    ``solver_lib:(\\w+)`` extractor can emit). Defense-in-depth: treat a name as
    unsafe if it is empty, contains a path separator, or contains a ``..``
    segment, so a future producer can never push a path-traversal key down to
    the staging writers.
    """
    if not name:
        return True
    if "/" in name or "\\" in name:
        return True
    if ".." in name:
        return True
    return False


def _binds_top_level_name(target: ast.expr, name: str) -> bool:
    """Whether an assignment ``target`` binds the bare top-level symbol ``name``."""
    if isinstance(target, ast.Name):
        return target.id == name
    if isinstance(target, (ast.Tuple, ast.List)):
        return any(_binds_top_level_name(elt, name) for elt in target.elts)
    if isinstance(target, ast.Starred):
        return _binds_top_level_name(target.value, name)
    return False


def _stmt_binds_module_name(stmt: ast.stmt, name: str) -> bool:
    """Whether a module-scope ``stmt`` binds the symbol ``name`` at module scope.

    Recurses into module-level *compound* statements (``try/except``, ``if``,
    ``with``, ``for``, ``while``) — none of which introduce a new scope, so a
    ``def`` / ``import`` / assignment inside them still binds at module scope and
    satisfies ``from .<name> import <name>``. It deliberately does NOT descend into
    ``def`` / ``async def`` / ``class`` bodies: those DO open a new scope, so a
    binding nested there is not importable from the module and must not be counted.
    """
    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return stmt.name == name
    if isinstance(stmt, ast.Assign):
        return any(_binds_top_level_name(t, name) for t in stmt.targets)
    if isinstance(stmt, ast.AnnAssign):
        return stmt.value is not None and _binds_top_level_name(stmt.target, name)
    if isinstance(stmt, (ast.Import, ast.ImportFrom)):
        return any(
            (alias.asname or alias.name.split(".")[0]) == name for alias in stmt.names
        )
    # Module-level compound statements keep module scope for their nested bodies.
    if isinstance(stmt, ast.Try):
        blocks = [stmt.body, stmt.orelse, stmt.finalbody]
        blocks += [handler.body for handler in stmt.handlers]
        return any(_stmt_binds_module_name(s, name) for block in blocks for s in block)
    if isinstance(stmt, ast.If):
        return any(_stmt_binds_module_name(s, name) for s in (*stmt.body, *stmt.orelse))
    if isinstance(stmt, (ast.With, ast.AsyncWith)):
        return any(_stmt_binds_module_name(s, name) for s in stmt.body)
    if isinstance(stmt, (ast.For, ast.AsyncFor)):
        return any(_stmt_binds_module_name(s, name) for s in (*stmt.body, *stmt.orelse))
    if isinstance(stmt, ast.While):
        return any(_stmt_binds_module_name(s, name) for s in (*stmt.body, *stmt.orelse))
    return False


def _importable_helper(name: str, source: str) -> bool:
    """Whether ``helpers/<name>.py`` can satisfy ``from .<name> import <name>``.

    The generated ``helpers/__init__.py`` re-export (see :meth:`_reexport`) emits
    an unconditional ``from .<name> import <name>`` for *every* staged Python
    helper, so importing the package executes that line for each helper in turn.
    If a single helper's source has a syntax error, or it parses but never binds
    a *module-scope* symbol named ``<name>`` (e.g. an Ω ``solver_lib:foo`` block
    whose body is ``def bar(): ...`` or just ``X = 1``), that one line raises
    ``SyntaxError`` / ``ImportError`` and poisons ``import helpers`` wholesale —
    silently zeroing helper adoption across the whole candidate. Gate staging on
    this predicate so one bad helper cannot break the import of the rest, mirroring
    the native staging path (``prepend_python_library`` skips helpers that fail
    ``validate_library_function`` — meta_layer.py).

    Module scope, not just the top-level body: a helper that binds ``<name>``
    inside a module-level compound statement — the common ``try: import numpy;
    def foo(): ... except ImportError: def foo(): ...`` import-fallback pattern, or
    an ``if``/``with``/``for``/``while`` guard — imports perfectly, so it must be
    recognised. Restricting to ``tree.body`` (the pre-fix behaviour) wrongly dropped
    such helpers from staging *while the advertising gate still promoted them*
    (``format_python_library_descriptions`` walks the whole tree), yielding an
    advertised-but-unstaged helper whose ``from helpers import <name>`` raises
    ImportError. :meth:`build` additionally advertises only from the staged
    subset, so staging can never be a strict subset of what is advertised.

    Pure static AST analysis — the (Ω-generated) source is never executed here.
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        # Beyond SyntaxError, ``ast.parse`` can raise ValueError (NUL bytes on
        # older Pythons), MemoryError (pathological expression nesting) and
        # RecursionError on adversarial Ω output — all mean "not stageable",
        # never "crash the plan build".
        return False
    return any(_stmt_binds_module_name(stmt, name) for stmt in tree.body)


class InjectionMapper:
    """Map a candidate's ``InjectedCode`` chain onto agent injection surfaces.

    Construction merges the libraries once (deepest layer wins on name clashes);
    :meth:`build` then assembles the per-task :class:`InjectionPlan` — running
    ``pre_process``, rendering library descriptions for the active language, and
    staging the helper files.

    Args:
        injected_codes: The candidate's ``InjectedCode`` blocks, ordered shallowest
            (outermost layer) → deepest. ``run_pre_process`` iterates this list
            deepest-first internally; an empty list is the gen0 baseline.
        solver_language: ``"python"`` (default) or ``"bash"``. Selects which
            description formatter runs and whether Python helpers are *also* staged
            as runnable ``helpers/_lib_<name>.py`` files (the workspace-relative
            ``./helpers/_lib_<name>.py`` form bash callers reference).
        sandbox_marker: A lightweight :class:`~meta_n.core.meta_layer.SandboxMarker`
            (``is_sandboxed=True``) passed to the description formatters so they skip
            host-side smoke tests / blocklist filtering — the agent runs helpers in
            its *own* sandbox (plan §3 FIX, minor #5).
    """

    def __init__(
        self,
        injected_codes: "list[InjectedCode]",
        solver_language: str,
        sandbox_marker: "SandboxMarker",
    ) -> None:
        # Merge once (later layers override earlier by name within each channel).
        self.merged_py, self.merged_bash = merge_code_libraries(injected_codes)
        self.injected_codes = injected_codes
        self.lang = solver_language
        self.sbx = sandbox_marker

    def build(self, task: "TaskDescription", additional_context: str = "") -> InjectionPlan:
        """Assemble the :class:`InjectionPlan` for one task.

        Runs the chain's ``pre_process`` blocks, renders library descriptions for the
        active language, and stages the helper files. The returned plan's ``prompt``
        carries the ``pre_process`` output + descriptions in ``system_suffix`` and the
        inter-layer ``additional_context`` in ``prefix``.

        Args:
            task: The task being solved (handed to ``pre_process`` blocks as ``task``).
            additional_context: Inter-layer guidance from a higher layer. Becomes the
                ``Prompt.prefix``; ``pre_process`` runs with ``outer_context=""`` to
                match the spine's ``execute()`` entry point (the re-solve ``solve()``
                path passes its own context as the prefix, plan §2.3).

        Returns:
            An :class:`InjectionPlan`. For empty ``injected_codes`` (and empty
            ``additional_context``) this is the gen0 vanilla baseline: empty
            ``system_suffix``, empty ``prefix``, no staged files, ``pre_process_ran``
            ``False``, ``utilities_available`` ``[]``.
        """
        # --- pre_process channel (deepest-first, validate_code rejection) ---
        ran, ctx = run_pre_process(
            self.injected_codes, task, additional_context, outer_context=""
        )

        # --- staging gate (single source of truth for advertise + stage + count) ---
        # Compute the actually-stageable subset BEFORE rendering descriptions so the
        # advertising channel can never promote a helper that staging drops. A safe
        # Python name is a bare ``[A-Za-z0-9_]`` token whose source binds ``<name>``
        # at module scope (so the ``helpers/__init__.py`` re-export imports cleanly);
        # a safe bash name only needs the traversal-key guard. ``safe_py`` / ``safe_bash``
        # then drive (a) the description formatters, (b) the staged files, and (c) the
        # ``utilities_available`` adoption denominator — the three sibling channels that
        # must agree (backend.py:90 documents ``utilities_available`` as "every STAGED
        # helper"). See ``_importable_helper`` / ``_unsafe_helper_name``.
        safe_py = {
            n: s
            for n, s in self.merged_py.items()
            if not _unsafe_helper_name(n) and _importable_helper(n, s)
        }
        safe_bash = {
            n: s for n, s in self.merged_bash.items() if not _unsafe_helper_name(n)
        }

        # --- library descriptions (routed by language; sandbox-marked) ---
        # Advertise ONLY from the staged subset (``safe_py`` / ``safe_bash``): a helper
        # dropped by the staging gate must never appear in the prompt with a
        # ``from helpers import <name>`` / ``python3 ./helpers/_lib_<name>.py`` note it
        # cannot satisfy. The formatters apply their own further filtering (advertising
        # blocklist for python, FunctionDef presence), so advertised ⊆ staged always.
        if self.lang == "bash":
            # The spine stages real Python helpers at the workspace-relative
            # ``helpers/_lib_<name>.py`` (not /tmp), so the bash descriptions must
            # point there for the path to resolve at runtime.
            desc = format_bash_library_descriptions(
                safe_py,
                safe_bash,
                self.sbx,
                lib_path_fmt="./helpers/_lib_{name}.py",
            )
        else:
            desc = format_python_library_descriptions(safe_py, self.sbx)
        # Gate the helpers-note on whether descriptions were ACTUALLY produced
        # (``desc``), not on the raw pre-validation merged dicts. The formatters
        # return "" when every helper is dropped by ``validate_library_function``
        # (syntax error / safety-blocklist flag), so ``self.merged_py`` can be
        # non-empty while ``desc`` is empty — appending the note then would set the
        # suffix to a bare "…the signatures above are the contract" referencing
        # signatures that were never emitted.
        if desc:
            note = _HELPERS_NOTE_BASH if self.lang == "bash" else _HELPERS_NOTE_PY
            desc = f"{desc}\n{note}"

        # System suffix = pre_process context + descriptions, skipping empty parts.
        suffix = "\n\n".join(part for part in (ctx, desc) if part)

        # --- code_library channels -> staged helper files under helpers/ ---
        # ``safe_py`` / ``safe_bash`` were computed above (the single staging gate):
        # names are a bare ``[A-Za-z0-9_]`` token with no ``..``/path-separator, and
        # every Python source binds ``<name>`` at module scope so the
        # ``helpers/__init__.py`` re-export imports cleanly. ``__init__`` is
        # constructed below, not a merged name, so it is safe.
        staged: dict[str, str] = {}
        for name, source in safe_py.items():
            staged[f"helpers/{name}.py"] = source
        if safe_py:
            # Python re-export so the agent can `from helpers import <name>`.
            staged["helpers/__init__.py"] = self._reexport(safe_py)
            if self.lang == "bash":
                # Bash callers reference `python3 ./helpers/_lib_<name>.py`; stage
                # the *real, runnable Python file* (helper source + __main__ CLI
                # dispatch) so those descriptions resolve. build_python_lib_file
                # returns FILE CONTENT — NOT the bash heredoc wrap_python_as_file
                # emits (which would land literal bash inside a .py and never run).
                for name, source in safe_py.items():
                    staged[f"helpers/_lib_{name}.py"] = build_python_lib_file(name, source)
        for name, source in safe_bash.items():
            staged[f"helpers/{name}.sh"] = source

        return InjectionPlan(
            prompt=Prompt(system_suffix=suffix, prefix=additional_context),
            staged_files=staged,
            # Denominator for the helper-adoption estimator: every helper the agent
            # could actually invoke == the STAGED set (backend.py:90), NOT the raw
            # merged dicts. Since the #20 staging filter, ``safe_*`` can be a strict
            # subset of ``merged_*`` (unsafe name / non-importable source); counting
            # never-staged helpers here inflated the denominator and deflated adoption.
            utilities_available=sorted([*safe_py, *safe_bash]),
            pre_process_ran=ran,
        )

    @staticmethod
    def _reexport(merged_py: dict[str, str]) -> str:
        """Build a ``helpers/__init__.py`` that re-exports every Python helper.

        Each helper lives in its own ``helpers/<name>.py`` module defining the
        function ``<name>``; the package ``__init__`` re-imports them by name so the
        agent can ``from helpers import <name>`` (or ``import helpers``). ``__all__``
        is emitted in sorted order for deterministic, reproducible output.

        Args:
            merged_py: The merged Python ``code_library`` (name -> source).

        Returns:
            The source of the ``helpers/__init__.py`` re-export module.
        """
        names = sorted(merged_py)
        lines = [
            '"""Auto-generated re-export of injected Python helpers.',
            "",
            "Each helper is defined in its sibling ``helpers/<name>.py`` module and",
            "re-exported here so it can be imported as ``from helpers import <name>``.",
            '"""',
            "",
        ]
        lines += [f"from .{name} import {name}" for name in names]
        lines.append("")
        all_items = ", ".join(f'"{name}"' for name in names)
        lines.append(f"__all__ = [{all_items}]")
        lines.append("")
        return "\n".join(lines)
