"""Is a solver's emitted program actually a program? Checked before execution.

Why this module exists (measured 2026-09-21): the extraction helpers in
``solver.py`` fall back to ``response.strip()`` when no fenced block matches, so
a completion truncated mid-program came back as a 172-character stub. The only
guard downstream was ``if not script.strip()``, which catches an empty string and
nothing else -- so every stub was handed to the executor as if it were a
solution, and its eventual ``SyntaxError`` was recorded as a *task difficulty*
failure rather than as a broken model call.

Two independent signals, both required:

* transport-level -- ``finish_reason == 'length'`` means the model was cut off
  by its token budget, so whatever it emitted is incomplete by construction.
  This is language-agnostic and applies to every spine.
* content-level -- for Python the program must parse AND define the entry point
  the harness will actually call.

Scope is deliberately narrow: only ``solution_language == 'python'`` gets the
syntax + entry-point treatment. CO-Bench -- the cohort every SRI number comes
from -- is python, and its runner indexes ``ns["solve"]`` (co_bench.py:156), so
"parses + defines a top-level ``solve``" is exactly its contract. The bash and
classify spines keep the previous non-empty test: their existing results were
produced without this check and no measurement covers changing them.
"""
import ast
from typing import Optional, Tuple

#: The function CO-Bench's runner invokes: ``solve_fn = ns["solve"]`` then
#: ``solve_fn(**instance)``. A script that parses but never defines this name
#: fails at load with a KeyError, which is a model failure, not a task failure.
PYTHON_ENTRY_POINT = "solve"


def validate_python_solve(script: str) -> Tuple[bool, str]:
    """(ok, reason). ``reason`` is a short stable slug, empty when ok.

    The slug is deliberately machine-readable -- it is written into the trace's
    ``error_summary`` and counted, so it must not be prose.
    """
    if not script or not script.strip():
        return False, "empty"

    try:
        tree = ast.parse(script)
    except SyntaxError as e:
        return False, "syntax_error: {}".format(e)
    except ValueError as e:                     # e.g. source with NUL bytes
        return False, "unparseable: {}".format(e)

    defined = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    if PYTHON_ENTRY_POINT not in defined:
        return False, "missing_entry_point:{}".format(PYTHON_ENTRY_POINT)

    return True, ""


def validate_solver_output(script: str, language: str,
                           finish_reason: Optional[str] = None
                           ) -> Tuple[bool, str]:
    """(ok, reason) for a solver script in ``language``.

    ``finish_reason`` is the value the LLM client reported for the call that
    produced ``script`` (see ``LLMClient.complete(_meta=...)``). ``None`` means
    the caller did not collect it -- treated as unknown, never as truncated.
    """
    if finish_reason == "length":
        # Checked first and for every language: a truncated program is not a
        # weaker solution, it is an incomplete one, and reporting it as a low
        # score would attribute a provider budget failure to task difficulty.
        return False, "truncated:finish_reason=length"

    if language == "python":
        return validate_python_solve(script)

    return (True, "") if (script or "").strip() else (False, "empty")
