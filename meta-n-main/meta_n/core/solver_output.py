"""Is a solver's emitted program actually a program? Classify BEFORE execution.

Why this module exists (measured 2026-09-21): the extraction helpers in
``solver.py`` fall back to ``response.strip()`` when no fenced block matches, so
a completion truncated mid-program came back as a 172-character stub, and the
only guard downstream was ``if not script.strip()``. Every such stub was handed
to the executor as if it were a solution and its eventual SyntaxError was
recorded as *task difficulty*.

The second half of the problem is classification. A response whose
``finish_reason`` is ``tool_calls`` carries its payload in
``message.tool_calls``, not in ``content`` -- reading only ``content`` made it
look like a plain empty completion. "Empty" and "the provider returned a
structured tool call we never asked for" are different events with different
fixes, and pooling them is what let this hide.

So the contract is one code per outcome, taken from a fixed vocabulary:

    BUDGET_EXHAUSTED     finish_reason='length' -- the model was cut off
    UNEXPECTED_TOOL_CALL a structured tool call; `content` is not the answer
    MODEL_REFUSAL        the provider reported a refusal
    EMPTY_COMPLETION     nothing usable in `content`, no other cause
    INVALID_SYNTAX       Python that does not parse
    MISSING_ENTRYPOINT   Python that parses but defines no top-level solve()
    VALID                a complete, callable program
    TRANSPORT_ERROR      (never returned here; the call raised -- see callers)

Scope is deliberately narrow: only ``solution_language == 'python'`` gets the
syntax + entry-point treatment. CO-Bench -- the cohort every SRI number comes
from -- is python, and its runner indexes ``ns["solve"]`` (co_bench.py:156), so
"parses + defines a top-level solve" is exactly its contract. The bash and
classify spines keep the previous non-empty test: their existing results were
produced without this check and no measurement covers changing them.
"""
import ast
from typing import Optional, Sequence, Tuple

#: The function CO-Bench's runner invokes: ``solve_fn = ns["solve"]`` then
#: ``solve_fn(**instance)``. A script that parses but never defines this name
#: fails at load with a KeyError -- a model failure, not a task failure.
PYTHON_ENTRY_POINT = "solve"

#: The closed vocabulary. Anything outside it is a bug in the classifier.
FAILURE_CODES = (
    "BUDGET_EXHAUSTED",
    "UNEXPECTED_TOOL_CALL",
    "MODEL_REFUSAL",
    "EMPTY_COMPLETION",
    "INVALID_SYNTAX",
    "MISSING_ENTRYPOINT",
    "TRANSPORT_ERROR",
    "VALID",
)


def validate_python_solve(script: str) -> Tuple[bool, str]:
    """(ok, code) for a Python solver body. ``code`` is VALID when ok."""
    if not script or not script.strip():
        return False, "EMPTY_COMPLETION"

    try:
        tree = ast.parse(script)
    except SyntaxError as e:
        return False, "INVALID_SYNTAX: {}".format(e)
    except ValueError as e:                     # e.g. source with NUL bytes
        return False, "INVALID_SYNTAX: {}".format(e)

    defined = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    if PYTHON_ENTRY_POINT not in defined:
        return False, "MISSING_ENTRYPOINT:{}".format(PYTHON_ENTRY_POINT)

    return True, "VALID"


def validate_solver_output(
    script: str,
    language: str,
    finish_reason: Optional[str] = None,
    tool_calls: Optional[Sequence[dict]] = None,
    refusal: Optional[str] = None,
) -> Tuple[bool, str]:
    """(ok, code) for a solver script.

    Args:
        script: what the extractor pulled out of the response.
        language: the task's ``solution_language``.
        finish_reason: the value the LLM client reported (via ``_meta``).
        tool_calls: the message's ``tool_calls`` payload, when any.
        refusal: the message's ``refusal`` field, when set.

    Every optional argument defaults to "the caller did not collect it", never
    to a guess -- a missing signal must not be read as a present one.
    """
    # Most specific first. A tool call explains an empty `content`, so testing
    # for emptiness before this would mislabel it.
    if tool_calls or finish_reason == "tool_calls":
        return False, "UNEXPECTED_TOOL_CALL"
    if refusal:
        return False, "MODEL_REFUSAL"
    if finish_reason == "length":
        # Language-agnostic: a truncated program is incomplete by construction,
        # and reporting it as a low score attributes a provider budget failure
        # to task difficulty.
        return False, "BUDGET_EXHAUSTED"

    if language == "python":
        return validate_python_solve(script)

    return (True, "VALID") if (script or "").strip() else (False,
                                                          "EMPTY_COMPLETION")
