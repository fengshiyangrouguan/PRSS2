"""Regression tests for audit finding #35 (builtin backend).

Finding #35: ``BuiltinBackend.solver_language`` is write-only — it is stored but
never read for routing, yet its constructor docstring falsely claimed it
"Selects the depth-1 ``solve()`` ... path vs the chain ``execute()`` path,
matching the orchestrator's dispatch". The actual single-shot vs chain/agentic
dispatch in ``_invoke_native`` branches solely on
``self.use_agentic or self.depth > 1``. The fix corrects the docstring to the
honest "metadata only" framing used by the sibling ``BuiltinTBBackend``.

These tests are LLM-free / offline: they only construct the backend and inspect
its docstring + routing behavior; no LM Studio, Docker, or network is touched.
"""

from __future__ import annotations

import asyncio

from meta_n.core.external_agents.backends.builtin import BuiltinBackend


class _RecordingSolver:
    """Minimal solver stub recording which native entrypoint was invoked."""

    def __init__(self) -> None:
        self.execute_called = False
        self.solve_called = False

    async def execute(self, task):  # noqa: ANN001
        self.execute_called = True

        class _Trace:
            success = True

        return _Trace(), 0

    async def solve(self, task):  # noqa: ANN001
        self.solve_called = True
        return "echo hi", "reasoning", 0


class _RecordingExecutor:
    async def execute(self, script, task):  # noqa: ANN001
        class _Trace:
            success = True

        return _Trace()


def _solver_language_doc_segment() -> str:
    """Return the ``solver_language`` slice of the constructor docstring."""
    doc = BuiltinBackend.__init__.__doc__ or ""
    # Fall back to the class docstring shape if needed; the Args block lives on
    # the class docstring for this backend.
    if "solver_language" not in doc:
        doc = BuiltinBackend.__doc__ or ""
    assert "solver_language" in doc, "solver_language must be documented"
    start = doc.index("solver_language")
    return doc[start:start + 600]


def test_solver_language_docstring_no_longer_claims_it_selects_routing():
    """The false 'Selects the depth-1 ... path vs the chain' claim must be gone."""
    segment = _solver_language_doc_segment()
    # The exact false phrasing from the un-fixed code must not survive.
    assert "Selects the\n            depth-1" not in segment
    assert "Selects the depth-1" not in segment.replace("\n", " ").replace(
        "            ", " "
    ).replace("  ", " ")
    # The honest framing must be present.
    assert "metadata only" in segment
    assert "not" in segment.lower() and "routing" in segment.lower()


def test_routing_ignores_solver_language_python_uses_single_shot():
    """depth-1 non-agentic routes via solve()+executor regardless of language."""
    for language in ("bash", "python"):
        solver = _RecordingSolver()
        backend = BuiltinBackend(
            solver=solver,
            llm_client=None,
            use_agentic=False,
            solver_language=language,
            executor=_RecordingExecutor(),
            depth=1,
        )

        class _Task:
            task_id = "t"

        asyncio.run(backend._invoke_native(_Task(), ctx=None))
        # Routing is independent of solver_language: depth-1 + non-agentic always
        # uses the single-shot solve() path, never execute().
        assert solver.solve_called is True
        assert solver.execute_called is False


def test_solver_language_is_stored_but_not_consulted_for_routing():
    """Attribute is preserved (kwarg parity) but does not flip the route."""
    solver = _RecordingSolver()
    backend = BuiltinBackend(
        solver=solver,
        use_agentic=True,
        solver_language="python",
        depth=1,
    )
    assert backend.solver_language == "python"

    class _Task:
        task_id = "t"

    # use_agentic=True forces the chain/agentic execute() route — solver_language
    # has no bearing on that decision.
    asyncio.run(backend._invoke_native(_Task(), ctx=None))
    assert solver.execute_called is True
    assert solver.solve_called is False
