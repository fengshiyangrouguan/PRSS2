"""Shared external-spine routing predicate.

CONTRACT: main.py's external-only startup guards and
``EvolutionaryOrchestrator._uses_external_spine`` must agree on ONE routing
rule; both call :func:`uses_external_spine`. Kept import-light (``logging``
only) so main.py can import it at module scope.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def uses_external_spine(base_solver: str | None, adapter: object | None) -> bool:
    """Whether a run drives the external-agent spine rather than the legacy
    native path.

    The spine is used when ``base_solver`` is a genuine external kind
    (``openhands`` / ``terminus2`` — anything other than ``None`` / ``builtin``),
    OR when ``base_solver == "builtin"`` AND the bound adapter advertises that it
    wants ``builtin`` on the spine (``advertises_spine_builtin()`` — only the
    terminal_bench adapter does, where the native executor cannot run the legacy
    task layout). ``base_solver in (None, "builtin")`` with a NON-advertising
    adapter (CO-Bench, text-classification, …) stays on the legacy native path,
    byte-for-byte unchanged — so CO-Bench's ``builtin`` baseline never regresses.
    ``adapter=None`` (the ``--tasks`` path) likewise stays legacy.

    Returns:
        ``True`` if the spine + ``ExternalAgentSolver`` should drive the run.
    """
    if base_solver not in (None, "builtin"):
        return True
    if base_solver == "builtin" and adapter is not None:
        advertise = getattr(adapter, "advertises_spine_builtin", None)
        if not callable(advertise):
            return False
        try:
            # Strict ``is True`` (not merely truthy): the contract is a bool,
            # and a MagicMock adapter (unit tests with a mocked executor whose
            # ``.adapter`` auto-mocks) would return a truthy MagicMock — which
            # must NOT silently promote builtin to the spine and stand up the
            # cost guard. Only a real adapter returning the literal ``True``
            # routes builtin through the spine.
            return advertise() is True
        except Exception:  # noqa: BLE001 - a faulty capability check → legacy
            logger.warning(
                "adapter.advertises_spine_builtin() raised; keeping builtin on "
                "the legacy native path",
                exc_info=True,
            )
            return False
    return False
