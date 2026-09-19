"""External-agent integration package — the reusable spine and its public surface.

This package lets meta-n drive *self-contained* external agents (OpenHands,
Terminus 2) — and the framework's own builtin solvers — behind a single
solver-shaped facade, :class:`ExternalAgentSolver`, that satisfies meta-n's
solver contract (``execute(task) -> (Trace, int)`` and
``solve(task, additional_context="") -> (script, reasoning, int)``; see plan
§2.1, §2.3). Around that spine sit a small set of strategy/boundary
abstractions:

* :class:`AgentBackend` — the pluggable strategy that actually drives an agent
  (builtin / OpenHands / Terminus 2). Lives in :mod:`.backend` together with the
  pure-data DTOs (``Prompt``/``InjectionPlan``/``AgentRunContext``/
  ``AgentRunResult``).
* :class:`AgentEnvProvider` and :class:`Scorer` — the per-benchmark
  provisioning/staging/extraction and scoring boundaries (:mod:`.env`); concrete
  implementations live next to their benchmark adapter, not here.
* :class:`InjectionMapper` — maps a candidate's ``InjectedCode`` chain onto the
  agent's injection surfaces (prompt suffix/prefix + staged helper files;
  :mod:`.injection`).
* :class:`AgentTelemetry` — the version-stamped run ledger and ``Trace``
  builder (:mod:`.telemetry`).
* :class:`DockerRunGuard` — the inner concurrency semaphore + per-run host
  isolation and cleanup (:mod:`.concurrency`).
* :class:`CostGuard` — the thin adapter onto the one ``CostTracker`` ledger
  (:mod:`.budget`).
* :class:`TerminatedBy` — the shared termination-reason enum (:mod:`.terminated`).

Import-cleanliness contract
---------------------------
This package **imports cleanly without** ``openhands``, ``terminal_bench`` or
``docker`` installed. Every external SDK is imported lazily *inside* the concrete
backend/provider methods, never at module top level — and this ``__init__`` only
pulls in the dependency-free spine modules (waves 0–2). The concrete backends
under :mod:`.backends` (and the per-benchmark providers/scorers in the adapter
modules) are **not** imported here; importing them is the caller's choice, kept
lazy so a missing optional SDK can never break ``import
meta_n.core.external_agents``.

:class:`ExternalAgentSolver` lives in :mod:`.solver` (a later wave that may not
yet be present). It is therefore exposed through a PEP 562 module-level
:func:`__getattr__` so that this package keeps importing cleanly until that
module lands, while ``from meta_n.core.external_agents import ExternalAgentSolver``
resolves correctly once it does.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# Dependency-free spine modules (waves 0–2). These import only the standard
# library plus already-present meta-n utilities, so pulling them in here cannot
# fail for want of an optional external SDK (openhands / terminal_bench / docker).
from .backend import (
    AgentBackend,
    AgentRunContext,
    AgentRunResult,
    InjectionPlan,
    Prompt,
)
from .budget import CostGuard
from .concurrency import DockerRunGuard
from .env import AgentEnvProvider, EnvLease, Scorer
from .injection import InjectionMapper
from .telemetry import AgentRunRecord, AgentTelemetry, compute_run_id
from .terminated import TerminatedBy

if TYPE_CHECKING:  # pragma: no cover - typing only; avoids importing the wave-3 module
    from .solver import ExternalAgentSolver


__all__ = [
    # Spine facade (lazily resolved via __getattr__ until solver.py lands).
    "ExternalAgentSolver",
    # Strategy boundary + DTOs.
    "AgentBackend",
    "AgentRunContext",
    "AgentRunResult",
    "InjectionPlan",
    "Prompt",
    # Env / scoring interfaces.
    "AgentEnvProvider",
    "Scorer",
    "EnvLease",
    # Spine collaborators.
    "InjectionMapper",
    "AgentTelemetry",
    "AgentRunRecord",
    "DockerRunGuard",
    "CostGuard",
    "TerminatedBy",
    "compute_run_id",
]


def __getattr__(name: str) -> object:
    """Resolve the spine facade lazily (PEP 562).

    :class:`ExternalAgentSolver` is defined in :mod:`.solver`, a later wave that
    depends on every spine module and may not yet be present. Deferring its import
    to first access keeps ``import meta_n.core.external_agents`` working before
    that module exists, while still letting
    ``from meta_n.core.external_agents import ExternalAgentSolver`` succeed once it
    does. Any other unknown attribute raises the standard :class:`AttributeError`.
    """
    if name == "ExternalAgentSolver":
        from .solver import ExternalAgentSolver

        return ExternalAgentSolver
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Include the lazily-exported spine facade in :func:`dir` output."""
    return sorted(set(globals()) | set(__all__))
