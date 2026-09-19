"""Concrete :class:`~meta_n.core.external_agents.backend.AgentBackend` strategies.

This subpackage hosts the pluggable agent strategies the spine
(:class:`~meta_n.core.external_agents.solver.ExternalAgentSolver`) and the
per-benchmark adapters can drive:

* :class:`BuiltinBackend` (``backends.builtin``) — wraps meta-n's native
  ``Layer1Solver`` / ``AgenticSolver``; pure-meta-n, no external SDK.
* :class:`OpenHandsBackend` (``backends.openhands``) — drives an OpenHands agent
  on a host workspace; imports ``openhands`` lazily *inside* its methods.
* :class:`Terminus2Backend` / :class:`OpenHandsTBBackend` /
  :class:`BuiltinTBBackend` (``backends.{terminus2,openhands_tb,builtin_tb}``) —
  the terminal-bench backend family, all subprocess-bridge children of
  ``backends._external_tb._ExternalTBBackend``.

Import discipline
-----------------
Importing this subpackage MUST be side-effect free with respect to the heavy /
optional third-party SDKs: ``import meta_n.core.external_agents.backends`` (or the
parent ``external_agents`` package) must succeed on a machine where neither
``openhands``, ``terminal_bench``, nor ``docker`` is installed. The concrete
backend modules import their SDKs lazily *inside* their methods, and this package
``__init__`` imports **no** backend module at import time — the five backend
names resolve lazily on first attribute access (PEP 562, mirroring the parent
package), so nothing is imported until a caller actually asks for a backend.
Because every backend module is itself import-clean without the SDKs, the
``ModuleNotFoundError`` for a missing optional SDK still only surfaces when a
caller actually constructs/runs that specific backend.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported eagerly
    from ..backend import AgentBackend  # noqa: F401
    from .builtin import BuiltinBackend  # noqa: F401
    from .builtin_tb import BuiltinTBBackend  # noqa: F401
    from .openhands import OpenHandsBackend  # noqa: F401
    from .openhands_tb import OpenHandsTBBackend  # noqa: F401
    from .terminus2 import Terminus2Backend  # noqa: F401

__all__ = [
    "BuiltinBackend",
    "BuiltinTBBackend",
    "OpenHandsBackend",
    "OpenHandsTBBackend",
    "Terminus2Backend",
]

#: Lazy attribute → defining submodule (all five are import-clean sans SDKs).
_LAZY_SUBMODULES = {
    "BuiltinBackend": "builtin",
    "BuiltinTBBackend": "builtin_tb",
    "OpenHandsBackend": "openhands",
    "OpenHandsTBBackend": "openhands_tb",
    "Terminus2Backend": "terminus2",
}


def __getattr__(name: str) -> object:
    """Resolve the concrete backends lazily (PEP 562).

    ``__all__`` advertises the five backend classes, so
    ``from meta_n.core.external_agents.backends import BuiltinBackend`` (and a
    star-import) must work — but importing them eagerly would defeat the
    package's zero-side-effect import discipline. Each name is bound on first
    access from its own submodule; any other unknown attribute raises the
    standard :class:`AttributeError`.
    """
    submodule = _LAZY_SUBMODULES.get(name)
    if submodule is not None:
        import importlib

        module = importlib.import_module(f".{submodule}", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Include the lazily-exported backend classes in :func:`dir` output."""
    return sorted(set(globals()) | set(__all__))
