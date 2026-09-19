"""Terminus 2 backend — SUBPROCESS BRIDGE (no ``terminal_bench`` in meta-n's env).

``terminal_bench`` and meta-n have conflicting pins, so meta-n's process must
**never** import ``terminal_bench``. This backend therefore drives Terminus 2 via
a *subprocess bridge*: :meth:`run` (inherited from :class:`_ExternalTBBackend`)
shells out to the dedicated external-agents interpreter running the standalone
runner (``scripts/t2_runner.py``), which imports the ``terminal_bench`` SDK
in-process, drives terminal-bench's OWN single-task trial machinery
(``Harness._run_trial`` → ``spin_up_terminal`` → ``TmuxSession`` → verifier →
binary reward), and writes a result JSON to a path meta-n chose. meta-n parses
that JSON into an :class:`AgentRunResult`.

This module imports **only the standard library** plus the zero-/wave-1 sibling
``backend`` / ``terminated`` / ``_external_tb`` modules — it has no top-level (or
method-local) ``import terminal_bench``. It stays importable with that absent.

Architecture (locked contract)
------------------------------
* Interpreter: ``.venv_external_agents/bin/python`` (the only env with the SDK).
* Runner: ``scripts/t2_runner.py``, invoked as ``-m t2_runner`` with
  ``PYTHONPATH`` pointing at the ``scripts/`` directory.
* Request JSON (meta-n → runner): task id, tasks dir, litellm routing, injection
  context (folded into ``additional_context`` — Terminus 2 has no suffix slot),
  staged files, episode/timeout caps, output dir.
* Result JSON (runner → meta-n): ``is_resolved`` (binary reward), authoritative
  token totals off ``AgentResult``, ``failure_mode``, parser results, transcript
  / pane artifact paths, command history.
* Token accounting is INNER (``outer_token_mode = False``); cost is priced from
  tokens by the spine's ``CostGuard`` (``cost_basis = "priced_from_tokens"``).

Teardown / never-raise / hard-timeout behavior is single-sourced in
:class:`_ExternalTBBackend`. This subclass overrides only the Terminus-2-specific
bridge identity (runner module / request+result filenames / compose labels), the
native-failure-tag resolver (:func:`from_t2_failure`), and the request key shape
(``parser_name`` + ``additional_context``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .._bridge import sanitize_compose_name
from ..backend import AgentRunContext
from ..terminated import from_t2_failure
from ._external_tb import _ExternalTBBackend

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..terminated import TerminatedBy

__all__ = ["Terminus2Backend"]

#: Fallback compose project name when the env handle carries no per-run label.
#: NORMALLY the trial's compose project is the lease's per-run-unique session
#: (read off ``ctx.workspace.run_label``), so concurrent T2 trials never collide
#: and a hard-timeout sweep only removes THIS run's container.
_TRIAL_COMPOSE_PROJECT = "t2-bridge"


def _sanitize_compose_label(label: str) -> str:
    """Reduce an arbitrary run label to a Docker-Compose-safe project name.

    Module-level thin wrapper over :func:`_bridge.sanitize_compose_name` pinned to
    the T2 fallback (``t2-bridge``) for an empty label. Kept as a module symbol for
    the unit suite; :class:`_ExternalTBBackend` uses the class-attr-driven
    ``_sanitize_compose_label`` method internally.
    """
    return sanitize_compose_name(label, fallback=_TRIAL_COMPOSE_PROJECT)


class Terminus2Backend(_ExternalTBBackend):
    """Drive terminal-bench's Terminus 2 agent via a subprocess bridge.

    Selected per candidate by the orchestrator (``--base-solver terminus2``) and
    handed to :class:`~meta_n.core.external_agents.solver.ExternalAgentSolver`.
    The whole agent run happens in a child process (the external-agents venv);
    meta-n's process touches no ``terminal_bench`` symbol. :meth:`run` (inherited)
    honors the :class:`AgentBackend` never-raise contract.

    Attributes:
        name: ``"terminus2"`` — the telemetry ``agent`` coordinate.
        outer_token_mode: ``False`` (inner-token accounting).
    """

    name: str = "terminus2"

    _RUNNER_MODULE = "t2_runner"
    _REQUEST_FILENAME = "t2_request.json"
    _RESULT_FILENAME = "t2_result.json"
    _RUN_LABEL_PREFIX = "t2"
    _TRIAL_COMPOSE_PROJECT = "t2-bridge"
    _LOG_NAME = "Terminus2Backend"

    def __init__(self, *, parser_name: str = "json", **kwargs) -> None:
        """Configure the bridge; ``parser_name`` is the Terminus-2-only extra.

        Args:
            parser_name: Terminus 2 output parser (``"json"`` / ``"xml"``).
            **kwargs: The shared :class:`_ExternalTBBackend` construction kwargs.
        """
        super().__init__(**kwargs)
        self._parser_name = parser_name

    # ------------------------------------------------------- subclass hooks --
    def _resolve_terminated(self, failure_mode: object) -> "TerminatedBy":
        """Resolve a native T2 failure tag to :class:`TerminatedBy`.

        Defers entirely to the canonical
        :func:`~meta_n.core.external_agents.terminated.from_t2_failure` — the
        shared table covers the runner/bridge-emitted tags (``env_error`` /
        ``output_length_exceeded`` / ``agent_installation_failed``) that were
        formerly backend-local overrides.
        """
        return from_t2_failure(failure_mode)

    def _extra_request_fields(self, ctx: AgentRunContext, max_episodes: int) -> dict:
        """Terminus 2 request extras: the parser name + the folded injection block.

        Terminus 2 has **no** system-suffix slot, so the injected ``system_suffix``
        and ``prefix`` are concatenated into a single ``additional_context`` block
        the runner prepends to the task instruction (raw task text comes LAST). At
        depth 1 both injection fields are empty → ``""``, making the run
        byte-identical to vanilla Terminus 2.
        """
        return {
            "parser_name": self._parser_name,
            "additional_context": self._compose_instruction_context(ctx),
        }
