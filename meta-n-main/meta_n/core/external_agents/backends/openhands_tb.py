"""TB-mode OpenHands backend — SUBPROCESS BRIDGE (no ``openhands`` in meta-n's env).

This is the terminal-bench analogue of :class:`OpenHandsBackend` (the CO-Bench /
host-workspace path) and a sibling of
:class:`~meta_n.core.external_agents.backends.terminus2.Terminus2Backend`. It lets
meta-n run the **OpenHands** agent against terminal-bench tasks while reusing the
*entire* Terminus 2 TB plumbing unchanged — the shared ``TBTerminus2EnvProvider``
(workspace handle carrying ``task_id`` / ``staged_files`` / ``run_label``) and the
shared ``TBTerminus2Scorer`` (which reads ``native_resolved`` / ``native_score``
off the run).

``openhands`` AND ``terminal_bench`` have pins that conflict with meta-n's, so
meta-n's process must **never** import either. This backend therefore drives
OpenHands-on-terminal-bench via a *subprocess bridge*: :meth:`run` (inherited from
:class:`_ExternalTBBackend`) shells out to the dedicated external-agents
interpreter (``.venv_external_agents/bin/python``) running the standalone runner
(``scripts/oh_tb_runner.py``), which imports BOTH SDKs in-process, registers a
custom OpenHands terminal-tool executor that drives terminal-bench's own
``TmuxSession`` / verifier, and writes a result JSON to a path meta-n chose.
meta-n parses that JSON into an :class:`AgentRunResult`.

CRITICAL: ``oh_tb_runner.py`` writes the **same result-JSON schema** that
``scripts/t2_runner.py`` writes (``ok`` / ``is_resolved`` / ``reward`` /
``failure_mode`` / ``total_input_tokens`` / ``total_output_tokens`` /
``agent_calls`` / ``command_history`` / ``transcript_path`` / ``post_agent_pane``
/ ``post_test_pane`` / ``wall_s`` / ``steps``), so the binary-reward → result
mapping is inherited from :class:`_ExternalTBBackend` verbatim, and the downstream
``TBTerminus2Scorer`` works on an OH run with no change.

This module imports **only the standard library** plus the zero-/wave-1 sibling
``backend`` / ``terminated`` / ``_external_tb`` modules — it has no top-level (or
method-local) ``import openhands`` / ``import terminal_bench``. It stays importable
with both SDKs absent.

Architecture (locked contract)
------------------------------
* Interpreter: ``.venv_external_agents/bin/python`` (the only env with the SDKs).
* Runner: ``scripts/oh_tb_runner.py``, invoked as ``-m oh_tb_runner`` with
  ``PYTHONPATH`` pointing at the ``scripts/`` directory.
* Request JSON (meta-n → runner): task id, tasks dir, litellm routing, injection
  context (forwarded as ``system_message_suffix`` — OH has a real suffix slot),
  staged files, episode/timeout caps, token budget, output dir.
* Result JSON (runner → meta-n): the SHARED t2-style schema (see above);
  ``is_resolved`` is the binary verifier reward.
* Token accounting is INNER (``outer_token_mode = False``); cost is priced from
  tokens by the spine's ``CostGuard`` (``cost_basis = "priced_from_tokens"``).

Teardown / never-raise / hard-timeout behavior is single-sourced in
:class:`_ExternalTBBackend`. This subclass overrides only the OH-specific bridge
identity (runner module / request+result filenames / compose labels), the
native-failure-tag resolver (:func:`from_oh_status`), the request key shape
(``system_message_suffix`` + ``max_iterations``), and the two OH-specific result
fields (``reasoning_summary`` = the agent's last message, ``agent_cached_tokens``).
The native ``failure_mode`` strings the runner emits follow OpenHands' status
vocabulary (``token_budget`` / ``max_iterations`` / ``context_window_exceeded`` /
``env_error`` / ``parse_error`` / ``agent_error``, plus the t2-style
``agent_timeout``).
"""

from __future__ import annotations

from typing import Any

from ..backend import AgentRunContext
from ..terminated import TerminatedBy, from_oh_status
from ._external_tb import _ExternalTBBackend

__all__ = ["OpenHandsTBBackend"]


class OpenHandsTBBackend(_ExternalTBBackend):
    """Drive the OpenHands agent on terminal-bench via a subprocess bridge.

    Selected per candidate by the orchestrator (``--base-solver openhands`` on the
    terminal_bench benchmark) and handed to
    :class:`~meta_n.core.external_agents.solver.ExternalAgentSolver`. The whole
    agent run happens in a child process (the external-agents venv); meta-n's
    process touches no ``openhands`` / ``terminal_bench`` symbol. :meth:`run`
    (inherited) honors the :class:`AgentBackend` never-raise contract.

    Attributes:
        name: ``"openhands"`` — the telemetry ``agent`` coordinate (shared with the
            CO-Bench OpenHands backend; the benchmark disambiguates).
        outer_token_mode: ``False`` (inner-token accounting).
    """

    name: str = "openhands"

    _RUNNER_MODULE = "oh_tb_runner"
    _REQUEST_FILENAME = "oh_tb_request.json"
    _RESULT_FILENAME = "oh_tb_result.json"
    _RUN_LABEL_PREFIX = "oh-tb"
    _TRIAL_COMPOSE_PROJECT = "oh-tb-bridge"
    _LOG_NAME = "OpenHandsTBBackend"

    # ------------------------------------------------------- subclass hooks --
    def _resolve_terminated(self, failure_mode: object) -> TerminatedBy:
        """Resolve a native OpenHands failure tag to :class:`TerminatedBy`.

        Defers to the canonical
        :func:`~meta_n.core.external_agents.terminated.from_oh_status`, which covers
        every tag the OH runner emits (``token_budget`` / ``max_iterations`` /
        ``context_window_exceeded`` / ``env_error`` / ``parse_error`` /
        ``agent_error`` / ``agent_timeout`` …) and falls back to
        :attr:`TerminatedBy.UNKNOWN` for an unmapped one.
        """
        return from_oh_status(failure_mode)

    def _extra_request_fields(self, ctx: AgentRunContext, max_episodes: int) -> dict:
        """OpenHands request extras: the suffix-slot injection + the iteration cap.

        OpenHands DOES have a system-suffix slot (``AgentContext.
        system_message_suffix``), which the runner wires from the composed injection
        block (it reads the request key ``system_message_suffix``, NOT
        ``additional_context``). The runner also caps OH iterations off
        ``max_iterations`` (its own field name); send BOTH spellings so a future
        runner that reads either key resolves. At depth 1 the injection block is
        ``""``, making the run byte-identical to vanilla OpenHands-on-terminal-bench.
        """
        return {
            "system_message_suffix": self._compose_instruction_context(ctx),
            "max_iterations": max_episodes,
        }

    def _extra_result_fields(self, data: dict[str, Any]) -> dict:
        """OpenHands result extras: the last-message summary + prompt-cache reads.

        ``reasoning_summary`` carries the agent's last message (bounded);
        ``agent_cached_tokens`` carries the prompt-cache read count the OH runner
        ships. Both default cleanly to empty/zero on the degraded / pre-run paths
        (an empty ``data`` dict).
        """
        return {
            "reasoning_summary": str(data.get("last_message", "") or "")[:4000],
            "agent_cached_tokens": int(data.get("cache_read_tokens", 0) or 0),
        }
