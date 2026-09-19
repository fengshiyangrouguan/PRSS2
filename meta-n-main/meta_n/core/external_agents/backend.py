"""Strategy boundary for external agents — the ``AgentBackend`` ABC and its DTOs.

This module defines the *pure-data* contract between meta-n's solver spine
(:class:`~meta_n.core.external_agents.solver.ExternalAgentSolver`) and the
pluggable agent strategies that actually drive an external agent (the builtin
Layer1Solver/AgenticSolver wrapper, OpenHands, Terminus 2).

It owns four dataclasses and one abstract base class:

* :class:`Prompt` — the two injection surfaces (system suffix + message prefix)
  produced by :class:`~meta_n.core.external_agents.injection.InjectionMapper`.
* :class:`InjectionPlan` — the full output of injection mapping: the prompt, the
  files to stage into the agent's workspace, the utility names available for
  telemetry attribution, and whether ``pre_process`` ran.
* :class:`AgentRunContext` — the immutable inputs handed to a backend's
  :meth:`AgentBackend.run` (instruction, prompt, workspace handle, and the
  per-run resource limits).
* :class:`AgentRunResult` — the uniform, backend-agnostic record a backend
  returns: transcript, reasoning, token/cost accounting, control metadata, and
  a :class:`~meta_n.core.external_agents.terminated.TerminatedBy` reason.
* :class:`AgentBackend` — the strategy ABC. ``run`` must *never* raise (the spine
  relies on this to keep the evaluation ``gather`` alive); errors are reported
  via the result's ``terminated_by``/``failure_mode`` fields.

Design notes
------------
This file is WAVE 1: it imports only the zero-dependency ``terminated`` sibling
and the standard library, so the ``external_agents`` package imports cleanly
without ``openhands``, ``terminal_bench``, or ``docker`` installed. Any external
SDK is imported lazily *inside* the concrete backends, never here.

The ``outer_token_mode`` flag on :class:`AgentBackend` is the discriminator that
keeps two token ledgers from silently mixing (see plan §2.3, §7.11): it is
``True`` *only* for the builtin backend, whose calls flow through the outer
``LLMClient`` and are therefore returned as native *outer* tokens; for OpenHands
and Terminus 2 the agent uses its own LLM client, so spend is reported as
``Trace.inner_*`` and the spine's outer-return token count is ``0``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from .terminated import TerminatedBy

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids runtime import cycles
    from .telemetry import AgentRunRecord, AgentTelemetry


__all__ = [
    "Prompt",
    "InjectionPlan",
    "AgentRunContext",
    "AgentRunResult",
    "AgentBackend",
    "task_from_workspace",
]


def task_from_workspace(
    ws: object, *, nested_requires: "tuple[str, ...]" = ("task_id",)
) -> "object | None":
    """Recover a task-shaped object from an env provider's workspace handle.

    Shared by the builtin backends (CO-Bench + TB): the handle may BE the task
    (exposes ``task_id`` AND ``description``) or bundle one under a
    conventional ``.task`` / ``.task_description`` attribute. A nested
    candidate must expose every attribute in ``nested_requires`` — the
    per-backend requirement set (``("task_id",)`` for the CO-Bench builtin
    control; ``("description", "task_id")`` for the TB one, whose caller
    synthesizes a task when the handle carries none).

    Args:
        ws: The :class:`AgentRunContext` workspace handle (or ``None``).
        nested_requires: Attributes a nested ``.task``/``.task_description``
            candidate must expose to count as a task.

    Returns:
        The recovered task object, or ``None`` when the handle carries no task.
    """
    if ws is None:
        return None
    # Direct task handle (has the description-shaped attributes we need).
    if hasattr(ws, "task_id") and hasattr(ws, "description"):
        return ws
    # Env object that bundles the task under a conventional attribute.
    for attr in ("task", "task_description"):
        cand = getattr(ws, attr, None)
        if cand is not None and all(hasattr(cand, a) for a in nested_requires):
            return cand
    return None


@dataclass
class Prompt:
    """The two injection surfaces shared by every backend.

    Attributes:
        system_suffix: ``pre_process`` output + library descriptions. Mapped to
            ``AgentContext.system_message_suffix`` for OpenHands and folded into
            the leading instruction for Terminus 2 (which has no suffix slot).
        prefix: Inter-layer ``additional_context``, prepended to the initial
            user/instruction message.

    A fully-empty ``Prompt`` (both fields ``""``) is the gen0 vanilla-agent
    baseline: no injected behavior at all (plan §3).
    """

    system_suffix: str = ""
    prefix: str = ""


@dataclass
class InjectionPlan:
    """The complete output of :class:`InjectionMapper.build`.

    Attributes:
        prompt: The :class:`Prompt` carrying the system suffix and message prefix.
        staged_files: Map of workspace-relative path (e.g. ``helpers/foo.py``) to
            file contents. Written into the agent's workspace before the run.
        utilities_available: Sorted names of every staged helper, used for
            non-invasive utility attribution in telemetry (plan §7.6).
        pre_process_ran: Whether a ``pre_process`` block executed during build.
    """

    prompt: Prompt
    staged_files: dict[str, str] = field(default_factory=dict)
    utilities_available: list[str] = field(default_factory=list)
    pre_process_ran: bool = False


@dataclass
class AgentRunContext:
    """Immutable inputs handed to :meth:`AgentBackend.run`.

    Bundles the task instruction, the injected :class:`Prompt`, an opaque
    workspace handle (whatever the env provider yields — a path, a container
    handle, a remote session id), and the per-run resource limits the backend
    must honor.

    Attributes:
        instruction: The task's natural-language description. ADVISORY on the
            TB subprocess backends (terminus2 / openhands_tb): they load the
            canonical instruction from the on-disk ``task.yaml`` and do NOT read
            this field, so a meta-layer that REWROTE (rather than appended via
            ``prompt``) the instruction would be silently void on those paths.
            Meta-layer effects reach the TB backends only through
            ``prompt.system_suffix`` / ``prompt.prefix`` (forwarded).
        prompt: Injection surfaces (system suffix + message prefix).
        workspace: Opaque, provider-specific workspace handle.
        time_limit_s: Soft wall-clock budget in seconds, or ``None`` for no soft
            limit (the spine still wraps the run in a hard timeout).
        max_turns: Maximum agent turns/episodes.
        token_budget: Maximum cumulative tokens the agent may spend.
        max_budget_usd: Maximum USD the agent may spend this run.
        logging_dir: Directory for the backend's own logs/artifacts.
    """

    instruction: str
    prompt: Prompt
    workspace: object
    time_limit_s: float | None
    max_turns: int
    token_budget: int
    max_budget_usd: float
    logging_dir: Path


@dataclass
class AgentRunResult:
    """Uniform, backend-agnostic record returned by every :meth:`AgentBackend.run`.

    Carries everything the spine needs to build a :class:`~meta_n.core.meta_layer.Trace`,
    feed the cost ledger, and write the telemetry row — regardless of which agent
    produced it.

    Token accounting (``agent_*``) is the agent's *inner* spend; for OpenHands
    and Terminus 2 it rides in ``Trace.inner_*`` while the spine returns ``0``
    outer tokens, whereas the builtin backend (``outer_token_mode=True``) returns
    these as native outer tokens instead (see plan §2.3, §7.1).

    Attributes:
        transcript: Full agent transcript text.
        reasoning_summary: Condensed reasoning/plan summary for the ``Trace``.
        stdout_tail: Tail of captured stdout.
        stderr_tail: Tail of captured stderr.
        artifacts_path: Path to archived run artifacts/logs.
        agent_tokens: Total inner tokens (= prompt + completion).
        agent_prompt_tokens: Inner prompt (input) tokens.
        agent_completion_tokens: Inner completion (output) tokens.
        agent_cached_tokens: Inner cached/prompt-cache tokens.
        agent_calls: Number of inner LLM calls.
        cost_usd: Native USD spend if the backend reports it (OpenHands); else
            ``0.0`` (priced later from tokens).
        cost_basis: ``"native_usd"`` when ``cost_usd`` is the agent's own ledger,
            ``"priced_from_tokens"`` when it must be derived from token counts.
        wall_s: Wall-clock seconds the agent ran.
        steps: Number of agent steps taken.
        command_history: Ordered shell/tool commands the agent issued, for
            utility attribution; ``[]`` when unavailable (plan §7.6).
        attribution_available: ``False`` when the backend cannot expose a command
            stream (so attribution must be reported as unknown, not empty).
        terminated_by: Why the run ended, as a :class:`TerminatedBy`.
        failure_mode: Short failure tag, or ``None`` on success.
        native_score: The backend's authoritative verifier signal (the binary
            reward for Terminus 2 / terminal-bench), carried so the scorer derives
            success/score FROM the verifier — not transitively from the agent-status
            ``terminated_by`` enum. ``None`` when the backend has no native verifier
            (the spine's own scorer is then authoritative).
        native_resolved: The backend's authoritative pass/fail flag (terminal-bench
            ``is_resolved``); ``None`` when not applicable. Kept distinct from
            ``terminated_by == COMPLETED`` so a future status-only change cannot
            fabricate a passing score.
        native_handle: Private backend handle (e.g. OpenHands conversation, T2
            chat) kept only for post-hoc metric collection / hard cancel.
    """

    transcript: str = ""
    reasoning_summary: str = ""
    stdout_tail: str = ""
    stderr_tail: str = ""
    artifacts_path: str = ""
    agent_tokens: int = 0
    agent_prompt_tokens: int = 0
    agent_completion_tokens: int = 0
    agent_cached_tokens: int = 0
    agent_calls: int = 0
    cost_usd: float = 0.0
    cost_basis: str = "priced_from_tokens"
    wall_s: float = 0.0
    steps: int = 0
    command_history: list[str] = field(default_factory=list)
    attribution_available: bool = False
    terminated_by: TerminatedBy = TerminatedBy.UNKNOWN
    failure_mode: str | None = None
    native_score: float | None = None
    native_resolved: bool | None = None
    native_handle: object | None = None


class AgentBackend(ABC):
    """Pluggable strategy that drives a single external agent for one run.

    Concrete backends (``builtin``, ``openhands``, ``terminus2``) translate an
    :class:`AgentRunContext` into agent activity and return a uniform
    :class:`AgentRunResult`. The strategy is selected per candidate by the
    orchestrator and supplied to :class:`ExternalAgentSolver`.

    Invariants:
        * :meth:`run` MUST NOT raise. The spine drives runs inside an evaluation
          ``gather``; a raised exception would cancel sibling tasks. Errors are
          reported through ``result.terminated_by`` / ``result.failure_mode``
          (e.g. :attr:`TerminatedBy.AGENT_ERROR`). The lone exception is
          :class:`asyncio.CancelledError`, which must propagate.
        * External SDKs are imported lazily inside the concrete ``run`` body so
          this package imports without those SDKs installed.

    Attributes:
        name: Stable backend identifier — ``"builtin"``, ``"openhands"`` or
            ``"terminus2"``. Used as the telemetry ``agent`` coordinate.
        outer_token_mode: ``True`` only for the builtin backend, whose tokens
            flow through the outer ``LLMClient`` and are returned as native outer
            tokens rather than remapped to ``Trace.inner_*`` (plan §2.3, §7.11).
    """

    #: Stable backend identifier; overridden as a class attribute by subclasses.
    name: str = "abstract"
    #: ``True`` only for the builtin backend (native outer-token accounting).
    outer_token_mode: bool = False

    @abstractmethod
    async def run(
        self,
        ctx: AgentRunContext,
        tel: "AgentTelemetry",
        rec: "AgentRunRecord",
    ) -> AgentRunResult:
        """Drive the agent for one task and return a uniform result.

        Args:
            ctx: Immutable run inputs (instruction, prompt, workspace, limits).
            tel: The active :class:`AgentTelemetry` (uniform signature; unused
                by all current backends).
            rec: The :class:`AgentRunRecord` started by the spine for this run.

        Returns:
            An :class:`AgentRunResult`. MUST NOT raise except for
            :class:`asyncio.CancelledError`; all other failures are encoded in
            the returned result's ``terminated_by`` / ``failure_mode``.
        """
        raise NotImplementedError

    async def collect_metrics(
        self, env: object, run: AgentRunResult
    ) -> AgentRunResult:
        """Augment ``run`` with post-hoc metrics read from the live env.

        Default is a no-op pass-through. OpenHands overrides this to read
        ``accumulated_cost`` / ``token_usage`` off the conversation before the
        env is torn down (plan §2.4). Like :meth:`run`, it must never raise — an
        override should ``try``/``except`` and fall back to the unmodified
        ``run`` on any error.

        Args:
            env: The live environment yielded by the env provider.
            run: The result produced by :meth:`run`.

        Returns:
            The (possibly augmented) :class:`AgentRunResult`.
        """
        return run
