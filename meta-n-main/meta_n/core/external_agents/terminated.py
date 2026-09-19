"""Shared ``terminated_by`` taxonomy for the external-agent backends.

This is WAVE 0 of the external-agents integration (plan §7.5): a single,
zero-dependency module that defines the canonical :class:`TerminatedBy` enum plus
the two native-status mapping tables (:data:`_OH_STATUS_TO_ENUM`,
:data:`_T2_FAILURE_TO_ENUM`) that every backend folds its own vocabulary into.

The enum lives here — and *only* here — so that ``backend.py``, ``solver.py``,
``telemetry.py`` and the per-agent backends (``backends/openhands.py``,
``backends/terminus2.py``) all agree on one set of terminal-state labels and one
serialized spelling. ``TerminatedBy`` is **never** a field on
:class:`meta_n.core.meta_layer.Trace`; it rides on ``AgentRunResult`` /
``AgentRunRecord`` and is written verbatim into ``telemetry/agent_runs.jsonl``
(plan §7.3, §7.9).

Design notes
------------
* ``TerminatedBy`` is a ``str``-backed enum so each member *is* its JSON value
  (``TerminatedBy.TIMEOUT == "timeout"``); the telemetry write path can dump it
  with no custom encoder, and ``json.loads`` round-trips to the same string.
* The task spec and the integration plan use two spellings for a few states
  (e.g. ``CONFIRMED`` / ``PERFECT_SCORE`` vs ``COMPLETED``, ``CONTEXT`` vs
  ``CONTEXT_LEN``). For *those*, both are defined: the plan name is the
  *canonical* member and the spec name is an **alias** (same value), so
  ``TerminatedBy.CONFIRMED is TerminatedBy.COMPLETED`` and either import resolves
  to one member with one serialized value.
* ``BUDGET_USD`` and ``BUDGET_DENIED`` are **NOT** aliases — they are two
  *distinct* canonical members with different values: ``BUDGET_USD``
  (``"budget_usd"``) is soft per-run USD budget exhaustion *during* a run, while
  ``BUDGET_DENIED`` (``"budget_denied"``) is the spine's up-front pre-check
  denying the run before it starts. Do not treat them as interchangeable.
* Mapping tables fold the *native* status / failure-mode vocabulary of each
  external agent into the shared enum. Anything not in a table falls back to
  :data:`TerminatedBy.UNKNOWN` via the :func:`from_oh_status` /
  :func:`from_t2_failure` helpers (and via ``dict.get(..., UNKNOWN)`` at the
  call sites in the backends).

This module imports only the standard library ``enum`` so the
``external_agents`` package stays importable without ``openhands``,
``terminal_bench`` or ``docker`` installed.
"""

from __future__ import annotations

import enum

__all__ = [
    "TerminatedBy",
    "from_oh_status",
    "from_t2_failure",
]


class TerminatedBy(str, enum.Enum):
    """Why an external-agent run stopped — the shared, serialized taxonomy.

    Members are ``str``-valued (``TerminatedBy.TIMEOUT == "timeout"``) so the
    enum drops straight into JSONL telemetry without a custom encoder. The
    canonical members follow integration plan §7.5; the spec-facing aliases
    (``CONFIRMED`` / ``PERFECT_SCORE`` → ``COMPLETED``, ``CONTEXT`` →
    ``CONTEXT_LEN``) are defined as additional names for an existing member so
    both spellings resolve to the same value. Note ``BUDGET_USD`` (soft per-run
    budget exhaustion) and ``BUDGET_DENIED`` (up-front spine pre-check denial)
    are *distinct* members, not aliases.
    """

    # --- canonical members (integration plan §7.5) ------------------------
    COMPLETED = "completed"        # agent declared/confirmed the task done
    MAX_TURNS = "max_turns"        # hit the iteration / episode ceiling
    TOKEN_BUDGET = "token_budget"  # inner LLM token budget exhausted
    BUDGET_USD = "budget_usd"      # per-run USD budget exhausted (soft stop)
    TIMEOUT = "timeout"            # wall-clock envelope expired
    ENV_ERROR = "env_error"        # sandbox/server/container fault (not agent)
    PARSE_ERROR = "parse_error"    # agent output could not be parsed
    CONTEXT_LEN = "context_len"    # ran out of model context window
    AGENT_ERROR = "agent_error"    # generic in-agent failure
    BUDGET_DENIED = "budget_denied"  # spine pre-check denied the run up front
    UNKNOWN = "unknown"            # unmapped / indeterminate terminal state

    # --- spec-facing aliases (same value → same member) -------------------
    # Defined as plain class attributes that reference an existing member so
    # ``enum`` treats them as aliases rather than new members. This keeps the
    # serialized vocabulary single-valued while accepting either spelling.
    CONFIRMED = COMPLETED          # alias of COMPLETED
    PERFECT_SCORE = COMPLETED      # alias of COMPLETED (scored a clean pass)
    CONTEXT = CONTEXT_LEN          # alias of CONTEXT_LEN

    def __str__(self) -> str:  # pragma: no cover - trivial
        """Return the bare string value (``"timeout"``), not ``"TerminatedBy.TIMEOUT"``."""
        return str(self.value)


# ---------------------------------------------------------------------------
# OpenHands: REST ``execution_status`` + soft ``failure_mode`` → TerminatedBy
# ---------------------------------------------------------------------------
# `# SPIKE` (plan §7.5): the exact agent-server REST `execution_status` surface
# is confirmed in the OpenHands verification spike. This table is the working
# hypothesis and is intentionally generous — it covers both the raw
# `ConversationInfo.execution_status` values and the soft `failure_mode`
# strings that `OpenHandsBackend._classify_oh_error` / `_term_for` emit
# (plan §4.6, §4.3). Unknown keys fall back to UNKNOWN (see `from_oh_status`).
_OH_STATUS_TO_ENUM: dict[str, TerminatedBy] = {
    # --- terminal `execution_status` values --------------------------------
    "finished": TerminatedBy.COMPLETED,
    "completed": TerminatedBy.COMPLETED,
    "success": TerminatedBy.COMPLETED,
    "idle": TerminatedBy.COMPLETED,        # idle-after-run == nothing left to do
    "stopped": TerminatedBy.COMPLETED,
    "stuck": TerminatedBy.AGENT_ERROR,     # agent looped / made no progress
    "error": TerminatedBy.AGENT_ERROR,
    "failed": TerminatedBy.AGENT_ERROR,
    "rejected": TerminatedBy.AGENT_ERROR,
    "paused": TerminatedBy.AGENT_ERROR,    # paused-on-confirmation, not done
    # non-terminal statuses that can surface if polling is cut short
    "running": TerminatedBy.UNKNOWN,
    "starting": TerminatedBy.UNKNOWN,
    "pending": TerminatedBy.UNKNOWN,
    # --- soft failure_mode strings from `_classify_oh_error` (plan §4.6) ----
    # OpenHands' in-run cap is a per-run USD ceiling (`max_budget_per_task`), so a
    # budget stop is labeled BUDGET_USD (the purpose-built soft member), NOT
    # TOKEN_BUDGET (which is reserved for genuine inner-token exhaustion).
    "budget_exhausted": TerminatedBy.BUDGET_USD,  # soft per §4.6 (USD cap)
    # The OH runner's per-call cumulative INNER-token kill switch (oh_runner.py
    # ``_install_token_budget_killswitch``) rewrites the SDK's overloaded ERROR
    # into ``status="token_budget"`` — a genuine inner-token exhaustion (parity
    # with Terminus 2's ``token_budget`` stop), NOT a USD-cap stop. Soft.
    "token_budget": TerminatedBy.TOKEN_BUDGET,
    "max_iterations": TerminatedBy.MAX_TURNS,
    "iteration_limit": TerminatedBy.MAX_TURNS,
    "agent_timeout": TerminatedBy.TIMEOUT,
    "timeout": TerminatedBy.TIMEOUT,
    "env_error": TerminatedBy.ENV_ERROR,        # server crash ≠ agent failure
    "server_error": TerminatedBy.ENV_ERROR,
    "context_window_exceeded": TerminatedBy.CONTEXT_LEN,
    "context_length_exceeded": TerminatedBy.CONTEXT_LEN,
    "parse_error": TerminatedBy.PARSE_ERROR,
    # Generic in-agent failure — the `_classify_oh_error` catch-all fallback.
    # WITHOUT this entry the fallback token mapped to UNKNOWN (a silent gap that
    # corrupted agent-quality vs env-fault attribution); it now maps correctly to
    # AGENT_ERROR. The regression test in test_terminated_mappings.py asserts that
    # EVERY token `_classify_oh_error` can emit is a key in this table.
    "agent_error": TerminatedBy.AGENT_ERROR,
}

# ---------------------------------------------------------------------------
# Terminus-2: `failure_mode` strings → TerminatedBy
# ---------------------------------------------------------------------------
# Covers the inline `failure_mode` strings set in `Terminus2Backend.run()`
# ("agent_timeout", "token_budget") plus the stable strings produced by
# `_classify_t2_retry` (which inspects the `RetryError.last_attempt.exception()`
# for Context/Output/Parse failures) and the names of `terminal_bench`'s
# `FailureMode` enum (verified post-install). Unknown keys fall back to UNKNOWN
# (see `from_t2_failure`). On the success path `failure_mode is None`, which
# `from_t2_failure` short-circuits to COMPLETED *before* indexing this table, so
# no bare-``None`` key is needed here (the ``"none"`` token covers FailureMode.NONE).
_T2_FAILURE_TO_ENUM: dict[str, TerminatedBy] = {
    "none": TerminatedBy.COMPLETED,              # FailureMode.NONE
    "completed": TerminatedBy.COMPLETED,
    # --- inline failure modes from Terminus2Backend.run() ------------------
    "agent_timeout": TerminatedBy.TIMEOUT,
    "timeout": TerminatedBy.TIMEOUT,
    "token_budget": TerminatedBy.TOKEN_BUDGET,
    # --- classified RetryError outcomes (`_classify_t2_retry`) -------------
    "context_length_exceeded": TerminatedBy.CONTEXT_LEN,
    "context_window_exceeded": TerminatedBy.CONTEXT_LEN,
    "context": TerminatedBy.CONTEXT_LEN,
    "output_limit": TerminatedBy.MAX_TURNS,
    "output": TerminatedBy.MAX_TURNS,
    "parse_error": TerminatedBy.PARSE_ERROR,
    "parse": TerminatedBy.PARSE_ERROR,
    # --- terminal_bench FailureMode enum names (post-install verified) -----
    "max_episodes": TerminatedBy.MAX_TURNS,
    "agent_max_episodes": TerminatedBy.MAX_TURNS,
    "unknown_agent_error": TerminatedBy.AGENT_ERROR,
    "fatal_llm_parse_error": TerminatedBy.PARSE_ERROR,
    "context_length_exceeded_error": TerminatedBy.CONTEXT_LEN,
    "test_timeout": TerminatedBy.ENV_ERROR,
    "agent_timeout_error": TerminatedBy.TIMEOUT,
    # --- runner/bridge-emitted tags (formerly backend-local overrides) ------
    # Emitted by the t2/builtin_tb runners and the meta-n bridge itself:
    # ``env_error`` (a runner-internal Docker/daemon fault or a meta-n launch
    # failure), ``output_length_exceeded`` (the LLM hit ``max_tokens`` — an
    # output-limit stop), and ``agent_installation_failed`` (the agent could not
    # be provisioned, an env fault).
    "env_error": TerminatedBy.ENV_ERROR,
    "output_length_exceeded": TerminatedBy.MAX_TURNS,  # output-limit stop
    "agent_installation_failed": TerminatedBy.ENV_ERROR,
}


def from_oh_status(status: object) -> TerminatedBy:
    """Map an OpenHands native status / failure-mode token to :class:`TerminatedBy`.

    Accepts a raw ``execution_status`` string, an enum-like value (its
    ``.value`` / ``.name`` / ``str()`` is normalized), or a soft ``failure_mode``
    string from ``_classify_oh_error``. Matching is case-insensitive. Anything
    unrecognized (including ``None``) returns :data:`TerminatedBy.UNKNOWN`.
    """
    key = _normalize(status)
    if key is None:
        return TerminatedBy.UNKNOWN
    return _OH_STATUS_TO_ENUM.get(key, TerminatedBy.UNKNOWN)


def from_t2_failure(failure_mode: object) -> TerminatedBy:
    """Map a Terminus-2 ``failure_mode`` token to :class:`TerminatedBy`.

    ``None`` (the success path) maps to :data:`TerminatedBy.COMPLETED`. Accepts a
    raw string or an enum-like ``FailureMode`` value; matching is
    case-insensitive. Anything unrecognized returns :data:`TerminatedBy.UNKNOWN`.
    """
    if failure_mode is None:
        return TerminatedBy.COMPLETED
    key = _normalize(failure_mode)
    if key is None:
        return TerminatedBy.UNKNOWN
    return _T2_FAILURE_TO_ENUM.get(key, TerminatedBy.UNKNOWN)


def _normalize(value: object) -> str | None:
    """Reduce a status/failure value to a lowercase lookup key, or ``None``.

    Handles plain strings and enum-like objects (uses ``.value`` when it is a
    string, else ``.name``, else ``str(value)``). The result is stripped and
    lower-cased so the mapping tables can be written in canonical lowercase.
    """
    if value is None:
        return None
    if isinstance(value, str):
        token = value
    else:
        raw = getattr(value, "value", None)
        if isinstance(raw, str):
            token = raw
        else:
            token = getattr(value, "name", None) or str(value)
    token = token.strip().lower()
    return token or None
