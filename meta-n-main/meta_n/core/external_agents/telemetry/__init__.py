"""Telemetry spine for the external-agent backends (plan §7.3-7.6).

This module owns the *write* side of the external-agents telemetry surface: the
version-stamped :class:`AgentRunRecord` schema, the deterministic,
resume-idempotent :func:`compute_run_id`, the secret-masking :func:`redact`,
the non-invasive utility-attribution helper :func:`attribute_utilities`, and
the :class:`AgentTelemetry` writer that flock-appends one JSONL row per
``execute()`` to ``telemetry/agent_runs.jsonl``.

Design contract (plan §7):

* **One ledger, two bases.** Token accounting is *inner* for OpenHands / Terminus
  2 (the agent *is* the executed program); the builtin backend is the documented
  exception and reports native *outer* tokens. Every row stamps
  ``token_basis ∈ {"outer", "inner"}`` and ``cost_basis ∈ {"native_usd",
  "priced_from_tokens"}`` so analysis never blends the two (§7.1, §7.11).
* **Deterministic run_id.** ``run_id = sha1(f"{generation}:{candidate_id}:
  {task_id}:{depth}")[:16]`` — *not* uuid-based. The ``ext-{task}-{uuid}``
  session/container name is OS-isolation only and is **not** the run_id. A
  re-executed task on resume yields the *same* run_id, so de-dup-on-read drops
  the duplicate instead of appending a second line (§7.3).
* **Non-invasive attribution.** We never instrument the staged helper source.
  Utility usage is recovered behaviorally from ``AgentRunResult.command_history``
  plus ``InjectionPlan.utilities_available``, with a hard ``None`` (unmeasurable
  on this backend) vs ``[]`` (measured, none used) distinction, and false
  positives for helpers named ``solve``/``run`` are avoided (§7.6).
* **Pointers, not blobs.** Native logs land under ``archive/.../agent_logs/`` and
  the JSONL row carries only relpaths (``transcript_ptr`` etc.) from
  ``output_dir`` (§7.7).
* **Pure-import.** WAVE 2: imports only stdlib + zero-/wave-1 ``external_agents``
  siblings + ``meta_n.core.meta_layer.Trace``; reuses the ``LLMIOLogger``
  flock-append write path (§7.7). No ``openhands`` / ``terminal_bench`` /
  ``docker`` import at module scope.
"""

from __future__ import annotations

from .attribution import (
    _GENERIC_HELPER_NAMES,
    _bash_helper_file_used,
    _bash_token_used,
    _py_helper_used,
    attribute_utilities,
)
from .records import SCHEMA_VERSION, AgentRunRecord, compute_run_id
from .redaction import _REDACTED, _SECRET_PATTERNS, _resolve_priced_model, redact
from .writer import (
    _DEGRADED_TERMINATED_BY,
    AgentTelemetry,
    iter_jsonl_objects,
    logger,
)

__all__ = [
    "SCHEMA_VERSION",
    "AgentRunRecord",
    "redact",
    "compute_run_id",
    "attribute_utilities",
    "iter_jsonl_objects",
    "AgentTelemetry",
]
