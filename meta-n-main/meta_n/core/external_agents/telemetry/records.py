"""Record schemas + deterministic run_id for the external-agents telemetry.

Split out of the former single-module ``telemetry.py`` (mechanical move; see
the package ``__init__`` for the full design contract). Owns the version-
stamped :class:`AgentRunRecord` row and the resume-idempotent
:func:`compute_run_id`.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Optional

from ..terminated import TerminatedBy
from .redaction import redact

__all__ = [
    "SCHEMA_VERSION",
    "AgentRunRecord",
    "compute_run_id",
]

# ---------------------------------------------------------------------------
# Version stamp
# ---------------------------------------------------------------------------
#: Bumped whenever the on-disk ``AgentRunRecord`` field set or the basis enum
#: vocabulary changes. Written once into ``telemetry/schema.json`` and stamped
#: onto every row so a mixed-version cross-run concat can be filtered.
#:
#: 2 — added ``AgentRunRecord.benchmark`` (additive; v1 rows read as "");
#:     removed the never-produced ``StepRecord``/``agent_steps.jsonl``
#:     channel (no producer ever existed; zero rows were ever written
#:     anywhere, so there is no v1 steps population to migrate). A resumed
#:     pre-v2 run dir gets its ``schema.json`` refreshed to v2 on the next
#:     writer construction; the rows themselves carry per-row
#:     ``schema_version`` stamps, so mixed v1/v2 files stay filterable.
#: 1 — initial AgentRunRecord/StepRecord schema.
SCHEMA_VERSION: int = 2

# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass
class AgentRunRecord:
    """One row in ``telemetry/agent_runs.jsonl`` — a single ``execute()`` (§7.3).

    The field set is **identical across backends** (OpenHands / Terminus 2 /
    builtin) — that uniformity is the fair-comparison requirement (§7.3, §7.11).
    Bases that would otherwise be silently mixed are carried as first-class
    discriminators (``token_basis``, ``cost_basis``) and the
    ``False``-vs-``None`` attribution distinction is preserved.

    Groups (plan §7.3):
        identity/coordinate — ``run_id``, ``generation``, ``candidate_id``,
            ``task_id``, ``agent``, ``depth``, ``parent_run_id``, ``phase``,
            ``benchmark``.
        token/cost — ``prompt_tokens``/``completion_tokens``/``total_tokens``/
            ``cached_tokens``/``cost_usd``, ``inner_tokens``/``inner_calls``,
            ``token_basis``, ``cost_basis``.
        control — ``steps``, ``max_turns``, ``terminated_by``.
        scoring — ``score``, ``raw_score``, ``success``, ``feasibility``,
            ``validity``.
        attribution — ``utilities_available``, ``utilities_called``,
            ``utilities_call_counts``, ``attribution_available``,
            ``pre_process_ran``, ``command_count`` (measured-zero vs lost-stream
            discriminator).
        timing — ``wall_s``, ``provision_s``, ``agent_s``, ``score_s``.
        diagnostics — ``failure_mode``, ``error_summary`` (≤200), ``eval_feedback``.
        pointers — ``transcript_ptr``, ``agent_logs_ptr`` (relpaths from
            ``output_dir`` root, never inline blobs).
    """

    # --- identity / coordinate --------------------------------------------
    run_id: str
    generation: int = 0
    candidate_id: str = ""
    task_id: str = ""
    agent: str = "builtin"
    depth: int = 1
    parent_run_id: Optional[str] = None
    #: Execution phase this row was produced under — ``"eval"`` (the default /
    #: full evaluation) or ``"gate"`` (the gate-check pre-screen). Co-determines
    #: the ``run_id`` basis (see :func:`compute_run_id`) and lets the fair-
    #: comparison reader default to the eval-phase rows so a gate subset is not
    #: double-counted. Rows predating this field are treated as ``"eval"``.
    phase: str = "eval"
    #: The benchmark this row was produced under (the adapter's ``name``, e.g.
    #: ``"co_bench"`` / ``"terminal_bench"``). Disambiguates same-named agents
    #: across benchmarks: the CO-Bench ``OpenHandsBackend`` and the TB
    #: ``OpenHandsTBBackend`` both report ``agent="openhands"`` (likewise the two
    #: ``builtin`` backends). Rows predating this field are read as ``""``
    #: (unknown) and are never split on.
    benchmark: str = ""

    # --- token / cost -----------------------------------------------------
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    cost_usd: float = 0.0
    inner_tokens: int = 0
    inner_calls: int = 0
    #: ``"outer"`` for builtin, ``"inner"`` for OpenHands/Terminus 2 (§7.1,§7.11).
    token_basis: str = "inner"
    #: ``"native_usd"`` for OpenHands, ``"priced_from_tokens"`` for T2/builtin.
    cost_basis: str = "priced_from_tokens"

    # --- control ----------------------------------------------------------
    steps: int = 0
    max_turns: int = 0
    terminated_by: str = TerminatedBy.UNKNOWN.value

    # --- scoring ----------------------------------------------------------
    score: float = 0.0
    raw_score: float = 0.0
    success: bool = False
    feasibility: bool = True
    validity: bool = True

    # --- attribution ------------------------------------------------------
    utilities_available: list[str] = field(default_factory=list)
    #: ``None`` == unmeasurable on this backend; ``[]`` == measured, none used.
    utilities_called: Optional[list[str]] = None
    utilities_call_counts: dict[str, int] = field(default_factory=dict)
    attribution_available: bool = False
    pre_process_ran: bool = False
    #: Number of commands captured in ``run.command_history``. This is the
    #: load-bearing discriminator between a MEASURED-zero (``command_count > 0``
    #: AND ``utilities_called == []`` → a real command stream was captured, none
    #: of the staged utilities were called) and a LOST/ambiguous stream
    #: (``command_count == 0`` → no stream was captured even though
    #: ``attribution_available`` may be True, so a ``[]`` / ``None``
    #: ``utilities_called`` is non-behavioral, NOT a measured zero). A serialized
    #: ``command_count`` makes the two provably distinct on read.
    command_count: int = 0

    # --- timing -----------------------------------------------------------
    wall_s: float = 0.0
    provision_s: float = 0.0
    agent_s: float = 0.0
    score_s: float = 0.0

    # --- diagnostics ------------------------------------------------------
    failure_mode: Optional[str] = None
    error_summary: str = ""
    eval_feedback: str = ""

    # --- pointers (relpaths from output_dir) ------------------------------
    transcript_ptr: Optional[str] = None
    agent_logs_ptr: Optional[str] = None

    # --- provenance -------------------------------------------------------
    schema_version: int = SCHEMA_VERSION

    def to_row(self) -> dict:
        """Serialize to a JSON-ready dict with all free-text fields redacted.

        ``error_summary``/``eval_feedback`` are truncated to 200 chars (the
        ``error_summary[:200]`` invariant, §7.3) and run through :func:`redact`,
        as are ``failure_mode`` and the command-derived attribution names.
        """
        row = asdict(self)
        row["error_summary"] = redact((self.error_summary or "")[:200])
        row["eval_feedback"] = redact(self.eval_feedback or "")
        if self.failure_mode is not None:
            row["failure_mode"] = redact(self.failure_mode)
        return row


# ---------------------------------------------------------------------------
# Deterministic run_id (plan §7.3)
# ---------------------------------------------------------------------------


def compute_run_id(
    generation: object,
    candidate_id: object,
    task_id: object,
    depth: object,
    phase: object = "eval",
    repeat_index: object = 0,
) -> str:
    """Return the deterministic, resume-idempotent ``run_id`` (plan §7.3).

    ``run_id = sha1(f"{generation}:{candidate_id}:{task_id}:{depth}")[:16]`` for
    the default/full-evaluation ``phase`` and ``repeat_index == 0``. It depends
    *only* on the four evaluation coordinates (plus the optional phase / repeat
    discriminators) and is **independent** of any session/container uuid (the
    ``ext-{task}-{uuid}`` name is OS-isolation only). Determinism is what makes
    the no-truncation resume correct: a re-executed task yields the same id, so
    de-dup-on-read drops the duplicate row instead of appending a second one.

    Execution-phase discriminator (de-dup correctness): a depth>1 (or any)
    spine candidate runs a subset of tasks TWICE — once in ``_gate_check`` and
    again in the full evaluation — at IDENTICAL ``(generation, candidate_id,
    task_id, depth)`` coordinates. Without a discriminator the two executions
    collide on one run_id and the full-eval row de-dups against the gate row, so
    ``agent_runs.jsonl`` undercounts. Folding ``phase`` into the basis for the
    NON-default phase (``"gate"``) gives the gate and eval executions distinct
    ids while keeping resume idempotency WITHIN each phase (a re-run gate still
    de-dups against the prior gate row).

    Repeat-index discriminator (median-of-R / gate-repeats de-dup): under
    ``--eval-repeats R`` (or ``--gate-repeats R``) the SAME ``(generation,
    candidate_id, task_id, depth, phase)`` coordinates are executed R times. Each
    sample carries a distinct ``repeat_index`` (the orchestrator stamps
    ``solver.repeat_index``), folded into the basis as ``:r<index>`` ONLY when it
    is truthy/non-zero, so the R samples write R distinct telemetry rows instead
    of de-dupping down to one (an ~R-fold undercount).

    The default ``phase in (None, "eval")`` with ``repeat_index`` falsy keeps the
    basis **byte-identical** to the historical formula, so existing run_ids (and
    any persisted / cross-run rows) are unchanged — only ``"gate"`` rows and
    non-zero repeat samples get a distinct id.

    Args:
        generation: The evolutionary generation / iteration index.
        candidate_id: The candidate chain id.
        task_id: The benchmark task id.
        depth: The solver depth.
        phase: The execution phase — ``"eval"`` (default / full evaluation) or
            ``"gate"`` (the gate-check pre-screen). Only a non-``eval`` phase
            changes the id.
        repeat_index: The 0-based repeated-eval / gate-repeat sample index. Only
            a truthy (non-zero) value changes the id.

    Returns:
        A 16-hex-char prefix of the SHA-1 of the joined coordinates.
    """
    basis = f"{generation}:{candidate_id}:{task_id}:{depth}"
    if phase not in (None, "eval"):
        basis += f":{phase}"
    if repeat_index:
        basis += f":r{repeat_index}"
    return hashlib.sha1(basis.encode("utf-8")).hexdigest()[:16]
