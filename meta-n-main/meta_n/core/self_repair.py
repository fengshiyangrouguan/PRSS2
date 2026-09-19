"""Self-repair event substrate (multi-granularity science, Stages 2 & 3).

This module owns the NON-behavioral record + sidecar writer that the two
error-driven self-repair features emit:

* **within-layer refine** (Stage 2): one extra Ω call that *fixes a buggy
  injection* on the gate-fail / regression-vs-parent branch — granularity
  ``"within_layer"``.
* **downward re-propagation** (Stage 3): regenerate an intermediate layer's
  injection from full-chain failure traces + the above-layer injections as
  downstream feedback — granularity ``"downstream"``.

Both are the same error-driven self-repair pattern at two granularities (the
plan's unifying insight). Each repair attempt emits ONE
:class:`SelfRepairEvent`. The record is a Pydantic twin of
:class:`~meta_n.core.external_agents.telemetry.AgentRunRecord` (a ``@dataclass``)
— the plan asks for a Pydantic record so it round-trips through
``model_dump``/``model_validate`` like every other meta-n core model.

Monotonic-archive discipline: a repair NEVER mutates the parent candidate or any
archived trace. It produces a NEW candidate (``candidate_id``) whose
``parent_id`` is the repaired candidate (``parent_candidate_id``); the event is a
provenance record of *that* construction, persisted as a candidate sidecar
mirroring the injected-code sidecar pattern in
``evolutionary_orchestrator._save_candidate_incremental``
(``repropagation_d{t}.json`` + a paired ``.txt``).

**Default-OFF byte-identity:** every current candidate carries an EMPTY
``self_repair_events`` list, so :func:`write_self_repair_sidecars` writes nothing
and ``_save_candidate_incremental`` stays byte-for-byte identical (the no-event
guard mirrors the existing ``if rollup is not None`` rollup guard). Only a
candidate that an opted-in Stage-2/3 loop actually repaired emits a sidecar.

Pure-import: stdlib + ``pydantic`` only. No ``openhands`` / ``terminal_bench`` /
``docker`` import, and no import of the orchestrator (the writer takes a plain
directory + the events, so this module never depends on the archive/orchestrator
that consume it).
"""

from __future__ import annotations

import difflib
import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from pydantic import BaseModel, Field

if TYPE_CHECKING:  # pragma: no cover - typing only (no runtime import cycle)
    from meta_n.core.meta_layer import InjectedCode

logger = logging.getLogger(__name__)

__all__ = [
    "SELF_REPAIR_SCHEMA_VERSION",
    "GRANULARITIES",
    "REPAIR_CLASSES",
    "SelfRepairEvent",
    "classify_repair",
    "write_self_repair_sidecars",
]

#: Bumped whenever the on-disk :class:`SelfRepairEvent` field set changes. Stamped
#: onto every event so a mixed-version cross-run read can be filtered (mirrors
#: ``telemetry.SCHEMA_VERSION``).
SELF_REPAIR_SCHEMA_VERSION: int = 2

#: The two self-repair granularities. ``within_layer`` == Stage-2 refine (fix the
#: just-generated injection); ``downstream`` == Stage-3 re-propagation (revise an
#: intermediate layer). The within-SOLVE granularity (R1 error-hint) emits NO
#: event — it is a base-agent observation lever, not an Ω self-repair (per the
#: plan: "R1 emits no event").
GRANULARITIES: frozenset[str] = frozenset({"within_layer", "downstream"})

#: The two classifier verdicts written into :attr:`SelfRepairEvent.classification`
#: by :func:`classify_repair`. ``error_correction`` == the repair kept the same
#: approach and made a localized fix (the FEAL-bound expectation for a
#: knowledge-bound same-model generator); ``novel`` == the repair changed the
#: approach (new helper / different algorithm class / failure-mode shift / drastic
#: rewrite). ``None`` (un-classified) is also a valid stored value.
REPAIR_CLASSES: frozenset[str] = frozenset({"error_correction", "novel"})

# Classifier thresholds (module-level so they are visible + tunable + testable).
#: Below this word-set Jaccard between the pre/post rationales the repair is a
#: rationale drift → ``novel``.
_RATIONALE_OVERLAP_MIN: float = 0.30
#: Below this :class:`difflib.SequenceMatcher` ratio between the pre/post
#: ``pre_process`` bodies the edit is a drastic rewrite (not localized) → ``novel``.
_PRE_PROCESS_SIM_MIN: float = 0.50


def _helper_names(ic: "InjectedCode | None") -> set[str]:
    """The union of an injection's Python + bash helper names (``set()`` if none)."""
    if ic is None:
        return set()
    return set(getattr(ic, "code_library", {}) or {}) | set(
        getattr(ic, "code_library_bash", {}) or {}
    )


def _word_set(text: str | None) -> set[str]:
    """Lower-cased identifier/word tokens of ``text`` (for rationale Jaccard)."""
    return set(re.findall(r"[a-z0-9_]+", (text or "").lower()))


def classify_repair(
    pre: "InjectedCode | None",
    post: "InjectedCode | None",
    *,
    pre_failure_class: str | None = None,
    post_failure_class: str | None = None,
    rationale_overlap_min: float = _RATIONALE_OVERLAP_MIN,
    pre_process_sim_min: float = _PRE_PROCESS_SIM_MIN,
) -> str:
    """Classify a pre→post injection repair as ``error_correction`` vs ``novel``.

    The Stage-3 error-correction-vs-novel classifier — the thing that tests the
    FEAL bound (a knowledge-bound same-model generator should be doing
    error-correction, not novel generation). It is a STRUCTURAL diff over the two
    injections (``meta_layer.classify_error`` supplies the optional failure-class
    signal via the caller), returning a label written into
    :attr:`SelfRepairEvent.classification`.

    ``novel`` if ANY of these fire (the repair changed the approach):

      1. **new helper** — ``post`` introduces a ``code_library`` /
         ``code_library_bash`` name absent from ``pre``;
      2. **failure-class shift** — ``pre_failure_class`` and ``post_failure_class``
         are both known and DIFFER (the intervention now targets a different
         failure mode / algorithm class);
      3. **rationale drift** — the pre/post rationale word-set Jaccard is below
         ``rationale_overlap_min`` (a different explanation = a different idea);
      4. **drastic rewrite** — the pre/post ``pre_process`` similarity
         (:class:`difflib.SequenceMatcher` ratio) is below ``pre_process_sim_min``
         (not a localized edit).

    Otherwise (same helpers, same failure target, overlapping rationale, localized
    edit) the verdict is ``error_correction``. Deterministic + pure: no LLM, no
    I/O — safe to call from the re-propagation loop (and the within-layer refine).

    A fully-resolved repair leaves ``post_failure_class`` empty (the post traces
    succeed), so signal 2 only fires when BOTH sides still fail with DIFFERENT
    classes — it never mislabels a successful localized fix as ``novel``.
    """
    # 1. New helper introduced → different algorithm surface → novel.
    if _helper_names(post) - _helper_names(pre):
        return "novel"

    # 2. Failure class shifted (both known + different) → novel.
    if pre_failure_class and post_failure_class and pre_failure_class != post_failure_class:
        return "novel"

    # 3. Rationale drift (low word-set Jaccard) → novel.
    pre_words, post_words = _word_set(getattr(pre, "rationale", "")), _word_set(
        getattr(post, "rationale", "")
    )
    if pre_words or post_words:
        inter = len(pre_words & post_words)
        union = len(pre_words | post_words) or 1
        if inter / union < rationale_overlap_min:
            return "novel"

    # 4. Drastic pre_process rewrite (low char-sequence similarity) → novel.
    pre_pp = getattr(pre, "pre_process", "") or ""
    post_pp = getattr(post, "pre_process", "") or ""
    if pre_pp or post_pp:
        sim = difflib.SequenceMatcher(None, pre_pp, post_pp).ratio()
        if sim < pre_process_sim_min:
            return "novel"

    return "error_correction"


class SelfRepairEvent(BaseModel):
    """One error-driven self-repair of a buggy Ω injection (Stage 2 / Stage 3).

    A pure provenance record: it describes the construction of a NEW candidate
    (``candidate_id``, ``parent_id == parent_candidate_id``) from a repair of an
    existing one — it never carries a reference that would let a reader mutate the
    parent. The pre/post injection are referenced by a ``code_hash`` (the
    sha256-prefix of the revised layer's ``pre_process``, parity
    ``Trace.code_hash``) and/or sidecar relpaths, never inline blobs; the raw Ω
    transcript of the repair call rides in ``raw_omega_prompt`` /
    ``raw_omega_response`` (excluded from the JSON sidecar, written to the paired
    ``.txt`` — mirroring how ``InjectedCode`` excludes its raw Ω text from
    ``injected_code_d{N}.json``).

    Fields (plan substrate spec):
        identity/coordinate — ``candidate_id`` (the NEW candidate),
            ``parent_candidate_id`` (the repaired candidate), ``granularity``
            (``within_layer`` | ``downstream``), ``target_depth`` (the layer
            depth whose injection was revised).
        injection refs — ``pre_code_hash`` / ``post_code_hash`` (the revised
            layer's ``pre_process`` hash before/after), ``pre_injection_ref`` /
            ``post_injection_ref`` (optional sidecar relpaths),
            ``omega_prompt_ref`` / ``omega_response_ref`` (optional relpaths to
            the repair-call transcript). As of schema v2, ``pre_injection_ref``
            / ``omega_prompt_ref`` / ``omega_response_ref`` are RESERVED and
            always serialize as ``null`` — only ``post_injection_ref`` is
            populated (by the orchestrator at event construction). A ``null``
            ref means "not wired yet", never "transcript absent": the repair
            transcript lives at the sidecar's paired ``{stem}.txt`` by naming
            convention regardless. Populating any of the three reserved refs
            MUST bump ``SELF_REPAIR_SCHEMA_VERSION`` and update the round-trip
            tests.
        scoring — ``mean_before`` / ``mean_after`` (the repaired-vs-original mean
            scored over a MATCHED denominator — the gate subset for
            ``within_layer``, the child's per-task set for ``downstream``),
            ``mean_after_full`` (the refined candidate's full evaluated mean,
            recorded for reference), ``per_task_delta`` (per-task
            ``after - before``), ``accepted`` (the SCORE-IMPROVEMENT verdict:
            ``mean_after >= mean_before`` on the matched denominator — NOT the
            keep decision), ``archived`` (the KEEP decision: ``within_layer``
            archives iff it cleared the gate, ``downstream`` archives
            unconditionally).
        classification — ``classification`` (``error_correction`` | ``novel`` |
            ``None``): filled by the Stage-3 error-correction-vs-novel classifier;
            ``None`` until classified.
        raw transcript — ``raw_omega_prompt`` / ``raw_omega_response`` (excluded
            from the JSON sidecar; written to the paired ``.txt``).
    """

    # --- identity / coordinate --------------------------------------------
    candidate_id: str = ""
    parent_candidate_id: str = ""
    #: ``"within_layer"`` (Stage-2 refine) or ``"downstream"`` (Stage-3
    #: re-propagation). Not validated to the enum here so an unknown future
    #: granularity round-trips rather than raising on read.
    granularity: str = "within_layer"
    target_depth: int = 0

    # --- injection refs (hashes + optional sidecar relpaths) --------------
    pre_code_hash: str = ""
    post_code_hash: str = ""
    # pre_injection_ref / omega_prompt_ref / omega_response_ref are RESERVED at
    # schema v2 (always null; populating them requires a version bump — see the
    # class docstring). Only post_injection_ref is populated today.
    pre_injection_ref: Optional[str] = None
    post_injection_ref: Optional[str] = None
    omega_prompt_ref: Optional[str] = None
    omega_response_ref: Optional[str] = None

    # --- scoring ----------------------------------------------------------
    #: ``mean_before`` and ``mean_after`` are scored over the SAME task keys (a
    #: MATCHED denominator) so their delta is comparable: the gate subset for
    #: ``within_layer``, the revised child's per-task set for ``downstream``.
    #: ``mean_after`` is NOT the refined candidate's full evaluated mean — see
    #: ``mean_after_full`` for that.
    mean_before: float = 0.0
    mean_after: float = 0.0
    #: The refined / re-propagated candidate's FULL evaluated mean (all tasks),
    #: recorded alongside the matched-denominator ``mean_after`` for reference.
    #: ``0.0`` when not separately recorded.
    mean_after_full: float = 0.0
    per_task_delta: dict[str, float] = Field(default_factory=dict)
    #: The SCORE-IMPROVEMENT verdict — ``mean_after >= mean_before`` on the
    #: matched denominator, unified across both granularities so a
    #: cross-granularity accept-rate compares like with like. This is NOT the
    #: keep decision: a ``within_layer`` refine can be archived (it cleared the
    #: gate) while ``accepted`` is False (the subset mean did not improve). See
    #: ``archived`` for the keep bit.
    accepted: bool = False
    #: The KEEP decision — whether the repaired candidate was archived. A
    #: ``within_layer`` refine is archived iff it CLEARED THE GATE; a
    #: ``downstream`` re-propagation is archived unconditionally. Distinct from
    #: ``accepted`` (the score-improvement verdict).
    archived: bool = False

    # --- classification (filled by the Stage-3 classifier; None until set) -
    classification: Optional[str] = None

    # --- raw Ω repair transcript (excluded from JSON; written to .txt) -----
    raw_omega_prompt: str = ""
    raw_omega_response: str = ""

    # --- provenance -------------------------------------------------------
    schema_version: int = SELF_REPAIR_SCHEMA_VERSION

    def to_sidecar_json(self) -> dict:
        """Serialize for ``repropagation_d{t}.json`` — raw Ω text excluded.

        Mirrors the injected-code sidecar (``injected_code_d{N}.json`` excludes
        ``raw_omega_prompt``/``raw_omega_response`` and persists them as paired
        ``.txt`` instead) so the JSON stays compact + diff-friendly.
        """
        return self.model_dump(exclude={"raw_omega_prompt", "raw_omega_response"})

    def to_sidecar_text(self) -> str:
        """Render the paired ``repropagation_d{t}.txt`` (human-diffable).

        Carries the raw Ω repair-call transcript (prompt → response) framed by a
        compact header so a reviewer can read what the refine/re-propagation
        actually changed. Returns ``""`` ONLY when there is no transcript AND no
        header would add signal (so the writer can skip an empty file) — in
        practice the header is always emitted, so the ``.txt`` is non-empty
        whenever an event is written.
        """
        head = (
            f"# self-repair: {self.granularity} @ depth d{self.target_depth}\n"
            f"# candidate={self.candidate_id} parent={self.parent_candidate_id}\n"
            f"# accepted={self.accepted} archived={self.archived} "
            f"mean_before={self.mean_before} mean_after={self.mean_after} "
            f"mean_after_full={self.mean_after_full}\n"
            f"# classification={self.classification}\n"
            f"# pre_code_hash={self.pre_code_hash} "
            f"post_code_hash={self.post_code_hash}\n"
        )
        parts = [head]
        if self.raw_omega_prompt:
            parts.append("\n===== Ω REFINE PROMPT =====\n")
            parts.append(self.raw_omega_prompt)
        if self.raw_omega_response:
            parts.append("\n===== Ω REFINE RESPONSE =====\n")
            parts.append(self.raw_omega_response)
        return "".join(parts)


def write_self_repair_sidecars(
    cand_dir: Path, events: Optional[list["SelfRepairEvent"]]
) -> list[Path]:
    """Write each event as ``repropagation_d{t}.json`` + paired ``.txt``.

    The candidate-sidecar writer for self-repair provenance, mirroring the
    injected-code sidecar loop in
    ``evolutionary_orchestrator._save_candidate_incremental``. **No-event
    byte-identity guard:** an empty / ``None`` ``events`` writes NOTHING and
    returns ``[]`` — exactly the path every current candidate takes — so the
    default ``_save_candidate_incremental`` output is byte-for-byte unchanged
    (the new sidecars appear ONLY for a candidate an opted-in Stage-2/3 loop
    repaired).

    Filenames key on ``target_depth`` (``repropagation_d{t}``); on the rare
    collision of two events at the same depth (e.g. a within-layer refine AND a
    downstream re-propagation both touching depth ``t``) the second+ get a
    deterministic ``_<n>`` suffix so neither overwrites the other.

    Best-effort: a failed write logs a warning and is skipped rather than sinking
    the candidate save (the sidecar is diagnostic, not load-bearing). Stdlib only.

    This loop is the designated future wiring chokepoint for the reserved
    ``omega_prompt_ref`` / ``omega_response_ref`` fields (``stem`` is in scope
    here; they would point at ``f"{stem}.txt"``), gated on a
    ``SELF_REPAIR_SCHEMA_VERSION`` bump.

    Args:
        cand_dir: The candidate's archive directory (``archive/<candidate_id>``).
        events: The candidate's ``self_repair_events`` (may be ``None`` / empty).

    Returns:
        The list of files actually written (``[]`` when no event was emitted).
    """
    if not events:
        return []
    written: list[Path] = []
    seen_depths: dict[int, int] = {}
    for event in events:
        try:
            t = int(getattr(event, "target_depth", 0) or 0)
            n = seen_depths.get(t, 0)
            seen_depths[t] = n + 1
            stem = f"repropagation_d{t}" if n == 0 else f"repropagation_d{t}_{n}"

            json_path = cand_dir / f"{stem}.json"
            with open(json_path, "w") as f:
                json.dump(event.to_sidecar_json(), f, indent=2)
            written.append(json_path)

            text = event.to_sidecar_text()
            if text:
                txt_path = cand_dir / f"{stem}.txt"
                with open(txt_path, "w") as f:
                    f.write(text)
                written.append(txt_path)
        except Exception:  # noqa: BLE001 - a lost sidecar must not sink the save
            logger.warning(
                "failed to write self-repair sidecar for candidate dir %s "
                "(target_depth=%s); skipping",
                cand_dir,
                getattr(event, "target_depth", "?"),
                exc_info=True,
            )
    return written
