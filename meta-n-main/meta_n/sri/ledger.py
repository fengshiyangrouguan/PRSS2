"""Proposal-slot ledger: one terminal row per NOMINAL recursive slot (§5, §2.3).

Why a ledger at all: the regression estimator (§2.2) must count rejected and
failed children, not just archive survivors. The search archive cannot hold
them (persisting a rejected candidate could make it eligible for parent
selection or Final Score), so every nominal slot gets a row here instead, and
the candidate material needed to re-execute it lives OUTSIDE the archive.

APPEND-ONLY, FOLDED ON READ. The file gets two record kinds:

    {"kind":"open",     ...pre-dispatch fields..., "created_at": ...}
    {"kind":"finalize", "slot_id": ..., ...terminal fields..., "finalized_at": ...}

`read_ledger` folds them into exactly one row per slot. Writing is append-only
so a crash can never corrupt an earlier observation; `finalize` refuses a second
finalization of the same slot, which is what makes "exactly once" checkable
rather than merely intended.

`resume_already_completed` (§5) is a FINALIZATION KIND, not a new slot: a resumed
iteration that re-enters an already-finalized slot references it and adds no
second experimental observation.

The nominal grid is enumerable -- `nominal_slot_ids(B, K, T)` -- so acceptance
criterion 1 ("every nominal recursive slot has exactly one terminal ledger row")
is a set comparison, not a hope.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from meta_n.sri.protocol import LEDGER_SCHEMA_VERSION, ProtocolError

# §5 terminal statuses. `evaluated_admitted`/`evaluated_rejected` are the two
# outcomes of a child that ran; the rest are the ways a slot terminates without
# a usable candidate. A slot is ALWAYS one of these -- there is no "unknown".
TERMINAL_STATUSES = frozenset({
    "evaluated_admitted",
    "evaluated_rejected",
    "gate_rejected",
    "empty_injection",
    "generation_error",
    "invalid_program",
    "execution_error",
    "timeout",
    "nonfinite_score",
    "no_eligible_parent",
    "budget_halt",
    "resume_already_completed",
})

# Terminal statuses that carry an EXECUTABLE child which the canonical audit must
# still evaluate (§2.2: "accepted, rejected, and gate-rejected children whenever
# the child is executable and can be canonically evaluated").
EXECUTABLE_STATUSES = frozenset({
    "evaluated_admitted",
    "evaluated_rejected",
    "gate_rejected",
    "nonfinite_score",     # ran, but produced a non-finite score
})

# §2.3: a slot that produced no valid edge-task observation at all.
FAILED_STATUSES = frozenset({
    "empty_injection",
    "generation_error",
    "invalid_program",
    "execution_error",
    "timeout",
    "no_eligible_parent",
    "budget_halt",
})


def slot_id_for(iteration: int, parent_slot: int, child_slot: int) -> str:
    """Stable, enumerable slot identity -- the nominal grid is built from these."""
    return "it{}-p{}-c{}".format(int(iteration), int(parent_slot),
                                 int(child_slot))


def nominal_slot_ids(B: int, K: int, T: int) -> List[str]:
    """Every nominal `(iteration, parent_slot, child_slot)` of one arm.

    `iteration` runs 0..T-1, `parent_slot` 0..B-1, `child_slot` 0..K-1, so the
    grid size is exactly B*K*T (§3.3). These are per SEED; a second search seed
    is a second ledger (its own run_id), never extra slots in this one.
    """
    out = []
    for it in range(int(T)):
        for p in range(int(B)):
            for c in range(int(K)):
                out.append(slot_id_for(it, p, c))
    return out


class SlotLedger:
    """Append-only ledger for one `(run_id, arm, search_seed)`.

    A logging failure is FATAL (§13): the ledger raises rather than degrading to
    a warning, because unlogged paid work must not continue.
    """

    def __init__(self, path, *, run_id: str, arm: str, backbone: str,
                 cohort_id: str, search_seed: int,
                 root_bundle_sha256: str,
                 gamma_checkpoint_sha256: Optional[str] = None,
                 schema_version: int = LEDGER_SCHEMA_VERSION) -> None:
        if not run_id or not arm:
            raise ProtocolError("ledger needs run_id and arm")
        if not root_bundle_sha256:
            raise ProtocolError("ledger needs root_bundle_sha256 (§4)")
        if arm == "predictive" and not gamma_checkpoint_sha256:
            raise ProtocolError(
                "the predictive arm must record its Gamma checkpoint hash")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._hdr = {
            "schema_version": int(schema_version),
            "run_id": run_id, "arm": arm, "backbone": backbone,
            "cohort_id": cohort_id, "search_seed": int(search_seed),
            "root_bundle_sha256": root_bundle_sha256,
            "gamma_checkpoint_sha256": gamma_checkpoint_sha256,
        }
        self._open: Dict[str, Dict[str, Any]] = {}
        self._final: Dict[str, Dict[str, Any]] = {}
        self._load()

    # -- persistence -------------------------------------------------------
    def _load(self) -> None:
        if not self.path.is_file():
            return
        for line in self.path.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError as e:
                raise ProtocolError(
                    "slot ledger {!r} has a corrupt line: {}".format(
                        str(self.path), e))
            sid = rec.get("slot_id")
            if rec.get("kind") == "open":
                self._open[sid] = rec
            elif rec.get("kind") == "finalize":
                if sid in self._final:
                    raise ProtocolError(
                        "slot {!r} was finalized twice in {!r}".format(
                            sid, str(self.path)))
                self._final[sid] = rec

    def _append(self, rec: Dict[str, Any]) -> None:
        rec = dict(rec)
        rec.update({k: v for k, v in self._hdr.items()
                    if k not in rec})
        try:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, sort_keys=True) + "\n")
                f.flush()
                os.fsync(f.fileno())
        except OSError as e:                                     # noqa: BLE001
            # §13: a logging failure is fatal before the next paid call.
            raise ProtocolError(
                "slot ledger write failed ({}); refusing to continue unlogged"
                .format(e))

    # -- lifecycle ---------------------------------------------------------
    def open_slot(self, *, iteration: int, parent_slot: int, child_slot: int,
                  parent_id: Optional[str], parent_structural_depth: int,
                  gate_configured: int, gate_effective: bool,
                  gate_reason: str, reduction_mode: str,
                  proposed_child_id: Optional[str] = None,
                  proposed_child_depth: Optional[int] = None,
                  temperature: Optional[float] = None,
                  focus_task: Optional[str] = None) -> str:
        """Create the slot row BEFORE dispatch. Returns the slot_id."""
        sid = slot_id_for(iteration, parent_slot, child_slot)
        if sid in self._open:
            raise ProtocolError(
                "slot {!r} already opened -- a nominal slot is opened once; a "
                "resumed iteration must reuse the existing row, not add one"
                .format(sid))
        rec = {
            "kind": "open", "slot_id": sid,
            "iteration": int(iteration), "parent_slot": int(parent_slot),
            "child_slot": int(child_slot),
            "parent_id": parent_id,
            "parent_structural_depth": int(parent_structural_depth),
            "proposed_child_id": proposed_child_id,
            "proposed_child_depth": proposed_child_depth,
            "temperature": temperature, "focus_task": focus_task,
            "reduction_mode": reduction_mode,
            "gate_configured": int(gate_configured),
            "gate_effective": bool(gate_effective),
            "gate_reason": gate_reason,
            "created_at": time.time(),
        }
        self._append(rec)
        # Store the FULL record: `rows()` folds open+finalize, so an in-memory
        # open record missing its pre-dispatch fields would silently drop them.
        self._open[sid] = rec
        return sid

    def finalize(self, slot_id: str, *, terminal_status: str,
                 gate_status: Optional[str] = None,
                 gate_reason: Optional[str] = None,
                 failure_class: Optional[str] = None,
                 failure_message: Optional[str] = None,
                 archive_admitted: Optional[bool] = None,
                 archive_rejection_reason: Optional[str] = None,
                 material_path: Optional[str] = None,
                 fresh_task_ids: Sequence[str] = (),
                 inherited_task_ids: Sequence[str] = (),
                 outer_calls: int = 0, inner_calls: int = 0,
                 evaluator_calls: int = 0,
                 prompt_tokens: int = 0, completion_tokens: int = 0,
                 inner_prompt_tokens: int = 0,
                 inner_completion_tokens: int = 0,
                 omega_wall_seconds: float = 0.0,
                 gate_wall_seconds: float = 0.0,
                 eval_wall_seconds: float = 0.0) -> None:
        """Finalize the slot exactly once."""
        if terminal_status not in TERMINAL_STATUSES:
            raise ProtocolError(
                "unknown terminal_status {!r}; known: {}".format(
                    terminal_status, sorted(TERMINAL_STATUSES)))
        if slot_id not in self._open:
            raise ProtocolError(
                "finalize({!r}) without a matching open_slot -- every slot row "
                "is created before dispatch".format(slot_id))
        if slot_id in self._final:
            raise ProtocolError(
                "slot {!r} already finalized as {!r}; a slot is finalized "
                "exactly once".format(slot_id,
                                      self._final[slot_id]["terminal_status"]))
        if terminal_status == "resume_already_completed":
            raise ProtocolError(
                "resume_already_completed is used by mark_resumed(), not "
                "finalize() -- it references an existing observation")
        wall = (float(omega_wall_seconds) + float(gate_wall_seconds)
                + float(eval_wall_seconds))
        frec = {
            "kind": "finalize", "slot_id": slot_id,
            "terminal_status": terminal_status,
            "gate_status": gate_status, "gate_reason": gate_reason,
            "proposal_status": "finalized",
            "failure_class": failure_class, "failure_message": failure_message,
            "archive_admitted": archive_admitted,
            "archive_rejection_reason": archive_rejection_reason,
            # §5: rejected/failed proposals keep enough immutable material
            # OUTSIDE the search archive to reconstruct an executable child, so
            # the canonical audit can still evaluate them. The path is what
            # makes "not restricted to archive survivors" enforceable.
            "material_path": material_path,
            "fresh_task_ids": list(fresh_task_ids),
            "inherited_task_ids": list(inherited_task_ids),
            "outer_calls": int(outer_calls), "inner_calls": int(inner_calls),
            "evaluator_calls": int(evaluator_calls),
            "prompt_tokens": int(prompt_tokens),
            "completion_tokens": int(completion_tokens),
            "inner_prompt_tokens": int(inner_prompt_tokens),
            "inner_completion_tokens": int(inner_completion_tokens),
            "omega_wall_seconds": float(omega_wall_seconds),
            "gate_wall_seconds": float(gate_wall_seconds),
            "eval_wall_seconds": float(eval_wall_seconds),
            "total_wall_seconds": wall,
            "finalized_at": time.time(),
        }
        self._append(frec)
        self._final[slot_id] = frec     # full record, so rows() folds correctly

    def mark_resumed(self, slot_id: str) -> None:
        """Record that a resumed run re-entered an ALREADY finalized slot.

        Adds no second observation: the referenced slot keeps its original
        terminal status (§5).
        """
        if slot_id not in self._final:
            raise ProtocolError(
                "mark_resumed({!r}) on a slot with no prior finalization"
                .format(slot_id))
        # Appended as a note, not a second finalize record.
        self._append({"kind": "resume_note", "slot_id": slot_id,
                      "references_terminal_status":
                          self._final[slot_id]["terminal_status"],
                      "created_at": time.time()})

    # -- reads -------------------------------------------------------------
    def is_finalized(self, slot_id: str) -> bool:
        return slot_id in self._final

    def rows(self) -> List[Dict[str, Any]]:
        """The folded view: exactly one row per opened slot.

        Only NON-None finalize values override the open record. A finalize that
        does not speak to a field (e.g. it passes gate_reason=None because the
        gate outcome needs no reason) must not erase what the pre-dispatch row
        already established -- that is how a configured value silently became
        null.
        """
        out = []
        for sid, op in self._open.items():
            row = dict(op)
            fin = self._final.get(sid)
            if fin is None:
                row["terminal_status"] = None
                row["proposal_status"] = "open"
            else:
                row.update({k: v for k, v in fin.items() if v is not None})
            out.append(row)
        return sorted(out, key=lambda r: r["slot_id"])

    def unfinalized(self) -> List[str]:
        return sorted(set(self._open) - set(self._final))

    def counts_by_status(self) -> Dict[str, int]:
        c: Dict[str, int] = {}
        for r in self.rows():
            k = r.get("terminal_status") or "OPEN"
            c[k] = c.get(k, 0) + 1
        return c


def read_ledger(path) -> List[Dict[str, Any]]:
    """Fold a ledger file into one row per slot, without an open handle."""
    led = SlotLedger.__new__(SlotLedger)
    led.path = Path(path)
    led._hdr = {}
    led._open, led._final = {}, {}
    led._load()
    return led.rows()


def assert_completeness(ledger: SlotLedger, B: int, K: int, T: int) -> None:
    """Acceptance criterion 1: every nominal slot has exactly one terminal row.

    Raises with the exact missing/extra sets, so an incomplete run is a loud
    failure rather than a quietly smaller denominator.
    """
    want = set(nominal_slot_ids(B, K, T))
    got = {r["slot_id"] for r in ledger.rows()}
    missing = sorted(want - got)
    extra = sorted(got - want)
    open_slots = ledger.unfinalized()
    if missing or extra or open_slots:
        raise ProtocolError(
            "slot ledger is not complete: missing={} extra={} unfinalized={}"
            .format(missing, extra, open_slots))


# --------------------------------------------------------------------------
# acceptance self-test (§14 items 1, 2, 3)
# --------------------------------------------------------------------------

def self_test() -> int:
    import tempfile
    print("ledger.py acceptance")
    print("=" * 68)

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "proposal_slots.jsonl"
        hdr = dict(run_id="r0", arm="official", backbone="gpt-5.5",
                   cohort_id="sri_primary6:6", search_seed=0,
                   root_bundle_sha256="R" * 64)
        led = SlotLedger(p, **hdr)

        # ---- §14.2: an empty injection consumes EXACTLY one slot -----------
        s0 = led.open_slot(iteration=0, parent_slot=0, child_slot=0,
                           parent_id="gen0_seed", parent_structural_depth=1,
                           gate_configured=3, gate_effective=False,
                           gate_reason="consolidation_focus",
                           reduction_mode="official")
        led.finalize(s0, terminal_status="empty_injection",
                     failure_class="empty_omega")
        assert len(led.rows()) == 1
        print("OK  empty injection    consumes exactly one slot ({})".format(s0))

        # ---- §14.1: a gate-REJECTED child is still in the ledger ----------
        s1 = led.open_slot(iteration=0, parent_slot=0, child_slot=1,
                           parent_id="gen0_seed", parent_structural_depth=1,
                           gate_configured=3, gate_effective=False,
                           gate_reason="consolidation_focus",
                           reduction_mode="official",
                           proposed_child_id="gen1_b0_k1")
        led.finalize(s1, terminal_status="gate_rejected",
                     gate_status="rejected", gate_reason="active",
                     archive_admitted=False,
                     archive_rejection_reason="gate")
        row = [r for r in led.rows() if r["slot_id"] == s1][0]
        assert row["terminal_status"] == "gate_rejected"
        assert row["archive_admitted"] is False
        print("OK  gate-rejected      recorded with archive_admitted=False "
              "(it can never become a parent or Final Score)")

        # ---- §14.3: failures are NOT backfilled ---------------------------
        s2 = led.open_slot(iteration=0, parent_slot=0, child_slot=2,
                           parent_id="gen0_seed", parent_structural_depth=1,
                           gate_configured=3, gate_effective=False,
                           gate_reason="consolidation_focus",
                           reduction_mode="official")
        led.finalize(s2, terminal_status="generation_error",
                     failure_class="BadRequestError",
                     failure_message="model_not_provisioned")
        assert len(led.rows()) == 3, len(led.rows())
        print("OK  no backfill        3 nominal slots -> 3 rows, none replaced")

        # ---- exactly once -------------------------------------------------
        try:
            led.finalize(s2, terminal_status="timeout")
            raise AssertionError("a second finalization was accepted")
        except ProtocolError as e:
            assert "exactly once" in str(e)
        print("OK  exactly once       a second finalize() on the same slot refused")

        # ---- opening twice is refused (resume must reuse the row) ---------
        try:
            led.open_slot(iteration=0, parent_slot=0, child_slot=0,
                          parent_id="gen0_seed", parent_structural_depth=1,
                          gate_configured=3, gate_effective=False,
                          gate_reason="consolidation_focus",
                          reduction_mode="official")
            raise AssertionError("a duplicate open was accepted")
        except ProtocolError as e:
            assert "already opened" in str(e)
        print("OK  reopen refused     a resumed slot reuses its row")

        # ---- resume_already_completed is a reference, not a second obs ----
        led.mark_resumed(s1)
        assert [r for r in led.rows() if r["slot_id"] == s1][0][
            "terminal_status"] == "gate_rejected"
        print("OK  resume reference   mark_resumed() adds no second observation")

        # ---- append-only, fold-on-read survives a reload ------------------
        led2 = SlotLedger(p, **hdr)
        assert [r["slot_id"] for r in led2.rows()] == \
               [r["slot_id"] for r in led.rows()]
        assert led2.counts_by_status()["gate_rejected"] == 1
        print("OK  fold on read       reload reproduces the same rows/statuses")

        # ---- nominal grid + completeness ---------------------------------
        ids = nominal_slot_ids(2, 2, 6)
        assert len(ids) == 24 and len(set(ids)) == 24
        assert ids[0] == "it0-p0-c0" and ids[-1] == "it5-p1-c1"
        print("OK  nominal grid       B*K*T = 2*2*6 = {} distinct slot ids"
              .format(len(ids)))

        try:
            assert_completeness(led, 2, 2, 6)
            raise AssertionError("an incomplete ledger passed completeness")
        except ProtocolError as e:
            assert "not complete" in str(e)
        print("OK  completeness       an incomplete ledger fails loudly")

        # a fully populated ledger passes
        p3 = Path(td) / "full.jsonl"
        led3 = SlotLedger(p3, **hdr)
        for sid in nominal_slot_ids(2, 2, 6):
            parts = sid.split("-")
            led3.open_slot(iteration=int(parts[0][2:]),
                           parent_slot=int(parts[1][1:]),
                           child_slot=int(parts[2][1:]),
                           parent_id="gen0_seed", parent_structural_depth=1,
                           gate_configured=3, gate_effective=False,
                           gate_reason="consolidation_focus",
                           reduction_mode="official")
            led3.finalize(sid, terminal_status="evaluated_admitted",
                          archive_admitted=True)
        assert_completeness(led3, 2, 2, 6)
        assert len(led3.rows()) == 24
        print("OK  full ledger        24/24 nominal slots finalized -> passes")

        # ---- predictive arm must carry a gamma hash ----------------------
        try:
            SlotLedger(Path(td) / "x.jsonl", run_id="r", arm="predictive",
                       backbone="b", cohort_id="c", search_seed=0,
                       root_bundle_sha256="R" * 64)
            raise AssertionError("predictive ledger without a gamma hash passed")
        except ProtocolError as e:
            assert "Gamma checkpoint hash" in str(e)
        print("OK  arm provenance     predictive requires its Gamma hash")

    print()
    print("VERDICT: ALL OK")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(self_test())
