"""Orchestrator-side SRI hooks: slot open/close, material capture, sweeping.

DESIGN CONSTRAINT: a non-formal run must stay byte-identical. Every hook here
defaults to a NO-OP (`SRIHooks.disabled()`), and the orchestrator holds one as a
plain attribute rather than a config-dataclass field, so nothing about the
existing search path changes when formal mode is off.

The orchestrator's breeding loop is long and already validated, so this module
avoids forcing a try/finally re-indent of the whole body. Instead:

  * `open(...)` is called once, after the resume-skip check, BEFORE dispatch;
  * each existing exit point calls `close(handle, "<terminal_status>")`;
  * `sweep_iteration()` finalizes anything still open at the end of an
    iteration as `generation_error` -- so "exactly one terminal row per nominal
    slot" holds even if a branch forgets to close (a visible failure, never a
    silent gap);
  * `finalize_dangling_on_resume()` closes rows left open by a crashed process,
    so a resume cannot leave a nominal slot permanently open.

MATERIAL CAPTURE (§5). A rejected or failed child keeps enough material OUTSIDE
the search archive to be re-executed by the canonical audit. `capture_material`
writes `<run>/rejected/<slot_id>/traces/<task>.py`; the ledger row stores that
path and never the archive. Because the material lives outside the archive, a
rejected candidate can never be re-admitted by persistence, nor become a parent
or a Final-Score candidate.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

from meta_n.sri.ledger import SlotLedger, slot_id_for
from meta_n.sri.protocol import ProtocolError

REJECTED_DIRNAME = "rejected"


class SlotHandle:
    """One open slot. `close` is idempotent-guarded so a double close is loud."""

    __slots__ = ("slot_id", "iteration", "parent_slot", "child_slot",
                 "t0", "closed", "extra")

    def __init__(self, slot_id: str, iteration: int, parent_slot: int,
                 child_slot: int) -> None:
        self.slot_id = slot_id
        self.iteration = int(iteration)
        self.parent_slot = int(parent_slot)
        self.child_slot = int(child_slot)
        self.t0 = time.time()
        self.closed = False
        self.extra: Dict[str, Any] = {}


class SRIHooks:
    """Ledger + material capture for one formal arm. Disabled by default."""

    def __init__(self, *, ledger: Optional[SlotLedger] = None,
                 run_dir: Optional[Path] = None,
                 gate_configured: int = 0,
                 gate_reason: str = "",
                 reduction_mode: str = "official",
                 cohort: Sequence[str] = (),
                 enabled: bool = True) -> None:
        self.ledger = ledger
        self.run_dir = Path(run_dir) if run_dir is not None else None
        self.gate_configured = int(gate_configured)
        self.gate_reason = str(gate_reason)
        self.reduction_mode = str(reduction_mode)
        self.cohort = list(cohort)
        self.enabled = bool(enabled and ledger is not None)
        self._open: Dict[str, SlotHandle] = {}

    @classmethod
    def disabled(cls) -> "SRIHooks":
        return cls(ledger=None, enabled=False)

    # -- lifecycle ---------------------------------------------------------
    def open(self, *, iteration: int, parent_slot: int, child_slot: int,
             parent_id: Optional[str], parent_structural_depth: int,
             gate_effective: bool, proposed_child_id: Optional[str] = None,
             proposed_child_depth: Optional[int] = None,
             temperature: Optional[float] = None,
             focus_task: Optional[str] = None) -> Optional[SlotHandle]:
        if not self.enabled:
            return None
        sid = slot_id_for(iteration, parent_slot, child_slot)
        if sid in self._open:
            raise ProtocolError(
                "slot {!r} opened twice in one process -- a nominal slot is "
                "opened once".format(sid))
        self.ledger.open_slot(
            iteration=iteration, parent_slot=parent_slot, child_slot=child_slot,
            parent_id=parent_id,
            parent_structural_depth=int(parent_structural_depth),
            gate_configured=self.gate_configured,
            gate_effective=bool(gate_effective),
            gate_reason=self.gate_reason, reduction_mode=self.reduction_mode,
            proposed_child_id=proposed_child_id,
            proposed_child_depth=proposed_child_depth,
            temperature=temperature,
            focus_task=focus_task)
        h = SlotHandle(sid, iteration, parent_slot, child_slot)
        self._open[sid] = h
        return h

    def close(self, handle: Optional[SlotHandle], terminal_status: str,
              *, gate_status: Optional[str] = None,
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
        if not self.enabled or handle is None:
            return
        if handle.closed:
            raise ProtocolError(
                "slot {!r} closed twice in one process".format(handle.slot_id))
        self.ledger.finalize(
            handle.slot_id, terminal_status=terminal_status,
            gate_status=gate_status,
            # keep the configured reason unless the caller supplies an outcome
            gate_reason=(gate_reason if gate_reason is not None
                         else self.gate_reason or None),
            failure_class=failure_class, failure_message=failure_message,
            archive_admitted=archive_admitted,
            archive_rejection_reason=archive_rejection_reason,
            material_path=material_path,
            fresh_task_ids=fresh_task_ids, inherited_task_ids=inherited_task_ids,
            outer_calls=outer_calls, inner_calls=inner_calls,
            evaluator_calls=evaluator_calls, prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            inner_prompt_tokens=inner_prompt_tokens,
            inner_completion_tokens=inner_completion_tokens,
            omega_wall_seconds=omega_wall_seconds,
            gate_wall_seconds=gate_wall_seconds,
            eval_wall_seconds=eval_wall_seconds)
        handle.closed = True
        self._open.pop(handle.slot_id, None)

    def resume_note(self, *, iteration: int, parent_slot: int,
                    child_slot: int) -> None:
        """A resumed run re-entered an already-finalized slot (§5).

        Adds no second observation. If the slot was never finalized (the crash
        happened mid-slot), it is NOT finalized here either -- it is swept by
        `sweep_iteration`, so the original failure stays visible.
        """
        if not self.enabled:
            return
        sid = slot_id_for(iteration, parent_slot, child_slot)
        if self.ledger.is_finalized(sid):
            self.ledger.mark_resumed(sid)

    def sweep_iteration(self, iteration: int, *, status: str = "generation_error",
                        message: str = "slot left open at end of iteration") -> int:
        """Finalize every still-open slot of this iteration. Returns how many."""
        if not self.enabled:
            return 0
        n = 0
        for sid, h in list(self._open.items()):
            if h.iteration != int(iteration):
                continue
            self.close(h, status, failure_class="unclosed_slot",
                       failure_message=message)
            n += 1
        return n

    def finalize_dangling_on_resume(self, *, status: str = "generation_error",
                                    message: str = "process ended before the "
                                                   "slot was finalized") -> int:
        """Close rows left open by a crashed process, at resume time.

        Without this a crash could leave a nominal slot permanently open and an
        incomplete ledger would look like a smaller run instead of a failure.
        """
        if not self.enabled:
            return 0
        n = 0
        for sid in self.ledger.unfinalized():
            try:
                self.ledger.finalize(sid, terminal_status=status,
                                     failure_class="crashed_process",
                                     failure_message=message)
                n += 1
            except ProtocolError:
                pass
        return n

    # -- material capture --------------------------------------------------
    def capture_material(self, *, slot_id: str, candidate_id: str,
                         structural_depth: int,
                         task_scripts: Mapping[str, str]) -> Optional[str]:
        """Write rejected/failed child material OUTSIDE the archive.

        Returns the directory path recorded on the ledger row, or None when the
        hooks are disabled or the run has no directory. A missing script is
        skipped: partial material is still useful, and `is_executable_on` /
        `build_edges` will report the edge invalid rather than guessing.
        """
        if not self.enabled or self.run_dir is None:
            return None
        d = self.run_dir / REJECTED_DIRNAME / slot_id
        (d / "traces").mkdir(parents=True, exist_ok=True)
        for task_id, src in task_scripts.items():
            if not str(src).strip():
                continue
            safe = str(task_id).replace("/", "_")
            (d / "traces" / (safe + ".py")).write_text(str(src),
                                                       encoding="utf-8")
        meta = {"candidate_id": candidate_id,
                "structural_depth": int(structural_depth),
                "slot_id": slot_id,
                "note": "out-of-archive material for the canonical transition "
                        "audit; never eligible for parent selection or Final "
                        "Score (§5)"}
        (d / "material.json").write_text(json.dumps(meta, indent=2),
                                         encoding="utf-8")
        return str(d)

    # -- provenance --------------------------------------------------------
    def root_bundle(self, extra: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        """The hashable root bundle (§4) -- what both arms must agree on."""
        from meta_n.sri.protocol import ROOT_BUNDLE_FIELDS
        base = {k: None for k in ROOT_BUNDLE_FIELDS}
        base.update({"cohort": list(self.cohort)})
        if self.ledger is not None:
            base["software_revision"] = "proposal_slots.jsonl"
            base["root_candidate_id"] = "gen0_seed"
        base.update(dict(extra or {}))
        return base


# --------------------------------------------------------------------------
# acceptance self-test (§14 items 1, 2, 3 exercised through the hooks)
# --------------------------------------------------------------------------

def self_test() -> int:
    import tempfile
    print("hooks.py acceptance")
    print("=" * 68)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # -- disabled hooks are a strict no-op -----------------------------
        off = SRIHooks.disabled()
        h = off.open(iteration=0, parent_slot=0, child_slot=0, parent_id="p",
                     parent_structural_depth=1, gate_effective=False)
        assert h is None
        off.close(None, "evaluated_admitted")
        assert off.sweep_iteration(0) == 0
        assert off.capture_material(slot_id="s", candidate_id="c",
                                    structural_depth=2,
                                    task_scripts={"T": "x"}) is None
        assert not any(td.iterdir())
        print("OK  disabled no-op     every hook returns without touching disk")

        led = SlotLedger(td / "proposal_slots.jsonl", run_id="r", arm="official",
                         backbone="gpt-5.5", cohort_id="sri_primary6:6",
                         search_seed=0, root_bundle_sha256="R" * 64)
        hk = SRIHooks(ledger=led, run_dir=td, gate_configured=3,
                      gate_reason="consolidation_focus",
                      reduction_mode="official", cohort=["T1", "T2"])

        h1 = hk.open(iteration=0, parent_slot=0, child_slot=0, parent_id="p2",
                     parent_structural_depth=2, gate_effective=False)
        hk.close(h1, "gate_rejected", gate_status="rejected",
                 archive_admitted=False, archive_rejection_reason="gate")
        row = led.rows()[0]
        assert row["terminal_status"] == "gate_rejected"
        assert row["gate_configured"] == 3 and row["gate_effective"] is False
        assert row["gate_reason"] == "consolidation_focus"
        print("OK  gate provenance   configured=3 but effective=false, "
              "reason=consolidation_focus recorded per slot")

        # -- sweep closes a forgotten slot, visibly -------------------------
        h2 = hk.open(iteration=0, parent_slot=0, child_slot=1, parent_id="p2",
                     parent_structural_depth=2, gate_effective=False)
        assert not led.is_finalized(h2.slot_id)
        assert hk.sweep_iteration(0) == 1
        r2 = [r for r in led.rows() if r["slot_id"] == h2.slot_id][0]
        assert r2["terminal_status"] == "generation_error"
        assert r2["failure_class"] == "unclosed_slot"
        print("OK  sweep              an unclosed slot becomes a visible "
              "generation_error, not a silent gap")

        # -- double close is loud ------------------------------------------
        try:
            hk.close(h1, "timeout")
            raise AssertionError("a double close was accepted")
        except ProtocolError as e:
            assert "closed twice" in str(e)
        print("OK  double close       refused")

        # -- material capture goes OUTSIDE the archive ---------------------
        mp = hk.capture_material(slot_id="it0-p0-c2", candidate_id="gen1_b0_k2",
                                 structural_depth=3,
                                 task_scripts={"T1": "def solve(): pass",
                                               "T2": ""})
        assert mp is not None and "rejected" in mp
        assert (Path(mp) / "traces" / "T1.py").is_file()
        assert not (Path(mp) / "traces" / "T2.py").is_file()
        assert "archive" not in mp
        print("OK  material outside   captured under rejected/ (never inside "
              "archive/), empty scripts skipped")

        h3 = hk.open(iteration=0, parent_slot=0, child_slot=2, parent_id="p2",
                     parent_structural_depth=2, gate_effective=False)
        hk.close(h3, "invalid_program", archive_admitted=False,
                 material_path=mp)
        r3 = [r for r in led.rows() if r["slot_id"] == h3.slot_id][0]
        assert r3["material_path"] == mp
        print("OK  ledger keeps path  the row carries material_path for the "
              "canonical audit")

        # -- resume: finalized slot gets a note, open slot stays visible ----
        led2 = SlotLedger(td / "proposal_slots.jsonl", run_id="r", arm="official",
                          backbone="gpt-5.5", cohort_id="sri_primary6:6",
                          search_seed=0, root_bundle_sha256="R" * 64)
        hk2 = SRIHooks(ledger=led2, run_dir=td, gate_configured=3,
                       gate_reason="consolidation_focus",
                       reduction_mode="official", cohort=["T1", "T2"])
        hk2.resume_note(iteration=0, parent_slot=0, child_slot=0)
        assert led2.rows()[0]["terminal_status"] == "gate_rejected"
        assert len(led2.rows()) == 3
        print("OK  resume note        re-entering a finalized slot adds no "
              "second observation")

        # a dangling OPEN row (crashed process) is closed on resume
        led3 = SlotLedger(td / "dangling.jsonl", run_id="r", arm="official",
                          backbone="gpt-5.5", cohort_id="sri_primary6:6",
                          search_seed=0, root_bundle_sha256="R" * 64)
        led3.open_slot(iteration=0, parent_slot=0, child_slot=0,
                       parent_id="p", parent_structural_depth=1,
                       gate_configured=3, gate_effective=False,
                       gate_reason="consolidation_focus",
                       reduction_mode="official")
        assert led3.unfinalized()
        hk3 = SRIHooks(ledger=led3, run_dir=td, cohort=["T1"])
        assert hk3.finalize_dangling_on_resume() == 1
        assert not led3.unfinalized()
        assert led3.rows()[0]["failure_class"] == "crashed_process"
        print("OK  dangling on resume a crashed slot is closed with "
              "failure_class=crashed_process")

        # root bundle is hashable and carries the cohort
        rb = hk.root_bundle({"dev_scores": {"T1": 0.5},
                             "task_order": ["T1", "T2"]})
        from meta_n.sri.protocol import root_bundle_sha256
        assert len(root_bundle_sha256(rb)) == 64
        assert rb["cohort"] == ["T1", "T2"]
        print("OK  root bundle       hashes and carries the cohort order")

    print()
    print("VERDICT: ALL OK")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(self_test())
