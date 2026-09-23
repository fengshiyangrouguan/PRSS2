"""Real offline Phase B on the REAL Official records (0 API).

Chain (user-frozen 2026-09-18):
    phase_b_runnable=true -> Phase B -> committed_steps >= 1 ?
        no  -> do NOT spend API on Phase C; report the blocker
        yes -> save the trained Gamma -> paired Phase C (Official vs Predictive)

This builds the task/protect windows from the materialised CutRecords with the
v4.2.1 split (tree_hash_v421), runs Phase B, and saves the frozen Gamma. It
prints everything needed to decide whether Phase C is worth spending on.

Usage:
    python scripts/run_phase_b.py [--steps 300] [--out runs/_gamma_phaseb.pt]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, ".")
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, os.environ.get("META_N_TORCH_THREADS", "1"))
torch.set_num_threads(int(os.environ.get("META_N_TORCH_THREADS", "1")))

from meta_n.rpbe import config as C                      # noqa: E402
from meta_n.rpbe.census import (SPLIT_PROTOCOL,          # noqa: E402
                                partition_tree_roles_v421)
from meta_n.rpbe.fusion import SlottedFusion             # noqa: E402
from meta_n.rpbe.records import CutMeta, CutRecord       # noqa: E402
from meta_n.rpbe.trainer import PhaseB                   # noqa: E402
from meta_n.rpbe.window import StatWindow                # noqa: E402

# NO DEFAULT RECORDS PATH. `runs/_official_records_latest.jsonl` was the
# PRIMARY6-era default, and a default is how the WRONG cohort's records get
# trained on by accident: the path looks plausible and the run proceeds. The
# records a Gamma is trained on must be named explicitly.


def load_records(path):
    out = []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        m = r["meta"]
        out.append(CutRecord(
            meta=CutMeta(candidate_id=m["candidate_id"], depth=m["depth"],
                         run_id=m["run_id"],
                         root_candidate_id=m["root_candidate_id"]),
            cut_id=tuple(r["cut_id"]), tree_id=tuple(r["tree_id"]),
            occurrence_seq=r["occurrence_seq"],
            X_v=torch.tensor(r["X_v"]), q_emb=torch.tensor(r["q_emb"]),
            mask=torch.tensor(r["mask_valid"]), p=torch.tensor(r["p"]),
            r=r["r"], weight=r["weight"]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=C.TRAIN_STEPS)
    ap.add_argument("--out", default="runs/_gamma_phaseb.pt")
    ap.add_argument("--phase-a-root", required=True,
                    help="the Structural6 Phase-A output root. --records must "
                         "live under it: the check is containment, not a name "
                         "pattern.")
    ap.add_argument("--records", required=True,
                    help="Structural6 Phase-A records JSONL. REQUIRED: the "
                         "Gamma must be trained on records built from THIS "
                         "cohort's archives.")
    ap.add_argument("--init-from", default=None,
                    help="CONTINUE training an existing Gamma checkpoint (warm start) instead of random init. Used to carry the Phase-B structure learned on one backbone onto another backbone's records.")
    args = ap.parse_args()

    # REFUSE THE OLD COHORT'S RECORDS BY CONTAINMENT, NOT BY NAME.
    #
    # Phase-A is cohort-coupled: a candidate's score is the mean over EVERY
    # cohort task and a record's label `r` is a difference of those means, so
    # PRIMARY6 records do not describe Structural6 -- training on them would fit
    # Gamma to the wrong task set while every diagnostic still looked healthy.
    #
    # A NAME BLOCKLIST DOES NOT WORK, and this was measured rather than assumed:
    # an earlier version refused a path containing `mtgem_run`, then accepted
    # `/root/autodl-tmp/_gem6_traceonly.jsonl` -- which IS the PRIMARY6 records
    # set, under a name that matches no pattern. The check is therefore POSITIVE:
    # the records must live under the Phase-A output root this cohort was built
    # in, which no amount of renaming can fake.
    _rr = Path(args.records).resolve()
    _pr = Path(args.phase_a_root).resolve()
    if _pr not in _rr.parents:
        raise SystemExit(
            "--records ({}) is not under --phase-a-root ({}). The Gamma must be "
            "trained on records built from THIS cohort's Phase-A output; "
            "cohort-coupled PRIMARY6 records (e.g. _gem5_records.jsonl, "
            "_gem6_traceonly.jsonl) do not describe Structural6."
            .format(_rr, _pr))

    ledger = os.environ.get("META_N_REQUEST_LEDGER", "").strip()
    paid0 = 0
    if ledger and os.path.isfile(ledger):
        from meta_n.rpbe.accounting import snapshot
        paid0 = snapshot()["backend_requests"]

    recs = load_records(args.records)
    trees = sorted({r.tree_id for r in recs})
    task_roots, protect_roots = partition_tree_roles_v421(trees)
    tw, pw = StatWindow(name="task"), StatWindow(name="protect")
    tw.add([r for r in recs if r.tree_id in set(task_roots)])
    pw.add([r for r in recs if r.tree_id in set(protect_roots)])

    print("=" * 78)
    print("PHASE B on REAL Official records")
    print("=" * 78)
    print("  split_protocol : %s" % SPLIT_PROTOCOL)
    print("  records        : %d over %d trees" % (len(recs), len(trees)))
    print("  task window    : %d rows / %d trees" % (tw.n_rows, tw.n_trees))
    print("  protect window : %d rows / %d trees" % (pw.n_rows, pw.n_trees))
    print("  steps          : %d" % args.steps)
    print()

    fusion = SlottedFusion()
    n_params = sum(p.numel() for p in fusion.parameters())

    # --init-from: warm start. w0 is captured AFTER the load on
    # purpose: `max|dtheta|` then reports what THIS training moved,
    # not what the original training did. If it stays tiny the
    # checkpoint is just the old Gamma relabelled -- which is exactly
    # the failure mode of a backbone mismatch, so it must be visible.
    if args.init_from:
        _ck = torch.load(args.init_from, map_location="cpu",
                         weights_only=False)
        _state = (_ck.get("state_dict")
                  if isinstance(_ck, dict) and "state_dict" in _ck
                  else _ck)
        _missing, _unexpected = fusion.load_state_dict(_state,
                                                       strict=False)
        print("  init_from      : %s" % args.init_from)
        print("    loaded %d entries  missing=%s  unexpected=%s"
              % (len(_state), list(_missing)[:4], list(_unexpected)[:4]))
        if _missing:
            raise SystemExit(
                "init_from left %d parameter(s) unset: %s"
                % (len(_missing), list(_missing)[:6]))

    w0 = [p.detach().clone() for p in fusion.parameters()]

    tb = PhaseB(fusion, tw, pw, steps=args.steps)
    hist = tb.run()

    moved = max(float((a - b).abs().max()) for a, b in
                zip(fusion.parameters(), w0))
    last = hist["history"][-1] if hist["history"] else {}

    print("  status          : %s" % hist.get("status"))
    print("  stop_reason     : %s" % hist.get("stop_reason"))
    print("  committed_steps : %d / %d" % (hist["steps"], hist["planned_steps"]))
    print("  ended_early     : %s" % hist.get("ended_early"))
    print("  blocked_at      : %s" % hist.get("blocked_at"))
    print("  gamma moved     : max|dtheta| = %.6e" % moved)
    print("  params          : %d" % n_params)
    if last:
        print("  last step       : J_task=%s J_LPSE=%s D_task=%s D_protect=%s"
              % (last.get("J_task"), last.get("J_LPSE"), last.get("D_task"),
                 last.get("D_protect")))
        print("  last oas alpha  : z_task=%s p_task=%s"
              % (last.get("oas_alpha_z_task"), last.get("oas_alpha_p_task")))
        js = [h["J_task"] for h in hist["history"]]
        print("  J_task trace    : %.6f -> %.6f (min %.6f max %.6f)"
              % (js[0], js[-1], min(js), max(js)))

    committed = int(hist["steps"]) >= 1
    print()
    if committed:
        meta = tb.save_gamma(args.out, extra={
            "records": args.records, "split_protocol": SPLIT_PROTOCOL,
            "task_trees": len(task_roots), "protect_trees": len(protect_roots),
            # provenance: a warm-started Gamma is NOT the same object
            # as a from-scratch one, and a reader must be able to tell.
            "init_from": args.init_from,
            "gamma_moved_max_abs": moved,
        })
        print("  GAMMA SAVED     : %s" % args.out)
        print("    checksum=%s params=%d steps=%d status=%s"
              % (meta.checksum, meta.param_count, meta.steps, meta.status))
        print("    -> committed_steps >= 1 : Phase C IS worth spending on.")
    else:
        print("  NO UPDATE COMMITTED -> the checkpoint would just be the")
        print("  INITIALISATION. Per the frozen chain, do NOT spend API on")
        print("  Phase C: its 'Predictive' score would not represent a trained")
        print("  Ours. Report this blocker instead.")

    paid1 = 0
    if ledger and os.path.isfile(ledger):
        from meta_n.rpbe.accounting import snapshot
        paid1 = snapshot()["backend_requests"]
    print()
    print("  API cost        : %d backend requests (must be 0)" % (paid1 - paid0))
    return 0 if (paid1 - paid0) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
