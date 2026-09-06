#!/usr/bin/env python3
"""Structural-ablation gate (Part III, pre-launch acceptance).

Runs ONE complete macro-group (``--group-batches`` exact-replay batches) for
each structural arm on a single seed from the same stage-1 checkpoint, plus
data-stream hash gates, and asserts the acceptance criteria:

Data-stream gates (identical across arms):
  * the three arms share the identical Y1+Y2-valid cut set (cut-set hash);
  * Y1 multiset hash identical across arms;
  * aligned and mispaired share the Y2 multiset (Counter equality + hash);
  * mispaired: unchanged_fraction == 0, every assigned Y2 strictly later
    than its receiving cut.

Training-group gate (per arm, per compressible tau):
  * M_unique_trees >= min_trees (actual counts recorded — never silently
    dropped);
  * below_threshold_groups == 0;
  * replay planned == matched (exact alignment across the two passes);
  * representation (auxiliary) gradient finite and nonzero.

``supervision_mode`` changes ONLY the cut builder's row construction; the
adapter trace (which roots are selected and which (tau, node, time) cuts
exist) is mode-independent.  So the data-stream gates walk the model once
and feed the identical trace to all three builders.

This script does NOT launch the full training.  Output JSON:
outputs/<outroot>/gate/ablation_gate.json

Usage:
    python -m scripts.ablation_gate --data wikipedia \\
        --data-dir old/processed_tgn_data --seed 0 --gpu 0 \\
        --pretrained-checkpoint outputs/t2_pretrain/best.pt \\
        --group-batches 56 --min-trees 896 [--hash-batches 120]
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np
import torch

torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rpbe.data.jodie import JodieDataset
from rpbe.records import (JodieFutureIndex, JodieCutBuilder, NODE_CLASS)
from rpbe.training.jodie_loop import select_trace_rows

ARMS = ["1obs", "2obs_aligned", "2obs_mispaired"]


def parse_args():
    p = argparse.ArgumentParser("structural ablation gate")
    p.add_argument("--data", default="wikipedia")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--pretrained-checkpoint", required=True)
    p.add_argument("--group-batches", type=int, default=56)
    p.add_argument("--kf-min-abs", type=int, default=896)
    p.add_argument("--min-trees", type=int, default=896)
    p.add_argument("--trace-roots", type=int, default=32)
    p.add_argument("--hash-batches", type=int, default=120)
    p.add_argument("--outroot", default="struct_abl_v2")
    p.add_argument("--ridge-eps", type=float, default=1e-3)
    p.add_argument("--sketch-dim", type=int, default=64)
    p.add_argument("--lambda-kf", type=float, default=0.088)
    return p.parse_args()


def _h16(seq):
    h = hashlib.sha256()
    for item in seq:
        h.update(repr(item).encode())
    return h.hexdigest()[:16]


def _seed_all(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build(args, device, dataset, supervision_mode):
    import scripts.train_jodie as tj
    _seed_all(args.seed)
    ns = argparse.Namespace(
        data=args.data, data_dir=args.data_dir,
        pretrained_checkpoint=args.pretrained_checkpoint, output="/tmp/gate",
        seed=args.seed, gpu=args.gpu, bs=200, n_degree=5, n_head=2,
        n_epoch=1, n_layer=3, lr=3e-4, patience=10, drop_out=0.1,
        message_dim=100, memory_dim=172, finetune_host=False,
        selection_metric="auc", rpbe=True, kf_lambda=args.lambda_kf,
        rpbe_width=128, sketch_dim=args.sketch_dim, kf_cuts_per_tau=1024,
        kf_min_ratio=2.0, kf_min_abs=args.kf_min_abs,
        kf_group_batches=args.group_batches, kf_estimator="exact_replay",
        kf_variant="full_balancing", n_observations=2,
        supervision_mode=supervision_mode, repr_lr=1e-3,
        ridge_eps=args.ridge_eps, rpbe_seed=0, trace_roots=args.trace_roots,
        trace_mode="evenly_spaced", train_eval_auc=False,
        no_early_stop=False, max_batches=0, grad_clip=5.0,
        monitor_every=50, checkpoint_every=0, resume_from="",
        no_fail_on_monitor_error=False, max_train=0, max_val=0, max_test=0)
    return tj.build_components(ns, device, dataset)


def _collect_trace_cuts(adapter, tgn, train, n_batches, bs, n_degree, seed,
                        trace_roots):
    """Walk ``n_batches``; return the mode-independent raw CutCandidates."""
    if tgn.use_memory:
        tgn.memory.__init_memory__()
    cuts = []
    for k in range(n_batches):
        s = k * bs
        e = min(len(train.sources), (k + 1) * bs)
        if e - s <= 0:
            break
        tr = select_trace_rows(np.zeros(e - s), trace_roots, seed, k,
                               "evenly_spaced")
        adapter.set_trace_source_rows(tr)
        with torch.no_grad():
            tgn.compute_temporal_embeddings(
                train.sources[s:e], train.destinations[s:e],
                train.destinations[s:e], train.timestamps[s:e],
                train.edge_idxs[s:e], n_degree)
        if adapter.trace and adapter.trace.cuts:
            cuts.extend(adapter.trace.cuts)
        adapter.clear_trace()
    return cuts


def main():
    args = parse_args()
    device = torch.device(
        "cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    dataset = JodieDataset(args.data, data_dir=args.data_dir,
                           use_validation=True)
    full, train, _, _ = dataset.splits()
    fail = []
    gate_dir = Path("outputs") / args.outroot / "gate"
    gate_dir.mkdir(parents=True, exist_ok=True)
    report = {"seed": args.seed, "min_trees": args.min_trees,
              "group_batches": args.group_batches,
              "kf_min_abs": args.kf_min_abs, "arms": {}, "gates": {},
              "train_gate": {}}

    # ------------------------------------------------------------------ hash
    print("=== data-stream hash gates ===", flush=True)
    components = _build(args, device, dataset, "1obs")   # mode-independent trace
    adapter = components["adapter"]
    tgn = components["tgn"]
    raw = _collect_trace_cuts(adapter, tgn, train, args.hash_batches, 200, 5,
                              args.seed, args.trace_roots)
    print("raw trace cuts collected:", len(raw), flush=True)
    idx = JodieFutureIndex(train)
    builders = {arm: JodieCutBuilder(idx, stage=NODE_CLASS, seed=args.seed,
                                     n_observations=2,
                                     supervision_mode=arm)
                for arm in ARMS}
    # feed the same trace object to each builder
    from rpbe.state import CompactCutTrace
    trace_obj = CompactCutTrace(
        root_rows=sorted({int(c.root_row) for c in raw}), cuts=raw)
    per_arm_rows = {}
    for arm in ARMS:
        rows = builders[arm].build(trace_obj, batch_seed=0)
        cut_ids = sorted(r.cut_id for r in rows)
        y1 = sorted(r.outcome_id for r in rows if r.horizon == 1)
        y2 = sorted(r.outcome_id for r in rows if r.horizon == 2)
        n_y2 = len(y2)
        n_y1 = len(y1)
        per_arm_rows[arm] = rows
        report["arms"][arm] = {
            "cut_set_hash": _h16(cut_ids),
            "y1_hash": _h16(y1),
            "y2_hash": _h16(y2),
            "n_cuts": len(set(cut_ids)),
            "n_y1": n_y1,
            "n_y2": n_y2,
        }
        print("  {} cuts={} y1={} y2={}".format(
            arm, len(set(cut_ids)), n_y1, n_y2), flush=True)

    cs = {report["arms"][a]["cut_set_hash"] for a in ARMS}
    y1h = {report["arms"][a]["y1_hash"] for a in ARMS}
    if len(cs) != 1:
        fail.append("cut-set hashes differ across arms")
    if len(y1h) != 1:
        fail.append("Y1 hashes differ across arms")
    if report["arms"]["2obs_aligned"]["y2_hash"] != \
            report["arms"]["2obs_mispaired"]["y2_hash"]:
        fail.append("aligned vs mispaired Y2 multiset hash differ")
    if report["arms"]["2obs_aligned"]["n_y1"] != \
            report["arms"]["2obs_aligned"]["n_cuts"]:
        fail.append("aligned: n_y1 != n_cuts (cuts missing Y1?)")
    # per-cut pairing + unchanged + legality
    align2 = {r.cut_id: r for r in per_arm_rows["2obs_aligned"]
              if r.horizon == 2}
    misp2 = {r.cut_id: r for r in per_arm_rows["2obs_mispaired"]
             if r.horizon == 2}
    time_of = {r.cut_id: r.time for r in per_arm_rows["2obs_aligned"]}
    unchanged = sum(1 for cid, r in align2.items()
                    if misp2.get(cid) is not None
                    and r.outcome_id == misp2[cid].outcome_id)
    illegal = sum(1 for cid, r in misp2.items()
                  if not (r.outcome_time is not None
                          and r.outcome_time > time_of[cid]))
    n_pair = len(align2)
    report["gates"]["mispaired"] = {
        "n_compared": n_pair,
        "unchanged": unchanged,
        "unchanged_fraction": unchanged / n_pair if n_pair else None,
        "illegal_assignments": illegal,
    }
    if unchanged != 0:
        fail.append("mispaired unchanged_fraction != 0 ({} unchanged)"
                    .format(unchanged))
    if illegal != 0:
        fail.append("mispaired {} illegal (non-future) Y2".format(illegal))
    del components, adapter, tgn, raw

    # ------------------------------------------------------------------ train
    print("=== training-group gate (1 macro-group per arm) ===", flush=True)
    from rpbe.monitoring import MonitorWriter
    from rpbe.training.jodie_loop import JodieNodeClassificationLoop
    for arm in ARMS:
        components = _build(args, device, dataset, arm)
        tgn = components["tgn"]
        decoder = components["decoder"]
        repr_opt = components["repr_optimizer"]
        head_opt = components["head_optimizer"]
        adapter = components["adapter"]
        cut_builder = components["cut_builder"]
        fixed_maps = components["fixed_maps"]
        rpbe_cfg = components["rpbe_cfg"]
        monitor = MonitorWriter(gate_dir / arm, fail_on_error=False)
        loop = JodieNodeClassificationLoop(
            tgn=tgn, decoder=decoder, repr_optimizer=repr_opt,
            head_optimizer=head_opt, device=device, batch_size=200,
            n_neighbors=5, grad_clip=5.0, monitor=monitor,
            seed=args.seed, finetune_host=True,
            adapter=adapter, cut_builder=cut_builder,
            fixed_maps=fixed_maps, rpbe_cfg=rpbe_cfg,
            trace_roots=args.trace_roots, trace_mode="evenly_spaced",
            kf_estimator="exact_replay")
        captured = {}
        orig_close = loop.kf_window.close_replay

        def wrap_close(*a, **kw):
            closed, plan, diag = orig_close(*a, **kw)
            captured["diag"] = diag
            return closed, plan, diag
        loop.kf_window.close_replay = wrap_close
        row = loop.train_epoch(0, 0, train, max_batches=args.group_batches)
        diag = captured.get("diag") or {}
        per_tau = {}
        for tau, d in diag.items():
            m_trees = d.get("M_unique_trees")
            per_tau[tau] = {
                "M_unique_trees": m_trees,
                "below_threshold": bool(d.get("below_threshold", False)),
                "failed": d.get("failed"),
            }
            if (m_trees or 0) < args.min_trees:
                fail.append("{} {} M_unique_trees={} < {}".format(
                    arm, tau, m_trees, args.min_trees))
            if d.get("below_threshold"):
                fail.append("{} {} below_threshold (window dropped)".format(
                    arm, tau))
        kf = row.get("kf") or {}
        g_norms = [float(p.grad.norm()) for p in loop.repr_params
                   if p.grad is not None]
        total_g = float(sum(g_norms))
        report["train_gate"][arm] = {
            "per_tau": per_tau,
            "below_threshold_groups": kf.get("below_threshold_groups"),
            "aux_batches": kf.get("aux_batches"),
            "replay_align_planned": kf.get("replay_align_planned"),
            "replay_align_matched": kf.get("replay_align_matched"),
            "replay_align_missing": kf.get("replay_align_missing"),
            "repr_grad_norm": total_g,
            "J_norm": kf.get("J_norm"),
        }
        print("  {} train_gate={}".format(arm, json.dumps(
            {k: v for k, v in report["train_gate"][arm].items()
             if k != "J_norm"}, allow_nan=True)), flush=True)
        if kf.get("below_threshold_groups"):
            fail.append("{} below_threshold_groups != 0".format(arm))
        if kf.get("replay_align_planned") != kf.get("replay_align_matched"):
            fail.append("{} replay plan/matched mismatch".format(arm))
        if not (np.isfinite(total_g) and total_g > 0.0):
            fail.append("{} representation grad not finite/nonzero ({})"
                        .format(arm, total_g))
        del components, loop, monitor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    report["gates"]["pass"] = not fail
    report["gates"]["failures"] = fail
    out_json = gate_dir / "ablation_gate.json"
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2, allow_nan=True)
    print(json.dumps(report, indent=2, allow_nan=True), flush=True)
    print("wrote {}".format(out_json), flush=True)
    print("GATE " + ("PASS" if not fail else "FAIL"), flush=True)
    if fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
