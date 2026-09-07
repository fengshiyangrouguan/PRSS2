#!/usr/bin/env python3
"""Measure how strong the mispaired intervention actually is (review verdict).

On a real window of Wikipedia cuts we quantify whether swapping the second
future observation under ``2obs_mispaired`` changes what the model sees:

  a) fraction of horizon-2 rows whose OUTCOME VALUE changes
     (outcome value, not outcome_id — Wikipedia positives are ~0.14%, so two
     distinct events usually carry the same 0 label);
  b) per-row and mean perturbation of the fixed joint test
     ``|| p_aligned - p_mispaired ||`` for the SAME cut, SAME Z;
  c) the window score difference ``J(aligned) - J(mispaired)`` under a FIXED
     Z stack (the only thing the arm changes).

If these are all near zero the intervention is vacuous: aligned ~= mispaired
downstream is then an artifact of a near-zero treatment, NOT evidence that
pairing/closure does not matter.

Usage:
    python -m scripts.mispaired_strength --data wikipedia \\
        --data-dir old/processed_tgn_data --gpu 0 \\
        [--trace-roots 32] [--hash-batches 60]
"""

import argparse
import math
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


def parse_args():
    p = argparse.ArgumentParser("mispaired intervention strength")
    p.add_argument("--data", default="wikipedia")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--trace-roots", type=int, default=32)
    p.add_argument("--hash-batches", type=int, default=60)
    p.add_argument("--dense", action="store_true",
                   help="use dense_future phi_Y (event-identity signature) "
                        "instead of the outcome-only signature")
    return p.parse_args()


def _seed_all(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    device = torch.device(
        "cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    dataset = JodieDataset(args.data, data_dir=args.data_dir,
                           use_validation=True)
    full, train, _, _ = dataset.splits()
    _seed_all(args.seed)

    import scripts.train_jodie as tj
    from rpbe.maps import FixedMaps
    from rpbe.loss import kf_score
    ns = argparse.Namespace(
        data=args.data, data_dir=args.data_dir,
        pretrained_checkpoint="outputs/t2_pretrain/best.pt", output="/tmp/x",
        seed=args.seed, gpu=args.gpu, bs=200, n_degree=5, n_head=2,
        n_epoch=1, n_layer=3, lr=3e-4, patience=10, drop_out=0.1,
        message_dim=100, memory_dim=172, finetune_host=False,
        selection_metric="auc", rpbe=True, kf_lambda=0.088, rpbe_width=128,
        sketch_dim=64, kf_cuts_per_tau=1024, kf_min_ratio=2.0,
        kf_min_abs=1024, kf_group_batches=56, kf_estimator="exact_replay",
        kf_variant="full_balancing", n_observations=2,
        supervision_mode="production", repr_lr=1e-3, ridge_eps=1e-3,
        rpbe_seed=0, trace_roots=args.trace_roots,
        trace_mode="evenly_spaced", train_eval_auc=False,
        no_early_stop=False, max_batches=0, grad_clip=5.0,
        monitor_every=50, checkpoint_every=0, resume_from="",
        no_fail_on_monitor_error=False, max_train=0, max_val=0, max_test=0,
        dense_future=args.dense)
    components = tj.build_components(ns, device, dataset)
    tgn = components["tgn"]
    adapter = components["adapter"]
    fixed_maps = components["fixed_maps"]

    # Collect a batch of traces (mode-independent) then build aligned and
    # mispaired rows for the same cuts.  The trace walk only needs the host
    # forward (compressor off does not matter for the trace shape).
    idx = JodieFutureIndex(train)
    b_align = JodieCutBuilder(idx, stage=NODE_CLASS, seed=args.seed,
                              n_observations=2, supervision_mode="2obs_aligned")
    b_misp = JodieCutBuilder(idx, stage=NODE_CLASS, seed=args.seed,
                             n_observations=2,
                             supervision_mode="2obs_mispaired")
    if tgn.use_memory:
        tgn.memory.__init_memory__()
    n_batch = min(args.hash_batches, math.ceil(len(train.sources) / 200))
    # rows per (cut,horizon) keyed by cut_id; aligned and mispaired built on
    # the SAME per-batch traces so cut_ids line up.
    from collections import defaultdict
    align_by_cut = {}
    misp_by_cut = {}
    for k in range(n_batch):
        s = k * 200
        e = min(len(train.sources), (k + 1) * 200)
        if e - s <= 0:
            break
        tr = select_trace_rows(np.zeros(e - s), args.trace_roots, args.seed,
                               k, "evenly_spaced")
        adapter.set_trace_source_rows(tr)
        with torch.no_grad():
            tgn.compute_temporal_embeddings(
                train.sources[s:e], train.destinations[s:e],
                train.destinations[s:e], train.timestamps[s:e],
                train.edge_idxs[s:e], 5)
        trace = adapter.trace
        if trace is not None and trace.cuts:
            for r in b_align.build(trace, batch_seed=k):
                align_by_cut[r.cut_id] = align_by_cut.get(r.cut_id, {})
                align_by_cut[r.cut_id][r.horizon] = r
            for r in b_misp.build(trace, batch_seed=k):
                misp_by_cut[r.cut_id] = misp_by_cut.get(r.cut_id, {})
                misp_by_cut[r.cut_id][r.horizon] = r
        adapter.clear_trace()

    shared = sorted(set(align_by_cut) & set(misp_by_cut))
    print("shared cuts:", len(shared), flush=True)

    def row_pair(row):
        """(context dict, outcome float, delta_t) -> fixed p via pv."""
        return fixed_maps.pv(row.context, row.outcome)

    # (a) outcome value change fraction on horizon-2 rows
    changed_val = 0
    n_h2 = 0
    # (b) p perturbation per horizon-2 row (same cut, same Z -> same context
    # except outcome/role/delta_t if the event changed)
    p_delta = []
    rows_align = []
    rows_misp = []
    for cid in shared:
        a2 = align_by_cut[cid].get(2)
        m2 = misp_by_cut[cid].get(2)
        a1 = align_by_cut[cid].get(1)
        if a2 is None or m2 is None or a1 is None:
            continue
        n_h2 += 1
        if float(a2.outcome) != float(m2.outcome):
            changed_val += 1
        pa = row_pair(a2)
        pm = row_pair(m2)
        p_delta.append(float((pa - pm).norm()))
        rows_align.append((cid, a1.z, a2))
        rows_misp.append((cid, a1.z, m2))
    frac_changed = changed_val / n_h2 if n_h2 else float("nan")
    p_delta = np.asarray(p_delta)
    print("(a) outcome-VALUE change on horizon-2 rows: {}/{} = {:.4f}"
          .format(changed_val, n_h2, frac_changed), flush=True)
    print("(b) ||p_align - p_misp||  mean={:.4e}  median={:.4e}  "
          "max={:.4e}".format(p_delta.mean(), np.median(p_delta),
                              p_delta.max()), flush=True)
    print("    |p| reference (aligned): "
          "{:.4e}".format(np.mean([float(row_pair(r[2]).norm())
                                   for r in rows_align])), flush=True)

    # (c) J(aligned) - J(mispaired) under FIXED Z per tau
    # Stack the aligned horizon-1+2 and horizon-2-only rows; Z is the same
    # between aligned/mispaired for a given cut, so any J change is purely
    # from the P swap.
    from collections import defaultdict
    by_tau = defaultdict(lambda: {"align": [], "misp": []})
    for cid in shared:
        a1 = align_by_cut[cid].get(1)
        a2 = align_by_cut[cid].get(2)
        m2 = misp_by_cut[cid].get(2)
        if a1 is None or a2 is None or m2 is None:
            continue
        tau = cid[2]
        # aligned emits Y1+Y2 rows; mispaired emits Y1+Y2 rows (Y2 swapped)
        by_tau[tau]["align"].append((a1, a2))
        by_tau[tau]["misp"].append((a1, m2))

    def j_for(rows):
        zs = [r[0].z for r in rows] + [r[1].z for r in rows]
        Z = torch.stack([r[0].z for r in rows] + [r[1].z for r in rows])
        P = torch.stack([row_pair(r[0]) for r in rows]
                        + [row_pair(r[1]) for r in rows])
        return kf_score(Z, P, eps=1e-3)
    print("(c) J under fixed Z (identical Z aligned vs mispaired):", flush=True)
    for tau, d in sorted(by_tau.items()):
        if len(d["align"]) < 64:
            print("   {} skipped ({} cuts < 64)".format(tau, len(d["align"])))
            continue
        ja = j_for(d["align"])
        jm = j_for(d["misp"])
        print("   {}  cuts={}  J_align={:.4f}  J_misp={:.4f}  "
              "delta={:+.4f}".format(tau, len(d["align"]),
                                     float(ja), float(jm),
                                     float(ja - jm)), flush=True)


if __name__ == "__main__":
    main()
