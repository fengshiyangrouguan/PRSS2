#!/usr/bin/env python3
"""Wiki-LR-Binary real-data gates (spec §3/§4.2/§8).

Runs on real tgbl-wiki train/val AFTER the compact manifest exists:

1. speed gate  : time a full-split val AP evaluation (n_neighbors=5, one neg
                 per positive); must beat the per-run cap on this GPU.
2. yield gate  : for macro-group sizes 8/10/12/16, run one exact-replay group
                 and record M_unique_trees per canonical tau.  Choose the
                 FIRST size where every canonical tau >= kf_min_trees AND the
                 forward fits in VRAM.  trace_roots=160 by default (fall to
                 128 only if 160 OOMs).
3. VRAM gate   : peak allocated memory must stay under the cap.

Output: a group-plan json shared by ALL arms/seeds (spec §4.2).

Usage:
    python -m scripts.gate_wiki_lr_binary --data-dir datasets \
        --negatives datasets/wiki_lr_binary_negatives.json --gpu 0 \
        --out datasets/wiki_lr_binary_group_plan.json
"""

import argparse
import json
import os
import sys
import time
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

from scripts.train_tgb_link import build_model, seed_all  # noqa: E402
from rpbe.data.wiki_binary_negatives import CompactNegatives  # noqa: E402
from rpbe.training.histrand_sampler import HistRandTrainSampler  # noqa: E402
from rpbe.training.tgb_link_loop import TGBPairLinkLoop  # noqa: E402


def make_args(**over):
    import argparse as _ap
    p = _ap.ArgumentParser("gate defaults")
    p.add_argument("--data-dir", default="datasets")
    p.add_argument("--arm", default="2obs_aligned")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_neighbors", type=int, default=5)
    p.add_argument("--n_layers", type=int, default=3)
    p.add_argument("--trace_pairs_per_parent", type=int, default=2)
    p.add_argument("--kf_group_batches", type=int, default=8)
    p.add_argument("--kf_min_trees", type=int, default=896)
    p.add_argument("--lambda_kf", type=float, default=0.088)
    p.add_argument("--ridge_eps", type=float, default=1e-3)
    p.add_argument("--sketch_dim", type=int, default=64)
    p.add_argument("--width_D", type=int, default=128)
    p.add_argument("--rpbe_seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-4)
    args = p.parse_args([])
    for k, v in over.items():
        setattr(args, k, v)
    return args


def main():
    p = argparse.ArgumentParser("wiki-LR-Binary real-data gate")
    p.add_argument("--data-dir", default="datasets")
    p.add_argument("--negatives", required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--out", default="datasets/wiki_lr_binary_group_plan.json")
    p.add_argument("--trace-roots", type=int, default=160)
    p.add_argument("--kf-min-trees", type=int, default=896)
    p.add_argument("--group-sizes", default="8,10,12,16")
    p.add_argument("--vram-cap-gb", type=float, default=30.0)
    p.add_argument("--time-cap-min", type=float, default=10.0)
    p.add_argument("--val-bs", type=int, default=200)
    args = p.parse_args()

    seed_all(0)
    device = torch.device("cuda:{}".format(args.gpu)
                          if torch.cuda.is_available() else "cpu")
    bargs = make_args(data_dir=args.data_dir, n_neighbors=5, n_layers=3,
                      kf_min_trees=args.kf_min_trees,
                      trace_pairs_per_parent=2)
    c = build_model(bargs, device)
    ds = c["ds"]
    train = ds.train
    negs = CompactNegatives(args.negatives)
    out = {"data": ds.name, "trace_roots": int(args.trace_roots),
           "kf_min_trees": int(args.kf_min_trees),
           "n_neighbors": 5, "n_layers": 3, "candidates": []}

    # ---------------- 1. full-split val speed gate ------------------
    from rpbe.training.wiki_binary_eval import evaluate_val
    if tgn_use_mem(c):
        c["tgn"].memory.__init_memory__()
    t0 = time.time()
    with torch.no_grad():
        vm = evaluate_val(c["tgn"], ds, negs, n_neighbors=5, bs=args.val_bs)
    speed_s = time.time() - t0
    print("full-val AP gate: {:.1f}s  ap_all={:.4f}".format(
        speed_s, vm["ap_all"]), flush=True)
    out["full_val_seconds"] = speed_s
    assert speed_s / 60.0 <= args.time_cap_min, \
        "full val {:.1f}s exceeds cap {:.1f}min; profile before 5-seed".format(
            speed_s, args.time_cap_min)

    # ---------------- 2./3. group-yield + VRAM gate -----------------
    sampler = HistRandTrainSampler(
        train.sources, train.destinations, train.timestamps, model_seed=0)
    chosen = None
    roots_tried = [int(args.trace_roots)]
    if int(args.trace_roots) > 160:
        roots_tried = [int(args.trace_roots)]
    for gs in [int(x) for x in args.group_sizes.split(",")]:
        for roots in list(roots_tried):
            torch.cuda.reset_peak_memory_stats()
            try:
                loop = TGBPairLinkLoop(
                    tgn=c["tgn"], device=device, batch_size=200,
                    n_neighbors=5, grad_clip=5.0, monitor=None, seed=0,
                    adapter=c["adapter"],
                    link_future_index=c["link_future_index"],
                    boundary_maps=c["boundary_maps"],
                    edge_table=c["edge_table"], arm=bargs.arm,
                    rpbe_cfg=c["rpbe_cfg"],
                    repr_optimizer=c["repr_optimizer"],
                    head_optimizer=c["head_optimizer"],
                    trace_roots=roots, trace_pairs_per_parent=2,
                    kf_group_batches=gs, kf_min_trees=args.kf_min_trees,
                    fail_below=False, train_neg_sampler=sampler)
                row = loop.train_epoch(0, 0, train, max_batches=gs)
                diag = row.get("window_diag", [])
                peak_gb = torch.cuda.max_memory_allocated() / 1e9
                yields = {d["tau"]: int(d["M_unique_trees"]) for d in diag}
                rec = {"group_batches": gs, "trace_roots": roots,
                       "yields": yields, "peak_gb": round(peak_gb, 2)}
                out["candidates"].append(rec)
                print(json.dumps(rec), flush=True)
                ok = yields and min(yields.values()) >= args.kf_min_trees \
                    and peak_gb <= args.vram_cap_gb
                if ok and chosen is None:
                    chosen = {"group_batches": gs, "trace_roots": roots}
                    print("CHOSEN group plan:", json.dumps(chosen), flush=True)
            except torch.cuda.OutOfMemoryError:
                print("OOM gs={} roots={}".format(gs, roots), flush=True)
                torch.cuda.empty_cache()
            if chosen is not None:
                break
        if chosen is not None:
            break
    if chosen is None:
        raise SystemExit("no group size met yield/VRAM gates; see candidates")
    out["chosen"] = chosen
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote", args.out)


def tgn_use_mem(c):
    return bool(c["tgn"].use_memory)


if __name__ == "__main__":
    main()
