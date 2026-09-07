#!/usr/bin/env python3
"""TGB tgbl-wiki recursive-closure link training runner (one arm/seed).

Builds the official multi-layer recursive Twitter-TGN on tgbl-wiki, attaches
the pair exact-replay trainer for one of the four arms, and writes rolling
checkpoints + per-epoch metrics.  Selection and final test MRR use the TGB
official evaluator in a separate evaluation step (this entry trains only).

Usage:
    python -m scripts.train_tgb_link --arm 2obs_aligned --seed 0 \
        --data-dir datasets --output outputs/tgbl_wiki_recursive_abl/seed0/2obs_aligned \
        --gpu 0
"""

import argparse
import json
import math
import os
import random
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

from rpbe.data.tgb_link import TGBLinkDataset
from rpbe.hosts.official_tgn import TGN, MLP, get_neighbor_finder
from rpbe.hosts.jodie_tgn import JodieTGNAdapter, TAU_TEMPLATE
from rpbe.config import RPBConfig
from rpbe.compressor import RecursiveCompressor
from rpbe.pair_maps import BoundaryMaps
from rpbe.link_records import LinkFutureIndex
from rpbe.training.tgb_link_loop import TGBPairLinkLoop, ARMS


def parse_args():
    p = argparse.ArgumentParser("tgbl-wiki pair link train")
    p.add_argument("--data-dir", default="datasets")
    p.add_argument("--arm", choices=list(ARMS), default="2obs_aligned")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--output", required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--bs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--n-neighbors", type=int, default=10)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--trace-roots", type=int, default=32)
    p.add_argument("--trace-pairs-per-parent", type=int, default=2)
    p.add_argument("--kf-group-batches", type=int, default=56)
    p.add_argument("--kf-min-trees", type=int, default=896)
    p.add_argument("--lambda-kf", type=float, default=0.088)
    p.add_argument("--ridge-eps", type=float, default=1e-3)
    p.add_argument("--sketch-dim", type=int, default=64)
    p.add_argument("--width-D", type=int, default=128)
    p.add_argument("--rpbe-seed", type=int, default=0)
    p.add_argument("--no-fail-on-monitor-error", action="store_true")
    p.add_argument("--max-batches", type=int, default=0,
                   help="cap each train epoch at N batches (0=full; smoke)")
    return p.parse_args()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2, allow_nan=True)


def build_model(args, device):
    ds = TGBLinkDataset(root=args.data_dir)
    full, train, val, test = ds.splits()
    finder = get_neighbor_finder(
        train, uniform=False, max_node_idx=int(ds.n_internal_nodes - 1))
    # time stats from train
    ts = train.timestamps.astype(np.float64)
    ms, ss = float(ts.mean()), float(ts.std()) + 1e-8

    tgn = TGN(
        neighbor_finder=finder,
        node_features=ds.node_features.astype(np.float32),
        edge_features=ds.edge_features.astype(np.float32),
        device=device,
        n_layers=args.n_layers,
        n_heads=2,
        dropout=0.1,
        use_memory=True,
        message_dimension=100,
        memory_dimension=172,
        memory_update_at_start=True,
        embedding_module_type="graph_attention",
        message_function="identity",
        aggregator_type="last",
        n_neighbors=args.n_neighbors,
        mean_time_shift_src=ms, std_time_shift_src=ss,
        mean_time_shift_dst=ms, std_time_shift_dst=ss,
    ).to(device)

    host_dim = int(tgn.embedding_dimension)
    taus = [TAU_TEMPLATE.format(l) for l in range(args.n_layers + 1)]
    rpbe_cfg = RPBConfig(
        state_dims={tau: host_dim for tau in taus},
        own_dims={tau: host_dim for tau in taus},
        width_D=args.width_D, m=args.sketch_dim,
        lambda_kf=args.lambda_kf, ridge_eps=args.ridge_eps,
        delta_t_scale=1.0,
        cuts_per_tau=1024, kf_min_ratio=2.0, kf_min_abs=args.kf_min_trees,
        kf_group_batches=args.kf_group_batches,
        kf_variant="full_balancing", n_observations=2,
        supervision_mode="production",
        kf_taus=list(taus[:-1]),
        rpbe_seed=args.rpbe_seed,
        dense_future=False)
    compressor = RecursiveCompressor(rpbe_cfg).to(device)
    adapter = JodieTGNAdapter(
        tgn.embedding_module, compressor=compressor,
        n_neighbors=args.n_neighbors,
        trace_pairs_per_parent=args.trace_pairs_per_parent)
    tgn.embedding_module = adapter
    # two optimizers: head = decoder params only (LinkPredictor is built-in);
    # repr = host + compressor
    head_params = [p for p in tgn.affinity_score.parameters()
                   if p.requires_grad]
    repr_params = [p for p in tgn.parameters() if p.requires_grad]
    seen = set()
    repr_params = [p for p in repr_params
                   if not (id(p) in seen or seen.add(id(p)))]
    head_optimizer = torch.optim.Adam(head_params, lr=args.lr)
    repr_optimizer = torch.optim.Adam(repr_params, lr=args.lr)
    # boundary maps
    d_msg = int(ds.msg_dim)
    boundary_maps = BoundaryMaps(
        d_ctx=int(rpbe_cfg.d_c), d_event=int(rpbe_cfg.d_f), m=int(rpbe_cfg.m),
        d_msg=d_msg, delta_t_scale=1.0, msg_scale=1e-3,
        num_counter_bins=4096, seed=int(args.rpbe_seed)).to(device)
    boundary_maps.message_for = lambda eid: torch.as_tensor(
        ds.edge_features[int(eid)], dtype=torch.float32, device=device)
    link_future_index = LinkFutureIndex(
        train.sources, train.destinations, train.timestamps,
        train.edge_idxs)
    return dict(ds=ds, tgn=tgn, rpbe_cfg=rpbe_cfg, adapter=adapter,
                compressor=compressor, head_optimizer=head_optimizer,
                repr_optimizer=repr_optimizer, boundary_maps=boundary_maps,
                link_future_index=link_future_index,
                edge_table=ds.edge_features)


def main():
    args = parse_args()
    seed_all(args.seed)
    device = torch.device(
        "cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    c = build_model(args, device)
    ds = c["ds"]
    train = ds.train
    val = ds.val
    monitor = None  # minimal; metrics go to jsonl
    loop = TGBPairLinkLoop(
        tgn=c["tgn"], device=device, batch_size=args.bs,
        n_neighbors=args.n_neighbors, grad_clip=5.0, monitor=monitor,
        seed=args.seed, adapter=c["adapter"],
        link_future_index=c["link_future_index"],
        boundary_maps=c["boundary_maps"], edge_table=c["edge_table"],
        arm=args.arm, rpbe_cfg=c["rpbe_cfg"],
        repr_optimizer=c["repr_optimizer"], head_optimizer=c["head_optimizer"],
        trace_roots=args.trace_roots,
        trace_pairs_per_parent=args.trace_pairs_per_parent,
        kf_group_batches=args.kf_group_batches,
        kf_min_trees=args.kf_min_trees)

    save_json(out / "config.json", {
        "data": "tgbl-wiki", "seed": args.seed, "arm": args.arm,
        "device": str(device), "epochs": args.epochs, "bs": args.bs,
        "n_neighbors": args.n_neighbors, "n_layers": args.n_layers,
        "lambda_kf": args.lambda_kf, "kf_group_batches": args.kf_group_batches,
        "kf_min_trees": args.kf_min_trees, "cli": vars(args)})
    metrics_path = out / "metrics.jsonl"
    metrics_path.unlink(missing_ok=True)

    best_val = -1e9
    bad = 0
    best_epoch = -1
    gs = 0
    for epoch in range(args.epochs):
        t0 = time.time()
        row = loop.train_epoch(
            epoch, gs, train,
            max_batches=args.max_batches if args.max_batches > 0 else None)
        gs = row.get("global_step", gs)
        row["epoch"] = epoch
        row["epoch_seconds"] = time.time() - t0
        with metrics_path.open("a") as f:
            f.write(json.dumps(row, allow_nan=True) + "\n")
        print(json.dumps({"epoch": epoch, **{k: row[k] for k in
              ("train_link_loss", "n_closed", "n_below", "n_aux_batches")}},
              allow_nan=True), flush=True)
        # rolling best (val MRR selection happens in eval step; here we keep
        # a placeholder improving on train link loss so a best.pt exists)
        score = -float(row.get("train_link_loss", 1e9))
        if score > best_val:
            best_val = score
            best_epoch = epoch
            bad = 0
            torch.save({
                "model": {"tgn": c["tgn"].state_dict(),
                          "compressor": c["compressor"].state_dict()},
                "epoch": epoch, "score": float(best_val),
                "arm": args.arm, "seed": args.seed,
            }, out / "best.pt")
        else:
            bad += 1
            if bad >= args.patience:
                print("early stop at epoch {}".format(epoch), flush=True)
                break
    summary = {"data": "tgbl-wiki", "seed": args.seed, "arm": args.arm,
               "best_epoch": int(best_epoch), "best_val_link": float(best_val)}
    save_json(out / "summary.json", summary)
    save_json(out / "_SUCCESS.json", {"status": "complete",
                                      "best_epoch": int(best_epoch)})
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
