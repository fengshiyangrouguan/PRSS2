#!/usr/bin/env python3
"""Re-score every UCI run under BOTH metric aggregations without touching
the historic results (fairness audit 2026-09-13).

* global     — whole-stream concatenation, one sklearn AP/AUC (our default);
* batch_mean — per-batch AP/AUC averaged (the official eval_edge_prediction
               convention).

For ours/taskonly runs: rebuild the host + adapter/compressor from
config.json, load best.pt (ck["model"]["tgn"]), re-run evaluate_test.
For vanilla runs: rebuild the OFFICIAL TGN structure (no adapter) and load
the official saved_models/TGN-uci.pth.

Writes <run>/retest_agg.json (never overwrites summary.json).

Usage (server):
    python scripts/retest_uci_agg.py \
        --root outputs/uci_formal_v2 \
        --data-dir /root/autodl-tmp/benchtemp/data_uci \
        [--only seed0_TGN_3hop/ours] [--gpu 0]
"""
import argparse
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

from rpbe.data.uci_link import UCILinkDataset
from rpbe.hosts.uci_tgn import UciTGNAdapter, TAU_TEMPLATE
from rpbe.config import RPBConfig
from rpbe.compressor import RecursiveCompressor
from rpbe.training.uci_eval import evaluate_test, evaluate_split
from rpbe.hosts.official_tgn import get_neighbor_finder

try:
    from model.tgn import TGN
except ImportError:
    for _p in ("/root/autodl-tmp/benchtemp/experimental_codes/"
               "tgn-jodie-dyrep",):
        if _p not in sys.path:
            sys.path.insert(0, _p)
    from model.tgn import TGN


def seed_all(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_official_tgn(ds, device, nl, nn):
    train = ds.train
    finder = get_neighbor_finder(
        train, uniform=False, max_node_idx=ds.n_nodes - 1)
    full_finder = get_neighbor_finder(
        ds.full, uniform=False, max_node_idx=ds.n_nodes - 1)
    ts = train.timestamps.astype(np.float64)
    ms, ss = float(ts.mean()), float(ts.std()) + 1e-8
    tgn = TGN(
        neighbor_finder=finder,
        node_features=ds.node_features.astype(np.float32),
        edge_features=ds.edge_features.astype(np.float32),
        device=device,
        n_layers=nl,
        n_heads=2,
        dropout=0.1,
        use_memory=True,
        message_dimension=100,
        memory_dimension=172,
        memory_update_at_start=True,
        embedding_module_type="graph_attention",
        message_function="identity",
        aggregator_type="last",
        n_neighbors=nn,
        mean_time_shift_src=ms, std_time_shift_src=ss,
        mean_time_shift_dst=ms, std_time_shift_dst=ss,
    ).to(device)
    return tgn, full_finder


def build_ours(ds, device, cfg):
    tgn, full_finder = build_official_tgn(
        ds, device, int(cfg["n_layers"]), int(cfg["n_neighbors"]))
    host_dim = int(tgn.embedding_dimension)
    nl = int(cfg["n_layers"])
    taus = [TAU_TEMPLATE.format(l) for l in range(nl + 1)]
    rpbe_cfg = RPBConfig(
        state_dims={tau: host_dim for tau in taus},
        own_dims={tau: host_dim for tau in taus},
        width_D=128, m=64,
        lambda_kf=float(cfg.get("lambda_kf", 0.0)),
        ridge_eps=float(cfg.get("ridge_eps", 1e-3)),
        delta_t_scale=1.0,
        cuts_per_tau=1024, kf_min_ratio=2.0, kf_min_abs=896,
        kf_group_batches=int(cfg.get("kf_group_batches", 40)),
        kf_variant="full_balancing", n_observations=2,
        supervision_mode="production",
        kf_taus=list(taus[:-1]),
        rpbe_seed=int(cfg.get("rpbe_seed", 0)),
        dense_future=False)
    compressor = RecursiveCompressor(rpbe_cfg).to(device)
    adapter = UciTGNAdapter(
        tgn.embedding_module, compressor=compressor,
        n_neighbors=int(cfg["n_neighbors"]),
        trace_pairs_per_parent=2)
    tgn.embedding_module = adapter
    return tgn, full_finder, compressor


def retest_one(run_dir, ds, device, gpu):
    run_dir = Path(run_dir)
    out_path = run_dir / "retest_agg.json"
    cfg_path = run_dir / "config.json"
    is_vanilla = (run_dir.name == "vanilla")
    if not is_vanilla and not (run_dir / "best.pt").exists():
        return None
    if is_vanilla and not (run_dir / "saved_models" / "TGN-uci.pth").exists():
        return None
    seed = 0
    nl = 3
    nn = 10
    if cfg_path.exists():
        cfg = json.load(open(cfg_path))
        seed = int(cfg.get("seed", 0))
        nl = int(cfg.get("n_layers", 3))
        nn = int(cfg.get("n_neighbors", 10))
    else:  # vanilla without config.json: guess from the path
        cfg = {}
        m = os.path.basename(str(run_dir.parent))
        if "2hop" in m:
            nl = 2
        if "4L5N" in m:
            nl, nn = 4, 5
        if "5N" in m:
            nn = 5
        if "3L10N" in m or "3hop" in m:
            nl, nn = 3, 10
    seed_all(seed)
    if is_vanilla:
        tgn, full_finder = build_official_tgn(ds, device, nl, nn)
        ck = torch.load(run_dir / "saved_models" / "TGN-uci.pth",
                        map_location=device, weights_only=False)
        tgn.load_state_dict(ck)
    else:
        tgn, full_finder, _comp = build_ours(ds, device, cfg)
        ck = torch.load(run_dir / "best.pt", map_location=device,
                        weights_only=False)
        tgn.load_state_dict(ck["model"]["tgn"])
    results = {"run": str(run_dir), "vanilla": is_vanilla,
               "seed": seed, "n_layers": nl, "n_neighbors": nn}
    for agg in ("global", "batch_mean"):
        # fresh memory replay per aggregation (identical scoring protocol)
        r = evaluate_test(tgn, ds, n_neighbors=nn, bs=200,
                          full_finder=full_finder, agg=agg)
        results["agg_" + agg] = {
            "ap": round(float(r["ap"]), 6),
            "auc": round(float(r["auc"]), 6),
            "n_pos": int(r["n_pos"])}
        print("[retest] {} {} ap={:.4f} auc={:.4f}".format(
            run_dir.name, agg, r["ap"], r["auc"]), flush=True)
    json.dump(results, open(out_path, "w"), indent=2)
    return results


def main():
    ap = argparse.ArgumentParser("UCI dual-aggregation retest")
    ap.add_argument("--root", required=True)
    ap.add_argument("--data-dir",
                    default="/root/autodl-tmp/benchtemp/data_uci")
    ap.add_argument("--only", default="")
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    device = torch.device("cuda:{}".format(args.gpu)
                          if torch.cuda.is_available() else "cpu")
    ds = UCILinkDataset(args.data_dir, data_name="uci")
    root = Path(args.root)
    runs = []
    for seed_dir in sorted(root.iterdir()):
        if not seed_dir.is_dir():
            continue
        for run_dir in sorted(seed_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            rel = str(run_dir.relative_to(root))
            if args.only and args.only not in rel:
                continue
            if (run_dir / "best.pt").exists() or (
                    run_dir.name == "vanilla" and
                    (run_dir / "saved_models" / "TGN-uci.pth").exists()):
                runs.append(run_dir)
    print("retesting {} runs".format(len(runs)), flush=True)
    done = 0
    for run_dir in runs:
        try:
            retest_one(run_dir, ds, device, args.gpu)
            done += 1
        except Exception as e:  # one bad run must not kill the sweep
            print("[retest-error] {}: {}".format(run_dir, e), flush=True)
    print("retest done: {}/{}".format(done, len(runs)), flush=True)


if __name__ == "__main__":
    main()
