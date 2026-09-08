#!/usr/bin/env python3
"""TGB tgbl-wiki MRR evaluation (read-only, strictly-equivalent).

Replays TRAIN memory online, then scores a split on the fixed query-id set
(shared across arms/seeds) via ``rpbe.link_eval.score_split``: source
embedding once per event, positive + full official negatives chunked against
the same pre-event memory, avg tie rank.  Memory is advanced over the split
per real scored event (each positive once; negatives never update state).

Metric: ``sampled_query_<split>_mrr`` (not a full-TGB-MRR claim).
``test`` never selects checkpoints.
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

from rpbe.data.tgb_link import TGBLinkDataset
from rpbe.hosts.official_tgn import TGN, get_neighbor_finder
from rpbe.hosts.jodie_tgn import JodieTGNAdapter, TAU_TEMPLATE
from rpbe.config import RPBConfig
from rpbe.compressor import RecursiveCompressor
from rpbe.link_eval import score_online_sampled


def parse_args():
    p = argparse.ArgumentParser("tgbl-wiki MRR eval (read-only chunked)")
    p.add_argument("--output", required=True)
    p.add_argument("--data-dir", default="datasets")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--query-sets", default="datasets/tgb_wiki_query_sets.json")
    p.add_argument("--max-n", type=int, default=0,
                   help="cap scored queries (0 = all in fixed set)")
    p.add_argument("--dst-chunk", type=int, default=64)
    return p.parse_args()


def _rebuild(cfg_cli, device, ds):
    args = argparse.Namespace(**cfg_cli)
    full, train, val, test = ds.splits()
    finder = get_neighbor_finder(
        train, uniform=False, max_node_idx=int(ds.n_internal_nodes - 1))
    ts = train.timestamps.astype(np.float64)
    ms, ss = float(ts.mean()), float(ts.std()) + 1e-8
    tgn = TGN(
        neighbor_finder=finder,
        node_features=ds.node_features.astype(np.float32),
        edge_features=ds.edge_features.astype(np.float32),
        device=device, n_layers=args.n_layers, n_heads=2, dropout=0.1,
        use_memory=True, message_dimension=100, memory_dimension=172,
        memory_update_at_start=True,
        embedding_module_type="graph_attention", message_function="identity",
        aggregator_type="last", n_neighbors=args.n_neighbors,
        mean_time_shift_src=ms, std_time_shift_src=ss,
        mean_time_shift_dst=ms, std_time_shift_dst=ss).to(device)
    host_dim = int(tgn.embedding_dimension)
    taus = [TAU_TEMPLATE.format(l) for l in range(args.n_layers + 1)]
    rpbe_cfg = RPBConfig(
        state_dims={tau: host_dim for tau in taus},
        own_dims={tau: host_dim for tau in taus},
        width_D=args.width_D, m=args.sketch_dim,
        lambda_kf=args.lambda_kf, ridge_eps=args.ridge_eps,
        delta_t_scale=1.0, cuts_per_tau=1024, kf_min_ratio=2.0,
        kf_min_abs=args.kf_min_trees, kf_group_batches=args.kf_group_batches,
        kf_variant="full_balancing", n_observations=2,
        supervision_mode="production", kf_taus=list(taus[:-1]),
        rpbe_seed=args.rpbe_seed, dense_future=False)
    comp = RecursiveCompressor(rpbe_cfg).to(device)
    adapter = JodieTGNAdapter(tgn.embedding_module, compressor=comp,
                              n_neighbors=args.n_neighbors,
                              trace_pairs_per_parent=0)
    tgn.embedding_module = adapter
    return tgn, comp


def _advance(tgn, src, dst, t, eidx, bs=1):
    """Advance memory over real positives once each (scores discarded)."""
    with torch.no_grad():
        tgn.compute_edge_probabilities(
            np.asarray(src, dtype=np.int64),
            np.asarray(dst, dtype=np.int64),
            np.asarray(dst, dtype=np.int64),
            np.asarray(t, dtype=np.float64),
            np.asarray(eidx, dtype=np.int64), 10)


def main():
    args = parse_args()
    outdir = Path(args.output)
    cfg = json.load(open(outdir / "config.json"))
    cli = cfg["cli"]
    device = torch.device(
        "cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    ds = TGBLinkDataset(root=args.data_dir)
    full, train, val, test = ds.splits()
    tgn, comp = _rebuild(cli, device, ds)
    best = torch.load(outdir / "best.pt", map_location=device,
                      weights_only=False)
    for k in ("tgn", "compressor"):
        if k in best["model"]:
            (tgn if k == "tgn" else comp).load_state_dict(best["model"][k])
    for p in tgn.parameters():
        p.requires_grad_(False)
    for p in comp.parameters():
        p.requires_grad_(False)
    tgn.eval()
    tgn.embedding_module.clear_trace()
    if args.split == "val":
        ds.load_val_ns()
        split_mode = "val"
    else:
        ds.load_test_ns()
        split_mode = "test"
    # fixed shared query ids
    qs = json.load(open(args.query_sets))
    qids = qs["{}_query_ids".format(split_mode)]
    if args.max_n > 0:
        qids = qids[:args.max_n]
    # replay train memory
    if tgn.use_memory:
        tgn.memory.__init_memory__()
    t0 = time.time()
    _advance(tgn, train.sources, train.destinations, train.timestamps,
             train.edge_idxs, bs=200)
    print("train replay {:.1f}s".format(time.time() - t0), flush=True)

    # correct online sampled-query MRR: memory advances over ALL val events
    # (true edge idx), not only the scored subset.
    t1 = time.time()
    res = score_online_sampled(tgn, ds, split_mode, qids,
                               n_neighbors=args.n_neighbors,
                               chunk=args.dst_chunk)
    res["scored_seconds"] = time.time() - t1
    metric = "sampled_query_{}_mrr".format(split_mode)
    out_json = outdir / "eval_{}.json".format(split_mode)
    with open(out_json, "w") as f:
        json.dump(res, f, indent=2, allow_nan=True)
    print(json.dumps(res, indent=2), flush=True)


if __name__ == "__main__":
    main()
