#!/usr/bin/env python3
"""TGB tgbl-wiki MRR evaluation for a trained arm (official protocol).

Replays train memory in online order (each real edge once), then evaluates
the val split with the OFFICIAL negative sampler and MRR.  ``--split test``
runs the held-out test (locked once).  The runner's best.pt is loaded; the
host runs in eval mode (auxiliary / future / trace / fixed maps off).

Usage:
    python -m scripts.eval_tgb_link --output <arm-dir> --data-dir datasets \
        --gpu 0 [--split val]
"""

import argparse
import json
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

from rpbe.data.tgb_link import TGBLinkDataset
from rpbe.hosts.official_tgn import TGN, get_neighbor_finder
from rpbe.hosts.jodie_tgn import JodieTGNAdapter, TAU_TEMPLATE
from rpbe.config import RPBConfig
from rpbe.compressor import RecursiveCompressor


def parse_args():
    p = argparse.ArgumentParser("tgbl-wiki MRR eval")
    p.add_argument("--output", required=True)   # arm run dir (has best.pt)
    p.add_argument("--data-dir", default="datasets")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--split", choices=["val", "test"], default="val")
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
        node_features=torch.as_tensor(ds.node_features, dtype=torch.float32),
        edge_features=torch.as_tensor(ds.edge_features, dtype=torch.float32),
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


def _mrr_from_scores(pos_scores, neg_scores_list):
    """pos_scores[i] vs neg_scores_list[i] (list of neg scores). MRR."""
    ranks = []
    for ps, ns in zip(pos_scores, neg_scores_list):
        # rank of the positive among {positive} U {negatives} (higher=better)
        r = 1.0 + float(sum(1 for x in ns if x > ps))
        ranks.append(1.0 / r)
    return float(np.mean(ranks)) if ranks else float("nan")


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
        split = val
    else:
        ds.load_test_ns()
        split_mode = "test"
        split = test
    raw_src, raw_dst, raw_t = ds.raw_split(split_mode)
    # rebuild memory online over train (each real edge once, memory update)
    if tgn.use_memory:
        tgn.memory.__init_memory__()
    bs = 200
    n = len(train.sources)
    with torch.no_grad():
        for s0 in range(0, n, bs):
            s1 = min(n, s0 + bs)
            tgn.compute_edge_probabilities(
                train.sources[s0:s1], train.destinations[s0:s1],
                train.destinations[s0:s1], train.timestamps[s0:s1],
                train.edge_idxs[s0:s1], cli["n_neighbors"])
    # score val/test positives against official negatives
    pos_scores = []
    neg_scores_all = []
    nv = len(raw_src)
    with torch.no_grad():
        for s0 in range(0, nv, bs):
            s1 = min(nv, s0 + bs)
            # positives as internal ids
            pos_src = (raw_src[s0:s1] + 1).astype(np.int64)
            pos_dst = (raw_dst[s0:s1] + 1).astype(np.int64)
            pos_t = raw_t[s0:s1]
            # gather official negatives for this row block (raw ids)
            negs = ds.query_negatives(raw_src[s0:s1], raw_dst[s0:s1],
                                      raw_t[s0:s1], split_mode=split_mode)
            # internal dst ids for all negatives of all rows in block
            flat = [int(x) for sub in negs for x in sub]
            flat_internal = np.asarray(flat, dtype=np.int64) + 1
            # score each positive once and all its negatives via a single
            # batch: query source against all candidate dsts
            size = len(pos_src)
            # build [pos_src repeated per candidate]; candidates per row vary
            p_scores = []
            n_scores = []
            for i in range(size):
                cands = np.concatenate([[pos_dst[i]], flat_internal[
                    sum(len(x) for x in negs[:i]):
                    sum(len(x) for x in negs[:i + 1])]])
                src_rep = np.full(len(cands), pos_src[i], dtype=np.int64)
                # TGN queries src,dst,neg = 3x each; approximate via
                # compute_edge_probabilities needs negatives; use
                # compute_temporal_embeddings + affinity manually
                emb_s, emb_d, _ = tgn.compute_temporal_embeddings(
                    src_rep, cands, cands,
                    np.full(len(cands), float(pos_t[i])),
                    np.full(len(cands), 0, dtype=np.int64),
                    cli["n_neighbors"])
                sc = tgn.affinity_score(
                    torch.cat([emb_s, emb_s], dim=0),
                    torch.cat([emb_d, emb_d], dim=0)).squeeze(0)
                ps = float(sc[0])
                p_scores.append(ps)
                n_scores.append([float(x) for x in sc[1:]])
            pos_scores.extend(p_scores)
            neg_scores_all.extend(n_scores)
    mrr = _mrr_from_scores(pos_scores, neg_scores_all)
    result = {"split": args.split, "mrr": mrr,
              "n_pos": len(pos_scores)}
    out_json = outdir / "eval_{}.json".format(args.split)
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2, allow_nan=True)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
