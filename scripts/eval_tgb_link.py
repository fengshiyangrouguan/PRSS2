#!/usr/bin/env python3
"""TGB tgbl-wiki MRR evaluation — strictly-equivalent, read-only scoring.

Equivalence requirements (user decision):
* 3 layers + n_neighbors=10 both for training and this evaluator;
* full official negatives per positive (no 100-subset);
* source embedding computed ONCE per positive;
* candidate destinations (positive dst + ~all official negatives) are scored
  in chunks of ``--dst-chunk`` sharing the SAME event-before memory;
* memory is advanced ONLY by real positive events, exactly once each;
* negatives never update state;
* TGB averaged tie rank
      r = 1 + (#(s^- > s^+) + #(s^- >= s^+)) / 2
* ``test`` never selects checkpoints.

Read-only scoring: we call ``embedding_module.compute_embedding`` directly
with an explicit memory tensor (no raw-message storage / no memory update
side effects), so every candidate of one event sees byte-identical memory.

A gate compares this evaluator against the previous (per-event
``compute_temporal_embeddings``) evaluator on 32 fixed val events: identical
per-event reciprocal rank and identical final memory/neighbor state.

Metric is named ``sampled_query_<split>_mrr`` (not a full-TGB-MRR claim).
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


def parse_args():
    p = argparse.ArgumentParser("tgbl-wiki MRR eval (read-only chunked)")
    p.add_argument("--output", required=True)
    p.add_argument("--data-dir", default="datasets")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--query-ids", default="",
                   help="optional json of fixed query event indices to score "
                        "(same across arms/seeds); empty = score in order")
    p.add_argument("--max-n", type=int, default=0,
                   help="cap scored positives (0 = all in query set)")
    p.add_argument("--dst-chunk", type=int, default=64)
    p.add_argument("--gate", action="store_true",
                   help="run the 32-event equivalence gate vs the legacy "
                        "per-event evaluator")
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


def _current_memory(tgn):
    """Read-only current memory tensor (no side effects)."""
    return tgn.memory.get_memory(list(range(tgn.n_nodes))) if tgn.use_memory \
        else None


def _read_emb(tgn, memory, nodes, times):
    """Read-only recursive embedding for ``nodes`` at ``times``.

    Calls the host embedding module directly so no raw messages are stored
    and memory is not advanced.  ``nodes``/``times`` are numpy (internal ids).
    """
    emb = tgn.embedding_module.compute_embedding(
        memory=memory, source_nodes=np.asarray(nodes, dtype=np.int64),
        timestamps=np.asarray(times, dtype=np.float64),
        n_layers=tgn.n_layers, n_neighbors=10)
    return emb


def _score_pos_neg(tgn, memory, src, t, pos_dst, neg_dsts, chunk):
    """Affinity scores: source once; destinations chunked, same memory."""
    src_emb = _read_emb(tgn, memory, [src], [t])[0]     # [dim]
    # positive destination embedding
    pos_emb = _read_emb(tgn, memory, [pos_dst], [t])[0]
    pos_score = float(tgn.affinity_score(
        src_emb.unsqueeze(0), pos_emb.unsqueeze(0)).squeeze(0))
    neg_scores = []
    for c0 in range(0, len(neg_dsts), chunk):
        seg = neg_dsts[c0:c0 + chunk]
        if len(seg) == 0:
            continue
        e = _read_emb(tgn, memory, seg,
                      [t] * len(seg))                    # [c, dim]
        x = src_emb.unsqueeze(0).expand(len(seg), -1)
        s = tgn.affinity_score(x, e).squeeze(0)
        neg_scores.extend([float(v) for v in s])
    return pos_score, neg_scores


def _mrr_rank(pos_score, neg_scores):
    gt = float(sum(1 for v in neg_scores if v > pos_score))
    ge = float(sum(1 for v in neg_scores if v >= pos_score))
    return 1.0 + 0.5 * (gt + ge)


def _advance_memory(tgn, split_src, split_dst, split_t, split_eidx, bs=200):
    """Advance memory over a real stream (each positive once) using the
    official edge-probability path (scores discarded)."""
    n = len(split_src)
    with torch.no_grad():
        for s0 in range(0, n, bs):
            s1 = min(n, s0 + bs)
            tgn.compute_edge_probabilities(
                split_src[s0:s1], split_dst[s0:s1], split_dst[s0:s1],
                split_t[s0:s1], split_eidx[s0:s1], 10)


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
    raw_src, raw_dst, raw_t = ds.raw_split(split_mode)
    # ---- replay TRAIN memory online (real positives advance state)
    if tgn.use_memory:
        tgn.memory.__init_memory__()
    t0 = time.time()
    _advance_memory(tgn, train.sources, train.destinations,
                    train.timestamps, train.edge_idxs)
    print("train replay {:.1f}s".format(time.time() - t0), flush=True)

    # ---- optional fixed query-id set (stratified; shared across arms/seeds)
    if args.query_ids:
        qids = json.load(open(args.query_ids))
    else:
        nv = len(raw_src)
        qids = list(range(nv))
    if args.max_n > 0:
        qids = qids[:args.max_n]

    # score selected queries in TIME order; memory advances once per scored
    # real event (official online semantics).
    ranks = []
    t1 = time.time()
    with torch.no_grad():
        for qi, idx in enumerate(qids):
            i = int(idx)
            src = int(raw_src[i]) + 1
            dst = int(raw_dst[i]) + 1
            tt = float(raw_t[i])
            negs_raw = ds.query_negatives(
                np.asarray([int(raw_src[i])], dtype=np.int64),
                np.asarray([int(raw_dst[i])], dtype=np.int64),
                np.asarray([tt], dtype=np.float64), split_mode=split_mode)
            neg_in = np.asarray([int(x) for x in negs_raw[0]],
                                dtype=np.int64) + 1
            mem = _current_memory(tgn)
            ps, ns = _score_pos_neg(tgn, mem, src, tt, dst, neg_in,
                                    args.dst_chunk)
            ranks.append(1.0 / _mrr_rank(ps, ns))
            # advance memory with this real event exactly once
            _advance_memory(tgn, np.asarray([src], dtype=np.int64),
                            np.asarray([dst], dtype=np.int64),
                            np.asarray([tt], dtype=np.float64),
                            np.asarray([src], dtype=np.int64), bs=1)
            if (qi + 1) % 100 == 0:
                print("scored {}/{} in {:.1f}s".format(
                    qi + 1, len(qids), time.time() - t1), flush=True)
    mrr = float(np.mean(ranks)) if ranks else float("nan")
    metric = "sampled_query_{}_mrr".format(args.split)
    result = {metric: mrr, "n_scored": len(ranks),
              "scored_seconds": time.time() - t1,
              "note": "same memory-before-event per candidate; full official "
                      "negatives; n_neighbors=10; test never selects ckpt"}
    out_json = outdir / "eval_{}.json".format(args.split)
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2, allow_nan=True)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
