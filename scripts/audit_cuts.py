#!/usr/bin/env python3
"""Cut-funnel audit on the JODIE train stream (no training, no_grad forward).

Answers the "sample population" questions before any window design:
how many compression calls exist, what fraction carry each consumption
kind, how many probes align, how many valid rows each (tau, horizon)
produces, how much overlap and outcome reuse there is — plus a
pooled-vs-separate Ky Fan comparison on the collected rows.

Usage:
    python -m scripts.audit_cuts -d wikipedia \
        --data-dir old/processed_tgn_data --gpu 0 \
        [--max-batches N] [--every K] [--pool-rows N] \
        [--output outputs/audit/cut_funnel.json]
"""

import argparse
import json
import math
import os
import sys
from collections import Counter
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
from rpbe.hosts.jodie_tgn import JodieTGNAdapter, TAU_TEMPLATE
from rpbe.hosts.official_tgn import TGN, get_neighbor_finder
from rpbe.loss import kf_score
from rpbe.maps import FixedMaps
from rpbe.config import RPBConfig
from rpbe.records import NODE_CLASS, build_edge_tables, JodieCutBuilder


def parse_args():
    p = argparse.ArgumentParser("cut-funnel audit (no training)")
    p.add_argument("-d", "--data", default="wikipedia")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--bs", type=int, default=100)
    p.add_argument("--n-layer", type=int, default=2)
    p.add_argument("--n-degree", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-batches", type=int, default=0,
                   help="0 = whole train region")
    p.add_argument("--every", type=int, default=5,
                   help="audit one batch every K (funnel estimates)")
    p.add_argument("--pool-rows", type=int, default=4096,
                   help="rows collected for the pooled-vs-separate score")
    p.add_argument("--warmup-batches", type=int, default=20,
                   help="batches skipped before pooling (memory warmup: "
                        "wikipedia node features are all-zero, so early z "
                        "rows are degenerate)")
    p.add_argument("--output", default="outputs/audit/cut_funnel.json")
    return p.parse_args()


def seed_all(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_host(dataset, args, device):
    full, train, _, _ = dataset.splits()
    finder = get_neighbor_finder(train, uniform=False,
                                 max_node_idx=max(full.unique_nodes))
    ms, ss, md, sd = dataset.time_stats()
    tgn = TGN(
        neighbor_finder=finder,
        node_features=dataset.node_features,
        edge_features=dataset.edge_features,
        device=device,
        n_layers=args.n_layer,
        n_heads=2,
        dropout=0.0,
        use_memory=True,
        message_dimension=100,
        memory_dimension=172,
        memory_update_at_start=True,
        embedding_module_type="graph_attention",
        message_function="identity",
        aggregator_type="last",
        n_neighbors=args.n_degree,
        mean_time_shift_src=ms,
        std_time_shift_src=ss,
        mean_time_shift_dst=md,
        std_time_shift_dst=sd,
    ).to(device)
    return tgn


def main():
    args = parse_args()
    seed_all(args.seed)
    device = torch.device(
        "cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")

    dataset = JodieDataset(args.data, data_dir=args.data_dir,
                           use_validation=True)
    full, train, val, test = dataset.splits()
    print("train={} val={} test={} val_time={:.1f}".format(
        len(train.sources), len(val.sources), len(test.sources),
        dataset.val_time), flush=True)

    tgn = build_host(dataset, args, device)
    (endpoints, labels_tbl, user_nodes, page_nodes,
     table_stats) = build_edge_tables(dataset)
    edge_tables = (endpoints, labels_tbl, user_nodes, page_nodes)
    adapter = JodieTGNAdapter(tgn.embedding_module, compressor=None,
                              n_neighbors=args.n_degree,
                              edge_tables=edge_tables)
    tgn.embedding_module = adapter

    taus = [TAU_TEMPLATE.format(l) for l in range(args.n_layer + 1)]
    host_dim = int(tgn.embedding_dimension)
    cfg = RPBConfig(
        state_dims={tau: host_dim for tau in taus},
        own_dims={tau: host_dim for tau in taus},
        m=64, rpbe_seed=args.seed,
        delta_t_scale=float(np.median(np.diff(np.sort(full.timestamps)))) or 1.0,
        # Audit must see the UNFILTERED population: the cap is bypassed
        # with a huge per-tau budget.
        cuts_per_tau=10 ** 9,
        kf_min_abs=10 ** 9)
    maps = FixedMaps(cfg).to(device)
    cut_builder = JodieCutBuilder(edge_tables,
                                  stage=NODE_CLASS, seed=args.seed,
                                  cuts_per_tau=10 ** 9)

    stats = {}
    pooled_rows = []
    pool_counts = {}
    n_batches_seen = 0
    num_batch = math.ceil(len(train.sources) / args.bs)
    max_batches = args.max_batches or num_batch

    tgn.eval()
    with torch.no_grad():
        for k in range(0, min(max_batches, num_batch), args.every):
            s = k * args.bs
            e = min(num_batch, k + 1) * args.bs
            sources = train.sources[s:e]
            dests = train.destinations[s:e]
            times = train.timestamps[s:e]
            edge_idxs = train.edge_idxs[s:e]
            labels_np = train.labels[s:e]
            size = len(sources)
            # Audit builds trees for EVERY row of the batch (training
            # traces a subset; the funnel must see the full population).
            adapter.set_trace_source_rows(list(range(size)))
            tgn.compute_temporal_embeddings(
                sources, dests, dests, times, edge_idxs, args.n_degree)
            trace = adapter.trace
            root_events = {
                row: {"dst": int(dests[row]), "label": float(labels_np[row]),
                      "time": float(times[row]),
                      "event_idx": int(edge_idxs[row])}
                for row in range(size)}
            rows = cut_builder.build(trace, root_events=root_events,
                                     batch_seed=k, stats=stats)
            # Stratified pool: postorder emits leaf rows first, so a plain
            # first-N pool would be flooded by layer0 and never see the
            # upper interfaces.  Each tau gets pool_rows / n_taus quota.
            # Pooling starts AFTER the warmup (memory is zero-initialized
            # and wikipedia node features are all-zero, so the first
            # batches carry degenerate z rows).
            if n_batches_seen > args.warmup_batches:
                for r in rows:
                    tau_quota = max(1, args.pool_rows // max(1, len(taus)))
                    if pool_counts.get(r.tau, 0) >= tau_quota:
                        continue
                    if len(pooled_rows) < args.pool_rows:
                        pooled_rows.append(r)
                        pool_counts[r.tau] = pool_counts.get(r.tau, 0) + 1
            n_batches_seen += 1
            if n_batches_seen % 20 == 0:
                print("batch {} / {} (rows so far {})".format(
                    k, min(max_batches, num_batch), len(pooled_rows)),
                    flush=True)

    # ------------------------------- pooled-vs-separate ----------------------
    pool_report = {}
    if pooled_rows:
        zs = torch.stack([r.z for r in pooled_rows]).to(device)
        ps = maps.pv_batch([r.context for r in pooled_rows],
                           [r.outcome for r in pooled_rows])
        z_all = zs.double().detach()
        p_all = ps.double().detach()
        by_tau = {}
        for i, r in enumerate(pooled_rows):
            by_tau.setdefault(r.tau, []).append(i)
        per_tau_j = {}
        for tau, idx in by_tau.items():
            j = kf_score(z_all[idx].float(), p_all[idx].float(), eps=cfg.ridge_eps)
            per_tau_j[tau] = float(j.detach())
        j_pool = kf_score(z_all.float(), p_all.float(), eps=cfg.ridge_eps)
        pool_report = {
            "n_rows": len(pooled_rows),
            "per_tau": per_tau_j,
            "sum_per_tau": float(sum(per_tau_j.values())),
            "pooled": float(j_pool.detach()),
            "rows_by_tau": {t: len(i) for t, i in by_tau.items()},
        }

    # ------------------------------- aggregate ------------------------------
    raw = stats.get("raw_occurrences", {})
    kinds = stats.get("consumption_kind", {})
    valid_rows = stats.get("valid_rows", {})
    overlap = stats.get("overlap_groups", {})
    outcome_use = stats.get("outcome_use", {})

    kind_by_tau = {}
    for (tau, kind), c in kinds.items():
        kind_by_tau.setdefault(tau, {})[kind] = c
    valid_by_tau = {}
    for (tau, h), c in valid_rows.items():
        valid_by_tau.setdefault(tau, {})["h{}".format(h)] = c

    def mult_summary(cnt):
        if not cnt:
            return {"n": 0, "max_mult": 0, "mean_mult": 0.0}
        vals = list(cnt.values())
        return {"n": len(cnt), "max_mult": max(vals),
                "mean_mult": float(np.mean(vals)),
                "mult_hist": {str(k): v for k, v in
                              Counter(vals).most_common(12)}}

    def ess_summary(cnt):
        vals = np.asarray(list(cnt.values()), dtype=np.float64)
        w = vals.sum()
        w2 = float((vals * vals).sum())
        return {"n_groups": len(cnt), "W": float(w),
                "w_eff": (w * w / w2) if w2 > 0 else 0.0}

    per_tau_cuts = Counter()
    per_tau_trees = {}
    tree_seen = set()
    # tree ids live on rows; reconstruct from pooled stats is partial, so
    # count per-tau trees only over the pooled subsample (flagged).
    for r in pooled_rows:
        per_tau_cuts[r.tau] += 1
        per_tau_trees.setdefault(r.tau, set()).add(r.tree_id)

    edge_outcomes = {str(k): v for k, v in outcome_use.items()
                     if k[0] == "edge"}
    root_outcomes = {str(k): v for k, v in outcome_use.items()
                     if k[0] == "root"}

    structure = {
        "cut_node_type_by_tau": {str(k): v
                                 for k, v in stats.get("cut_node_type",
                                                       {}).items()},
        "owner_position_by_tau": {str(k): v
                                  for k, v in stats.get("owner_position",
                                                        {}).items()},
        "parent_child_same_type": stats.get("parent_child_same_type", 0),
        "bipartite_alternation_ok": stats.get("parent_child_same_type", 0) == 0,
    }

    report = {
        "args": vars(args),
        "n_batches_seen": n_batches_seen,
        "edge_table": table_stats,
        "structure": structure,
        "funnel": {
            "raw_occurrences_by_tau": raw,
            "consumption_kind_by_tau": kind_by_tau,
            "aligned_probes": stats.get("aligned_probes", 0),
            "unaligned_probes": stats.get("unaligned_probes", 0),
            "self_steps_skipped": stats.get("self_steps_skipped", 0),
            "root_records_used": stats.get("root_records_used", 0),
            "depth_terminated_walks": stats.get("depth_terminated", 0),
            "valid_rows_by_tau": valid_by_tau,
        },
        "overlap": {"rows": sum(overlap.values()),
                    "groups": len(overlap),
                    **ess_summary(overlap)},
        "outcome": {
            "edge": mult_summary(edge_outcomes),
            "root": mult_summary(root_outcomes),
            "unique_outcomes": len(outcome_use),
            "outcome_cluster_ess": ess_summary(outcome_use),
        },
        "per_tau_pool_subsample": {
            tau: {"n_cuts": c, "n_trees": len(per_tau_trees.get(tau, set()))}
            for tau, c in per_tau_cuts.items()},
        "r_tau": {tau: host_dim for tau in taus},
        "pooled_vs_separate": pool_report,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(report, f, indent=2, allow_nan=True)
    print(json.dumps(report, indent=2), flush=True)
    print("wrote {}".format(out), flush=True)


if __name__ == "__main__":
    main()
