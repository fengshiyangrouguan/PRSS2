#!/usr/bin/env python3
"""A0 conditional-moment tree compression on the official TGN host.

Independent entry point for the four-phase A0 protocol (theory doc
2026-08-20): phase A calibrates per-interface rank-r coordinate maps from the
conditional future moment; phase B fits per-constructor recursive operators
by one-shot convex ridge; phase C audits prediction/closure/support/gain with
the G0-G4 gates; phase D trains the A0 readout (z_root = R x_root) and the
baseline decoder (x_root) head-to-head on the same frozen host forward.

The train_jodie line is untouched; the vanilla PRSS core only produces the
trace.  Outputs the same four files as the JODIE line (config.json,
metrics.jsonl, summary.json, _SUCCESS.json).

Example:
    python -m scripts.train_a0 -d wikipedia --data-dir old/processed_tgn_data \
        --pretrained-checkpoint outputs/pretrained/tgn-wikipedia.pth \
        --output outputs/a0/r32_seed0 --r 32
"""

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from prss.a0.probes import A0Probes
from prss.config import InterfaceSpec, PRSSConfig
from prss.core import PRSSCore
from prss.data.jodie import JodieDataset
from prss.hosts.jodie_tgn import (JodieTGNAdapter, TAU_TEMPLATE,
                                  jodie_preagg_dim)
from prss.hosts.official_tgn import TGN, get_neighbor_finder
from prss.monitoring import MonitorWriter
from prss.training.a0_loop import A0NodeClassificationLoop


def parse_args():
    p = argparse.ArgumentParser("A0 conditional-moment tree compression (TGN host)")
    p.add_argument("-d", "--data", default="wikipedia")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--pretrained-checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu", type=int, default=0)
    # Upstream TGN names/defaults (must match the pretrained checkpoint).
    p.add_argument("--bs", type=int, default=100)
    p.add_argument("--n-degree", type=int, default=10)
    p.add_argument("--n-head", type=int, default=2)
    p.add_argument("--n-layer", type=int, default=2)
    p.add_argument("--drop-out", type=float, default=0.1)
    p.add_argument("--message-dim", type=int, default=100)
    p.add_argument("--memory-dim", type=int, default=172)
    # A0 calibration.
    p.add_argument("--r", type=int, default=32, help="Fixed rank budget")
    p.add_argument("--d-context", type=int, default=32,
                   help="Context probe width (fixed random projection)")
    p.add_argument("--lambda-x", type=float, default=1e-4,
                   help="Phase-A ridge on C_xx")
    p.add_argument("--lambda-gamma", type=float, default=1e-3,
                   help="Phase-B ridge on the interaction design")
    p.add_argument("--lambda-audit", type=float, default=1e-3,
                   help="Phase-C audit ridge")
    p.add_argument("--frac-a", type=float, default=0.2)
    p.add_argument("--frac-b", type=float, default=0.2)
    p.add_argument("--frac-c", type=float, default=0.2)
    p.add_argument("--d-slice-only", action="store_true",
                   help="Restrict phase D to the post-C rows (default: full train)")
    p.add_argument("--trace-roots", type=int, default=16)
    p.add_argument("--trace-mode", default="evenly_spaced",
                   choices=["positive_first", "evenly_spaced", "off"])
    # Phase D readout training.
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--n-epoch", type=int, default=10)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--selection-metric", choices=["auc", "ap"], default="auc")
    # Failure gates (None = report only; set to enforce in stop mode).
    p.add_argument("--gate-mode", choices=["report", "stop"], default="report")
    p.add_argument("--g1-max-rank-tail", type=float, default=None)
    p.add_argument("--g2-max-closure-resid", type=float, default=None)
    p.add_argument("--g3-max-gain-product", type=float, default=None)
    p.add_argument("--g4-min-auc-delta", type=float, default=None)
    # Smoke caps / monitoring.
    p.add_argument("--monitor-every", type=int, default=50)
    p.add_argument("--no-fail-on-monitor-error", action="store_true")
    p.add_argument("--max-train", type=int, default=0)
    p.add_argument("--max-val", type=int, default=0)
    p.add_argument("--max-test", type=int, default=0)
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


def load_torch(path, device):
    return torch.load(path, map_location=device, weights_only=False)


def unwrap_state(obj):
    if isinstance(obj, dict):
        for key in ("model_state_dict", "tgn"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
    return obj


def cap_data(data, cap):
    if not cap or len(data.sources) <= cap:
        return data
    return data.slice(0, cap)


def build_components(args, device, dataset):
    """Assembly order matters: pretrained keys validated and host frozen
    BEFORE the embedding module is swapped for the adapter (same contract as
    train_jodie).  The PRSS core is vanilla — an identity trace producer."""
    full, train, val, test = dataset.splits()
    train_finder = get_neighbor_finder(train, uniform=False,
                                       max_node_idx=max(full.unique_nodes))
    ms, ss, md, sd = dataset.time_stats()
    tgn = TGN(
        neighbor_finder=train_finder,
        node_features=dataset.node_features,
        edge_features=dataset.edge_features,
        device=device,
        n_layers=args.n_layer,
        n_heads=args.n_head,
        dropout=args.drop_out,
        use_memory=True,
        message_dimension=args.message_dim,
        memory_dimension=args.memory_dim,
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
    state = unwrap_state(load_torch(args.pretrained_checkpoint, device))
    # Same whitelist contract as train_jodie: runtime memory keys skipped
    # (they are sized for the unpadded node set), affinity_score.* allowed
    # missing (self-supervised decoder, replaced by the supervised heads).
    MEMORY_STATE_KEYS = ("memory.memory", "memory.last_update",
                         "memory_updater.memory.memory",
                         "memory_updater.memory.last_update")
    DECODER_STATE_KEYS = ("affinity_score.fc1.weight", "affinity_score.fc1.bias",
                          "affinity_score.fc2.weight", "affinity_score.fc2.bias")
    for key in MEMORY_STATE_KEYS:
        state.pop(key, None)
    allowed_missing = set(MEMORY_STATE_KEYS) | set(DECODER_STATE_KEYS)
    missing, unexpected = tgn.load_state_dict(state, strict=False)
    if not (set(missing) <= allowed_missing) or unexpected:
        raise RuntimeError(
            "pretrained TGN mismatch: missing={}, unexpected={}".format(
                sorted(missing), sorted(unexpected)))
    for p in tgn.parameters():
        p.requires_grad_(False)
    tgn.eval()

    # Vanilla PRSS core: every interface is identity (candidate_dim ==
    # host_dim), so the adapter produces the trace with zero behavior change.
    host_dim = int(tgn.embedding_dimension)
    time_dim = int(tgn.embedding_module.n_time_features)
    edge_dim = int(tgn.n_edge_features)
    taus = [TAU_TEMPLATE.format(l) for l in range(args.n_layer + 1)]
    interfaces = {
        tau: InterfaceSpec(tau, raw_dim=host_dim, candidate_dim=host_dim,
                           host_dim=host_dim, response_dim=1)
        for tau in taus}
    preagg_dim = jodie_preagg_dim(host_dim, time_dim, edge_dim, args.n_degree)
    config = PRSSConfig(
        interfaces=interfaces,
        context_dim=args.d_context,
        root_metadata_dim=1,
        parent_local_dim=preagg_dim,
        relation_count=2,
        relation_dim=16,
        outside_layers=2,
        variant="vanilla",
    )
    prss_core = PRSSCore(config, variant="vanilla").to(device)
    adapter = JodieTGNAdapter(tgn.embedding_module, prss_core,
                              n_neighbors=args.n_degree)
    probes = A0Probes(preagg_dim=preagg_dim, d_context=args.d_context,
                      seed=args.seed, device=device)
    # Swap AFTER pretrained-key validation and freezing.
    tgn.embedding_module = adapter

    return dict(tgn=tgn, adapter=adapter, prss_core=prss_core, probes=probes,
                host_dim=host_dim, preagg_dim=preagg_dim), config


def main():
    args = parse_args()
    seed_all(args.seed)
    device = torch.device("cuda:{}".format(args.gpu)
                          if torch.cuda.is_available() else "cpu")
    print("device:", device, flush=True)

    dataset = JodieDataset(args.data, data_dir=args.data_dir,
                           use_validation=True)
    full, train, val, test = dataset.splits()
    train = cap_data(train, args.max_train)
    val = cap_data(val, args.max_val)
    test = cap_data(test, args.max_test)
    print("splits: train={} val={} test={} (positives {}/{}/{})".format(
        train.n_interactions, val.n_interactions, test.n_interactions,
        int(train.labels.sum()), int(val.labels.sum()), int(test.labels.sum())),
        flush=True)

    components, prss_config = build_components(args, device, dataset)
    tgn = components["tgn"]

    gates = {
        "G1": args.g1_max_rank_tail,
        "G2": args.g2_max_closure_resid,
        "G3": args.g3_max_gain_product,
        "G4": args.g4_min_auc_delta,
    }
    config_out = {
        "protocol": "a0",
        "variant": "a0",
        "data": args.data,
        "seed": args.seed,
        "device": str(device),
        "host_dim": components["host_dim"],
        "preagg_dim": components["preagg_dim"],
        "r": args.r,
        "d_context": args.d_context,
        "lambda_x": args.lambda_x,
        "lambda_gamma": args.lambda_gamma,
        "lambda_audit": args.lambda_audit,
        "frac_a": args.frac_a,
        "frac_b": args.frac_b,
        "frac_c": args.frac_c,
        "d_slice_only": args.d_slice_only,
        "gate_mode": args.gate_mode,
        "gates": gates,
        "train": {"pairs": train.n_interactions,
                  "positives": int(train.labels.sum())},
        "val": {"pairs": val.n_interactions,
                "positives": int(val.labels.sum())},
        "test": {"pairs": test.n_interactions,
                 "positives": int(test.labels.sum())},
        "pretrained_checkpoint": args.pretrained_checkpoint,
        "cli": vars(args),
    }
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / "config.json", config_out)

    monitor = MonitorWriter(
        out, fail_on_error=not args.no_fail_on_monitor_error, reset_files=True)

    loop = A0NodeClassificationLoop(
        tgn=tgn,
        adapter=components["adapter"],
        prss_core=components["prss_core"],
        probes=components["probes"],
        device=device,
        batch_size=args.bs,
        n_neighbors=args.n_degree,
        trace_roots=args.trace_roots,
        trace_mode=args.trace_mode,
        rank_r=args.r,
        lambda_x=args.lambda_x,
        lambda_gamma=args.lambda_gamma,
        lambda_audit=args.lambda_audit,
        frac_a=args.frac_a,
        frac_b=args.frac_b,
        frac_c=args.frac_c,
        d_slice_only=args.d_slice_only,
        gates=gates,
        gate_mode=args.gate_mode,
        monitor=monitor,
        seed=args.seed,
        out_dir=out,
        lr=args.lr,
        n_epoch=args.n_epoch,
        patience=args.patience,
        drop_out=args.drop_out,
        selection_metric=args.selection_metric,
    )
    summary = loop.run(train, val, test)
    loop.finalize(summary)

    print("status:", summary["status"], flush=True)
    if summary.get("stop_reason"):
        print("STOP reason:", summary["stop_reason"], flush=True)
    if summary.get("a0_readout"):
        a0_t = summary["a0_readout"]["test"]
        b_t = summary["baseline_decoder"]["test"]
        print("A0 readout   test auc={:.4f} ap={:.4f} (best_epoch={})".format(
            a0_t.get("auc", float("nan")), a0_t.get("ap", float("nan")),
            summary["a0_readout"]["best_epoch"]), flush=True)
        print("baseline dec test auc={:.4f} ap={:.4f} (best_epoch={})".format(
            b_t.get("auc", float("nan")), b_t.get("ap", float("nan")),
            summary["baseline_decoder"]["best_epoch"]), flush=True)
        print("delta_auc={:+.4f} delta_ap={:+.4f}".format(
            summary.get("delta_auc", float("nan")),
            summary.get("delta_ap", float("nan"))), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
