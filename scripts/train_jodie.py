#!/usr/bin/env python3
"""JODIE node-classification training on the official TGN host (stage 2 entry).

Replicates the v1 matched protocol: pretrained official TGN + MLP decoder,
BCE over natural labels, held-out early stopping, best.pt, then zero-memory
train+val replay before the held-out test.

Example:
    python -m scripts.train_jodie --data wikipedia \
        --data-dir old/processed_tgn_data \
        --pretrained-checkpoint outputs/pretrained/wikipedia/best.pt \
        --output outputs/jodie/wikipedia_seed000 --gpu 0

RPBE wiring (--rpbe switch, Ky Fan term, adapter) lands in a later step; this
entry currently runs the pure host only.
"""

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rpbe.data.jodie import JodieDataset
from rpbe.hosts.official_tgn import MLP, TGN, get_neighbor_finder
from rpbe.monitoring import MonitorWriter
from rpbe.training.checkpoint import CheckpointManager
from rpbe.training.jodie_loop import JodieNodeClassificationLoop


def parse_args():
    p = argparse.ArgumentParser("Official TGN node classification (stage 2)")
    p.add_argument("-d", "--data", default="wikipedia")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--pretrained-checkpoint", default="",
                   help="Stage-1 host weights; empty = train from scratch")
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu", type=int, default=0)
    # Upstream TGN names/defaults (hyphenated for this CLI).
    p.add_argument("--bs", type=int, default=100)
    p.add_argument("--n-degree", type=int, default=10)
    p.add_argument("--n-head", type=int, default=2)
    p.add_argument("--n-epoch", type=int, default=10)
    p.add_argument("--n-layer", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--drop-out", type=float, default=0.1)
    p.add_argument("--message-dim", type=int, default=100)
    p.add_argument("--memory-dim", type=int, default=172)
    p.add_argument("--finetune-host", action="store_true",
                   help="Jointly fine-tune the pretrained host (default: frozen)")
    p.add_argument("--selection-metric", choices=["auc", "ap"], default="auc")
    # Diagnostics.
    p.add_argument("--no-early-stop", action="store_true")
    # Monitoring / resume / smoke caps.
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--monitor-every", type=int, default=50)
    p.add_argument("--checkpoint-every", type=int, default=50,
                   help="Rolling exact checkpoint every N train batches; 0 disables")
    p.add_argument("--resume-from", default="")
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
    """Locate the TGN state dict inside the checkpoint formats we may load.

    Supports: a bare state dict; ``{"model_state_dict"|"tgn": {...}}`` (official
    self-supervised checkpoints); and the two-level nested ``{"model": {"tgn":
    ...}}`` layout written by this repository's ``best.pt``.
    """
    if isinstance(obj, dict):
        for key in ("model_state_dict", "tgn"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
        if "model" in obj and isinstance(obj["model"], dict):
            inner = obj["model"]
            for key in ("model_state_dict", "tgn"):
                if key in inner and isinstance(inner[key], dict):
                    return inner[key]
    return obj


def cap_data(data, cap):
    if not cap or len(data.sources) <= cap:
        return data
    return data.slice(0, cap)


def build_components(args, device, dataset):
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
    if args.pretrained_checkpoint:
        state = unwrap_state(load_torch(args.pretrained_checkpoint, device))
        # Memory is a runtime state produced by the pretraining stream, not a
        # parameter to carry across runs (shape differs across node sets).
        MEMORY_STATE_KEYS = ("memory.memory", "memory.last_update",
                             "memory_updater.memory.memory",
                             "memory_updater.memory.last_update")
        for key in MEMORY_STATE_KEYS:
            state.pop(key, None)
        # The vendored TGN carries its self-supervised link-prediction decoder
        # (affinity_score.*); stage-2 runs build their own decoder.
        DECODER_STATE_KEYS = ("affinity_score.fc1.weight", "affinity_score.fc1.bias",
                              "affinity_score.fc2.weight", "affinity_score.fc2.bias")
        allowed_missing = set(MEMORY_STATE_KEYS) | set(DECODER_STATE_KEYS)
        missing, unexpected = tgn.load_state_dict(state, strict=False)
        if not (set(missing) <= allowed_missing) or unexpected:
            raise RuntimeError(
                "pretrained TGN mismatch: missing={}, unexpected={}".format(
                    sorted(missing), sorted(unexpected)))
    if not args.finetune_host:
        for p in tgn.parameters():
            p.requires_grad_(False)
        tgn.eval()

    # Exact upstream decoder type/dimension.
    decoder = MLP(dataset.node_features.shape[1], drop=args.drop_out).to(device)

    main_params = list(decoder.parameters())
    if args.finetune_host:
        main_params.extend(p for p in tgn.parameters() if p.requires_grad)
    seen = set()
    main_params = [p for p in main_params
                   if not (id(p) in seen or seen.add(id(p)))]
    optimizer = torch.optim.Adam(main_params, lr=args.lr)
    return dict(tgn=tgn, decoder=decoder, optimizer=optimizer)


def main():
    args = parse_args()
    seed_all(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    metrics_path = out / "metrics.jsonl"
    if not args.resume_from and metrics_path.exists():
        metrics_path.unlink()
    monitor = MonitorWriter(out, fail_on_error=not args.no_fail_on_monitor_error,
                            reset_files=not bool(args.resume_from))

    dataset = JodieDataset(args.data, data_dir=args.data_dir,
                           use_validation=True)
    full, train, val, test = dataset.splits()
    train = cap_data(train, args.max_train)
    val = cap_data(val, args.max_val)
    test = cap_data(test, args.max_test)
    components = build_components(args, device, dataset)
    tgn = components["tgn"]
    loop = JodieNodeClassificationLoop(
        tgn=tgn, decoder=components["decoder"],
        optimizer=components["optimizer"],
        device=device, batch_size=args.bs, n_neighbors=args.n_degree,
        grad_clip=args.grad_clip, monitor=monitor,
        seed=args.seed, finetune_host=args.finetune_host,
        selection_metric=args.selection_metric)

    save_json(out / "config.json", {
        "data": args.data,
        "data_dir": str(Path(args.data_dir).resolve()),
        "seed": args.seed,
        "device": str(device),
        "host_dim": int(tgn.embedding_dimension),
        "n_time_features": int(tgn.embedding_module.n_time_features),
        "n_edge_features": int(tgn.n_edge_features),
        "train_pairs": len(train.sources),
        "val_pairs": len(val.sources),
        "test_pairs": len(test.sources),
        "train_positives": int(train.labels.sum()),
        "val_positives": int(val.labels.sum()),
        "test_positives": int(test.labels.sum()),
        "pretrained_checkpoint": str(args.pretrained_checkpoint),
        "finetune_host": bool(args.finetune_host),
        "protocol": "upstream TGN node classification + matched corrections "
                    "(held-out val early stop, zero-memory replay test)",
        "rpbe": False,
        "cli": vars(args),
    })

    # ------------------------- resume / train state -------------------------
    ckpt = CheckpointManager(out / "rolling_step.pt")
    resume = {}
    global_step = 0
    start_epoch = 0
    best_score = -float("inf")
    best_epoch = -1
    bad_rounds = 0
    if args.resume_from:
        resume = ckpt.load(
            model_components={"tgn": tgn, "decoder": components["decoder"]},
            optimizer=components["optimizer"], device=device)
        start_epoch = int(resume["epoch"])
        global_step = int(resume["global_step"])
        best_score = float(resume.get("best_score", best_score))
        best_epoch = int(resume.get("best_epoch", best_epoch))
        bad_rounds = int(resume.get("bad_rounds", bad_rounds))
        extra = resume.get("extra") or {}
        if tgn.use_memory and extra.get("memory_backup") is not None:
            tgn.memory.restore_memory(extra["memory_backup"])
        print(f"RESUME epoch={start_epoch} global_step={global_step} "
              f"from={args.resume_from}", flush=True)

    # ------------------------------ train loop ------------------------------
    patience = args.patience if not args.no_early_stop else 10 ** 9
    for epoch in range(start_epoch, args.n_epoch):
        t0 = time.time()
        train_row = loop.train_epoch(epoch, global_step, train)
        global_step = train_row["global_step"]

        val_row = loop.evaluate_split(val, reset=False)

        score = val_row[args.selection_metric]
        row = {
            "epoch": epoch, "global_step": global_step,
            "train": {k: v for k, v in train_row.items() if k != "global_step"},
            "val": val_row,
            "epoch_seconds": time.time() - t0,
        }
        with metrics_path.open("a") as f:
            f.write(json.dumps(row, allow_nan=True) + "\n")
        monitor.write_epoch(row)
        print(f"epoch={epoch} "
              f"train_auc={train_row['train']['auc']:.5f} "
              f"train_ap={train_row['train']['ap']:.5f} "
              f"val_auc={val_row['auc']:.5f} val_ap={val_row['ap']:.5f} "
              f"val_nll={val_row['nll']:.5f} val_pos={val_row['positives']} "
              f"sec={row['epoch_seconds']:.1f}", flush=True)

        score_is_finite = bool(np.isfinite(score))
        improved = (best_epoch < 0) or (
            score_is_finite and score > best_score + 1e-12)
        if improved:
            best_score = float(score) if score_is_finite else -float("inf")
            best_epoch = epoch
            bad_rounds = 0
            torch.save({
                "model": {"decoder": components["decoder"].state_dict(),
                          "tgn": tgn.state_dict()},
                "epoch": epoch, "score": float(best_score),
            }, out / "best.pt")
        else:
            bad_rounds += 1
            if bad_rounds >= patience:
                print(f"early_stop epoch={epoch} best_epoch={best_epoch} "
                      f"best_{args.selection_metric}={best_score:.6f}",
                      flush=True)
                break
        if args.checkpoint_every > 0 and (epoch + 1) % args.checkpoint_every == 0:
            ckpt.save(model_components={
                "tgn": tgn, "decoder": components["decoder"]},
                optimizer=components["optimizer"],
                epoch=epoch + 1, next_batch=0, global_step=global_step,
                best_score=best_score, best_epoch=best_epoch,
                bad_rounds=bad_rounds, train_state={},
                extra_payload={"memory_backup":
                               tgn.memory.backup_memory() if tgn.use_memory else None})

    # ---------------------------- final test --------------------------------
    best = load_torch(out / "best.pt", device)
    for name in ("decoder", "tgn"):
        components[name].load_state_dict(best["model"][name])

    loop.reset_memory()
    loop.replay_split(train)
    loop.replay_split(val)
    test_row = loop.evaluate_split(test, reset=False)

    summary = {
        "data": args.data,
        "seed": args.seed,
        "best_epoch": int(best_epoch),
        "selection_metric": args.selection_metric,
        "best_validation_score": float(best_score),
        "test": test_row,
        "pretrained_checkpoint": str(args.pretrained_checkpoint),
    }
    save_json(out / "summary.json", summary)
    monitor.finalize(summary)
    save_json(out / "_SUCCESS.json", {"status": "complete",
                                      "best_epoch": int(best_epoch),
                                      "test": test_row})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
