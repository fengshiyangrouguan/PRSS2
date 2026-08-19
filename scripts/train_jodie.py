#!/usr/bin/env python3
"""JODIE node-classification training on the official TGN host, PRSS switch.

Replicates the v1 matched protocol on the v2 architecture: pretrained official
TGN (frozen by default) + MLP decoder, 10 supervised epochs, BCE over natural
labels, independent held-out early stopping, best.pt, then zero-memory
train+val replay before the held-out test with spectral-isolation audit.

Example:
    python -m scripts.train_jodie --variant vanilla --data wikipedia \
        --data-dir old/processed_tgn_data \
        --pretrained-checkpoint outputs/pretrained/tgn-wikipedia.pth \
        --output outputs/jodie/vanilla__seed000 --gpu 0

--variant vanilla runs the plain official host (no PRSS core/adapter/bridge);
every other variant adds the pluggable compressor on top of the same host.
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

from prss.config import InterfaceSpec, PRSSConfig
from prss.core import PRSSCore
from prss.data.jodie import JodieDataset
from prss.hosts.jodie_bridge import JodieNodeClassificationBridge
from prss.hosts.jodie_tgn import (JodieTGNAdapter, TAU_TEMPLATE,
                                  jodie_preagg_dim)
from prss.hosts.official_tgn import MLP, TGN, get_neighbor_finder
from prss.monitoring import MonitorWriter
from prss.training.checkpoint import CheckpointManager
from prss.training.jodie_loop import JodieNodeClassificationLoop


def parse_args():
    p = argparse.ArgumentParser("Official TGN node classification + PRSS switch")
    p.add_argument("--variant", default="spectral",
                   choices=["vanilla", "random", "pca", "direct", "spectral"])
    p.add_argument("-d", "--data", default="wikipedia")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--pretrained-checkpoint", required=True)
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
                   help="Fine-tune the same pretrained TGN host in both matched modes")
    p.add_argument("--selection-metric", choices=["auc", "ap"], default="auc")
    # PRSS.
    p.add_argument("--candidate-dim", type=int, default=256)
    p.add_argument("--candidate-hidden", type=int, default=128)
    p.add_argument("--context-dim", type=int, default=64)
    p.add_argument("--reader-hidden", type=int, default=128)
    p.add_argument("--lambda-resp", type=float, default=1.0)
    p.add_argument("--lambda-spec", type=float, default=0.1)
    p.add_argument("--gram-ema", type=float, default=0.05)
    p.add_argument("--spectral-warmup", type=int, default=200)
    p.add_argument("--spectral-interval", type=int, default=200)
    p.add_argument("--spectral-step-size", type=float, default=0.25)
    p.add_argument("--trace-roots", type=int, default=8)
    p.add_argument("--trace-mode", default="positive_first",
                   choices=["positive_first", "evenly_spaced", "off"],
                   help="B1 hook: how traced roots are picked per batch")
    # Diagnostics (v1 known issues).
    p.add_argument("--no-early-stop", action="store_true",
                   help="B4 hook: keep training past the val plateau")
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
    BEFORE the embedding module is swapped for the adapter."""
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
    # Memory is a runtime state produced by the pretraining stream, not a
    # parameter to carry across runs. The v1 checkpoint's memory cells are
    # sized for the un-padded node set (9228) while the vendored TGN pads one
    # zero node (9229, upstream convention), so those keys are skipped and the
    # memory starts zeroed (reset_memory) — exactly the held-out protocol.
    MEMORY_STATE_KEYS = ("memory.memory", "memory.last_update",
                         "memory_updater.memory.memory",
                         "memory_updater.memory.last_update")
    for key in MEMORY_STATE_KEYS:
        state.pop(key, None)
    # The vendored TGN carries its self-supervised link-prediction decoder
    # (affinity_score.*). Official self-supervised checkpoints include it,
    # v1 pretrained checkpoints predate it — either way the supervised run
    # builds its own decoder, so those keys are allowed to be missing.
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

    prss_core = adapter = bridge = None
    if args.variant != "vanilla":
        host_dim = int(tgn.embedding_dimension)
        time_dim = int(tgn.embedding_module.n_time_features)
        edge_dim = int(tgn.n_edge_features)
        taus = [TAU_TEMPLATE.format(l) for l in range(args.n_layer + 1)]
        interfaces = {
            TAU_TEMPLATE.format(0): InterfaceSpec(
                TAU_TEMPLATE.format(0), raw_dim=host_dim,
                candidate_dim=host_dim, host_dim=host_dim, response_dim=1),
        }
        for layer in range(1, args.n_layer + 1):
            interfaces[TAU_TEMPLATE.format(layer)] = InterfaceSpec(
                TAU_TEMPLATE.format(layer), raw_dim=host_dim,
                candidate_dim=args.candidate_dim, host_dim=host_dim,
                response_dim=1)
        config = PRSSConfig(
            interfaces=interfaces,
            context_dim=args.context_dim,
            root_metadata_dim=1,
            parent_local_dim=jodie_preagg_dim(host_dim, time_dim, edge_dim,
                                              args.n_degree),
            relation_count=2,
            relation_dim=16,
            outside_layers=2,
            reader_hidden_dim=args.reader_hidden,
            candidate_hidden_dim=args.candidate_hidden,
            lambda_resp=args.lambda_resp,
            lambda_spec=args.lambda_spec,
            gram_ema_rho=args.gram_ema,
            spectral_update_interval=args.spectral_interval,
            spectral_warmup_steps=args.spectral_warmup,
            spectral_step_size=args.spectral_step_size,
            variant=args.variant,
        )
        prss_core = PRSSCore(config, variant=args.variant).to(device)
        adapter = JodieTGNAdapter(tgn.embedding_module, prss_core,
                                  n_neighbors=args.n_degree)
        train_logt = np.log1p(train.timestamps.astype(np.float64))
        bridge = JodieNodeClassificationBridge(
            adapter, prss_core,
            log_time_mean=float(train_logt.mean()),
            log_time_std=float(train_logt.std() + 1e-8))
        # Swap AFTER pretrained-key validation and freezing.
        tgn.embedding_module = adapter

    # Exact upstream decoder type/dimension.
    decoder = MLP(dataset.node_features.shape[1], drop=args.drop_out).to(device)

    unrestricted_params = []
    main_params = list(decoder.parameters())
    if args.finetune_host:
        main_params.extend(p for p in tgn.parameters() if p.requires_grad)
    if prss_core is not None:
        main_params.extend(p for p in prss_core.parameters()
                           if p.requires_grad)
        unrestricted_params = list(prss_core.unrestricted.parameters())
        seen = {id(p) for p in unrestricted_params}
        main_params = [p for p in main_params if id(p) not in seen]
    seen = set()
    main_params = [p for p in main_params
                   if not (id(p) in seen or seen.add(id(p)))]
    optimizer = torch.optim.Adam(main_params, lr=args.lr)
    unrestricted_optimizer = (torch.optim.Adam(unrestricted_params, lr=args.lr)
                              if unrestricted_params else None)
    return dict(tgn=tgn, decoder=decoder, prss_core=prss_core,
                adapter=adapter, bridge=bridge, optimizer=optimizer,
                unrestricted_optimizer=unrestricted_optimizer), \
        (prss_core.config if prss_core is not None else None)


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
    components, prss_config = build_components(args, device, dataset)
    tgn = components["tgn"]
    loop = JodieNodeClassificationLoop(
        tgn=tgn, decoder=components["decoder"],
        adapter=components["adapter"], bridge=components["bridge"],
        prss_core=components["prss_core"], optimizer=components["optimizer"],
        unrestricted_optimizer=components["unrestricted_optimizer"],
        device=device, batch_size=args.bs, n_neighbors=args.n_degree,
        grad_clip=args.grad_clip, lambda_resp=args.lambda_resp,
        lambda_spec=args.lambda_spec, trace_roots=args.trace_roots,
        trace_mode=args.trace_mode, spectral_warmup=args.spectral_warmup,
        spectral_interval=args.spectral_interval, monitor=monitor,
        seed=args.seed, finetune_host=args.finetune_host,
        selection_metric=args.selection_metric)

    save_json(out / "config.json", {
        "variant": args.variant,
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
        "protocol": "upstream TGN node classification + matched corrections "
                    "(held-out val early stop, zero-memory replay test); "
                    "vanilla|compressor switch",
        "prss": prss_config.as_dict() if prss_config is not None else None,
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
            model_components={"tgn": tgn, "decoder": components["decoder"],
                              **({"prss_core": components["prss_core"]}
                                 if components["prss_core"] is not None else {})},
            optimizer=components["optimizer"],
            unrestricted_optimizer=components["unrestricted_optimizer"],
            memory_msg_stores=None, device=device)
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

        before_counts, before_r = loop.audit_before()
        val_row = loop.evaluate_split(val, reset=False)
        loop.audit_after(before_counts, before_r, "validation")
        loop.reenable_spectral()

        score = val_row[args.selection_metric]
        row = {
            "epoch": epoch, "global_step": global_step, "variant": args.variant,
            "train": {k: v for k, v in train_row.items() if k != "global_step"},
            "val": val_row,
            "spectral": (components["prss_core"].snapshots()
                         if components["prss_core"] is not None else {}),
            "validation_spectral_isolation": "passed",
            "epoch_seconds": time.time() - t0,
        }
        with metrics_path.open("a") as f:
            f.write(json.dumps(row, allow_nan=True) + "\n")
        monitor.write_epoch(row)
        print(f"epoch={epoch} variant={args.variant} "
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
                          "tgn": tgn.state_dict(),
                          **({"prss_core": components["prss_core"].state_dict()}
                             if components["prss_core"] is not None else {})},
                "epoch": epoch, "score": float(best_score),
                "variant": args.variant,
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
                "tgn": tgn, "decoder": components["decoder"],
                **({"prss_core": components["prss_core"]}
                   if components["prss_core"] is not None else {})},
                optimizer=components["optimizer"],
                unrestricted_optimizer=components["unrestricted_optimizer"],
                epoch=epoch + 1, next_batch=0, global_step=global_step,
                best_score=best_score, best_epoch=best_epoch,
                bad_rounds=bad_rounds, train_state={}, memory_msg_stores=None,
                extra_payload={"memory_backup":
                               tgn.memory.backup_memory() if tgn.use_memory else None})

    # ---------------------------- final test --------------------------------
    best = load_torch(out / "best.pt", device)
    for name in ("decoder", "tgn"):
        components[name].load_state_dict(best["model"][name])
    if components["prss_core"] is not None:
        components["prss_core"].load_state_dict(best["model"]["prss_core"])

    loop.reset_memory()
    loop.replay_split(train)
    loop.replay_split(val)
    before_counts, before_r = loop.audit_before()
    test_row = loop.evaluate_split(test, reset=False)
    loop.audit_after(before_counts, before_r, "test")
    loop.reenable_spectral()

    summary = {
        "variant": args.variant,
        "data": args.data,
        "seed": args.seed,
        "best_epoch": int(best_epoch),
        "selection_metric": args.selection_metric,
        "best_validation_score": float(best_score),
        "test": test_row,
        "spectral": (components["prss_core"].snapshots()
                     if components["prss_core"] is not None else {}),
        "pretrained_checkpoint": str(args.pretrained_checkpoint),
        "inference_isolation": "passed",
    }
    save_json(out / "summary.json", summary)
    monitor.finalize(summary)
    save_json(out / "_SUCCESS.json", {"status": "complete",
                                      "best_epoch": int(best_epoch),
                                      "test": test_row})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
