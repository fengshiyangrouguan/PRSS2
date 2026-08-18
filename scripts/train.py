#!/usr/bin/env python3
"""PRSS2 training entry: TGB link prediction with a pluggable compressor.

Example:
    python -m scripts.train --variant spectral --dataset tgbl-wiki --seed 0 \
        --output outputs/tgbl-wiki/tgbl-wiki__spectral__seed000 --gpu 0

Precedence: explicit CLI > YAML experiment file > code defaults.
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from prss.config import InterfaceSpec, PRSSConfig
from prss.core import PRSSCore
from prss.data.tgb_link import TGBLinkDataset
from prss.hosts.tgn_pyg import TAU, PyGTGNAdapter, pyg_preagg_dim
from prss.hosts.tgn_pyg_bridge import TGBLinkOutsideBridge
from prss.monitoring import MonitorWriter
from prss.training.checkpoint import CheckpointManager
from prss.training.event_loop import TGBLinkPredictionLoop


def parse_args():
    p = argparse.ArgumentParser("PRSS2 TGB link-prediction training")
    p.add_argument("--config", default="", help="YAML experiment file for defaults")
    p.add_argument("--variant", default="spectral",
                   choices=["vanilla", "random", "pca", "direct", "spectral"])
    p.add_argument("--dataset", default="tgbl-wiki")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--output", required=True)
    # Host / training
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--bs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--n-neighbors", type=int, default=10)
    p.add_argument("--mem-dim", type=int, default=100)
    p.add_argument("--time-dim", type=int, default=100)
    p.add_argument("--emb-dim", type=int, default=100)
    # PRSS
    p.add_argument("--candidate-dim", type=int, default=256)
    p.add_argument("--candidate-hidden", type=int, default=128)
    p.add_argument("--context-dim", type=int, default=64)
    p.add_argument("--reader-hidden", type=int, default=128)
    p.add_argument("--lambda-resp", type=float, default=1.0)
    p.add_argument("--lambda-spec", type=float, default=0.1)
    p.add_argument("--gram-ema", type=float, default=0.05)
    p.add_argument("--spectral-warmup", type=int, default=100)
    p.add_argument("--spectral-interval", type=int, default=100)
    p.add_argument("--spectral-step-size", type=float, default=0.25)
    p.add_argument("--trace-roots", type=int, default=8)
    # Monitoring / resuming / smoke caps
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--monitor-every", type=int, default=100)
    p.add_argument("--checkpoint-every", type=int, default=0)
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


def build_components(args, device, dataset, time_stats):
    from prss.hosts.pyg_models.decoder import LinkPredictor
    from prss.hosts.pyg_models.emb_module import GraphAttentionEmbedding
    from prss.hosts.pyg_models.memory_module import TGNMemory
    from prss.hosts.pyg_models.msg_agg import LastAggregator
    from prss.hosts.pyg_models.msg_func import IdentityMessage
    from prss.hosts.pyg_models.neighbor_loader import LastNeighborLoader

    neighbor_loader = LastNeighborLoader(dataset.num_nodes, size=args.n_neighbors,
                                         device=device)
    memory = TGNMemory(dataset.num_nodes, dataset.msg_dim, args.mem_dim, args.time_dim,
                       message_module=IdentityMessage(dataset.msg_dim, args.mem_dim,
                                                      args.time_dim),
                       aggregator_module=LastAggregator()).to(device)
    gnn = GraphAttentionEmbedding(in_channels=args.mem_dim, out_channels=args.emb_dim,
                                  msg_dim=dataset.msg_dim,
                                  time_enc=memory.time_enc).to(device)
    link_pred = LinkPredictor(in_channels=args.emb_dim).to(device)
    criterion = torch.nn.BCEWithLogitsLoss()

    prss_core = adapter = bridge = None
    if args.variant != "vanilla":
        config = PRSSConfig(
            interfaces={TAU: InterfaceSpec(TAU, raw_dim=args.emb_dim,
                                           candidate_dim=args.candidate_dim,
                                           host_dim=args.emb_dim)},
            context_dim=args.context_dim,
            root_metadata_dim=args.emb_dim + 2,
            parent_local_dim=pyg_preagg_dim(args.mem_dim, args.time_dim,
                                            dataset.msg_dim, args.n_neighbors),
            relation_count=4,
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
        adapter = PyGTGNAdapter(
            memory, gnn, prss_core, n_neighbors=args.n_neighbors,
            mem_dim=args.mem_dim, time_dim=args.time_dim, msg_dim=dataset.msg_dim,
            emb_dim=args.emb_dim, time_mean=time_stats[0], time_std=time_stats[1])
        bridge = TGBLinkOutsideBridge(adapter, prss_core,
                                      time_mean=time_stats[0], time_std=time_stats[1])

    # Parameter groups: main path (host + core); unrestricted reader stays isolated.
    main_params = list(memory.parameters()) + list(gnn.parameters()) + \
        list(link_pred.parameters())
    unrestricted_params = []
    if prss_core is not None:
        main_params += list(prss_core.parameters())
        unrestricted_params = list(prss_core.unrestricted.parameters())
        seen = {id(p) for p in unrestricted_params}
        main_params = [p for p in main_params if id(p) not in seen]
    seen = set()
    main_params = [p for p in main_params if not (id(p) in seen or seen.add(id(p)))]
    optimizer = torch.optim.Adam(main_params, lr=args.lr)
    unrestricted_optimizer = (torch.optim.Adam(unrestricted_params, lr=args.lr)
                              if unrestricted_params else None)
    return dict(memory=memory, gnn=gnn, link_pred=link_pred, neighbor_loader=neighbor_loader,
                criterion=criterion, prss_core=prss_core, adapter=adapter, bridge=bridge,
                optimizer=optimizer, unrestricted_optimizer=unrestricted_optimizer), \
        (prss_core.config if prss_core is not None else None)


def main():
    args = parse_args()
    defaults = {}
    if args.config:
        import yaml
        with open(args.config) as f:
            spec = yaml.safe_load(f)
        defaults = dict(spec.get("defaults", {}))
    for key in defaults:
        if getattr(args, key.replace("-", "_"), None) is None:
            setattr(args, key.replace("-", "_"), defaults[key])
    seed_all(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    monitor = MonitorWriter(out, fail_on_error=not args.no_fail_on_monitor_error,
                            reset_files=not bool(args.resume_from))

    dataset = TGBLinkDataset(name=args.dataset, device=device)
    if args.max_train:
        dataset._train_data = dataset._train_data[:args.max_train]
    if args.max_val:
        dataset._val_data = dataset._val_data[:args.max_val]
    if args.max_test:
        dataset._test_data = dataset._test_data[:args.max_test]
    time_stats = dataset.time_stats()
    components, prss_config = build_components(args, device, dataset, time_stats)

    loop = TGBLinkPredictionLoop(
        dataset=dataset, memory=components["memory"], gnn=components["gnn"],
        link_pred=components["link_pred"],
        neighbor_loader=components["neighbor_loader"],
        adapter=components["adapter"], bridge=components["bridge"],
        prss_core=components["prss_core"], optimizer=components["optimizer"],
        unrestricted_optimizer=components["unrestricted_optimizer"],
        criterion=components["criterion"], device=device, batch_size=args.bs,
        n_neighbors=args.n_neighbors, grad_clip=args.grad_clip,
        lambda_resp=args.lambda_resp, lambda_spec=args.lambda_spec,
        trace_roots=args.trace_roots, spectral_warmup=args.spectral_warmup,
        spectral_interval=args.spectral_interval, monitor=monitor, seed=args.seed)

    save_json(out / "config.json", {
        "variant": args.variant,
        "dataset": args.dataset,
        "seed": args.seed,
        "device": str(device),
        "host_dim": args.emb_dim,
        "candidate_dim": args.candidate_dim if prss_config else None,
        "sanity": dataset.sanity_check(),
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
            model_components={"memory": components["memory"], "gnn": components["gnn"],
                              "link_pred": components["link_pred"],
                              **({"prss_core": components["prss_core"]}
                                 if components["prss_core"] is not None else {})},
            optimizer=components["optimizer"],
            unrestricted_optimizer=components["unrestricted_optimizer"],
            memory_msg_stores={"s": components["memory"].msg_s_store,
                               "d": components["memory"].msg_d_store},
            device=device)
        start_epoch = int(resume["epoch"])
        global_step = int(resume["global_step"])
        best_score = float(resume.get("best_score", best_score))
        best_epoch = int(resume.get("best_epoch", best_epoch))
        bad_rounds = int(resume.get("bad_rounds", bad_rounds))

    # ------------------------------ train loop ------------------------------
    metrics_path = out / "metrics.jsonl"
    if not args.resume_from and metrics_path.exists():
        metrics_path.unlink()
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_row = loop.train_epoch(epoch, global_step)
        global_step = train_row["global_step"]

        before_counts, before_r = loop.audit_before()
        val_row = loop.evaluate_split("val")
        loop.audit_after(before_counts, before_r, "validation")
        loop.reenable_spectral()

        score = val_row[f"val_{dataset.eval_metric}"]
        row = {
            "epoch": epoch, "global_step": global_step, "variant": args.variant,
            "train": {k: v for k, v in train_row.items() if k != "global_step"},
            "val": val_row,
            "spectral": (components["prss_core"].snapshots()
                         if components["prss_core"] is not None else {}),
            "epoch_seconds": time.time() - t0,
        }
        with metrics_path.open("a") as f:
            f.write(json.dumps(row, allow_nan=True) + "\n")
        monitor.write_epoch(row)
        print(f"epoch={epoch} variant={args.variant} "
              f"train_loss={train_row['train_task_loss']:.5f} "
              f"val_{dataset.eval_metric}={score:.5f} "
              f"sec={row['epoch_seconds']:.1f}", flush=True)

        if score > best_score + 1e-9:
            best_score = float(score)
            best_epoch = epoch
            bad_rounds = 0
            torch.save({
                "model": {"memory": components["memory"].state_dict(),
                          "gnn": components["gnn"].state_dict(),
                          "link_pred": components["link_pred"].state_dict(),
                          **({"prss_core": components["prss_core"].state_dict()}
                             if components["prss_core"] is not None else {})},
                "epoch": epoch, "score": score, "variant": args.variant,
            }, out / "best.pt")
        else:
            bad_rounds += 1
            if bad_rounds >= args.patience:
                print(f"early_stop epoch={epoch} best_epoch={best_epoch} "
                      f"best={best_score:.6f}", flush=True)
                break
        if args.checkpoint_every > 0 and (epoch + 1) % args.checkpoint_every == 0:
            ckpt.save(model_components={
                "memory": components["memory"], "gnn": components["gnn"],
                "link_pred": components["link_pred"],
                **({"prss_core": components["prss_core"]}
                   if components["prss_core"] is not None else {})},
                optimizer=components["optimizer"],
                unrestricted_optimizer=components["unrestricted_optimizer"],
                epoch=epoch + 1, next_batch=0, global_step=global_step,
                best_score=best_score, best_epoch=best_epoch, bad_rounds=bad_rounds,
                train_state={}, memory_msg_stores={
                    "s": components["memory"].msg_s_store,
                    "d": components["memory"].msg_d_store})

    # ---------------------------- final test --------------------------------
    best = torch.load(out / "best.pt", map_location=device, weights_only=False)
    for name in ("memory", "gnn", "link_pred"):
        components[name].load_state_dict(best["model"][name])
    if components["prss_core"] is not None:
        components["prss_core"].load_state_dict(best["model"]["prss_core"])

    loop.reset_stream()
    loop.replay_split("train")
    loop.replay_split("val")
    before_counts, before_r = loop.audit_before()
    test_row = loop.evaluate_split("test")
    loop.audit_after(before_counts, before_r, "test")
    loop.reenable_spectral()

    summary = {
        "variant": args.variant,
        "dataset": args.dataset,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_validation_score": best_score,
        "test": test_row,
        "spectral": (components["prss_core"].snapshots()
                     if components["prss_core"] is not None else {}),
        "inference_isolation": "passed",
    }
    save_json(out / "summary.json", summary)
    monitor.finalize(summary)
    save_json(out / "_SUCCESS.json", {"status": "complete", "best_epoch": best_epoch,
                                      "test": test_row})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
