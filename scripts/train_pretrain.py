#!/usr/bin/env python3
"""Stage-1 self-supervised pretraining entry (official TGN protocol).

Clones ``old/tgn/train_self_supervised.py``: BCE link prediction with negative
destinations, val-AP early stopping, memory reset per epoch, detach after
every backward.  With ``--stage1-rpbe`` the RPBE component (compressor + fixed
maps + LINK cut builder) trains jointly with the host from step 0.

Example:
    python -m scripts.train_pretrain -d wikipedia \
        --data-dir old/processed_tgn_data --output outputs/pretrained/wikipedia
"""

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse
import json
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

from rpbe.compressor import RecursiveCompressor
from rpbe.config import RPBConfig
from rpbe.data.jodie import JodieDataset
from rpbe.hosts.jodie_bridge import build_cut_builder
from rpbe.hosts.jodie_tgn import JodieTGNAdapter, TAU_TEMPLATE
from rpbe.hosts.official_tgn import TGN, get_neighbor_finder
from rpbe.maps import FixedMaps
from rpbe.monitoring import MonitorWriter
from rpbe.records import LINK, build_edge_tables
from rpbe.training.checkpoint import CheckpointManager
from rpbe.training.pretrain_loop import TGNPretrainLoop


def parse_args():
    p = argparse.ArgumentParser("Official TGN self-supervised pretraining (stage 1)")
    p.add_argument("-d", "--data", default="wikipedia")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--bs", type=int, default=200)
    p.add_argument("--n-degree", type=int, default=10)
    p.add_argument("--n-head", type=int, default=2)
    p.add_argument("--n-epoch", type=int, default=50)
    p.add_argument("--n-layer", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=10,
                   help="val 曲线抖动大，早停阈值放宽")
    p.add_argument("--drop-out", type=float, default=0.1)
    p.add_argument("--message-dim", type=int, default=100)
    p.add_argument("--memory-dim", type=int, default=172)
    p.add_argument("--backprop-every", type=int, default=1)
    p.add_argument("--grad-clip", type=float, default=0.0,
                   help="0 disables (official protocol has no clipping)")
    # RPBE (stage-1 joint pretraining).
    p.add_argument("--stage1-rpbe", action="store_true",
                   help="Jointly train host + Gamma_theta with the Ky Fan term")
    p.add_argument("--kf-lambda", type=float, default=1.0)
    p.add_argument("--rpbe-width", type=int, default=128)
    p.add_argument("--sketch-dim", type=int, default=64)
    p.add_argument("--kf-cuts-per-tau", type=int, default=1024,
                   help="per-tau per-batch sampling cap (sampling-probability "
                        "corrected)")
    p.add_argument("--kf-min-ratio", type=float, default=2.0)
    p.add_argument("--kf-min-abs", type=int, default=1024)
    p.add_argument("--ridge-eps", type=float, default=1e-4)
    p.add_argument("--rpbe-seed", type=int, default=0)
    p.add_argument("--trace-roots", type=int, default=32)
    # Monitoring / resume / caps.
    p.add_argument("--monitor-every", type=int, default=50)
    p.add_argument("--no-fail-on-monitor-error", action="store_true")
    p.add_argument("--checkpoint-every", type=int, default=1,
                   help="Exact rolling checkpoint every N epochs; 0 disables")
    p.add_argument("--resume-from", default="")
    p.add_argument("--max-train", type=int, default=0)
    p.add_argument("--max-val", type=int, default=0)
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


def cap_data(data, cap):
    if not cap or len(data.sources) <= cap:
        return data
    return data.slice(0, cap)


def make_tb_writer(out):
    """TensorBoard writer under <out>/tb; degrades to None without the dep."""
    try:
        from torch.utils.tensorboard import SummaryWriter
        return SummaryWriter(log_dir=str(Path(out) / "tb"))
    except ImportError:
        print("WARN: tensorboard not installed; skipping TB output", flush=True)
        return None


def main():
    args = parse_args()
    seed_all(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    monitor = MonitorWriter(out, fail_on_error=not args.no_fail_on_monitor_error,
                            reset_files=True)
    tb_writer = make_tb_writer(out)

    dataset = JodieDataset(args.data, data_dir=args.data_dir,
                           use_validation=True)
    full, train, val, test = dataset.splits()
    train = cap_data(train, args.max_train)
    val = cap_data(val, args.max_val)
    train_finder = get_neighbor_finder(train, uniform=False,
                                       max_node_idx=max(full.unique_nodes))
    full_finder = get_neighbor_finder(full, uniform=False,
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

    compressor = adapter = fixed_maps = cut_builder = rpbe_cfg = None
    if args.stage1_rpbe:
        host_dim = int(tgn.embedding_dimension)
        taus = [TAU_TEMPLATE.format(l) for l in range(args.n_layer + 1)]
        delta_scale = float(np.median(np.diff(np.sort(full.timestamps)))) or 1.0
        # The root interface (highest layer) has no upward walk and is
        # excluded from the KF windows EXPLICITLY (kf_taus whitelist).
        rpbe_cfg = RPBConfig(
            state_dims={tau: host_dim for tau in taus},
            own_dims={tau: host_dim for tau in taus},
            width_D=args.rpbe_width, m=args.sketch_dim,
            lambda_kf=args.kf_lambda, ridge_eps=args.ridge_eps,
            delta_t_scale=delta_scale,
            cuts_per_tau=args.kf_cuts_per_tau, kf_min_ratio=args.kf_min_ratio,
            kf_min_abs=args.kf_min_abs,
            kf_taus=list(taus[:-1]),
            rpbe_seed=args.rpbe_seed)
        compressor = RecursiveCompressor(rpbe_cfg).to(device)
        # Explicit edge tables: consumption-record stamping (adapter) and
        # probe Y lookup (cut builder) share ONE source of truth.
        (endpoints, labels_tbl, user_nodes, page_nodes,
         edge_table_stats) = build_edge_tables(dataset)
        edge_tables = (endpoints, labels_tbl, user_nodes, page_nodes)
        adapter = JodieTGNAdapter(tgn.embedding_module, compressor,
                                  n_neighbors=args.n_degree,
                                  edge_tables=edge_tables)
        tgn.embedding_module = adapter
        fixed_maps = FixedMaps(rpbe_cfg).to(device)
        cut_builder = build_cut_builder(dataset, stage=LINK, cfg=rpbe_cfg,
                                        seed=args.rpbe_seed,
                                        delta_t_scale=delta_scale,
                                        tables=edge_tables)

    main_params = [p for p in tgn.parameters() if p.requires_grad]
    if compressor is not None:
        main_params.extend(p for p in compressor.parameters()
                           if p.requires_grad)
    seen = set()
    main_params = [p for p in main_params
                   if not (id(p) in seen or seen.add(id(p)))]
    optimizer = torch.optim.Adam(main_params, lr=args.lr)
    loop = TGNPretrainLoop(
        tgn=tgn, optimizer=optimizer, device=device, batch_size=args.bs,
        n_neighbors=args.n_degree, backprop_every=args.backprop_every,
        grad_clip=args.grad_clip, monitor=monitor, seed=args.seed,
        adapter=adapter, cut_builder=cut_builder, fixed_maps=fixed_maps,
        rpbe_cfg=rpbe_cfg,
        trace_roots=args.trace_roots)

    save_json(out / "config.json", {
        "data": args.data,
        "data_dir": str(Path(args.data_dir).resolve()),
        "seed": args.seed,
        "device": str(device),
        "host_dim": int(tgn.embedding_dimension),
        "train_pairs": len(train.sources),
        "val_pairs": len(val.sources),
        "stage1_rpbe": bool(args.stage1_rpbe),
        "rpbe": rpbe_cfg.as_dict() if rpbe_cfg is not None else None,
        "cli": vars(args),
    })

    global_step = 0
    best_ap = -1.0
    best_epoch = -1
    bad_rounds = 0
    best_kf = None
    start_epoch = 0
    ckpt = CheckpointManager(out / "rolling_step.pt")
    if args.resume_from:
        payload = ckpt.load(
            model_components={"tgn": tgn,
                              **({"compressor": compressor}
                                 if compressor is not None else {})},
            optimizer=optimizer, device=device)
        start_epoch = int(payload["epoch"])
        global_step = int(payload["global_step"])
        best_ap = float(payload.get("best_score", -1.0))
        best_epoch = int(payload.get("best_epoch", -1))
        bad_rounds = int(payload.get("bad_rounds", 0))
        print(f"RESUME epoch={start_epoch} global_step={global_step} "
              f"from={args.resume_from}", flush=True)

    for epoch in range(start_epoch, args.n_epoch):
        t0 = time.time()
        row = loop.train_epoch(epoch, global_step, train)
        global_step = row["global_step"]

        # Official semantics: validation runs on the full graph's neighbor
        # finder; memory resets at the next epoch start so no backup needed.
        tgn.embedding_module.neighbor_finder = full_finder
        val_row = loop.evaluate_edge_prediction(val, neg_seed=0)
        tgn.embedding_module.neighbor_finder = train_finder

        epoch_row = {"epoch": epoch, "global_step": global_step,
                     "train": row, "val": val_row,
                     "epoch_seconds": time.time() - t0}
        if fixed_maps is not None:
            monitor.save_fingerprint(epoch, fixed_maps.isolation_fingerprint())
        monitor.write_epoch(epoch_row)
        print(f"epoch={epoch} train_link={row['train_link_loss']:.4f} "
              f"train_kf={row['train_kf_loss']:.6f} "
              f"val_ap={val_row['val_ap']:.4f} val_auc={val_row['val_auc']:.4f} "
              f"sec={epoch_row['epoch_seconds']:.1f}", flush=True)
        if tb_writer is not None:
            tb_writer.add_scalar("epoch/val_ap", val_row["val_ap"], epoch)
            tb_writer.add_scalar("epoch/val_auc", val_row["val_auc"], epoch)
            tb_writer.add_scalar("epoch/train_link_loss",
                                 row["train_link_loss"], epoch)
            if row.get("kf"):
                tb_writer.add_scalar("epoch/kf_loss",
                                     row["kf"]["kf_loss"], epoch)
                for tau, frac in row["kf"]["J_frac"].items():
                    tb_writer.add_scalar("epoch/J_frac/" + tau, frac, epoch)

        if val_row["val_ap"] > best_ap + 1e-12:
            best_ap = float(val_row["val_ap"])
            best_epoch = epoch
            best_kf = row.get("kf")
            bad_rounds = 0
            # Save the host in its OFFICIAL namespace: temporarily swap the
            # adapter out so tgn.state_dict() has upstream keys
            # (embedding_module.attention_models.*), not the wrapped
            # adapter namespace (embedding_module.host.*).
            if adapter is not None:
                tgn.embedding_module = adapter.host
            host_sd = tgn.state_dict()
            if adapter is not None:
                tgn.embedding_module = adapter
            payload = {"model": {"tgn": host_sd},
                       "epoch": epoch, "val_ap": best_ap}
            if compressor is not None:
                payload["model"]["compressor"] = compressor.state_dict()
            torch.save(payload, out / "best.pt")
        else:
            bad_rounds += 1
            if bad_rounds >= args.patience:
                print(f"early_stop epoch={epoch} best_epoch={best_epoch} "
                      f"best_ap={best_ap:.6f}", flush=True)
                break
        if args.checkpoint_every > 0 and (epoch + 1) % args.checkpoint_every == 0:
            ckpt.save(model_components={
                "tgn": tgn,
                **({"compressor": compressor}
                   if compressor is not None else {})},
                optimizer=optimizer,
                epoch=epoch + 1, next_batch=0, global_step=global_step,
                best_score=best_ap, best_epoch=best_epoch,
                bad_rounds=bad_rounds, train_state={},
                extra_payload={"memory_backup":
                               tgn.memory.backup_memory()
                               if tgn.use_memory else None})

    summary = {
        "data": args.data, "seed": args.seed, "best_epoch": int(best_epoch),
        "best_val_ap": float(best_ap), "stage1_rpbe": bool(args.stage1_rpbe),
        "kf": best_kf if best_kf is not None else None,
    }
    save_json(out / "summary.json", summary)
    if tb_writer is not None:
        tb_writer.close()
    monitor.finalize(summary)
    save_json(out / "_SUCCESS.json", {"status": "complete",
                                      "best_epoch": int(best_epoch)})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
