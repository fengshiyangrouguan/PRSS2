#!/usr/bin/env python3
"""PRSS2 inference entry: held-out one-vs-many MRR from a saved training run.

Example:
    python -m scripts.inference --checkpoint outputs/tgbl-wiki/tgbl-wiki__spectral__seed000/best.pt \
        --split test --output outputs/tgbl-wiki/tgbl-wiki__spectral__seed000/inference_test.json

Rebuilds the model of the recorded variant/seed, loads the checkpoint, replays
train+val from zero memory, then evaluates the requested split.  All training-only
components (outside encoder, readers, Gram/SVD) are hard-disabled and audited.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SRC = ROOT / "src"
for path in (str(SRC), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from prss.data.tgb_link import TGBLinkDataset
from prss.training.isolation import assert_clean, counts_of_spectral, r_copies


def parse_args():
    p = argparse.ArgumentParser("PRSS2 TGB link-prediction inference")
    p.add_argument("--checkpoint", required=True, help="path to best.pt from train.py")
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--output", required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--bs", type=int, default=200)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config_path = Path(args.checkpoint).resolve().parent / "config.json"
    run_config = json.loads(config_path.read_text()) if config_path.exists() else {}
    cli = run_config.get("cli", {})
    variant = run_config.get("variant", ckpt.get("variant", "spectral"))
    emb_dim = int(cli.get("emb_dim", 100))

    dataset = TGBLinkDataset(name=run_config.get("dataset", "tgbl-wiki"), device=device)
    time_stats = dataset.time_stats()

    # Rebuild the model exactly as train.py did (host + PRSS when applicable).
    from scripts.train import build_components  # noqa: E402

    class _Args:
        pass

    train_args = _Args()
    for key, value in cli.items():
        setattr(train_args, key, value)
    for key, value in vars(args).items():
        if not hasattr(train_args, key):
            setattr(train_args, key, value)
    train_args.variant = variant
    train_args.dataset = run_config.get("dataset", "tgbl-wiki")
    train_args.seed = int(run_config.get("seed", 0))

    components, _ = build_components(train_args, device, dataset, time_stats)
    for name in ("memory", "gnn", "link_pred"):
        components[name].load_state_dict(ckpt["model"][name])
    prss_core = components["prss_core"]
    if prss_core is not None:
        prss_core.load_state_dict(ckpt["model"]["prss_core"])
        prss_core.eval()
        prss_core.set_spectral_updates_allowed(False)
    adapter = components["adapter"]
    if adapter is not None:
        adapter.clear_trace()

    from prss.hosts.pyg_models.decoder import LinkPredictor  # noqa: F401
    from prss.monitoring import MonitorWriter  # noqa: F401
    from prss.training.event_loop import TGBLinkPredictionLoop

    out_dir = Path(args.output).parent
    loop = TGBLinkPredictionLoop(
        dataset=dataset, memory=components["memory"], gnn=components["gnn"],
        link_pred=components["link_pred"],
        neighbor_loader=components["neighbor_loader"],
        adapter=adapter, bridge=None, prss_core=prss_core,
        optimizer=components["optimizer"],
        unrestricted_optimizer=components["unrestricted_optimizer"],
        criterion=components["criterion"], device=device, batch_size=args.bs,
        n_neighbors=int(cli.get("n_neighbors", 10)), grad_clip=0.0,
        lambda_resp=0.0, lambda_spec=0.0, trace_roots=0,
        spectral_warmup=10 ** 9, spectral_interval=10 ** 9,
        monitor=MonitorWriter(out_dir, fail_on_error=True, reset_files=False),
        seed=int(run_config.get("seed", 0)))

    loop.reset_stream()
    loop.replay_split("train")
    loop.replay_split("val")
    before_counts, before_r = loop.audit_before()
    result = loop.evaluate_split(args.split)
    trace_created = bool(adapter is not None and adapter.trace is not None)
    assert_clean(before_counts, before_r, prss_core, trace_created,
                 f"inference({args.split})")

    payload = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "split": args.split,
        "variant": variant,
        "result": result,
        "spectral_isolation": "passed",
        "spectral_state": prss_core.snapshots() if prss_core is not None else {},
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2, allow_nan=True)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
