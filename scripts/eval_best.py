#!/usr/bin/env python3
"""Evaluate the best.pt of a finished (or manually killed) run on test.

Manual early stopping companion: after killing a training run, this loads
its best.pt and the run's config.json, rebuilds the components, replays
train+val (zero-memory protocol) and evaluates the held-out test.

Usage:
    python -m scripts.eval_best --output outputs/t3_rpbe_stage2_lambda-13 \
        --data-dir old/processed_tgn_data --gpu 0
"""

import argparse
import json
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

from rpbe.data.jodie import JodieDataset
from rpbe.monitoring import MonitorWriter
from rpbe.training.jodie_loop import JodieNodeClassificationLoop


def parse_args():
    p = argparse.ArgumentParser("evaluate best.pt on test")
    p.add_argument("--output", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--gpu", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.output)
    cfg = json.load(open(out / "config.json"))
    cli = cfg["cli"]
    device = torch.device(
        "cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")

    # Rebuild the same components as the original run.
    dataset = JodieDataset(cli["data"], data_dir=args.data_dir,
                           use_validation=True)
    full, train, val, test = dataset.splits()

    import scripts.train_jodie as tj
    ns = argparse.Namespace(**cli)
    ns.data_dir = args.data_dir
    components = tj.build_components(ns, device, dataset)
    tgn = components["tgn"]
    best = torch.load(out / "best.pt", map_location=device, weights_only=False)
    for name in ("decoder", "tgn"):
        components[name].load_state_dict(best["model"][name])
    if components["compressor"] is not None and \
            "compressor" in best["model"]:
        components["compressor"].load_state_dict(best["model"]["compressor"])

    loop = JodieNodeClassificationLoop(
        tgn=tgn, decoder=components["decoder"],
        optimizer=components["optimizer"], device=device,
        batch_size=cli.get("bs", 200), n_neighbors=cli.get("n_degree", 5),
        grad_clip=cli.get("grad_clip", 5.0), monitor=MonitorWriter(out),
        seed=cli.get("seed", 0), finetune_host=cli.get("finetune_host", False),
        selection_metric=cli.get("selection_metric", "auc"),
        adapter=components["adapter"], cut_builder=components["cut_builder"],
        fixed_maps=components["fixed_maps"], rpbe_cfg=components["rpbe_cfg"],
        trace_roots=cli.get("trace_roots", 32),
        trace_mode=cli.get("trace_mode", "positive_first"))

    loop.reset_memory()
    loop.replay_split(train)
    loop.replay_split(val)
    test_row = loop.evaluate_split(test, reset=False)
    print(json.dumps({"output": str(out),
                      "test": test_row}, indent=2), flush=True)


if __name__ == "__main__":
    main()
