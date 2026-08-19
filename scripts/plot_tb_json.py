#!/usr/bin/env python3
"""Plot TensorBoard scalar data previously exported to JSON (see tb_export).

Input JSON format (produced by the cloud-side EventAccumulator dump):
    {"tag": [[step, value], ...], ...}

Usage:
    python scripts/plot_tb_json.py --json path/to/tb.json --outdir path/to/png
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EPOCH_STEPS = 552  # tgbl-wiki: batches per epoch at bs=200


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--json", required=True, help="TB scalar JSON exported from EventAccumulator")
    p.add_argument("--outdir", required=True, help="PNG output dir")
    p.add_argument("--epoch-steps", type=int, default=EPOCH_STEPS,
                   help="global steps per epoch (for x-axis conversion)")
    args = p.parse_args()

    with open(args.json) as f:
        data = json.load(f)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    written = []

    # ---- per-epoch curves (val/test) -----------------------------------
    epoch_tags = {t: pts for t, pts in data.items() if t.startswith("epoch/")}
    for tag, pts in sorted(epoch_tags.items()):
        xs = [s / args.epoch_steps for s, _ in pts]
        ys = [v for _, v in pts]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(xs, ys, "-o", ms=3)
        ax.set_xlabel("epoch")
        ax.set_ylabel(tag.replace("epoch/", ""))
        ax.set_title(tag.replace("epoch/", ""))
        ax.grid(alpha=0.3)
        fig.tight_layout()
        name = tag.replace("/", "__") + ".png"
        fig.savefig(outdir / name, dpi=150)
        plt.close(fig)
        written.append(name)

    # ---- final test markers ---------------------------------------------
    final_tags = {t: pts for t, pts in data.items() if t.startswith("final/")}
    if final_tags:
        fig, ax = plt.subplots(figsize=(7, 4))
        labels = [t.replace("final/", "") for t in final_tags]
        vals = [pts[-1][1] for pts in final_tags.values()]
        ax.bar(labels, vals)
        ax.set_title("final test metrics")
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        name = "final_test_metrics.png"
        fig.savefig(outdir / name, dpi=150)
        plt.close(fig)
        written.append(name)

    # ---- train loss over steps -----------------------------------------
    if "train/task_loss" in data:
        pts = data["train/task_loss"]
        xs = [s / args.epoch_steps for s, _ in pts]
        ys = [v for _, v in pts]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(xs, ys, lw=0.8)
        ax.set_xlabel("epoch")
        ax.set_ylabel("train task loss")
        ax.set_title("train task loss (per-100-step samples)")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        name = "train_task_loss.png"
        fig.savefig(outdir / name, dpi=150)
        plt.close(fig)
        written.append(name)

    print(f"wrote {len(written)} PNGs to {outdir}:")
    for name in written:
        print(f"  {name}")


if __name__ == "__main__":
    main()
