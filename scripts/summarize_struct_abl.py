#!/usr/bin/env python3
"""Summarise structural-ablation arms (Part III) into a comparison table.

Reads ``outputs/<outroot>/seed{N}/{1obs,2obs_aligned,2obs_mispaired}/
summary.json`` (each written by train_jodie with val-AUC early stopping) and
prints per-arm, per-seed task metrics plus the per-arm mean over seeds.

NOTE on fair comparison: these are each arm's own checkpoints selected by the
SAME protocol (val AUC).  Per spec Part III.7 they must NOT be read as the
decisive structural comparison on their own — the shared held-out aligned
audit (scripts/audit_aligned_heldout.py) is the common objective.  This table
reports the downstream task metrics only.

Usage:
    python -m scripts.summarize_struct_abl --outroot struct_abl \
        [--seeds 0 1 2 3 4] [--metrics auc ap nll]
"""

import argparse
import json
from pathlib import Path

ARMS = ["1obs", "2obs_aligned", "2obs_mispaired"]


def parse_args():
    p = argparse.ArgumentParser("structural ablation summary")
    p.add_argument("--outroot", default="struct_abl")
    p.add_argument("--seeds", default="0 1 2 3 4")
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--metrics", default="auc ap")
    return p.parse_args()


def main():
    args = parse_args()
    seeds = [int(x) for x in args.seeds.split()]
    metrics = args.metrics.split()
    base = Path(args.output_dir) / args.outroot
    print("structural-ablation task metrics  (each arm: its own checkpoint, "
          "val-AUC selected)")
    header = "seed | arm | " + " | ".join(m.upper() for m in metrics)
    print(header)
    print("-" * len(header))
    per_arm = {a: {m: [] for m in metrics} for a in ARMS}
    missing = []
    for seed in seeds:
        for arm in ARMS:
            p = base / "seed{}".format(seed) / arm / "summary.json"
            if not p.exists():
                missing.append(str(p))
                continue
            s = json.load(open(p))
            t = s["test"]
            vals = [t.get(m) for m in metrics]
            for m, v in zip(metrics, vals):
                per_arm[arm][m].append(v)
            print("{} | {} | {}".format(seed, arm,
                                        " | ".join(
                                            "{:.5f}".format(v) if v is not None
                                            else "NA" for v in vals)))
    print("-" * len(header))
    print("MEAN")
    for arm in ARMS:
        if not per_arm[arm][metrics[0]]:
            continue
        print("{} | {}".format(arm, " | ".join(
            "{:.5f}".format(sum(v) / len(v))
            for v in [per_arm[arm][m] for m in metrics])))
    if missing:
        print("\nmissing summaries ({}):".format(len(missing)))
        for m in missing:
            print("  {}".format(m))


if __name__ == "__main__":
    main()
