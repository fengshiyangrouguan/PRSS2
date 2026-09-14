#!/usr/bin/env python3
"""Figure 5: Robustness of LPSE (UCI, 3-hop setting, additive arm).

Three panels, per the paper caption:

  (a) sensitivity to the LPSE weight beta  (x: beta, log scale)
  (b) sensitivity to the compression ratio d_z / d_u  (x: d/172)
  (c) sensitivity to the statistical support used to estimate LPSE
      (x: fraction of window trees kept)

Performance = test AP (final_test.ap from summary.json).  Anchors
(beta=0.00668 additive, P0 task-only) come from the Table-2 four-card
runs (3 seeds each, mean + std); new Fig-5 points are 1 seed each.
beta=0 is P0 by construction (additive with zero LPSE weight recovers
the matched Task-only model).

Usage:
    python scripts/fig5_robustness.py --dir fig5_out --dir-nma1 nma1_out \
        --anchor table2_root --out fig5_robustness.png
"""
import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OURS = "#2a78d6"
HOST = "#eb6834"
RAW_DIM = 172.0


def load_ap(summary_path):
    with open(summary_path) as f:
        s = json.load(f)
    test = s.get("test") or {}
    ap = test.get("ap")
    if ap is None:
        raise ValueError("no test ap in {}".format(summary_path))
    return float(ap)


def mean_std(vals):
    v = np.asarray(vals, dtype=np.float64)
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0


def scan_dir(root, name):
    """Gather 1-seed points for ``name`` across seed dirs in root."""
    root = Path(root)
    pts = []
    for d in sorted(root.glob("seed0_*")):
        if d.name.endswith(name) and (d / "summary.json").exists():
            pts.append(load_ap(d / "summary.json"))
    return mean_std(pts) if pts else (None, None)


def anchor(root, sub, seed=1):
    """Table-2 four-card anchor: SINGLE seed (Fig-5 is 1-seed everywhere).

    seed 1 is the fixed anchor seed for all three panels; every new Fig-5
    point is also seed 0-1 of its own run, so the figure stays consistent.
    """
    p = Path(root) / "seed{}_TGN_UCI_3L10N".format(seed) / sub / "summary.json"
    if not p.exists():
        raise SystemExit("anchor {} seed{} missing under {}".format(
            sub, seed, root))
    return load_ap(p), 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="fig5 serial outputs")
    ap.add_argument("--dir-nma1", required=True, help="nma1 pair outputs")
    ap.add_argument("--anchor", required=True,
                    help="Table-2 root (four-card, 3-seed anchors)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # ---- anchors (3 seeds) ----------------------------------------------
    p0_mean, p0_std = anchor(args.anchor, "P0")
    add_mean, add_std = anchor(args.anchor, "additive")
    print("[anchor] P0 {:.4f}+-{:.4f}  additive {:.4f}+-{:.4f}".format(
        p0_mean, p0_std, add_mean, add_std))

    # ---- (a) beta scan: beta=0 -> P0; beta=0.00668 -> additive anchor ----
    betas, ours_a, err_a = [], [], []
    betas.append(0.0)
    ours_a.append(p0_mean)
    err_a.append(p0_std)
    for b in (0.00167, 0.00334, 0.00668, 0.01336, 0.02672):
        if abs(b - 0.00668) < 1e-9:
            m, s = add_mean, add_std
        else:
            m, s = scan_dir(args.dir, "b{:05d}".format(int(round(b * 1e5))))
            if m is None:
                raise SystemExit("missing beta={} run".format(b))
        betas.append(b)
        ours_a.append(m)
        err_a.append(s)
    p0_line_a = [p0_mean] * len(betas)

    # ---- (b) compression ratio scan --------------------------------------
    ds = [64, 96, 128, 172]
    ratios = [d / RAW_DIM for d in ds]
    ours_b, err_b, p0_b, err_p0b = [], [], [], []
    for d in ds:
        if d == 172:
            m, s = add_mean, add_std
            mp, sp = p0_mean, p0_std
        else:
            m, s = scan_dir(args.dir_nma1, "d{}_ours".format(d))
            mp, sp = scan_dir(args.dir_nma1, "d{}_p0".format(d))
            if m is None or mp is None:
                raise SystemExit("missing d={} pair".format(d))
        ours_b.append(m)
        err_b.append(s)
        p0_b.append(mp)
        err_p0b.append(sp)

    # ---- (c) statistical support scan ------------------------------------
    fracs = [0.25, 0.5, 1.0]
    ours_c, err_c = [], []
    for f in fracs:
        if abs(f - 1.0) < 1e-9:
            m, s = add_mean, add_std
        else:
            m, s = scan_dir(args.dir, "sup{:03d}".format(int(f * 100)))
            if m is None:
                raise SystemExit("missing support={} run".format(f))
        ours_c.append(m)
        err_c.append(s)
    p0_line_c = [p0_mean] * len(fracs)

    # ---- render ----------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))

    # (a)
    ax = axes[0]
    ax.errorbar(betas, p0_line_a, yerr=None, color=HOST, ls="--", lw=1.8,
                marker="o", ms=4.5, label="Task-only (P0)", zorder=3)
    ax.errorbar(betas, ours_a, yerr=err_a, color=OURS, ls="-", lw=1.8,
                marker="o", ms=4.5, label="Ours (LPSE)", zorder=3)
    ax.set_xscale("log")
    ax.set_xlabel(r"LPSE weight $\beta$ (log scale)", fontsize=9)
    ax.set_ylabel("test AP on UCI (3-hop)", fontsize=9)

    # (b)
    ax = axes[1]
    ax.errorbar(ratios, p0_b, yerr=err_p0b, color=HOST, ls="--", lw=1.8,
                marker="o", ms=4.5, label="Task-only (P0)", zorder=3)
    ax.errorbar(ratios, ours_b, yerr=err_b, color=OURS, ls="-", lw=1.8,
                marker="o", ms=4.5, label="Ours (LPSE)", zorder=3)
    ax.set_xlabel(r"compression ratio $d_z / d_u$  ($d_u$ = 172)", fontsize=9)
    ax.set_xticks(ratios)
    ax.set_xticklabels(["{:.2f}".format(r) for r in ratios], fontsize=8)

    # (c)
    ax = axes[2]
    ax.errorbar(fracs, p0_line_c, color=HOST, ls="--", lw=1.8, marker="o",
                ms=4.5, label="Task-only (P0)", zorder=3)
    ax.errorbar(fracs, ours_c, yerr=err_c, color=OURS, ls="-", lw=1.8,
                marker="o", ms=4.5, label="Ours (LPSE)", zorder=3)
    ax.set_xlabel("statistical support\n(fraction of window trees)",
                  fontsize=9)

    for ax in axes:
        ax.grid(True, axis="y", color="#e5e8ea", lw=0.6, zorder=0)
        ax.legend(fontsize=8, frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
