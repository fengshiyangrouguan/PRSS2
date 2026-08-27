#!/usr/bin/env python3
"""Plot the per-interface Ky Fan scores (J_frac = J/m) from j_curves.json.

Small multiples: one panel per training run, one line per interface
(tjo:layer0/1/2).  The total kf loss (negative sum of J) is reported in the
panel subtitle as first -> last so the overall trend stays readable without a
second axis.

Usage:
    python -m scripts.plot_j_curves --input outputs/j_curves.json \
        --output outputs/j_curves.png
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Categorical slots 1-3 of the reference palette (light surface).
SERIES = {
    "tjo:layer0": "#2a78d6",   # blue
    "tjo:layer1": "#eb6834",   # orange
    "tjo:layer2": "#1baf7a",   # aqua
}
PANELS = [
    ("task1_s1r",   "Task1 s0 full-RPBE — stage 1 (50 ep, running)"),
    ("task2_full",  "Task2 s0 — stage 2 RPBE full run (10 ep)"),
    ("task2_early", "Task2 s0 — stage 2 RPBE early-stopped (6 ep)"),
]
INK = "#0b0b0b"
MUTED = "#52514e"
SURFACE = "#fcfcfb"
GRID = "#d9d8d4"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="outputs/j_curves.json")
    ap.add_argument("--output", default="outputs/j_curves.png")
    args = ap.parse_args()

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))

    fig, axes = plt.subplots(1, len(PANELS), figsize=(4.6 * len(PANELS), 4.0),
                             sharey=True)
    fig.patch.set_facecolor(SURFACE)
    fig.suptitle("Per-interface Ky Fan score  J/m   (m = 256)",
                 fontsize=12, color=INK, y=0.97)

    for ax, (key, title) in zip(axes, PANELS):
        run = data.get(key, {})
        ax.set_facecolor(SURFACE)
        ax.set_title(title, fontsize=10, color=INK, pad=8)
        if not run:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    color=MUTED, transform=ax.transAxes)
            ax.set_axis_off()
            continue

        for tau, color in SERIES.items():
            pts = run.get(tau, [])
            if not pts:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(xs, ys, color=color, lw=2.0, marker="o", ms=8,
                    markerfacecolor=SURFACE, markeredgecolor=color,
                    markeredgewidth=2.0, label=tau.split(":")[1], zorder=3)
            # Direct label at the line end (series identity without color).
            ax.annotate(tau.split(":")[1], (xs[-1], ys[-1]),
                        xytext=(6, 0), textcoords="offset points",
                        color=MUTED, fontsize=8, va="center")

        kf = run.get("kf_loss", [])
        if kf:
            ax.text(0.02, 0.96,
                    "kf_loss: {:.1f} → {:.1f}".format(kf[0][1], kf[-1][1]),
                    transform=ax.transAxes, fontsize=8, color=MUTED,
                    va="top", ha="left")

        ax.set_xlabel("epoch", fontsize=9, color=MUTED)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.tick_params(colors=MUTED, labelsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)

    axes[0].set_ylabel("J / m", fontsize=10, color=INK)
    handles = [plt.Line2D([], [], color=c, lw=2.0, marker="o", ms=8,
                          markerfacecolor=SURFACE, markeredgecolor=c,
                          markeredgewidth=2.0)
               for c in SERIES.values()]
    fig.legend(handles, ["layer0", "layer1", "layer2"],
               loc="lower center", ncol=3, frameon=False,
               fontsize=9, labelcolor=MUTED)

    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(args.output, dpi=150, facecolor=SURFACE)
    print("wrote", args.output)


if __name__ == "__main__":
    main()
