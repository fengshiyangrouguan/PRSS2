#!/usr/bin/env python3
"""Figure 4 renderer for retention_audit_v3.json (NLL-based MI protocol).

One 1x3 panel figure, x-axis = physical position along the path
(leaf / a2 / a1 / root), y-axis = normalized conditional future-predictive
information R = I_k / I_source (R_0 = 1), two trajectories per panel
(host dashed, ours solid) with cluster-bootstrap CI bands.

The plotted quantity is the held-out joint-future NLL drop (bits/sample)
of the SAME source predictive signal, normalized by the source value —
the Figure-4 caption semantics ("fraction retained after each compression,
normalized by the signal available at the source").  R>1 is possible and is
NOT clipped (flagged in the JSON as R_gt1_flag).

Usage:
    python scripts/render_retention_v3.py \
        --model host:path/to/task_only/retention_audit_v3.json \
        --model ours:path/to/exact_replay/retention_audit_v3.json \
        --out fig4_retention.png
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

POS_LABELS = {0: "leaf\n(U0)", 1: "a2\n(Z1)", 2: "a1\n(Z2)", 3: "root\n(Z3)"}
PANELS = [
    ("Y_leaf", "3-hop source (U0)"),
    ("Y_a2", "2-hop source (Z1)"),
    ("Y_a1", "1-hop source (Z2)"),
]
STYLE = {
    "host": dict(color="#eb6834", ls="--", marker="o"),
    "task_only": dict(color="#eb6834", ls="--", marker="o"),
    "ours": dict(color="#2a78d6", ls="-", marker="o"),
    "exact_replay": dict(color="#2a78d6", ls="-", marker="o"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True,
                    metavar="ARM:PATH",
                    help="arm (host/ours) and its retention_audit_v3.json; "
                         "repeatable")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    models = []
    for m in args.model:
        arm, path = m.split(":", 1)
        with open(path) as f:
            models.append((arm, json.load(f)))

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2), sharey=True)
    for ax, (line, title) in zip(axes, PANELS):
        ax.axhline(0.0, color="#b7bcc2", lw=0.8, ls=":", zorder=1)
        for arm, report in models:
            src = report.get("sources", {}).get(line)
            if not src:
                continue
            i0 = src.get("info_source_bits")
            pts = src.get("points", [])
            if i0 is None or abs(i0) <= 1e-9 or not pts:
                ax.text(0.5, 0.92, "no data", transform=ax.transAxes,
                        ha="center", fontsize=8, color="#6a7178")
                continue
            xpos = [p["phys"] for p in pts]
            R = [p["R"] for p in pts]
            lo = [c[0] / i0 for p in pts for c in [p.get("ci95",
                                                         [float("nan"),
                                                          float("nan")])]]
            hi = [c[1] / i0 for p in pts for c in [p.get("ci95",
                                                         [float("nan"),
                                                          float("nan")])]]
            st = STYLE.get(arm, {})
            ax.plot(xpos, R, lw=1.8, ms=4.5, zorder=3, label=arm, **st)
            ax.fill_between(xpos, lo, hi, color=st.get("color", "#333"),
                            alpha=0.10, lw=0, zorder=2)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(list(POS_LABELS.keys()))
        ax.set_xticklabels([POS_LABELS[k] for k in sorted(POS_LABELS)],
                           fontsize=8)
        ax.grid(True, axis="y", color="#e5e8ea", lw=0.6, zorder=0)
        ax.legend(fontsize=8, frameon=False, loc="lower left")
    axes[0].set_ylabel("normalized conditional future-predictive\n"
                       "information R = I$_k$ / I$_{source}$ (R$_0$ = 1)",
                       fontsize=9)
    fig.suptitle("Predictive signal retention through recursive compression "
                 "-- host vs ours (same measurement protocol)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(args.out, dpi=150)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
