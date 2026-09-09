#!/usr/bin/env python3
"""Render the gated retention figure from retention_audit_v2.json.

One figure, a common physical leaf-to-root axis
    0 = leaf(U0) -> 1 = a_2(Z1) -> 2 = a_1(Z2) -> 3 = root(Z3),
with three staggered source lines (3-hop / 2-hop / 1-hop), each starting at
100% at its own source origin and estimated as *direct* same-source explained
variances (never a chain product).  Only lines whose gates all pass
(line_ok in the JSON: source signal > shuffle-null p95, identity ~1,
delete-source floor ~0, mismatched-delta no-inflation, memory parity) are
drawn; failing lines are listed in the caption instead.

Usage:
    python scripts/render_retention_fig.py --json <retention_audit_v2.json> \
        --out <fig.png>
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LINE_COLORS = {3: "#2a78d6", 2: "#eb6834", 1: "#1baf7a"}
POS_LABELS = {0: "leaf\n(U0)", 1: "a2\n(Z1)", 2: "a1\n(Z2)", 3: "root\n(Z3)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.json) as f:
        report = json.load(f)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.axhline(1.0, color="#8a9199", lw=1.0, ls="--", zorder=1)
    ax.axhline(0.0, color="#b7bcc2", lw=0.8, ls=":", zorder=1)

    ok = []
    dropped = []
    for s in sorted((int(k) for k in report["sources"])):
        src = report["sources"][str(s)]
        main = src["same_tail"]               # main result: same-tail causal fit
        name, color = src["name"], LINE_COLORS[s]
        pts = main["points"]
        phys = [p["phys"] for p in pts]
        R = [max(-0.25, min(1.0, p["R"])) for p in pts]
        lo = [max(-0.25, min(1.0, p["ci_lo"])) for p in pts]
        hi = [max(-0.25, min(1.0, p["ci_hi"])) for p in pts]
        if not main["line_ok"]:
            fails = []
            for g, v in [("signal", main["signal"]),
                         ("identity", main["identity"]),
                         ("delete_floor", main["delete_floor"]),
                         ("mismatch", main["mismatched_delta"])]:
                if not v["ok"]:
                    fails.append("{}={:.3f}".format(g, v["value"]))
            dropped.append("{}hop {} ({})".format(s, name, ", ".join(fails)))
            continue
        ok.append((s, name, color, phys, R, lo, hi))

    if not ok:
        ax.text(0.5, 0.5, "no source line passed the same-tail gates",
                ha="center", va="center", transform=ax.transAxes)
    for s, name, color, phys, R, lo, hi in ok:
        ax.plot(phys, R, "-o", color=color, lw=1.8, ms=4.5,
                label="{}hop source {}".format(s, name), zorder=3)
        ax.fill_between(phys, lo, hi, color=color, alpha=0.12, lw=0,
                        zorder=2)

    ax.set_xlabel("position along the leaf-to-root path (compressions)")
    ax.set_ylabel("retention of source prediction component\n"
                  "R = 1 - SSE / SST (same-tail causal fit, direct regression)")
    ax.set_xticks(list(POS_LABELS.keys()))
    ax.set_xticklabels([POS_LABELS[k] for k in sorted(POS_LABELS)])
    ax.set_ylim(-0.3, 1.08)
    ax.set_xlim(-0.2, 3.2)
    ax.grid(True, axis="y", color="#e5e8ea", lw=0.6, zorder=0)
    if ok:
        ax.legend(loc="lower left", fontsize=8, frameon=False)
    if dropped:
        ax.text(0.02, -0.24, "not drawn (same-tail gates): "
                + "; ".join(dropped),
                transform=ax.transAxes, fontsize=7, color="#6a7178",
                va="top")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
