#!/usr/bin/env python3
"""Render the gated retention figure as a Task-only vs Ours overlay.

One 1x3 panel figure: one panel per source depth (3-hop U0, 2-hop Z1,
1-hop Z2).  Each panel shows the canonical-strength-weighted corr²
recoverability of the SAME fixed Q_s along that source's leaf-to-root path
(its own staggered origin at its own position) for TWO models, e.g. the
Task-only arm and the Ours (exact-replay) arm, on the same window/protocol.

The plotted value is correlation-type linear recoverability in [0,1] -- it is
NOT a retained-information fraction and NOT R = 1 - SSE/SST; that formula is
kept only as the ev_raw diagnostic in the JSON.  Only lines whose same-tail
gates all pass (line_ok) are drawn.

Usage:
    python scripts/render_retention_fig.py \
        --model task_only:outputs/.../task_only/retention_audit_v2.json \
        --model ours:outputs/.../exact_replay/retention_audit_v2.json \
        --out fig.png
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

POS_LABELS = {0: "leaf\n(U0)", 1: "a2\n(Z1)", 2: "a1\n(Z2)", 3: "root\n(Z3)"}
PANELS = {
    3: ("3-hop source (U0)", 0),
    2: ("2-hop source (Z1)", 1),
    1: ("1-hop source (Z2)", 2),
}
STYLE = {
    "task_only": dict(color="#eb6834", ls="--", marker="o"),
    "ours": dict(color="#2a78d6", ls="-", marker="o"),
    "exact_replay": dict(color="#2a78d6", ls="-", marker="o"),
}


def _pick_arm(json_path):
    """Best-effort arm label from the json path parent dir or file name."""
    import os
    parent = os.path.basename(os.path.dirname(json_path))
    if "exact_replay" in parent or "exact" in parent:
        return "ours"
    if "task_only" in parent:
        return "task_only"
    return parent or "model"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True,
                    metavar="ARM:PATH",
                    help="model arm name (task_only/ours) and its "
                         "retention_audit_v2.json; repeatable")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    models = []
    for m in args.model:
        arm, path = m.split(":", 1)
        with open(path) as f:
            report = json.load(f)
        models.append((arm, path, report))

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2), sharey=True)
    for (s, (title, ax_i)) in PANELS.items():
        ax = axes[ax_i]
        ax.axhline(0.0, color="#b7bcc2", lw=0.8, ls=":", zorder=1)
        for arm, path, report in models:
            if str(s) not in report["sources"]:
                continue
            src = report["sources"][str(s)]
            main_res = src["same_tail"]
            if not main_res.get("line_ok", False):
                ax.text(0.5, 0.92, "gates fail", transform=ax.transAxes,
                        ha="center", fontsize=8, color="#6a7178")
                continue
            pts = main_res["points"]
            phys = [p["phys"] for p in pts]
            R = [max(-0.05, min(1.05, p["Rw"])) for p in pts]
            lo = [max(-0.05, min(1.05, p["ciw_lo"])) for p in pts]
            hi = [max(-0.05, min(1.05, p["ciw_hi"])) for p in pts]
            st = STYLE.get(arm, {})
            ax.plot(phys, R, lw=1.8, ms=4.5, zorder=3,
                    label="{}({})".format(arm, src["name"]),
                    **st)
            ax.fill_between(phys, lo, hi, color=st.get("color", "#333"),
                            alpha=0.10, lw=0, zorder=2)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(list(POS_LABELS.keys()))
        ax.set_xticklabels([POS_LABELS[k] for k in sorted(POS_LABELS)],
                           fontsize=8)
        ax.grid(True, axis="y", color="#e5e8ea", lw=0.6, zorder=0)
        ax.legend(fontsize=8, frameon=False, loc="lower left")
    axes[0].set_ylabel("strength-weighted corr² recoverability\n"
                       "of fixed Q_s (not R=1-SSE/SST; not a\n"
                       "retained-info fraction)", fontsize=9)
    fig.suptitle("Same fixed source component after 1/2/3 recursive "
                 "aggregations -- Task-only vs Ours (same window/protocol)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(args.out, dpi=150)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
