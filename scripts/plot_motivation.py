#!/usr/bin/env python3
"""Plot-ready figures from a motivation_probe.json report (Part II.10).

Reads the JSON emitted by ``scripts.motivation_probe`` and draws:

* Figure A — available gain A_d and retained gain K_d vs aggregation depth,
  with 95% cluster-bootstrap CIs as shaded bands (the point CIs live in the
  JSON; bands are connected between the CI endpoints).
* Figure B — PUR_d vs depth, with the identity and permutation controls as
  dashed reference curves when ``--controls`` was run.

Also writes a plot-ready CSV: one row per (depth, quantity, control) with
point value and CI.

Usage:
    python -m scripts.plot_motivation \\
        --json outputs/task_only/motivation_probe.json \\
        --outdir outputs/task_only/motivation_figs
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TITLES = {
    "audit": "TGN task-only baseline",
    "z_equals_u": "control: Z=U (identity)",
    "random_z": "control: random Z",
    "permuted_target": "control: permuted target",
}


def parse_args():
    p = argparse.ArgumentParser("motivation figure plotter")
    p.add_argument("--json", required=True, help="motivation_probe.json")
    p.add_argument("--outdir", default="", help="figure output dir")
    return p.parse_args()


def _collect(report):
    """Return per-depth rows with A_d/K_d/PUR + CIs + controls."""
    base = report.get("audit") or {}
    rows = []
    for d_str, r in sorted(base.items(), key=lambda kv: int(kv[0])):
        if "skipped" in r:
            continue
        rows.append({
            "depth": int(d_str),
            "label": d_str,
            "A_d": r.get("A_d"), "K_d": r.get("K_d"),
            "PUR_d": r.get("PUR_d"),
            "A_d_ci": r.get("A_d_ci"), "K_d_ci": r.get("K_d_ci"),
            "kind": "audit",
        })
    controls = report.get("controls") or {}
    for name, ctrl in controls.items():
        for d_str, r in sorted(ctrl.items(), key=lambda kv: int(kv[0])):
            if not isinstance(r, dict) or "skipped" in r:
                continue
            rows.append({
                "depth": int(d_str),
                "label": "{}_{}".format(name, d_str),
                "A_d": r.get("A_d"), "K_d": r.get("K_d"),
                "PUR_d": r.get("PUR_d"),
                "A_d_ci": r.get("A_d_ci"), "K_d_ci": r.get("K_d_ci"),
                "kind": name,
            })
    return rows


def main():
    args = parse_args()
    report = json.load(open(args.json))
    rows = _collect(report)
    outdir = Path(args.outdir) if args.outdir else Path(args.json).parent
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- plot-ready CSV
    csv_path = outdir / "plot_ready.csv"
    with open(csv_path, "w") as f:
        f.write("kind,depth,A_d,A_d_lo,A_d_hi,K_d,K_d_lo,K_d_hi,PUR_d\n")
        for r in rows:
            a_ci = r["A_d_ci"] or ["", ""]
            k_ci = r["K_d_ci"] or ["", ""]
            f.write("{},{},{},{},{},{},{},{},{}\n".format(
                r["kind"], r["depth"],
                _fmt(r["A_d"]), _fmt(a_ci[0] if len(a_ci) > 0 else ""),
                _fmt(a_ci[1] if len(a_ci) > 1 else ""),
                _fmt(r["K_d"]), _fmt(k_ci[0] if len(k_ci) > 0 else ""),
                _fmt(k_ci[1] if len(k_ci) > 1 else ""),
                _fmt(r["PUR_d"])))

    audit_rows = [r for r in rows if r["kind"] == "audit"]
    ctrl_rows = {k: [r for r in rows if r["kind"] == k]
                 for k in TITLES if k != "audit"}

    if not audit_rows:
        print("no audit rows; nothing to plot", flush=True)
        return

    # ---- Figure A: available / retained gain vs depth
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    x = [r["depth"] for r in audit_rows]
    xs = np.arange(len(x))
    a = [r["A_d"] for r in audit_rows]
    k = [r["K_d"] for r in audit_rows]
    ax.plot(xs, a, "-o", color="#1f77b4", label="available $A_d$")
    ax.plot(xs, k, "-s", color="#d62728", label="retained $K_d$")
    for i, r in enumerate(audit_rows):
        if r["A_d_ci"]:
            ax.plot([i, i], [r["A_d_ci"][0], r["A_d_ci"][1]],
                    color="#1f77b4", alpha=0.4, lw=2)
        if r["K_d_ci"]:
            ax.plot([i, i], [r["K_d_ci"][0], r["K_d_ci"][1]],
                    color="#d62728", alpha=0.4, lw=2)
    ax.set_xticks(xs)
    ax.set_xticklabels(x)
    ax.set_xlabel("aggregation depth (recursion layer)")
    ax.set_ylabel("predictive gain (normalized MSE reduction)")
    ax.set_title("TGN task-only: future info available vs retained")
    ax.axhline(0, color="gray", lw=0.8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "figure_A_depth_gain.png", dpi=150)
    plt.close(fig)

    # ---- Figure B: PUR vs depth with controls
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    x_idx = {r["depth"]: i for i, r in enumerate(audit_rows)}

    def _xs(kind):
        return [x_idx.get(r["depth"], float("nan"))
                for r in rows if r["kind"] == kind
                and r["PUR_d"] is not None]

    ax.plot(_xs("audit"), [r["PUR_d"] for r in audit_rows
                           if r["PUR_d"] is not None],
            "-o", color="#2ca02c", label="task-only baseline")
    for name in TITLES:
        if name == "audit":
            continue
        pts = [(r["depth"], r["PUR_d"]) for r in ctrl_rows.get(name, [])
               if r["PUR_d"] is not None]
        if not pts:
            continue
        xv = [x_idx.get(d, float("nan")) for d, _ in pts]
        yv = [p for _, p in pts]
        ax.plot(xv, yv, "--o", label=TITLES.get(name, name))
    ax.set_xticks(range(len(audit_rows)))
    ax.set_xticklabels([r["depth"] for r in audit_rows])
    ax.set_xlabel("aggregation depth (recursion layer)")
    ax.set_ylabel("PUR$_d$ = $K_d / A_d$")
    ax.set_title("Predictive Utilization Ratio")
    ax.axhline(0, color="gray", lw=0.8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "figure_B_pur.png", dpi=150)
    plt.close(fig)
    print("wrote {} and figures to {}".format(csv_path, outdir), flush=True)


def _fmt(v):
    if v is None:
        return ""
    if isinstance(v, (list, tuple)):
        return ",".join(_fmt(x) for x in v)
    try:
        return "{:.6g}".format(float(v))
    except (TypeError, ValueError):
        return ""


if __name__ == "__main__":
    main()
