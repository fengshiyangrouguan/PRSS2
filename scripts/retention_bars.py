#!/usr/bin/env python3
"""Figure 4: predictive signal retention through recursive compression.

Tracks ONE source predictive signal through successive recursive
transformations and reports the fraction retained after each compression,
NORMALIZED by the signal available at the source (R_0 = 1).  Ours is compared
with the corresponding host under the SAME measurement protocol.

Computed directly from cached retention_rows.pkl files (no model rerun).
All groups share ONE root prediction target

    S_root = (Y_{a1}, Y_root)                 (phi-composed, 256-d)

and each group tracks ONLY its own source's paired-removal delta chain -- never
the full ancestor state (which would mix in the parent's own / sibling / new
neighbor information):

    3-hop source U0 :  U0  -> d32 -> d31 -> d3r      (depth 0..3 trajectory)
    2-hop source Z1 :  Z1  -> d21 -> d2r             (depth 0..2 trajectory)
    1-hop source Z2 :  Z2  -> d1r                    (depth 0..1 trajectory)

Rendered as LINE charts: retained fraction (normalized, R_0 = 1) vs recursive
depth, one subplot per source, host vs ours trajectories in the same axes.

Protocol (all directions/maps fixed on CALIB; nothing refit on audit):
  1. canonical directions between the source state X_s and S_root are fixed on
     calib -> source component Q_s = X_s_std @ W and the common root-target
     directions T = S_root_std @ B (weights = canonical strengths squared);
  2. the source bar is the weighted corr^2 between Q_s and T on audit -- the
     signal AVAILABLE at the source; every bar is divided by it, so the
     source reads exactly 1 and downstream bars are retained FRACTIONS;
  3. each ancestor bar fits the map d_{s->k} -> Q_s on calib, applies it on
     audit, and scores the RECOVERED component against the SAME root target T.

Both arms share the SAME protocol: same frozen phi (FIXED_SEED), same
GROUPS, same n_dir / ridge lambdas, same normalization.  Each arm fits its
own calib directions (its own model), which is the point of the comparison.
No J*R splicing, no audit-side refit.

Usage:
    python scripts/retention_bars.py --rows <host_rows.pkl> \
        --rows-ours <ours_rows.pkl> --out fig4_retention.png
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import retention_stats as rs  # noqa: E402

FIXED_SEED = 20260909
P_DIM = 128


def _hash01(vals, seed):
    h = (np.asarray(vals, dtype=np.int64) * 2654435761 + seed) & 0xFFFFFFFF
    return h / float(2 ** 32)


def fixed_phi_S(edge_feat, delta_t, counterpart, src_node):
    rng = np.random.RandomState(FIXED_SEED)
    raw = np.stack([np.log1p(delta_t),
                    _hash01(counterpart, 101),
                    _hash01(src_node, 202)], axis=1)
    W = rng.normal(0.0, 0.2, size=(P_DIM // 4, 3))
    b = rng.uniform(0, 2 * np.pi, size=(P_DIM // 4,))
    rff = np.cos(raw @ W.T + b) * np.sqrt(2.0 / (P_DIM // 4))
    e = np.asarray(edge_feat, dtype=np.float64)
    e = e / (np.abs(e).max(axis=1, keepdims=True) + 1e-8)
    signs = rng.choice([-1.0, 1.0], size=(P_DIM // 2, e.shape[1]))
    sk = e @ signs.T / np.sqrt(e.shape[1])
    return np.concatenate([rff, sk, np.ones((len(delta_t), P_DIM // 4))],
                          axis=1)


def _y_phi(rows, key):
    ef = np.stack([r[key][0] for r in rows])
    dt = np.asarray([r[key][1] for r in rows], dtype=np.float64)
    cp = np.asarray([r[key][2] for r in rows], dtype=np.int64)
    sn = np.asarray([r[key][3] for r in rows], dtype=np.int64)
    return fixed_phi_S(ef, dt, cp, sn)


def _col(rows, key):
    return np.stack([r[key] for r in rows])


GROUPS = [
    ("3-hop source (U0)", "u0", ["d32", "d31", "d3r"]),
    ("2-hop source (Z1)", "z1", ["d21", "d2r"]),
    ("1-hop source (Z2)", "z2", ["d1r"]),
]


def arm_bars(calib, audit, n_dir, lam, lam_ret):
    """Absolute weighted corr^2 per bar, one arm, same protocol.

    Returns {title: [(bar_name, abs_corr2), ...]} with source first; the
    caller normalizes each group by its source value (R_0 = 1).
    """
    S_c = np.concatenate([_y_phi(calib, "Y_a1"), _y_phi(calib, "Y_root")],
                         axis=1)
    S_a = np.concatenate([_y_phi(audit, "Y_a1"), _y_phi(audit, "Y_root")],
                         axis=1)
    out = {}
    for title, src_key, deltas in GROUPS:
        # 1. fix source->root directions on calib (common root-target dirs)
        mpd = rs.canonical_dirs(_col(calib, src_key), S_c, n_dir, lam=lam)
        wts = np.asarray(mpd["sv"], dtype=np.float64) ** 2
        T_a = rs.project_future(mpd, S_a)          # root target directions
        Qc = rs.predict_source_component(mpd, _col(calib, src_key))
        Qa = rs.predict_source_component(mpd, _col(audit, src_key))
        bars = [("source", rs.weighted_dir_sqcorr(Qa, T_a, wts))]
        # 2. each delta recovers the SAME source component; score vs root
        for dk in deltas:
            mp = rs.fit_ridge_map(_col(calib, dk), Qc, lam=lam_ret)
            qhat = rs.apply_ridge_map(mp, _col(audit, dk))
            bars.append((dk, rs.weighted_dir_sqcorr(qhat, T_a, wts)))
        out[title] = bars
    return out


def load_rows(path):
    with open(path, "rb") as f:
        rd = pickle.load(f)
    return rd["calib"], rd["audit"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True,
                    help="host (task-only) retention_rows.pkl")
    ap.add_argument("--rows-ours", default=None,
                    help="ours retention_rows.pkl (same protocol)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-dir", type=int, default=5)
    ap.add_argument("--lam", type=float, default=1e-2)
    ap.add_argument("--lam-ret", type=float, default=1e-2)
    args = ap.parse_args()

    arms = [("host", load_rows(args.rows))]
    if args.rows_ours:
        arms.append(("ours", load_rows(args.rows_ours)))

    norm = {}
    for name, (calib, audit) in arms:
        abs_bars = arm_bars(calib, audit, args.n_dir, args.lam, args.lam_ret)
        norm[name] = {}
        for title, bars in abs_bars.items():
            s0 = bars[0][1]
            guard = s0 if abs(s0) > 1e-8 else float("nan")
            norm[name][title] = [(nm, v / guard) for nm, v in bars]
            print("[{}] {}  ".format(name, title)
                  + "  ".join("{}={:.3f}".format(nm, rv)
                              for nm, rv in norm[name][title]),
                  flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))
    c_host = "#9aa5b1"
    c_ours = "#2a78d6"
    two = len(arms) == 2
    for ax, (title, _sk, deltas) in zip(axes, GROUPS):
        # line chart: retained fraction vs recursive depth (source = depth 0)
        xpos = np.arange(len(deltas) + 1)
        for aname, _raw in arms:
            vals = [v for _nm, v in norm[aname][title]]
            color = c_ours if aname == "ours" else c_host
            ls = "-" if aname == "ours" else "--"
            ax.plot(xpos, vals, marker="o", ms=4, lw=1.8, color=color,
                    ls=ls, zorder=3)
            for x, v in zip(xpos, vals):
                if np.isfinite(v):
                    ax.text(x, v + 0.03, "{:.2f}".format(v),
                            ha="center", fontsize=8,
                            color="#333a44" if aname == "ours" else "#7a838e")
        ax.set_xticks(xpos)
        ax.set_xticklabels(["source\n(depth 0)"]
                           + ["depth {}".format(i + 1)
                              for i in range(len(deltas))], fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0.0, 1.15)              # source = 1; noise may push ~1.0x
        ax.grid(True, axis="y", color="#e5e8ea", lw=0.6, zorder=0)
    if two:
        from matplotlib.lines import Line2D
        axes[0].legend(handles=[
            Line2D([0], [0], color=c_host, ls="--", marker="o", ms=4,
                   label="host (task-only)"),
            Line2D([0], [0], color=c_ours, ls="-", marker="o", ms=4,
                   label="ours")],
            fontsize=8, loc="upper right", frameon=False)
    axes[0].set_ylabel("fraction of the source predictive signal retained\n"
                       "(normalized to the source, R$_0$ = 1)", fontsize=9)
    fig.suptitle("Predictive signal retention through recursive compression",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(args.out, dpi=150)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
