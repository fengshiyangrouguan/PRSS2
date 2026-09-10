#!/usr/bin/env python3
"""Absolute source-signal decay along the path, toward the ROOT boundary.

Task-only TGN, computed directly from a cached retention_rows.pkl (no model
rerun).  All groups share ONE root prediction target

    S_root = (Y_{a1}, Y_root)                 (phi-composed, 256-d)

and each group tracks ONLY its own source's paired-removal delta chain -- never
the full ancestor state (which would mix in the parent's own / sibling / new
neighbor information):

    3-hop source U0 :  U0  -> d32 -> d31 -> d3r      (4 bars)
    2-hop source Z1 :  Z1  -> d21 -> d2r             (3 bars)
    1-hop source Z2 :  Z2  -> d1r                    (2 bars)

Protocol (all directions/maps fixed on CALIB; nothing refit on audit):
  1. canonical directions between the source state X_s and S_root are fixed on
     calib -> source component Q_s = X_s_std @ W and the common root-target
     directions T = S_root_std @ B (weights = canonical strengths squared);
  2. the source bar is the weighted corr^2 between Q_s and T on audit -- the
     source's ABSOLUTE strength toward the root target, so it is not 1;
  3. each ancestor bar fits the map d_{s->k} -> Q_s on calib, applies it on
     audit, and scores the RECOVERED component against the SAME root target T.

The value is a weighted squared-correlation in [0, 1] (percentage-readable).
No J*R splicing, no source normalization, no audit-side refit.

Usage:
    python scripts/retention_bars.py --rows <retention_rows.pkl> --out fig.png
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-dir", type=int, default=5)
    ap.add_argument("--lam", type=float, default=1e-2)
    ap.add_argument("--lam-ret", type=float, default=1e-2)
    args = ap.parse_args()

    with open(args.rows, "rb") as f:
        rd = pickle.load(f)
    calib, audit = rd["calib"], rd["audit"]
    S_c = np.concatenate([_y_phi(calib, "Y_a1"), _y_phi(calib, "Y_root")],
                         axis=1)
    S_a = np.concatenate([_y_phi(audit, "Y_a1"), _y_phi(audit, "Y_root")],
                         axis=1)

    out = {}
    for title, src_key, deltas in GROUPS:
        # 1. fix source->root directions on calib (common root-target dirs)
        mpd = rs.canonical_dirs(_col(calib, src_key), S_c, args.n_dir,
                                lam=args.lam)
        wts = np.asarray(mpd["sv"], dtype=np.float64) ** 2
        T_a = rs.project_future(mpd, S_a)          # root target directions
        Qc = rs.predict_source_component(mpd, _col(calib, src_key))
        Qa = rs.predict_source_component(mpd, _col(audit, src_key))

        bars = [("source", rs.weighted_dir_sqcorr(Qa, T_a, wts))]
        # 3. each delta recovers the SAME source component; score vs root target
        for dk in deltas:
            mp = rs.fit_ridge_map(_col(calib, dk), Qc, lam=args.lam_ret)
            qhat = rs.apply_ridge_map(mp, _col(audit, dk))
            bars.append((dk, rs.weighted_dir_sqcorr(qhat, T_a, wts)))
        out[title] = bars
        print("[{}] ".format(title)
              + "  ".join("{}={:.4f}".format(nm, v) for nm, v in bars),
              flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))
    colors = {0: "#2a78d6", 1: "#eb6834", 2: "#1baf7a", 3: "#8a5cd6"}
    for ax, (title, _sk, deltas) in zip(axes, GROUPS):
        names = ["source"] + deltas
        vals = [v for _nm, v in out[title]]
        xpos = np.arange(len(names))
        ax.bar(xpos, vals, color=[colors[0]] + [colors[i + 1]
               for i in range(len(deltas))], zorder=3)
        ax.set_xticks(xpos)
        ax.set_xticklabels(["source\n({})".format(_sk)]
                           + ["+{} agg".format(i + 1)
                              for i in range(len(deltas))], fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0.0, 1.0)                      # zero baseline, %-range
        ax.grid(True, axis="y", color="#e5e8ea", lw=0.6, zorder=0)
        for x, v in zip(xpos, vals):
            ax.text(x, v + 0.02, "{:.3f}".format(v), ha="center", fontsize=8)
    axes[0].set_ylabel("weighted corr² of the source's own component\n"
                       "with the shared root target S_root=(Y_a1,Y_root)",
                       fontsize=9)
    fig.suptitle("Source signal decay along the path (absolute, not "
                 "normalized) -- Task-only, same root target, no J×R",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(args.out, dpi=150)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
