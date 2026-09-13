#!/usr/bin/env python3
"""Figure 4: predictive signal retention through recursive compression.

Tracks ONE source predictive signal through successive recursive
transformations and reports the fraction retained after each compression,
NORMALIZED by the signal available at the source (R_0 = 1).  Ours is compared
with the corresponding host under the SAME measurement protocol.

Rendered in the audit_v2_fig layout: one 1x3 panel figure, x-axis = physical
position along the path (leaf / a2 / a1 / root), y-axis = normalized retained
fraction, two trajectories per panel (host dashed, ours solid) with
bootstrap CI bands.

Computed directly from cached retention_rows.pkl files (no model rerun).
All groups share ONE root prediction target

    S_root = (Y_{a1}, Y_root)                 (phi-composed, 256-d)

and each group tracks ONLY its own source's paired-removal delta chain -- never
the full ancestor state (which would mix in the parent's own / sibling / new
neighbor information):

    3-hop source U0 :  U0  -> d32 -> d31 -> d3r      (leaf..root, 4 points)
    2-hop source Z1 :  Z1  -> d21 -> d2r             (a2..root,    3 points)
    1-hop source Z2 :  Z2  -> d1r                    (a1..root,    2 points)

Protocol (all directions/maps fixed on CALIB; nothing refit on audit):
  1. canonical directions between the source state X_s and S_root are fixed on
     calib -> source component Q_s = X_s_std @ W and the common root-target
     directions T = S_root_std @ B (weights = canonical strengths squared);
  2. the source bar is the weighted corr^2 between Q_s and T on audit -- the
     signal AVAILABLE at the source; every point is divided by it, so the
     source reads exactly 1 and downstream points are retained FRACTIONS;
  3. each downstream point fits the map d_{s->k} -> Q_s on calib, applies it
     on audit, and scores the RECOVERED component against the SAME root
     target T.
  4. CI bands: paired cluster-bootstrap over audit rows (same resample index
     for source and every downstream point), so the normalized ratio's
     distribution is estimated jointly -- percentile 2.5/97.5.

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


# title, source-state key, downstream delta chain, x position of the source
GROUPS = [
    ("3-hop source (U0)", "u0", ["d32", "d31", "d3r"], 0),
    ("2-hop source (Z1)", "z1", ["d21", "d2r"], 1),
    ("1-hop source (Z2)", "z2", ["d1r"], 2),
]

# physical position on the shared leaf->root axis (audit_v2_fig layout)
POS_LABELS = {0: "leaf\n(U0)", 1: "a2\n(Z1)", 2: "a1\n(Z2)", 3: "root\n(Z3)"}


def arm_bars(calib, audit, n_dir, lam, lam_ret, n_boot, seed):
    """Normalized retention trajectory per group, with paired-bootstrap CIs.

    Returns {title: [(name, frac, ci_lo, ci_hi), ...]} with source first
    (frac == 1); CI bands from a paired cluster-bootstrap over audit rows.
    """
    S_c = np.concatenate([_y_phi(calib, "Y_a1"), _y_phi(calib, "Y_root")],
                         axis=1)
    S_a = np.concatenate([_y_phi(audit, "Y_a1"), _y_phi(audit, "Y_root")],
                         axis=1)
    out = {}
    for title, src_key, deltas, _x0 in GROUPS:
        mpd = rs.canonical_dirs(_col(calib, src_key), S_c, n_dir, lam=lam)
        wts = np.asarray(mpd["sv"], dtype=np.float64) ** 2
        T_a = rs.project_future(mpd, S_a)
        Qc = rs.predict_source_component(mpd, _col(calib, src_key))
        Qa = rs.predict_source_component(mpd, _col(audit, src_key))
        maps = {dk: rs.fit_ridge_map(_col(calib, dk), Qc, lam=lam_ret)
                for dk in deltas}
        d_audit = {dk: _col(audit, dk) for dk in deltas}

        def eval_at(idx):
            ta = T_a[idx]
            s0 = rs.weighted_dir_sqcorr(Qa[idx], ta, wts)
            vs = [rs.weighted_dir_sqcorr(
                rs.apply_ridge_map(maps[dk], d_audit[dk][idx]), ta, wts)
                for dk in deltas]
            return s0, vs

        s0_full, vs_full = eval_at(np.arange(len(audit)))
        r_full = np.asarray(vs_full, dtype=np.float64) / max(s0_full, 1e-8)
        rng = np.random.RandomState(seed)
        n = len(audit)
        boot = np.zeros((n_boot, len(deltas)))
        for b in range(n_boot):
            idx = rng.choice(n, size=n, replace=True)
            s0b, vsb = eval_at(idx)
            boot[b] = np.asarray(vsb) / max(s0b, 1e-8)
        lo = np.percentile(boot, 2.5, axis=0)
        hi = np.percentile(boot, 97.5, axis=0)
        bars = [("source", 1.0, 1.0, 1.0)]
        for i, dk in enumerate(deltas):
            bars.append((dk, float(r_full[i]), float(lo[i]), float(hi[i])))
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
    ap.add_argument("--n-boot", type=int, default=200)
    args = ap.parse_args()

    arms = [("host", load_rows(args.rows))]
    if args.rows_ours:
        arms.append(("ours", load_rows(args.rows_ours)))

    norm = {}
    for name, (calib, audit) in arms:
        norm[name] = arm_bars(calib, audit, args.n_dir, args.lam,
                              args.lam_ret, args.n_boot,
                              seed=20260909 + (1 if name == "ours" else 0))
        for title, bars in norm[name].items():
            print("[{}] {}  ".format(name, title)
                  + "  ".join("{}={:.3f}[{:.3f},{:.3f}]".format(
                      nm, rv, lo, hi) for nm, rv, lo, hi in bars),
                  flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2), sharey=True)
    for ax, (title, _sk, deltas, x0) in zip(axes, GROUPS):
        xpos = np.arange(x0, 4)             # physical leaf->root positions
        ax.axhline(0.0, color="#b7bcc2", lw=0.8, ls=":", zorder=1)
        for aname, _raw in arms:
            vals = [max(-0.05, min(1.05, v))
                    for _nm, v, _l, _h in norm[aname][title]]
            los = [max(-0.05, min(1.05, l))
                   for _nm, _v, l, _h in norm[aname][title]]
            his = [max(-0.05, min(1.05, h))
                   for _nm, _v, _l, h in norm[aname][title]]
            color = "#2a78d6" if aname == "ours" else "#eb6834"
            ls = "-" if aname == "ours" else "--"
            ax.plot(xpos, vals, lw=1.8, ms=4.5, zorder=3, color=color,
                    ls=ls, marker="o", label=aname)
            ax.fill_between(xpos, los, his, color=color, alpha=0.10,
                            lw=0, zorder=2)
        ax.set_xticks(list(POS_LABELS.keys()))
        ax.set_xticklabels([POS_LABELS[k] for k in sorted(POS_LABELS)],
                           fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, axis="y", color="#e5e8ea", lw=0.6, zorder=0)
        ax.legend(fontsize=8, frameon=False, loc="lower left")
    axes[0].set_ylabel("fraction of the source predictive signal\n"
                       "retained (normalized to the source, R$_0$ = 1)",
                       fontsize=9)
    fig.suptitle("Predictive signal retention through recursive compression "
                 "-- host vs ours (same measurement protocol)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(args.out, dpi=150)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
