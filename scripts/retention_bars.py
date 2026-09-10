#!/usr/bin/env python3
"""Absolute per-position predictive strength toward the ROOT boundary.

Task-only TGN only; computed directly from a cached retention_rows.pkl (no
model rerun).  All three source depths share ONE prediction target -- the root
two-level boundary

    S_root = (Y_{a1}, Y_root)                 (phi-composed, 256-d)

and every path position is scored by the absolute squared-canonical strength
between its state and S_root:

    J(H_k, S_root) = tr[(C_ss+eps I)^-1 C_sh (C_hh+lam I)^-1 C_hs]

No normalization to 1 at the source, no J*R splicing: each bar is the raw
strength.  Three zero-baseline groups (bars are the states along that source's
leaf-to-root path):
    3-hop source (U0):  U0, Z1, Z2, Z3      (4 bars)
    2-hop source (Z1):  Z1, Z2, Z3          (3 bars)
    1-hop source (Z2):  Z2, Z3              (2 bars)

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
    """Same fixed 128-d future witness as the audit (deterministic)."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="audit", choices=["audit", "calib"])
    ap.add_argument("--n-bootstrap", type=int, default=200)
    ap.add_argument("--lam", type=float, default=1e-2)
    args = ap.parse_args()

    with open(args.rows, "rb") as f:
        rd = pickle.load(f)
    rows = rd[args.split]

    # unified ROOT boundary target for every source: (Y_a1, Y_root)
    S_root = np.concatenate([_y_phi(rows, "Y_a1"), _y_phi(rows, "Y_root")],
                            axis=1)
    state_key = {"U0": "u0", "Z1": "z1", "Z2": "z2", "Z3": "z3"}
    states = {name: np.stack([r[key] for r in rows])
              for name, key in state_key.items()}

    def J(X, idx):
        return rs.centered_sq_corr_strength(X[idx], S_root[idx],
                                            lam=args.lam)

    idx_all = np.arange(len(rows))
    strength = {}
    for name, X in states.items():
        strength[name] = float(J(X, idx_all))
        print("[bar] {:>2}  J(state, S_root)={:.5f}".format(
            name, strength[name]), flush=True)

    groups = [
        ("3-hop source (U0)", ["U0", "Z1", "Z2", "Z3"]),
        ("2-hop source (Z1)", ["Z1", "Z2", "Z3"]),
        ("1-hop source (Z2)", ["Z2", "Z3"]),
    ]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    xs = {"U0": 0, "Z1": 1, "Z2": 2, "Z3": 3}
    colors = {"U0": "#2a78d6", "Z1": "#eb6834", "Z2": "#1baf7a",
              "Z3": "#8a5cd6"}
    for ax, (title, names) in zip(axes, groups):
        xpos = np.arange(len(names))
        vals = [strength[nm] for nm in names]
        ax.bar(xpos, vals, color=[colors[nm] for nm in names], zorder=3)
        ax.set_xticks(xpos)
        ax.set_xticklabels(["{}\n({})".format(nm, "leaf" if nm == "U0"
                           else "root" if nm == "Z3" else "anc")
                            for nm in names], fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.grid(True, axis="y", color="#e5e8ea", lw=0.6, zorder=0)
        ax.set_ylim(bottom=0.0)          # zero baseline
    axes[0].set_ylabel("J(state, S_root)  absolute squared-canonical strength\n"
                       "toward the shared root boundary (Task-only)", fontsize=9)
    fig.suptitle("Absolute predictive strength toward the same root boundary "
                 "S_root=(Y_a1,Y_root) -- no source normalization", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(args.out, dpi=150)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
