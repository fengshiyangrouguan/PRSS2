#!/usr/bin/env python3
"""Is the B2 probe able to detect a signal that is *identifiable given the
candidates*?  -- the positive control the earlier Z-only controls got wrong.

Why the earlier controls were invalid
-------------------------------------
The label is "which presentation position holds the true candidate".  A state
that perfectly memorises the true future NEIGHBOUR still cannot say where that
neighbour was randomly placed unless the candidate order is visible.  Formally:
let Z hold the two true future neighbours exactly; with the candidates permuted
independently the best Z-ONLY 4-class accuracy is 25%, while Z together with the
permuted candidates reaches 100%.  So "Z alone is at chance", and "the real
labels look like shuffled labels", are GUARANTEED by the design and say nothing
about whether the state carries predictive signal.

What this control does instead
------------------------------
On the REAL feature geometry (real candidate pairs, real E and C) plant a state
consistent with the candidate ordering,

    Z_planted = alpha * (E(c_s_true) + E(p_true)) + noise

so that <W_Z Z, E(c_s^a)> is large exactly at the true position.  The probe
family is unchanged and the protocol is the formal one: lambda is chosen by the
out-of-fold split INSIDE calibration and the audit block is scored once.  The
dose-response over alpha is the instrument's detection threshold -- what "is
this probe sensitive enough at this sample size" actually means.

Ablations: Z = pure noise should give J ~ 0, and a state built from the WRONG
candidates should not beat noise.

Usage:
    PYTHONPATH=<repo>/src:<repo>/scripts python scripts/diag_b2/probe_positive_control.py \
        --rows <...>/ours/retention_rows.pkl --data-dir <...>/uci --line Y_leaf
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
for _p in (str(SCRIPTS), str(SCRIPTS.parent / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import retention_probe_joint as rj          # noqa: E402
import audit_rows_schema as ars             # noqa: E402
from retention_probe_joint_run import _future_end, build_folds   # noqa: E402

LINE_NODES = {"Y_leaf": ("leaf", "a2"), "Y_a2": ("a2", "a1"),
              "Y_a1": ("a1", "root")}
LAMS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)


def _index(d, split, line):
    out = {}
    for r in d[split]:
        if r["line"] != line:
            continue
        out.setdefault(rj.pair_key(tuple(int(x) for x in r["pair_id"])), {})[
            int(r["phys"])] = r
    return out


def _align(im, keys, phys):
    """One row per key at position ``phys[0]`` plus the pair id."""
    rows = [im[k][phys[0]] for k in keys]
    return rows, [tuple(int(x) for x in r["pair_id"]) for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--data-name", default="uci")
    ap.add_argument("--line", default="Y_leaf")
    ap.add_argument("--phys", type=int, default=0)
    ap.add_argument("--svd-rank", type=int, default=16)
    ap.add_argument("--n-folds", type=int, default=3)
    ap.add_argument("--min-fold-rows", type=int, default=20)
    ap.add_argument("--alphas", default="0,0.25,0.5,1,2,4")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from rpbe.data.uci_link import UCILinkDataset
    d = pickle.load(open(args.rows, "rb"))
    man = {tuple(int(x) for x in m["pair_id"]): m for m in d["manifest"]}
    sk, pk = LINE_NODES[args.line]

    cal, aud = _index(d, "calib", args.line), _index(d, "audit", args.line)
    phys = sorted({p for v in cal.values() for p in v}
                  & {p for v in aud.values() for p in v})
    ck = sorted(k for k, v in cal.items() if set(v) == set(phys))
    ak = sorted(k for k, v in aud.items() if set(v) == set(phys))
    if args.phys not in phys:
        raise SystemExit("phys %d not present (have %s)" % (args.phys, phys))

    ds = UCILinkDataset(args.data_dir, data_name=args.data_name)
    cut = min(float(man[tuple(int(x) for x in r["pair_id"])]["t_root"])
              for r in d["calib"])
    keep = np.asarray(ds.train.timestamps, np.float64) < cut
    E, _ = rj.build_svd_encoder(ds.train.sources[keep],
                                ds.train.destinations[keep],
                                int(ds.n_nodes), rank=args.svd_rank,
                                random_state=0)
    Es = E / (np.sqrt((E ** 2).mean()) + 1e-12)     # unit-RMS embeddings

    def packed(im, keys):
        rows, pids = _align(im, keys, phys)
        sp, pp, ys, yp, C, Z = [], [], [], [], [], []
        for r, pid in zip(rows, pids):
            a, b, c, e = rj.ordered_pair_ids(man[pid], sk, pk)
            sp.append(a)
            pp.append(b)
            ys.append(c)
            yp.append(e)
            C.append(ars.ctx_shared(r["ctx"]))
            Z.append(np.asarray(r["keep"], np.float64))
        Phi = rj.phi_tensor(E, np.array(sp), np.array(pp))
        return (np.array(sp), np.array(pp), np.array(ys), np.array(yp),
                np.stack(C), np.stack(Z), Phi)

    sc, pc, ysc, ypc, Cc, Zc, Phic = packed(cal, ck)
    sa, pa, ysa, ypa, Ca, Za, Phia = packed(aud, ak)
    yc, ya = rj.joint_class(ysc, ypc), rj.joint_class(ysa, ypa)

    # the probe's folds, with the same future-time purge as the formal runner
    t_root_c = np.array([float(man[p]["t_root"]) for p in
                         [tuple(int(x) for x in cal[k][phys[0]]["pair_id"])
                          for k in ck]], np.float64)
    t_end_c = np.array([_future_end(man[p], sk, pk, None) for p in
                        [tuple(int(x) for x in cal[k][phys[0]]["pair_id"])
                         for k in ck]], np.float64)
    folds, attempts = build_folds(t_root_c, t_end_c, yc, args.line,
                                 args.n_folds, args.min_fold_rows)
    print("[geom] line=%s phys=%d calib=%d audit=%d classes=%s folds=%d"
          % (args.line, args.phys, len(yc), len(ya), rj.class_counts(yc),
             len(folds)), flush=True)
    print("[folds] attempts=%s" % (attempts,), flush=True)

    # ---- the FORMAL calib -> audit boundary must also be purged by future
    # time: a calibration row whose supervised future lands inside the audit
    # window leaks that window's events into the probe's training.
    audit_start = float(man[tuple(int(x) for x in
                         aud[ak[0]][phys[0]]["pair_id"])]["t_root"])
    cal_ok = t_end_c < audit_start
    print("[purge] audit starts at t_root=%.1f ; calib rows with "
          "t_future < that: %d / %d (max calib t_future=%.1f)"
          % (audit_start, int(cal_ok.sum()), len(cal_ok), t_end_c.max()),
          flush=True)
    if 0 < cal_ok.sum() < len(cal_ok):
        keep_idx = np.where(cal_ok)[0]
        t_root_c = t_root_c[keep_idx]
        t_end_c = t_end_c[keep_idx]
        yc = yc[keep_idx]
        Cc = Cc[keep_idx]
        Zc = Zc[keep_idx]
        Phic = Phic[keep_idx]
        sc, pc, ysc, ypc = sc[keep_idx], pc[keep_idx], ysc[keep_idx], ypc[keep_idx]
        folds, attempts2 = build_folds(t_root_c, t_end_c, yc, args.line,
                                       args.n_folds, args.min_fold_rows)
        print("[purge] after filtering: calib=%d folds=%d attempts=%s"
              % (len(yc), len(folds), attempts2), flush=True)

    base, oof_base, _ = rj.select_joint(Phic, Cc, np.zeros((len(yc), 1)), yc,
                                        folds, lams=LAMS, use_state=False)

    def probe(tag, Zc_, Za_):
        full, oof_full, diag = rj.select_joint(Phic, Cc, Zc_, yc, folds,
                                               lams=LAMS, use_state=True)
        n0 = float(rj.joint_row_nll(base, Phia, Ca, None, ya).mean())
        n1 = float(rj.joint_row_nll(full, Phia, Ca, Za_, ya).mean())
        print("  %-46s J = %+.6f bits  (fallback=%s, lam=%g)"
              % (tag, (n0 - n1) / np.log(2.0), diag["used_fallback"],
                 full["lam"]), flush=True)

    rng = np.random.RandomState(args.seed)
    r = Es.shape[1]          # candidate-embedding width; Z is 172-d

    def plant(a, s_, p_, ys_, yp_, shape):
        """Noise everywhere plus a*E(true candidate) in the first r dims.

        phi_ab starts with E(c_s^a), so a planted candidate identity in Z[:, :r]
        gives W_Z something to align with exactly at the true position -- a
        signal that IS identifiable given the candidates.
        """
        Z = rng.randn(*shape)
        Z[:, :r] += a * (Es[s_[np.arange(len(s_)), ys_]] +
                         Es[p_[np.arange(len(p_)), yp_]])
        return Z

    def plant_random(a, s_, p_, shape):
        """Ablation: plant a RANDOM node id -- uncorrelated with the label.

        (Planting the OTHER candidate's embedding is NOT an ablation: phi
        carries both candidates, so "the other one" is a deterministic function
        of the label and a linear family just absorbs the sign.)
        """
        Z = rng.randn(*shape)
        rid = rng.randint(0, Es.shape[0] - 1, size=len(s_)) + 1
        Z[:, :r] += a * Es[rid]
        return Z

    for a in [float(x) for x in args.alphas.split(",")]:
        if a == 0.0:
            probe("Z = pure noise", rng.randn(*Zc.shape), rng.randn(*Za.shape))
        else:
            probe("Z = %.2f*E(true candidates)+noise" % a,
                  plant(a, sc, pc, ysc, ypc, Zc.shape),
                  plant(a, sa, pa, ysa, ypa, Za.shape))
    probe("Z = 2.0*E(random nodes)+noise  [ablation]",
          plant_random(2.0, sc, pc, Zc.shape),
          plant_random(2.0, sa, pa, Za.shape))
    probe("REAL state", Zc, Za)


if __name__ == "__main__":
    main()
