#!/usr/bin/env python3
"""Offline v3 nested mechanism-retention statistics from a retention_rows.pkl.

Consumes the pkl written by audit_retention_v2 (rows + manifest) and computes,
per source line (Y_leaf / Y_a2 / Y_a1), the nested source-contribution
information J_{s->k} and retention R_{s->k}=J_{s->k}/J_{s->s}, plus a
permutation null for the source gate.  Pure numpy/sklearn -- NO model, NO torch,
NO re-extraction (the existing pkl is enough; this is the "offline mechanism
stats" half of the two-track split).

Fairness:
  * C is the SHARED-ONLY context (the three model-dependent other-neighbor
    blocks are zeroed) -> identical across arms;
  * the candidate PAIR (presented vs the other of pos/neg) is fed with the
    manifest's swap label (which one is the true future destination);
  * the source position uses Delta_{s->s} = U_s; downstream uses keep - remove.
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import retention_stats as rs  # noqa: E402

# model-dependent blocks of C (other-neighbor hidden-state projections)
CTX_MODEL_DEP_BLOCKS = ((0, 16), (24, 40), (48, 64))
LINE_NODES = {"Y_leaf": ("leaf", "a2"), "Y_a2": ("a2", "a1"),
              "Y_a1": ("a1", "root")}
ORIGIN_PHYS = {"Y_leaf": 0, "Y_a2": 1, "Y_a1": 2}


def ctx_shared(ctx):
    o = np.array(ctx, dtype=np.float64, copy=True)
    for a, b in CTX_MODEL_DEP_BLOCKS:
        o[a:b] = 0.0
    return o


def node_emb(nid, d_emb=8, seed=20260909):
    """Arm-shared frozen candidate encoder E(node) (deterministic)."""
    g = np.random.RandomState((int(nid) * 2654435761 + int(seed)) & 0xFFFFFFFF)
    return g.normal(0.0, 1.0, d_emb) / np.sqrt(d_emb)


def _pair_arrays(rows, man, line):
    """pair_id -> {C, qs, qp, ys, yp, D:{phys:Delta}, t} for one line."""
    sk, pk = LINE_NODES[line]
    bypair = {}
    for r in rows:
        if r.get("line") != line:
            continue
        pid = tuple(int(x) for x in r["pair_id"])
        m = man.get(pid)
        if m is None:
            continue
        e = bypair.get(pid)
        if e is None:
            pres_s = int(m["presented"][sk]); pos_s = int(m["pos_cand"][sk])
            neg_s = int(m["neg_cand"][sk])
            oth_s = neg_s if pres_s == pos_s else pos_s
            pres_p = int(m["presented"][pk]); pos_p = int(m["pos_cand"][pk])
            neg_p = int(m["neg_cand"][pk])
            oth_p = neg_p if pres_p == pos_p else pos_p
            e = {"C": ctx_shared(r["ctx"]),
                 "qs": node_emb(pres_s) - node_emb(oth_s),
                 "qp": node_emb(pres_p) - node_emb(oth_p),
                 "ys": int(r["y_s"]), "yp": int(r["y_p"]),
                 "t": float(m["t_root"]), "D": {}}
            bypair[pid] = e
        e["D"][int(r["phys"])] = (np.asarray(r["keep"], np.float64)
                                  - np.asarray(r["rem"], np.float64))
    return bypair


def _stack(bypair, pids, phys_list):
    C = np.stack([bypair[p]["C"] for p in pids])
    qs = np.stack([bypair[p]["qs"] for p in pids])
    qp = np.stack([bypair[p]["qp"] for p in pids])
    ys = np.array([bypair[p]["ys"] for p in pids])
    yp = np.array([bypair[p]["yp"] for p in pids])
    D = {ph: np.stack([bypair[p]["D"][ph] for p in pids]) for ph in phys_list}
    t = np.array([bypair[p]["t"] for p in pids])
    return C, qs, qp, ys, yp, D, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-null", type=int, default=50)
    ap.add_argument("--ranks", default="4,8,16")
    ap.add_argument("--lam", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    d = pickle.load(open(args.rows, "rb"))
    man = {tuple(int(x) for x in m["pair_id"]): m for m in d["manifest"]}
    ranks = tuple(int(x) for x in args.ranks.split(","))
    calib_rows, audit_rows = d["calib"], d["audit"]

    report = {"rows_file": args.rows, "meta": d.get("meta", {}),
              "n_calib_rows": len(calib_rows), "n_audit_rows": len(audit_rows),
              "sources": {}}

    for line in ("Y_leaf", "Y_a2", "Y_a1"):
        cb = _pair_arrays(calib_rows, man, line)
        ab = _pair_arrays(audit_rows, man, line)
        cps, aps = sorted(cb), sorted(ab)
        if not cps or not aps:
            report["sources"][line] = {"error": "no pairs"}
            continue
        # calib and audit pairs are DISJOINT time blocks (fit on calib,
        # evaluate on audit) -- only the POSITION set must agree.
        phys_all = sorted(set.intersection(
            *[set(cb[p]["D"]) for p in cps],
            *[set(ab[p]["D"]) for p in aps]))
        origin = ORIGIN_PHYS[line]
        if origin not in phys_all:
            report["sources"][line] = {"error": "origin phys missing"}
            continue
        Cc, qsc, qpc, ysc, ypc, Dc, tc = _stack(cb, cps, phys_all)
        Ca, qsa, qpa, ysa, ypa, Da, ta = _stack(ab, aps, phys_all)
        curve = rs.nested_curve(Cc, qsc, qpc, ysc, ypc,
                                Ca, qsa, qpa, ysa, ypa,
                                Dc, Da, origin_pos=origin, ranks=ranks,
                                lam=args.lam, seed=args.seed)
        # permutation null on the source position (shuffle Delta in time blocks)
        nb = 8
        strata = np.digitize(ta, np.quantile(ta, np.linspace(0, 1, nb + 1)[1:-1]))
        null = rs.nested_null_source(
            Cc, qsc, qpc, ysc, ypc, Ca, qsa, qpa, ysa, ypa,
            Dc[origin], Da[origin], strata, ranks=ranks, lam=args.lam,
            seed=args.seed + 3, n_null=int(args.n_null))
        q95 = float(np.percentile(null, 95)) if null.size else float("nan")
        j0 = curve["J_source"]
        gate = bool(j0 is not None and j0 > q95 and j0 > 0)
        pts = []
        for ph in phys_all:
            r = curve["points"][ph]
            pts.append({"phys": int(ph), "J": float(r["J"]),
                        "R": float(r["R"]), "rank": int(r["rank"]),
                        "nll_base": float(r["nll_base"]),
                        "nll_full": float(r["nll_full"])})
        report["sources"][line] = {
            "origin_phys": int(origin), "J_source": float(j0) if j0 is not None else None,
            "null_q95": q95, "null_median": float(np.median(null)) if null.size else None,
            "gate_source_gt_null": gate,
            "n_pairs_calib": len(cps), "n_pairs_audit": len(aps),
            "points": pts}

    out = args.out or (str(Path(args.rows).parent / "retention_v3_nested.json"))
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(json.dumps(report, indent=2, default=str))
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
