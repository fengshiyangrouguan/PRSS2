#!/usr/bin/env python3
"""Rows / schema audit for the B2 joint probe (spec 2026-09-14 §2.3, §10).

Before the joint probe may treat existing ``retention_rows.pkl`` files as
reusable, the fields it consumes must be IDENTICAL across arms for every shared
row.  This script proves (or fails) that, and never assumes it.

Shared fields checked across arms, joined on ``(split, pair_key, line, phys)``:

  * ``pair_id`` (globally unique tuple, compared via a string key -- never
    ``np.unique`` on a tuple array, which flattens),
  * ``t_root`` and the branch ``nodes``,
  * ``joint_label`` = 2*Y_s + Y_p (also cross-checked against the row's own
    ``y_s``/``y_p``),
  * the ORDERED candidate pair per node (presented, other) and the raw
    pos/neg candidate ids,
  * ``future_event_ids`` and ``candidate_seed`` per node,
  * ``C_shared``: the model-independent 32-d context.  The rows' ``ctx`` is 80-d
    and its blocks ``[0:16]``, ``[24:40]``, ``[48:64]`` are projections of the
    arm's own neighbour hidden states, i.e. NOT shared; only ``[16:24]``,
    ``[40:48]``, ``[64:80]`` (path edge features/times + scalars) are kept,
  * the ``probe split`` membership (calib / audit / head) and the layout.

State fields (``keep``/``rem``) are model-dependent by design and are NOT
compared; only the shared inputs are.

Usage:
    python scripts/audit_rows_schema.py --rows A.pkl B.pkl [C.pkl ...] \
        [--out report.json] [--label arm_a=path]
Exit code is nonzero if any shared field differs across arms.
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
import retention_probe_joint as rj  # noqa: E402

CTX_DIM = 80
CTX_DROPPED = ((0, 16), (24, 40), (48, 64))    # model-dependent neighbour means
CTX_SHARED_BLOCKS = ((16, 24), (40, 48), (64, 80))
SHARED_DIM = sum(b - a for a, b in CTX_SHARED_BLOCKS)

NODE_KEYS = ("leaf", "a2", "a1", "root")


def ctx_shared(ctx):
    """The 32-d model-independent context (drop the neighbour hidden blocks)."""
    ctx = np.asarray(ctx, dtype=np.float64).reshape(-1)
    if ctx.size != CTX_DIM:
        raise ValueError("ctx has {} dims, expected {}".format(ctx.size, CTX_DIM))
    return np.concatenate([ctx[a:b] for a, b in CTX_SHARED_BLOCKS])


def _i(x):
    return int(x)


def arm_records(d):
    """Per-row shared-field records keyed by (split, pair_key, line, phys)."""
    man = {tuple(int(x) for x in m["pair_id"]): m for m in d.get("manifest", [])}
    recs = {}
    intra = {"rows_missing_manifest": 0, "label_conflict": 0}
    for split in ("calib", "audit", "head"):
        for r in d.get(split, []):
            pid = tuple(int(x) for x in r["pair_id"])
            m = man.get(pid)
            if m is None:
                intra["rows_missing_manifest"] += 1
                continue
            s_key, p_key = {"Y_leaf": ("leaf", "a2"), "Y_a2": ("a2", "a1"),
                            "Y_a1": ("a1", "root")}[r["line"]]
            os_, op_, ys, yp = rj.ordered_pair_ids(m, s_key, p_key)
            if (ys, yp) != (int(r["y_s"]), int(r["y_p"])):
                intra["label_conflict"] += 1
            key = (split, rj.pair_key(pid), r["line"], int(r["phys"]))
            recs[key] = {
                "pair_id": list(pid),
                "t_root": float(m["t_root"]),
                "nodes": {k: _i(m["nodes"][k]) for k in NODE_KEYS},
                "joint_label": int(rj.joint_class(ys, yp)),
                "ordered_s": list(os_),
                "ordered_p": list(op_),
                "pos_cand": {k: _i(m["pos_cand"][k]) for k in NODE_KEYS},
                "neg_cand": {k: _i(m["neg_cand"][k]) for k in NODE_KEYS},
                "presented": {k: _i(m["presented"][k]) for k in NODE_KEYS},
                "future_event_id": {k: _i(m["pos_future_event_id"][k])
                                    for k in NODE_KEYS},
                "candidate_seed": {k: _i(m["candidate_seed"][k])
                                   for k in NODE_KEYS},
                "c_shared": ctx_shared(r["ctx"]),
            }
    return recs, intra


def compare(arm_recs, labels, atol=0.0):
    """Field-by-field comparison over the intersection of row keys."""
    keys = None
    for recs in arm_recs:
        k = set(recs)
        keys = k if keys is None else (keys & k)
    keys = sorted(keys or [])
    fields = ["pair_id", "t_root", "nodes", "joint_label", "ordered_s",
              "ordered_p", "pos_cand", "neg_cand", "presented",
              "future_event_id", "candidate_seed", "c_shared"]
    mism = {f: 0 for f in fields}
    examples = []
    for k in keys:
        base = arm_recs[0][k]
        for f in fields:
            vals = [recs[k][f] for recs in arm_recs]
            if f == "c_shared":
                bad = any(not np.allclose(v, vals[0], atol=1e-12, rtol=1e-9)
                          for v in vals[1:])
            else:
                bad = any(v != vals[0] for v in vals[1:])
            if bad:
                mism[f] += 1
                if len(examples) < 8:
                    examples.append({"key": list(k), "field": f,
                                     "values": [str(v) for v in vals]})
    # keys present in only some arms
    all_keys = set()
    for recs in arm_recs:
        all_keys |= set(recs)
    n_only = len(all_keys) - len(keys)
    ok = (sum(mism.values()) == 0) and (n_only == 0)
    return {"n_common_rows": len(keys), "n_keys_not_in_all_arms": n_only,
            "n_rows_per_arm": [len(r) for r in arm_recs],
            "mismatch_counts": mism, "examples": examples, "ok": bool(ok),
            "labels": labels}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--json", action="store_true",
                    help="print the full JSON report")
    args = ap.parse_args()

    arm_recs, metas, intras = [], [], []
    for p in args.rows:
        with open(p, "rb") as f:
            d = pickle.load(f)
        recs, intra = arm_records(d)
        arm_recs.append(recs)
        intras.append(intra)
        m = d.get("meta", {})
        metas.append({k: m.get(k) for k in (
            "model_kind", "arm", "seed", "n_layers", "n_neighbors", "bs",
            "data_name", "dataset_hash", "manifest_sha", "manifest_in",
            "memory_parity")})
    labels = [Path(p).parent.name + "/" + Path(p).name for p in args.rows]

    # meta-level agreement that must hold for any cross-arm comparison
    meta_keys = ["n_layers", "n_neighbors", "bs", "data_name", "dataset_hash",
                 "manifest_sha"]
    meta_mism = {k: [str(m.get(k)) for m in metas] for k in meta_keys
                 if len({str(m.get(k)) for m in metas}) > 1}

    report = {"rows": args.rows, "labels": labels, "meta": metas,
              "intra_arm": intras,
              "meta_mismatch": meta_mism,
              "ctx_shared_dim": SHARED_DIM,
              "compare": compare(arm_recs, labels)}
    report["ok"] = bool(report["compare"]["ok"]
                        and not meta_mism
                        and all(i["label_conflict"] == 0 for i in intras))

    summary = [
        "rows/schema audit: {}".format("OK" if report["ok"] else "FAILED"),
        "  C_shared dim: {} (dropped ctx blocks {} )".format(
            SHARED_DIM, list(CTX_DROPPED)),
        "  common rows across arms: {}".format(
            report["compare"]["n_common_rows"]),
        "  keys not present in every arm: {}".format(
            report["compare"]["n_keys_not_in_all_arms"]),
    ]
    if meta_mism:
        summary.append("  META mismatch: {}".format(
            {k: v for k, v in meta_mism.items()}))
    bad = {k: v for k, v in report["compare"]["mismatch_counts"].items() if v}
    if bad:
        summary.append("  FIELD mismatches: {}".format(bad))
        for ex in report["compare"]["examples"][:3]:
            summary.append("    e.g. {} {} -> {}".format(
                ex["key"], ex["field"], ex["values"]))
    print("\n".join(summary), flush=True)

    out = args.out or str(Path(args.rows[0]).parent / "rows_schema_audit.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print("wrote", out, flush=True)
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
