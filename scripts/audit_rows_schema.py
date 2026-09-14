#!/usr/bin/env python3
"""Rows / schema audit for the B2 joint probe (spec 2026-09-14 §2.3, §10).

Before the joint probe may treat existing ``retention_rows.pkl`` files as
reusable, the fields it consumes must be IDENTICAL across arms for every shared
row.  This module proves (or fails) that; it never assumes it.  The same
``full_audit`` is used by this CLI and by the probe runner, so the runner cannot
bypass a check the CLI would have raised.

Shared fields checked across arms, joined on ``(split, pair_key, line, phys)``:

  * ``pair_id`` (globally unique tuple, compared via a string key -- never
    ``np.unique`` on a tuple array, which flattens),
  * ``t_root`` and the branch ``nodes``,
  * the canonical ``joint_label`` = 2*Y_s + Y_p, cross-checked against the
    row's own ``y_s``/``y_p`` under the declared/inferred label convention,
  * the ORDERED candidate pair per node (presented, other) and the raw pos/neg
    candidate ids,
  * ``future_event_ids`` and ``candidate_seed`` per node,
  * ``C_shared``: the model-independent 32-d context.  The rows' ``ctx`` is 80-d
    and its blocks ``[0:16]``, ``[24:40]``, ``[48:64]`` are projections of the
    arm's own neighbour hidden states, i.e. NOT shared; only ``[16:24]``,
    ``[40:48]``, ``[64:80]`` (path edge features/times + scalars) are kept,
  * the probe split membership (calib / audit / head) and the layout.

Label convention: the canonical label is the PRESENTATION POSITION of the true
candidate.  Legacy ``audit_retention_v2`` rows store ``Y = 1`` iff the presented
candidate is the true one, i.e. ``Y_canonical = 1 - Y_legacy`` -- resolved
explicitly from ``meta.label_kind`` or inferred, never guessed silently.

State fields (``keep``/``rem``) are model-dependent by design and are NOT
compared; only the shared inputs are.

Usage:
    python scripts/audit_rows_schema.py --rows A.pkl B.pkl [C.pkl ...] \
        [--out report.json] [--print-json]
Exit code is nonzero if anything fails.
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
LINE_NODES = {"Y_leaf": ("leaf", "a2"), "Y_a2": ("a2", "a1"),
              "Y_a1": ("a1", "root")}
SPLITS = ("calib", "audit", "head")
META_KEYS = ("n_layers", "n_neighbors", "bs", "data_name", "dataset_hash",
             "manifest_sha", "layout")
LABEL_KIND_ALIASES = {"canonical": rj.CANONICAL_LABEL_KIND,
                      "legacy": rj.LEGACY_LABEL_KIND}
LABEL_KIND_CHOICES = list(rj.LABEL_KINDS) + list(LABEL_KIND_ALIASES)
COMPARE_FIELDS = ("pair_id", "t_root", "nodes", "joint_label", "ordered_s",
                  "ordered_p", "pos_cand", "neg_cand", "presented",
                  "future_event_id", "future_event_time", "candidate_seed",
                  "c_shared")


def ctx_shared(ctx):
    """The 32-d model-independent context (drop the neighbour hidden blocks)."""
    ctx = np.asarray(ctx, dtype=np.float64).reshape(-1)
    if ctx.size != CTX_DIM:
        raise ValueError("ctx has {} dims, expected {}".format(ctx.size, CTX_DIM))
    return np.concatenate([ctx[a:b] for a, b in CTX_SHARED_BLOCKS])


def arm_identity(d):
    """Arm identity from the checkpoint meta -- never from the file order."""
    m = d.get("meta", {})
    arm = m.get("arm")
    sha = m.get("ckpt_sha256")
    seed = m.get("seed")
    if not arm or not sha:
        raise ValueError(
            "checkpoint meta is missing arm/ckpt_sha256 (have {})".format(
                {k: m.get(k) for k in ("arm", "seed", "ckpt_sha256",
                                       "model_kind")}))
    return "{}|seed{}|{}".format(arm, "NA" if seed is None else seed, sha)


def resolve_label_kind(d, declared=None):
    """(label_kind, problems) for one arm's stored row labels."""
    if declared is not None:
        declared = LABEL_KIND_ALIASES.get(declared, declared)
        if declared not in rj.LABEL_KINDS:
            return None, ["unknown declared label_kind {!r}".format(declared)]
        return declared, []
    meta_kind = d.get("meta", {}).get("row_label_kind") or \
        d.get("meta", {}).get("label_kind")
    if meta_kind in rj.LABEL_KINDS:
        return meta_kind, []
    man, _dups = _manifest_index(d)
    n_can = n_leg = 0
    for split in SPLITS:
        for r in d.get(split, []):
            m = man.get(tuple(int(x) for x in r["pair_id"]))
            if m is None:
                continue
            sk, pk = LINE_NODES[r["line"]]
            cs, cp = rj.canonical_bits(m, sk), rj.canonical_bits(m, pk)
            got = (int(r["y_s"]), int(r["y_p"]))
            if got == (cs, cp):
                n_can += 1
            elif got == (1 - cs, 1 - cp):
                n_leg += 1
    if n_can and n_leg:
        return None, ["mixed label conventions: {} canonical, {} legacy rows"
                      .format(n_can, n_leg)]
    if not n_can and not n_leg:
        return None, ["cannot infer the label convention (no rows matched the "
                      "manifest)"]
    return (rj.CANONICAL_LABEL_KIND if n_can else rj.LEGACY_LABEL_KIND), []


def _manifest_index(d, problems=None):
    """pair_id -> manifest row, refusing silently-overwritten duplicates."""
    man, dups = {}, []
    for m in d.get("manifest", []):
        pid = tuple(int(x) for x in m["pair_id"])
        if pid in man:
            dups.append(pid)
        man[pid] = m
    return man, dups


def arm_records(d, label_kind):
    """Per-row shared-field records keyed by (split, pair_key, line, phys)."""
    problems = []
    man, dups = _manifest_index(d)
    if dups:
        problems.append("manifest has {} duplicate pair_id(s), e.g. {}".format(
            len(dups), list(dups[:3])))
    recs = {}
    intra = {"label_kind": label_kind, "n_rows": 0,
             "rows_missing_manifest": 0, "label_conflict": 0,
             "duplicate_row_keys": 0, "split_overlap": 0,
             "head_overlap": 0,
             "manifest_duplicate_pair_ids": len(dups)}
    seen = {}
    for split in SPLITS:
        for r in d.get(split, []):
            intra["n_rows"] += 1
            pid = tuple(int(x) for x in r["pair_id"])
            m = man.get(pid)
            if m is None:
                intra["rows_missing_manifest"] += 1
                continue
            sk, pk = LINE_NODES[r["line"]]
            os_, op_, ys, yp = rj.ordered_pair_ids(m, sk, pk)
            if label_kind is not None and (
                    rj.row_label_to_canonical(r["y_s"], label_kind),
                    rj.row_label_to_canonical(r["y_p"], label_kind)) != (ys, yp):
                intra["label_conflict"] += 1
            bare = (rj.pair_key(pid), r["line"], int(r["phys"]))
            if bare in seen:
                prev = seen[bare]
                if prev == split:
                    intra["duplicate_row_keys"] += 1
                elif {prev, split} == {"calib", "audit"}:
                    # leak-critical: the probe fits on calib and scores audit
                    intra["split_overlap"] += 1
                else:
                    # the optional head-calibration block re-scans the stream
                    # head by construction; it is not used by the probe, so an
                    # overlap with calib is informational, not a failure
                    intra["head_overlap"] += 1
            else:
                seen[bare] = split
            recs[(split, rj.pair_key(pid), r["line"], int(r["phys"]))] = {
                "pair_id": list(pid),
                "t_root": float(m["t_root"]),
                "nodes": {k: int(m["nodes"][k]) for k in NODE_KEYS},
                "joint_label": int(rj.joint_class(ys, yp)),
                "ordered_s": list(os_),
                "ordered_p": list(op_),
                "pos_cand": {k: int(m["pos_cand"][k]) for k in NODE_KEYS},
                "neg_cand": {k: int(m["neg_cand"][k]) for k in NODE_KEYS},
                "presented": {k: int(m["presented"][k]) for k in NODE_KEYS},
                "future_event_id": {k: int(m["pos_future_event_id"][k])
                                    for k in NODE_KEYS},
                "future_event_time": (
                    {k: float(m["future_event_time"][k]) for k in NODE_KEYS}
                    if m.get("future_event_time") is not None else None),
                "candidate_seed": {k: int(m["candidate_seed"][k])
                                   for k in NODE_KEYS},
                "c_shared": ctx_shared(r["ctx"]),
            }
    return recs, intra


def compare(arm_recs, labels, atol=1e-12):
    """Field-by-field comparison over the intersection of row keys."""
    keys = None
    for recs in arm_recs:
        k = set(recs)
        keys = k if keys is None else (keys & k)
    keys = sorted(keys or [])
    mism = {f: 0 for f in COMPARE_FIELDS}
    examples = []
    for k in keys:
        base = arm_recs[0][k]
        for f in COMPARE_FIELDS:
            vals = [recs[k][f] for recs in arm_recs]
            if f == "c_shared":
                bad = any(not np.allclose(v, vals[0], atol=atol, rtol=1e-9)
                          for v in vals[1:])
            else:
                bad = any(v != vals[0] for v in vals[1:])
            if bad:
                mism[f] += 1
                if len(examples) < 8:
                    examples.append({"key": list(k), "field": f,
                                     "values": [str(v) for v in vals]})
    all_keys = set()
    for recs in arm_recs:
        all_keys |= set(recs)
    n_only = len(all_keys) - len(keys)
    return {"n_common_rows": len(keys), "n_keys_not_in_all_arms": n_only,
            "n_rows_per_arm": [len(r) for r in arm_recs],
            "mismatch_counts": mism, "examples": examples,
            "ok": bool(sum(mism.values()) == 0 and n_only == 0),
            "labels": labels}


def full_audit(arms, declared_label_kind=None):
    """The one audit used by both this CLI and the probe runner.

    Fails (``ok`` False) on ANY of: label-convention problems, intra-arm
    ``label_conflict``, missing-manifest rows, duplicate row keys, split
    overlap, meta/layout disagreement, duplicate arm identity, or any cross-arm
    field mismatch.
    """
    per_arm, recs, labels, problems = [], [], [], []
    for i, d in enumerate(arms):
        try:
            labels.append(arm_identity(d))
        except ValueError as e:
            labels.append("arm{}".format(i))
            problems.append("arm {}: {}".format(i, e))
        lk, errs = resolve_label_kind(d, declared_label_kind)
        if lk is None:
            problems.append("arm {}: {}".format(i, "; ".join(errs)))
        r, intra = arm_records(d, lk)
        recs.append(r)
        per_arm.append(intra)

    if len(set(labels)) != len(labels):
        problems.append("duplicate arm identities: {}".format(labels))
    meta_mism = {}
    for k in META_KEYS:
        vals = {str(d.get("meta", {}).get(k)) for d in arms}
        if len(vals) > 1:
            meta_mism[k] = [str(d.get("meta", {}).get(k)) for d in arms]
    if meta_mism:
        problems.append("meta/layout mismatch: {}".format(sorted(meta_mism)))
    for i, intra in enumerate(per_arm):
        for k in ("label_conflict", "rows_missing_manifest",
                  "duplicate_row_keys", "split_overlap",
                  "manifest_duplicate_pair_ids"):
            if intra[k]:
                problems.append("arm {} {}={}".format(i, k, intra[k]))

    cmp_rep = compare(recs, labels)
    return {"ok": bool(not problems and cmp_rep["ok"]),
            "labels": labels, "arm_identity": labels, "per_arm": per_arm,
            "problems": problems, "meta_mismatch": meta_mism,
            "compare": cmp_rep, "ctx_shared_dim": SHARED_DIM}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--row-label-kind", default=None,
                    choices=LABEL_KIND_CHOICES,
                    help="override the stored-row label convention")
    ap.add_argument("--print-json", action="store_true")
    args = ap.parse_args()

    arms = []
    for p in args.rows:
        with open(p, "rb") as f:
            arms.append(pickle.load(f))
    rep = full_audit(arms, declared_label_kind=args.row_label_kind)
    rep["rows"] = args.rows

    summary = ["rows/schema audit: {}".format("OK" if rep["ok"] else "FAILED"),
               "  C_shared dim: {} (dropped ctx blocks {})".format(
                   rep["ctx_shared_dim"], list(CTX_DROPPED)),
               "  arms: {}".format(rep["labels"]),
               "  label kinds: {}".format(
                   [a["label_kind"] for a in rep["per_arm"]]),
               "  common rows across arms: {}".format(
                   rep["compare"]["n_common_rows"])]
    if rep["problems"]:
        summary.append("  PROBLEMS:")
        summary += ["    - {}".format(p) for p in rep["problems"]]
    bad = {k: v for k, v in rep["compare"]["mismatch_counts"].items() if v}
    if bad:
        summary.append("  FIELD mismatches: {}".format(bad))
        for ex in rep["compare"]["examples"][:3]:
            summary.append("    e.g. {} {} -> {}".format(
                ex["key"], ex["field"], ex["values"]))
    print("\n".join(summary), flush=True)

    out = args.out or str(Path(args.rows[0]).parent / "rows_schema_audit.json")
    with open(out, "w") as f:
        json.dump(rep, f, indent=2, default=str)
    print("wrote", out, flush=True)
    if args.print_json:
        print(json.dumps(rep, indent=2, default=str))
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
