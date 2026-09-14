#!/usr/bin/env python3
"""Offline B2 joint-probe runner: rows pkl(s) -> J, R, RootGain.

Consumes the existing ``retention_rows.pkl`` files (no model, no GPU, no
re-extraction) and produces the fixed statistics of the reviewed spec:

    J_k      = mean_i (NLL0_i - NLL1_{k,i}) / ln2         [bits / joint query]
    R_k      = J_k / J_source                             (paired bootstrap)
    RootGain = 1000 * J_root                              [millibits]

Fixed design:
  * ``C_shared`` is the 32-d model-independent context (audit_rows_schema);
  * the candidate features ``phi_ab`` and ``C_shared`` are per-PAIR -- they do
    not depend on the downstream position, only the state ``Z_k`` does;
  * the base probe is fit ONCE per source line and shared across arms and
    positions; the full probe is fit per arm/line/position;
  * lambda is chosen on an inner expanding-window time split of the calib block
    with a purge gap; the audit block selects nothing;
  * each full probe's lambda is compared against the base reproduced with
    ``W_Z = 0``, so a full probe can never lose to the base by construction;
  * ONE block-bootstrap index set (root-event time blocks) is shared by every
    arm, position and metric, and R is computed inside each replicate.

The runner refuses to proceed unless the rows/schema audit passes across arms.

Usage:
    python scripts/retention_probe_joint_run.py \
        --rows armA.pkl armB.pkl --labels ours taskonly \
        --data-dir /path/to/processed --data-name uci --out-dir ./joint_out
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
for _p in (str(SRC), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import retention_probe_joint as rj          # noqa: E402
import audit_rows_schema as ars             # noqa: E402
from retention_probe_joint import build_svd_encoder  # noqa: E402

LINES = ("Y_leaf", "Y_a2", "Y_a1")
LINE_NODES = {"Y_leaf": ("leaf", "a2"), "Y_a2": ("a2", "a1"),
              "Y_a1": ("a1", "root")}
ORIGIN_PHYS = {"Y_leaf": 0, "Y_a2": 1, "Y_a1": 2}
PRI_PHYS = {"Y_leaf": (1, 2, 3), "Y_a2": (2, 3), "Y_a1": (3,)}


# ------------------------------------------------------------------ helpers
def expanding_folds(times, n_folds=3, purge_frac=0.02):
    """(fit_idx, tune_idx): fit = expanding prefix, tune = the next slice.

    A purge gap separates fit from tune so that supervised futures of the fit
    rows cannot cross into the tune block.
    """
    order = np.argsort(np.asarray(times, np.float64), kind="stable")
    n = len(order)
    if n < 40:
        h = max(1, n // 2)
        return [(order[:h], order[h:])]
    purge = max(1, int(round(purge_frac * n)))
    folds = []
    for k in range(1, int(n_folds) + 1):
        cut = int(round(n * (k + 1) / (n_folds + 1)))
        fit, tune = order[:max(1, cut - purge)], order[cut:]
        if len(fit) >= 20 and len(tune) >= 20:
            folds.append((fit, tune))
    return folds or [(order[:n // 2], order[n // 2:])]


def select_probe(Phi, C, Z, y, times, use_state, lams, n_folds,
                 fallback_fn=None):
    """Expanding-window lambda selection; returns (params, mean tune NLL)."""
    folds = expanding_folds(times, n_folds=n_folds)
    return rj.select_joint(Phi, C, Z, y, folds, lams=lams,
                           use_state=use_state, fallback_fn=fallback_fn)


def _index_by_pair(d, split, line):
    """{pair_key: {'pair_id':..., 'rows': {phys: row}}} for one line/split."""
    out = {}
    for r in d.get(split, []):
        if r["line"] != line:
            continue
        pid = tuple(int(x) for x in r["pair_id"])
        k = rj.pair_key(pid)
        e = out.setdefault(k, {"pair_id": pid, "rows": {}})
        e["rows"][int(r["phys"])] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="+", required=True)
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--data-name", default="uci")
    ap.add_argument("--encoder-npz", default=None,
                    help="precomputed SVD table (npz with 'E'); takes priority "
                         "over --data-dir so a locked SHA can be reused")
    ap.add_argument("--svd-rank", type=int, default=16)
    ap.add_argument("--n-folds", type=int, default=3)
    ap.add_argument("--n-blocks", type=int, default=20)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lams", default="1e-4,1e-3,1e-2,1e-1,1,10,100")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--allow-schema-fail", action="store_true",
                    help="override the fail-closed audit (debug only)")
    args = ap.parse_args()

    lams = tuple(float(x) for x in args.lams.split(","))
    labels = args.labels or [Path(p).parent.name for p in args.rows]
    out_dir = Path(args.out_dir or Path(args.rows[0]).parent / "joint_probe")
    out_dir.mkdir(parents=True, exist_ok=True)

    arms = []
    for p in args.rows:
        with open(p, "rb") as f:
            arms.append(pickle.load(f))

    # ---- fail-closed cross-arm schema audit --------------------------------
    recs = [ars.arm_records(d)[0] for d in arms]
    cmp_rep = ars.compare(recs, labels)
    if not cmp_rep["ok"] and not args.allow_schema_fail:
        raise SystemExit(
            "rows/schema audit FAILED across arms: mismatches={} keys_not_in_"
            "all_arms={}. Re-extract the missing fields before running the "
            "probe (--allow-schema-fail only for debugging).".format(
                cmp_rep["mismatch_counts"], cmp_rep["n_keys_not_in_all_arms"]))
    print("[schema] common rows:", cmp_rep["n_common_rows"],
          "ok:", cmp_rep["ok"], flush=True)

    # ---- shared SVD encoder, cut off before the earliest calib event -------
    man0 = {tuple(int(x) for x in m["pair_id"]): m for m in arms[0]["manifest"]}
    calib_t = [float(man0[tuple(int(x) for x in r["pair_id"])]["t_root"])
               for r in arms[0].get("calib", [])
               if tuple(int(x) for x in r["pair_id"]) in man0]
    cutoff = float(min(calib_t)) if calib_t else float("inf")
    if args.encoder_npz:
        z = np.load(args.encoder_npz, allow_pickle=True)
        E = np.asarray(z["E"], np.float64)
        emb_meta = {k: (z[k].item() if hasattr(z[k], "item") else z[k])
                    for k in z.files if k != "E"}
        emb_meta.setdefault("encoder", "svd_locked_npz")
        emb_meta["encoder_npz"] = args.encoder_npz
    else:
        if not args.data_dir:
            raise SystemExit("--data-dir (or --encoder-npz) is required")
        from rpbe.data.uci_link import UCILinkDataset
        ds = UCILinkDataset(args.data_dir, data_name=args.data_name)
        keep = np.asarray(ds.train.timestamps, np.float64) < cutoff
        E, emb_meta = build_svd_encoder(
            ds.train.sources[keep], ds.train.destinations[keep],
            int(ds.n_nodes), rank=args.svd_rank, random_state=0)
        emb_meta.update({"encoder": "svd_train_prefix", "cutoff_time": cutoff,
                         "prefix_edges": int(keep.sum())})
    print("[encoder]", {k: emb_meta.get(k) for k in
                        ("svd_rank", "embedding_hash", "cutoff_time")},
          flush=True)

    report = {"encoder": emb_meta, "labels": labels, "rows": args.rows,
              "schema_ok": bool(cmp_rep["ok"]), "n_boot": int(args.n_boot),
              "n_blocks": int(args.n_blocks), "lams": list(lams),
              "arms": {}}
    row_losses = {}

    for line in LINES:
        sk, pk = LINE_NODES[line]
        origin = ORIGIN_PHYS[line]
        idx = {s: _index_by_pair(arms[0], s, line) for s in ("calib", "audit")}
        # pairs present with EXACTLY the same position set in both splits
        pos_cal = {k: sorted(v["rows"])
                   for k, v in idx["calib"].items()}
        pos_aud = {k: sorted(v["rows"]) for k, v in idx["audit"].items()}
        phys_set = sorted({p for v in pos_cal.values() for p in v}
                          & {p for v in pos_aud.values() for p in v})
        cal_keys = sorted(k for k, v in pos_cal.items()
                          if set(v) == set(phys_set))
        aud_keys = sorted(k for k, v in pos_aud.items()
                          if set(v) == set(phys_set))
        common_phys = {k: [p for p in phys_set if p in pos_aud[k]]
                       for k in aud_keys}
        aud_keys = [k for k in aud_keys if common_phys[k] == phys_set]
        if not cal_keys or not aud_keys:
            report["arms"][line] = {"error": "no aligned pairs"}
            continue

        def _pair_arrays(keys, side):
            """Per-pair shared features: Phi, C_shared, joint label, t_root."""
            fields = [rj.ordered_pair_ids(man0[idx[side][k]["pair_id"]], sk, pk)
                      for k in keys]
            s_pair = np.array([f[0] for f in fields], np.int64)
            p_pair = np.array([f[1] for f in fields], np.int64)
            y = np.array([rj.joint_class(f[2], f[3]) for f in fields], np.int64)
            C = np.stack([ars.ctx_shared(
                idx[side][k]["rows"][phys_set[0]]["ctx"]) for k in keys])
            t = np.array([float(man0[idx[side][k]["pair_id"]]["t_root"])
                          for k in keys], np.float64)
            return rj.phi_tensor(E, s_pair, p_pair), C, y, t

        Phi_c, C_c, y_c, t_c = _pair_arrays(cal_keys, "calib")
        Phi_a, C_a, y_a, t_a = _pair_arrays(aud_keys, "audit")

        # ---- base probe: ONE per line, shared across arms and positions ----
        base_params, base_val = select_probe(
            Phi_c, C_c, np.zeros((len(cal_keys), 1)), y_c, t_c,
            use_state=False, lams=lams, n_folds=args.n_folds)
        nll0_a = rj.joint_row_nll(base_params, Phi_a, C_a,
                                  np.zeros((len(aud_keys), 1)), y_a)

        # ---- one shared bootstrap index set for the whole line -------------
        blocks = rj.time_blocks(t_a, args.n_blocks)
        boot_idx = rj.bootstrap_indices(blocks, args.n_boot, args.seed)

        for ai, d in enumerate(arms):
            lab = labels[ai]
            d_cal = _index_by_pair(d, "calib", line)
            d_aud = _index_by_pair(d, "audit", line)
            nll1 = {}
            for p in phys_set:
                Zc = np.stack([np.asarray(d_cal[k]["rows"][p]["keep"], np.float64)
                               for k in cal_keys])
                Za = np.stack([np.asarray(d_aud[k]["rows"][p]["keep"], np.float64)
                               for k in aud_keys])
                fb = rj.base_as_full(base_params, d_z=Zc.shape[1])
                full, _ = select_probe(
                    Phi_c, C_c, Zc, y_c, t_c, use_state=True, lams=lams,
                    n_folds=args.n_folds, fallback_fn=lambda fb=fb: fb)
                nll1[p] = rj.joint_row_nll(full, Phi_a, C_a, Za, y_a)
                row_losses[(lab, line, p)] = {
                    "pair_keys": aud_keys, "nll_base": nll0_a,
                    "nll_full": nll1[p]}

            jrep = rj.paired_bootstrap(boot_idx, nll0_a, nll1)
            j_src = jrep.get(origin)
            out = {"n_calib_pairs": len(cal_keys), "n_audit_pairs": len(aud_keys),
                   "base_val_nll": float(base_val), "positions": {}}
            for p in phys_set:
                lo, hi = np.percentile(jrep[p], [2.5, 97.5])
                e = {"J_point": rj.gain_bits(nll0_a, nll1[p]),
                     "J_ci95": [float(lo), float(hi)],
                     "nll_base_mean": float(nll0_a.mean()),
                     "nll_full_mean": float(nll1[p].mean())}
                if j_src is not None:
                    R, rlo, rhi, ident = rj.ratio_ci(j_src, jrep[p])
                    e["R"] = R if ident else "ratio_not_identifiable"
                    e["R_ci95"] = [rlo, rhi] if ident else None
                out["positions"][int(p)] = e
            out["J_source"] = (rj.gain_bits(nll0_a, nll1[origin])
                               if origin in phys_set else None)
            out["RootGain_millibits"] = (
                1000.0 * rj.gain_bits(nll0_a, nll1[3]) if 3 in phys_set else None)
            pri = [float(np.mean(jrep[p])) for p in PRI_PHYS[line]
                   if p in jrep]
            out["PRI_bits"] = float(np.mean(pri)) if pri else None
            report["arms"].setdefault(line, {})[lab] = out

    with open(out_dir / "joint_probe_rows.pkl", "wb") as f:
        pickle.dump({"row_losses": row_losses,
                     "meta": {"encoder": emb_meta, "labels": labels,
                              "lams": list(lams), "seed": args.seed}}, f)
    with open(out_dir / "joint_probe.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print("[out]", out_dir / "joint_probe.json", flush=True)
    return report


if __name__ == "__main__":
    main()
