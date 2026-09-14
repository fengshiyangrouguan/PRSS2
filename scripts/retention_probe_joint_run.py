#!/usr/bin/env python3
"""Offline B2 joint-probe runner: rows pkl(s) -> J, R, RootGain.

Consumes the existing ``retention_rows.pkl`` files (no model, no GPU, no
re-extraction) and produces the fixed statistics of the reviewed spec:

    J_k      = mean_i (NLL0_i - NLL1_{k,i}) / ln2         [bits / joint query]
    R_k      = J_k / J_source                             (paired bootstrap)
    RootGain = 1000 * J_root                              [millibits]

Fixed design:
  * ``C_shared`` is the 32-d model-independent context (audit_rows_schema);
  * the candidate features ``phi_ab`` and ``C_shared`` are per-PAIR; only the
    downstream state ``Z_k`` differs per position;
  * the base probe is fit ONCE per source line and shared across arms and
    positions; the full probe is fit per arm/line/position;
  * lambda and the base-vs-full family choice are made on OUT-OF-FOLD
    predictions whose fit side is purged by
    ``t_end = max(t_future_s, t_future_p)``; a candidate lambda that fails in
    any fold is marked invalid and skipped, and only an all-invalid set aborts;
  * every calibration block, every usable fold and the audit block must contain
    all four joint classes -- a missing class makes the family nearly separable
    and the optimiser unreliable, so it is an error, never a silent prune;
  * after selection everything is refit once on the full calibration block and
    the audit block is scored exactly once;
  * ONE block-bootstrap index set (root-event time blocks) is shared by every
    arm, position and metric.

Comparisons are restricted to PRE-REGISTERED same-seed, one-direction pairs
(``--ours-role`` minus each baseline role at the same seed); cross-seed
aggregation is reported separately.  The formal p-value is a paired time-block
sign-flip randomization test; the percentile bootstrap supplies the CIs.

Fail-closed: the runner aborts unless the full rows/schema audit passes, the
encoder verifies against its recorded hash/rank/cutoff/dataset hash, the
future-event-id table is validated, and the chosen fits converge.

Usage:
    python scripts/retention_probe_joint_run.py \
        --rows ours_s0.pkl taskonly_s0.pkl --data-dir /path/to/processed \
        --data-name uci --out-dir ./joint_out
"""

import argparse
import hashlib
import json
import pickle
import re
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

LINES = ("Y_leaf", "Y_a2", "Y_a1")
LINE_NODES = {"Y_leaf": ("leaf", "a2"), "Y_a2": ("a2", "a1"),
              "Y_a1": ("a1", "root")}
ORIGIN_PHYS = {"Y_leaf": 0, "Y_a2": 1, "Y_a1": 2}
PRI_PHYS = {"Y_leaf": (1, 2, 3), "Y_a2": (2, 3), "Y_a1": (3,)}
HOP_LINE = {3: "Y_leaf", 2: "Y_a2", 1: "Y_a1"}
# positions the extractor emits for each line; a pair missing one is an error
EXPECTED_PHYS = {"Y_leaf": (0, 1, 2, 3), "Y_a2": (1, 2, 3), "Y_a1": (2, 3)}

# Terminology is fixed; the numbers below are probe-estimated predictive gains
# under a restricted family, not Shannon mutual information.
NAMING = {
    "J_s_k": "held-out predictive gain (bits per joint query) of the state at "
             "position k about the source's joint local future -- NOT mutual "
             "information",
    "retention_ratio": "R_{s->k} = J_{s->k} / J_{s->s} -- predictive-signal "
                       "retention ratio",
    "RootGain_h": "predictive gain at the root of the h-hop source signal",
    "R_source": "R_{s->s} = 1 is only the normalisation origin; it does not "
                "mean the source carries one bit",
    "p_signflip": "paired time-block sign-flip randomization p-value "
                  "(one-sided, pre-registered direction)",
    "p_tail_boot": "Monte-Carlo bootstrap tail probability (descriptive; not "
                   "a sign test, not used for the Holm decision)",
}


# ------------------------------------------------------------------ helpers
class RunnerError(RuntimeError):
    """A fail-closed condition in the runner (never a silently pruned sample)."""


def expanding_folds(t_root, t_end, n_folds=3, min_rows=20):
    """Purged expanding-window folds over contiguous, non-overlapping slices.

    Rows are cut into ``n_folds + 1`` contiguous time slices by ``t_root``.
    Slice 0 is always available as fit data; for fold ``k`` in 1..n_folds the
    validation block is slice ``k`` and the fit set is every row whose TARGET
    FUTURE has already finished before that block starts:
    ``t_end_i < min(t_root over the block)``.  That purges label leakage across
    the boundary; ties on ``t_root`` cannot enter the fit set because
    ``t_end >= t_root``.
    """
    t_root = np.asarray(t_root, np.float64)
    t_end = np.asarray(t_end, np.float64)
    order = np.argsort(t_root, kind="stable")
    n = len(order)
    if n < 2 * int(min_rows):
        raise RunnerError(
            "not enough rows for purged folds: n={} min_rows={}".format(
                n, min_rows))
    edges = np.quantile(t_root[order],
                        np.linspace(0.0, 1.0, int(n_folds) + 2))
    cuts = np.searchsorted(t_root[order], edges)
    folds = []
    for k in range(1, int(n_folds) + 1):
        lo, hi = int(cuts[k]), int(cuts[k + 1])
        if hi - lo < int(min_rows):
            continue
        tune = order[lo:hi]
        tval = float(t_root[tune].min())
        fit = np.where(t_end < tval)[0]
        if len(fit) < int(min_rows):
            continue
        folds.append((fit, tune))
    return folds


def build_folds(t_root, t_end, y, line, n_folds, min_rows):
    """Folds that additionally carry all four joint classes on BOTH sides.

    Fewer/larger slices are tried in turn; if no configuration yields a usable
    fold the line is an error rather than an under-powered silent fallback.
    """
    attempts = []
    for nf in range(int(n_folds), 0, -1):
        try:
            folds = expanding_folds(t_root, t_end, n_folds=nf,
                                    min_rows=min_rows)
        except RunnerError as e:
            attempts.append({"n_folds": nf, "error": str(e)})
            continue
        good, rejected = [], []
        for fi, (fit, tune) in enumerate(folds):
            try:
                rj.require_all_classes(y[fit],
                                       where="{} fold{} fit".format(line, fi))
                rj.require_all_classes(y[tune],
                                       where="{} fold{} tune".format(line, fi))
                good.append((fit, tune))
            except rj.ProbeFitError as e:
                rejected.append({"fold": fi, "reason": str(e)})
        attempts.append({"n_folds": nf, "built": len(folds),
                         "usable": len(good), "rejected": rejected})
        if good:
            return good, attempts
    raise RunnerError(
        "line {}: no usable validation fold (all four joint classes required "
        "on both sides of every fold).  attempts={}".format(line, attempts))


def _role_seed(d):
    """(role, seed) from checkpoint meta; role is the arm name minus the seed."""
    m = d.get("meta", {})
    arm = str(m.get("arm") or "")
    seed = m.get("seed")
    role = re.sub(r"^seed\d+[_-]?", "", arm) or arm
    if seed is None:
        mm = re.search(r"seed(\d+)", arm)
        seed = int(mm.group(1)) if mm else None
    return role, seed


def build_eid_time_map(ds):
    """edge_id -> timestamp, validated.

    Edge ids are only required to be UNIQUE and lookable-up; they are NOT
    required to run contiguously from 0 (UCI/BenchTemp may reserve 0 as padding
    with real edges from 1).  Unassigned slots stay NaN so that looking up an id
    that never appears in the stream is a detectable error rather than a silent
    time of 0.0.
    """
    eid = np.asarray(ds.full.edge_idxs, np.int64)
    if eid.size == 0:
        raise RunnerError("edge_idxs is empty; cannot map future event ids")
    if eid.min() < 0:
        raise RunnerError("edge_idxs contains negative ids")
    n_uniq = int(np.unique(eid).size)
    if n_uniq != int(eid.size):
        raise RunnerError(
            "edge_idxs is not a bijection ({} distinct of {} rows); an "
            "edge_id -> time map would silently overwrite entries".format(
                n_uniq, eid.size))
    out = np.full(int(eid.max()) + 1, np.nan, dtype=np.float64)
    ts = np.asarray(ds.full.timestamps, np.float64)
    if not np.all(np.isfinite(ts)):
        raise RunnerError("stream timestamps contain non-finite values")
    out[eid] = ts
    if not np.all(np.isfinite(out[eid])):
        raise RunnerError("some edge_id -> time assignments are not finite")
    return out


def _future_end(man, sk, pk, fut_map):
    """``max(t_future_s, t_future_p)`` for one pair.

    Prefers ``manifest['future_event_time']`` (new schema); otherwise maps the
    stored future EVENT ID through the validated edge-id -> time table.  An id
    that is out of range or was never assigned a timestamp is an error.
    """
    ft = man.get("future_event_time")
    if ft is not None:
        return max(float(ft[sk]), float(ft[pk]))
    if fut_map is None:
        raise RunnerError(
            "manifest lacks future_event_time and no dataset was supplied, so "
            "the future-event-time purge cannot be done")
    eid = man["pos_future_event_id"]
    vals = []
    for k in (sk, pk):
        j = int(eid[k])
        if not (0 <= j < fut_map.size):
            raise RunnerError(
                "future_event_id {} (node {}) is outside the edge-id table "
                "(size {})".format(j, k, fut_map.size))
        v = float(fut_map[j])
        if not np.isfinite(v):
            raise RunnerError(
                "future_event_id {} (node {}) is not an edge id that occurs in "
                "the stream (no timestamp assigned)".format(j, k))
        vals.append(v)
    return max(vals)


def _file_sha16(paths):
    """Same content hash the extractor records as ``dataset_hash``."""
    h = hashlib.sha256()
    for pp in paths:
        fh = hashlib.sha256()
        with open(pp, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                fh.update(chunk)
        h.update(fh.digest())
    return h.hexdigest()[:16]


def _index_by_pair(d, split, line):
    """{pair_key: {'pair_id':..., 'rows': {phys: row}}} for one line/split."""
    out = {}
    for r in d.get(split, []):
        if r["line"] != line:
            continue
        pid = tuple(int(x) for x in r["pair_id"])
        e = out.setdefault(rj.pair_key(pid), {"pair_id": pid, "rows": {}})
        e["rows"][int(r["phys"])] = r
    return out


def _require_positions(idx, line, split):
    """Every pair must carry exactly the expected positions for its line."""
    want = set(EXPECTED_PHYS[line])
    bad = {k: sorted(v["rows"]) for k, v in idx.items()
           if set(v["rows"]) != want}
    if bad:
        ex = list(bad.items())[:3]
        raise RunnerError(
            "{} {}: {} pair(s) do not have the expected positions {} "
            "(examples: {}).  Re-extract rather than pruning rows.".format(
                line, split, len(bad), sorted(want), ex))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="+", required=True)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--data-name", default="uci")
    ap.add_argument("--encoder-npz", default=None,
                    help="locked SVD table (npz with 'E' + hashes); verified")
    ap.add_argument("--svd-rank", type=int, default=16)
    ap.add_argument("--row-label-kind", default=None,
                    choices=ars.LABEL_KIND_CHOICES,
                    help="override the stored-row label convention")
    ap.add_argument("--ours-role", default="ours",
                    help="role compared as the primary arm (pre-registered)")
    ap.add_argument("--n-folds", type=int, default=3)
    ap.add_argument("--min-fold-rows", type=int, default=20)
    ap.add_argument("--n-blocks", type=int, default=20)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lams", default="1e-4,1e-3,1e-2,1e-1,1,10,100")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    lams = tuple(float(x) for x in args.lams.split(","))
    out_dir = Path(args.out_dir or Path(args.rows[0]).parent / "joint_probe")
    out_dir.mkdir(parents=True, exist_ok=True)

    arms = []
    for p in args.rows:
        with open(p, "rb") as f:
            arms.append(pickle.load(f))

    # ---- fail-closed audit (same function the CLI runs) --------------------
    audit = ars.full_audit(arms, declared_label_kind=args.row_label_kind)
    if not audit["ok"]:
        raise SystemExit(
            "rows/schema audit FAILED; refusing to run the probe.\n  "
            "problems: {}\n  field mismatches: {}\n  keys not in all arms: {}"
            .format(audit["problems"], audit["compare"]["mismatch_counts"],
                    audit["compare"]["n_keys_not_in_all_arms"]))
    labels = audit["labels"]
    print("[schema] ok; arms =", labels, flush=True)

    ds = None
    if args.data_dir:
        from rpbe.data.uci_link import UCILinkDataset
        ds = UCILinkDataset(args.data_dir, data_name=args.data_name)

    # ---- shared SVD encoder, cut off before the earliest calib event -------
    man0 = {tuple(int(x) for x in m["pair_id"]): m for m in arms[0]["manifest"]}
    calib_root_t = [float(man0[tuple(int(x) for x in r["pair_id"])]["t_root"])
                    for r in arms[0].get("calib", [])
                    if tuple(int(x) for x in r["pair_id"]) in man0]
    cutoff = float(min(calib_root_t)) if calib_root_t else float("inf")
    ds_hash = arms[0].get("meta", {}).get("dataset_hash")
    if args.encoder_npz:
        z = np.load(args.encoder_npz, allow_pickle=True)
        E = np.asarray(z["E"], dtype=np.float64)
        emb_meta = {k: (z[k].item() if hasattr(z[k], "item") else z[k])
                    for k in z.files if k != "E"}
        emb_meta.setdefault("encoder", "svd_locked_npz")
        emb_meta["encoder_npz"] = args.encoder_npz
    else:
        if ds is None:
            raise SystemExit("--data-dir (or --encoder-npz) is required")
        keep = np.asarray(ds.train.timestamps, np.float64) < cutoff
        E, emb_meta = rj.build_svd_encoder(
            ds.train.sources[keep], ds.train.destinations[keep],
            int(ds.n_nodes), rank=args.svd_rank, random_state=0)
        emb_meta.update({"encoder": "svd_train_prefix", "cutoff_time": cutoff,
                         "prefix_edges": int(keep.sum())})
        try:
            emb_meta["dataset_hash"] = _file_sha16([
                str(Path(args.data_dir) / "ml_{}.csv".format(args.data_name)),
                str(Path(args.data_dir) / "ml_{}.npy".format(args.data_name)),
                str(Path(args.data_dir) / "ml_{}_node.npy".format(args.data_name))])
        except OSError as e:
            raise SystemExit("cannot hash the dataset files: {}".format(e))
    problems = rj.verify_encoder(E, emb_meta, expect_rank=args.svd_rank,
                                 cutoff_max=cutoff, dataset_hash=ds_hash)
    if problems:
        raise SystemExit("encoder verification FAILED: {}".format(problems))
    print("[encoder]", {k: emb_meta.get(k) for k in
                        ("svd_rank", "embedding_hash", "cutoff_time")},
          flush=True)

    fut_map = build_eid_time_map(ds) if ds is not None else None

    report = {"encoder": emb_meta, "labels": labels, "rows": args.rows,
              "naming": NAMING,
              "schema": {"ok": audit["ok"], "problems": audit["problems"],
                         "label_kinds": [a["label_kind"]
                                         for a in audit["per_arm"]]},
              "n_boot": int(args.n_boot), "n_blocks": int(args.n_blocks),
              "ours_role": args.ours_role, "lams": list(lams), "arms": {}}
    row_losses = {}
    root = {}          # (label, line) -> {'reps','point','nll1','blocks'}

    for line in LINES:
        sk, pk = LINE_NODES[line]
        origin = ORIGIN_PHYS[line]
        idx = {s: _index_by_pair(arms[0], s, line) for s in ("calib", "audit")}
        for s in ("calib", "audit"):
            _require_positions(idx[s], line, s)
        phys_set = list(EXPECTED_PHYS[line])
        cal_keys = sorted(idx["calib"])
        aud_keys = sorted(idx["audit"])
        if not cal_keys or not aud_keys:
            raise RunnerError("line {}: empty calib or audit block".format(line))

        def _pair_arrays(keys, side):
            fields = [rj.ordered_pair_ids(man0[idx[side][k]["pair_id"]], sk, pk)
                      for k in keys]
            s_pair = np.array([f[0] for f in fields], np.int64)
            p_pair = np.array([f[1] for f in fields], np.int64)
            y = np.array([rj.joint_class(f[2], f[3]) for f in fields], np.int64)
            C = np.stack([ars.ctx_shared(
                idx[side][k]["rows"][phys_set[0]]["ctx"]) for k in keys])
            t = np.array([float(man0[idx[side][k]["pair_id"]]["t_root"])
                          for k in keys], np.float64)
            te = np.array([_future_end(man0[idx[side][k]["pair_id"]], sk, pk,
                                       fut_map) for k in keys], np.float64)
            return rj.phi_tensor(E, s_pair, p_pair), C, y, t, te

        Phi_c, C_c, y_c, t_c, te_c = _pair_arrays(cal_keys, "calib")
        Phi_a, C_a, y_a, t_a, _ = _pair_arrays(aud_keys, "audit")
        count_c = rj.require_all_classes(y_c, where="{} calib".format(line))
        count_a = rj.require_all_classes(y_a, where="{} audit".format(line))
        folds, fold_attempts = build_folds(t_c, te_c, y_c, line,
                                           args.n_folds, args.min_fold_rows)

        base_params, cv_base, base_diag = rj.select_joint(
            Phi_c, C_c, np.zeros((len(cal_keys), 1)), y_c, folds,
            lams=lams, use_state=False)
        nll0_a = rj.joint_row_nll(base_params, Phi_a, C_a,
                                  np.zeros((len(aud_keys), 1)), y_a)

        blocks = rj.time_blocks(t_a, args.n_blocks)
        boot_idx = rj.bootstrap_indices(blocks, args.n_boot, args.seed)

        for ai, d in enumerate(arms):
            lab = labels[ai]
            d_cal = _index_by_pair(d, "calib", line)
            d_aud = _index_by_pair(d, "audit", line)
            nll1, chosen = {}, {}
            for p in phys_set:
                Zc = np.stack([np.asarray(d_cal[k]["rows"][p]["keep"], np.float64)
                               for k in cal_keys])
                Za = np.stack([np.asarray(d_aud[k]["rows"][p]["keep"], np.float64)
                               for k in aud_keys])
                fb = (rj.base_as_full(base_params, d_z=Zc.shape[1]), cv_base)
                full, cv_full, diag = rj.select_joint(
                    Phi_c, C_c, Zc, y_c, folds, lams=lams, use_state=True,
                    fallback=fb)
                nll1[p] = rj.joint_row_nll(full, Phi_a, C_a, Za, y_a)
                chosen[p] = {"lam": float(full["lam"]),
                             "used_base_fallback": bool(diag["used_fallback"]),
                             "cv_oof_nll": float(cv_full),
                             "n_invalid_lambda": int(diag["n_invalid"]),
                             "lambda_trials": diag["trials"]}
                row_losses[(lab, line, p)] = {
                    "pair_keys": aud_keys, "nll_base": nll0_a,
                    "nll_full": nll1[p]}

            jrep = rj.paired_bootstrap(boot_idx, nll0_a, nll1)
            j_src = jrep.get(origin)
            j_src_pt = (rj.gain_bits(nll0_a, nll1[origin])
                        if origin in phys_set else None)
            out = {"n_calib_pairs": len(cal_keys), "n_audit_pairs": len(aud_keys),
                   "n_folds": len(folds), "fold_attempts": fold_attempts,
                   "class_counts_calib": count_c, "class_counts_audit": count_a,
                   "audit_pair_keys": aud_keys,
                   "audit_block_ids": [int(b) for b in blocks],
                   "base_probe": {"cv_oof_nll": float(cv_base),
                                  "lam": float(base_params["lam"]),
                                  "n_invalid_lambda": int(base_diag["n_invalid"])},
                   "positions": {}}
            for p in phys_set:
                lo, hi = np.percentile(jrep[p], [2.5, 97.5])
                pt = rj.gain_bits(nll0_a, nll1[p])
                e = {"J_point": pt, "J_ci95": [float(lo), float(hi)],
                     "nll_base_mean": float(nll0_a.mean()),
                     "nll_full_mean": float(nll1[p].mean()),
                     "probe": chosen[p]}
                if j_src is not None and j_src_pt is not None:
                    R, rlo, rhi, ident = rj.ratio_ci(
                        j_src, jrep[p], j_src_pt, pt)
                    e["retention_ratio"] = (R if ident
                                            else "ratio_not_identifiable")
                    e["retention_ratio_ci95"] = [rlo, rhi] if ident else None
                out["positions"][int(p)] = e
            out["J_source"] = j_src_pt
            out["RootGain_millibits"] = (
                1000.0 * rj.gain_bits(nll0_a, nll1[3]) if 3 in phys_set else None)
            pri = [float(np.mean(jrep[p])) for p in PRI_PHYS[line] if p in jrep]
            out["PRI_bits"] = float(np.mean(pri)) if pri else None
            report["arms"].setdefault(line, {})[lab] = out
            if 3 in phys_set:
                root[(lab, line)] = {"reps": jrep[3],
                                     "point": rj.gain_bits(nll0_a, nll1[3]),
                                     "nll1": nll1[3]}

    # ---- pre-registered, same-seed, ONE-DIRECTION comparisons --------------
    info = {lab: _role_seed(arms[i]) for i, lab in enumerate(labels)}
    by_seed = {}
    for lab, (role, seed) in info.items():
        by_seed.setdefault(seed, {})[role] = lab
    comparisons, skipped = [], []
    for seed in sorted(by_seed, key=lambda s: (s is None, s)):
        roles = by_seed[seed]
        ours = roles.get(args.ours_role)
        if ours is None:
            skipped.append({"seed": seed, "roles": sorted(roles),
                            "reason": "no '{}' arm".format(args.ours_role)})
            continue
        for role in sorted(roles):
            if role == args.ours_role:
                continue
            base = roles[role]
            hops, pvals, keys = {}, [], []
            for h, line in HOP_LINE.items():
                ka, kb = (ours, line), (base, line)
                if ka not in root or kb not in root:
                    continue
                dpt, lo, hi, p_tail = rj.paired_diff_ci(
                    root[ka]["reps"], root[kb]["reps"],
                    root[ka]["point"], root[kb]["point"])
                # formal test: sign-flip the RAW paired audit rows, not the
                # bootstrap replicates.  d_i = (NLL1_other,i - NLL1_ours,i)/ln2
                # is exactly the per-row contribution to
                # RootGain_ours - RootGain_other.
                d_rows = (root[kb]["nll1"] - root[ka]["nll1"]) / np.log(2.0)
                p_sf = rj.block_signflip_p(
                    d_rows, blocks, n_perm=args.n_perm, seed=args.seed + h,
                    alternative="greater")
                hops[str(h)] = {"line": line, "diff_point": dpt,
                                "diff_ci95": [lo, hi],
                                "p_tail_boot": p_tail, "p_signflip": p_sf}
                pvals.append(p_sf)
                keys.append(str(h))
            if not pvals:
                continue
            adj = rj.holm_adjust(pvals)
            for k, p in zip(keys, adj):
                hops[k]["p_holm"] = float(p)
                hops[k]["significant_05"] = bool(p < 0.05)
            comparisons.append({
                "name": "{} - {} (seed {})".format(ours, base, seed),
                "ours": ours, "ours_role": info[ours][0], "base": base,
                "base_role": role, "seed": seed, "hops": hops})

    # ---- cross-seed summary: seed is the OUTER independent replicate ------
    # Each seed contributes ONE already-paired delta per hop; rows are never
    # pooled across seeds (that would fake a narrow interval).  With 2-3 seeds
    # no cross-seed interval is claimed -- only the mean and the seed points.
    across = {}
    for cmp_ in comparisons:
        key = "{} - {}".format(cmp_["ours_role"], cmp_["base_role"])
        for h, e in cmp_["hops"].items():
            across.setdefault(key, {}).setdefault(h, []).append(
                {"seed": cmp_["seed"], "diff_point": e["diff_point"]})
    report["comparisons"] = {c["name"]: c["hops"] for c in comparisons}
    report["comparisons_skipped"] = skipped
    report["comparisons_across_seed"] = {
        k: {h: {"n_seeds": len(v),
                "mean_diff_point": float(np.mean([x["diff_point"]
                                                  for x in v])),
                "per_seed": sorted(v, key=lambda x: (x["seed"] is None,
                                                     x["seed"])),
                "ci_claimed": False}
            for h, v in hs.items()}
        for k, hs in across.items()}

    with open(out_dir / "joint_probe_rows.pkl", "wb") as f:
        pickle.dump({"row_losses": row_losses,
                     "meta": {"encoder": emb_meta, "labels": labels,
                              "lams": list(lams), "seed": args.seed,
                              "schema": report["schema"]}}, f)
    with open(out_dir / "joint_probe.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print("[out]", out_dir / "joint_probe.json", flush=True)
    return report


if __name__ == "__main__":
    main()
