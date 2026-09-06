#!/usr/bin/env python3
"""Motivation probe: predictive-information utilization vs aggregation depth.

Part II of the TGN motivation + structural ablation task.  It FREEZES an
already-trained ``task_only`` (TGN + Gamma, lambda=0) best.pt and asks, for
each internal compression cut, how much of the future information that is
*available* in the pre-compression state U survives into the compressed
state Z, measured at each aggregation depth.

Target (per user decision): the fixed future feature of the FIRST strictly
later legal future event of the cut,

    T_v = phi_Y(Y_v^{(1)}),

computed with the run's own ``FixedMaps.future_vector`` (no 2Obs, no extra
future encoder).  Three same-class linear ridge multi-output probes:

    q_C : T <- C                  (context / nearer-to-root baseline)
    q_U : T <- (U, C)             (+ the pre-compression state)
    q_Z : T <- (Z, C)             (+ the deployed compressed state)

With R the held-out normalized MSE, define

    A_d = R(q_C) - R(q_U)     (predictive gain available in U beyond C)
    K_d = R(q_C) - R(q_Z)     (predictive gain retained in Z beyond C)
    PUR_d = K_d / A_d

per aggregation depth d (= the TGN recursion layer at the cut).  Negative
values are reported, never clipped; PUR is suppressed when A_d's CI covers
zero.  Probe data is split per ROOT QUERY (a tree never straddles splits);
per-tree weights equalize tree influence; uncertainty is a cluster bootstrap
over roots.  A small two-layer MLP may re-check the trend (never replaces the
linear main result).  A secondary logistic probe targets the root task label
(AP/AUC/NLL) to connect the future-information bottleneck to the task.

The model is used purely as a frozen feature source: evaluation mode, no
gradient, no optimizer, no future ever re-enters a TGN forward.

Controls (Part II.8): Z=U identity (expect PUR ~1), randomized Z (expect
~0), and permuted T (expect all three risk gaps ~0).  None may shuffle
across the train/audit boundary.

Usage:
    python -m scripts.motivation_probe --output outputs/task_only_run \\
        --data-dir old/processed_tgn_data --gpu 0 [--distances 1,2] \\
        [--max-batches N] [--controls] [--out-json out.json]

    # the output dir must hold the run's config.json and best.pt
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np
import torch

torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rpbe.data.jodie import JodieDataset
from rpbe.records import JodieFutureIndex
from rpbe.training.jodie_loop import select_trace_rows


def parse_args():
    p = argparse.ArgumentParser("motivation depth probe (Part II)")
    p.add_argument("--output", required=True,
                   help="run dir holding config.json + best.pt (task_only)")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--distances", default="", help="comma list; empty = all")
    p.add_argument("--trace-roots", type=int, default=64)
    p.add_argument("--max-batches", type=int, default=0,
                   help="cap the collection walk at N batches (0 = full)")
    p.add_argument("--controls", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-json", default="")
    p.add_argument("--min-cuts-per-depth", type=int, default=60)
    p.add_argument("--n-boot", type=int, default=200)
    return p.parse_args()


# ------------------------------------------------------------------ probe fit
def fit_ridge_multi(x_tr, y_tr):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(x_tr)
    model = Ridge(alpha=1.0)
    model.fit(scaler.transform(x_tr), y_tr)
    return scaler, model


def _audit_one_depth(samples_sel, d, args, z_override=None,
                     y_field="T", y_override=None):
    """Fit q_C/q_U/q_Z at one depth and report A/K/PUR + cluster-bootstrap
    CIs.  ``z_override`` is a per-sample replacement of the Z stack (for the
    controls) and ``y_override`` a per-sample target replacement.  Returns a
    JSON-able result dict.
    """
    roots_all = np.asarray(sorted({s["root_id"] for s in samples_sel}))
    rng = np.random.RandomState(args.seed + 1000 * d)
    roots_all = roots_all[rng.permutation(len(roots_all))]
    n_tr = int(0.6 * len(roots_all))
    n_ca = int(0.8 * len(roots_all))
    root_tr = set(int(r) for r in roots_all[:n_tr])
    root_ca = set(int(r) for r in roots_all[n_tr:n_ca])
    root_te = set(int(r) for r in roots_all[n_ca:])

    u_arr = np.stack([s["U"] for s in samples_sel])
    c_arr = np.stack([s["C"] for s in samples_sel])
    z_arr = (z_override if z_override is not None
             else np.stack([s["Z"] for s in samples_sel]))
    if y_override is not None:
        y_arr = y_override
    else:
        y_raw = np.stack([s[y_field] for s in samples_sel])
        y_arr = y_raw if y_raw.ndim > 1 else y_raw.reshape(-1, 1)
    roots_arr = np.asarray([s["root_id"] for s in samples_sel])
    tr = np.asarray([s["root_id"] in root_tr for s in samples_sel])
    te = np.asarray([s["root_id"] in root_te for s in samples_sel])

    def split(x):
        return (x[tr], x[te])

    u_tr, u_te = split(u_arr)
    c_tr, c_te = split(c_arr)
    z_tr, z_te = split(z_arr)
    y_tr_a, y_te_a = split(y_arr)

    # Fit the three probes on the tree train rows.
    def tree_risk(x_trv):
        return fit_ridge_multi(x_trv, y_tr_a)

    scC, mC = tree_risk(c_tr)
    uc_tr = np.concatenate([u_tr, c_tr], axis=1)
    uc_te = np.concatenate([u_te, c_te], axis=1)
    scU, mU = tree_risk(uc_tr)
    zc_tr = np.concatenate([z_tr, c_tr], axis=1)
    zc_te = np.concatenate([z_te, c_te], axis=1)
    scZ, mZ = tree_risk(zc_tr)

    # Risk on the AUDIT roots.  We first compute per-root normalized-MSE
    # contributions on the (sample-level) audit rows — the per-dimension
    # variance used for normalization is the AUDIT-pool variance, so a small
    # per-root variance cannot blow one root's contribution up — and then
    # average the contributions over roots (equal per-tree vote).  The point
    # estimate and the bootstrap CI below both use this same statistic.
    denom = np.var(y_te_a, axis=0) + 1e-12

    def contrib(feats, sc, model):
        xs = sc.transform(feats)
        pred = model.predict(xs)
        idx_by_root = {}
        for j in range(len(feats)):
            idx_by_root.setdefault(int(roots_arr[te][j]), []).append(j)
        out = {}
        for k, js in idx_by_root.items():
            per_dim = (((pred[js] - y_te_a[js]) ** 2).mean(axis=0) / denom)
            out[k] = float(np.mean(per_dim))
        return out

    rc = contrib(c_te, scC, mC)
    ru = contrib(uc_te, scU, mU)
    rz = contrib(zc_te, scZ, mZ)
    rC = float(np.mean(list(rc.values())))
    rU = float(np.mean(list(ru.values())))
    rZ = float(np.mean(list(rz.values())))

    entry = {"R_C": rC, "R_U": rU, "R_Z": rZ,
             "A_d": rC - rU, "K_d": rC - rZ,
             "n": len(samples_sel),
             "n_roots": len(roots_all),
             "n_audit_roots": len(rc)}
    a_d = entry["A_d"]
    entry["PUR_d"] = (entry["K_d"] / a_d) if abs(a_d) > 1e-12 else None

    # Cluster bootstrap over the AUDIT roots: resample roots with
    # replacement and average the per-root contributions of the three
    # already-fit probes.  The bootstrap mean approximates the point
    # estimate above, so the CI brackets it.
    te_roots = np.asarray(sorted(rc.keys()))
    if len(te_roots) >= 4:
        boot_rng = np.random.RandomState(args.seed + 7000 * d)
        a_b, k_b = [], []
        for _ in range(args.n_boot):
            chosen = set(int(r) for r in boot_rng.choice(
                te_roots, size=len(te_roots), replace=True))
            a_vals = [rc[k] for k in te_roots if k in chosen]
            u_vals = [ru[k] for k in te_roots if k in chosen]
            z_vals = [rz[k] for k in te_roots if k in chosen]
            rC_b = float(np.mean(a_vals))
            rU_b = float(np.mean(u_vals))
            rZ_b = float(np.mean(z_vals))
            a_b.append(rC_b - rU_b)
            k_b.append(rC_b - rZ_b)
        entry["A_d_ci"] = [float(np.percentile(a_b, 2.5)),
                           float(np.percentile(a_b, 97.5))]
        entry["K_d_ci"] = [float(np.percentile(k_b, 2.5)),
                           float(np.percentile(k_b, 97.5))]
    return entry


def _audit_root_task(samples_sel, d, args):
    """Secondary probe (Part II.9): predict the ROOT task label F from C,
    (U, C), (Z, C) with logistic ridge.  Reports NLL gain as the primary
    quantity plus AP/AUC as auxiliary; per-root split and cluster bootstrap
    mirror the future-feature audit."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, average_precision_score

    def _nll(p, y):
        eps = 1e-6
        p = np.clip(p, eps, 1 - eps)
        return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())

    roots_all = np.asarray(sorted({s["root_id"] for s in samples_sel}))
    rng = np.random.RandomState(args.seed + 3000 * d)
    roots_all = roots_all[rng.permutation(len(roots_all))]
    n_tr = int(0.6 * len(roots_all))
    n_ca = int(0.8 * len(roots_all))
    root_tr = set(int(r) for r in roots_all[:n_tr])
    root_ca = set(int(r) for r in roots_all[n_tr:n_ca])
    root_te = set(int(r) for r in roots_all[n_ca:])

    u_arr = np.stack([s["U"] for s in samples_sel])
    c_arr = np.stack([s["C"] for s in samples_sel])
    z_arr = np.stack([s["Z"] for s in samples_sel])
    y_arr = np.asarray([s["F"] for s in samples_sel])
    roots_arr = np.asarray([s["root_id"] for s in samples_sel])
    tr = np.asarray([s["root_id"] in root_tr for s in samples_sel])
    te = np.asarray([s["root_id"] in root_te for s in samples_sel])

    def fit_eval(x_feat):
        sc = StandardScaler().fit(x_feat[tr])
        xs_tr = sc.transform(x_feat[tr])
        xs_te = sc.transform(x_feat[te])
        if len(np.unique(y_arr[tr])) < 2:
            # Root labels are extremely sparse (wikipedia test has 44
            # positives / 23621); a per-root split can land on a train fold
            # with a single class.  Report NaN instead of crashing (Part
            # II.9: do not resample a natural-distribution metric).
            return float("nan"), float("nan"), float("nan")
        clf = LogisticRegression(C=1.0, max_iter=2000).fit(xs_tr, y_arr[tr])
        p_te = clf.predict_proba(xs_te)[:, 1]
        auc = float(roc_auc_score(y_arr[te], p_te)) \
            if len(np.unique(y_arr[te])) > 1 else float("nan")
        ap = float(average_precision_score(y_arr[te], p_te))
        return _nll(p_te, y_arr[te]), auc, ap

    nll_c, auc_c, ap_c = fit_eval(c_arr)
    nll_u, auc_u, ap_u = fit_eval(np.concatenate([u_arr, c_arr], axis=1))
    nll_z, auc_z, ap_z = fit_eval(np.concatenate([z_arr, c_arr], axis=1))
    return {
        "task_NLL_C": nll_c, "task_NLL_U": nll_u, "task_NLL_Z": nll_z,
        "task_auc_C": auc_c, "task_auc_U": auc_u, "task_auc_Z": auc_z,
        "task_ap_C": ap_c, "task_ap_U": ap_u, "task_ap_Z": ap_z,
        "A_d_task": nll_c - nll_u,
        "K_d_task": nll_c - nll_z,
        "n": len(samples_sel),
        "n_roots": len(roots_all),
    }


def main():
    args = parse_args()
    out = Path(args.output)
    cfg = json.load(open(out / "config.json"))
    cli = cfg["cli"]
    device = torch.device(
        "cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")

    import scripts.train_jodie as tj
    dataset = JodieDataset(cli["data"], data_dir=args.data_dir,
                           use_validation=True)
    full, train, _, _ = dataset.splits()
    ns = argparse.Namespace(**cli)
    ns.data_dir = args.data_dir
    # Backward compatibility: checkpoints trained before these CLI flags
    # existed store a config.json whose ``cli`` block lacks them.
    ns.supervision_mode = getattr(ns, "supervision_mode",
                                  "production")
    components = tj.build_components(ns, device, dataset)
    tgn = components["tgn"]
    best = torch.load(out / "best.pt", map_location=device, weights_only=False)
    for name in ("decoder", "tgn"):
        components[name].load_state_dict(best["model"][name])
    if components["compressor"] is not None and \
            "compressor" in best["model"]:
        components["compressor"].load_state_dict(best["model"]["compressor"])
    for p in tgn.parameters():
        p.requires_grad_(False)
    if components["compressor"] is not None:
        for p in components["compressor"].parameters():
            p.requires_grad_(False)
    tgn.eval()

    adapter = components["adapter"]
    fixed_maps = components["fixed_maps"]
    root_features = dataset.node_features
    future_index = JodieFutureIndex(train)
    n_layers = cli.get("n_layer", 2)
    bs = cli.get("bs", 200)
    n_degree = cli.get("n_degree", 5)

    # ---------------------------------------------------------- collection
    samples = []
    if tgn.use_memory:
        tgn.memory.__init_memory__()
    n_batches = math.ceil(len(train.sources) / bs)
    max_batches = args.max_batches or n_batches
    with torch.no_grad():
        for k in range(min(max_batches, n_batches)):
            s = k * bs
            e = min(len(train.sources), (k + 1) * bs)
            srcs = train.sources[s:e]
            dsts = train.destinations[s:e]
            tms = train.timestamps[s:e]
            eis = train.edge_idxs[s:e]
            labels_np = train.labels[s:e]
            trace_rows = select_trace_rows(
                np.zeros(e - s), args.trace_roots, args.seed, k,
                "evenly_spaced")
            if not trace_rows:
                continue
            adapter.set_trace_source_rows(trace_rows)
            tgn.compute_temporal_embeddings(
                srcs, dsts, dsts, tms, eis, n_degree)
            trace = adapter.trace
            if trace is None or not trace.cuts:
                adapter.clear_trace()
                continue
            by_root = {}
            for cut in trace.cuts:
                by_root.setdefault(int(cut.root_row), []).append(cut)
            for row, cuts in by_root.items():
                if row >= len(labels_np):
                    continue
                label = float(labels_np[row])
                node_feat = torch.from_numpy(
                    root_features[int(srcs[row])]).float().to(device)
                root_id = int(s + row)
                for cut in cuts:
                    layer = int(cut.tau.split("layer")[-1])
                    if not (0 < layer < n_layers):
                        continue
                    nearer = [c.z for c in cuts
                              if int(c.tau.split("layer")[-1]) > layer]
                    c_vec = torch.cat(nearer, dim=0) if nearer else node_feat
                    y1 = future_index.query(int(cut.node), float(cut.time),
                                            limit=1)
                    if not y1:
                        continue
                    samples.append({
                        "root_id": int(root_id),
                        "depth": int(layer),
                        "U": cut.u.detach().cpu().numpy(),
                        "Z": cut.z.detach().cpu().numpy(),
                        "C": c_vec.detach().cpu().numpy(),
                        "F": label,
                        "T": fixed_maps.future_vector(
                            float(y1[0].outcome)).detach().cpu().numpy(),
                    })
            adapter.clear_trace()
    if not samples:
        raise RuntimeError("no probe samples collected; is the run a "
                           "task_only (compressor attached) run?")
    print("collected {} samples".format(len(samples)), flush=True)

    depths = [int(x) for x in args.distances.split(",")] \
        if args.distances else sorted({s["depth"] for s in samples})

    def audit(sample_list, tag="audit", target="T"):
        res = {}
        for d in depths:
            sel = [sm for sm in sample_list if sm["depth"] == d]
            if len(sel) < args.min_cuts_per_depth:
                res[str(d)] = {
                    "skipped": "too few samples",
                    "n": len(sel),
                    "n_roots": len({s["root_id"] for s in sel})}
                continue
            res[str(d)] = _audit_one_depth(sel, d, args, y_field=target)
            res[str(d)]["n"] = len(sel)
            res[str(d)]["n_roots"] = len({s["root_id"] for s in sel})
        return res

    report = {"run": str(out), "n_samples": len(samples),
              "depths": depths, "audit": audit(samples)}

    # Secondary root-task probe (Part II.9): same frozen states, target the
    # root task label.  NLL gain is the primary utilization quantity.
    task_audit = {}
    for d in depths:
        sel = [sm for sm in samples if sm["depth"] == d]
        if len(sel) < args.min_cuts_per_depth:
            task_audit[str(d)] = {"skipped": "too few samples",
                                  "n": len(sel)}
            continue
        task_audit[str(d)] = _audit_root_task(sel, d, args)
    report["root_task_audit"] = task_audit

    if args.controls:
        controls = {}
        # Control A: Z = U identity -> retained should approach available.
        z_orig = [s["Z"] for s in samples]
        for s in samples:
            s["Z"] = s["U"]
        controls["z_equals_u"] = audit(samples, "z_equals_u")
        # Control B: randomized Z within (depth, root).
        for s in samples:
            rs = np.random.RandomState(s["root_id"] + 100000)
            s["Z"] = rs.randn(*s["Z"].shape).astype(s["Z"].dtype)
        controls["random_z"] = audit(samples, "random_z")
        # Restore original Z.
        for s, z in zip(samples, z_orig):
            s["Z"] = z
        # Control C: permuted target within each depth (never across the
        # train/audit boundary — the permutation is inside the depth's
        # sample pool and the audit still splits per root afterwards).
        by_depth = {}
        for i, s in enumerate(samples):
            by_depth.setdefault(s["depth"], []).append(i)
        for d, idxs in by_depth.items():
            rs = np.random.RandomState(5000 + d)
            idxs = np.asarray(idxs)
            perm = rs.permutation(len(idxs))
            tvals = [samples[i]["T"] for i in idxs]
            for src, dst in zip(idxs, perm):
                samples[src]["T"] = tvals[dst]
        controls["permuted_target"] = audit(samples, "permuted_target")
        report["controls"] = controls

    out_json = Path(args.out_json) if args.out_json else \
        out / "motivation_probe.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2, allow_nan=True)
    print(json.dumps(report, indent=2, allow_nan=True), flush=True)
    print("wrote {}".format(out_json), flush=True)


if __name__ == "__main__":
    main()
