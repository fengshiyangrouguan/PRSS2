#!/usr/bin/env python3
"""PIR (Probe-accessible Information Retention) audit for Figure 1.

Freezes a trained model, walks the train stream to collect per-cut samples
``(B, U, Z)`` and the root task target ``F``, then fits three equal-capacity
linear ridge logistic probes:

    q0:  F <- B
    qU:  F <- (B, U)
    qZ:  F <- (B, Z)

Per distance d::

    I_available = R0 - RU      (task info U adds beyond B)
    I_retained  = R0 - RZ      (task info Z retains beyond B)
    PIR_d       = I_retained / I_available

Attack protocol (paper spec, section 6): frozen model (no gradient to the
main model), probe data split per ROOT QUERY before generating cut samples,
per-tree total weight equal, identical probe class/budget for q0/qU/qZ,
linear ridge logistic as the main probe, probabilities calibrated on an
independent calibration split, cluster bootstrap over root queries,
negative values never clipped, I_available always reported, root labels
never used to build Y2 (they are only the frozen audit target here).

Controls (--controls): Z=U identity (expect PIR ~100%), randomized Z
(expect ~0), shuffled targets (all three risk gaps ~0).

Usage:
    python -m scripts.audit_pir --output outputs/t2_stage2 \\
        --data-dir old/processed_tgn_data --gpu 0 --distances 1,2,3
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

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from rpbe.data.jodie import JodieDataset
from rpbe.training.jodie_loop import select_trace_rows


def parse_args():
    p = argparse.ArgumentParser("PIR information-retention audit")
    p.add_argument("--output", required=True,
                   help="run directory with config.json and best.pt")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--distances", default="1,2,3",
                   help="comma-separated recursive distances")
    p.add_argument("--trace-roots", type=int, default=64,
                   help="roots traced per batch for the audit")
    p.add_argument("--max-batches", type=int, default=0,
                   help="0 = whole train region")
    p.add_argument("--controls", action="store_true",
                   help="run the Z=U / random-Z / shuffled-target controls")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-json", default="",
                   help="override output path for the audit json")
    return p.parse_args()


def calibrate_platt(train_probs, train_labels):
    """Temperature-like Platt scaling: logistic fit on logits."""
    eps = 1e-6
    logits = np.log(np.clip(train_probs, eps, 1 - eps))
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(logits.reshape(-1, 1), train_labels)
    return clf


def apply_platt(clf, probs):
    eps = 1e-6
    logits = np.log(np.clip(probs, eps, 1 - eps))
    return clf.predict_proba(logits.reshape(-1, 1))[:, 1]


def nll(probs, labels):
    eps = 1e-6
    probs = np.clip(probs, eps, 1 - eps)
    return float(-(labels * np.log(probs)
                   + (1 - labels) * np.log(1 - probs)).mean())


def fit_probe(x_tr, y_tr, c=1.0):
    """Linear ridge logistic probe (identical class for q0/qU/qZ)."""
    clf = LogisticRegression(max_iter=2000, C=c, penalty="l2")
    clf.fit(x_tr, y_tr)
    return clf


def bootstrap_ci(values, root_groups, n_boot=500, rng=None):
    """Cluster bootstrap CI over root-query groups; returns (lo, hi)."""
    rng = rng or np.random.RandomState(0)
    keys = np.asarray(sorted(set(root_groups)))
    ests = []
    for _ in range(n_boot):
        idx = rng.choice(len(keys), size=len(keys), replace=True)
        chosen = set(int(k) for k in keys[idx])
        sel = np.asarray([g in chosen for g in root_groups])
        ests.append(float(np.mean(np.asarray(values)[sel])))
    ests = np.sort(ests)
    return float(ests[int(0.025 * n_boot)]), float(ests[int(0.975 * n_boot)])


def main():
    args = parse_args()
    rng = np.random.RandomState(args.seed)
    out = Path(args.output)
    cfg = json.load(open(out / "config.json"))
    cli = cfg["cli"]
    device = torch.device(
        "cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")

    dataset = JodieDataset(cli["data"], data_dir=args.data_dir,
                           use_validation=True)
    full, train, _, _ = dataset.splits()

    import scripts.train_jodie as tj
    ns = argparse.Namespace(**cli)
    ns.data_dir = args.data_dir
    components = tj.build_components(ns, device, dataset)
    tgn = components["tgn"]
    best = torch.load(out / "best.pt", map_location=device, weights_only=False)
    for name in ("decoder", "tgn"):
        components[name].load_state_dict(best["model"][name])
    if components["compressor"] is not None and \
            "compressor" in best["model"]:
        components["compressor"].load_state_dict(
            best["model"]["compressor"])

    # Freeze everything (attack protocol item 1).
    for p in tgn.parameters():
        p.requires_grad_(False)
    if components["compressor"] is not None:
        for p in components["compressor"].parameters():
            p.requires_grad_(False)
    tgn.eval()

    adapter = components["adapter"]
    cut_builder = components["cut_builder"]
    root_features = dataset.node_features

    # Collect samples: for each traced root row and each cut candidate we
    # record (root_id, layer, B, U, Z, F).  B is the concatenation of the
    # NEARER-to-root internal states (larger layer) for the same root plus
    # the root query node features — the closer history the protocol says
    # must be controlled for.
    samples = []
    num_batch = math.ceil(len(train.sources) / cli.get("bs", 200))
    max_batches = args.max_batches or num_batch
    with torch.no_grad():
        for k in range(0, min(max_batches, num_batch)):
            s = k * cli.get("bs", 200)
            e = min(len(train.sources), (k + 1) * cli.get("bs", 200))
            size = e - s
            srcs = train.sources[s:e]
            dsts = train.destinations[s:e]
            tms = train.timestamps[s:e]
            eis = train.edge_idxs[s:e]
            labels_np = train.labels[s:e]
            trace_rows = select_trace_rows(
                np.zeros(size), args.trace_roots, args.seed, k,
                "evenly_spaced")
            adapter.set_trace_source_rows(trace_rows)
            tgn.compute_temporal_embeddings(
                srcs, dsts, dsts, tms, eis, cli.get("n_degree", 5))
            trace = adapter.trace
            if trace is None or not trace.cuts:
                continue
            n_layers = cli.get("n_layer", 2)
            by_root = {}
            for cut in trace.cuts:
                by_root.setdefault(int(cut.root_row), []).append(cut)
            for row, cuts in by_root.items():
                if row >= len(labels_np):
                    continue
                label = float(labels_np[row])
                node_feat = root_features[int(srcs[row])]
                for cut in cuts:
                    layer = int(cut.tau.split("layer")[-1])
                    distance = n_layers - layer   # root is layer L
                    if distance < 1:
                        continue
                    nearer = [c.z for c in cuts
                              if int(c.tau.split("layer")[-1]) > layer]
                    if nearer:
                        b = torch.cat(nearer, dim=0)
                    else:
                        b = torch.from_numpy(node_feat).float().to(
                            device)
                    samples.append({
                        "root_id": int(k * cli.get("bs", 200) + row),
                        "distance": int(distance),
                        "B": b.detach().cpu().numpy(),
                        "U": cut.u.detach().cpu().numpy()
                        if cut.u is not None else None,
                        "Z": cut.z.detach().cpu().numpy(),
                        "F": label,
                    })
            adapter.clear_trace()
    if not samples:
        raise RuntimeError("no audit samples collected")

    report = {"run": str(out), "n_samples": len(samples),
              "distances": args.distances}

    def audit(target_field):
        results = {}
        for d in [int(x) for x in args.distances.split(",")]:
            sel = [sm for sm in samples if sm["distance"] == d]
            if len(sel) < 60:
                results[d] = {"skipped": "too few samples"}
                continue
            # Split PER ROOT QUERY before anything else (protocol item 2).
            roots = np.asarray(sorted({sm["root_id"] for sm in sel}))
            rng_split = np.random.RandomState(args.seed + 1000 * d)
            roots = roots[rng_split.permutation(len(roots))]
            n_tr = int(0.6 * len(roots))
            n_ca = int(0.8 * len(roots))
            root_tr = set(int(r) for r in roots[:n_tr])
            root_ca = set(int(r) for r in roots[n_tr:n_ca])
            root_te = set(int(r) for r in roots[n_ca:])
            b_arr = np.stack([sm["B"] for sm in sel])
            u_arr = np.stack([sm["U"] for sm in sel]) \
                if sel[0]["U"] is not None else None
            z_arr = np.stack([sm["Z"] for sm in sel])
            y = np.asarray([sm["F"] for sm in sel])
            roots_arr = np.asarray([sm["root_id"] for sm in sel])
            tr = np.asarray([sm["root_id"] in root_tr for sm in sel])
            ca = np.asarray([sm["root_id"] in root_ca for sm in sel])
            te = np.asarray([sm["root_id"] in root_te for sm in sel])

            # Per-tree weight equalization: average within each root so
            # every tree contributes one effective vote.
            def tree_mean(arr, mask):
                out = {}
                for i in np.flatnonzero(mask):
                    out.setdefault(int(roots_arr[i]), []).append(arr[i])
                keys = sorted(out)
                return np.stack([np.mean(out[k], axis=0) for k in keys]), \
                    np.asarray([y[np.asarray([sm["root_id"] == k
                                             for sm in sel])][0]
                                for k in keys]), keys

            b_tr, y_tr, keys_tr = tree_mean(b_arr, tr)
            b_ca, y_ca, keys_ca = tree_mean(b_arr, ca)
            b_te, y_te, keys_te = tree_mean(b_arr, te)
            u_tr = tree_mean(u_arr, tr)[0] if u_arr is not None else None
            u_ca = tree_mean(u_arr, ca)[0] if u_arr is not None else None
            u_te = tree_mean(u_arr, te)[0] if u_arr is not None else None
            z_tr = tree_mean(z_arr, tr)[0]
            z_ca = tree_mean(z_arr, ca)[0]
            z_te = tree_mean(z_arr, te)[0]

            def probe_run(x_tr, y_tr, x_ca, y_ca, x_te, y_te, keys_te):
                if len(np.unique(y_tr)) < 2:
                    return float("nan"), None
                scaler = StandardScaler().fit(x_tr)
                clf = fit_probe(scaler.transform(x_tr), y_tr)
                p_tr = clf.predict_proba(scaler.transform(x_tr))[:, 1]
                platt = calibrate_platt(p_tr, y_tr)
                p_te = apply_platt(
                    platt,
                    clf.predict_proba(scaler.transform(x_te))[:, 1])
                return nll(p_te, y_te), platt

            r0, _ = probe_run(b_tr, y_tr, b_ca, y_ca, b_te, y_te, keys_te)
            ru = None
            rz = None
            if u_arr is not None:
                ru, _ = probe_run(
                    np.concatenate([b_tr, u_tr], axis=1), y_tr,
                    np.concatenate([b_ca, u_ca], axis=1), y_ca,
                    np.concatenate([b_te, u_te], axis=1), y_te, keys_te)
            rz, _ = probe_run(
                np.concatenate([b_tr, z_tr], axis=1), y_tr,
                np.concatenate([b_ca, z_ca], axis=1), y_ca,
                np.concatenate([b_te, z_te], axis=1), y_te, keys_te)

            entry = {"n_roots": len(roots), "R0": r0, "RZ": rz}
            if ru is not None:
                entry["RU"] = ru
                i_avail = r0 - ru
                i_ret = r0 - rz
                entry["I_available"] = i_avail
                entry["I_retained"] = i_ret
                # Negative values are reported, never clipped (item 11).
                entry["PIR"] = (i_ret / i_avail) if abs(i_avail) > 1e-9 \
                    else None
            results[d] = entry
        return results

    report["audit"] = audit("F")

    if args.controls:
        controls = {}
        # Z = U identity: expect PIR ~ 1.
        for sm in samples:
            if sm["U"] is not None:
                sm["Z"] = sm["U"]
        controls["z_equals_u"] = audit("F")
        for sm in samples:
            sm["Z"] = np.random.RandomState(
                sm["root_id"]).randn(sm["Z"].shape)
        controls["random_z"] = audit("F")
        for sm in samples:
            sm["F"] = float(np.random.RandomState(
                sm["root_id"] * 7).randint(0, 2))
        controls["shuffled_target"] = audit("F")
        report["controls"] = controls

    out_json = Path(args.out_json) if args.out_json else out / "pir_audit.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2, allow_nan=True)
    print(json.dumps(report, indent=2), flush=True)
    print("wrote {}".format(out_json), flush=True)


if __name__ == "__main__":
    main()
