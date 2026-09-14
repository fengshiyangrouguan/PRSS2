#!/usr/bin/env python3
"""Cross-model root-state readout on canonical, model-independent targets.

For each model m and each hop h in {1,2,3}:

    J_root^(h,m) = CE(q0(S^(h) | B^(h))) - CE(q1(S^(h) | B^(h), Z_root^m))

with
    S^(1) = (Y_a1, Y_root),  S^(2) = (Y_a2, Y_a1),  S^(3) = (Y_leaf, Y_a2)
    B^(h) = (C_shared from the canonical path, the ordered candidate pair)
    Z_root^m = the model's OWN natively-formed root state

The targets and the inputs B are built once, from the canonical model-free
manifest, and are identical for every model.  Each model's Z_root comes from its
own native forward (its own neighbour sampler, recursion and memory), so the
question answered is:

    for the same model-independent h-hop historical source target, how much
    readable predictive signal does each model's natively-formed root state
    retain?

A model whose native sampling never covered that long-range source simply reads
out less -- that IS its effective long-range coverage; nothing is forced.

Everything here is pure numpy.  Reuses the probe family and the lambda/CV
machinery from ``retention_probe_joint``.

Usage:
    python scripts/retention_root_joint.py \
        --canonical canonical_manifest.json \
        --model ours=zroot_ours_seed0.pkl taskonly=... tgn=... tgat=... \
        --data-dir <.../uci> --out-dir root_joint
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

import retention_probe_joint as rj                       # noqa: E402

LAMS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
HOP_NODES = {1: ("a1", "root"), 2: ("a2", "a1"), 3: ("leaf", "a2")}


def _fixed_proj(x, dim, seed):
    x = np.asarray(x, np.float64).reshape(-1)
    rng = np.random.RandomState(seed)
    W = rng.normal(0.0, 1.0 / np.sqrt(max(1, len(x))), size=(dim, len(x)))
    return W @ x


def _hash01v(v, seed):
    return ((np.asarray([v], np.int64) * 2654435761 + seed) & 0xFFFFFFFF) \
        / float(2 ** 32)


def ctx_shared_canonical(ev, edge_feat_rows):
    """The 32-d model-independent context of the canonical path.

    Same field list as audit_retention_v2's shared block: the three hop edges'
    projected features, plus root/leaf id hashes and log-scaled node/edge times.
    Nothing model-dependent enters.
    """
    pt = ev["path_times"]
    nt = [pt["a2"], pt["a1"], ev["t_root"]]
    parts = [_fixed_proj(edge_feat_rows[i], 8, 700 + i) for i in range(3)]
    parts.append(np.asarray([
        _hash01v(ev["root_node"], 401)[0], _hash01v(ev["path"]["leaf"], 402)[0],
        np.log1p(nt[0]) / 12.0, np.log1p(nt[1]) / 12.0, np.log1p(nt[2]) / 12.0,
        np.log1p(nt[0]) / 12.0, np.log1p(nt[1]) / 12.0,
        np.log1p(ev["t_root"]) / 12.0], np.float64))
    out = np.concatenate([np.asarray(p, np.float64).reshape(-1) for p in parts])
    assert out.size == 32, out.size
    return out


def _probs(params, Phi, C, Z):
    """4-class probabilities of the fitted joint probe."""
    Cs = np.asarray(C, np.float64) / params["sC"]
    if params["use_state"]:
        Zs = np.asarray(Z, np.float64) / params["sZ"]
    else:
        Zs = np.zeros((len(Phi), 0))
    lg = rj._logits(params["w_phi"], params["W_C"],
                    params["W_Z"] if params["use_state"] else None,
                    Phi, Cs, Zs)
    m = lg.max(axis=1, keepdims=True)
    e = np.exp(lg - m)
    return e / e.sum(axis=1, keepdims=True)


def marginal_j(base, full, Phie, Ce, Ze, y_node, which):
    """Per-NODE readout, marginalized out of the SAME fitted joint probe.

    ``which=0`` marginalises to the source node's bit (the high bit of
    ``class = 2*Y_src + Y_parent``), ``which=1`` to the parent's bit.  Using one
    probe for the joint and both marginals keeps the three hop readouts strictly
    comparable.
    """
    p0 = _probs(base, Phie, Ce, Ze)
    p1 = _probs(full, Phie, Ce, Ze)
    if which == 0:
        P0 = np.stack([p0[:, 0] + p0[:, 1], p0[:, 2] + p0[:, 3]], axis=1)
        P1 = np.stack([p1[:, 0] + p1[:, 1], p1[:, 2] + p1[:, 3]], axis=1)
    else:
        P0 = np.stack([p0[:, 0] + p0[:, 2], p0[:, 1] + p0[:, 3]], axis=1)
        P1 = np.stack([p1[:, 0] + p1[:, 2], p1[:, 1] + p1[:, 3]], axis=1)
    i = np.arange(len(y_node))
    n0 = float(-np.log(np.clip(P0[i, y_node], 1e-12, 1.0)).mean())
    n1 = float(-np.log(np.clip(P1[i, y_node], 1e-12, 1.0)).mean())
    return (n0 - n1) / np.log(2.0), n0, n1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--canonical", required=True)
    ap.add_argument("--model", nargs="+", required=True, metavar="NAME=ZROOT_PKL")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--data-name", default="uci")
    ap.add_argument("--svd-rank", type=int, default=16)
    ap.add_argument("--fit-frac", type=float, default=0.7,
                    help="earlier fraction (by t_root) used to fit; the rest is "
                         "the held-out evaluation block")
    ap.add_argument("--n-folds", type=int, default=3)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    from rpbe.data.uci_link import UCILinkDataset
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    man = json.load(open(args.canonical))
    events = {int(e["raw_row"]): e for e in man["events"]}
    print("[canonical] {} events (sha {})".format(
        len(events), str(man.get("manifest_sha"))[:16]), flush=True)

    models = {}
    for spec in args.model:
        name, path = spec.split("=", 1)
        with open(path, "rb") as f:
            d = pickle.load(f)
        models[name] = {"zroot": d["zroot"], "meta": d["meta"]}
        print("[model] {} {} roots".format(name, len(d["zroot"])), flush=True)

    ds = UCILinkDataset(args.data_dir, data_name=args.data_name)
    edge_feat_all = np.asarray(ds.edge_features, np.float64)

    # rows present in EVERY model and in the canonical manifest
    common = set(events)
    for m in models.values():
        common &= {int(k) for k in m["zroot"]}
    rows = sorted(common)
    print("[align] {} canonical root events shared by all models "
          "(models cover {})".format(
              len(rows), {k: len(v["zroot"]) for k, v in models.items()}),
          flush=True)
    if not rows:
        raise SystemExit("no shared canonical root events")

    t_root = np.array([events[r]["t_root"] for r in rows], np.float64)
    order = np.argsort(t_root, kind="stable")
    cut = int(round(args.fit_frac * len(order)))
    fit_idx, eval_idx = order[:cut], order[cut:]
    print("[split] fit={} eval={} (t_fit<= {:.0f} < t_eval)".format(
        len(fit_idx), len(eval_idx), t_root[fit_idx].max()), flush=True)

    # shared encoder: train prefix strictly before the earliest canonical fit row
    keep = np.asarray(ds.train.timestamps, np.float64) < float(t_root.min())
    E, emb = rj.build_svd_encoder(ds.train.sources[keep],
                                  ds.train.destinations[keep],
                                  int(ds.n_nodes), rank=args.svd_rank,
                                  random_state=0)
    emb["cutoff_time"] = float(t_root.min())
    print("[encoder]", {k: emb[k] for k in ("svd_rank", "embedding_hash",
                                            "cutoff_time")}, flush=True)

    report = {"canonical_sha": man.get("manifest_sha"), "encoder": emb,
              "n_shared_roots": len(rows), "fit_frac": args.fit_frac,
              "models": sorted(models), "hops": {}}

    for h, (sk, pk) in HOP_NODES.items():
        C, Phi, y, y_src, y_par = [], [], [], [], []
        for r in rows:
            ev = events[r]
            er = [edge_feat_all[ev["path_edge_rows"][k]]
                  for k in ("a1", "a2", "leaf")]
            C.append(ctx_shared_canonical(ev, er))
            c_s, c_p = ev["cands"][sk], ev["cands"][pk]
            osp = (c_s["presented"],
                   c_s["neg"] if c_s["presented"] == c_s["pos"] else c_s["pos"])
            opp = (c_p["presented"],
                   c_p["neg"] if c_p["presented"] == c_p["pos"] else c_p["pos"])
            Phi.append(rj.phi_tensor(E, np.asarray([osp]), np.asarray([opp]))[0])
            y.append(2 * c_s["Y"] + c_p["Y"])
            y_src.append(int(c_s["Y"]))
            y_par.append(int(c_p["Y"]))
        C, Phi = np.stack(C), np.stack(Phi)
        y = np.asarray(y, np.int64)
        y_src, y_par = np.asarray(y_src, np.int64), np.asarray(y_par, np.int64)
        e2i = {r: i for i, r in enumerate(rows)}
        fi = np.asarray([e2i[rows[i]] for i in fit_idx])
        ei = np.asarray([e2i[rows[i]] for i in eval_idx])

        # expanding-window folds inside the fit block, ordered by root time
        fo = np.argsort(t_root[fi], kind="stable")
        folds = []
        for k in range(1, 3):
            c = int(round(len(fo) * (k + 1) / 3))
            if c - 20 >= 20 and len(fo) - c >= 20:
                folds.append((fi[fo[:c - 20]], fi[fo[c:]]))
        if not folds:
            folds = [(fi[fo[:len(fo) // 2]], fi[fo[len(fo) // 2:]])]

        Cc, Phic, yc = C[fi], Phi[fi], y[fi]
        Ce, Phie, ye = C[ei], Phi[ei], y[ei]
        base, cv_base, _ = rj.select_joint(Phic, Cc, np.zeros((len(fi), 1)),
                                           yc, folds, lams=LAMS, use_state=False)
        nll0 = rj.joint_row_nll(base, Phie, Ce, np.zeros((len(ei), 1)), ye)
        entry = {"n_fit": len(fi), "n_eval": len(ei),
                 "class_counts_eval": rj.class_counts(ye),
                 "base_nll": float(nll0.mean()), "models": {}}
        for name, m in models.items():
            Z = np.stack([np.asarray(m["zroot"][r], np.float64) for r in rows])
            Zc, Ze = Z[fi], Z[ei]
            full, cv_full, diag = rj.select_joint(Phic, Cc, Zc, yc, folds,
                                                  lams=LAMS, use_state=True)
            nll1 = rj.joint_row_nll(full, Phie, Ce, Ze, ye)
            j = float((nll0.mean() - nll1.mean()) / np.log(2.0))
            # paired time-block bootstrap over the evaluation roots
            blocks = rj.time_blocks(t_root[ei], 20)
            idx = rj.bootstrap_indices(blocks, args.n_boot, args.seed)
            reps = rj.paired_bootstrap(idx, nll0, {0: nll1})[0]
            lo, hi = np.percentile(reps, [2.5, 97.5])
            entry["models"][name] = {
                "J_bits": j, "J_millibits": 1000.0 * j,
                "J_ci95_bits": [float(lo), float(hi)],
                "nll_full_mean": float(nll1.mean()),
                "lam": float(full["lam"]),
                "n_cv": float(cv_full)}
            # per-NODE readouts, marginalized out of the same joint probe:
            # hop(h) = (src=.., parent=..) so J_src is the h-hop source node's own
            # future and J_parent the next-inward node's
            js, ns0, ns1 = marginal_j(base, full, Phie, Ce, Ze, y_src[ei], 0)
            jp, npp0, npp1 = marginal_j(base, full, Phie, Ce, Ze, y_par[ei], 1)
            entry["models"][name]["J_src_bits"] = float(js)
            entry["models"][name]["J_src_millibits"] = 1000.0 * float(js)
            entry["models"][name]["J_parent_bits"] = float(jp)
            entry["models"][name]["J_parent_millibits"] = 1000.0 * float(jp)
            entry["models"][name]["src_node"] = sk
            entry["models"][name]["parent_node"] = pk
        report["hops"][str(h)] = entry
        print("[hop{}] base NLL={:.5f}".format(h, entry["base_nll"]), flush=True)
        for name, e in entry["models"].items():
            print("    {:<9} J={:+.6f} bits ({:+.3f} mbit) CI=[{:+.6f},{:+.6f}]"
                  .format(name, e["J_bits"], e["J_millibits"],
                          e["J_ci95_bits"][0], e["J_ci95_bits"][1]), flush=True)

    with open(out_dir / "root_joint.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print("[out]", out_dir / "root_joint.json", flush=True)


if __name__ == "__main__":
    main()
