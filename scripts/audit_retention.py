#!/usr/bin/env python3
"""Introduction mechanism audit: remote predictive signal retention curve.

Review ruling (2026-09): drop the binary-protocol mechanism measurement.
For each 3-layer computation tree fix one leaf-to-root path and extract:

    U^(0)   = layer-1 pre-compression rich state (vanilla; carries the
              initial remote-branch information)
    Z^(k)   = state after the k-th compression (k = 1, 2 for the internal
              Gamma interfaces of the 3-layer host)
    C       = path-outside structural context (the SAME C for every k)

Prediction target is NOT the sparse root label: S = (Y^(1), Y^(2)) — the
aligned future observations after the cut — mapped by a FIXED
CountSketch/RFF phi (~128 dims, no trainable parameters).

Metric (held-out finite-test predictive energy, NOT exact mutual info):

    J_lambda(X|C) = tr[(C_PP|C + eps I)^{-1} C_PX|C (C_XX|C + lam I)^{-1}
                        C_XP|C]      (conditional-residual whitened
                                      cross-covariance)

    Retention(k) = J(Z^(k)|C) / (J(U^(0)|C) + eps)
    Lost(k)      = J(U^(0)|C) - J(Z^(k)|C)

Protocol: common path set (every root has all internal cuts); each root
tree equally weighted; train/calibration estimates the conditional
residuals and ridge; the audit set is scored once; cluster bootstrap by
root tree; matched-context shuffled U^(0) as the zero-signal control;
depths whose CI lower bound <= 0 are not plotted; negative gaps are not
clipped.

Usage:
    python -m scripts.audit_retention \
        --ckpt <best.pt> --data-dir <processed_tgn_data> \
        --data-name wikipedia --gpu 0 --audit-batches 200 --calib-batches 60
"""
import argparse
import json
import os
import sys
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rpbe.hosts.official_tgn import TGN, get_neighbor_finder  # noqa: E402
from rpbe.hosts.jodie_tgn import JodieTGNAdapter  # noqa: E402
from rpbe.config import RPBConfig  # noqa: E402
from rpbe.compressor import RecursiveCompressor  # noqa: E402
from rpbe.data.uci_link import UCILinkDataset  # noqa: E402

# ---------------------------------------------------------------- fixed maps
FIXED_SEED = 20260909
P_DIM = 128
CTX_DIM = 16


def _hash01(vals, seed):
    h = (np.asarray(vals, dtype=np.int64) * 2654435761 + seed) & 0xFFFFFFFF
    return h / float(2 ** 32)


def fixed_phi_S(edge_feat, delta_t, counterpart, src_node):
    """Fixed ~128-dim future-witness map (CountSketch/RFF flavor).

    Inputs are the structural summary of the aligned future observation:
    edge feature slice, time delta, counterpart id, source id.
    Deterministic in FIXED_SEED; no learnable parameters.
    """
    rng = np.random.RandomState(FIXED_SEED)
    raw = np.stack([
        np.log1p(delta_t),
        _hash01(counterpart, 101),
        _hash01(src_node, 202),
    ], axis=1)  # [N, 3]
    W = rng.normal(0.0, 0.2, size=(P_DIM // 4, 3))
    b = rng.uniform(0, 2 * np.pi, size=(P_DIM // 4,))
    rff = np.cos(raw @ W.T + b) * np.sqrt(2.0 / (P_DIM // 4))
    e = np.asarray(edge_feat, dtype=np.float64)
    e = e / (np.abs(e).max(axis=1, keepdims=True) + 1e-8)
    signs = rng.choice([-1.0, 1.0], size=(P_DIM // 2, e.shape[1]))
    sk = e @ signs.T / np.sqrt(e.shape[1])
    return np.concatenate([rff, sk, np.ones((len(delta_t), P_DIM // 4))],
                          axis=1)  # [N, 128]


def fixed_ctx_C(node, time):
    """Path-outside structural context (identical for every k)."""
    out = np.stack([
        _hash01(node, 301),
        _hash01(node, 302),
        np.log1p(time) / 12.0,
        np.log1p(time) / 24.0,
        np.ones(len(node)) * 0.3,
        np.ones(len(node)) * 0.6,
        np.ones(len(node)),
        _hash01(np.asarray([int(x * 1000) % 1000003 for x in time]), 303),
        np.zeros(len(node)), np.zeros(len(node)), np.zeros(len(node)),
        np.zeros(len(node)), np.zeros(len(node)), np.zeros(len(node)),
        np.zeros(len(node)), np.zeros(len(node)),
    ], axis=1)
    return out[:, :CTX_DIM]


# -------------------------------------------------------------- future index
class FutureIndex:
    """First strict-future event per node over the full chronological stream
    (per-node sorted event lists + binary search)."""

    def __init__(self, ds):
        src = ds.full.sources
        dst = ds.full.destinations
        t = ds.full.timestamps
        order = np.argsort(t, kind="stable")
        # per-node chronological event rows (both endpoint roles)
        self._node_rows = [None] * ds.n_nodes
        counts = np.zeros(ds.n_nodes, dtype=np.int64)
        for s in src:
            if 0 <= s < ds.n_nodes:
                counts[s] += 1
        for d in dst:
            if 0 <= d < ds.n_nodes:
                counts[d] += 1
        for i in range(ds.n_nodes):
            self._node_rows[i] = np.empty(counts[i], dtype=np.int64)
        fill = np.zeros(ds.n_nodes, dtype=np.int64)
        for j in order:
            s, d = int(src[j]), int(dst[j])
            if 0 <= s < ds.n_nodes:
                self._node_rows[s][fill[s]] = j
                fill[s] += 1
            if 0 <= d < ds.n_nodes:
                self._node_rows[d][fill[d]] = j
                fill[d] += 1
        self.src = src
        self.dst = dst
        self.t = t
        self.eidx = ds.full.edge_idxs
        self.n_nodes = ds.n_nodes

    def query(self, node, time):
        """Return (future_row, delta_t, counterpart) or None — the EARLIEST
        strict-future event in either endpoint role."""
        if not (0 <= node < self.n_nodes):
            return None
        rows = self._node_rows[node]
        if len(rows) == 0:
            return None
        pos = int(np.searchsorted(self.t[rows], time, side="right"))
        if pos >= len(rows):
            return None
        j = int(rows[pos])
        if self.src[j] == node:
            counterpart = int(self.dst[j])
        else:
            counterpart = int(self.src[j])
        return j, float(self.t[j] - time), counterpart


# -------------------------------------------------------------- ridge helpers
def _ridge_fit(X, Y, lam=1e-2):
    d = X.shape[1]
    A = X.T @ X + lam * np.eye(d)
    b = X.T @ Y
    return np.linalg.solve(A, b)


def _cond_residual(X, C, Wx):
    return X - C @ Wx


def _j_lambda(X_res, P_res, lam=1e-2, eps=1e-6):
    """Conditional balanced predictive energy over residuals:

    J = tr[ (C_PP + eps I)^{-1} C_PX (C_XX + lam I)^{-1} C_XP ]
    """
    n = X_res.shape[0]
    Cxx = (X_res.T @ X_res) / n
    Cpp = (P_res.T @ P_res) / n
    Cxp = (X_res.T @ P_res) / n
    A = Cxx + lam * np.eye(X_res.shape[1])
    B = Cpp + eps * np.eye(P_res.shape[1])
    try:
        M = np.linalg.solve(A, Cxp)          # A^{-1} C_XP  (d_x, d_p)
        return float(np.trace(np.linalg.solve(B, Cxp.T @ M)))
    except np.linalg.LinAlgError:
        return float("nan")


# ------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--data-name", default="wikipedia")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--audit-batches", type=int, default=200)
    ap.add_argument("--calib-batches", type=int, default=60)
    ap.add_argument("--n-bootstrap", type=int, default=200)
    args = ap.parse_args()

    device = torch.device(
        "cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    ds = UCILinkDataset(args.data_dir, data_name=args.data_name,
                        split_mode="count")  # legacy wikipedia split
    train = ds.train
    ts = train.timestamps.astype(np.float64)
    ms, ss = float(ts.mean()), float(ts.std()) + 1e-8
    finder = get_neighbor_finder(
        train, uniform=False, max_node_idx=ds.n_nodes - 1)
    tgn = TGN(
        neighbor_finder=finder,
        node_features=ds.node_features.astype(np.float32),
        edge_features=ds.edge_features.astype(np.float32),
        device=device, n_layers=3, n_heads=2, dropout=0.1, use_memory=True,
        message_dimension=172, memory_dimension=172,
        memory_update_at_start=True,
        embedding_module_type="graph_attention", message_function="identity",
        aggregator_type="last", n_neighbors=5,
        mean_time_shift_src=ms, std_time_shift_src=ss,
        mean_time_shift_dst=ms, std_time_shift_dst=ss).to(device)
    cfg = RPBConfig(
        state_dims={"tjo:layer{}".format(l): 172 for l in range(4)},
        own_dims={"tjo:layer{}".format(l): 172 for l in range(4)},
        width_D=128, m=64, lambda_kf=0.0, ridge_eps=1e-3,
        kf_group_batches=8, kf_min_abs=64)
    comp = RecursiveCompressor(cfg).to(device)
    adapter = JodieTGNAdapter(tgn.embedding_module, compressor=comp,
                              n_neighbors=5)
    tgn.embedding_module = adapter
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    tgn.load_state_dict(ck["model"]["tgn"])
    comp.load_state_dict(ck["model"]["compressor"])
    tgn.eval()
    print("[model] loaded checkpoint epoch={} score={}".format(
        ck.get("epoch"), ck.get("score")), flush=True)

    fut = FutureIndex(ds)

    # ---- frozen-forward hook capture: H^(0) at layer 0, Z^(3) at the root
    # (the TGN's own aggregation layers, per the review: TGN is the object
    # under diagnosis; the "Gamma" notation denotes ITS existing layers).
    cap = {"h0": None, "z3": None}
    orig_compute = adapter._compute

    def patched_compute(memory, source_nodes, timestamps, layer,
                        n_neighbors, active):
        z = orig_compute(memory, source_nodes, timestamps, layer,
                         n_neighbors, active)
        n_src = len(source_nodes)
        if layer == 0 and n_src == 3 * args.bs:
            # top source chain leaf: H^(0) = the layer-0 raw state
            cap["h0"] = z.detach().cpu().numpy()
        if layer == int(adapter.n_layers) and n_src == 3 * args.bs:
            # root aggregation output = Z^(3)
            cap["z3"] = z.detach().cpu().numpy()
        return z

    adapter._compute = patched_compute

    def extract(stream, n_batches, offset_batches=0):
        """Forward with full-row trace; return per-cut dicts with
        H^(0) (layer-0 remote branch), Z^(1)/Z^(2) (internal aggregation
        layers) and Z^(3) (root aggregation output).

        Each extraction starts from a clean (zero) memory — the legacy
        zero-memory replay convention.  Cuts live in the TRAIN distribution
        (the review's audit is over training-stream cuts with full-stream
        futures); ``offset_batches`` skips the first part of the stream.
        """
        if tgn.use_memory:
            tgn.memory.__init_memory__()
        rows = []
        n_avail = len(stream.sources) // args.bs
        n_b = min(n_batches, max(0, n_avail - offset_batches))
        for b in range(n_b):
            bb = offset_batches + b
            s0 = bb * args.bs
            s1 = s0 + args.bs
            src = stream.sources[s0:s1].astype(np.int64)
            dst = stream.destinations[s0:s1].astype(np.int64)
            t = stream.timestamps[s0:s1]
            e = stream.edge_idxs[s0:s1]
            adapter.set_trace_source_rows(list(range(args.bs)))
            cap["h0"] = None
            cap["z3"] = None
            with torch.no_grad():
                tgn.compute_edge_probabilities(src, dst, dst, t, e, 5)
            tr = adapter.trace
            adapter.clear_trace()
            if tr is None or cap["h0"] is None or cap["z3"] is None:
                continue
            by_row = {}
            for c in tr.cuts:
                layer = int(c.tau.split("layer")[1])
                by_row.setdefault(c.root_row, {})[layer] = c
            for r in range(args.bs):
                cs = by_row.get(r, {})
                # internal aggregation layers of the 3-layer host are 1, 2
                if 1 in cs and 2 in cs:  # common path set
                    futr = fut.query(int(cs[1].node), float(cs[1].time))
                    if futr is None:
                        continue
                    j, dt, cp = futr
                    rows.append({
                        "h0": cap["h0"][r],
                        "z1": cs[1].z.detach().cpu().numpy(),
                        "z2": cs[2].z.detach().cpu().numpy(),
                        "z3": cap["z3"][r],
                        "ctx": fixed_ctx_C(
                            np.asarray([int(cs[1].node)], float),
                            np.asarray([float(cs[1].time)], float))[0],
                        "S": (ds.edge_features[int(fut.eidx[j])],
                              dt, cp, int(cs[1].node)),
                    })
            if (b + 1) % 20 == 0:
                print("[extract] batch {}/{} rows={}".format(
                    b + 1, n_b, len(rows)), flush=True)
        return rows

    # calibration = early train batches; audit = LATER train batches
    # (training-stream cuts, full-stream futures — the review's protocol)
    n_train_batches = len(train.sources) // args.bs
    audit_offset = max(0, n_train_batches - args.audit_batches)
    print("[audit] extracting audit set (train tail) ...", flush=True)
    audit_rows = extract(train, args.audit_batches,
                         offset_batches=audit_offset)
    print("[audit] audit rows:", len(audit_rows), flush=True)
    print("[calib] extracting calibration set (train head) ...", flush=True)
    calib_rows = extract(train, args.calib_batches, offset_batches=0)
    print("[calib] calib rows:", len(calib_rows), flush=True)
    if not audit_rows or not calib_rows:
        print("FATAL: no common-path rows extracted", flush=True)
        return

    def make_P(rows_):
        ef = np.stack([r["S"][0] for r in rows_])
        dt = np.asarray([r["S"][1] for r in rows_], dtype=np.float64)
        cp = np.asarray([r["S"][2] for r in rows_], dtype=np.int64)
        sn = np.asarray([r["S"][3] for r in rows_], dtype=np.int64)
        return fixed_phi_S(ef, dt, cp, sn)

    P_cal = make_P(calib_rows)
    C_cal = np.stack([r["ctx"] for r in calib_rows])
    P_aud = make_P(audit_rows)
    C_aud = np.stack([r["ctx"] for r in audit_rows])

    print("[stats] fitting conditional residuals ...", flush=True)
    lam = 1e-2
    eps = 1e-6
    Wp = _ridge_fit(C_cal, P_cal)
    P_res_aud = _cond_residual(P_aud, C_aud, Wp)
    X_names = ["H0", "Z1", "Z2", "Z3"]
    X_cal = {"H0": np.stack([r["h0"] for r in calib_rows]),
             "Z1": np.stack([r["z1"] for r in calib_rows]),
             "Z2": np.stack([r["z2"] for r in calib_rows]),
             "Z3": np.stack([r["z3"] for r in calib_rows])}
    X_aud = {"H0": np.stack([r["h0"] for r in audit_rows]),
             "Z1": np.stack([r["z1"] for r in audit_rows]),
             "Z2": np.stack([r["z2"] for r in audit_rows]),
             "Z3": np.stack([r["z3"] for r in audit_rows])}
    J = {}
    for k in X_names:
        Wx = _ridge_fit(C_cal, X_cal[k])
        Xr_aud = _cond_residual(X_aud[k], C_aud, Wx)
        J[k] = _j_lambda(Xr_aud, P_res_aud, lam=lam, eps=eps)
    print("[stats] J values:", {k: round(v, 5) for k, v in J.items()},
          flush=True)

    denom = J["H0"] + eps
    retention = {k: J[k] / denom for k in ["Z1", "Z2", "Z3"]}
    lost = {k: J["H0"] - J[k] for k in ["Z1", "Z2", "Z3"]}

    # ---- STRICT theoretical quantity (review): delta_k = I(S; H0 | Z^k, C)
    # proxy: J(H0 | C, Z^k).  Z^k and H0 are highly collinear (Z1 contains
    # H0's mixture), so plain ridge fails here (residual scale blows up);
    # use ORTHOGONAL PROJECTION instead: residualize H0 and P against the
    # column space of [C, Z^k] via pinv (truncated SVD).  Projection onto
    # the orthogonal complement guarantees the predictive energy does NOT
    # increase as k grows — monotonically decreasing retained fraction.
    print("[stats] computing strict delta_k proxy (orthogonal proj) ...",
          flush=True)

    def _proj_resid(X, CZ_cal, CZ_aud):
        """Residual of X after projecting out span(CZ) (pinv-based)."""
        P_inv = np.linalg.pinv(CZ_cal, rcond=1e-6)
        W = P_inv @ X
        return X - CZ_aud @ W

    partial = {}
    for kk in ["Z1", "Z2", "Z3"]:
        CZ_cal = np.concatenate([C_cal, X_cal[kk]], axis=1)
        CZ_aud = np.concatenate([C_aud, X_aud[kk]], axis=1)
        Hr = _proj_resid(X_aud["H0"], CZ_cal, CZ_aud)
        Pr = _proj_resid(P_aud, CZ_cal, CZ_aud)
        partial[kk] = _j_lambda(Hr, Pr, lam=lam, eps=eps)
    partial["K0"] = J["H0"]
    partial_ret = {kk: 1.0 - partial[kk] / (partial["K0"] + eps)
                   for kk in ["Z1", "Z2", "Z3"]}
    print("[stats] partial J (delta proxy):",
          {kk: round(v, 5) for kk, v in partial.items()}, flush=True)
    print("[stats] strict retained fraction:",
          {kk: round(v, 5) for kk, v in partial_ret.items()}, flush=True)

    # ---- cluster bootstrap (by root tree = by row, since each row is one
    # root's fixed path) ----
    rng = np.random.RandomState(FIXED_SEED)
    n_aud = len(audit_rows)
    boot_ret = {k: [] for k in retention}
    boot_partial_ret = {k: [] for k in ["Z1", "Z2", "Z3"]}
    idx_all = np.arange(n_aud)
    for _ in range(args.n_bootstrap):
        idx = rng.choice(idx_all, size=n_aud, replace=True)
        Jb = {}
        for k in X_names:
            Wx = _ridge_fit(C_cal, X_cal[k])
            Xr2 = _cond_residual(X_aud[k][idx], C_aud[idx], Wx)
            Jb[k] = _j_lambda(Xr2, P_res_aud[idx], lam=lam, eps=eps)
        db = Jb["H0"] + eps
        for k in retention:
            boot_ret[k].append(Jb[k] / db)
        for kk in ["Z1", "Z2", "Z3"]:
            CZ_cal = np.concatenate([C_cal, X_cal[kk]], axis=1)
            CZ_aud = np.concatenate([C_aud[idx], X_aud[kk][idx]], axis=1)
            Hr = _proj_resid(X_aud["H0"][idx], CZ_cal, CZ_aud)
            Pr = _proj_resid(P_aud[idx], CZ_cal, CZ_aud)
            pk = _j_lambda(Hr, Pr, lam=lam, eps=eps)
            p0 = Jb["H0"]
            boot_partial_ret[kk].append(1.0 - pk / (p0 + eps))
    ci = {}
    for k in retention:
        arr = np.asarray(boot_ret[k])
        ci[k] = (float(np.percentile(arr, 2.5)),
                 float(np.percentile(arr, 97.5)))
    ci_partial = {}
    for kk in ["Z1", "Z2", "Z3"]:
        arr = np.asarray(boot_partial_ret[kk])
        ci_partial[kk] = (float(np.percentile(arr, 2.5)),
                          float(np.percentile(arr, 97.5)))

    # ---- matched-context shuffle null (H0 permuted within coarse time
    # buckets of C) ----
    h0 = X_aud["H0"].copy()
    strata = np.digitize(C_aud[:, 2], bins=np.linspace(0, 1.2, 8))
    null_J = []
    for _ in range(20):
        h0s = h0.copy()
        for s in np.unique(strata):
            m = strata == s
            perm = rng.permutation(m.sum())
            h0s[m] = h0[m][perm]
        Wx = _ridge_fit(C_cal, X_cal["H0"])
        Xr = _cond_residual(h0s, C_aud, Wx)
        null_J.append(_j_lambda(Xr, P_res_aud, lam=lam, eps=eps))
    null_lo = float(np.percentile(null_J, 5))
    print("[stats] H0 null (shuffled) J p5:", round(null_lo, 5), flush=True)

    report = {
        "J": J,
        "retention": retention,
        "lost": lost,
        "ci95": ci,
        "h0_null_p5": null_lo,
        "partial_J": partial,
        "strict_retained": partial_ret,
        "ci95_partial": ci_partial,
        "n_audit_rows": n_aud,
        "n_calib_rows": len(calib_rows),
        "lam": lam, "eps": eps,
        "fixed_seed": FIXED_SEED,
    }
    out = Path(args.ckpt).parent / "retention_audit.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2), flush=True)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
