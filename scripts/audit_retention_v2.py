#!/usr/bin/env python3
"""Introduction mechanism audit v2 — leaf-to-root path extraction.

Review correction (2026-09): the measured object is ONE fixed 3-hop remote
branch signal propagating along the computational dependency, NOT the same
root node at different layers.

Path:  a_3 (3-hop leaf) -> a_2 (2-hop) -> a_1 (1-hop) -> a_0 (root)

    U^(0) = h_{a_3}^{(0)}                      (leaf layer-0 state — the
                                                branch tensor actually fed
                                                into the first parent
                                                aggregation)
    Z^(1) = h_{a_2}^{(1)} = Gamma_1(U^(0), N1)
    Z^(2) = h_{a_1}^{(2)} = Gamma_2(Z^(1), N2)
    Z^(3) = h_{a_0}^{(3)} = Gamma_3(Z^(2), N3)

X-axis: # recursive aggregations (0 = 3-hop leaf, 0 compressions;
3 = root, 3 compressions).  Retention is NOT forced monotonic; C contains
summaries of the N_{1:3} added along the path.

Path selection: per root, a FIXED hash (root_id, layer) picks the child
slot at every step (never "first non-zero slot" — slot order correlates
with time and would bias the sample).  All k use the SAME full paths.
Root trees equally weighted (one path per root).

Model: frozen task-only TGN checkpoint (main-branch code); the hook is a
re-implementation of JodieTGNAdapter._compute with path sampling inserted
after each neighbor recursion — no weight updates, no backprop.
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
from rpbe.hosts.jodie_tgn import JodieTGNAdapter, TAU_TEMPLATE  # noqa: E402
from rpbe.config import RPBConfig  # noqa: E402
from rpbe.compressor import RecursiveCompressor  # noqa: E402
from rpbe.data.uci_link import UCILinkDataset  # noqa: E402

FIXED_SEED = 20260909
P_DIM = 128
# branch-wide context: path-OUTSIDE N_k projections (other-neighbor vector
# mean 16 + edge features 8 per step, times/structure) — self raw state is
# part of the branch and is NOT conditioned.  The SAME C conditions every
# layer of the path.
CTX_DIM = 80


def _fixed_proj(x, dim, seed):
    """Deterministic Gaussian projection [d] -> [dim] (fixed seed)."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    rng = np.random.RandomState(seed)
    W = rng.normal(0.0, 1.0 / np.sqrt(max(1, len(x))), size=(dim, len(x)))
    return W @ x


def _other_neighbor_vec(neighbor_lower, flat_row, slot, n_neighbors_):
    """Vector mean of the OTHER neighbors (excluding the path child slot)."""
    arr = neighbor_lower[flat_row].detach().cpu().numpy()  # [K, d]
    mask = np.ones(arr.shape[0], dtype=bool)
    mask[slot] = False
    if mask.sum() == 0:
        return np.zeros(arr.shape[1], dtype=np.float64)
    return arr[mask].mean(axis=0)


# ---------------------------------------------------------------- fixed maps
def _hash01(vals, seed):
    h = (np.asarray(vals, dtype=np.int64) * 2654435761 + seed) & 0xFFFFFFFF
    return h / float(2 ** 32)


def fixed_phi_S(edge_feat, delta_t, counterpart, src_node):
    rng = np.random.RandomState(FIXED_SEED)
    raw = np.stack([np.log1p(delta_t),
                    _hash01(counterpart, 101),
                    _hash01(src_node, 202)], axis=1)
    W = rng.normal(0.0, 0.2, size=(P_DIM // 4, 3))
    b = rng.uniform(0, 2 * np.pi, size=(P_DIM // 4,))
    rff = np.cos(raw @ W.T + b) * np.sqrt(2.0 / (P_DIM // 4))
    e = np.asarray(edge_feat, dtype=np.float64)
    e = e / (np.abs(e).max(axis=1, keepdims=True) + 1e-8)
    signs = rng.choice([-1.0, 1.0], size=(P_DIM // 2, e.shape[1]))
    sk = e @ signs.T / np.sqrt(e.shape[1])
    return np.concatenate([rff, sk, np.ones((len(delta_t), P_DIM // 4))],
                          axis=1)


def _ctx_vector(root_id, leaf_id, times, other_neighbor_vecs, edge_feats,
                self_raws=None):
    """Branch-wide context: fixed projections of the path-OUTSIDE N_k
    (vector mean of the OTHER neighbors, edge features, time) added at
    each step.  self raw state is part of the remote branch itself and
    MUST NOT be conditioned out — conditioning it removes the signal being
    measured (null control fails).  The SAME C conditions every layer."""
    del self_raws
    parts = []
    for i in range(3):
        parts.append(_fixed_proj(other_neighbor_vecs[i], 16, 600 + i))
        parts.append(_fixed_proj(edge_feats[i], 8, 700 + i))
    parts.append(np.asarray([
        _hash01(np.asarray([int(root_id)]), 401)[0],
        _hash01(np.asarray([int(leaf_id)]), 402)[0],
        np.log1p(times[0]) / 12.0,
        np.log1p(times[1]) / 12.0,
        np.log1p(times[2]) / 12.0,
    ], dtype=np.float64))
    out = np.concatenate([np.asarray(p, dtype=np.float64).reshape(-1)
                          for p in parts])
    if len(out) < CTX_DIM:
        out = np.concatenate(
            [out, np.zeros(CTX_DIM - len(out), dtype=np.float64)])
    return out[:CTX_DIM]


# -------------------------------------------------------------- future index
class FutureIndex:
    def __init__(self, ds):
        src = ds.full.sources
        dst = ds.full.destinations
        t = ds.full.timestamps
        order = np.argsort(t, kind="stable")
        self._rows = [None] * ds.n_nodes
        counts = np.zeros(ds.n_nodes, dtype=np.int64)
        for s in src:
            if 0 <= s < ds.n_nodes:
                counts[s] += 1
        for d in dst:
            if 0 <= d < ds.n_nodes:
                counts[d] += 1
        for i in range(ds.n_nodes):
            self._rows[i] = np.empty(counts[i], dtype=np.int64)
        fill = np.zeros(ds.n_nodes, dtype=np.int64)
        for j in order:
            s, d = int(src[j]), int(dst[j])
            if 0 <= s < ds.n_nodes:
                self._rows[s][fill[s]] = j
                fill[s] += 1
            if 0 <= d < ds.n_nodes:
                self._rows[d][fill[d]] = j
                fill[d] += 1
        self.src = src
        self.dst = dst
        self.t = t
        self.eidx = ds.full.edge_idxs
        self.n_nodes = ds.n_nodes

    def query(self, node, time):
        if not (0 <= node < self.n_nodes):
            return None
        rows = self._rows[node]
        if len(rows) == 0:
            return None
        pos = int(np.searchsorted(self.t[rows], time, side="right"))
        if pos >= len(rows):
            return None
        j = int(rows[pos])
        cp = int(self.dst[j]) if self.src[j] == node else int(self.src[j])
        return j, float(self.t[j] - time), cp


# -------------------------------------------------------------- ridge helpers
def _ridge_fit(X, Y, lam=1e-2):
    d = X.shape[1]
    return np.linalg.solve(X.T @ X + lam * np.eye(d), X.T @ Y)


def _cond_residual(X, C, Wx):
    return X - C @ Wx


def _j_lambda(X_res, P_res, lam=1e-2, eps=1e-6):
    n = X_res.shape[0]
    Cxx = (X_res.T @ X_res) / n
    Cpp = (P_res.T @ P_res) / n
    Cxp = (X_res.T @ P_res) / n
    A = Cxx + lam * np.eye(X_res.shape[1])
    B = Cpp + eps * np.eye(P_res.shape[1])
    try:
        M = np.linalg.solve(A, Cxp)
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
                        split_mode="count")
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

    # ---- leaf-to-root path-tracking compute (re-implemented hook) ----
    n_neighbors = 5
    n_layers = 3
    bs = args.bs
    # path sampling state: for the CURRENT batch
    path_state = {
        # layer -> {flat_row_in_parent_neighbor_recursion: (root_row, depth)}
        "by_layer": {1: {}, 2: {}},
        # root_row -> collected path pieces
        "recs": {},
        # per-step record stash keyed by (root_row, depth)
        "stash": {},
    }

    def _slot_hash(node_id, layer):
        return int((int(node_id) * 2654435761 + int(layer) * 104729)
                   % 1000003)

    orig_compute = adapter._compute

    cur_remove = {"d": None}

    def traced_compute(memory, source_nodes, timestamps, layer, n_neighbors_,
                       trace_paths, remove_depth=None):
        if remove_depth is None:
            remove_depth = cur_remove["d"]
        device_ = adapter.device
        source_nodes_t = torch.from_numpy(source_nodes).long().to(device_)
        timestamps_t = torch.from_numpy(timestamps).float().to(device_)
        timestamps_t = timestamps_t.unsqueeze(1)
        source_time = adapter.host.time_encoder(
            torch.zeros_like(timestamps_t))
        raw_source = adapter.host.node_features[source_nodes_t]
        if adapter.use_memory:
            raw_source = memory[source_nodes] + raw_source
        if layer == 0:
            # leaf states may be needed by the parent recursion's sampling
            return raw_source

        source_paths = {
            int(row): list(path) + [(0, 0.0)]
            for row, path in trace_paths.items()}
        source_lower = traced_compute(
            memory, source_nodes, timestamps, layer - 1, n_neighbors_,
            source_paths, remove_depth=remove_depth)

        neighbors, edge_idxs_np, edge_times = \
            adapter.neighbor_finder.get_temporal_neighbor(
                source_nodes, timestamps, n_neighbors=n_neighbors_)
        neighbors_t = torch.from_numpy(neighbors).long().to(device_)
        edge_idxs = torch.from_numpy(edge_idxs_np).long().to(device_)
        edge_deltas_np = timestamps[:, None] - edge_times
        edge_deltas = torch.from_numpy(edge_deltas_np).float().to(device_)
        flat_neighbors = neighbors.reshape(-1)
        repeated_times = np.repeat(timestamps, n_neighbors_)

        # ---------- path slot selection BEFORE the child recursion ----------
        is_top = (layer == n_layers and len(source_nodes) == 3 * bs)
        if is_top:
            # init per-batch state and pick each root's first child slot;
            # the child recursion (layer-1 of the neighbor chain) consumes
            # by_layer below.
            path_state["recs"] = {}
            path_state["stash"] = {}
            path_state["by_layer"] = {1: {}, 2: {}}
            for r in range(bs):
                s = _slot_hash(int(source_nodes[r]), 3) % n_neighbors_
                if int(neighbors[r, s]) != 0:
                    path_state["recs"][r] = {
                        "root": int(source_nodes[r]),
                        "t_root": float(timestamps[r]),
                        "slot3": s,
                    }
                    path_state["by_layer"][2][(r, s)] = r
        # non-top path rows: this call's source_nodes are the flat neighbors
        # of the parent call; a path row (parent_row, slot) lives at
        # flat_row = parent_row * n_neighbors + slot.  Pick the next child
        # slot BEFORE recursing into the children (so layer-1 sees it).
        self_path_choices = {}
        for (pr, s), root_r in list(path_state["by_layer"].get(
                layer, {}).items()):
            flat_row = pr * n_neighbors_ + s
            if flat_row >= len(source_nodes):
                continue
            node = int(source_nodes[flat_row])
            if node == 0:
                continue
            if layer >= 2:
                s2 = _slot_hash(node, layer - 1) % n_neighbors_
                child = int(neighbors[flat_row, s2])
                if child != 0:
                    self_path_choices[(root_r, layer)] = {
                        "node": node,
                        "t": float(timestamps[flat_row]),
                        "slot": s2,
                        "flat_row": flat_row,
                    }
                    path_state["by_layer"][layer - 1][
                        (flat_row, s2)] = root_r
            else:
                # layer 1 path rows: child slot is the parent-fixed s;
                # record the choice so the post-recursion block writes U0
                child = int(neighbors[flat_row, s])
                if child != 0:
                    self_path_choices[(root_r, layer)] = {
                        "node": node,
                        "t": float(timestamps[flat_row]),
                        "slot": s,
                        "flat_row": flat_row,
                    }

        neighbor_lower = traced_compute(
            memory, flat_neighbors, repeated_times, layer - 1, n_neighbors_,
            {}, remove_depth=remove_depth)
        neighbor_lower = neighbor_lower.view(
            len(source_nodes), n_neighbors_, -1)
        edge_time = adapter.host.time_encoder(edge_deltas)
        edge_features = adapter.host.edge_features[edge_idxs]
        mask = neighbors_t == 0
        # ---- paired removal: zero the selected child's message at the
        # aggregation layer that consumes it (remove_depth=3 -> layer 1,
        # 2 -> layer 2, 1 -> layer 3/top); everything else untouched ----
        if remove_depth is not None and layer == (4 - remove_depth):
            if remove_depth == 1 and is_top:
                for r, rec in path_state["recs"].items():
                    neighbor_lower = neighbor_lower.clone()
                    neighbor_lower[r, rec["slot3"]] = 0.0
                    mask = mask.clone()
                    mask[r, rec["slot3"]] = True
            else:
                for (pr, s), root_r in list(path_state["by_layer"].get(
                        layer, {}).items()):
                    flat_row = pr * n_neighbors_ + s
                    if flat_row >= len(source_nodes):
                        continue
                    neighbor_lower = neighbor_lower.clone()
                    neighbor_lower[flat_row, s] = 0.0
                    mask = mask.clone()
                    mask[flat_row, s] = True
        vanilla = adapter.host.aggregate(
            layer, source_lower, source_time, neighbor_lower, edge_time,
            edge_features, mask)
        if adapter.compressor is not None and 0 < layer < adapter.n_layers:
            tau = TAU_TEMPLATE.format(layer)
            z = adapter.compressor.compress(
                tau=tau, own_input=raw_source, aggregate_output=vanilla)
        else:
            z = vanilla

        # ---------- path state recording AFTER z is available ----------
        if is_top:
            for r, rec in path_state["recs"].items():
                rec["z3"] = z[r].detach().cpu().numpy()
                rec["other_neighbors"] = _other_neighbor_vec(
                    neighbor_lower, r, rec["slot3"], n_neighbors_)
        for (root_r, lv), ch in self_path_choices.items():
            flat_row = ch["flat_row"]
            if lv >= 2:
                path_state["stash"][(root_r, lv)] = {
                    "z": z[flat_row].detach().cpu().numpy(),
                    "node": ch["node"],
                    "t": ch["t"],
                    "edge_feat": adapter.host.edge_features[
                        int(edge_idxs_np[flat_row, ch["slot"]])]
                        .detach().cpu().numpy(),
                    # VECTOR mean of the OTHER neighbors (path-outside)
                    "other_neighbors": _other_neighbor_vec(
                        neighbor_lower, flat_row, ch["slot"],
                        n_neighbors_),
                }
            else:
                # layer 1: child slot -> LEAF layer-0 state = U0
                child = int(neighbors[flat_row, ch["slot"]])
                path_state["stash"][(root_r, 1)] = {
                    "z": z[flat_row].detach().cpu().numpy(),
                    "node": ch["node"],
                    "t": ch["t"],
                    "u0": neighbor_lower[flat_row, ch["slot"]]
                          .detach().cpu().numpy(),
                    "leaf": child,
                    "edge_feat": adapter.host.edge_features[
                        int(edge_idxs_np[flat_row, ch["slot"]])]
                        .detach().cpu().numpy(),
                    "other_neighbors": _other_neighbor_vec(
                        neighbor_lower, flat_row, ch["slot"],
                        n_neighbors_),
                }
        return z

    adapter._compute = traced_compute

    def extract(stream, n_batches, offset_batches=0):
        if tgn.use_memory:
            tgn.memory.__init_memory__()
        rows = []
        n_avail = len(stream.sources) // bs
        n_b = min(n_batches, max(0, n_avail - offset_batches))
        for b in range(n_b):
            bb = offset_batches + b
            s0 = bb * bs
            s1 = s0 + bs
            src = stream.sources[s0:s1].astype(np.int64)
            dst = stream.destinations[s0:s1].astype(np.int64)
            t = stream.timestamps[s0:s1]
            e = stream.edge_idxs[s0:s1]
            adapter.set_trace_source_rows([])  # no SELF-spine trace needed
            # paired variants: 0=original, 3/2/1 = remove that source
            var_states = {}
            for rm in (None, 3, 2, 1):
                cur_remove["d"] = rm
                path_state["recs"] = {}
                path_state["stash"] = {}
                path_state["by_layer"] = {1: {}, 2: {}}
                mem_bak = (tgn.memory.backup_memory()
                           if tgn.use_memory else None)
                with torch.no_grad():
                    tgn.compute_edge_probabilities(src, dst, dst, t, e,
                                                   n_neighbors)
                if mem_bak is not None:
                    tgn.memory.restore_memory(mem_bak)
                adapter.clear_trace()
                var_states[rm] = {
                    "recs": dict(path_state["recs"]),
                    "stash": {k: dict(v) for k, v
                              in path_state["stash"].items()},
                }
            cur_remove["d"] = None
            recs = var_states[None]["recs"]
            stash = var_states[None]["stash"]
            rm3 = var_states[3]
            rm2 = var_states[2]
            rm1 = var_states[1]
            for r, rec in recs.items():
                # need all three steps: stash keys (r,1),(r,2)
                if (r, 1) not in stash or (r, 2) not in stash:
                    continue
                st1 = stash[(r, 1)]
                st2 = stash[(r, 2)]
                root = rec["root"]
                futr = fut.query(root, rec["t_root"])
                if futr is None:
                    continue
                j, dt, cp = futr
                times = [st1["t"], st2["t"], rec["t_root"]]
                om = [st1["other_neighbors"],
                      st2["other_neighbors"],
                      rec["other_neighbors"]]
                ef = [st1["edge_feat"], st2["edge_feat"],
                      ds.edge_features[int(fut.eidx[j])]]
                ctx = _ctx_vector(root, st1["leaf"], times, om, ef)
                # paired-removal deltas: source-component at each position
                def _dz(var, key, orig_z):
                    if key == "z3":
                        rr = var["recs"].get(r)
                        return (orig_z - rr[key]) if rr is not None \
                            and key in rr else None
                    ss = var["stash"].get((r, {"z1": 1, "z2": 2}[key]))
                    return (orig_z - ss["z"]) if ss is not None else None
                z1 = st1["z"]; z2 = st2["z"]; z3 = rec["z3"]
                dz3_2 = _dz(rm3, "z1", z1)  # 3-hop source at the 2-hop node
                dz3_1 = _dz(rm3, "z2", z2)
                dz3_r = _dz(rm3, "z3", z3)
                dz2_1 = _dz(rm2, "z2", z2)
                dz2_r = _dz(rm2, "z3", z3)
                dz1_r = _dz(rm1, "z3", z3)
                if any(v is None for v in
                       (dz3_2, dz3_1, dz3_r, dz2_1, dz2_r, dz1_r)):
                    continue
                rows.append({
                    "u0": st1["u0"],           # Delta_{3->3} = U3 (leaf)
                    "z1": st1["z"],            # Delta_{2->2} = U2
                    "z2": st2["z"],            # Delta_{1->1} = U1
                    "z3": z3,                  # original root state
                    "d32": dz3_2, "d31": dz3_1, "d3r": dz3_r,
                    "d21": dz2_1, "d2r": dz2_r,
                    "d1r": dz1_r,
                    "ctx": ctx,
                    "S": (ds.edge_features[int(fut.eidx[j])],
                          dt, cp, root),
                })
            if (b + 1) % 20 == 0:
                print("[extract] batch {}/{} rows={}".format(
                    b + 1, n_b, len(rows)), flush=True)
        return rows

    n_train_batches = len(train.sources) // bs
    audit_offset = max(0, n_train_batches - args.audit_batches)
    print("[audit] extracting audit set (train tail) ...", flush=True)
    audit_rows = extract(train, args.audit_batches,
                         offset_batches=audit_offset)
    print("[audit] audit rows:", len(audit_rows), flush=True)
    print("[calib] extracting calibration set (train head) ...", flush=True)
    calib_rows = extract(train, args.calib_batches, offset_batches=0)
    print("[calib] calib rows:", len(calib_rows), flush=True)
    if not audit_rows or not calib_rows:
        print("FATAL: no leaf-to-root paths extracted", flush=True)
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
    X_names = ["U0", "Z1", "Z2", "Z3"]
    X_cal = {k: np.stack([r[k.lower()] for r in calib_rows])
             for k in X_names}
    X_aud = {k: np.stack([r[k.lower()] for r in audit_rows])
             for k in X_names}
    J = {}
    for k in X_names:
        Wx = _ridge_fit(C_cal, X_cal[k])
        Xr_aud = _cond_residual(X_aud[k], C_aud, Wx)
        J[k] = _j_lambda(Xr_aud, P_res_aud, lam=lam, eps=eps)
    print("[stats] J values:", {k: round(v, 5) for k, v in J.items()},
          flush=True)

    # ---- per-source depth retention (review: three staggered lines) ----
    # Same path data, three source depths:
    #   3-hop line: U3=u0 -> T32=z1 -> T31=z2 -> T3r=z3
    #   2-hop line: U2=z1        -> T21=z2 -> T2r=z3
    #   1-hop line: U1=z2                   -> T1r=z3
    # Each line normalized by its own source J(U_d | C); same C and same
    # future target S_rho for all positions of a line.
    # ---- paired-removal retention: source COMPONENT at each position ----
    # Delta_{s->k} = h_k(keep s) - h_k(remove s); parent self and siblings
    # cancel in the subtraction.  Each line normalized by its own source
    # component Delta_{s->s}.
    line_keys = {
        "D3s": "U0", "D32": "D32", "D31": "D31", "D3r": "D3R",
        "D2s": "Z1", "D21": "D21", "D2r": "D2R",
        "D1s": "Z2", "D1r": "D1R",
    }
    # build delta tensors (calib/audit)
    X_cal_d = dict(X_cal)
    X_aud_d = dict(X_aud)
    for kk, fld in [("D32", "d32"), ("D31", "d31"), ("D3R", "d3r"),
                    ("D21", "d21"), ("D2R", "d2r"), ("D1R", "d1r")]:
        X_cal_d[kk] = np.stack([r[fld] for r in calib_rows])
        X_aud_d[kk] = np.stack([r[fld] for r in audit_rows])
    J_line = {}
    for name, key in line_keys.items():
        Wx = _ridge_fit(C_cal, X_cal_d[key])
        Xr = _cond_residual(X_aud_d[key], C_aud, Wx)
        J_line[name] = _j_lambda(Xr, P_res_aud, lam=lam, eps=eps)
    print("[stats] paired-removal J:", {k: round(v, 4)
                                        for k, v in J_line.items()},
          flush=True)
    # RAW variant: same paired-removal deltas, no context residualization
    # on X or P (review: raw/keep trace is the defensible primary line).
    J_raw = {}
    for name, key in line_keys.items():
        J_raw[name] = _j_lambda(X_aud_d[key], P_aud, lam=lam, eps=eps)
    print("[stats] paired-removal raw J:",
          {k: round(v, 4) for k, v in J_raw.items()}, flush=True)

    def _line_ret(us, ts):
        d = J_line[us] + eps
        return {t: J_line[t] / d for t in ts}

    line3 = _line_ret("D3s", ["D32", "D31", "D3r"])
    line2 = _line_ret("D2s", ["D21", "D2r"])
    line1 = _line_ret("D1s", ["D1r"])
    print("[stats] line3 (3-hop source, paired removal):",
          {k: round(v, 4) for k, v in line3.items()}, flush=True)
    print("[stats] line2 (2-hop source, paired removal):",
          {k: round(v, 4) for k, v in line2.items()}, flush=True)
    print("[stats] line1 (1-hop source, paired removal):",
          {k: round(v, 4) for k, v in line1.items()}, flush=True)

    # ---- cluster bootstrap (root = cluster) for the delta energies ----
    rng = np.random.RandomState(FIXED_SEED)
    n_aud = len(audit_rows)
    boot_J = {name: [] for name in line_keys}
    idx_all = np.arange(n_aud)
    for _ in range(args.n_bootstrap):
        idx = rng.choice(idx_all, size=n_aud, replace=True)
        for name, key in line_keys.items():
            Wx = _ridge_fit(C_cal, X_cal_d[key])
            Xr2 = _cond_residual(X_aud_d[key][idx], C_aud[idx], Wx)
            boot_J[name].append(_j_lambda(Xr2, P_res_aud[idx],
                                          lam=lam, eps=eps))
    ci_J = {}
    for name, arr in boot_J.items():
        a = np.asarray(arr)
        ci_J[name] = (float(np.percentile(a, 2.5)),
                      float(np.percentile(a, 97.5)))
    print("[stats] CI on delta J:",
          {k: [round(a, 4), round(b, 4)] for k, (a, b) in ci_J.items()},
          flush=True)

    # ---- matched-context shuffle nulls on the three sources ----
    strata = np.digitize(C_aud[:, 2], bins=np.linspace(0, 1.2, 8))
    null_lo = {}
    for src in ("U0", "Z1", "Z2"):
        src_x = X_aud[src].copy()
        null_J = []
        for _ in range(20):
            src_s = src_x.copy()
            for s in np.unique(strata):
                m = strata == s
                perm = rng.permutation(m.sum())
                src_s[m] = src_x[m][perm]
            Wx = _ridge_fit(C_cal, X_cal[src])
            Xr = _cond_residual(src_s, C_aud, Wx)
            null_J.append(_j_lambda(Xr, P_res_aud, lam=lam, eps=eps))
        null_lo[src] = float(np.percentile(null_J, 5))
    print("[stats] source null (shuffled) J p5:",
          {k: round(v, 5) for k, v in null_lo.items()}, flush=True)

    report = {
        "J": J,
        "paired_removal_J": J_line,
        "paired_removal_raw_J": J_raw,
        "retention_lines": {"line3": line3, "line2": line2, "line1": line1},
        "ci_delta_J": ci_J,
        "source_null_p5": null_lo,
        "n_audit_rows": n_aud,
        "n_calib_rows": len(calib_rows),
        "lam": lam, "eps": eps,
        "fixed_seed": FIXED_SEED,
        "extraction": "leaf-to-root paired-removal path",
    }
    out = Path(args.ckpt).parent / "retention_audit_v2.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2), flush=True)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
