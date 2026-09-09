#!/usr/bin/env python3
"""Introduction-mechanism retention audit v2 — corrected protocol.

This version supersedes the audit committed at 6bcaa81 whose JSON/figure were
rejected by review: values were not retention fractions (ratios of unlike
quantities, a zero denominator padded with eps -> 129k, chained products such
as 0.503*1.035*1.691 = 0.88, CIs excluding their own point estimate), the CC
context leaked the *future* event's edge feature (the same edge that builds
the prediction target), and TGN memory never advanced (each batch restored a
backup, and the "train tail" started from a zeroed memory without replaying
the prefix).

Corrected protocol
------------------
Path (one fixed leaf-to-root branch per root, slot chosen by a fixed hash):
a_3 (leaf) -> a_2 -> a_1 -> a_0 (root); states
    U0 = leaf layer-0 state
    Z1 = h_{a_2} = Gamma_1(U0, N1)
    Z2 = h_{a_1} = Gamma_2(Z1, N2)
    Z3 = h_{a_0} = Gamma_3(Z2, N3)

History flow: memory is advanced exactly like a real single-pass stream.  A
pass first replays the prefix with normal keep-only forwards, then processes
its window batch by batch: snapshot pre-memory -> run the KEEP forward (this
IS the advance) -> snapshot post-memory -> run each remove variant from the
same pre-memory and restore after each -> restore post-memory.  A
--memory-parity-batches pre-check asserts the window scheme leaves memory
bit-identical to a pristine keep-only single pass.

Context C is leak-free: it contains ONLY historical path structure (the three
historical edges' edge_feat/edge_time and node times, path-outside
other-neighbor means, root/leaf ids).  The future event appears ONLY in the
prediction target P (fixed witness phi_S).

Retention (fixed same-source signal, bounded <= 1, no chaining).  The figure's
question is how much of a source node's useful signal survives one/two/three
recursive aggregations toward the root.  For source depth s let X_s be its
representation (3: U0, 2: Z1, 1: Z2) and fix the source signal ONCE as

    Q_s = ridge(X_s -> P)          (fit on calib; P = root future witness)

Q_s is never re-estimated per layer and never regressed against the context C.
C is NOT subtracted from the source signal: the paired keep/remove happens on
the SAME tree, so the environment (context, siblings, edges, times) is fixed by
construction.  At each downstream position k the paired-removal delta
Delta_{s->k} (ancestor state keep minus remove of this source) is measured, and
retention is how much of the SAME Q_s the audit-set Delta recovers:

    R_{s->k} = 1 - ||Q_s - Qhat_s||_F^2 / (||Q_s - mean Q_s||_F^2 + eps)

with Qhat_s a ridge prediction of Q_s from Delta_{s->k} alone.  Every point is
an independent direct regression against the same source component -- never a
chain of local factors, never a ratio of per-layer J, never a per-layer
re-prediction of the future.  SSE >= 0 makes R <= 1 by construction; negative
values are reported honestly.

Gates run before plotting: source signal (explained variance of audit P by
Q_s) must exceed a within-strata shuffle null (95th pct) or the source is
marked NOT IDENTIFIABLE and its line is not drawn; identity at the source ~1;
remove-source (Delta = 0) ~0; a mismatched (permuted) delta must not recover
Q_s; memory parity is stored.  The figure only draws lines whose gates
all pass.  Calibration for the MAIN result is the contiguous same-tail block
(just before the audit block, silent gap between); a head-calibration ->
tail-audit fit is kept only as a cross-temporal TRANSFER stress test.

Gates run before plotting: source predictive signal above a matched-context
shuffle null (95th pct), identity at the source ~1, delete-source (Delta = 0,
C-only) floor ~0, and a mismatched-delta control (Delta permuted within
strata) must not recover Q.  A memory-parity flag is stored.  The figure only
draws lines whose gates all pass.

Usage:
    python scripts/audit_retention_v2.py \
        --ckpt <best.pt> --data-dir <processed_tgn_data> --data-name wikipedia \
        --gpu 0 --bs 64 --audit-batches 200 --calib-batches 60 \
        --n-bootstrap 200 --memory-parity-batches 5
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
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from rpbe.hosts.official_tgn import TGN, get_neighbor_finder  # noqa: E402
from rpbe.hosts.jodie_tgn import JodieTGNAdapter, TAU_TEMPLATE  # noqa: E402
from rpbe.config import RPBConfig  # noqa: E402
from rpbe.compressor import RecursiveCompressor  # noqa: E402
from rpbe.data.uci_link import UCILinkDataset  # noqa: E402

import retention_stats as rs  # noqa: E402

FIXED_SEED = 20260909
P_DIM = 128
# branch-wide leak-free context: per step the path-OUTSIDE other-neighbor vector
# mean (16) + the *historical* path edge feature (8), plus 2 id hashes, 3 node
# times and 3 historical edge times.  No future index may enter C.
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


def _ctx_vector(root_id, leaf_id, node_times, edge_feats, edge_times,
                other_neighbor_vecs):
    """Leak-free path context (dim CTX_DIM).  Only historical structure:
    the three path edges (edge_feat + edge_time), the node times, path-outside
    other-neighbor means and root/leaf ids.  Deliberately NO future index."""
    parts = []
    for i in range(3):
        parts.append(_fixed_proj(other_neighbor_vecs[i], 16, 600 + i))
        parts.append(_fixed_proj(edge_feats[i], 8, 700 + i))
    parts.append(np.asarray([
        _hash01(np.asarray([int(root_id)]), 401)[0],
        _hash01(np.asarray([int(leaf_id)]), 402)[0],
        np.log1p(node_times[0]) / 12.0,
        np.log1p(node_times[1]) / 12.0,
        np.log1p(node_times[2]) / 12.0,
        np.log1p(edge_times[0]) / 12.0,
        np.log1p(edge_times[1]) / 12.0,
        np.log1p(edge_times[2]) / 12.0,
    ], dtype=np.float64))
    out = np.concatenate([np.asarray(p, dtype=np.float64).reshape(-1)
                          for p in parts])
    assert len(out) == CTX_DIM, (len(out), CTX_DIM)
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


# ------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--data-name", default="wikipedia")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--audit-batches", type=int, default=200)
    ap.add_argument("--calib-batches", type=int, default=60,
                    help="same-tail calibration block length (contiguous, "
                         "immediately before the audit block, separated by "
                         "--gap-batches)")
    ap.add_argument("--gap-batches", type=int, default=8,
                    help="silent batches between the same-tail calib block "
                         "and the audit block")
    ap.add_argument("--transfer-batches", type=int, default=None,
                    help="head calibration block length for the head->tail "
                         "transfer stress test (default = --calib-batches)")
    ap.add_argument("--audit-start-batches", type=int, default=None,
                    help="override: start the audit window here (default: "
                         "train tail = --audit-batches worth of batches)")
    ap.add_argument("--n-bootstrap", type=int, default=200)
    ap.add_argument("--n-null", type=int, default=200)
    ap.add_argument("--memory-parity-batches", type=int, default=5)
    ap.add_argument("--lam", type=float, default=1e-2)
    ap.add_argument("--lam-ret", type=float, default=1e-2)
    ap.add_argument("--identity-min", type=float, default=0.90)
    ap.add_argument("--floor-max", type=float, default=0.05)
    args = ap.parse_args()

    eps = 1e-6
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
    path_state = {
        "by_layer": {1: {}, 2: {}},
        "recs": {},
        "stash": {},
    }
    cur = {"d": None, "record": False}

    def _slot_hash(node_id, layer):
        return int((int(node_id) * 2654435761 + int(layer) * 104729)
                   % 1000003)

    def traced_compute(memory, source_nodes, timestamps, layer, n_neighbors_,
                       trace_paths, remove_depth=None, record=None):
        if remove_depth is None:
            remove_depth = cur["d"]
        if record is None:
            record = cur["record"]
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
            return raw_source

        # ---- bookkeeping only when record=True; the tensor math below is
        # bit-identical to the plain host recursion either way, so silent
        # prefix replay advances memory exactly like a single real pass ----
        source_paths = {}
        if record:
            source_paths = {
                int(row): list(path) + [(0, 0.0)]
                for row, path in trace_paths.items()}

        source_lower = traced_compute(
            memory, source_nodes, timestamps, layer - 1, n_neighbors_,
            source_paths, remove_depth=remove_depth, record=record)

        neighbors, edge_idxs_np, edge_times_np = \
            adapter.neighbor_finder.get_temporal_neighbor(
                source_nodes, timestamps, n_neighbors=n_neighbors_)
        neighbors_t = torch.from_numpy(neighbors).long().to(device_)
        edge_idxs = torch.from_numpy(edge_idxs_np).long().to(device_)
        edge_deltas_np = timestamps[:, None] - edge_times_np
        edge_deltas = torch.from_numpy(edge_deltas_np).float().to(device_)
        flat_neighbors = neighbors.reshape(-1)
        repeated_times = np.repeat(timestamps, n_neighbors_)

        # ---------- path slot selection (record only; never used by the
        # tensor math of the plain pass) ----------
        is_top = (layer == n_layers and len(source_nodes) == 3 * bs)
        self_path_choices = {}
        if record:
            if is_top:
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
                            "edge_feat_top": adapter.host.edge_features[
                                int(edge_idxs_np[r, s])].detach().cpu().numpy(),
                            "edge_time_top": float(edge_times_np[r, s]),
                        }
                        path_state["by_layer"][2][(r, s)] = r
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
            {}, remove_depth=remove_depth, record=record)
        neighbor_lower = neighbor_lower.view(
            len(source_nodes), n_neighbors_, -1)
        edge_time = adapter.host.time_encoder(edge_deltas)
        edge_features = adapter.host.edge_features[edge_idxs]
        mask = neighbors_t == 0
        # ---- paired removal: replace ONLY the tracked source state (zero the
        # selected child's hidden contribution).  The neighbor slot, its edge
        # feature, time diff and mask are left untouched so the removal does
        # not change the surrounding environment -- C is fixed by the paired
        # keep/remove on the same tree.
        if record and remove_depth is not None and layer == (4 - remove_depth):
            if remove_depth == 1 and is_top:
                for r, rec in path_state["recs"].items():
                    neighbor_lower = neighbor_lower.clone()
                    neighbor_lower[r, rec["slot3"]] = 0.0
            else:
                for (pr, s), root_r in list(path_state["by_layer"].get(
                        layer, {}).items()):
                    flat_row = pr * n_neighbors_ + s
                    if flat_row >= len(source_nodes):
                        continue
                    neighbor_lower = neighbor_lower.clone()
                    neighbor_lower[flat_row, s] = 0.0
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
        if record:
            if is_top:
                for r, rec in path_state["recs"].items():
                    rec["z3"] = z[r].detach().cpu().numpy()
                    rec["other_neighbors"] = _other_neighbor_vec(
                        neighbor_lower, r, rec["slot3"], n_neighbors_)
            for (root_r, lv), ch in self_path_choices.items():
                flat_row = ch["flat_row"]
                base = {
                    "z": z[flat_row].detach().cpu().numpy(),
                    "node": ch["node"],
                    "t": ch["t"],
                    "edge_feat": adapter.host.edge_features[
                        int(edge_idxs_np[flat_row, ch["slot"]])]
                        .detach().cpu().numpy(),
                    "edge_time": float(edge_times_np[flat_row, ch["slot"]]),
                    "other_neighbors": _other_neighbor_vec(
                        neighbor_lower, flat_row, ch["slot"], n_neighbors_),
                }
                if lv >= 2:
                    path_state["stash"][(root_r, lv)] = base
                else:
                    child = int(neighbors[flat_row, ch["slot"]])
                    base["u0"] = neighbor_lower[flat_row, ch["slot"]] \
                        .detach().cpu().numpy()
                    base["leaf"] = child
                    path_state["stash"][(root_r, 1)] = base
        return z

    adapter._compute = traced_compute

    def _reset_path_state():
        path_state["recs"] = {}
        path_state["stash"] = {}
        path_state["by_layer"] = {1: {}, 2: {}}

    def _forward_batch(bb, rm, record=True):
        cur["d"] = rm
        cur["record"] = record
        _reset_path_state()
        s0 = bb * bs
        s1 = s0 + bs
        src = train.sources[s0:s1].astype(np.int64)
        dst = train.destinations[s0:s1].astype(np.int64)
        t = train.timestamps[s0:s1]
        e = train.edge_idxs[s0:s1]
        adapter.set_trace_source_rows([])
        with torch.no_grad():
            tgn.compute_edge_probabilities(src, dst, dst, t, e, n_neighbors)
        adapter.clear_trace()
        if not record:
            return None
        return {
            "recs": dict(path_state["recs"]),
            "stash": {k: dict(v) for k, v in path_state["stash"].items()},
        }

    def _backup_mem():
        return tgn.memory.backup_memory() if tgn.use_memory else None

    def _restore_mem(bak):
        if tgn.use_memory and bak is not None:
            tgn.memory.restore_memory(bak)

    def window_batch(bb):
        """Keep forward advances memory exactly once; removes run from the
        same pre-memory and are rolled back; post-keep state is restored so
        the stream continues like a pure keep-only single pass."""
        mem_pre = _backup_mem()
        keep = _forward_batch(bb, None, record=True)
        mem_post = _backup_mem()
        rm_snaps = {}
        for rm in (3, 2, 1):
            _restore_mem(mem_pre)
            rm_snaps[rm] = _forward_batch(bb, rm, record=True)
            _restore_mem(mem_pre)
        _restore_mem(mem_post)
        return keep, rm_snaps

    def silent_batch(bb):
        _forward_batch(bb, None, record=False)

    def _rows_from(keep, rm_snaps):
        recs = keep["recs"]
        stash = keep["stash"]
        rm3, rm2, rm1 = rm_snaps[3], rm_snaps[2], rm_snaps[1]
        rows = []
        for r, rec in recs.items():
            if (r, 1) not in stash or (r, 2) not in stash:
                continue
            st1 = stash[(r, 1)]
            st2 = stash[(r, 2)]
            root = rec["root"]
            futr = fut.query(root, rec["t_root"])
            if futr is None:
                continue
            j, dt, cp = futr
            node_times = [st1["t"], st2["t"], rec["t_root"]]
            edge_feats = [st1["edge_feat"], st2["edge_feat"],
                          rec["edge_feat_top"]]
            edge_times = [st1["edge_time"], st2["edge_time"],
                          rec["edge_time_top"]]
            om = [st1["other_neighbors"],
                  st2["other_neighbors"],
                  rec["other_neighbors"]]
            ctx = _ctx_vector(root, st1["leaf"], node_times, edge_feats,
                              edge_times, om)
            # paired-removal deltas: source component at each position
            def _dz(var, key, orig_z):
                if key == "z3":
                    rr = var["recs"].get(r)
                    return (orig_z - rr[key]) if rr is not None \
                        and key in rr else None
                ss = var["stash"].get((r, {"z1": 1, "z2": 2}[key]))
                return (orig_z - ss["z"]) if ss is not None else None
            z1 = st1["z"]; z2 = st2["z"]; z3 = rec["z3"]
            dz3_2 = _dz(rm3, "z1", z1)
            dz3_1 = _dz(rm3, "z2", z2)
            dz3_r = _dz(rm3, "z3", z3)
            dz2_1 = _dz(rm2, "z2", z2)
            dz2_r = _dz(rm2, "z3", z3)
            dz1_r = _dz(rm1, "z3", z3)
            if any(v is None for v in
                   (dz3_2, dz3_1, dz3_r, dz2_1, dz2_r, dz1_r)):
                continue
            rows.append({
                "u0": st1["u0"],
                "z1": z1,
                "z2": z2,
                "z3": z3,
                "d32": dz3_2, "d31": dz3_1, "d3r": dz3_r,
                "d21": dz2_1, "d2r": dz2_r,
                "d1r": dz1_r,
                "ctx": ctx,
                "S": (ds.edge_features[int(fut.eidx[j])],
                      dt, cp, root),
            })
        return rows

    def _silent_batches(a, b, label):
        for i, bb in enumerate(range(a, b)):
            silent_batch(bb)
            if (i + 1) % 200 == 0:
                print("[{}] prefix replay {}/{}".format(
                    label, i + 1, b - a), flush=True)

    def _collect_rows(a, b, label):
        rows = []
        for i, bb in enumerate(range(a, b)):
            keep, rm_snaps = window_batch(bb)
            rows += _rows_from(keep, rm_snaps)
            if (i + 1) % 20 == 0:
                print("[{}] batch {}/{} rows={}".format(
                    label, i + 1, b - a, len(rows)), flush=True)
        return rows

    def mem_identical(bak_a, bak_b):
        if not tgn.use_memory:
            return True
        if not (torch.equal(bak_a[0], bak_b[0])
                and torch.equal(bak_a[1], bak_b[1])):
            return False
        if set(bak_a[2]) != set(bak_b[2]):
            return False
        for k in bak_a[2]:
            if len(bak_a[2][k]) != len(bak_b[2][k]):
                return False
            for (ta, ea), (tb, eb) in zip(bak_a[2][k], bak_b[2][k]):
                if not (torch.equal(ta, tb) and torch.equal(ea, eb)):
                    return False
        return True

    n_train_batches = len(train.sources) // bs
    if args.audit_start_batches is None:
        audit_offset = max(0, n_train_batches - args.audit_batches)
    else:
        audit_offset = int(args.audit_start_batches)
    # same-tail causal layout: audit block [audit_lo, audit_hi) is the held-out
    # target; calibration is the CONTIGUOUS block just before it (no row-level
    # interleaving), separated by a silent gap.  Memory is advanced by the real
    # single-pass flow only.
    audit_lo = audit_offset
    audit_hi = min(n_train_batches, audit_lo + args.audit_batches)
    gap = max(0, int(args.gap_batches))
    calib_hi = max(0, audit_lo - gap)
    calib_lo = max(0, calib_hi - args.calib_batches)
    transfer_hi = min(n_train_batches, int(
        args.transfer_batches if args.transfer_batches is not None
        else args.calib_batches))

    # ---- memory-parity pre-check: the keep/remove window scheme must leave
    # memory bit-identical to a pristine keep-only single pass.  Run over the
    # first few train batches (fresh memory == the real stream start) so the
    # mechanics are validated without an extra full prefix replay.
    mem_parity = {"ok": False, "n_batches": args.memory_parity_batches,
                  "detail": "not run (use_memory={})".format(tgn.use_memory)}
    if tgn.use_memory and args.memory_parity_batches > 0:
        if tgn.use_memory:
            tgn.memory.__init_memory__()
        base = _backup_mem()
        for w in range(args.memory_parity_batches):
            window_batch(w)                          # keep + rolled-back removes
        state_window = _backup_mem()
        _restore_mem(base)
        for w in range(args.memory_parity_batches):
            silent_batch(w)                          # pristine keep-only pass
        state_single = _backup_mem()
        _restore_mem(base)
        ok = mem_identical(state_window, state_single)
        mem_parity = {"ok": ok, "n_batches": args.memory_parity_batches,
                      "detail": "bit-identical vs keep-only single pass"
                      if ok else "MISMATCH vs keep-only single pass"}
        print("[parity] memory {} vs keep-only single pass".format(
            "bit-identical" if ok else "MISMATCH"), flush=True)
        if not ok:
            print("FATAL: memory parity check failed; aborting", flush=True)
            return

    print("[tail] one full prefix pass: silent replay 0..{}, then calib "
          "[{},{}), silent gap, audit [{},{})".format(
              calib_lo, calib_lo, calib_hi, audit_lo, audit_hi), flush=True)
    if tgn.use_memory:
        tgn.memory.__init_memory__()
    _silent_batches(0, calib_lo, "tail")
    calib_rows = _collect_rows(calib_lo, calib_hi, "calib")
    _silent_batches(calib_hi, audit_lo, "gap")
    audit_rows = _collect_rows(audit_lo, audit_hi, "audit")
    print("[tail] same-tail calib rows:", len(calib_rows),
          "audit rows:", len(audit_rows), flush=True)
    # ---- head calibration block for the head->tail TRANSFER stress test
    # (a second, independent pass from fresh memory over the stream head)
    if tgn.use_memory:
        tgn.memory.__init_memory__()
    calib_h_rows = _collect_rows(0, transfer_hi, "head-calib")
    print("[head] head calib rows:", len(calib_h_rows), flush=True)
    if not audit_rows or not calib_rows or not calib_h_rows:
        print("FATAL: no leaf-to-root paths extracted", flush=True)
        return

    # ------------------------------------------------------------ statistics
    def make_P(rows_):
        ef = np.stack([r["S"][0] for r in rows_])
        dt = np.asarray([r["S"][1] for r in rows_], dtype=np.float64)
        cp = np.asarray([r["S"][2] for r in rows_], dtype=np.int64)
        sn = np.asarray([r["S"][3] for r in rows_], dtype=np.int64)
        return fixed_phi_S(ef, dt, cp, sn)

    def col(rows_, key):
        return np.stack([r[key] for r in rows_])

    def catF(D, C):
        return np.concatenate([D, C], axis=1)

    lines = {
        3: {"name": "U0", "source_key": "u0", "origin_phys": 0,
            "points": [(1, "d32"), (2, "d31"), (3, "d3r")]},
        2: {"name": "Z1", "source_key": "z1", "origin_phys": 1,
            "points": [(2, "d21"), (3, "d2r")]},
        1: {"name": "Z2", "source_key": "z2", "origin_phys": 2,
            "points": [(3, "d1r")]},
    }

    P_aud = make_P(audit_rows)
    C_aud = col(audit_rows, "ctx")
    # matched-context key = a historical node-time scalar in C (index 74:
    # root/leaf hashes and one node time precede the three edge times).
    strata = rs.strata_ids(C_aud[:, 74])

    def compute_line(s, spec, calib_rows_, audit_rows_, do_ci):
        """Retention of ONE fixed source signal Q_s along the path.

        The figure's question is: how much of a source node's useful signal
        survives one/two/three recursive aggregations on the way to the root?
        Q_s is fixed once at the source -- Q_s = ridge(X_s -> P) fit on calib,
        where P is the root's future witness -- and is never re-estimated per
        layer and never regressed against C.  The context C is NOT subtracted:
        the paired keep/remove intervention happens on the SAME tree, so C is
        held fixed by construction.  At each ancestor position the paired-
        removal delta Delta_{s->k} (keep minus remove of that source) is
        regressed onto the SAME Q_s, and retention is how much of Q_s a
        held-out Delta recovers:
            R_{s->k} = 1 - ||Q_s - Qhat_s||^2 / (||Q_s - mean Q_s||^2 + eps).
        No chaining, no per-layer J ratios, no re-predicting the future at any
        layer."""
        Xc = col(calib_rows_, spec["source_key"])
        Xa = col(audit_rows_, spec["source_key"])
        Pc = make_P(calib_rows_)
        Qc, Qa = rs.source_component(Xc, Pc, Xa, lam=args.lam)

        # ---- source signal: how much of the (non-residual) audit P does the
        # fixed Q_s explain; gate = above the 95th pct of a within-strata
        # shuffle null (permuting X_s destroys the source-future link).
        sig = rs.explained_var(P_aud, Qa, eps=eps)
        rng_null = np.random.RandomState(FIXED_SEED + 5000 + s)
        nulls = []
        for _ in range(args.n_null):
            Xp = rs.permute_within_strata(Xa, strata, rng_null)
            nulls.append(rs.explained_var(P_aud,
                                          rs.apply_ridge_map(
                                              rs.fit_ridge_map(
                                                  Xc, Pc, lam=args.lam),
                                              Xp),
                                          eps=eps))
        null_p95 = float(np.percentile(nulls, 95))
        sig_ok = bool(sig > null_p95)

        # ---- identity at the source: X_s itself recovers Q_s (~1) ----
        mp_id = rs.fit_ridge_map(Xc, Qc, lam=args.lam_ret)
        R_id = rs.retention_map_R(mp_id, Qa, Xa, eps=eps)
        id_ok = bool(R_id >= args.identity_min)

        # ---- remove-source floor: Delta = 0 recovers ~nothing ----
        mp_floor = rs.fit_ridge_map(np.zeros_like(Xc), Qc,
                                    lam=args.lam_ret)
        R_floor = rs.retention_map_R(mp_floor, Qa, np.zeros_like(Xa),
                                     eps=eps)
        floor_ok = bool(abs(R_floor) <= args.floor_max)

        # ---- per-point retention from the paired-removal deltas ----
        points = [{"phys": spec["origin_phys"], "delta": "source",
                   "R": float(R_id), "ci_lo": float(R_id),
                   "ci_hi": float(R_id), "ok": True}]
        first_dc = first_da = None
        for phys, dk in spec["points"]:
            Dc = col(calib_rows_, dk)
            Da = col(audit_rows_, dk)
            if first_dc is None:
                first_dc, first_da = Dc, Da
            mp_k = rs.fit_ridge_map(Dc, Qc, lam=args.lam_ret)
            R_k = rs.retention_map_R(mp_k, Qa, Da, eps=eps)
            if do_ci:
                boot = rs.retention_bootstrap(
                    mp_k, Qa, Da, np.arange(len(audit_rows_)),
                    args.n_bootstrap, FIXED_SEED + 9000 + s * 10 + phys,
                    eps=eps)
                lo, hi = rs.retention_ci(boot)
            else:
                lo = hi = float(R_k)
            points.append({"phys": int(phys), "delta": dk,
                           "R": float(R_k), "ci_lo": lo, "ci_hi": hi,
                           "ok": True})

        # ---- mismatched-delta control (permuted Delta recovers ~0) ----
        if first_dc is not None:
            perm = rs.permute_within_strata(first_da, strata,
                                            np.random.RandomState(
                                                FIXED_SEED + 7000 + s))
            mp_perm = rs.fit_ridge_map(first_dc, Qc, lam=args.lam_ret)
            R_perm = rs.retention_map_R(mp_perm, Qa, perm, eps=eps)
            perm_ok = bool(R_perm - R_floor <= 0.02)
        else:
            R_perm, perm_ok = float("nan"), True

        line_ok = bool(sig_ok and id_ok and floor_ok and perm_ok
                       and mem_parity["ok"])
        return {
            "signal": {"value": float(sig), "null_p95": null_p95,
                       "ok": sig_ok},
            "identity": {"value": float(R_id), "ok": id_ok},
            "delete_floor": {"value": float(R_floor), "ok": floor_ok},
            "mismatched_delta": {"value": float(R_perm), "ok": perm_ok},
            "points": points,
            "line_ok": line_ok,
        }

    results = {}
    for s, spec in sorted(lines.items()):
        same_tail = compute_line(s, spec, calib_rows, audit_rows, do_ci=True)
        head_tail = compute_line(s, spec, calib_h_rows, audit_rows,
                                 do_ci=False)
        results[str(s)] = {
            "name": spec["name"], "source_key": spec["source_key"],
            "origin_phys": spec["origin_phys"],
            "same_tail": same_tail,
            "head_to_tail": head_tail,
        }
        st = same_tail
        ht = head_tail
        print("[line {}] same-tail sig {:.4f}(null {:.4f}) id {:.4f} "
              "floor {:.4f} | head->tail sig {:.4f}(null {:.4f})".format(
                  s, st["signal"]["value"], st["signal"]["null_p95"],
                  st["identity"]["value"], st["delete_floor"]["value"],
                  ht["signal"]["value"], ht["signal"]["null_p95"]),
              flush=True)
        for tag, res in (("same-tail", st), ("head->tail", ht)):
            print("  [{}] line_ok={} points={}".format(
                tag, res["line_ok"],
                [(p["phys"], round(p["R"], 3)) for p in res["points"]]),
                flush=True)

    report = {
        "protocol": "same-source explained-variance retention, leak-free "
                    "historical C, direct (non-chained) points; same-tail "
                    "causal calibration is the main result, head->tail is a "
                    "transfer stress test",
        "layout": {
            "audit_batches": args.audit_batches,
            "audit_block": [audit_lo, audit_hi],
            "calib_batches": args.calib_batches,
            "same_tail_calib_block": [calib_lo, calib_hi],
            "gap_batches": gap,
            "head_calib_block": [0, transfer_hi],
        },
        "n_audit_rows": len(audit_rows),
        "n_same_tail_calib_rows": len(calib_rows),
        "n_head_calib_rows": len(calib_h_rows),
        "lam": args.lam, "lam_ret": args.lam_ret, "eps": eps,
        "identity_min": args.identity_min, "floor_max": args.floor_max,
        "fixed_seed": FIXED_SEED,
        "memory_parity": mem_parity,
        "sources": results,
    }
    out = Path(args.ckpt).parent / "retention_audit_v2.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2), flush=True)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
