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
import hashlib
import json
import os
import pickle
import re
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


def _cand_hash(node, seed):
    """Fixed 16-d candidate encoding (node-id hash; symmetric pos/neg)."""
    g = np.random.RandomState((int(node) * 2654435761 + int(seed)) & 0xFFFFFFFF)
    return g.normal(0.0, 1.0, 16) / 4.0


def _sample_neg(node, root_t, pool, salt=0):
    """Official-sampler negative dst for a strict-future query (cached by the
    caller; TGN and ours share the SAME candidate)."""
    g = np.random.RandomState(((int(node) * 1000003 + int(root_t)) + int(salt))
                              % (2 ** 31))
    return int(pool[g.randint(len(pool))])


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
def model_meta(ckpt_path):
    """Arm / seed / epoch / score + content hash of the audited checkpoint, so
    a result can never be read without knowing exactly which model produced
    it."""
    p = Path(ckpt_path)
    arm = p.parent.name
    gp = p.parent.parent.name
    m = re.search(r"seed(\d+)", gp)
    seed = int(m.group(1)) if m else None
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return {
        "arm": arm,
        "seed": seed,
        "epoch": ck.get("epoch"),
        "score": ck.get("score"),
        "ckpt": str(p),
        "ckpt_sha256": hashlib.sha256(p.read_bytes()).hexdigest()[:16],
    }


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
    ap.add_argument("--n-dir", type=int, default=5,
                    help="number of fixed predictive directions extracted for "
                         "Q_s (top predictable part of the local future)")
    ap.add_argument("--recompute-from", default=None,
                    help="recompute statistics from a saved retention_rows.pkl "
                         "(no model run); ignores the extraction/model args")
    ap.add_argument("--memory-parity-batches", type=int, default=5)
    ap.add_argument("--lam", type=float, default=1e-2)
    ap.add_argument("--lam-ret", type=float, default=1e-2)
    ap.add_argument("--identity-min", type=float, default=0.90)
    ap.add_argument("--floor-max", type=float, default=0.05)
    ap.add_argument("--model-kind", default="ours", choices=["ours", "tgn"],
                    help="ours = TGN+Gamma (load tgn+compressor); tgn = native "
                         "TGN (compressor=None, load tgn only)")
    ap.add_argument("--n-neighbors", type=int, default=5)
    ap.add_argument("--n-layers", type=int, default=3)
    args = ap.parse_args()
    if args.recompute_from:
        with open(args.recompute_from, "rb") as f:
            rd = pickle.load(f)
        run_stats(rd["calib"], rd["audit"], rd["head"], args,
                  {"ok": True,
                   "n_batches": args.memory_parity_batches,
                   "detail": "recomputed from saved rows (parity was verified "
                             "at extraction)"},
                  layout=rd.get("meta", {}).get("layout"))
        return

    eps = 1e-6
    device = torch.device(
        "cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    ds = UCILinkDataset(args.data_dir, data_name=args.data_name)

    train = ds.train
    ts = train.timestamps.astype(np.float64)
    ms, ss = float(ts.mean()), float(ts.std()) + 1e-8
    n_neighbors = int(args.n_neighbors)
    n_layers = int(args.n_layers)
    finder = get_neighbor_finder(
        train, uniform=False, max_node_idx=ds.n_nodes - 1)
    tgn = TGN(
        neighbor_finder=finder,
        node_features=ds.node_features.astype(np.float32),
        edge_features=ds.edge_features.astype(np.float32),
        device=device, n_layers=n_layers, n_heads=2, dropout=0.1,
        use_memory=True,
        message_dimension=172, memory_dimension=172,
        memory_update_at_start=True,
        embedding_module_type="graph_attention", message_function="identity",
        aggregator_type="last", n_neighbors=n_neighbors,
        mean_time_shift_src=ms, std_time_shift_src=ss,
        mean_time_shift_dst=ms, std_time_shift_dst=ss).to(device)
    cfg = RPBConfig(
        state_dims={"tjo:layer{}".format(l): 172 for l in range(n_layers + 1)},
        own_dims={"tjo:layer{}".format(l): 172 for l in range(n_layers + 1)},
        width_D=128, m=64, lambda_kf=0.0, ridge_eps=1e-3,
        kf_group_batches=8, kf_min_abs=64)
    comp = (RecursiveCompressor(cfg).to(device)
            if args.model_kind == "ours" else None)
    adapter = JodieTGNAdapter(tgn.embedding_module, compressor=comp,
                              n_neighbors=n_neighbors)
    tgn.embedding_module = adapter
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    tgn.load_state_dict(ck["model"]["tgn"])
    if comp is not None and "compressor" in ck["model"]:
        comp.load_state_dict(ck["model"]["compressor"])
    tgn.eval()
    print("[model] kind={} n_layers={} n_neighbors={} loaded checkpoint "
          "epoch={} score={}".format(args.model_kind, n_layers, n_neighbors,
                                     ck.get("epoch"), ck.get("score")),
          flush=True)

    fut = FutureIndex(ds)

    # ---- leaf-to-root path-tracking compute (re-implemented hook) ----
    # n_neighbors / n_layers come from args (must match the checkpoint)
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
            return raw_source, raw_source

        # ---- bookkeeping only when record=True; the tensor math below is
        # bit-identical to the plain host recursion either way, so silent
        # prefix replay advances memory exactly like a single real pass ----
        source_paths = {}
        if record:
            source_paths = {
                int(row): list(path) + [(0, 0.0)]
                for row, path in trace_paths.items()}

        source_lower, _ = traced_compute(
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
                    # unified path slot: pick the child at THIS node via the
                    # same hash rule (never reuse the parent's slot index s).
                    s2 = _slot_hash(node, layer - 1) % n_neighbors_
                    child = int(neighbors[flat_row, s2])
                    if child != 0:
                        self_path_choices[(root_r, layer)] = {
                            "node": node,
                            "t": float(timestamps[flat_row]),
                            "slot": s2,
                            "flat_row": flat_row,
                        }

        neighbor_lower, _ = traced_compute(
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
        # ---- paired removal of the WHOLE source interface at this layer:
        # replace that interface's host pre-compression aggregate U0 with a
        # reference (0) and then run Gamma normally (ours) / pass vanilla (TGN).
        # This removes the source node's entire U0 -- NOT a single leaf-child
        # slot -- which is what the retention question asks about.
        rm_rows = []
        if record and remove_depth is not None and layer == (4 - remove_depth):
            if remove_depth == 1 and is_top:
                for r, rec in path_state["recs"].items():
                    rm_rows.append(int(r))
            else:
                for (pr, s), root_r in list(path_state["by_layer"].get(
                        layer, {}).items()):
                    flat_row = pr * n_neighbors_ + s
                    if flat_row < len(source_nodes):
                        rm_rows.append(int(flat_row))
        vanilla = adapter.host.aggregate(
            layer, source_lower, source_time, neighbor_lower, edge_time,
            edge_features, mask)
        if rm_rows:
            vanilla = vanilla.clone()
            vanilla[rm_rows] = 0.0
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
                    # U0 = host PRE-COMPRESSION aggregate at this interface
                    # (NOT a selected leaf-child state); h0/node_info are the
                    # raw memory / node-feature states kept separately so the
                    # three quantities are never conflated.
                    base["u0"] = vanilla[flat_row].detach().cpu().numpy()
                    if adapter.use_memory:
                        base["h0"] = memory[int(ch["node"])] \
                            .detach().cpu().numpy()
                    else:
                        base["h0"] = np.zeros_like(base["u0"])
                    base["node_info"] = adapter.host.node_features[
                        int(ch["node"])].detach().cpu().numpy()
                    base["leaf"] = child
                    path_state["stash"][(root_r, 1)] = base
        return z, z

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
        pool = ds.full_dst_pool
        rows = []
        for r, rec in recs.items():
            if (r, 1) not in stash or (r, 2) not in stash:
                continue
            st1 = stash[(r, 1)]
            st2 = stash[(r, 2)]
            root = rec["root"]
            node_times = [st1["t"], st2["t"], rec["t_root"]]
            edge_feats = [st1["edge_feat"], st2["edge_feat"],
                          rec["edge_feat_top"]]
            edge_times = [st1["edge_time"], st2["edge_time"],
                          rec["edge_time_top"]]
            om = [st1["other_neighbors"], st2["other_neighbors"],
                  rec["other_neighbors"]]
            ctx = _ctx_vector(root, st1["leaf"], node_times, edge_feats,
                              edge_times, om)

            def _dz(var, key, orig_z):
                if key == "z3":
                    rr = var["recs"].get(r)
                    return (orig_z - rr[key]) if rr is not None                         and key in rr else None
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
            leaf_node = int(st1["leaf"])
            a2_node = int(st1["node"])
            a1_node = int(st2["node"])
            root_t = float(rec["t_root"])
            # ---- strict-future link queries for the four path nodes; the real
            # future edge is y=1 and the official-sampler negative is y=0.
            # candidates are cached per (node, root_t) and shared by TGN/ours.
            futq = {}
            okf = True
            for k, node in [("leaf", leaf_node), ("a2", a2_node),
                            ("a1", a1_node), ("root", root)]:
                q = fut.query(node, root_t)
                if q is None:
                    okf = False
                    break
                jq, dtq, cpq = q
                futq[k] = {"cp": int(cpq),
                           "cn": _sample_neg(node, root_t, pool)}
            if not okf:
                continue
            zk = {1: z1, 2: z2, 3: z3}
            # per source line: (src key, parent key, source state, origin phys,
            # downstream [(phys, delta, keep-key)])
            lines_local = [
                ("Y_leaf", "Y_a2", st1["u0"], 0,
                 [(1, dz3_2, "z1"), (2, dz3_1, "z2"), (3, dz3_r, "z3")]),
                ("Y_a2", "Y_a1", z1, 1,
                 [(2, dz2_1, "z2"), (3, dz2_r, "z3")]),
                ("Y_a1", "Y_root", z2, 2,
                 [(3, dz1_r, "z3")]),
            ]
            for sk, pk, src_state, origin, pts in lines_local:
                fs = futq[sk.replace("Y_", "")]
                fp = futq[pk.replace("Y_", "")]
                combos = [(1, 1, fs["cp"], fp["cp"]), (1, 0, fs["cp"], fp["cn"]),
                          (0, 1, fs["cn"], fp["cp"]), (0, 0, fs["cn"], fp["cn"])]
                for y_s, y_p, cand_s, cand_p in combos:
                    hs = _cand_hash(cand_s, 11)
                    hp = _cand_hash(cand_p, 22)
                    rows.append({"line": sk, "phys": origin, "ctx": ctx,
                                 "rem": np.zeros_like(src_state),
                                 "delta": src_state, "y_s": y_s, "y_p": y_p,
                                 "cand_s": hs, "cand_p": hp})
                    for phys, dk, zkey in pts:
                        rows.append({"line": sk, "phys": phys, "ctx": ctx,
                                     "rem": zk[zkey] - dk, "delta": dk,
                                     "y_s": y_s, "y_p": y_p,
                                     "cand_s": hs, "cand_p": hp})
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
    # persist the extracted row matrices so a later statistics-only change can
    # be recomputed without re-running the model (15 min -> seconds)
    dump_path = Path(args.ckpt).parent / "retention_rows.pkl"
    meta = model_meta(args.ckpt)
    meta.update({"model_kind": args.model_kind, "n_layers": args.n_layers,
                 "n_neighbors": args.n_neighbors, "bs": args.bs,
                 "data_name": args.data_name,
                 "dataset_hash": hashlib.sha256(
                     (str(Path(args.data_dir).resolve()) + args.data_name)
                     .encode()).hexdigest()[:16],
                 "layout": {"audit_block": [audit_lo, audit_hi],
                            "same_tail_calib_block": [calib_lo, calib_hi],
                            "head_calib_block": [0, transfer_hi]}})
    with open(dump_path, "wb") as f:
        pickle.dump({"audit": audit_rows, "calib": calib_rows,
                     "head": calib_h_rows, "meta": meta}, f)
    print("[dump] rows saved to", dump_path, flush=True)

    run_stats(calib_rows, audit_rows, calib_h_rows, args, mem_parity,
              layout={"audit_batches": args.audit_batches,
                      "audit_block": [audit_lo, audit_hi],
                      "calib_batches": args.calib_batches,
                      "same_tail_calib_block": [calib_lo, calib_hi],
                      "gap_batches": gap,
                      "head_calib_block": [0, transfer_hi]})
    return

def run_stats(calib_rows, audit_rows, calib_h_rows, args, mem_parity,
              layout=None):
    """Conditional future-predictive information via held-out joint-future NLL.

    Per source line s: I_{s,k} = (NLL(C,Z^-_{s->k}) - NLL(C,Z^-_{s->k},Delta))
    / ln2  [bits/sample]; target = 2*Y_s + Y_pa(s) in {0,1,2,3} (Y=1 for the
    real strict-future edge, 0 for the official-sampler negative candidate).
    R_{s,k} = I_{s,k} / I_{s,0}.  NOT clipped: R>1 is flagged as a
    DPI / finite-sample violation, never silently clamped.
    """
    lam = args.lam_ret
    LINES = ["Y_leaf", "Y_a2", "Y_a1"]

    def _feat(rows_):
        ctx = np.stack([r["ctx"] for r in rows_])
        rem = np.stack([r["rem"] for r in rows_])
        dl = np.stack([r["delta"] for r in rows_])
        cs = np.stack([r["cand_s"] for r in rows_])
        cp = np.stack([r["cand_p"] for r in rows_])
        base = np.concatenate([ctx, rem, cs, cp], axis=1)
        full = np.concatenate([base, dl], axis=1)
        return base, full

    def _tgt(rows_):
        return np.asarray([2 * r["y_s"] + r["y_p"] for r in rows_],
                          dtype=np.int64)

    results = {}
    for line in LINES:
        cr = [r for r in calib_rows if r["line"] == line]
        ar = [r for r in audit_rows if r["line"] == line]
        if len(cr) < 40 or len(ar) < 40:
            continue
        cb, cf = _feat(cr)
        ab, af = _feat(ar)
        yc = _tgt(cr)
        ya = _tgt(ar)
        phs = sorted({r["phys"] for r in cr} & {r["phys"] for r in ar})
        info = {}
        for ph in phs:
            cidx = [i for i, r in enumerate(cr) if r["phys"] == ph]
            aidx = [i for i, r in enumerate(ar) if r["phys"] == ph]
            if len(cidx) < 40 or len(aidx) < 40:
                continue
            yc_ph, ya_ph = yc[cidx], ya[aidx]
            if len(np.unique(yc_ph)) < 2 or len(np.unique(ya_ph)) < 2:
                continue
            ib = rs.conditional_info_bits(
                cb[cidx], cf[cidx], yc_ph, ab[aidx], af[aidx], ya_ph, lam=lam)
            info[ph] = float(ib)
        origin = 0 if line == "Y_leaf" else (1 if line == "Y_a2" else 2)
        i0 = info.get(origin)
        pts = []
        for ph in sorted(info):
            R = (info[ph] / i0) if (i0 is not None and i0 > 1e-9)                 else float("nan")
            pts.append({"phys": int(ph), "info_bits": float(info[ph]),
                        "R": float(R),
                        "R_gt1_flag": bool(R == R and R > 1.0)})
        results[line] = {"origin_phys": origin, "info_source_bits": i0,
                         "n_calib": len(cr), "n_audit": len(ar),
                         "points": pts}

    report = {
        "protocol": "conditional future-predictive information (held-out "
                    "joint-future NLL drop, bits/sample); R_k = I_k / I_source; "
                    "no clipping (R>1 flagged)",
        "model": model_meta(args.ckpt),
        "model_kind": args.model_kind,
        "n_audit_rows": len(audit_rows), "n_calib_rows": len(calib_rows),
        "lam_ret": lam, "identity_min": args.identity_min,
        "floor_max": args.floor_max, "memory_parity": mem_parity,
        "layout": layout,
        "sources": results,
    }
    out = Path(args.ckpt).parent / "retention_audit_v3.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(json.dumps(report, indent=2, default=str), flush=True)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
