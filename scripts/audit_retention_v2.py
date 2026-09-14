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

Predictive-information retention (v3, NLL convention).  The figure asks how
much of a source node's *conditional predictive information* about the joint
leaf-to-root future survives one/two/three recursive aggregations toward the
root.  For each fixed leaf->root branch the audit measures, on held-out rows,
the joint-future NLL of a scorer fit TWO ways at the same dimension and same
architecture:

    base : uses Z^- = the ancestor state with this source's contribution
           removed (paired delete of the leaf interface vector);
    full : uses Z^+ = Z^- + Delta (the kept state),
    both scored with a candidate x state interaction feature
    psi = [C, S, e(c_s), e(c_p), S*(P e(c_s)), S*(P e(c_p))].

    I_k = (NLL_base - NLL_full)/ln2   (bits/sample), and the reported curve is

    R_k = I_k / I_source              ("normalized source-specific predictive
                                       gain", NOT a strict [0,1] fraction).

No chaining, no ratio of per-layer J, no per-layer future re-prediction.  The
source signal is fixed once as the joint-future NLL drop at the leaf position;
C is never subtracted (the keep/remove pair is on the SAME tree, so context,
siblings, edges and times are held fixed by construction).

Candidate protocol: for each branch node the future event is fetched strictly
after the root time (source->destination direction), paired with ONE sampled
negative; the label is "balanced sampled candidate existence" (the presented
candidate is randomly the positive or the negative; the probe must judge
whether the presented candidate is the true future).  The candidate manifest
is written once (--manifest-out) and consumed by every arm (--manifest-in),
with a SHA recorded in the pkl/report; cross-arm comparisons join on
(pair_id, line, phys) with pair_id = the globally-unique root event + branch
nodes.  dneg==dpos collisions are handled explicitly (--neg-collision).

Gates (run before plotting): source identifiable above a matched shuffle null;
delete-source (Delta = 0) floor; permutation null on Delta; A/A (same arm vs
itself); erase/source-dependence control; memory parity stored.  CI/curve
bootstrap is paired over the shared pair clusters.

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


# The three model-dependent blocks of C are the fixed projections of the
# path-outside OTHER-NEIGHBOR hidden states (a checkpoint quantity).  They are
# NOT shared across arms, so a fair cross-arm context must drop them; only the
# historical edge features/times, node/edge ids, selected slot and other-
# neighbour RAW structure may remain.
CTX_MODEL_DEP_BLOCKS = ((0, 16), (24, 40), (48, 64))


def _future_time_of(eid, eid2time):
    """Timestamp of a future event, refusing an id the stream never produced."""
    j = int(eid)
    if not (0 <= j < eid2time.size):
        raise RuntimeError(
            "future_event_id {} is outside the edge-id table (size {})".format(
                j, eid2time.size))
    t = float(eid2time[j])
    if not np.isfinite(t):
        raise RuntimeError(
            "future_event_id {} was never assigned a timestamp".format(j))
    return t


def ctx_shared_only(ctx):
    """Zero the three model-dependent other-neighbor blocks of a stored C."""
    out = np.array(ctx, dtype=np.float64, copy=True)
    for a, b in CTX_MODEL_DEP_BLOCKS:
        out[a:b] = 0.0
    return out


def _mask_rows_ctx(rows):
    for r in rows:
        if "ctx" in r and r["ctx"] is not None:
            r["ctx"] = ctx_shared_only(r["ctx"])
    return rows


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

    def query_src(self, node, time):
        """First event with ``node`` as SOURCE strictly after ``time`` (UCI
        link convention: predict the node's next out-edge)."""
        if not (0 <= node < self.n_nodes):
            return None
        rows = self._rows[node]
        if len(rows) == 0:
            return None
        pos = int(np.searchsorted(self.t[rows], time, side="right"))
        for k in range(pos, len(rows)):
            j = int(rows[k])
            if int(self.src[j]) == int(node):
                return j, float(self.t[j] - time), int(self.dst[j])
        return None


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
    ap.add_argument("--ckpt", default=None,
                    help="checkpoint to audit (required unless --recompute-from)")
    ap.add_argument("--manifest-out", default=None,
                    help="write the shared candidate manifest here (one file "
                         "consumed by every arm)")
    ap.add_argument("--manifest-in", default=None,
                    help="read candidates from a shared manifest instead of "
                         "re-sampling (cross-arm pairing)")
    ap.add_argument("--neg-collision", default="resample",
                    choices=["resample", "flip"],
                    help="when the sampled negative equals the positive "
                         "(dneg==dpos): resample deterministically until "
                         "different (default), or 'flip' to reproduce the "
                         "BenchTemp 1-bit pseudo-negative semantics")
    ap.add_argument("--ctx-shared-only", action="store_true",
                    help="drop the three model-dependent other-neighbor "
                         "blocks of C so the context is identical across arms")
    ap.add_argument("--ctx-candidate-pairs", action="store_true",
                    help="(v3) feed BOTH the positive and the negative "
                         "candidate of each node with a random swap label")
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
    ap.add_argument("--no-stats", action="store_true",
                    help="stop after dumping retention_rows.pkl; skip the "
                         "legacy in-process v2 statistics (superseded by the "
                         "offline B2 runner, and hours-slow on large streams)")
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
    assert int(args.n_layers) == 3, (
        'retention path bookkeeping currently assumes n_layers == 3')
    if args.recompute_from:
        with open(args.recompute_from, "rb") as f:
            rd = pickle.load(f)
        _m = rd.get("meta", {})
        run_stats(rd["calib"], rd["audit"], rd.get("head", []), args,
                  _m.get("memory_parity",
                         {"ok": False,
                          "detail": "memory parity not stored in pkl"}),
                  layout=_m.get("layout"), model_meta_override=_m)
        return
    if not args.ckpt:
        raise SystemExit("--ckpt is required unless --recompute-from is given")

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
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if args.model_kind == "ours":
        # project checkpoint: keys are embedding_module.host.* -> wrap FIRST.
        if ("model" not in ck) or ("tgn" not in ck["model"]):
            raise RuntimeError(
                "model-kind=ours expects a project checkpoint with "
                "model['tgn'] (and 'compressor'); got keys {}".format(
                    sorted(ck.keys())))
        if "compressor" not in ck["model"]:
            raise RuntimeError(
                "model-kind=ours but checkpoint has no 'compressor' block; "
                "use --model-kind tgn for a native TGN checkpoint")
        adapter = JodieTGNAdapter(tgn.embedding_module, compressor=comp,
                                  n_neighbors=n_neighbors)
        tgn.embedding_module = adapter
        tgn.load_state_dict(ck["model"]["tgn"], strict=True)
        comp.load_state_dict(ck["model"]["compressor"], strict=True)
    else:
        # native official TGN state_dict: keys are embedding_module.* and the
        # adapter must NOT be installed yet (it renames them to
        # embedding_module.host.*), otherwise load_state_dict misses/unexpected.
        raw = ck["model"]["tgn"] if ("model" in ck and
                                     isinstance(ck.get("model"), dict) and
                                     "tgn" in ck["model"]) else ck
        tgn.load_state_dict(raw, strict=True)
        adapter = JodieTGNAdapter(tgn.embedding_module, compressor=None,
                                  n_neighbors=n_neighbors)
        tgn.embedding_module = adapter
    tgn.eval()
    print("[model] kind={} n_layers={} n_neighbors={} loaded checkpoint "
          "epoch={} score={}".format(args.model_kind, n_layers, n_neighbors,
                                     ck.get("epoch"), ck.get("score")),
          flush=True)

    fut = FutureIndex(ds)
    # future-event TIMES by edge index, so the manifest can record when each
    # supervised future actually happens (needed to purge probe fits by future
    # time).  edge_idxs is a row index into the edge feature table, not
    # necessarily the stream position, so map it rather than assume identity.
    # Ids must be unique (a duplicate would silently overwrite) and are not
    # assumed to start at 0 (0 may be a padding id), so unassigned slots stay
    # NaN and a bad lookup is detectable rather than a silent time of 0.0.
    _eidx_all = np.asarray(ds.full.edge_idxs, np.int64)
    if np.unique(_eidx_all).size != _eidx_all.size:
        raise RuntimeError(
            "edge_idxs is not a bijection; the edge_id -> future_event_time "
            "map would silently overwrite entries")
    eid2time = np.full(int(_eidx_all.max()) + 1, np.nan, dtype=np.float64)
    eid2time[_eidx_all] = np.asarray(ds.full.timestamps, np.float64)

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
        # ---- paired removal of the source interface: zero the ENTIRE 172-d
        # interface vector that this source child hands UP into its parent's
        # aggregation (neighbor_lower[parent_row, source_slot, :]).  This keeps
        # the parent's own state, all siblings, edge feature/time and mask
        # unchanged -- only the selected source's propagated state is removed.
        if record and remove_depth is not None and layer == (4 - remove_depth):
            if remove_depth == 1 and is_top:
                for r, rec in path_state["recs"].items():
                    neighbor_lower = neighbor_lower.clone()
                    neighbor_lower[r, rec["slot3"], :] = 0.0
            else:
                for (pr, s), root_r in list(path_state["by_layer"].get(
                        layer, {}).items()):
                    flat_row = pr * n_neighbors_ + s
                    if flat_row >= len(source_nodes):
                        continue
                    neighbor_lower = neighbor_lower.clone()
                    neighbor_lower[flat_row, s, :] = 0.0
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
                    # 3-hop source = the selected leaf child's full interface
                    # state handed up into a2 (NOT a2's own aggregate).  h0 /
                    # node_info are kept separately and never conflated.
                    base["u0"] = neighbor_lower[flat_row, ch["slot"]] \
                        .detach().cpu().numpy()
                    # h0/node_info describe the LEAF child (the 3-hop source),
                    # not the a2 aggregator that consumed it.
                    if adapter.use_memory:
                        base["h0"] = memory[int(child)] \
                            .detach().cpu().numpy()
                    else:
                        base["h0"] = np.zeros_like(base["u0"])
                    base["node_info"] = adapter.host.node_features[
                        int(child)].detach().cpu().numpy()
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
        cur["eidx"] = e   # root event ids for globally-unique pair_id
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

    manifest_rows = []
    neg_collisions = 0
    manifest_in = None
    manifest_in_sha = None
    if args.manifest_in:
        with open(args.manifest_in, "rb") as _mf:
            _mj = json.load(_mf)
        _rows_in = _mj["rows"] if isinstance(_mj, dict) else _mj
        # self-consistency: recompute the manifest SHA and compare to stored
        _sha = hashlib.sha256(json.dumps(
            _rows_in, sort_keys=True, default=str).encode()).hexdigest()
        if isinstance(_mj, dict) and _mj.get("manifest_sha") not in (None, _sha):
            raise RuntimeError(
                "manifest_in SHA mismatch: stored {} != recomputed {}".format(
                    _mj.get("manifest_sha"), _sha))
        manifest_in_sha = _sha
        manifest_in = {tuple(m["pair_id"]): m for m in _rows_in}
        print("[manifest] loaded {} pairs from {} (sha {})".format(
            len(manifest_in), args.manifest_in, _sha[:16]), flush=True)

    def _rows_from(keep, rm_snaps):
        nonlocal neg_collisions
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
                    return (orig_z - rr[key]) if (rr is not None and key in rr) else None
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
            leaf_node = int(st1["leaf"]); a2_node = int(st1["node"])
            a1_node = int(st2["node"]); root_t = float(rec["t_root"])
            # globally-unique pair id: the root EVENT plus the branch nodes
            root_eid = int(cur["eidx"][r])
            pair_id = (root_eid, int(root), leaf_node, a2_node, a1_node)
            pair_key = "{}:{}:{}:{}:{}".format(*pair_id)
            nodes = {"leaf": leaf_node, "a2": a2_node, "a1": a1_node,
                     "root": root}
            F = {}
            okf = True
            if manifest_in is not None:
                mrow = manifest_in.get(pair_id)
                if mrow is None:
                    okf = False
                else:
                    for k, v in nodes.items():
                        if (int(mrow["nodes"][k]) != int(v)
                                or float(mrow["t_root"]) != root_t):
                            raise RuntimeError(
                                "manifest_in mismatch: pair {} node {} "
                                "stored nodes {} t_root {} vs runtime "
                                "nodes {} t_root {}".format(
                                    pair_id, k, mrow["nodes"],
                                    mrow["t_root"], nodes, root_t))
                        pres = int(mrow["presented"][k])
                        dpos = int(mrow["pos_cand"][k])
                        dneg = int(mrow["neg_cand"][k])
                        F[k] = {"eid": int(mrow["pos_future_event_id"][k]),
                                "dpos": dpos, "dneg": dneg,
                                "presented": pres,
                                "Y": 1 if pres == dpos else 0,
                                "cand_seed": int(mrow["candidate_seed"][k])}
            else:
                for k, v in nodes.items():
                    q = fut.query_src(v, root_t)
                    if q is None:
                        okf = False
                        break
                    jq, _dt, dpos = q
                    cand_seed = (int(v) * 1000003 + int(root_t)) % (2 ** 31)
                    g = np.random.RandomState(cand_seed)
                    dneg = int(pool[g.randint(len(pool))])
                    collided = int(dneg == int(dpos))
                    if collided and args.neg_collision == "resample":
                        for _try in range(64):
                            d2 = int(pool[g.randint(len(pool))])
                            if d2 != int(dpos):
                                dneg = d2
                                break
                        else:
                            raise RuntimeError(
                                "no non-colliding negative for node {} t {}"
                                .format(v, root_t))
                        collided = 0
                    elif collided:
                        neg_collisions += 1
                    bit = int(g.randint(2))
                    presented = int(dpos) if bit == 0 else int(dneg)
                    Y = 1 if presented == int(dpos) else 0
                    if args.neg_collision == "flip" and collided:
                        # BenchTemp pseudo-negative parity: when dneg==dpos the
                        # label is the flipped bit instead of a real negative.
                        Y = 1 - Y
                    F[k] = {"eid": int(fut.eidx[jq]), "dpos": int(dpos),
                            "dneg": int(dneg), "presented": int(presented),
                            "Y": int(Y), "cand_seed": int(cand_seed)}
            if not okf:
                continue
            manifest_rows.append({
                "pair_id": list(pair_id), "root_event_id": int(root_eid),
                "root_node": int(root), "t_root": root_t, "nodes": nodes,
                "pos_cand": {k: F[k]["dpos"] for k in F},
                "neg_cand": {k: F[k]["dneg"] for k in F},
                "presented": {k: F[k]["presented"] for k in F},
                "presented_bits": {k: (0 if F[k]["presented"] == F[k]["dpos"]
                                       else 1) for k in F},
                "pos_future_event_id": {k: F[k]["eid"] for k in F},
                # when each supervised future event actually happens, so a
                # probe fit can be purged by future time without re-extraction
                "future_event_time": {k: _future_time_of(F[k]["eid"], eid2time)
                                      for k in F},
                "candidate_seed": {k: F[k]["cand_seed"] for k in F},
                "sampler_formula_version":
                    "seed=(v*1000003+root_t)%2**31;dneg=randint;bit=randint",
                "label_kind": "balanced_sampled_candidate_existence"})
            # keys must match the "z1"/"z2"/"z3" strings used by pts below
            zk = {"z1": z1, "z2": z2, "z3": z3}
            lines_local = [
                ("Y_leaf", "Y_a2", st1["u0"], 0,
                 [(1, dz3_2, "z1"), (2, dz3_1, "z2"), (3, dz3_r, "z3")]),
                ("Y_a2", "Y_a1", z1, 1,
                 [(2, dz2_1, "z2"), (3, dz2_r, "z3")]),
                ("Y_a1", "Y_root", z2, 2,
                 [(3, dz1_r, "z3")]),
            ]
            for sk, pk, src_state, origin, pts in lines_local:
                fsk = sk.replace("Y_", ""); fpk = pk.replace("Y_", "")
                Ys = int(F[fsk]["Y"]); Yp = int(F[fpk]["Y"])
                hs = _cand_hash(F[fsk]["presented"], 11)
                hp = _cand_hash(F[fpk]["presented"], 22)
                rows.append({"line": sk, "phys": origin, "ctx": ctx,
                             "rem": np.zeros_like(src_state),
                             "keep": src_state, "y_s": Ys, "y_p": Yp,
                             "cand_s": hs, "cand_p": hp,
                             "pair_id": pair_id, "pair_key": pair_key,
                             "row_id": (pair_key, sk, int(origin))})
                for phys, dk, zkey in pts:
                    rows.append({"line": sk, "phys": phys, "ctx": ctx,
                                 "rem": zk[zkey] - dk, "keep": zk[zkey],
                                 "y_s": Ys, "y_p": Yp,
                                 "cand_s": hs, "cand_p": hp,
                                 "pair_id": pair_id, "pair_key": pair_key,
                                 "row_id": (pair_key, sk, int(phys))})
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
    if not audit_rows or not calib_rows:
        print("FATAL: no leaf-to-root paths extracted", flush=True)
        return
    if not calib_h_rows:
        # the head->tail transfer stress test is optional
        print("[head] WARNING: no head-calibration rows "
              "(head->tail transfer stress skipped)", flush=True)
    # persist the extracted row matrices so a later statistics-only change can
    # be recomputed without re-running the model (15 min -> seconds)
    dump_path = Path(args.ckpt).parent / "retention_rows.pkl"
    def _file_sha(paths):
        h = hashlib.sha256()
        per = {}
        for pp in paths:
            fh_ = hashlib.sha256()
            try:
                with open(pp, "rb") as fh:
                    for chunk in iter(lambda: fh.read(1 << 20), b""):
                        fh_.update(chunk)
            except OSError as ex:
                raise FileNotFoundError(
                    "dataset file missing/unreadable: {} ({})".format(pp, ex))
            per[Path(pp).name] = fh_.hexdigest()[:16]
            h.update(fh_.digest())
        return h.hexdigest()[:16], per

    _dd = Path(args.data_dir)
    meta = model_meta(args.ckpt)
    _dsha, _dper = _file_sha([
        str(_dd / "ml_{}.csv".format(args.data_name)),
        str(_dd / "ml_{}.npy".format(args.data_name)),
        str(_dd / "ml_{}_node.npy".format(args.data_name))])
    # shared manifest: write once (from any arm) and record its SHA so every
    # arm can be proven to have consumed the identical candidate set.
    _mrows_json = json.dumps(manifest_rows, sort_keys=True, default=str)
    manifest_sha = hashlib.sha256(_mrows_json.encode()).hexdigest()
    if args.manifest_out:
        with open(args.manifest_out, "w") as f:
            json.dump({"manifest_sha": manifest_sha,
                       "neg_collisions": int(neg_collisions),
                       "sampler_formula_version":
                           "seed=(v*1000003+root_t)%2**31",
                       "rows": manifest_rows}, f, default=str)
        print("[manifest] wrote {} pairs (sha {}) to {}".format(
            len(manifest_rows), manifest_sha[:16], args.manifest_out),
            flush=True)
    if args.manifest_in:
        if manifest_sha != manifest_in_sha:
            raise RuntimeError(
                "manifest_in sha {} != regenerated sha {} (arm candidates "
                "differ)".format(manifest_in_sha, manifest_sha))
    meta.update({"model_kind": args.model_kind, "n_layers": args.n_layers,
                 "n_neighbors": args.n_neighbors, "bs": args.bs,
                 "data_name": args.data_name,
                 "dataset_hash": _dsha, "dataset_file_hashes": _dper,
                 "manifest_sha": manifest_sha,
                 "neg_collisions": int(neg_collisions),
                 "manifest_in": args.manifest_in,
                 "memory_parity": mem_parity,
                 # the rows store Y = 1 iff the PRESENTED candidate is the true
                 # future; the canonical probe label is the true candidate's
                 # presentation POSITION, so Y_canonical = 1 - Y_row
                 "row_label_kind": "presented_is_positive",
                 "schema_version": 2,
                 "layout": {"audit_block": [audit_lo, audit_hi],
                            "same_tail_calib_block": [calib_lo, calib_hi],
                            "head_calib_block": [0, transfer_hi]}})
    with open(dump_path, "wb") as f:
        pickle.dump({"audit": audit_rows, "calib": calib_rows,
                     "head": calib_h_rows, "meta": meta,
                     "manifest": manifest_rows}, f)
    print("[dump] rows saved to", dump_path, flush=True)

    if args.no_stats:
        print("[stats] skipped (--no-stats); rows are the deliverable here",
              flush=True)
        return
    run_stats(calib_rows, audit_rows, calib_h_rows, args, mem_parity,
              layout={"audit_batches": args.audit_batches,
                      "audit_block": [audit_lo, audit_hi],
                      "calib_batches": args.calib_batches,
                      "same_tail_calib_block": [calib_lo, calib_hi],
                      "gap_batches": gap,
                      "head_calib_block": [0, transfer_hi]})
    return

def run_stats(calib_rows, audit_rows, calib_h_rows, args, mem_parity,
              layout=None, model_meta_override=None):
    """Conditional future-predictive information via held-out joint-future NLL
    with a candidate x state INTERACTION scorer and SAME-dim base/full
    (base = Z^-; full = Z^+ = Z^- + Delta).  Target = 2*Y_s + Y_pa(s) in
    {0,1,2,3} (Y=1 iff the presented candidate is the real future dest).
    R_{s,k} = I_{s,k} / I_{s,0}; NOT clipped (R>1 flagged)."""
    lam = args.lam_ret

    if not audit_rows or not calib_rows:
        print("FATAL: no rows to score", flush=True)
        return
    if getattr(args, "ctx_shared_only", False):
        for _rows in (calib_rows, audit_rows, calib_h_rows):
            _mask_rows_ctx(_rows)
        print("[ctx] model-dependent other-neighbor blocks zeroed "
              "(shared-only context)", flush=True)
    d_state = calib_rows[0]["rem"].shape[0]
    d_cand = calib_rows[0]["cand_s"].shape[0]
    P = rs.fixed_proj(d_cand, d_state, FIXED_SEED + 77)

    def _A(rows_, key):
        return np.stack([r[key] for r in rows_])

    def _Y(rows_):
        return np.asarray([2 * r["y_s"] + r["y_p"] for r in rows_],
                          dtype=np.int64)

    LINES = ["Y_leaf", "Y_a2", "Y_a1"]
    results = {}
    for line in LINES:
        cr = [r for r in calib_rows if r["line"] == line]
        ar = [r for r in audit_rows if r["line"] == line]
        if len(cr) < 40 or len(ar) < 40:
            continue
        phs = sorted({r["phys"] for r in cr} & {r["phys"] for r in ar})
        info, ci = {}, {}
        for ph in phs:
            csel = [r for r in cr if r["phys"] == ph]
            asel = [r for r in ar if r["phys"] == ph]
            if len(csel) < 40 or len(asel) < 40:
                continue
            yc = _Y(csel)
            ya = _Y(asel)
            if len(np.unique(yc)) < 4:
                raise RuntimeError(
                    "calibration missing a joint-future class for line {} "
                    "phys {}: {}".format(line, ph, sorted(np.unique(yc))))
            args_c = (_A(csel, "ctx"), _A(csel, "rem"), _A(csel, "keep"),
                      _A(csel, "cand_s"), _A(csel, "cand_p"), yc)
            args_a = (_A(asel, "ctx"), _A(asel, "rem"), _A(asel, "keep"),
                      _A(asel, "cand_s"), _A(asel, "cand_p"), ya)
            ib = rs.info_bits_interaction(*args_c, *args_a, P, lam=lam)
            info[ph] = float(ib)
            groups_a = np.asarray([r["pair_key"] for r in asel], dtype=object)
            ci[ph] = rs.cluster_bootstrap_ci(
                *args_c, *args_a, groups_a, P, n_boot=args.n_bootstrap,
                lam=lam)
        origin = 0 if line == "Y_leaf" else (1 if line == "Y_a2" else 2)
        i0 = info.get(origin)
        i0ci = ci.get(origin)
        pts = []
        for ph in sorted(info):
            R = (info[ph] / i0) if (i0 is not None and i0 > 1e-9)                 else float("nan")
            pts.append({"phys": int(ph), "info_bits": float(info[ph]),
                        "ci95": [float(ci[ph][0]), float(ci[ph][1])]
                        if ph in ci else [float("nan"), float("nan")],
                        "R": float(R),
                        "R_gt1_flag": bool(R == R and R > 1.0)})
        results[line] = {
            "origin_phys": origin, "info_source_bits": i0,
            "info_source_ci95": [float(i0ci[0]), float(i0ci[1])]
            if i0ci else None, "n_calib": len(cr), "n_audit": len(ar),
            "points": pts}

    report = {
        "protocol": "conditional future-predictive information: held-out "
                    "joint-future NLL drop (bits/sample) with a candidate x "
                    "state interaction scorer and same-dim base/full "
                    "(Z^- vs Z^+).  R_k=I_k/I_source; no clipping (R>1 "
                    "flagged).  Naming: normalized source-specific predictive "
                    "gain (not a strict information-retention fraction).",
        "model": model_meta_override or model_meta(args.ckpt),
        "model_kind": ((model_meta_override or {}).get("model_kind")
                       or args.model_kind),
        "n_audit_rows": len(audit_rows), "n_calib_rows": len(calib_rows),
        "lam_ret": lam, "memory_parity": mem_parity, "layout": layout,
        "sources": results,
    }
    # provenance: in recompute mode the SOURCE of truth is the stored meta and
    # the pkl's directory -- never a possibly-unrelated --ckpt.
    if args.recompute_from:
        out = Path(args.recompute_from).parent / "retention_audit_v3.json"
    else:
        out = Path(args.ckpt).parent / "retention_audit_v3.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(json.dumps(report, indent=2, default=str), flush=True)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
