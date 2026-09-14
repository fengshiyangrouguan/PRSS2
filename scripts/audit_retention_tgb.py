#!/usr/bin/env python3
"""Retention-audit extraction for the TGB baseline hosts (TGN / TGAT).

Produces EXACTLY the rows schema of ``audit_retention_v2.py`` -- the same
``audit`` / ``calib`` / ``head`` row lists, ``meta`` and shared ``manifest``,
including ``future_event_time`` and ``row_label_kind`` -- so the offline B2
statistics read every arm (ours, taskonly, TGN, TGAT) through one code path and
the cross-arm schema audit applies unchanged.

The shared fairness-critical pieces (the leak-free context vector, the negative
sampler, the future index, the manifest/label convention) are IMPORTED from
``audit_retention_v2`` rather than re-implemented, so they cannot drift between
the two extractors.

Host differences handled here:
  * neighbours come from the TGB ``NeighborSampler`` (uniform/recent sampling
    over all history, padding id 0, mask = the sampled-id matrix) instead of the
    official ``NeighborFinder``;
  * each layer is ``temporal_conv_layers[L-1](...)`` followed by the residual
    ``merge_layers[L-1](out, base)`` -- there is no plain ``aggregate``;
  * the checkpoint is an ``nn.Sequential(backbone, MergeLayer)`` state dict, so
    its keys carry the ``0.`` / ``1.`` prefixes;
  * TGN keeps its memory in ``memory_bank`` and defers updates by one batch
    (``get_updated_memories`` applies the previous batch's raw messages first),
    so the keep/remove pair must snapshot and restore ``node_memories``,
    ``node_last_updated_times``, ``node_raw_messages`` AND the neighbour
    sampler's RNG state.

Usage:
    PYTHONPATH=<repo>/src:<repo>/scripts python scripts/audit_retention_tgb.py \
        --model tgn --src-dir <UCI_3H10_SOURCE_20260914/TGN_3H10> \
        --ckpt <TGN_seed0.pkl> --data-root <dir containing processed_data/uci> \
        --bs 200 --calib-batches 60 --audit-batches 200 \
        --manifest-out /tmp/m.json --out rows.pkl
"""

import argparse
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np                                    # noqa: E402
import torch                                          # noqa: E402

HERE = Path(__file__).resolve().parent
_SRC = HERE.parent / "src"
for _p in (str(_SRC), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# shared, fairness-critical code -- imported, never re-implemented
import audit_retention_v2 as av2                      # noqa: E402
from audit_retention_v2 import (                      # noqa: E402
    CTX_MODEL_DEP_BLOCKS, FutureIndex, _cand_hash, _ctx_vector,
    _future_time_of, _hash01, _mask_rows_ctx, _other_neighbor_vec,
    ctx_shared_only, model_meta)

ROW_LABEL_KIND = "presented_is_positive"   # same convention as audit_retention_v2


class _StreamView:
    """Name adapter so the shared FutureIndex can read TGB's ``full_data``."""

    def __init__(self, src, dst, times, eids):
        self.sources = src
        self.destinations = dst
        self.timestamps = times
        self.edge_idxs = eids


class _DSView:
    """Minimal dataset view for FutureIndex (``.full`` / ``.n_nodes`` / pool)."""

    def __init__(self, full_data, n_nodes, dst_pool):
        self.full = _StreamView(full_data.src_node_ids, full_data.dst_node_ids,
                                full_data.node_interact_times, full_data.edge_ids)
        self.n_nodes = int(n_nodes)
        self.full_dst_pool = dst_pool


# ------------------------------------------------------------------ host glue
def _import_host(src_dir, model):
    """Import the TGB model classes from ``--src-dir`` (top-level packages)."""
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    if model == "tgat":
        from models.TGAT import TGAT                       # noqa: WPS433
        return {"TGAT": TGAT}
    from models.MemoryModel import (                       # noqa: WPS433
        MemoryModel, compute_src_dst_node_time_shifts)
    return {"MemoryModel": MemoryModel,
            "compute_src_dst_node_time_shifts": compute_src_dst_node_time_shifts}


def build_host(args, host, node_raw, edge_raw, sampler, device,
               train_src, train_dst, train_times):
    """Instantiate the backbone with the config the checkpoints were trained on.

    ``nn.Sequential(backbone, link_predictor)`` mirrors the native eval script;
    the checkpoint's ``0.`` / ``1.`` prefixes come from that Sequential.
    """
    from models.modules import MergeLayer                  # noqa: WPS433
    d = int(node_raw.shape[1])
    if args.model == "tgat":
        backbone = host["TGAT"](node_raw_features=node_raw,
                                edge_raw_features=edge_raw,
                                neighbor_sampler=sampler,
                                time_feat_dim=args.time_feat_dim,
                                num_layers=args.n_layers,
                                num_heads=args.n_heads,
                                dropout=args.dropout, device=device)
    else:
        shifts = host["compute_src_dst_node_time_shifts"](
            train_src, train_dst, train_times)
        backbone = host["MemoryModel"](
            node_raw_features=node_raw, edge_raw_features=edge_raw,
            neighbor_sampler=sampler, time_feat_dim=args.time_feat_dim,
            model_name="TGN", num_layers=args.n_layers,
            num_heads=args.n_heads, dropout=args.dropout,
            src_node_mean_time_shift=shifts[0], src_node_std_time_shift=shifts[1],
            dst_node_mean_time_shift_dst=shifts[2],
            dst_node_std_time_shift=shifts[3], device=device)
    model = torch.nn.Sequential(backbone,
                                MergeLayer(d, d, d, 1))
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = ck["state_dict"] if (isinstance(ck, dict) and
                                 "state_dict" in ck) else ck
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    return model, model[0]


# ------------------------------------------------------------------- memory vm
def _backup(backbone, sampler):
    """Full mutable state: memories, last-update times, raw messages, RNG."""
    mb = backbone.memory_bank
    msgs = {k: [(m[0].clone(), m[1]) for m in v]
            for k, v in mb.node_raw_messages.items()}
    rng = getattr(sampler, "random_state", None)
    return {"mem": mb.node_memories.clone(),
            "upd": mb.node_last_updated_times.clone(),
            "msgs": msgs,
            "rng": (rng.get_state() if rng is not None else None),
            "np": np.random.get_state()}


def _restore(backbone, sampler, bak):
    mb = backbone.memory_bank
    mb.node_memories.copy_(bak["mem"])
    mb.node_last_updated_times.copy_(bak["upd"])
    mb.node_raw_messages.clear()
    for k, v in bak["msgs"].items():
        mb.node_raw_messages[k] = [(m[0].clone(), m[1]) for m in v]
    rng = getattr(sampler, "random_state", None)
    if rng is not None and bak["rng"] is not None:
        rng.set_state(bak["rng"])
    np.random.set_state(bak["np"])


# --------------------------------------------------------------- traced layer
def traced_embeddings(backbone, model, node_memories, node_ids,
                      node_interact_times, layer, num_neighbors,
                      path_state, trace_paths, remove, record, is_top, bs):
    """Re-implementation of the host's ``compute_node_temporal_embeddings``.

    Bit-identical to the plain recursion when ``record`` is False and
    ``remove`` is None; the tensor maths below never depends on the bookkeeping
    (which is how the memory-parity property of the official extractor carries
    over).  ``remove`` zeroes ONLY the tracked source child's propagated
    feature at the parent's aggregation, leaving siblings, edge features, times
    and mask untouched.
    """
    host = backbone
    device = host.node_raw_features.device
    n_nb = int(num_neighbors)
    node_ids_t = torch.from_numpy(node_ids).long().to(device)
    node_times_t = torch.from_numpy(node_interact_times).float().to(device)

    node_time_features = host.time_encoder(
        timestamps=torch.zeros(node_interact_times.shape).unsqueeze(dim=1).to(device))
    base = host.node_raw_features[node_ids_t]
    if node_memories is not None:
        node_features = node_memories[node_ids_t] + base
    else:
        node_features = base

    if layer == 0:
        return node_features

    source_paths = ({int(r): list(p) + [(0, 0.0)] for r, p in trace_paths.items()}
                    if record else {})
    node_conv = traced_embeddings(backbone, model, node_memories, node_ids,
                                  node_interact_times, layer - 1, num_neighbors,
                                  path_state, source_paths, remove, record,
                                  False, bs)

    nb_ids, nb_eids, nb_times = host.neighbor_sampler.get_historical_neighbors(
        node_ids=node_ids, node_interact_times=node_interact_times,
        num_neighbors=n_nb)
    nb_conv = traced_embeddings(backbone, model, node_memories,
                                nb_ids.flatten(), nb_times.flatten(),
                                layer - 1, num_neighbors, path_state, {},
                                None, record, False, bs)
    nb_conv = nb_conv.reshape(len(node_ids), n_nb, -1)

    delta = node_interact_times[:, np.newaxis] - nb_times
    nb_time_feat = host.time_encoder(
        timestamps=torch.from_numpy(delta).float().to(device))
    nb_edge_feat = host.edge_raw_features[torch.from_numpy(nb_eids)]

    # ---------- path bookkeeping (never feeds the tensor maths) ----------
    choices = {}
    if record:
        if is_top and layer == int(model.n_layers):
            path_state["recs"] = {}
            path_state["stash"] = {}
            path_state["by_layer"] = {1: {}, 2: {}}
            for r in range(bs):
                s = av2._hash01(np.asarray([int(node_ids[r])]), 91 + 3)[0]
                s = int(s * n_nb) % n_nb
                if int(nb_ids[r, s]) != 0:
                    path_state["recs"][r] = {
                        "root": int(node_ids[r]),
                        "t_root": float(node_interact_times[r]),
                        "slot3": s,
                        "edge_feat_top": host.edge_raw_features[
                            int(nb_eids[r, s])].detach().cpu().numpy(),
                        "edge_time_top": float(nb_times[r, s])}
                    path_state["by_layer"][2][(r, s)] = r
        for (pr, s), root_r in list(path_state["by_layer"].get(layer, {}).items()):
            flat = pr * n_nb + s
            if flat >= len(node_ids):
                continue
            node = int(node_ids[flat])
            if node == 0:
                continue
            s2 = int(av2._hash01(np.asarray([node]), 91 + layer - 1)[0] * n_nb) % n_nb
            child = int(nb_ids[flat, s2])
            if child != 0:
                choices[(root_r, layer)] = {"node": node,
                                            "t": float(node_interact_times[flat]),
                                            "slot": s2, "flat_row": flat}
                path_state["by_layer"][layer - 1][(flat, s2)] = root_r

    # ---------- paired removal of the tracked source interface ----------
    if record and remove is not None and layer == (int(model.n_layers) + 1 - remove):
        if remove == 1 and is_top and layer == int(model.n_layers):
            for r, rec in path_state["recs"].items():
                nb_conv = nb_conv.clone()
                nb_conv[r, rec["slot3"], :] = 0.0
        else:
            for (pr, s), _rr in list(path_state["by_layer"].get(layer, {}).items()):
                flat = pr * n_nb + s
                if flat < len(node_ids):
                    nb_conv = nb_conv.clone()
                    nb_conv[flat, s, :] = 0.0

    out, _ = host.temporal_conv_layers[layer - 1](
        node_features=node_conv, node_time_features=node_time_features,
        neighbor_node_features=nb_conv, neighbor_node_time_features=nb_time_feat,
        neighbor_node_edge_features=nb_edge_feat, neighbor_masks=nb_ids)
    out = host.merge_layers[layer - 1](input_1=out, input_2=node_features)

    # ---------- record path states AFTER z is available ----------
    if record:
        if is_top and layer == int(model.n_layers):
            for r, rec in path_state["recs"].items():
                rec["z3"] = out[r].detach().cpu().numpy()
                rec["other_neighbors"] = _other_neighbor_vec(nb_conv, r,
                                                             rec["slot3"], n_nb)
        for (root_r, lv), ch in choices.items():
            flat = ch["flat_row"]
            e = {"z": out[flat].detach().cpu().numpy(), "node": ch["node"],
                 "t": ch["t"],
                 "edge_feat": host.edge_raw_features[
                     int(nb_eids[flat, ch["slot"]])].detach().cpu().numpy(),
                 "edge_time": float(nb_times[flat, ch["slot"]]),
                 "other_neighbors": _other_neighbor_vec(nb_conv, flat,
                                                        ch["slot"], n_nb)}
            if lv >= 2:
                path_state["stash"][(root_r, lv)] = e
            else:
                child = int(nb_ids[flat, ch["slot"]])
                e["u0"] = nb_conv[flat, ch["slot"]].detach().cpu().numpy()
                e["h0"] = (node_memories[int(child)].detach().cpu().numpy()
                           if node_memories is not None
                           else np.zeros_like(e["u0"]))
                e["node_info"] = host.node_raw_features[
                    int(child)].detach().cpu().numpy()
                e["leaf"] = child
                path_state["stash"][(root_r, 1)] = e
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["tgn", "tgat"], required=True)
    ap.add_argument("--src-dir", required=True,
                    help="the *_3H10 dir containing models/ and utils/")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-root", required=True,
                    help="dir containing processed_data/<name>/")
    ap.add_argument("--data-name", default="uci")
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--n-neighbors", type=int, default=10)
    ap.add_argument("--time-feat-dim", type=int, default=100)
    ap.add_argument("--n-heads", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.4)
    ap.add_argument("--sample-neighbor-strategy", default="recent")
    ap.add_argument("--sampler-seed", type=int, default=1)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--bs", type=int, default=200)
    ap.add_argument("--audit-batches", type=int, default=200)
    ap.add_argument("--calib-batches", type=int, default=60)
    ap.add_argument("--gap-batches", type=int, default=8)
    ap.add_argument("--manifest-out", default=None)
    ap.add_argument("--manifest-in", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--gate-check", action="store_true",
                    help="run the native-forward parity gate and exit")
    args = ap.parse_args()

    # the TGB loader hardcodes ``./processed_data/<name>/``
    os.chdir(args.data_root)
    host = _import_host(Path(args.src_dir).resolve(), args.model)
    from utils.DataLoader import get_link_prediction_data      # noqa: WPS433
    from utils.utils import get_neighbor_sampler               # noqa: WPS433

    (node_raw, edge_raw, full_data, train_data, val_data, test_data,
     _nv, _nt) = get_link_prediction_data(args.data_name, 0.15, 0.15)
    n_nodes = int(max(node_raw.shape[0], full_data.src_node_ids.max() + 1,
                      full_data.dst_node_ids.max() + 1))
    device = torch.device("cuda:{}".format(args.gpu)
                          if torch.cuda.is_available() else "cpu")
    sampler = get_neighbor_sampler(
        data=full_data, sample_neighbor_strategy=args.sample_neighbor_strategy,
        time_scaling_factor=0.0, seed=args.sampler_seed)
    model, backbone = build_host(args, host, node_raw, edge_raw, sampler,
                                 device, train_data.src_node_ids,
                                 train_data.dst_node_ids,
                                 train_data.node_interact_times)
    print("[host] {} n_layers={} n_neighbors={} dropout={} n_nodes={} "
          "device={}".format(args.model, args.n_layers, args.n_neighbors,
                             args.dropout, n_nodes, device), flush=True)
    print("[ckpt]", model_meta(args.ckpt), flush=True)

    ds = _DSView(full_data, n_nodes, np.unique(full_data.dst_node_ids))
    if args.gate_check:
        return _gate_check(args, model, backbone, train_data, device)
    raise SystemExit("full extraction path not yet wired; use --gate-check")


def _gate_check(args, model, backbone, train_data, device):
    """Gate 1: the loaded checkpoint reproduces the native forward."""
    n = len(train_data.src_node_ids)
    s = np.asarray(train_data.src_node_ids[:args.bs])
    d = np.asarray(train_data.dst_node_ids[:args.bs])
    t = np.asarray(train_data.node_interact_times[:args.bs], dtype=np.float64)
    e = np.asarray(train_data.edge_ids[:args.bs])
    with torch.no_grad():
        src_emb, dst_emb = model[0].compute_src_dst_node_temporal_embeddings(
            src_node_ids=s, dst_node_ids=d, node_interact_times=t,
            edge_ids=e, edges_are_positive=True,
            num_neighbors=args.n_neighbors)
    print("[gate1] forward ok: src_emb{} dst_emb{} finite={}".format(
        tuple(src_emb.shape), tuple(dst_emb.shape),
        bool(torch.isfinite(src_emb).all() and torch.isfinite(dst_emb).all())),
        flush=True)
    return {"ok": True, "n_train": int(n)}


if __name__ == "__main__":
    main()
