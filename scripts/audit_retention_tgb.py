#!/usr/bin/env python3
"""TGB baseline host adapter for the retention audit -- alignment gates first.

Scope of this file: load the TGB ``MemoryModel`` (TGN) / ``TGAT`` checkpoints
exactly as the native evaluation does, and prove the traced forward is
bit-identical to the native one.  Row extraction is deliberately NOT wired yet;
the review requires native-forward alignment before any state extraction.

What is replaced, and what is not
---------------------------------
The ONLY thing swapped out is the backbone's ``compute_node_temporal_embeddings``,
by a same-signature traced function returning a bit-identical tensor.  The
model's own memory update, neighbour sampler and time recursion all run exactly
as in the native evaluation, so each trained baseline keeps its own sampling,
temporal recursion and memory semantics -- which is what comparing trained
native models requires.

Gates implemented here
----------------------
  --gate-load    checkpoint loads with strict=True into the rebuilt backbone
  --gate-native  the native validation pass reproduces the published metric
                 (AP / AUC / MRR) for that checkpoint
  --gate-trace   the traced forward returns exactly the native embeddings, and a
                 no-op remove (Delta = 0) leaves them unchanged

Usage:
    PYTHONPATH=<repo>/src:<repo>/scripts python scripts/audit_retention_tgb.py \
        --model tgn --src-dir <.../TGN_3H10> --ckpt <TGN_seed0.pkl> \
        --data-root <dir with processed_data/uci> --gate all
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

from audit_retention_v2 import _hash01, _other_neighbor_vec, model_meta  # noqa: E402


class RunnerError(RuntimeError):
    pass


def _import_host(src_dir, model):
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    if model == "tgat":
        from models.TGAT import TGAT                        # noqa: WPS433
        return {"TGAT": TGAT}
    from models.MemoryModel import (                        # noqa: WPS433
        MemoryModel, compute_src_dst_node_time_shifts)
    return {"MemoryModel": MemoryModel,
            "compute_src_dst_node_time_shifts": compute_src_dst_node_time_shifts}


def build_model(args, host, node_raw, edge_raw, sampler, device, train_data):
    """Rebuild ``nn.Sequential(backbone, MergeLayer)`` and load the state dict."""
    from models.modules import MergeLayer                   # noqa: WPS433
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
            train_data.src_node_ids, train_data.dst_node_ids,
            train_data.node_interact_times)
        backbone = host["MemoryModel"](
            node_raw_features=node_raw, edge_raw_features=edge_raw,
            neighbor_sampler=sampler, time_feat_dim=args.time_feat_dim,
            model_name="TGN", num_layers=args.n_layers,
            num_heads=args.n_heads, dropout=args.dropout,
            src_node_mean_time_shift=shifts[0], src_node_std_time_shift=shifts[1],
            dst_node_mean_time_shift_dst=shifts[2],
            dst_node_std_time_shift=shifts[3], device=device)
    model = torch.nn.Sequential(backbone, MergeLayer(d, d, d, 1))
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = ck["state_dict"] if (isinstance(ck, dict) and "state_dict" in ck) else ck
    model.load_state_dict(state, strict=True)          # gate-load
    return model.to(device).eval(), model[0]


class StateSnapshot:
    """Everything a remove variant must restore before it runs.

    TGN defers its memory update by one batch, so the keep/remove pair has to
    start from the same ``node_memories``, ``node_last_updated_times`` and
    ``node_raw_messages`` -- and the neighbour sampler's RNG state, so both
    variants see the SAME sampled neighbours.
    """

    def __init__(self, backbone, sampler):
        self.backbone, self.sampler = backbone, sampler

    def backup(self):
        mb = getattr(self.backbone, "memory_bank", None)
        mem = None
        if mb is not None:
            mem = (mb.node_memories.clone(), mb.node_last_updated_times.clone(),
                   {k: [(m[0].clone(), m[1]) for m in v]
                    for k, v in mb.node_raw_messages.items()})
        rng = getattr(self.sampler, "random_state", None)
        return {"mem": mem,
                "rng": (rng.get_state() if rng is not None else None),
                "np": np.random.get_state()}

    def restore(self, bak):
        mb = getattr(self.backbone, "memory_bank", None)
        if mb is not None and bak["mem"] is not None:
            mem, upd, msgs = bak["mem"]
            mb.node_memories.copy_(mem)
            mb.node_last_updated_times.copy_(upd)
            mb.node_raw_messages.clear()
            for k, v in msgs.items():
                mb.node_raw_messages[k] = [(m[0].clone(), m[1]) for m in v]
        rng = getattr(self.sampler, "random_state", None)
        if rng is not None and bak["rng"] is not None:
            rng.set_state(bak["rng"])
        np.random.set_state(bak["np"])






class Tracer:
    """Same-signature replacement for ``compute_node_temporal_embeddings``.

    Bit-identical to the host recursion when ``record`` is False and ``remove``
    is None.  Path bookkeeping is registered BEFORE the neighbour recursion --
    registering it afterwards means the child call at layer-1 sees no tracked
    rows, so the leaf->a2->a1 chain is never followed.
    """

    def __init__(self, backbone, num_layers, n_neighbors):
        self.backbone = backbone
        self.num_layers = int(num_layers)
        self.n_neighbors = int(n_neighbors)
        self.reset(False, None, 0)

    def reset(self, record, remove, bs):
        self.path_state = {"recs": {}, "stash": {}, "by_layer": {1: {}, 2: {}}}
        self.record, self.remove, self.bs = bool(record), remove, int(bs)

    def __call__(self, node_memories=None, node_ids=None,
                 node_interact_times=None, current_layer_num=None,
                 num_neighbors=None, **kw):
        if node_ids is None or node_interact_times is None:
            raise RunnerError("traced call missing node_ids/interact_times")
        node_ids = np.asarray(node_ids)
        # The root call is the top-layer call whose rows are exactly the batch:
        # TGN concatenates [src ; dst] (2*bs rows) and calls once, TGAT calls
        # once per side (bs rows).  Any deeper call sees bs*n_neighbors rows.
        is_top = (int(current_layer_num) == self.num_layers
                  and len(node_ids) in (self.bs, 2 * self.bs))
        return self._rec(node_memories, node_ids,
                         np.asarray(node_interact_times),
                         int(current_layer_num), int(num_neighbors), {}, is_top)

    def _rec(self, node_memories, node_ids, node_interact_times, layer,
             n_nb, trace_paths, is_top):
        host = self.backbone
        device = host.node_raw_features.device
        node_ids_t = torch.from_numpy(node_ids).long().to(device)
        node_time_features = host.time_encoder(timestamps=torch.zeros(
            node_interact_times.shape).unsqueeze(1).to(device))
        base = host.node_raw_features[node_ids_t]
        node_features = (node_memories[node_ids_t] + base
                         if node_memories is not None else base)
        if layer == 0:
            return node_features

        source_paths = ({int(r): list(p) + [(0, 0.0)]
                         for r, p in trace_paths.items()} if self.record else {})
        node_conv = self._rec(node_memories, node_ids, node_interact_times,
                              layer - 1, n_nb, source_paths, False)
        nb_ids, nb_eids, nb_times = host.neighbor_sampler.get_historical_neighbors(
            node_ids=node_ids, node_interact_times=node_interact_times,
            num_neighbors=n_nb)

        choices = {}
        if self.record:
            if is_top and layer == self.num_layers:
                for r in range(min(self.bs, len(node_ids))):
                    s = int(_hash01(np.asarray([int(node_ids[r])]), 94)[0]
                            * n_nb) % n_nb
                    if int(nb_ids[r, s]) != 0:
                        self.path_state["recs"][r] = {
                            "root": int(node_ids[r]),
                            "t_root": float(node_interact_times[r]), "slot3": s,
                            "edge_feat_top": host.edge_raw_features[
                                int(nb_eids[r, s])].detach().cpu().numpy(),
                            "edge_time_top": float(nb_times[r, s])}
                        self.path_state["by_layer"][2][(r, s)] = r
            for (pr, s), root_r in list(
                    self.path_state["by_layer"].get(layer, {}).items()):
                flat = pr * n_nb + s
                if flat >= len(node_ids):
                    continue
                node = int(node_ids[flat])
                if node == 0:
                    continue
                s2 = int(_hash01(np.asarray([node]), 90 + layer)[0] * n_nb) % n_nb
                if int(nb_ids[flat, s2]) != 0:
                    choices[(root_r, layer)] = {
                        "node": node, "t": float(node_interact_times[flat]),
                        "slot": s2, "flat_row": flat}
                    # only descend while there is a level below to register;
                    # at layer 1 the leaf is captured through its u0 state
                    if layer >= 2:
                        self.path_state["by_layer"][layer - 1][(flat, s2)] = root_r

        nb_conv = self._rec(node_memories, nb_ids.flatten(), nb_times.flatten(),
                            layer - 1, n_nb, {}, False)
        nb_conv = nb_conv.reshape(len(node_ids), n_nb, -1)
        delta = node_interact_times[:, np.newaxis] - nb_times
        nb_time_feat = host.time_encoder(
            timestamps=torch.from_numpy(delta).float().to(device))
        nb_edge_feat = host.edge_raw_features[torch.from_numpy(nb_eids)]

        if self.record and self.remove is not None \
                and layer == (self.num_layers + 1 - self.remove):
            if self.remove == 1 and is_top and layer == self.num_layers:
                for r, rec in self.path_state["recs"].items():
                    nb_conv = nb_conv.clone()
                    nb_conv[r, rec["slot3"], :] = 0.0
            else:
                for (pr, s), _rr in list(
                        self.path_state["by_layer"].get(layer, {}).items()):
                    flat = pr * n_nb + s
                    if flat < len(node_ids):
                        nb_conv = nb_conv.clone()
                        nb_conv[flat, s, :] = 0.0

        out, _ = host.temporal_conv_layers[layer - 1](
            node_features=node_conv, node_time_features=node_time_features,
            neighbor_node_features=nb_conv,
            neighbor_node_time_features=nb_time_feat,
            neighbor_node_edge_features=nb_edge_feat, neighbor_masks=nb_ids)
        out = host.merge_layers[layer - 1](input_1=out, input_2=node_features)

        if self.record:
            if is_top and layer == self.num_layers:
                for r, rec in self.path_state["recs"].items():
                    rec["z3"] = out[r].detach().cpu().numpy()
                    rec["other_neighbors"] = _other_neighbor_vec(
                        nb_conv, r, rec["slot3"], n_nb)
            for (root_r, lv), ch in choices.items():
                flat = ch["flat_row"]
                e = {"z": out[flat].detach().cpu().numpy(), "node": ch["node"],
                     "t": ch["t"],
                     "edge_feat": host.edge_raw_features[
                         int(nb_eids[flat, ch["slot"]])].detach().cpu().numpy(),
                     "edge_time": float(nb_times[flat, ch["slot"]]),
                     "other_neighbors": _other_neighbor_vec(
                         nb_conv, flat, ch["slot"], n_nb)}
                if lv >= 2:
                    self.path_state["stash"][(root_r, lv)] = e
                else:
                    child = int(nb_ids[flat, ch["slot"]])
                    e["u0"] = nb_conv[flat, ch["slot"]].detach().cpu().numpy()
                    e["leaf"] = child
                    self.path_state["stash"][(root_r, 1)] = e
        return out




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["tgn", "tgat"], required=True)
    ap.add_argument("--src-dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--data-name", default="uci")
    ap.add_argument("--train-frac", type=float, default=0.70,
                    help="train split as a quantile of the full stream, to match "
                         "UCILinkDataset (the official extractor's loader)")
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--n-neighbors", type=int, default=10)
    ap.add_argument("--time-feat-dim", type=int, default=100)
    ap.add_argument("--n-heads", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.4)
    ap.add_argument("--sample-neighbor-strategy", default="recent")
    ap.add_argument("--sampler-seed", type=int, default=1)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--bs", type=int, default=200)
    ap.add_argument("--audit-batches", type=int, default=60)
    ap.add_argument("--calib-batches", type=int, default=80)
    ap.add_argument("--gap-batches", type=int, default=8)
    ap.add_argument("--transfer-batches", type=int, default=None)
    ap.add_argument("--manifest-out", default=None)
    ap.add_argument("--manifest-in", default=None)
    ap.add_argument("--neg-collision", default="resample",
                    choices=["resample", "flip"])
    ap.add_argument("--arm", default=None, help="override the meta arm name")
    ap.add_argument("--seed-id", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--reset-memory", action="store_true", default=True,
                    help="zero the checkpoint memory before the stream replay "
                         "(the stored memory is the end-of-training state and "
                         "cannot be replayed backwards)")
    ap.add_argument("--audit", action="store_true",
                    help="run the full rows extraction (not just the gates)")
    ap.add_argument("--zroot-only", action="store_true",
                    help="only record the native root state per raw csv row, for "
                         "the canonical model-independent cross-model targets")
    ap.add_argument("--gate", default="all",
                    choices=["load", "native", "trace", "all"])
    args = ap.parse_args()

    os.chdir(args.data_root)
    host = _import_host(Path(args.src_dir).resolve(), args.model)
    from utils.DataLoader import get_link_prediction_data      # noqa: WPS433
    from utils.utils import get_neighbor_sampler               # noqa: WPS433
    from utils.DataLoader import get_idx_data_loader           # noqa: WPS433

    (node_raw, edge_raw, full_data, train_data, val_data, test_data,
     _nv, _nt) = get_link_prediction_data(args.data_name, 0.15, 0.15)

    # Put the TGB arms on the SAME root events as the RPBE arms.  UCILinkDataset
    # (the official extractor's loader) defines train as
    # ``timestamps <= quantile(full, 0.70)``; the TGB loader uses its own ratio
    # split, which yields a different event set and makes the shared manifest
    # unusable.  Both read the same csv in the same order, so re-deriving the
    # mask here reproduces the official split exactly.
    _ft = np.asarray(full_data.node_interact_times, np.float64)
    _cut = float(np.quantile(_ft, args.train_frac))
    _mask = _ft <= _cut

    class _Split:
        def __init__(self, fd, m):
            self.src_node_ids = fd.src_node_ids[m]
            self.dst_node_ids = fd.dst_node_ids[m]
            self.node_interact_times = fd.node_interact_times[m]
            self.edge_ids = fd.edge_ids[m]

    train_data = _Split(full_data, _mask)
    print("[split] train = {} events (t <= {:.0f}, the {:.0f}th percentile of "
          "the full stream) -- matching UCILinkDataset".format(
              int(_mask.sum()), _cut, args.train_frac * 100), flush=True)
    n_nodes = int(max(node_raw.shape[0], full_data.src_node_ids.max() + 1,
                      full_data.dst_node_ids.max() + 1))
    device = torch.device("cuda:{}".format(args.gpu)
                          if torch.cuda.is_available() else "cpu")
    sampler = get_neighbor_sampler(
        data=full_data, sample_neighbor_strategy=args.sample_neighbor_strategy,
        time_scaling_factor=0.0, seed=args.sampler_seed)
    model, backbone = build_model(args, host, node_raw, edge_raw, sampler,
                                  device, train_data)
    print("[gate-load] OK  keys={} strict=True  {}".format(
        len(model.state_dict()), model_meta(args.ckpt)), flush=True)
    if args.gate == "load":
        return

    ns = int(args.n_neighbors)
    bs = 200
    # the object whose compute_node_temporal_embeddings is replaced, i.e. the
    # owner of time_encoder / edge_raw_features / neighbor_sampler / conv layers
    target = (backbone.embedding_module if args.model == "tgn" else backbone)

    def memory_tensor():
        mb = getattr(backbone, "memory_bank", None)
        if mb is None:
            return None
        return mb.get_memories(np.arange(backbone.num_nodes))

    def call_embedding(mem, ids, times, tracer):
        """Call the recursion DIRECTLY, so no memory update (and no native
        monotonicity assert) is involved -- this isolates exactly what the
        adapter replaces."""
        orig = target.compute_node_temporal_embeddings
        if tracer is not None:
            target.compute_node_temporal_embeddings = tracer
        try:
            with torch.no_grad():
                if args.model == "tgn":
                    out = target.compute_node_temporal_embeddings(
                        node_memories=mem, node_ids=ids,
                        node_interact_times=times,
                        current_layer_num=args.n_layers, num_neighbors=ns)
                else:
                    out = target.compute_node_temporal_embeddings(
                        node_ids=ids, node_interact_times=times,
                        current_layer_num=args.n_layers, num_neighbors=ns)
        finally:
            target.compute_node_temporal_embeddings = orig
        return out.detach().cpu().numpy()

    # ---- gate-trace: traced == native, and a no-op remove is the identity ----
    # feed the concatenated [src ; dst] root rows exactly as the real call does,
    # so the path bookkeeping is exercised too
    mem = memory_tensor()
    _s = np.asarray(train_data.src_node_ids[:bs]).astype(np.int64)
    _d = np.asarray(train_data.dst_node_ids[:bs]).astype(np.int64)
    _t = np.asarray(train_data.node_interact_times[:bs], np.float64)
    ids = np.concatenate([_s, _d])
    times = np.concatenate([_t, _t])
    nat = call_embedding(mem, ids, times, None)
    tr = Tracer(target, args.n_layers, ns)
    tr.reset(True, None, bs)
    trc = call_embedding(mem, ids, times, tr)
    same = np.array_equal(nat, trc)
    ps = tr.path_state
    print("[gate-trace] traced == native embeddings (exact): {}  "
          "max|d|={:.3g}  roots={} stash={}".format(
              same, float(np.abs(nat - trc).max()), len(ps["recs"]),
              sorted({k[1] for k in ps["stash"]})), flush=True)
    tr2 = Tracer(target, args.n_layers, ns)
    tr2.reset(True, 0, bs)          # remove=0 never fires -> no-op
    noop = call_embedding(mem, ids, times, tr2)
    print("[gate-trace] no-op remove leaves embeddings unchanged (exact): {}"
          .format(np.array_equal(nat, noop)), flush=True)
    if args.gate == "trace":
        return

    # ---- gate-native: reproduce the published metric ---------------------
    # The native evaluation deliberately SKIPS validation for memory models --
    # the stored memory has already been advanced past those times, so replaying
    # them trips the "update memory to time in the past" assert.  Reproducing the
    # published number faithfully therefore means running the host's own
    # evaluate_link_prediction.py with its own negative sampler, which is a
    # separate subprocess step rather than something this harness re-implements.
    print("[gate-native] SKIPPED here by design: run the host's own "
          "evaluate_link_prediction.py (it skips val for memory models and "
          "uses its own NegativeEdgeSampler) to reproduce the published "
          "AP/AUC.", flush=True)

    if not args.audit and not args.zroot_only:
        return

    # the gate section restores the original after each call, so the tracer is
    # installed for the whole extraction here (both modes need it)
    target.compute_node_temporal_embeddings = tr
    _meta = model_meta(args.ckpt)
    if args.arm:
        _meta["arm"] = args.arm
    if args.seed_id is not None:
        _meta["seed"] = int(args.seed_id)
    _meta["model_kind"] = args.model

    # ============================ Z_root extraction ==========================
    # The cross-model comparison (canonical targets) needs ONLY each model's
    # natively-formed root state per canonical root event -- not the model's
    # own sampled path.  This mode runs the plain keep forward over the audit
    # window and records z3 against the RAW csv row index of the root event.
    if args.zroot_only:
        src_all = np.asarray(train_data.src_node_ids)
        dst_all = np.asarray(train_data.dst_node_ids)
        t_all = np.asarray(train_data.node_interact_times)
        e_all = np.asarray(train_data.edge_ids)
        bs = int(args.bs)
        n_batches = len(src_all) // bs
        lo = max(0, n_batches - int(args.audit_batches))
        hi = n_batches
        mb0 = getattr(backbone, "memory_bank", None)
        if mb0 is not None:
            mb0.node_memories.zero_()
            mb0.node_last_updated_times.fill_(0)
            mb0.node_raw_messages.clear()
        zroot, times = {}, {}
        for bb in range(0, hi):
            s0 = bb * bs
            tr.reset(True, None, bs)
            with torch.no_grad():
                if args.model == "tgn":
                    backbone.compute_src_dst_node_temporal_embeddings(
                        src_node_ids=src_all[s0:s0 + bs],
                        dst_node_ids=dst_all[s0:s0 + bs],
                        node_interact_times=t_all[s0:s0 + bs].astype(np.float64),
                        edge_ids=e_all[s0:s0 + bs], edges_are_positive=True,
                        num_neighbors=ns)
                else:
                    backbone.compute_src_dst_node_temporal_embeddings(
                        src_node_ids=src_all[s0:s0 + bs],
                        dst_node_ids=dst_all[s0:s0 + bs],
                        node_interact_times=t_all[s0:s0 + bs].astype(np.float64),
                        num_neighbors=ns)
            if bb >= lo:
                for r, rec in tr.path_state["recs"].items():
                    raw = s0 + r
                    zroot[int(raw)] = np.asarray(rec["z3"], np.float32)
                    times[int(raw)] = float(rec["t_root"])
            if (bb + 1) % 40 == 0:
                print("[zroot] batch {}/{} collected={}".format(
                    bb + 1, hi, len(zroot)), flush=True)
        out_path = args.out or str(Path(args.ckpt).parent /
                                   "zroot_{}_seed0.pkl".format(args.model))
        with open(out_path, "wb") as f:
            pickle.dump({"meta": {"model_kind": args.model,
                                  "arm": _meta.get("arm"),
                                  "seed": _meta.get("seed"),
                                  "ckpt_sha256": _meta.get("ckpt_sha256"),
                                  "n_layers": args.n_layers,
                                  "n_neighbors": args.n_neighbors, "bs": bs,
                                  "audit_block": [lo, hi],
                                  "raw_row_convention":
                                      "full-csv row index (== csv idx column)"},
                         "zroot": zroot, "t_root": times}, f)
        print("[zroot] saved {} root states to {}".format(len(zroot), out_path),
              flush=True)
        return

    # ============================ rows extraction ============================
    from audit_retention_v2 import _cand_hash, _ctx_vector      # noqa: WPS433
    from audit_retention_v2 import FutureIndex as _FutureIndex  # noqa: WPS433

    class _Full:
        def __init__(self, fd):
            self.sources = fd.src_node_ids
            self.destinations = fd.dst_node_ids
            self.timestamps = fd.node_interact_times
            self.edge_idxs = fd.edge_ids

    class _DS:
        def __init__(self, fd, n, pool):
            self.full, self.n_nodes, self.full_dst_pool = _Full(fd), int(n), pool

    ds = _DS(full_data, n_nodes, np.unique(full_data.dst_node_ids))
    fut = _FutureIndex(ds)
    pool = ds.full_dst_pool
    _eidx_all = np.asarray(ds.full.edge_idxs, np.int64)
    if np.unique(_eidx_all).size != _eidx_all.size:
        raise RunnerError("edge_idxs is not a bijection; the edge_id -> time map "
                          "would silently overwrite entries")
    _e2t = np.full(int(_eidx_all.max()) + 1, np.nan, dtype=np.float64)
    _e2t[_eidx_all] = np.asarray(ds.full.timestamps, np.float64)

    def _ftime(eid):
        j = int(eid)
        if not (0 <= j < _e2t.size) or not np.isfinite(_e2t[j]):
            raise RunnerError(
                "future_event_id {} has no timestamp in the stream".format(j))
        return float(_e2t[j])

    manifest_in, manifest_in_sha = None, None
    if args.manifest_in:
        with open(args.manifest_in, "rb") as f:
            mj = json.load(f)
        rows_in = mj["rows"] if isinstance(mj, dict) else mj
        sha = hashlib.sha256(json.dumps(
            rows_in, sort_keys=True, default=str).encode()).hexdigest()
        if isinstance(mj, dict) and mj.get("manifest_sha") not in (None, sha):
            raise RunnerError("manifest_in SHA mismatch")
        manifest_in_sha = sha
        manifest_in = {tuple(int(x) for x in m["pair_id"]): m for m in rows_in}
        print("[manifest] loaded {} pairs (sha {})".format(
            len(manifest_in), sha[:16]), flush=True)

    src_all = np.asarray(train_data.src_node_ids)
    dst_all = np.asarray(train_data.dst_node_ids)
    t_all = np.asarray(train_data.node_interact_times)
    e_all = np.asarray(train_data.edge_ids)
    bs = int(args.bs)
    n_batches = len(src_all) // bs
    audit_hi = n_batches
    audit_lo = max(0, n_batches - int(args.audit_batches))
    gap = max(0, int(args.gap_batches))
    calib_hi = max(0, audit_lo - gap)
    calib_lo = max(0, calib_hi - int(args.calib_batches))
    transfer_hi = min(n_batches, int(args.transfer_batches
                                     if args.transfer_batches is not None
                                     else args.calib_batches))
    print("[tail] silent 0..{}, calib [{},{}), gap, audit [{},{})".format(
        calib_lo, calib_lo, calib_hi, audit_lo, audit_hi), flush=True)

    snap = StateSnapshot(backbone, sampler)
    cur = {"eidx": None}
    neg_collisions = [0]

    # The checkpoint's stored memory is the END-of-training state, so replaying
    # the stream from its beginning would run it backwards (the native assert
    # "update memory to time in the past" catches exactly that).  The audit
    # evaluates a single forward pass from the SAME initial state the native
    # training uses, so the memory is reset to zero here -- same discipline the
    # official extractor applies to the RPBE arms.
    mb0 = getattr(backbone, "memory_bank", None)
    if mb0 is not None and args.reset_memory:
        mb0.node_memories.zero_()
        mb0.node_last_updated_times.fill_(0)
        mb0.node_raw_messages.clear()
        print("[memory] reset to the training-time initial state (zeros)",
              flush=True)

    def forward(bb, record, remove):
        s0 = bb * bs
        tr.reset(record, remove, bs)
        cur["eidx"] = e_all[s0:s0 + bs]
        with torch.no_grad():
            backbone.compute_src_dst_node_temporal_embeddings(
                src_node_ids=src_all[s0:s0 + bs], dst_node_ids=dst_all[s0:s0 + bs],
                node_interact_times=t_all[s0:s0 + bs].astype(np.float64),
                edge_ids=e_all[s0:s0 + bs], edges_are_positive=True,
                num_neighbors=ns)
        if not record:
            return None
        ps = tr.path_state
        return {"recs": dict(ps["recs"]),
                "stash": {k: dict(v) for k, v in ps["stash"].items()}}

    def window_batch(bb):
        pre = snap.backup()
        keep = forward(bb, True, None)      # this IS the memory advance
        post = snap.backup()
        rm = {}
        for d_ in (3, 2, 1):
            snap.restore(pre)
            rm[d_] = forward(bb, True, d_)
            snap.restore(pre)               # every variant past the advance point
        snap.restore(post)
        return keep, rm

    manifest_rows, seen_pid = [], set()
    dbg = {"done": False}

    def rows_from(keep, rm):
        recs, stash = keep["recs"], keep["stash"]
        if os.environ.get("TGB_DEBUG") and not dbg["done"]:
            dbg["done"] = True
            print("[dbg] recs={} stash={} eidx0={} src0={}".format(
                len(recs), sorted({k[1] for k in stash}),
                int(cur["eidx"][0]) if cur["eidx"] is not None else None,
                int(src_all[0])), flush=True)
            mk = list(manifest_in)[:2] if manifest_in else []
            print("[dbg] sample manifest keys:", mk, flush=True)
            for r0 in list(recs)[:2]:
                if (r0, 1) in stash and (r0, 2) in stash:
                    print("[dbg] would-be pair_id:",
                          (int(cur["eidx"][r0]), int(recs[r0]["root"]),
                           int(stash[(r0, 1)]["leaf"]),
                           int(stash[(r0, 1)]["node"]),
                           int(stash[(r0, 2)]["node"])),
                          "in_manifest=", ((int(cur["eidx"][r0]),
                                           int(recs[r0]["root"]),
                                           int(stash[(r0, 1)]["leaf"]),
                                           int(stash[(r0, 1)]["node"]),
                                           int(stash[(r0, 2)]["node"]))
                                          in (manifest_in or {})), flush=True)
        out = []
        for r, rec in recs.items():
            if (r, 1) not in stash or (r, 2) not in stash:
                continue
            st1, st2 = stash[(r, 1)], stash[(r, 2)]
            root = int(rec["root"])
            ctx = _ctx_vector(root, st1["leaf"],
                              [st1["t"], st2["t"], rec["t_root"]],
                              [st1["edge_feat"], st2["edge_feat"],
                               rec["edge_feat_top"]],
                              [st1["edge_time"], st2["edge_time"],
                               rec["edge_time_top"]],
                              [st1["other_neighbors"], st2["other_neighbors"],
                               rec["other_neighbors"]])
            leaf, a2, a1 = int(st1["leaf"]), int(st1["node"]), int(st2["node"])
            root_t = float(rec["t_root"])
            pair_id = (int(cur["eidx"][r]), root, leaf, a2, a1)
            nodes = {"leaf": leaf, "a2": a2, "a1": a1, "root": root}
            F, ok = {}, True
            if manifest_in is not None:
                mrow = manifest_in.get(pair_id)
                if mrow is None:
                    continue
                for k, v in nodes.items():
                    if int(mrow["nodes"][k]) != int(v) or \
                            float(mrow["t_root"]) != root_t:
                        raise RunnerError(
                            "manifest_in mismatch on pair {}".format(pair_id))
                    pres, dpos, dneg = (int(mrow["presented"][k]),
                                        int(mrow["pos_cand"][k]),
                                        int(mrow["neg_cand"][k]))
                    F[k] = {"eid": int(mrow["pos_future_event_id"][k]),
                            "dpos": dpos, "dneg": dneg, "presented": pres,
                            "cand_seed": int(mrow["candidate_seed"][k]),
                            "Y": 1 if pres == dpos else 0}
            else:
                for k, v in nodes.items():
                    q = fut.query_src(v, root_t)
                    if q is None:
                        ok = False
                        break
                    jq, _dt, dpos = q
                    cseed = (int(v) * 1000003 + int(root_t)) % (2 ** 31)
                    g = np.random.RandomState(cseed)
                    dneg = int(pool[g.randint(len(pool))])
                    collided = int(dneg == int(dpos))
                    if collided and args.neg_collision == "resample":
                        for _try in range(64):
                            d2 = int(pool[g.randint(len(pool))])
                            if d2 != int(dpos):
                                dneg = d2
                                break
                        else:
                            raise RunnerError("no non-colliding negative")
                        collided = 0
                    elif collided:
                        neg_collisions[0] += 1
                    bit = int(g.randint(2))
                    pres = int(dpos) if bit == 0 else int(dneg)
                    y = 1 if pres == int(dpos) else 0
                    if args.neg_collision == "flip" and collided:
                        y = 1 - y
                    F[k] = {"eid": int(fut.eidx[jq]), "dpos": int(dpos),
                            "dneg": int(dneg), "presented": pres,
                            "cand_seed": int(cseed), "Y": int(y)}
            if not ok:
                continue
            if pair_id not in seen_pid:
                seen_pid.add(pair_id)
                if manifest_in is not None:
                    # adopt the shared row verbatim: the arm must not silently
                    # rebuild it (it would drop fields such as future_event_time)
                    manifest_rows.append(manifest_in[pair_id])
                else:
                    manifest_rows.append({
                        "pair_id": list(pair_id),
                        "root_event_id": int(pair_id[0]),
                        "root_node": root, "t_root": root_t, "nodes": nodes,
                        "pos_cand": {k: F[k]["dpos"] for k in F},
                        "neg_cand": {k: F[k]["dneg"] for k in F},
                        "presented": {k: F[k]["presented"] for k in F},
                        "presented_bits": {k: (0 if F[k]["presented"] ==
                                               F[k]["dpos"] else 1) for k in F},
                        "pos_future_event_id": {k: F[k]["eid"] for k in F},
                        "future_event_time": {k: _ftime(F[k]["eid"])
                                              for k in F},
                        "candidate_seed": {k: F[k]["cand_seed"] for k in F},
                        "sampler_formula_version":
                            "seed=(v*1000003+root_t)%2**31;dneg=randint;bit=randint",
                        "label_kind": "balanced_sampled_candidate_existence"})

            def _dz(var, key, orig):
                if key == "z3":
                    rr = var["recs"].get(r)
                    return (orig - rr["z3"]) if (rr and "z3" in rr) else None
                ss = var["stash"].get((r, {"z1": 1, "z2": 2}[key]))
                return (orig - ss["z"]) if ss else None

            z1, z2, z3 = st1["z"], st2["z"], rec["z3"]
            d3_2, d3_1, d3_r = (_dz(rm[3], "z1", z1), _dz(rm[3], "z2", z2),
                                _dz(rm[3], "z3", z3))
            d2_1, d2_r = _dz(rm[2], "z2", z2), _dz(rm[2], "z3", z3)
            d1_r = _dz(rm[1], "z3", z3)
            if any(v is None for v in (d3_2, d3_1, d3_r, d2_1, d2_r, d1_r)):
                continue
            zk = {"z1": z1, "z2": z2, "z3": z3}
            pkey = ":".join(str(x) for x in pair_id)
            for sk, pk_, src_state, origin, pts in (
                    ("Y_leaf", "Y_a2", st1["u0"], 0,
                     [(1, d3_2, "z1"), (2, d3_1, "z2"), (3, d3_r, "z3")]),
                    ("Y_a2", "Y_a1", z1, 1, [(2, d2_1, "z2"), (3, d2_r, "z3")]),
                    ("Y_a1", "Y_root", z2, 2, [(3, d1_r, "z3")])):
                fsk, fpk = sk.replace("Y_", ""), pk_.replace("Y_", "")
                ys, yp = int(F[fsk]["Y"]), int(F[fpk]["Y"])
                hs = _cand_hash(F[fsk]["presented"], 11)
                hp = _cand_hash(F[fpk]["presented"], 22)
                common = {"line": sk, "ctx": ctx, "y_s": ys, "y_p": yp,
                          "cand_s": hs, "cand_p": hp, "pair_id": pair_id,
                          "pair_key": pkey}
                out.append(dict(common, phys=origin,
                                rem=np.zeros_like(src_state), keep=src_state))
                for phys, dk, zkey in pts:
                    out.append(dict(common, phys=phys,
                                    rem=zk[zkey] - dk, keep=zk[zkey]))
        return out

    def collect(a, b, label):
        acc = []
        for i, bb in enumerate(range(a, b)):
            keep, rm = window_batch(bb)
            acc += rows_from(keep, rm)
            if (i + 1) % 20 == 0:
                print("[{}] batch {}/{} rows={}".format(
                    label, i + 1, b - a, len(acc)), flush=True)
        return acc

    def silent(a, b, label):
        for i, bb in enumerate(range(a, b)):
            forward(bb, False, None)

    silent(0, calib_lo, "tail")
    calib_rows = collect(calib_lo, calib_hi, "calib")
    silent(calib_hi, audit_lo, "gap")
    audit_rows = collect(audit_lo, audit_hi, "audit")
    print("[tail] calib rows:", len(calib_rows), "audit rows:", len(audit_rows),
          flush=True)
    if not calib_rows or not audit_rows:
        raise RunnerError("no rows extracted")

    # manifest de-duplication, mirroring the official extractor
    by_pid, dedup = {}, []
    for m in manifest_rows:
        k = tuple(int(x) for x in m["pair_id"])
        if k in by_pid:
            if m != by_pid[k]:
                raise RunnerError("manifest pair_id {} has two different "
                                  "rows".format(k))
            continue
        by_pid[k] = m
        dedup.append(m)
    if len(dedup) != len(manifest_rows):
        print("[manifest] de-duplicated {} repeated pair_id(s) ({} -> {})"
              .format(len(manifest_rows) - len(dedup), len(manifest_rows),
                      len(dedup)), flush=True)
    manifest_rows = dedup
    manifest_sha = hashlib.sha256(json.dumps(
        manifest_rows, sort_keys=True, default=str).encode()).hexdigest()
    if args.manifest_out:
        with open(args.manifest_out, "w") as f:
            json.dump({"manifest_sha": manifest_sha,
                       "neg_collisions": int(neg_collisions[0]),
                       "sampler_formula_version":
                           "seed=(v*1000003+root_t)%2**31",
                       "rows": manifest_rows}, f, default=str)
        print("[manifest] wrote {} pairs (sha {}) to {}".format(
            len(manifest_rows), manifest_sha[:16], args.manifest_out), flush=True)
    if manifest_in is not None:
        # the arm must AGREE with the shared manifest on every pair it used;
        # the shared file may legitimately cover more pairs (it also carries the
        # optional head block, which this pass does not produce)
        bad = [k for k in by_pid if k in manifest_in
               and by_pid[k] != manifest_in[k]]
        if bad:
            raise RunnerError(
                "{} regenerated pairs disagree with the shared manifest, e.g. {}"
                .format(len(bad), bad[:3]))
        print("[manifest] verified {} pairs against the shared manifest "
              "({} more in the shared file)".format(
                  len(by_pid), len(manifest_in) - len(by_pid)), flush=True)

    meta.update({
        "n_layers": args.n_layers, "n_neighbors": args.n_neighbors,
        "bs": bs, "data_name": args.data_name, "manifest_sha": manifest_sha,
        "neg_collisions": int(neg_collisions[0]),
        "manifest_in": args.manifest_in,
        "layout": {"audit_block": [audit_lo, audit_hi],
                   "same_tail_calib_block": [calib_lo, calib_hi],
                   "head_calib_block": [0, transfer_hi]}})
    out_path = args.out or str(Path(args.ckpt).parent / "retention_rows.pkl")
    with open(out_path, "wb") as f:
        pickle.dump({"audit": audit_rows, "calib": calib_rows, "head": [],
                     "meta": meta, "manifest": manifest_rows}, f)
    print("[dump] rows saved to", out_path, flush=True)


if __name__ == "__main__":
    main()
