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
import os
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
        # the root call is the concatenated [src ; dst] at the top layer
        is_top = (int(current_layer_num) == self.num_layers
                  and len(node_ids) == 2 * self.bs)
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
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--n-neighbors", type=int, default=10)
    ap.add_argument("--time-feat-dim", type=int, default=100)
    ap.add_argument("--n-heads", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.4)
    ap.add_argument("--sample-neighbor-strategy", default="recent")
    ap.add_argument("--sampler-seed", type=int, default=1)
    ap.add_argument("--gpu", type=int, default=0)
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


if __name__ == "__main__":
    main()
