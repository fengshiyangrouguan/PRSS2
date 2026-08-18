#!/usr/bin/env python3
"""Official-TGN-mother dynamic node classification with one PRSS switch.

Upstream mother: ``official_tgn/train_supervised.py`` (preserved byte-for-byte separately).

This derivative intentionally keeps the upstream task path intact:
  * ``get_data_node_classification`` and chronological TGN data;
  * the upstream ``TGN`` constructor and pretrained self-supervised checkpoint;
  * the full upstream call ``compute_temporal_embeddings(src,dst,dst,...)``;
  * the upstream ``utils.utils.MLP`` node decoder;
  * memory reset at epoch start and chronological train -> validation -> test replay.

The single scientific switch is ``--mode vanilla|prss``.  In PRSS mode only, the official
recursive embedding module is wrapped so that each recursive occurrence forms a rich candidate
from the exact host aggregate inputs, projects it with the shared per-layer predictive quotient,
and returns only the host-width quotient to its parent.  Training-only outside contexts generate
future-reading matrices B(C); same-layer B(C) rows form one conceptual operator bank whose right
singular subspace is solved through ``eigh(mean B^T B)``.  There is no PCA or alternate reduction
path in this runtime.

The exact upstream script is still run separately as a provenance/task anchor.  This derivative
adds a proper validation holdout and held-out test replay identically to vanilla and PRSS so that
model selection is fair.  ``--finetune-host`` is also applied identically in the matched pair.
"""
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = Path(__file__).resolve().parent
METHOD_DIR = HERE.parent
TGN_DIR = Path(os.environ.get("TGN_DIR", METHOD_DIR / "official_tgn" / "source")).resolve()
if str(TGN_DIR) not in sys.path:
    sys.path.insert(0, str(TGN_DIR))
if str(METHOD_DIR) not in sys.path:
    sys.path.insert(0, str(METHOD_DIR))

from model.tgn import TGN
from utils.utils import get_neighbor_finder, MLP
from utils.data_processing import compute_time_statistics, get_data_node_classification
from prss import PRSSCore, PRSSTGNEmbeddingAdapter
from prss.auxiliary import build_auxiliary
from prss.monitoring import MonitorWriter, candidate_stats, grad_l2, matrix_stats, module_finiteness


def parse_args():
    p = argparse.ArgumentParser("Official TGN node-classification mother + PRSS switch")
    p.add_argument("--mode", choices=["vanilla", "prss"], required=True)
    p.add_argument("-d", "--data", default="wikipedia")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--pretrained-checkpoint", required=True)
    p.add_argument("--output", required=True)

    # Upstream TGN names/defaults, only hyphenated for this derivative CLI.
    p.add_argument("--bs", type=int, default=100)
    p.add_argument("--n-degree", type=int, default=10)
    p.add_argument("--n-head", type=int, default=2)
    p.add_argument("--n-epoch", type=int, default=10)
    p.add_argument("--n-layer", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--drop-out", type=float, default=0.1)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--use-memory", action="store_true")
    p.add_argument("--embedding-module", default="graph_attention", choices=["graph_attention", "graph_sum"])
    p.add_argument("--message-function", default="identity", choices=["mlp", "identity"])
    p.add_argument("--aggregator", default="last")
    p.add_argument("--memory-update-at-end", action="store_true")
    p.add_argument("--message-dim", type=int, default=100)
    p.add_argument("--memory-dim", type=int, default=172)
    p.add_argument("--uniform", action="store_true")
    p.add_argument("--use-destination-embedding-in-message", action="store_true")
    p.add_argument("--use-source-embedding-in-message", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--selection-metric", choices=["auc", "ap"], default="auc")
    p.add_argument("--finetune-host", action="store_true",
                   help="Fine-tune the same pretrained TGN host in both matched modes")

    # PRSS-only.  There is deliberately no alternative reduction selector.
    p.add_argument("--candidate-dim", type=int, default=256)
    p.add_argument("--candidate-hidden", type=int, default=128)
    p.add_argument("--context-dim", type=int, default=64)
    p.add_argument("--reader-hidden", type=int, default=128)
    p.add_argument("--lambda-response", type=float, default=1.0)
    p.add_argument("--lambda-spectral", type=float, default=0.1)
    p.add_argument("--gram-ema", type=float, default=0.05)
    p.add_argument("--spectral-warmup", type=int, default=200)
    p.add_argument("--spectral-interval", type=int, default=200)
    p.add_argument("--spectral-step-size", type=float, default=0.25,
                   help="Trust-region step toward the exact SVD/eigenspace target")
    p.add_argument("--trace-roots", type=int, default=8)

    # Monitoring only; never changes the scientific objective.
    p.add_argument("--monitor-every", type=int, default=50)
    p.add_argument("--no-fail-on-monitor-error", action="store_true")
    p.add_argument("--response-gap-warn", type=float, default=0.25)
    p.add_argument("--grad-clip", type=float, default=5.0,
                   help="Clip matched main-path gradient norm; 0 disables")
    p.add_argument("--unrestricted-grad-clip", type=float, default=5.0,
                   help="Clip monitoring-only unrestricted reader gradient norm; 0 disables")
    p.add_argument("--checkpoint-every", type=int, default=50,
                   help="Overwrite a rolling exact mid-epoch checkpoint every N train batches; 0 disables")
    p.add_argument("--resume-from", default="",
                   help="Resume an exact mid-epoch checkpoint produced by this script")

    # Smoke-only caps. Full experiment uses 0.
    p.add_argument("--max-train", type=int, default=0)
    p.add_argument("--max-val", type=int, default=0)
    p.add_argument("--max-test", type=int, default=0)
    return p.parse_args()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_torch(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def unwrap_state(obj):
    if isinstance(obj, dict) and "model_state_dict" in obj:
        return obj["model_state_dict"]
    return obj


def metric_bundle(labels, probs):
    labels = np.asarray(labels).astype(np.float64)
    probs = np.clip(np.asarray(probs).astype(np.float64), 1e-7, 1 - 1e-7)
    auc = float(roc_auc_score(labels, probs)) if len(np.unique(labels)) > 1 else float("nan")
    ap = float(average_precision_score(labels, probs)) if labels.sum() > 0 else 0.0
    nll = float(-(labels * np.log(probs) + (1-labels) * np.log(1-probs)).mean())
    pos = labels > 0.5
    neg = ~pos
    return {
        "auc": auc,
        "ap": ap,
        "nll": nll,
        "positive_nll": float(-np.log(probs[pos]).mean()) if pos.any() else float("nan"),
        "negative_nll": float(-np.log(1-probs[neg]).mean()) if neg.any() else float("nan"),
        "positives": int(pos.sum()),
        "pairs": int(len(labels)),
        "positive_rate": float(pos.mean()),
        "mean_prob_positive": float(probs[pos].mean()) if pos.any() else float("nan"),
        "mean_prob_negative": float(probs[neg].mean()) if neg.any() else float("nan"),
    }


def cap_data(data, cap):
    if not cap or len(data.sources) <= cap:
        return data
    from utils.data_processing import Data
    sl = slice(0, cap)
    return Data(data.sources[sl], data.destinations[sl], data.timestamps[sl],
                data.edge_idxs[sl], data.labels[sl])


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2, allow_nan=True)


def append_jsonl(path, obj):
    with Path(path).open("a") as f:
        f.write(json.dumps(obj, allow_nan=True) + "\n")


def make_tgn(args, train_finder, node_features, edge_features, device, time_stats):
    # Constructor arguments are the upstream train_supervised.py arguments.
    ms, ss, md, sd = time_stats
    tgn = TGN(
        neighbor_finder=train_finder,
        node_features=node_features,
        edge_features=edge_features,
        device=device,
        n_layers=args.n_layer,
        n_heads=args.n_head,
        dropout=args.drop_out,
        use_memory=args.use_memory,
        message_dimension=args.message_dim,
        memory_dimension=args.memory_dim,
        memory_update_at_start=not args.memory_update_at_end,
        embedding_module_type=args.embedding_module,
        message_function=args.message_function,
        aggregator_type=args.aggregator,
        n_neighbors=args.n_degree,
        mean_time_shift_src=ms,
        std_time_shift_src=ss,
        mean_time_shift_dst=md,
        std_time_shift_dst=sd,
        use_destination_embedding_in_message=args.use_destination_embedding_in_message,
        use_source_embedding_in_message=args.use_source_embedding_in_message,
    ).to(device)
    state = unwrap_state(load_torch(args.pretrained_checkpoint, device))
    missing, unexpected = tgn.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"pretrained TGN mismatch: missing={missing}, unexpected={unexpected}")
    if not args.finetune_host:
        for p in tgn.parameters():
            p.requires_grad_(False)
        tgn.eval()
    return tgn


def select_trace_rows(labels, max_roots, batch_index, seed):
    """Trace positives first, then deterministic negatives; tracing never changes main forward."""
    if max_roots <= 0:
        return []
    labels = np.asarray(labels)
    pos = np.flatnonzero(labels > 0.5).tolist()
    neg = np.flatnonzero(labels <= 0.5).tolist()
    chosen = pos[:max_roots]
    remain = max_roots - len(chosen)
    if remain > 0 and neg:
        rng = np.random.RandomState(seed + 104729 * (batch_index + 1))
        chosen.extend(neg if len(neg) <= remain else rng.choice(neg, size=remain, replace=False).tolist())
    return sorted(chosen)


def reset_memory(tgn):
    if tgn.use_memory:
        tgn.memory.__init_memory__()


def full_official_embedding_call(tgn, sources, destinations, timestamps, edge_idxs, n_neighbors,
                                 grad_enabled):
    """Exactly the source/destination/destination call used by upstream train_supervised.py."""
    ctx = torch.enable_grad() if grad_enabled else torch.no_grad()
    with ctx:
        return tgn.compute_temporal_embeddings(
            sources, destinations, destinations, timestamps, edge_idxs, n_neighbors)


def replay_stream(tgn, data, bs, n_degree):
    tgn.eval()
    if hasattr(tgn.embedding_module, "clear_trace"):
        tgn.embedding_module.clear_trace()
    with torch.no_grad():
        for k in range(math.ceil(len(data.sources) / bs)):
            s, e = k * bs, min(len(data.sources), (k + 1) * bs)
            full_official_embedding_call(
                tgn, data.sources[s:e], data.destinations[s:e], data.timestamps[s:e],
                data.edge_idxs[s:e], n_degree, grad_enabled=False)


def evaluate_stream(tgn, decoder, data, bs, n_degree, reset=False):
    if reset:
        reset_memory(tgn)
    tgn.eval()
    decoder.eval()
    if hasattr(tgn.embedding_module, "clear_trace"):
        tgn.embedding_module.clear_trace()
    probs, labels = [], []
    observed_dims = None
    with torch.no_grad():
        for k in range(math.ceil(len(data.sources) / bs)):
            s, e = k * bs, min(len(data.sources), (k + 1) * bs)
            src, dst, neg = full_official_embedding_call(
                tgn, data.sources[s:e], data.destinations[s:e], data.timestamps[s:e],
                data.edge_idxs[s:e], n_degree, grad_enabled=False)
            if observed_dims is None:
                observed_dims = {
                    "source": int(src.shape[-1]),
                    "destination": int(dst.shape[-1]),
                    "negative": int(neg.shape[-1]),
                }
            probs.append(decoder(src).sigmoid().detach().cpu().numpy())
            labels.append(data.labels[s:e])
    out = metric_bundle(np.concatenate(labels), np.concatenate(probs))
    out["embedding_dims_observed"] = observed_dims or {}
    return out


def counts_of_spectral(prss):
    if prss is None:
        return {"gram": 0, "svd": 0}
    return {
        "gram": sum(int(q.reader_gram_updates_t.item()) for q in prss.quotients.values()),
        "svd": sum(int(q.spectral_updates_t.item()) for q in prss.quotients.values()),
    }


def r_copies(prss):
    if prss is None:
        return {}
    return {k: q.R.detach().cpu().clone() for k, q in prss.quotients.items()}


def r_max_change(before, prss):
    if prss is None:
        return 0.0
    vals = []
    for k, old in before.items():
        vals.append(float((old - prss.quotients[k].R.detach().cpu()).abs().max().item()))
    return max(vals) if vals else 0.0


def _rng_state():
    out = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        out["cuda"] = torch.cuda.get_rng_state_all()
    return out


def _cpu_byte_rng_state(x):
    """Normalize an RNG state for PyTorch generator APIs.

    Rolling checkpoints are loaded with ``map_location=device``.  That remaps the saved CPU
    RNG ByteTensor to CUDA along with model tensors, but ``torch.set_rng_state`` requires a
    CPU ByteTensor.  CUDA generator state setters also accept CPU ByteTensors, so restore RNG
    states after explicitly moving them back to CPU.
    """
    if not torch.is_tensor(x):
        x = torch.as_tensor(x, dtype=torch.uint8)
    return x.detach().to(device="cpu", dtype=torch.uint8).contiguous()


def _restore_rng(state):
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(_cpu_byte_rng_state(state["torch"]))
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all([_cpu_byte_rng_state(x) for x in state["cuda"]])


def _save_rolling(path, *, tgn, decoder, optimizer, unrestricted_optimizer, epoch, next_batch, global_step,
                  best_score, best_epoch, bad_rounds, train_state):
    payload = {
        "tgn": tgn.state_dict(),
        "decoder": decoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "unrestricted_optimizer": unrestricted_optimizer.state_dict() if unrestricted_optimizer is not None else None,
        "epoch": int(epoch),
        "next_batch": int(next_batch),
        "global_step": int(global_step),
        "best_score": float(best_score),
        "best_epoch": int(best_epoch),
        "bad_rounds": int(bad_rounds),
        "train_state": train_state,
        "rng": _rng_state(),
        "memory_backup": tgn.memory.backup_memory() if tgn.use_memory else None,
    }
    tmp = Path(str(path) + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _load_rolling(path, *, tgn, decoder, optimizer, unrestricted_optimizer, device):
    ck = load_torch(path, device)
    tgn.load_state_dict(ck["tgn"])
    decoder.load_state_dict(ck["decoder"])
    optimizer.load_state_dict(ck["optimizer"])
    if unrestricted_optimizer is not None and ck.get("unrestricted_optimizer") is not None:
        unrestricted_optimizer.load_state_dict(ck["unrestricted_optimizer"])
    if tgn.use_memory and ck.get("memory_backup") is not None:
        tgn.memory.restore_memory(ck["memory_backup"])
    _restore_rng(ck.get("rng"))
    return ck


def main():
    args = parse_args()
    seed_all(args.seed)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    metrics_path = out / "metrics.jsonl"
    if not args.resume_from and metrics_path.exists():
        metrics_path.unlink()
    monitor = MonitorWriter(
        out, fail_on_error=not args.no_fail_on_monitor_error,
        response_gap_warn=args.response_gap_warn, reset_files=not bool(args.resume_from))

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Upstream loader hardcodes ./data.  The symlink is the only filesystem adapter.
    os.chdir(METHOD_DIR)
    data_link = METHOD_DIR / "data"
    target_data = Path(args.data_dir).resolve()
    if data_link.is_symlink():
        if data_link.resolve() != target_data:
            data_link.unlink()
            data_link.symlink_to(target_data, target_is_directory=True)
    elif not data_link.exists():
        data_link.symlink_to(target_data, target_is_directory=True)
    elif data_link.resolve() != target_data:
        raise RuntimeError(f"{data_link} exists and is not requested data directory {target_data}")

    # Same upstream loader/task; true validation is enabled only for the matched comparison.
    full, node_features, edge_features, train, val, test = get_data_node_classification(
        args.data, use_validation=True)
    train = cap_data(train, args.max_train)
    val = cap_data(val, args.max_val)
    test = cap_data(test, args.max_test)
    max_idx = max(full.unique_nodes)
    train_finder = get_neighbor_finder(train, uniform=args.uniform, max_node_idx=max_idx)
    time_stats = compute_time_statistics(full.sources, full.destinations, full.timestamps)
    tgn = make_tgn(args, train_finder, node_features, edge_features, device, time_stats)

    prss = None
    adapter = None
    if args.mode == "prss":
        host = tgn.embedding_module
        prss = PRSSCore(
            host_dim=tgn.embedding_dimension,
            edge_dim=tgn.n_edge_features,
            time_dim=tgn.n_node_features,
            n_neighbors=args.n_degree,
            n_layers=args.n_layer,
            candidate_dim=args.candidate_dim,
            candidate_hidden=args.candidate_hidden,
            context_dim=args.context_dim,
            reader_hidden=args.reader_hidden,
            gram_ema=args.gram_ema,
            spectral_step_size=args.spectral_step_size,
        ).to(device)
        adapter = PRSSTGNEmbeddingAdapter(host, prss).to(device)
        tgn.embedding_module = adapter

    # Exact upstream decoder type/dimension.
    decoder = MLP(node_features.shape[1], drop=args.drop_out).to(device)

    # Monitoring-only unrestricted reader gets its own optimizer and detached inputs.
    unrestricted_ids = set()
    unrestricted_optimizer = None
    if prss is not None:
        unrestricted_params = list(prss.unrestricted.parameters())
        unrestricted_ids = {id(p) for p in unrestricted_params}
        unrestricted_optimizer = torch.optim.Adam(unrestricted_params, lr=args.lr)

    main_params = list(decoder.parameters())
    if args.finetune_host:
        main_params.extend(p for p in tgn.parameters() if p.requires_grad and id(p) not in unrestricted_ids)
    elif prss is not None:
        main_params.extend(p for p in prss.parameters() if p.requires_grad and id(p) not in unrestricted_ids)
    # De-duplicate shared module references by parameter identity.
    seen = set()
    main_params = [p for p in main_params if not (id(p) in seen or seen.add(id(p)))]
    optimizer = torch.optim.Adam(main_params, lr=args.lr)

    train_logt = np.log1p(train.timestamps.astype(np.float64))
    log_time_mean = float(train_logt.mean())
    log_time_std = float(train_logt.std() + 1e-8)

    official_sha = (METHOD_DIR / "official_tgn" / "train_supervised.sha256").read_text().strip()
    upstream_commit = (METHOD_DIR / "official_tgn" / "UPSTREAM_COMMIT").read_text().strip()
    config = vars(args).copy()
    config.update({
        "device": str(device),
        "host_dim": int(tgn.embedding_dimension),
        "edge_dim": int(tgn.n_edge_features),
        "node_dim_actual": int(tgn.n_node_features),
        "train_pairs": len(train.sources), "val_pairs": len(val.sources), "test_pairs": len(test.sources),
        "train_positives": int(train.labels.sum()), "val_positives": int(val.labels.sum()),
        "test_positives": int(test.labels.sum()),
        "official_mother_sha256": official_sha,
        "official_upstream_commit": upstream_commit,
        "protocol": "upstream TGN node-classification mother + identical held-out corrections; one vanilla|prss switch",
        "compression_runtime": "future-reader operator bank -> mean(B^T B) -> eigh/right-singular subspace only",
    })
    save_json(out / "config.json", config)

    best_score = -float("inf")
    best_epoch = -1
    bad_rounds = 0
    global_step = 0
    best_path = out / "best.pt"
    rolling_path = out / "rolling_step.pt"
    num_batch = math.ceil(len(train.sources) / args.bs)
    resume_epoch = 0
    resume_batch = 0
    resume_train_state = None
    if args.resume_from:
        ck = _load_rolling(
            args.resume_from, tgn=tgn, decoder=decoder, optimizer=optimizer,
            unrestricted_optimizer=unrestricted_optimizer, device=device)
        resume_epoch = int(ck["epoch"])
        resume_batch = int(ck["next_batch"])
        global_step = int(ck["global_step"])
        best_score = float(ck.get("best_score", best_score))
        best_epoch = int(ck.get("best_epoch", best_epoch))
        bad_rounds = int(ck.get("bad_rounds", bad_rounds))
        resume_train_state = ck.get("train_state")
        print(f"RESUME epoch={resume_epoch} next_batch={resume_batch} global_step={global_step} from={args.resume_from}", flush=True)

    for epoch in range(resume_epoch, args.n_epoch):
        epoch_start = time.time()
        start_batch = resume_batch if epoch == resume_epoch else 0
        if start_batch == 0:
            reset_memory(tgn)
        tgn.train(args.finetune_host)
        decoder.train()
        if prss is not None:
            prss.train()

        if start_batch > 0 and resume_train_state is not None:
            train_probs = resume_train_state["train_probs"]
            train_labels = resume_train_state["train_labels"]
            response_struct_probs = resume_train_state["response_struct_probs"]
            response_unres_probs = resume_train_state["response_unres_probs"]
            response_targets = resume_train_state["response_targets"]
            total_task = float(resume_train_state["total_task"])
            total_resp = float(resume_train_state["total_resp"])
            total_spec = float(resume_train_state["total_spec"])
            total_unres = float(resume_train_state["total_unres"])
            occurrence_counts = dict(resume_train_state["occurrence_counts"])
            svd_events = list(resume_train_state["svd_events"])
        else:
            train_probs, train_labels = [], []
            response_struct_probs, response_unres_probs, response_targets = [], [], []
            total_task = total_resp = total_spec = total_unres = 0.0
            occurrence_counts = {l: 0 for l in range(args.n_layer + 1)}
            svd_events = []

        for k in range(start_batch, num_batch):
            s, e = k * args.bs, min(len(train.sources), (k + 1) * args.bs)
            sources = train.sources[s:e]
            dests = train.destinations[s:e]
            times = train.timestamps[s:e]
            edge_idxs = train.edge_idxs[s:e]
            labels_np = train.labels[s:e]
            labels_t = torch.from_numpy(labels_np).float().to(device)
            optimizer.zero_grad(set_to_none=True)
            if unrestricted_optimizer is not None:
                unrestricted_optimizer.zero_grad(set_to_none=True)

            trace_rows = []
            if adapter is not None:
                trace_rows = select_trace_rows(labels_np, args.trace_roots, global_step, args.seed)
                adapter.set_trace_source_rows(trace_rows)

            # Full official source/destination/destination computation; no source-only shortcut.
            src_emb, _, _ = full_official_embedding_call(
                tgn, sources, dests, times, edge_idxs, args.n_degree,
                grad_enabled=(args.finetune_host or prss is not None))
            logits = decoder(src_emb)
            pred = logits.sigmoid()
            # Keep the upstream node-classification objective exactly: sigmoid + BCELoss.
            task_loss = F.binary_cross_entropy(pred, labels_t)

            aux = None
            resp_v = spec_v = unres_v = 0.0
            if prss is not None and trace_rows:
                selected_labels = labels_t[trace_rows]
                selected_times = torch.from_numpy(times[trace_rows]).float().to(device)
                aux = build_auxiliary(
                    prss, adapter.trace, trace_rows, selected_labels, selected_times,
                    log_time_mean, log_time_std)
                # Fail at the *source* of a non-finite reader, before backward/optimizer can
                # obscure whether the problem came from continuation context or reader weights.
                for layer, C in aux.contexts_by_layer.items():
                    if not bool(torch.isfinite(C).all()):
                        monitor.alert("error", "outside_context_nonfinite", f"layer {layer}", step=global_step)
                for layer, B in aux.matrices_by_layer.items():
                    if not bool(torch.isfinite(B).all()):
                        rs = module_finiteness(prss.readers[str(layer)])
                        os_ = module_finiteness(prss.outside)
                        monitor.alert(
                            "error", "reader_matrix_nonfinite_pre_backward",
                            f"layer {layer} reader_param_finite={rs['finite_fraction']:.6f} "
                            f"reader_param_max={rs['max_abs']:.3e} outside_finite={os_['finite_fraction']:.6f} "
                            f"outside_max={os_['max_abs']:.3e}", step=global_step)
                main_loss = (task_loss + args.lambda_response * aux.response_loss +
                             args.lambda_spectral * aux.spectral_loss)
                resp_v = float(aux.response_loss.detach().item())
                spec_v = float(aux.spectral_loss.detach().item())
                unres_v = float(aux.unrestricted_loss.detach().item())
                if aux.targets.numel() > 0:
                    response_struct_probs.append(aux.structured_logits.sigmoid().cpu().numpy())
                    response_unres_probs.append(aux.unrestricted_logits.sigmoid().cpu().numpy())
                    response_targets.append(aux.targets.cpu().numpy())
            else:
                main_loss = task_loss

            monitor.validate_losses({
                "task": task_loss.detach().item(),
                "response": resp_v,
                "spectral": spec_v,
                "unrestricted_monitor": unres_v,
                "main_total": main_loss.detach().item(),
            }, global_step)

            main_loss.backward()
            # The future-reader loss is auxiliary to the official TGN task and can see very
            # sparse positive batches.  Bound its optimizer step without changing the forward
            # objective or the spectral statistic.  clip_grad_norm_ returns the pre-clip norm.
            main_grad_preclip = 0.0
            if args.grad_clip > 0:
                main_grad_preclip = float(torch.nn.utils.clip_grad_norm_(
                    main_params, max_norm=args.grad_clip, error_if_nonfinite=True).item())
            # Record gradients after clipping; also log the pre-clip total norm.
            do_heavy_monitor = (k % args.monitor_every == 0)
            if do_heavy_monitor:
                prss_grad = {
                    "main_total_preclip": main_grad_preclip,
                    "candidate_builders": grad_l2(prss.builders) if prss is not None else 0.0,
                    "structured_readers": grad_l2(prss.readers) if prss is not None else 0.0,
                    "outside_encoder": grad_l2(prss.outside) if prss is not None else 0.0,
                    "decoder": grad_l2(decoder),
                    "host_embedding": grad_l2(adapter.host if adapter is not None else tgn.embedding_module)
                                      if args.finetune_host else 0.0,
                    "host_time_encoder": grad_l2(tgn.time_encoder) if args.finetune_host else 0.0,
                    "host_memory_updater": grad_l2(tgn.memory_updater)
                                           if args.finetune_host and tgn.use_memory else 0.0,
                }
            else:
                prss_grad = {}
            optimizer.step()

            if unrestricted_optimizer is not None and aux is not None:
                aux.unrestricted_loss.backward()
                unres_grad_preclip = 0.0
                if args.unrestricted_grad_clip > 0:
                    unres_grad_preclip = float(torch.nn.utils.clip_grad_norm_(
                        list(prss.unrestricted.parameters()), max_norm=args.unrestricted_grad_clip,
                        error_if_nonfinite=True).item())
                if do_heavy_monitor:
                    prss_grad["unrestricted_total_preclip"] = unres_grad_preclip
                    prss_grad["unrestricted_reader"] = grad_l2(prss.unrestricted)
                unrestricted_optimizer.step()

            # Same truncation invariant used by upstream self-supervised TGN when host gradients exist.
            if tgn.use_memory and (args.finetune_host or prss is not None):
                tgn.memory.detach_memory()

            if prss is not None:
                for lname, mod in (("readers", prss.readers), ("outside", prss.outside), ("builders", prss.builders)):
                    st = module_finiteness(mod)
                    if st["finite_fraction"] < 1.0:
                        monitor.alert("error", "prss_parameter_nonfinite_after_step",
                                      f"module={lname} finite={st['finite_fraction']:.6f} max_abs={st['max_abs']:.3e}",
                                      step=global_step)

            # Block-coordinate spectral statistic/update.  R is never a gradient parameter.
            if prss is not None and aux is not None:
                with torch.no_grad():
                    for layer, B in aux.matrices_by_layer.items():
                        prss.quotients[str(layer)].accumulate(B)
                        occurrence_counts[layer] += int(aux.occurrence_counts.get(layer, 0))
                    completed = global_step + 1
                    if completed >= args.spectral_warmup and completed % args.spectral_interval == 0:
                        for layer in range(1, args.n_layer + 1):
                            q = prss.quotients[str(layer)]
                            if q.update(completed):
                                snap = q.snapshot()
                                event = {"step": completed, "layer": layer, **snap}
                                svd_events.append(event)
                                print(
                                    f"SVD_UPDATE step={completed} layer={layer} total={snap['spectral_updates']} "
                                    f"rank={snap['effective_predictive_rank']} energy@k={snap['energy_at_k']:.6f} "
                                    f"tail@k={snap['tail_at_k']:.6f} proj_dist={snap['projector_distance']:.5f} "
                                    f"step={snap['accepted_spectral_step']:.3f} "
                                    f"gain={snap['captured_energy_gain']:.6f}", flush=True)

            probs = pred.detach().cpu().numpy()
            train_probs.append(probs)
            train_labels.append(labels_np)
            total_task += float(task_loss.detach().item())
            total_resp += resp_v
            total_spec += spec_v
            total_unres += unres_v

            if do_heavy_monitor:
                spectral_now = prss.snapshots() if prss is not None else {}
                monitor.validate_spectral(spectral_now, global_step)
                cstats = candidate_stats(adapter.trace) if adapter is not None else {}
                bstats = matrix_stats(aux.matrices_by_layer) if aux is not None else {}
                for layer, st in cstats.items():
                    if st.get("finite_fraction", 1.0) < 1.0:
                        monitor.alert("error", "candidate_nonfinite", f"layer {layer}", step=global_step)
                for layer, st in bstats.items():
                    # Hard invariants must use exact integer counts / torch.isfinite(...).all(),
                    # never a floating-point reduction of the finite mask.  The pre-backward
                    # check above already establishes finiteness of this immutable snapshot; this
                    # second check is retained only to detect genuine post-build corruption.
                    if int(st.get("nonfinite_count", 0)) > 0:
                        Bchk = aux.matrices_by_layer[int(layer)]
                        bad = (~torch.isfinite(Bchk)).nonzero(as_tuple=False)
                        first_bad = bad[0].detach().cpu().tolist() if bad.numel() else []
                        rs = module_finiteness(prss.readers[str(layer)])
                        os_ = module_finiteness(prss.outside)
                        monitor.alert(
                            "error", "reader_snapshot_nonfinite_after_backward",
                            f"layer {layer} nonfinite={st.get('nonfinite_count')}/{st.get('elements')} "
                            f"first_bad={first_bad} reader_param_finite={rs['finite_fraction']:.6f} "
                            f"reader_param_max={rs['max_abs']:.3e} outside_finite={os_['finite_fraction']:.6f} "
                            f"outside_max={os_['max_abs']:.3e}",
                            step=global_step)
                step_row = {
                    "epoch": epoch, "batch": k, "global_step": global_step, "mode": args.mode,
                    "batch_positives": int(labels_np.sum()), "batch_pairs": int(len(labels_np)),
                    "loss": {"task": float(task_loss.detach().item()), "response": resp_v,
                             "spectral": spec_v, "unrestricted_monitor": unres_v,
                             "main_total": float(main_loss.detach().item())},
                    "grad_l2": prss_grad,
                    "candidate": cstats,
                    "future_reader_B": bstats,
                    "spectral": spectral_now,
                    "traced_roots": int(len(trace_rows)),
                }
                monitor.write_step(step_row)
                svd_total = counts_of_spectral(prss)["svd"] if prss is not None else 0
                print(
                    f"epoch={epoch} batch={k}/{num_batch} mode={args.mode} pos={int(labels_np.sum())} "
                    f"task={task_loss.item():.5f} resp={resp_v:.5f} spec={spec_v:.5f} "
                    f"unres_monitor={unres_v:.5f} svd_total={svd_total}", flush=True)
            global_step += 1
            if args.checkpoint_every > 0 and ((k + 1) % args.checkpoint_every == 0):
                train_state = {
                    "train_probs": train_probs, "train_labels": train_labels,
                    "response_struct_probs": response_struct_probs,
                    "response_unres_probs": response_unres_probs, "response_targets": response_targets,
                    "total_task": total_task, "total_resp": total_resp, "total_spec": total_spec,
                    "total_unres": total_unres, "occurrence_counts": occurrence_counts,
                    "svd_events": svd_events,
                }
                _save_rolling(rolling_path, tgn=tgn, decoder=decoder, optimizer=optimizer,
                              unrestricted_optimizer=unrestricted_optimizer, epoch=epoch,
                              next_batch=k + 1, global_step=global_step, best_score=best_score,
                              best_epoch=best_epoch, bad_rounds=bad_rounds, train_state=train_state)

        # Any next epoch starts from a fresh chronological memory stream.
        resume_batch = 0
        resume_train_state = None
        train_metrics = metric_bundle(np.concatenate(train_labels), np.concatenate(train_probs))
        if response_targets:
            rt = np.concatenate(response_targets)
            response_structured = metric_bundle(rt, np.concatenate(response_struct_probs))
            response_unrestricted = metric_bundle(rt, np.concatenate(response_unres_probs))
        else:
            response_structured = {}
            response_unrestricted = {}

        # Validation begins from the memory produced by the chronological train stream, matching
        # upstream semantics.  PRSS trace/Gram/SVD are disabled and audited around the call.
        before_counts = counts_of_spectral(prss)
        before_R = r_copies(prss)
        val_metrics = evaluate_stream(tgn, decoder, val, args.bs, args.n_degree, reset=False)
        after_counts = counts_of_spectral(prss)
        val_r_change = r_max_change(before_R, prss)
        if before_counts != after_counts or val_r_change != 0.0:
            monitor.alert("error", "validation_mutated_spectral_state",
                          f"counts {before_counts}->{after_counts}, R_change={val_r_change}", step=global_step)

        score = val_metrics[args.selection_metric]
        spectral = prss.snapshots() if prss is not None else {}
        if response_structured and response_unrestricted:
            response_gap = response_structured["nll"] - response_unrestricted["nll"]
            if np.isfinite(response_gap) and response_gap > args.response_gap_warn:
                monitor.alert("warning", "structured_reader_gap",
                              f"structured-unrestricted NLL gap={response_gap:.4f}",
                              epoch=epoch, step=global_step)
        else:
            response_gap = float("nan")

        row = {
            "epoch": epoch, "global_step": global_step, "mode": args.mode,
            "train_task_loss": total_task / max(num_batch, 1),
            "train_response_loss": total_resp / max(num_batch, 1),
            "train_spectral_loss": total_spec / max(num_batch, 1),
            "train_unrestricted_monitor_loss": total_unres / max(num_batch, 1),
            "train": train_metrics, "val": val_metrics,
            "response_structured": response_structured,
            "response_unrestricted": response_unrestricted,
            "response_nll_gap_structured_minus_unrestricted": response_gap,
            "spectral": spectral,
            "occurrence_counts": occurrence_counts,
            "svd_events_this_epoch": svd_events,
            "validation_spectral_isolation": {
                "counts_before": before_counts, "counts_after": after_counts,
                "max_R_change": val_r_change,
            },
            "epoch_seconds": time.time() - epoch_start,
        }
        append_jsonl(metrics_path, row)
        monitor.write_epoch(row)
        monitor.save_projection_snapshot(epoch, prss)
        print(
            f"epoch={epoch} mode={args.mode} train_auc={train_metrics['auc']:.5f} "
            f"train_ap={train_metrics['ap']:.5f} val_auc={val_metrics['auc']:.5f} "
            f"val_ap={val_metrics['ap']:.5f} val_nll={val_metrics['nll']:.5f} "
            f"val_pos={val_metrics['positives']} sec={row['epoch_seconds']:.1f}", flush=True)

        score_is_finite = bool(np.isfinite(score))
        improved = (best_epoch < 0) or (score_is_finite and score > best_score + 1e-12)
        if improved:
            best_score = float(score) if score_is_finite else -float("inf")
            best_epoch = epoch
            bad_rounds = 0
            torch.save({"decoder": decoder.state_dict(), "tgn": tgn.state_dict(),
                        "epoch": epoch, "score": score}, best_path)
        else:
            bad_rounds += 1
            if bad_rounds >= args.patience:
                print(f"early_stop epoch={epoch} best_epoch={best_epoch} "
                      f"best_{args.selection_metric}={best_score:.6f}", flush=True)
                break

    best = load_torch(best_path, device)
    decoder.load_state_dict(best["decoder"])
    tgn.load_state_dict(best["tgn"])
    decoder.eval()
    tgn.eval()
    if prss is not None:
        prss.eval()
        adapter.clear_trace()

    # Held-out test is reconstructed from zero memory with best parameters, then train+val replay.
    reset_memory(tgn)
    replay_stream(tgn, train, args.bs, args.n_degree)
    replay_stream(tgn, val, args.bs, args.n_degree)
    before_test_counts = counts_of_spectral(prss)
    before_test_R = r_copies(prss)
    test_metrics = evaluate_stream(tgn, decoder, test, args.bs, args.n_degree, reset=False)
    after_test_counts = counts_of_spectral(prss)
    test_r_change = r_max_change(before_test_R, prss)
    test_trace_created = bool(adapter is not None and adapter.trace is not None)
    if before_test_counts != after_test_counts or test_r_change != 0.0 or test_trace_created:
        monitor.alert("error", "test_inference_isolation_failed",
                      f"counts {before_test_counts}->{after_test_counts}, R_change={test_r_change}, "
                      f"trace={test_trace_created}", step=global_step)

    summary = {
        "mode": args.mode,
        "host_training": "finetuned" if args.finetune_host else "frozen_official_style",
        "best_epoch": int(best_epoch),
        "selection_metric": args.selection_metric,
        "best_validation_score": float(best_score),
        "test": test_metrics,
        "spectral": prss.snapshots() if prss is not None else {},
        "pretrained_checkpoint": str(args.pretrained_checkpoint),
        "inference_isolation": {
            "counts_before": before_test_counts, "counts_after": after_test_counts,
            "max_R_change": test_r_change, "trace_created": test_trace_created,
        },
    }
    save_json(out / "summary.json", summary)
    monitor.finalize(summary)
    save_json(out / "_SUCCESS.json", {"status": "complete", "best_epoch": best_epoch, "test": test_metrics})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
