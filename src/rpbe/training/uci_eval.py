"""UCI full-split val/test AP/AUC under the official BenchTemp protocol.

Official evaluation semantics (transductive main protocol):
* fixed-seed negative sampler over the FULL stream (val seed=0, test seed=2)
* one negative per positive, sampled once per evaluation call
* per-batch online advance: each scored positive batch updates memory
* metric: sklearn AUC (primary) / AP (secondary)
* memory CONTINUES across splits: per-epoch val is scored from the train-end
  memory the loop leaves behind; the final test resets memory and replays
  train + val before scoring test (mirrors ``main``'s
  reset -> replay train -> replay val -> test).  No split is ever scored cold.

The model is switched to eval() (dropout off, official convention).  The
caller's next training epoch resets memory, so no memory restore is needed
between val calls.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


def _advance(tgn, src, dst, neg, t, eidx, n_neighbors, bs):
    """One scored batch: pos pairs update memory, negs do not."""
    with torch.no_grad():
        pos_prob, neg_prob = tgn.compute_edge_probabilities(
            np.asarray(src, dtype=np.int64),
            np.asarray(dst, dtype=np.int64),
            np.asarray(neg, dtype=np.int64),
            np.asarray(t, dtype=np.float64),
            np.asarray(eidx, dtype=np.int64), n_neighbors)
    return (pos_prob.squeeze().cpu().numpy(),
            neg_prob.squeeze().cpu().numpy())


def replay_split(tgn, stream, *, n_neighbors=10, bs=200):
    """Advance cross-batch memory over a split's positive edges, no scoring.

    Replays the official positive-edge update path (identity messages depend
    only on memory + edge features + time, so the discarded embeddings and the
    dummy negatives leave memory identical to a real scoring pass).
    """
    n = len(stream.sources)
    if n == 0 or not tgn.use_memory:
        return
    for start in range(0, n, bs):
        end = min(n, start + bs)
        with torch.no_grad():
            tgn.compute_edge_probabilities(
                np.asarray(stream.sources[start:end], dtype=np.int64),
                np.asarray(stream.destinations[start:end], dtype=np.int64),
                np.asarray(stream.destinations[start:end], dtype=np.int64),
                np.asarray(stream.timestamps[start:end], dtype=np.float64),
                np.asarray(stream.edge_idxs[start:end], dtype=np.int64),
                n_neighbors)


def evaluate_split(tgn, ds, split="val", *, n_neighbors=10, bs=200,
                   seed=None, full_finder=None, reset=False):
    """Score one split; returns {ap, auc, n_pos}.

    Memory is NOT reset here: it continues from the caller's current state
    (train-end for per-epoch val; train+val replay for the final test).  Pass
    ``reset=True`` only when scoring cold is truly intended.

    Official protocol: evaluation neighbor sampling uses the FULL graph
    finder (train+val+test edges), so the finder is swapped in for the
    scoring pass and swapped back afterwards.
    """
    stream = ds.val if split == "val" else ds.test
    n = len(stream.sources)
    if n == 0:
        return {"ap": float("nan"), "auc": float("nan"), "n_pos": 0}
    if seed is None:
        seed = 0 if split == "val" else 2
    _, neg_dst = ds.val_negatives(n, seed=seed)

    train_finder = None
    if full_finder is not None:
        train_finder = tgn.embedding_module.neighbor_finder
        tgn.embedding_module.neighbor_finder = full_finder
    tgn.eval()
    if reset and tgn.use_memory:
        tgn.memory.__init_memory__()
    pos_probs, neg_probs = [], []
    for start in range(0, n, bs):
        end = min(n, start + bs)
        pp, np_ = _advance(
            tgn, stream.sources[start:end], stream.destinations[start:end],
            neg_dst[start:end], stream.timestamps[start:end],
            stream.edge_idxs[start:end], n_neighbors, bs)
        pos_probs.append(pp)
        neg_probs.append(np_)
    pos = np.concatenate(pos_probs)
    neg = np.concatenate(neg_probs)
    labels = np.concatenate([np.ones_like(pos), np.zeros_like(neg)])
    scores = np.concatenate([pos, neg])
    if train_finder is not None:
        tgn.embedding_module.neighbor_finder = train_finder
    return {"ap": float(average_precision_score(labels, scores)),
            "auc": float(roc_auc_score(labels, scores)),
            "n_pos": int(n)}


def evaluate_val(tgn, ds, *, n_neighbors=10, bs=200, full_finder=None):
    """Official val evaluation; keys match the runner's val_row contract.

    Scored from the current memory state (train-end after the loop's
    train_epoch), which is exactly the official val protocol.
    """
    r = evaluate_split(tgn, ds, "val", n_neighbors=n_neighbors, bs=bs,
                       full_finder=full_finder)
    return {
        "ap_all": r["ap"],
        "auc_all": r["auc"],
        "nll_all": float("nan"),
        "paired_acc": float("nan"),
        "ap_hist": float("nan"), "auc_hist": float("nan"),
        "nll_hist": float("nan"),
        "n_hist": 0, "n_random": r["n_pos"],
    }


def evaluate_test(tgn, ds, *, n_neighbors=10, bs=200, full_finder=None):
    """Final test: reset memory, replay train then val, score test.

    Mirrors ``main``'s ``reset_memory() -> replay_split(train) ->
    replay_split(val) -> evaluate_split(test, reset=False)``.  Read exactly
    once after checkpoint selection (never used for early stopping).
    """
    if tgn.use_memory:
        tgn.memory.__init_memory__()
        replay_split(tgn, ds.train, n_neighbors=n_neighbors, bs=bs)
        replay_split(tgn, ds.val, n_neighbors=n_neighbors, bs=bs)
    return evaluate_split(tgn, ds, "test", n_neighbors=n_neighbors, bs=bs,
                          full_finder=full_finder)
