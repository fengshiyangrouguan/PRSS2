"""Wiki-LR-Binary full-stream global metrics (spec §2.4, §3).

One positive -> one fixed negative.  Evaluation walks a WHOLE split in
chronological order; each batch scores positive/negative BEFORE that batch's
events update memory (the official host applies the *previous* batch's stored
messages at the start of the next call, so a forward-then-stage loop is
exactly pre-event online semantics), then memory advances only over real
positive events with their TRUE edge_idx.  All metrics are computed over the
concatenated per-event scores of the FULL split — never by averaging per-batch
values.

Metric definitions are deterministic (a fixed tie convention), so a
scalar/reference implementation over the same score sequence reproduces them
exactly (spec §8 unit gate).
"""

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch

from rpbe.data.wiki_binary_negatives import CompactNegatives


# ------------------------------------------------------------------ metrics
def _ap(pos: np.ndarray, neg: np.ndarray) -> float:
    """Average precision (positives-first within score ties).

    Candidates sorted by (desc score, desc label).  Ties therefore resolve
    positives ahead of negatives; the definition is deterministic and matches
    sklearn's when there are no score ties.
    """
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    scores = np.concatenate([pos, neg])
    labels = np.concatenate([np.ones(pos.size), np.zeros(neg.size)])
    order = np.lexsort((-labels, -scores))  # desc label within desc score
    n_correct = 0
    total = 0.0
    ap = 0.0
    for idx in order:
        total += 1
        if labels[idx] == 1:
            n_correct += 1
            ap += n_correct / total
    return float(ap / pos.size)


def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """ROC-AUC via the pair-count rank statistic (ties count 0.5)."""
    pos = np.sort(np.asarray(pos, dtype=np.float64))
    neg = np.sort(np.asarray(neg, dtype=np.float64))
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    # for each negative, how many positives are strictly below / tied
    below = np.searchsorted(pos, neg, side="right")   # pos <= neg
    strictly = np.searchsorted(pos, neg, side="left")  # pos < neg
    wins = pos.size - strictly                        # pos > neg
    ties = below - strictly                            # pos == neg
    return float((wins + 0.5 * ties).sum() / (pos.size * neg.size))


def _nll(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mean binary cross-entropy over all positives (label 1) + negs (0)."""
    pos = np.clip(np.asarray(pos, dtype=np.float64), 1e-7, 1 - 1e-7)
    neg = np.clip(np.asarray(neg, dtype=np.float64), 1e-7, 1 - 1e-7)
    return float(-(np.log(pos).mean() + np.log1p(-neg).mean()) / 2.0)


def global_metrics(pos: np.ndarray, neg: np.ndarray,
                   hist_mask: Optional[np.ndarray] = None) -> Dict:
    """Global metrics over full (or masked-historical) pos/neg score arrays."""
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    assert pos.shape == neg.shape, "one neg per pos required"
    out = {
        "n_pos": int(pos.size),
        "ap_all": _ap(pos, neg),
        "auc_all": _auc(pos, neg),
        "nll_all": _nll(pos, neg),
        "paired_acc": float((pos > neg).mean() + 0.5 * (pos == neg).mean()),
    }
    if hist_mask is not None:
        hm = np.asarray(hist_mask, dtype=bool)
        out["n_hist"] = int(hm.sum())
        out["n_random"] = int((~hm).sum())
        if hm.any():
            out["ap_hist"] = _ap(pos[hm], neg[hm])
            out["auc_hist"] = _auc(pos[hm], neg[hm])
            out["nll_hist"] = _nll(pos[hm], neg[hm])
        else:
            out.update({"ap_hist": float("nan"), "auc_hist": float("nan"),
                        "nll_hist": float("nan")})
        rm = ~hm
        if rm.any():
            out["ap_random"] = _ap(pos[rm], neg[rm])
            out["auc_random"] = _auc(pos[rm], neg[rm])
            out["nll_random"] = _nll(pos[rm], neg[rm])
        else:
            out.update({"ap_random": float("nan"),
                        "auc_random": float("nan"),
                        "nll_random": float("nan")})
    return out


# ------------------------------------------------------------- stream scorer
def _iter_batches(n: int, bs: int):
    for lo in range(0, n, bs):
        yield lo, min(n, lo + bs)


def score_split_binary(tgn, ds, split, negatives: CompactNegatives, *,
                       n_neighbors: int = 5, bs: int = 200,
                       collect: bool = True) -> Dict:
    """Score a whole split one-neg-per-positive, advancing memory online.

    ``memory`` must already be at the correct start state (train-end for val;
    train+val replayed for test).  Returns full-split global metrics plus the
    raw (pos, neg, hist) score arrays when ``collect`` (evaluator internals /
    replay both share this forward path; replay passes collect=False).
    """
    split_obj = {"val": ds.val, "test": ds.test, "train": ds.train}[split]
    sources = split_obj.sources
    dst_pos = split_obj.destinations
    timestamps = split_obj.timestamps
    eidx = split_obj.edge_idxs
    n = len(sources)

    pos_scores = np.zeros(n, dtype=np.float64) if collect else None
    neg_scores = np.zeros(n, dtype=np.float64) if collect else None
    hist_flags = np.zeros(n, dtype=bool) if collect else None

    with torch.no_grad():
        for lo, hi in _iter_batches(n, bs):
            idx = slice(lo, hi)
            src_b = sources[idx]
            pos_b = dst_pos[idx]
            t_b = timestamps[idx]
            eidx_b = eidx[idx]
            neg_b = np.asarray(
                [int(negatives.neg_for(split, int(r) - 1))
                 + 1 if negatives.neg_for(split, int(r) - 1) is not None
                 else int(pos_b[k]) for k, r in enumerate(eidx_b)],
                dtype=np.int64)
            p, ng = tgn.compute_edge_probabilities(
                src_b, pos_b, neg_b, t_b, eidx_b, n_neighbors)
            if collect:
                pos_scores[idx] = p.squeeze().cpu().numpy()
                neg_scores[idx] = ng.squeeze().cpu().numpy()
                hist_flags[idx] = [int(negatives.type_for(
                    split, int(r) - 1) or "rand") == "hist"
                    for r in eidx_b]
    if not collect:
        return {}
    return global_metrics(pos_scores, neg_scores, hist_flags)


def replay_memory(tgn, ds, split, *, n_neighbors: int = 5,
                  bs: int = 200) -> None:
    """Advance memory over a split's real positives (scores discarded)."""
    split_obj = {"val": ds.val, "test": ds.test, "train": ds.train}[split]
    sources = split_obj.sources
    dst_pos = split_obj.destinations
    timestamps = split_obj.timestamps
    eidx = split_obj.edge_idxs
    n = len(sources)
    with torch.no_grad():
        for lo, hi in _iter_batches(n, bs):
            idx = slice(lo, hi)
            # negative placeholder never stored: memory only advances over src
            # + positive dst with the true edge idx.
            tgn.compute_edge_probabilities(
                sources[idx], dst_pos[idx], dst_pos[idx], timestamps[idx],
                eidx[idx], n_neighbors)


def evaluate_val(tgn, ds, negatives: CompactNegatives, *,
                 n_neighbors: int = 5, bs: int = 200) -> Dict:
    """Score val from the current memory state, then RESTORE that state.

    Training ends each epoch at train-end memory; val evaluation advances
    memory over val, and we must put train-end back so the next epoch starts
    clean (spec §2.4).
    """
    backup = tgn.memory.backup_memory() if tgn.use_memory else None
    try:
        return score_split_binary(tgn, ds, "val", negatives,
                                  n_neighbors=n_neighbors, bs=bs)
    finally:
        if backup is not None:
            tgn.memory.restore_memory(backup)


def run_test_protocol(tgn, ds, negatives: CompactNegatives, *,
                      n_neighbors: int = 5, bs: int = 200) -> Dict:
    """Fresh test protocol: reset, replay train then val, score test."""
    if tgn.use_memory:
        tgn.memory.__init_memory__()
    replay_memory(tgn, ds, "train", n_neighbors=n_neighbors, bs=bs)
    replay_memory(tgn, ds, "val", n_neighbors=n_neighbors, bs=bs)
    return score_split_binary(tgn, ds, "test", negatives,
                              n_neighbors=n_neighbors, bs=bs)
