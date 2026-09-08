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

from contextlib import contextmanager
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch

from rpbe.data.wiki_binary_negatives import CompactNegatives
from rpbe.hosts.official_tgn import get_neighbor_finder


@contextmanager
def _full_finder_ctx(tgn, ds):
    """Serve recursive neighbors from the FULL stream during an evaluation.

    Training uses a train-only finder; when we evaluate val/test the memory
    advances over events the train finder cannot see, so the recursion must
    also see them.  ``find_before`` keeps strictly-before-cut-time neighbours,
    so a full-stream finder never leaks future edges.  The caller's finder is
    restored afterwards (spec §1.1 #3).
    """
    full = getattr(ds, "_wiki_full_finder", None)
    if full is None:
        full = get_neighbor_finder(
            ds.full, uniform=False,
            max_node_idx=int(ds.n_internal_nodes - 1))
        ds._wiki_full_finder = full
    orig = tgn.neighbor_finder
    tgn.set_neighbor_finder(full)
    try:
        yield
    finally:
        tgn.set_neighbor_finder(orig)


# ------------------------------------------------------------------ metrics
def _ap(pos: np.ndarray, neg: np.ndarray) -> float:
    """Average precision with tie-groups processed as one block.

    Candidates are sorted by descending score (stable); every candidate with
    the SAME score forms one tie-group, and the group's positives all receive
    the precision reached AFTER the whole group is added.  With all scores
    equal this returns the positive rate (1:1 -> 0.5), never ~1.  When there
    are no ties it matches the usual running-precision definition.
    """
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    scores = np.concatenate([pos, neg])
    labels = np.concatenate([np.ones(pos.size), np.zeros(neg.size)])
    order = np.argsort(-scores, kind="stable")
    ss = scores[order]
    ll = labels[order]
    n = ss.size
    # group boundaries at distinct descending scores
    cuts = np.flatnonzero(np.diff(ss) != 0) + 1
    starts = np.concatenate([[0], cuts])
    ends = np.concatenate([cuts, [n]])
    lens = ends - starts
    grp_pos = np.add.reduceat(ll, starts)
    cum_pos = np.cumsum(grp_pos)
    cum_tot = np.cumsum(lens)
    ap = float((grp_pos * (cum_pos / cum_tot)).sum() / pos.size)
    return ap


def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """ROC-AUC via the pair-count rank statistic (strict >; ties count 0.5)."""
    pos = np.sort(np.asarray(pos, dtype=np.float64))
    neg = np.sort(np.asarray(neg, dtype=np.float64))
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    below = np.searchsorted(pos, neg, side="right")    # #pos <= neg
    strictly = np.searchsorted(pos, neg, side="left")   # #pos < neg
    wins = pos.size - below                              # #pos > neg
    ties = below - strictly                              # #pos == neg
    out = (wins + 0.5 * ties).sum() / (pos.size * neg.size)
    return float(min(1.0, max(0.0, out)))


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
            # fail-closed: every split event must be present in the manifest
            # (spec §1.1 #4); a missing negative is an error, never a fallback
            # to the positive dst.
            neg_vals = np.zeros(len(eidx_b), dtype=np.int64)
            if collect:
                hist_flags[idx] = False
            for k, r in enumerate(eidx_b):
                gr = int(r) - 1
                nv = negatives.neg_for(split, gr)
                if nv is None:
                    raise KeyError(
                        "manifest missing negative: split={} global_row={}".format(
                            split, gr))
                neg_vals[k] = int(nv) + 1
                if collect:
                    hist_flags[idx][k] = (negatives.type_for(split, gr)
                                          == "hist")
            p, ng = tgn.compute_edge_probabilities(
                src_b, pos_b, neg_vals, t_b, eidx_b, n_neighbors)
            if collect:
                pos_scores[idx] = p.squeeze().cpu().numpy()
                neg_scores[idx] = ng.squeeze().cpu().numpy()
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
        with _full_finder_ctx(tgn, ds):
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
    with _full_finder_ctx(tgn, ds):
        replay_memory(tgn, ds, "val", n_neighbors=n_neighbors, bs=bs)
        return score_split_binary(tgn, ds, "test", negatives,
                                  n_neighbors=n_neighbors, bs=bs)
