"""Read-only sampled-query MRR scoring shared by eval and checkpoint select.

Scoring is strictly-equivalent and side-effect free w.r.t. model state:
the caller is responsible for the memory being correct *before* each scored
event (train replay / online advance handled by the caller).  For each
scored event the source embedding is computed once; candidate destinations
(positive dst + full official negatives) are scored in chunks against the
same pre-event memory.
"""

from typing import List, Optional, Sequence

import numpy as np
import torch


def _read_emb(tgn, memory, nodes, times, n_neighbors):
    return tgn.embedding_module.compute_embedding(
        memory=memory, source_nodes=np.asarray(nodes, dtype=np.int64),
        timestamps=np.asarray(times, dtype=np.float64),
        n_layers=tgn.n_layers, n_neighbors=n_neighbors)


def _current_memory(tgn):
    return tgn.memory.get_memory(list(range(tgn.n_nodes))) \
        if tgn.use_memory else None


def score_one(tgn, memory, src, t, pos_dst, neg_dsts, n_neighbors, chunk):
    """(pos_score, neg_scores) with source computed once; dsts chunked."""
    src_emb = _read_emb(tgn, memory, [src], [t], n_neighbors)[0]
    pos_emb = _read_emb(tgn, memory, [pos_dst], [t], n_neighbors)[0]
    pos_score = float(tgn.affinity_score(
        src_emb.unsqueeze(0), pos_emb.unsqueeze(0)).squeeze(0))
    neg_scores = []
    for c0 in range(0, len(neg_dsts), chunk):
        seg = neg_dsts[c0:c0 + chunk]
        if not len(seg):
            continue
        e = _read_emb(tgn, memory, seg, [t] * len(seg), n_neighbors)
        x = src_emb.unsqueeze(0).expand(len(seg), -1)
        s = tgn.affinity_score(x, e).squeeze(0)
        neg_scores.extend([float(v) for v in s])
    return pos_score, neg_scores


def avg_tie_rank(pos_score, neg_scores):
    gt = float(sum(1 for v in neg_scores if v > pos_score))
    ge = float(sum(1 for v in neg_scores if v >= pos_score))
    return 1.0 + 0.5 * (gt + ge)


def score_split(tgn, ds, split_mode, qids: Sequence[int], *,
                n_neighbors: int = 10, chunk: int = 64,
                advance_stream=None) -> dict:
    """Score the given split event indices against official negatives.

    ``qids`` are indices into the split stream (row order == chronological).
    ``advance_stream`` optionally advances memory per scored real event
    (a callable invoked as advance_stream(src, dst, t, eidx)); when None the
    memory is left untouched (caller already replayed it).  Reads official
    negatives via the dataset's sampler.
    """
    raw_src, raw_dst, raw_t = ds.raw_split(split_mode)
    ranks = []
    with torch.no_grad():
        for i in qids:
            i = int(i)
            src = int(raw_src[i]) + 1
            dst = int(raw_dst[i]) + 1
            tt = float(raw_t[i])
            negs_raw = ds.query_negatives(
                np.asarray([int(raw_src[i])], dtype=np.int64),
                np.asarray([int(raw_dst[i])], dtype=np.int64),
                np.asarray([tt], dtype=np.float64), split_mode=split_mode)
            neg_in = np.asarray([int(x) for x in negs_raw[0]],
                                dtype=np.int64) + 1
            mem = _current_memory(tgn)
            ps, ns = score_one(tgn, mem, src, tt, dst, neg_in,
                               n_neighbors, chunk)
            ranks.append(1.0 / avg_tie_rank(ps, ns))
            if advance_stream is not None:
                advance_stream(
                    np.asarray([src], dtype=np.int64),
                    np.asarray([dst], dtype=np.int64),
                    np.asarray([tt], dtype=np.float64),
                    np.asarray([src], dtype=np.int64))
    mrr = float(np.mean(ranks)) if ranks else float("nan")
    return {("sampled_query_{}_mrr".format(split_mode)): mrr,
            "n_scored": len(ranks)}
