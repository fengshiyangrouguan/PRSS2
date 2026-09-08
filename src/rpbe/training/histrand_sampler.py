"""Prefix-causal HistRand train negative sampler (Wiki-LR-Binary §2.3, §1.1 #5).

For each train positive event the root task needs one negative destination.
With probability 0.5 the negative is a *historical* destination of that src
(a dst the src interacted with at an EARLIER train event); otherwise it is
drawn from the train destination universe, EXCLUDING every dst this src has
already seen (so hist and random are genuinely disjoint partitions).  The
current positive dst — and any other true positive destination of the same
(src, t) — are excluded from both branches.

Determinism: per-event seed ``(model_seed, epoch, global_event_row)``.

Efficiency (§1.1 #5): prefix structures are built once as per-src first-
occurrence tables (dst dedup for free); the hist branch slices one array, the
random branch uses bounded rejection sampling over the numpy universe — never
a per-event python rebuild of the whole universe list.

The sampler only ever serves the root link task: negatives never enter the
future index, boundary records, or any PRBE auxiliary window.
"""

import bisect
import hashlib
from typing import Dict, List, Sequence, Tuple

import numpy as np

_UNIVERSE_MAX_TRIES = 64


def _row_seed(model_seed: int, epoch: int, row: int) -> int:
    canonical = "histrand|{}|{}|{}".format(
        int(model_seed), int(epoch), int(row))
    return int.from_bytes(hashlib.sha256(canonical.encode("utf-8")).digest()
                          [:8], "little") % (2 ** 32)


class HistRandTrainSampler:
    """One negative per positive over a chronological train stream."""

    def __init__(self, sources: Sequence[int], destinations: Sequence[int],
                 timestamps: Sequence[float], model_seed: int = 0,
                 hist_p: float = 0.5, rng_seed: int = 0):
        self.sources = np.asarray(sources, dtype=np.int64)
        self.destinations = np.asarray(destinations, dtype=np.int64)
        self.timestamps = np.asarray(timestamps, dtype=np.float64)
        self.model_seed = int(model_seed)
        self.hist_p = float(hist_p)
        self.rng_seed = int(rng_seed)
        n = int(len(self.sources))
        # per-src FIRST-occurrence tables: fr_rows[s] ascending global rows at
        # which each distinct dst was first seen; fr_dst[s] parallel dst ids.
        # hist candidates of (s, r) = fr_dst[s][:bisect(fr_rows[s], r)] (dst
        # dedup is built in).  first_map[s] = {dst: first_row} for O(1) seen.
        fr_rows: Dict[int, List[int]] = {}
        fr_dst: Dict[int, List[int]] = {}
        first_map: Dict[int, Dict[int, int]] = {}
        seen: Dict[int, set] = {}
        for i in range(n):
            s = int(self.sources[i])
            d = int(self.destinations[i])
            if d not in seen.get(s, ()):
                fr_rows.setdefault(s, []).append(i)
                fr_dst.setdefault(s, []).append(d)
                first_map.setdefault(s, {})[d] = i
                seen.setdefault(s, set()).add(d)
        self._fr_rows = fr_rows
        self._fr_dst = fr_dst
        self._first_map = first_map
        # true positive dsts per (src, t)
        self._pos_by_st: Dict[Tuple[int, float], List[int]] = {}
        for i in range(n):
            k = (int(self.sources[i]), float(self.timestamps[i]))
            self._pos_by_st.setdefault(k, []).append(int(self.destinations[i]))
        self._universe = np.unique(self.destinations)

    # --------------------------------------------------------------- sampling
    def sample(self, epoch: int, row_lo: int,
               src: Sequence[int], dst_pos: Sequence[int],
               t: Sequence[float], global_rows: Sequence[int]) -> np.ndarray:
        """Sample one negative per positive; all ids INTERNAL (+1)."""
        del row_lo
        src = np.asarray(src, dtype=np.int64)
        dst_pos = np.asarray(dst_pos, dtype=np.int64)
        gr = np.asarray(global_rows, dtype=np.int64)
        neg = np.zeros(len(src), dtype=np.int64)
        for j in range(len(src)):
            s = int(src[j]); d = int(dst_pos[j]); tt = float(t[j])
            h = _row_seed(self.model_seed, epoch, int(gr[j]))
            rng = np.random.RandomState(int(h))
            excluded = set(self._pos_by_st.get((s, tt), []))
            use_hist = rng.rand() < self.hist_p
            hist = self._hist_dsts(s, int(gr[j]), excluded)
            if use_hist and hist:
                neg[j] = int(hist[int(rng.randint(len(hist)))])
            else:
                neg[j] = int(self._random_dst(rng, s, int(gr[j]),
                                              excluded, d))
        return neg

    def _hist_dsts(self, src: int, global_row: int,
                   excluded: set) -> List[int]:
        """Unique dsts of ``src`` first seen at events strictly before row."""
        rows = self._fr_rows.get(src)
        if not rows:
            return []
        k = bisect.bisect_left(rows, int(global_row))
        dsts = self._fr_dst[src]
        if not excluded:
            return dsts[:k]
        return [d for d in dsts[:k] if d not in excluded]

    def _random_dst(self, rng, src: int, global_row: int, excluded: set,
                    cur_pos: int) -> int:
        """Random dst in universe \\ (prefix-seen(src) U excluded U {cur})."""
        fm = self._first_map.get(src, {})
        uni = self._universe
        for _ in range(_UNIVERSE_MAX_TRIES):
            cand = int(uni[int(rng.randint(len(uni)))])
            if cand == cur_pos or cand in excluded:
                continue
            if cand in fm and fm[cand] < global_row:
                continue  # already-seen dst is reserved for the hist branch
            return cand
        # bounded rejection failed (pathological tiny universe): explicit mask
        keep = np.asarray([int(u) for u in uni if u != cur_pos
                           and u not in excluded
                           and not (u in fm and fm[u] < global_row)],
                          dtype=np.int64)
        if keep.size == 0:
            raise ValueError(
                "no legal random negative (src={} row={})".format(
                    src, global_row))
        return int(keep[int(rng.randint(keep.size))])
