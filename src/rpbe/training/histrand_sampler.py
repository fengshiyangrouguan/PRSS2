"""Prefix-causal HistRand train negative sampler (Wiki-LR-Binary §2.3).

For each train positive event the root task needs one negative destination.
With probability 0.5 the negative is a *historical* destination of that src
(a dst the src has interacted with at an EARLIER train event); otherwise it
is drawn from the train destination universe.  The current positive dst —
and any other true positive destination of the same (src, t) — are excluded.

The sampler is deterministic in ``(model_seed, epoch, global_event_row)`` and
only ever serves the root link task: negatives never enter the future index,
boundary records, or any PRBE auxiliary window.
"""

import bisect
import hashlib
from typing import Dict, List, Sequence, Tuple

import numpy as np


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
        # per-src prefix structure: rows and dsts in chronological order
        rows_by_src: Dict[int, List[int]] = {}
        dst_by_src: Dict[int, List[int]] = {}
        for i in range(n):
            s = int(self.sources[i])
            rows_by_src.setdefault(s, []).append(i)
            dst_by_src.setdefault(s, []).append(int(self.destinations[i]))
        self._rows_by_src = rows_by_src
        self._dst_by_src = dst_by_src
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
        """Sample one negative per positive; all ids INTERNAL (+1).

        ``global_rows`` gives each event its global (model-wide) event row used
        for the deterministic seed; ``row_lo`` is the stream offset of the
        current batch (unused given global_rows, kept for interface clarity).
        """
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
                pool = [u for u in self._universe.tolist()
                        if u not in excluded]
                if not pool:  # degenerate tiny universe: fall back to any != d
                    pool = [u for u in self._universe.tolist() if u != d]
                neg[j] = int(pool[int(rng.randint(len(pool)))])
        return neg

    def _hist_dsts(self, src: int, global_row: int,
                   excluded: set) -> List[int]:
        """Historical dsts of ``src`` at events strictly before global_row."""
        rows = self._rows_by_src.get(src, [])
        if not rows:
            return []
        # strictly earlier (stream row < global_row), per prefix causality
        k = bisect.bisect_left(rows, int(global_row))
        dsts = self._dst_by_src[src]
        return [d for d in dsts[:k] if d not in excluded]
