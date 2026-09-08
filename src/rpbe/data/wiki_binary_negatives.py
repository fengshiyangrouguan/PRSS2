"""Wiki-LR-Binary fixed-negative manifest (compact, shared across arms/seeds).

Every val/test positive event is paired with EXACTLY ONE negative, drawn from
the official TGB negative pool, preferred to be "historical" (a (src, dst)
pair that already occurred in the TRAIN positives).  Selection is a stable
hash over ``(protocol_seed, split, raw_event_row, src, dst_pos, t)`` so all
arms and model seeds read the SAME manifest and never re-sample.

The manifest is generated once by ``scripts/build_wiki_binary_negatives.py``
and stored as a compact JSON keyed by GLOBAL event row (0-based row in the
full chronological stream).  Internal edge id == global row + 1, so the
evaluator / trainer can look a negative up from a split event's ``edge_idxs``.

Only train positive events may ever appear in Y/P/aux windows; negatives are
root-task scoring targets only and never touch memory or the future index.
"""

import hashlib
import json
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def _stable_digest(protocol_seed: int, split: str, raw_event_row: int,
                   src: int, dst: int, t: float) -> int:
    """Deterministic int from the spec-§2.2 selection input tuple.

    Uses hashlib (immune to PYTHONHASHSEED); no float formatting ambiguity
    because ``t`` is embedded via repr with full precision.
    """
    canonical = "{}|{}|{}|{}|{}|{!r}".format(
        int(protocol_seed), split, int(raw_event_row), int(src), int(dst),
        float(t))
    return int.from_bytes(hashlib.sha256(canonical.encode("utf-8")).digest()
                          [:8], "little")


def _pick(negatives: np.ndarray, h: int) -> int:
    return int(negatives[int(h) % len(negatives)])


def build_compact_negatives(ds, protocol_seed: int = 20260908,
                            splits: Sequence[str] = ("val", "test"),
                            max_check: int = 0) -> Dict:
    """Select one fixed negative per val/test positive; return report.

    ``ds`` is a :class:`rpbe.data.tgb_link.TGBLinkDataset`.  Requires the
    official negative pools (``ds.load_val_ns()`` / ``load_test_ns()``) to be
    loadable.  Raw ids throughout (0-based TGB node ids).

    Returns the manifest-plus-report dict (see caller for persistence) and
    applies the §2.2 collision gate: chosen neg != dst_pos, and never equals
    any true positive destination of the same (src, t).
    """
    # directed train positive edge set (raw ids) for the historical test
    train_src = ds.train.sources - 1
    train_dst = ds.train.destinations - 1
    train_edges = set(zip(train_src.tolist(), train_dst.tolist()))
    # true positive dst per (src, t) — multiple rows can share (src, t)
    pos_by_st: Dict[Tuple[int, float], List[int]] = {}

    manifest: Dict[str, Dict[str, object]] = {}
    counts = {"val_hist": 0, "val_random": 0, "test_hist": 0,
              "test_random": 0, "collisions": 0}
    for split in splits:
        raw_src, raw_dst, raw_t = ds.raw_split(split)
        n = len(raw_src)
        if split == "val":
            ds.load_val_ns()
        else:
            ds.load_test_ns()
        cands = ds.query_negatives(
            np.asarray(raw_src, dtype=np.int64),
            np.asarray(raw_dst, dtype=np.int64),
            np.asarray(raw_t, dtype=np.float64), split_mode=split)
        # official sampler returns per-sample candidate arrays
        per_pos = cands
        if isinstance(cands, dict):
            per_pos = cands.get("neg_dst", cands.get("candidate", []))
        rows: Dict[str, object] = {}
        for i in range(n):
            src = int(raw_src[i]); dst = int(raw_dst[i]); t = float(raw_t[i])
            gid = "{}|{}".format(split, i)
            key = (src, t)
            pos_by_st.setdefault(key, []).append(dst)
        # second pass needs the full per-(src,t) positive sets; keep cands list
        for i in range(n):
            src = int(raw_src[i]); dst = int(raw_dst[i]); t = float(raw_t[i])
            row_global = int(_global_row(ds, split, i))
            gid = str(row_global)
            negs = np.asarray(per_pos[i], dtype=np.int64)
            if negs.ndim == 0:
                negs = negs.reshape(1)
            h = _stable_digest(protocol_seed, split, i, src, dst, t)
            hist = [int(x) for x in negs if (src, int(x)) in train_edges]
            pool = hist if hist else [int(x) for x in negs]
            chosen = _pick(np.asarray(pool, dtype=np.int64), h)
            ntype = "hist" if hist else "random"
            counts[split + "_" + ntype] += 1
            # collision gate: not the positive dst, and not any true positive
            # dst at the same (src, t)
            if chosen == dst or chosen in pos_by_st[(src, t)]:
                counts["collisions"] += 1
                # fall back deterministically to the first legal candidate
                legal = [int(x) for x in negs if int(x) != dst
                         and int(x) not in pos_by_st[(src, t)]]
                if not legal:
                    raise ValueError(
                        "no legal negative for split={} row={}".format(
                            split, i))
                chosen = legal[int(h) % len(legal)]
            rows[gid] = {"neg": int(chosen), "type": ntype}
        manifest[split] = rows
    manifest["_meta"] = {
        "protocol_seed": int(protocol_seed),
        "counts": counts,
        "dataset": ds.name,
    }
    return manifest


def _global_row(ds, split: str, i: int) -> int:
    """Global (full-stream) 0-based row for split event i."""
    split_obj = {"val": ds.val, "test": ds.test, "train": ds.train}[split]
    return int(split_obj.edge_idxs[i]) - 1


def manifest_sha256(payload: dict) -> str:
    """Stable SHA-256 of the manifest content (meta counts excluded)."""
    body = {k: v for k, v in payload.items() if k != "_meta"}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class CompactNegatives:
    """Read-only lookup of fixed negatives keyed by global event row."""

    def __init__(self, path: str):
        with open(path) as f:
            self._data = json.load(f)
        self.meta = self._data.get("_meta", {})
        self._rows = {k: v for k, v in self._data.items() if k != "_meta"}

    def neg_for(self, split: str, global_row: int) -> Optional[int]:
        """Raw (0-based) negative dst for a global row, or None."""
        row = self._rows.get(split, {}).get(str(int(global_row)))
        return None if row is None else int(row["neg"])

    def type_for(self, split: str, global_row: int) -> Optional[str]:
        row = self._rows.get(split, {}).get(str(int(global_row)))
        return None if row is None else str(row["type"])

    def counts(self) -> Dict:
        return dict(self.meta.get("counts", {}))
