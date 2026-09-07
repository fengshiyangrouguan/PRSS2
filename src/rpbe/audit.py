"""audit.py — read-only hash sidecar for the ablation comparison protocol.

Spec §26 item 13 requires a ``comparison_audit.json`` per run proving that
each arm changed ONLY its designated variable relative to R0.  This module
implements that sidecar.

All digests are built ONLY from structural integers (pair ids, event ids,
roles, rows) — never from floating-point tensors — so they are deterministic
across runs on the same commit/seed and directly comparable across arms.
The accumulator performs no tensor work and touches no training state:
it is a pure audit side-channel.

A2 acceptance relations (spec §24.5):

* data_flow / valid_cut / occurrence_order identical across the four arms;
* aligned and mispaired share the same y1/y2 multiset (population);
* aligned and mispaired differ in pairing (mispaired consumes donor futures);
* replay_pairing digest equals the pass-1 pairing digest (exact replay);
* task_only records the underlying population even though its objective
  consumes no Y at all (common data population across the four arms).
"""
from __future__ import annotations

import hashlib
from typing import Dict, List, Optional, Tuple


def _flatten(keys) -> list:
    """Recursively flatten tuple/list keys (pair_id may be a tuple)."""
    out = []
    for k in keys:
        if isinstance(k, (tuple, list)):
            out.extend(_flatten(k))
        else:
            out.append(int(k))
    return out


def _int_key(*ints: int) -> bytes:
    return "|".join(str(i) for i in _flatten(ints)).encode("utf-8")


def _digest8(key: bytes) -> int:
    return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(),
                          "little")


_M64 = (1 << 64) - 1


class _Stream:
    """Order-dependent blake2b stream."""

    def __init__(self, name: str):
        self.name = name
        self._h = hashlib.blake2b(digest_size=16)
        self.n = 0

    def add(self, *ints: int) -> None:
        self._h.update(_int_key(*ints))
        self._h.update(b"\x00")
        self.n += 1

    def hexdigest(self) -> str:
        return self._h.hexdigest()


class _Multiset:
    """Order-invariant counter: sum of per-key digests mod 2^64."""

    def __init__(self, name: str):
        self.name = name
        self._total = 0
        self.n = 0

    def add(self, *ints: int) -> None:
        self._total = (self._total + _digest8(_int_key(*ints))) & _M64
        self.n += 1

    def hexdigest(self) -> str:
        return "%016x" % self._total


class _Set:
    """Order-invariant unique-key counter."""

    def __init__(self, name: str):
        self.name = name
        self._seen = set()
        self._total = 0
        self.n = 0

    def add(self, *ints: int) -> None:
        k = _int_key(*ints)
        if k in self._seen:
            return
        self._seen.add(k)
        self._total = (self._total + _digest8(k)) & _M64
        self.n += 1

    def hexdigest(self) -> str:
        return "%016x" % self._total


def _event_ids(rec) -> Tuple[int, ...]:
    cf = rec.child_future
    pf = rec.parent_future
    y1 = (int(cf.event_id), int(cf.counterpart), int(cf.role),
          int(cf.message_idx)) if cf is not None else (-1,)
    y2 = (int(pf.event_id), int(pf.counterpart), int(pf.role),
          int(pf.message_idx)) if pf is not None else (-1,)
    return y1, y2


class AuditAccumulator:
    """Streaming structural hash sidecar (pure audit, zero training effect)."""

    def __init__(self) -> None:
        self.data_flow = _Stream("data_flow")
        self.valid_cut = _Set("valid_cut")
        self.occurrence = _Stream("occurrence_order")
        self.y1 = _Multiset("y1_multiset")
        self.y2 = _Multiset("y2_multiset")
        self.pairing = _Stream("pairing")
        self.closed = _Stream("closed_window")
        self.replay = _Stream("replay_pairing")
        # consumed parent future (donor for mispaired) per pair_id, filled
        # during pass 1 so pass 2 replay can reproduce the SAME pairing keys
        # (pair_id is a hashable tuple — used directly as the dict key)
        self._y2_consumed: Dict[tuple, int] = {}
        # pass-1 pairing insertion order + pass-2 replay rows; the replay
        # stream is replayed in pairing order at dump time so the two
        # order-dependent digests are directly comparable
        self._pairing_order: List[tuple] = []
        self._replay_rows: Dict[tuple, Tuple[int, int]] = {}
        self._replay_finalized = False
        self.n_population = 0
        self.n_pairing = 0

    # ------------------------------------------------------------- pass 1
    def add_population(self, rec) -> None:
        """One record of the shared surviving set (all four arms).

        Records the common population: data flow, valid-cut membership,
        occurrence order, and the underlying Y1/Y2 candidate multisets
        (aligned parent futures only — derangement is a pairing concern).
        """
        y1, y2 = _event_ids(rec)
        self.data_flow.add(rec.pair_id, int(rec.root_row))
        self.valid_cut.add(rec.pair_id)
        self.occurrence.add(rec.pair_id)
        if y1 != (-1,):
            self.y1.add(*y1)
        if y2 != (-1,):
            self.y2.add(*y2)
        self.n_population += 1

    def add_pairing(self, rec, consumed_parent_event_id: int) -> None:
        """One CONSUMED pairing (window row): pair_id -> (Y1, Y2_used).

        ``consumed_parent_event_id`` is the donor future for the mispaired
        arm and the true parent future for every other arm, so aligned and
        mispaired share y2_multiset but differ in pairing.
        """
        y1, _ = _event_ids(rec)
        y1_id = y1[0] if y1 != (-1,) else -1
        self.pairing.add(rec.pair_id, y1_id, int(consumed_parent_event_id))
        self._y2_consumed[rec.pair_id] = int(consumed_parent_event_id)
        self._pairing_order.append(rec.pair_id)
        self.n_pairing += 1

    def add_window_close(self, tau: str, pair_ids: List[tuple]) -> None:
        self.closed.add(_digest8(tau.encode("utf-8")) & 0x7FFFFFFF,
                        len(pair_ids))
        for pid in sorted(pair_ids, key=_int_key):
            self.closed.add(pid)

    # ------------------------------------------------------------- pass 2
    def add_replay(self, rec) -> None:
        """Collect one replayed pair in pass 2 (stream order differs from
        pass 1 — finalize_replay restores pairing order at dump time)."""
        y1, _ = _event_ids(rec)
        y1_id = y1[0] if y1 != (-1,) else -1
        y2_used = self._y2_consumed.get(rec.pair_id, -1)
        self._replay_rows[rec.pair_id] = (int(y1_id), int(y2_used))

    def _finalize_replay(self) -> None:
        if self._replay_finalized:
            return
        self._missing_replay: List[tuple] = []
        self._extra_replay: List[tuple] = []
        for pid in self._pairing_order:
            if pid not in self._replay_rows:
                self._missing_replay.append(pid)
            y1_id, y2_used = self._replay_rows.get(pid, (-1, -1))
            self.replay.add(pid, y1_id, y2_used)
        for pid in self._replay_rows:
            if pid not in self._y2_consumed:
                self._extra_replay.append(pid)
        self._replay_finalized = True

    # ------------------------------------------------------------- dump
    def dump(self, *, commit: str, config_hash: str,
             fixed_feature: Dict[str, object]) -> Dict[str, object]:
        self._finalize_replay()
        return {
            "commit": commit,
            "config_hash": config_hash,
            "fixed_feature": fixed_feature,
            "counts": {
                "population": self.n_population,
                "pairing": self.n_pairing,
                "valid_cut": self.valid_cut.n,
                "y1": self.y1.n,
                "y2": self.y2.n,
                "closed": self.closed.n,
                "replay": self.replay.n,
            },
            "digests": {
                "data_flow": self.data_flow.hexdigest(),
                "valid_cut": self.valid_cut.hexdigest(),
                "occurrence_order": self.occurrence.hexdigest(),
                "y1_multiset": self.y1.hexdigest(),
                "y2_multiset": self.y2.hexdigest(),
                "pairing": self.pairing.hexdigest(),
                "closed_window": self.closed.hexdigest(),
                "replay_pairing": self.replay.hexdigest(),
            },
            "replay_debug": {
                "n_missing": len(self._missing_replay),
                "n_extra": len(self._extra_replay),
                "first_missing": self._missing_replay[:5],
                "first_extra": self._extra_replay[:5],
            },
        }
