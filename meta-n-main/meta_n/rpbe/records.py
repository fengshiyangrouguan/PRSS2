"""Cut records for the Meta^n x RPBE port (task book §2.4).

This module records events that ALREADY happened. It deliberately does NOT
select, sketch the future, reduce context, or train -- those are `selector`,
`future`, `context_reduction` and `trainer`. Keeping that boundary sharp is
what stops a "recording bug" being mistaken for a "model bug".

SCHEMA (two independent keys -- neither substitutes for the other):

    CutMeta                       provenance of the cut's own candidate
        candidate_id              run-qualified candidate identity
        depth                     recursion layer d
        run_id                    which Omega run
        root_candidate_id         lineage root (tree identity)

    CutRecord
        cut_id          (candidate_id, occurrence_seq)   "which candidate"
        tree_id         (run_id, root_candidate_id)      unique-tree gate
        occurrence_seq  int
        X_v             [n_v, 256]   frozen encoder output
        q_emb           [256]        frozen query embedding
        mask            [4, n_v]     padding-only additive mask
        p               [4, 8]       detached future sketch, one row per branch
        r               float        two-step REAL return
        weight          float        parent-normalised weight

`cut_id` answers "which candidate produced this cut". `tree_id` answers "which
independent history does it belong to" and drives the window gate. Parent /
child / grandchild relationships are NOT stored here -- `lineage.py` owns them.

CANDIDATE IDENTITY IS RUN-QUALIFIED. The orchestrator's own id ("gen0_seed",
"gen1_b0_k0") is NOT unique across runs: every run mints its own "gen0_seed".
Use :func:`make_candidate_id` to build the qualified form. A bare orchestrator
id is rejected, because it would collapse every run into one candidate -- the
same class of bug as a bare `tree_id`.

Invariants enforced here:
  * no tensor may carry a graph -- a record is a snapshot, not a live node;
  * `p` is [N_BRANCHES, M_SKETCH]; branches stay separate, never concatenated;
  * `r` must be finite: a dropped return is a DROPPED cut, not a NaN record.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from meta_n.rpbe import config as C

TreeId = Tuple[str, str]        # (run_id, root_candidate_id)
CutId = Tuple[str, int]         # (candidate_id, occurrence_seq)

NEG_INF = float("-inf")
QUALIFIER = ":"                 # run-qualifier separator in candidate ids


def make_candidate_id(run_id: str, candidate_id: str) -> str:
    """Run-qualified candidate identity, e.g. ``run_0:gen1_b0_k0``.

    Returns the input unchanged if it is already qualified (idempotent), so it
    is safe to apply at every construction site.
    """
    run_id, candidate_id = str(run_id), str(candidate_id)
    if not run_id or not candidate_id:
        raise ValueError("run_id and candidate_id must both be non-empty")
    if candidate_id.startswith(run_id + QUALIFIER):
        return candidate_id
    return run_id + QUALIFIER + candidate_id


def parse_candidate_id(candidate_id: str) -> Tuple[Optional[str], str]:
    """Canonical inverse of :func:`make_candidate_id`.

    Returns ``(run_id, bare_candidate_id)``; ``run_id`` is None when the id is
    unqualified. Every module that needs to split a candidate id MUST use this
    -- hand-rolled ``f"{run_id}:{bare}"`` / ``f"{run_id}/{bare}"`` formatting
    drifts and silently breaks the identity space.
    """
    s = str(candidate_id)
    if QUALIFIER in s:
        run_id, bare = s.split(QUALIFIER, 1)
        return run_id, bare
    return None, s


@dataclass(frozen=True)
class CutMeta:
    """Provenance of the candidate a cut came from. No lineage, no tensors."""

    candidate_id: str            # run-qualified (see make_candidate_id)
    depth: int                   # recursion layer d
    run_id: str
    root_candidate_id: str       # lineage root within this run

    def __post_init__(self) -> None:
        q_run, bare = parse_candidate_id(self.candidate_id)
        if q_run is None:
            raise ValueError(
                "candidate_id {!r} is not run-qualified. The orchestrator's "
                "own ids ('gen0_seed') repeat in every run, so a bare id would "
                "collapse all runs into one candidate. Build it with "
                "make_candidate_id(run_id, candidate_id).".format(
                    self.candidate_id))
        if q_run != self.run_id:
            raise ValueError(
                "candidate_id {!r} is qualified with run {!r}, but run_id is "
                "{!r}".format(self.candidate_id, q_run, self.run_id))
        if int(self.depth) < 0:
            raise ValueError("depth must be >= 0")
        if not self.root_candidate_id:
            raise ValueError("root_candidate_id must be non-empty")
        object.__setattr__(self, "depth", int(self.depth))

    @property
    def bare_candidate_id(self) -> str:
        return parse_candidate_id(self.candidate_id)[1]

    def to_dict(self) -> Dict[str, Any]:
        return {"candidate_id": self.candidate_id, "depth": self.depth,
                "run_id": self.run_id,
                "root_candidate_id": self.root_candidate_id}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CutMeta":
        return cls(candidate_id=d["candidate_id"], depth=int(d["depth"]),
                   run_id=d["run_id"],
                   root_candidate_id=d["root_candidate_id"])


@dataclass(eq=False)
class CutRecord:
    """One RPBE cut: a snapshot of what was observed, with no live graph.

    `eq=False` because torch tensors do not compare with `==`; use
    :meth:`fingerprint` for equality.
    """

    meta: CutMeta
    cut_id: CutId
    tree_id: TreeId
    occurrence_seq: int
    X_v: torch.Tensor
    q_emb: torch.Tensor
    mask: torch.Tensor
    p: torch.Tensor
    r: float
    weight: float

    # -- construction ------------------------------------------------------
    def __post_init__(self) -> None:
        if not isinstance(self.meta, CutMeta):
            raise TypeError("meta must be a CutMeta, got {}".format(
                type(self.meta).__name__))
        self.cut_id = _as_cut_id(self.cut_id)
        self.tree_id = _as_tree_id(self.tree_id)
        self.occurrence_seq = int(self.occurrence_seq)
        self.r = float(self.r)
        self.weight = float(self.weight)
        self.validate()

    def validate(self) -> "CutRecord":
        """Assert every schema invariant. Returns self for chaining."""
        # Key identity: cut_id names the candidate; tree_id names the history.
        # They are INDEPENDENT (a cut's candidate need not be the tree root).
        if self.cut_id[0] != self.meta.candidate_id:
            raise ValueError(
                "cut_id[0] {!r} must equal meta.candidate_id {!r}".format(
                    self.cut_id[0], self.meta.candidate_id))
        if self.occurrence_seq != self.cut_id[1]:
            raise ValueError("occurrence_seq {} != cut_id[1] {}".format(
                self.occurrence_seq, self.cut_id[1]))
        if len(self.tree_id) != 2 or not all(
                isinstance(x, str) and x for x in self.tree_id):
            raise ValueError("tree_id must be a non-empty (run_id, root) pair")
        if self.tree_id[0] != self.meta.run_id:
            raise ValueError(
                "tree_id[0] {!r} must equal meta.run_id {!r}".format(
                    self.tree_id[0], self.meta.run_id))
        if self.tree_id[1] != self.meta.root_candidate_id:
            raise ValueError(
                "tree_id[1] {!r} must equal meta.root_candidate_id {!r}".format(
                    self.tree_id[1], self.meta.root_candidate_id))

        for name in ("X_v", "q_emb", "mask", "p"):
            t = getattr(self, name)
            if not torch.is_tensor(t):
                raise TypeError("{} must be a tensor, got {}".format(
                    name, type(t).__name__))
            if t.requires_grad:
                raise ValueError(
                    "{} carries a graph; a CutRecord is a detached snapshot "
                    "(no graph)".format(name))

        if self.X_v.dim() != 2 or self.X_v.shape[1] != C.D_E:
            raise ValueError("X_v must be [n_v, {}], got {}".format(
                C.D_E, tuple(self.X_v.shape)))
        n_v = int(self.X_v.shape[0])
        if self.q_emb.shape != (C.D_E,):
            raise ValueError("q_emb must be [{}], got {}".format(
                C.D_E, tuple(self.q_emb.shape)))
        if self.mask.shape != (C.N_SLOTS, n_v):
            raise ValueError("mask must be [{}, {}], got {}".format(
                C.N_SLOTS, n_v, tuple(self.mask.shape)))
        if self.p.shape != (C.N_BRANCHES, C.M_SKETCH):
            raise ValueError("p must be [{}, {}], got {}".format(
                C.N_BRANCHES, C.M_SKETCH, tuple(self.p.shape)))
        for name in ("X_v", "q_emb", "p"):
            if getattr(self, name).dtype != torch.float32:
                raise ValueError("{} must be float32, got {}".format(
                    name, getattr(self, name).dtype))
        if not torch.isfinite(self.p).all():
            raise ValueError("p must be finite (a sketch, not a gate)")
        if not math.isfinite(self.r):
            raise ValueError("r must be finite; a dropped return is a DROPPED "
                             "cut, not a NaN record")
        if not (math.isfinite(self.weight) and self.weight > 0.0):
            raise ValueError("weight must be a positive finite float")
        return self

    # -- provenance --------------------------------------------------------
    @property
    def candidate_id(self) -> str:
        return self.meta.candidate_id

    @property
    def run_id(self) -> str:
        return self.meta.run_id

    @property
    def root_candidate_id(self) -> str:
        return self.meta.root_candidate_id

    @property
    def depth(self) -> int:
        return self.meta.depth

    @property
    def n_v(self) -> int:
        return int(self.X_v.shape[0])

    def provenance(self) -> Dict[str, Any]:
        """Everything the record alone answers about where it came from.

        Parent / child / grandchild are NOT here: `lineage.py` owns those.
        """
        return {
            "candidate_id": self.meta.candidate_id,
            "bare_candidate_id": self.meta.bare_candidate_id,
            "depth": self.meta.depth,
            "run_id": self.meta.run_id,
            "root_candidate_id": self.meta.root_candidate_id,
            "occurrence_seq": self.occurrence_seq,
            "n_v": self.n_v,
        }

    # -- serialisation -----------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe dict. `mask` is stored as a boolean validity matrix
        because its padding entry is -inf, which JSON cannot represent; the
        additive mask is rebuilt on load, so the round-trip is exact."""
        return {
            "meta": self.meta.to_dict(),
            "cut_id": [self.cut_id[0], self.cut_id[1]],
            "tree_id": [self.tree_id[0], self.tree_id[1]],
            "occurrence_seq": self.occurrence_seq,
            "X_v": _tensor_to_nested(self.X_v),
            "q_emb": _tensor_to_nested(self.q_emb),
            "mask_valid": _mask_to_valid(self.mask),
            "p": _tensor_to_nested(self.p),
            "r": self.r,
            "weight": self.weight,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CutRecord":
        valid = torch.tensor(d["mask_valid"], dtype=torch.bool)
        mask = torch.zeros(valid.shape, dtype=torch.float32).masked_fill(
            ~valid, NEG_INF)
        return cls(
            meta=CutMeta.from_dict(d["meta"]),
            cut_id=(d["cut_id"][0], int(d["cut_id"][1])),
            tree_id=(d["tree_id"][0], d["tree_id"][1]),
            occurrence_seq=int(d["occurrence_seq"]),
            X_v=_nested_to_tensor(d["X_v"]),
            q_emb=_nested_to_tensor(d["q_emb"]),
            mask=mask,
            p=_nested_to_tensor(d["p"]),
            r=float(d["r"]),
            weight=float(d["weight"]),
        )

    # -- identity ----------------------------------------------------------
    def fingerprint(self) -> str:
        """Deterministic content hash; two extractions must agree here."""
        h = hashlib.sha256()
        h.update(json.dumps(
            {"meta": self.meta.to_dict(), "cut_id": list(self.cut_id),
             "tree_id": list(self.tree_id), "occ": self.occurrence_seq,
             "r": self.r, "weight": self.weight},
            sort_keys=True).encode("utf-8"))
        for name in ("X_v", "q_emb", "mask", "p"):
            h.update(_tensor_bytes(getattr(self, name)))
        return h.hexdigest()


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _as_tree_id(x) -> TreeId:
    if isinstance(x, str):
        raise TypeError(
            "tree_id must be a (run_id, root_candidate_id) tuple, not the bare "
            "string {!r}: 'gen0_seed' is hard-coded in the orchestrator and "
            "would collapse every run into a single tree".format(x))
    seq = tuple(x)
    if len(seq) != 2:
        raise ValueError("tree_id must have 2 elements, got {}".format(len(seq)))
    return (str(seq[0]), str(seq[1]))


def _as_cut_id(x) -> CutId:
    seq = tuple(x)
    if len(seq) != 2:
        raise ValueError("cut_id must have 2 elements (candidate_id, "
                         "occurrence_seq), got {}".format(len(seq)))
    if isinstance(seq[1], str) or not isinstance(seq[0], str):
        raise ValueError("cut_id must be (candidate_id: str, occurrence: int)")
    return (str(seq[0]), int(seq[1]))


def _tensor_bytes(t: torch.Tensor) -> bytes:
    a = t.detach().cpu().contiguous()
    return (struct.pack("<B", {torch.float32: 0, torch.float64: 1,
                               torch.bool: 2}.get(a.dtype, 9)) +
            struct.pack("<H", a.dim()) +
            b"".join(struct.pack("<I", s) for s in a.shape) +
            a.numpy().tobytes())


def _tensor_to_nested(t: torch.Tensor):
    return t.detach().cpu().tolist()


def _nested_to_tensor(x):
    return torch.tensor(x, dtype=torch.float32)


def _mask_to_valid(mask: torch.Tensor):
    return (~torch.isinf(mask)).tolist()


# --------------------------------------------------------------------------
# collection helpers
# --------------------------------------------------------------------------

def sort_records(records: Sequence[CutRecord]) -> List[CutRecord]:
    """Deterministic order: (run, candidate, occurrence) then fingerprint."""
    return sorted(records, key=lambda r: (r.run_id, r.candidate_id,
                                          r.occurrence_seq, r.fingerprint()))


def archive_fingerprint(records: Sequence[CutRecord]) -> str:
    h = hashlib.sha256()
    for r in sort_records(records):
        h.update(r.fingerprint().encode("ascii"))
    return h.hexdigest()


def save_jsonl(path: str, records: Sequence[CutRecord]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in sort_records(records):
            f.write(json.dumps(r.to_dict(), sort_keys=True,
                               separators=(",", ":")) + "\n")


def load_jsonl(path: str) -> List[CutRecord]:
    out: List[CutRecord] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(CutRecord.from_dict(json.loads(line)))
    return out


# --------------------------------------------------------------------------
# acceptance self-test -- deterministic / round-trip / no-leakage / zero-cost
# --------------------------------------------------------------------------

def _fixture(seed: int = 0, runs: Tuple[str, ...] = ("run_0", "run_1"),
             chain: int = 3) -> List[CutRecord]:
    """A synthetic archive shaped like the real AlgoTune ones: one root per
    run, a single chain of candidates, one cut per candidate."""
    recs: List[CutRecord] = []
    g = torch.Generator().manual_seed(seed)
    for run in runs:
        root_bare = "gen0_seed"
        for occ in range(chain):
            bare = root_bare if occ == 0 else "gen{}_b0_k0".format(occ)
            n_v = 5 + occ
            valid = torch.ones(C.N_SLOTS, n_v, dtype=torch.bool)
            valid[:, -1] = False                       # one padded item
            recs.append(CutRecord(
                meta=CutMeta(candidate_id=make_candidate_id(run, bare),
                             depth=occ, run_id=run,
                             root_candidate_id=root_bare),
                cut_id=(make_candidate_id(run, bare), occ),
                tree_id=(run, root_bare),
                occurrence_seq=occ,
                X_v=torch.randn(n_v, C.D_E, generator=g),
                q_emb=torch.randn(C.D_E, generator=g),
                mask=torch.zeros(C.N_SLOTS, n_v).masked_fill(~valid, NEG_INF),
                p=torch.randn(C.N_BRANCHES, C.M_SKETCH, generator=g),
                r=0.1 * occ, weight=1.0,
            ))
    return recs


def self_test() -> int:
    import tempfile

    print("records.py acceptance")
    print("=" * 68)

    # 1 -- DETERMINISTIC
    a, b = _fixture(seed=0), _fixture(seed=0)
    fa, fb = archive_fingerprint(a), archive_fingerprint(b)
    assert fa == fb, (fa, fb)
    print("OK  deterministic      same archive -> identical fingerprint {}"
          .format(fa[:12]))

    # 2 -- ROUND-TRIP
    with tempfile.TemporaryDirectory() as td:
        p1, p2 = os.path.join(td, "a.jsonl"), os.path.join(td, "b.jsonl")
        save_jsonl(p1, a)
        back = load_jsonl(p1)
        save_jsonl(p2, back)
        assert open(p1, "rb").read() == open(p2, "rb").read()
        assert archive_fingerprint(back) == fa
    print("OK  round-trip         reload + re-save is byte-identical")

    # 3 -- NO LEAKAGE + schema guards
    for r in a:
        assert not any(t.requires_grad for t in (r.X_v, r.q_emb, r.mask, r.p))
    val = a[1].provenance()
    assert val["candidate_id"] == "run_0:gen1_b0_k0", val
    assert val["bare_candidate_id"] == "gen1_b0_k0" and val["depth"] == 1
    print("OK  provenance         candidate_id={} depth={} root={}".format(
        val["candidate_id"], val["depth"], val["root_candidate_id"]))

    def rejects(label, **kw):
        base = dict(meta=a[0].meta, cut_id=a[0].cut_id, tree_id=a[0].tree_id,
                    occurrence_seq=0, X_v=torch.randn(5, C.D_E),
                    q_emb=torch.randn(C.D_E), mask=torch.zeros(C.N_SLOTS, 5),
                    p=torch.randn(C.N_BRANCHES, C.M_SKETCH), r=0.0, weight=1.0)
        base.update(kw)
        try:
            CutRecord(**base)
        except Exception:                                       # noqa: BLE001
            return
        raise AssertionError("accepted an invalid record: " + label)

    rejects("graph-carrying X_v", X_v=torch.randn(5, C.D_E).requires_grad_(True))
    rejects("NaN return", r=float("nan"))
    rejects("bare-string tree_id", tree_id="gen0_seed")
    rejects("p with shape [8]", p=torch.randn(C.M_SKETCH))
    rejects("cut_id disagreeing with meta",
            cut_id=("run_0:OTHER", 0))
    print("OK  schema guards      graph / NaN-r / bare tree_id / p=[8] / key "
          "mismatch rejected")

    try:
        CutMeta(candidate_id="gen0_seed", depth=0, run_id="run_0",
                root_candidate_id="gen0_seed")
        raise AssertionError("unqualified candidate_id accepted")
    except ValueError as e:
        assert "run-qualified" in str(e)
    print("OK  candidate id       bare 'gen0_seed' rejected as run-ambiguous")

    # 4 -- ZERO COST
    paid = 0
    led = os.environ.get("META_N_REQUEST_LEDGER", "").strip()
    if led and os.path.isfile(led):
        from meta_n.rpbe.accounting import snapshot
        paid = snapshot()["backend_requests"]
    assert paid == 0, "paid backend requests observed: {}".format(paid)
    print("OK  zero-cost          paid backend_requests = {}".format(paid))

    print()
    print("VERDICT: ALL OK")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(self_test())
