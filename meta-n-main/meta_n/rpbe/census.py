"""Archive census for the Meta^n x RPBE port (task book v4.1 §4.3).

Runs BEFORE any training code. Answers the only question that decides whether
the frozen window gates are satisfiable at all:

  1. how many eligible two-step lineages the archive actually contains
  2. how many unique parents
  3. how many unique trees, keyed (run_id, root_candidate_id)
  4. how many complete window pairs N_TREE_MIN=32 permits

Reads only ``<run_dir>/archive/index.json`` (written by
``run_persistence.py:428`` as ``archive.to_dict()``) plus the per-candidate
``mean_score`` / ``per_task_scores`` it already contains. No torch, no Meta^n
imports -- this must run against a run directory on any machine.

A run's archive is a single chain (``gen0_seed`` root, ``beam_width=1``,
``beam_candidates=1``), so ``unique trees == number of runs`` that produced a
usable archive. That is the structural fact that drove N_TREE_MIN from 128 to
32; this script measures it instead of assuming it.

Usage:
    python -m meta_n.rpbe.census <run_dir> [<run_dir> ...]
    python -m meta_n.rpbe.census --root <parent_of_run_dirs>
    python -m meta_n.rpbe.census --self-test
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

N_TREE_MIN = 32            # formal-data TARGET (v4.2: a readiness status)
N_RUNS_BUDGET = 64         # Phase A run BUDGET, NOT a success requirement


# --------------------------------------------------------------------------
# archive parsing
# --------------------------------------------------------------------------

@dataclass
class Cand:
    candidate_id: str
    parent_id: Optional[str]
    depth: int
    mean_score: float
    per_task_scores: dict


@dataclass
class RunArchive:
    run_id: str
    path: Path
    candidates: list[Cand] = field(default_factory=list)

    @property
    def roots(self) -> list[str]:
        return [c.candidate_id for c in self.candidates if c.parent_id is None]


def load_run(run_dir: Path) -> RunArchive:
    """Load one run's archive index. Raises on a missing/unreadable index."""
    index = run_dir / "archive" / "index.json"
    if not index.is_file():
        # tolerate being pointed straight at the archive dir
        alt = run_dir / "index.json"
        if alt.is_file():
            index = alt
        else:
            raise FileNotFoundError("no archive index under {}".format(run_dir))
    with open(index, "r", encoding="utf-8") as f:
        blob = json.load(f)
    ra = RunArchive(run_id=run_dir.name, path=run_dir)
    for raw in blob.get("candidates", []):
        ra.candidates.append(Cand(
            candidate_id=str(raw["candidate_id"]),
            parent_id=(None if raw.get("parent_id") is None
                       else str(raw["parent_id"])),
            depth=int(raw.get("depth", 1)),
            mean_score=float(raw.get("mean_score", 0.0)),
            per_task_scores=dict(raw.get("per_task_scores") or {}),
        ))
    return ra


# --------------------------------------------------------------------------
# lineage / tree census
# --------------------------------------------------------------------------

def _finite(x: float) -> bool:
    return isinstance(x, float) and math.isfinite(x)


def eligible_lineages(ra: RunArchive) -> tuple[list[tuple[str, str, str]], dict]:
    """All real c_v -> c_{v+1} -> c_{v+2} lineages.

    A lineage is eligible only when both steps exist in the archive AND
    R_v = mean_score(c_{v+2}) - mean_score(c_v) is well defined (v4.1 §2.9):
    both endpoints finite and drawn from the identical task cohort.
    Returns (lineages, drop_counts).
    """
    by_id = {c.candidate_id: c for c in ra.candidates}
    children: dict[str, list[str]] = {}
    for c in ra.candidates:
        if c.parent_id is not None:
            children.setdefault(c.parent_id, []).append(c.candidate_id)

    out: list[tuple[str, str, str]] = []
    drops = {"no_child": 0, "no_grandchild": 0,
             "nonfinite_score": 0, "cohort_mismatch": 0}
    for c in ra.candidates:
        kids = children.get(c.candidate_id, [])
        if not kids:
            drops["no_child"] += 1
            continue
        progressed = False
        for ch in kids:
            gks = children.get(ch, [])
            if not gks:
                continue
            for gk in gks:
                grand = by_id[gk]
                if not (_finite(c.mean_score) and _finite(grand.mean_score)):
                    drops["nonfinite_score"] += 1
                    continue
                if set(c.per_task_scores) != set(grand.per_task_scores):
                    drops["cohort_mismatch"] += 1
                    continue
                out.append((c.candidate_id, ch, gk))
                progressed = True
        if not progressed and kids:
            drops["no_grandchild"] += 1
    return out, drops


def tree_id(run_id: str, root_id: str) -> tuple[str, str]:
    """(run_id, root_candidate_id) -- never a bare 'gen0_seed' (v4.1 §2.4)."""
    return (run_id, root_id)


def _root_of(cand_id: str, by_id: dict[str, Cand]) -> Optional[str]:
    seen = set()
    cur = cand_id
    while cur is not None and cur not in seen:
        seen.add(cur)
        c = by_id.get(cur)
        if c is None:
            return None
        if c.parent_id is None:
            return cur
        cur = c.parent_id
    return None


def fixed_hash_split(trees: Iterable[tuple[str, str]],
                     n_task: int = 32) -> tuple[list, list]:
    """Deterministic task/protect split (v4.1 §4.2).

    Ordering is a stable sha256 of the tree id, so the split is reproducible
    across machines and never re-cut on results.
    """
    uniq = sorted(set(trees))

    def key(t):
        return hashlib.sha256("|".join(t).encode("utf-8")).hexdigest()

    ordered = sorted(uniq, key=key)
    return ordered[:n_task], ordered[n_task:]


# --------------------------------------------------------------------------
# v4.2.1 split protocol -- per-tree fixed hash, NO salt (frozen 2026-09-18)
# --------------------------------------------------------------------------
#
# The legacy truncation split above is KEPT for reproducing historical results
# and for explicit control use. It is NOT the v4.2.1 path: it encodes `32` into
# the allocation mechanism, so with n < 33 trees the protect window is always
# empty, and `ordered[:32]` shifts as the set grows -- the same tree can change
# role from one batch to the next. See SPLIT_PROTOCOL_V4.2_DRAFT.md.

SPLIT_PROTOCOL = "tree_hash_v421"


def _canonical_tree_bytes(tree_id) -> bytes:
    """Unambiguous binary encoding of ``(run_id, root_candidate_id)``.

    LENGTH-PREFIXED, not delimiter-joined: a delimiter would be ambiguous if an
    id ever contained it, and unlike a string join this cannot be made to
    collide by naming. The encoding is FROZEN -- any change re-rolls every
    role, which is exactly the result-dependent tuning this protocol forbids.
    """
    run_b = str(tree_id[0]).encode("utf-8")
    root_b = str(tree_id[1]).encode("utf-8")
    return (len(run_b).to_bytes(4, "big") + run_b
            + len(root_b).to_bytes(4, "big") + root_b)


def tree_role_v421(tree_id) -> str:
    """The frozen role of ONE tree. Depends only on the tree's immutable id.

    There is deliberately NO salt. A tunable salt is a degree of freedom that
    invites choosing the one that yields a pleasing ratio; removing it removes
    the possibility. ``digest[0] & 1 == 0`` -> task, otherwise protect.

    Never adjust this function, its encoding, or its constant on results.
    """
    if len(tree_id) != 2 or not all(str(x) for x in tree_id):
        raise ValueError("tree_id must be a non-empty (run_id, root) pair")
    digest = hashlib.sha256(_canonical_tree_bytes(tree_id)).digest()
    return "task" if (digest[0] & 1) == 0 else "protect"


def partition_tree_roles_v421(trees):
    """``(task, protect)`` by per-tree role. Order-independent, set-stable.

    Invariants (SPLIT_PROTOCOL_V4.2_DRAFT.md §3.1):
      I1 role is a pure function of the tree's own immutable id;
      I2 it is decided before any score / cut / lineage / D / alpha / J is read
         -- this function takes ONLY tree ids and reads nothing else;
      I3 adding a tree never re-roles an existing one (no positional slicing);
      I4 no salt, no re-hash, no manual moving of trees to force a ratio.
    """
    task, protect = [], []
    for t in sorted({tuple(x) for x in trees}):
        (task if tree_role_v421(t) == "task" else protect).append(t)
    return task, protect


@dataclass
class CensusReport:
    n_runs: int
    n_candidates: int
    n_lineages: int
    n_unique_parents: int
    n_unique_trees: int
    tree_ids: list
    drops: dict
    per_run: list

    @property
    def window_pairs(self) -> int:
        return self.n_unique_trees // N_TREE_MIN

    @property
    def formal_data_ready(self) -> bool:
        """Readiness STATUS, not a training gate (v4.2 §1/§4.3).

        True when the census can supply the frozen full pair. A False here means
        'label this pilot / underpowered', never 'refuse to train'.
        """
        return self.n_unique_trees >= N_RUNS_BUDGET

    def as_dict(self) -> dict:
        task, protect = fixed_hash_split(self.tree_ids)
        return {
            "n_runs": self.n_runs,
            "n_candidates": self.n_candidates,
            "n_lineages": self.n_lineages,
            "n_unique_parents": self.n_unique_parents,
            "n_unique_trees": self.n_unique_trees,
            "n_window_pairs_at_min": self.window_pairs,
            "n_tree_min": N_TREE_MIN,
            "formal_data_ready": self.formal_data_ready,
            "split": {"task": len(task), "protect": len(protect),
                      "disjoint": not (set(task) & set(protect))},
            "drops": self.drops,
            "per_run": self.per_run,
        }


def census(run_dirs: list[Path]) -> CensusReport:
    trees: set = set()
    parents: set = set()
    n_lineages = 0
    n_candidates = 0
    drops_total = {"no_child": 0, "no_grandchild": 0,
                   "nonfinite_score": 0, "cohort_mismatch": 0}
    per_run = []
    for d in run_dirs:
        ra = load_run(d)
        by_id = {c.candidate_id: c for c in ra.candidates}
        lins, drops = eligible_lineages(ra)
        roots = ra.roots
        for r in roots:
            trees.add(tree_id(ra.run_id, r))
        for (a, _, _) in lins:
            parents.add((ra.run_id, a))
        n_lineages += len(lins)
        n_candidates += len(ra.candidates)
        for k, v in drops.items():
            drops_total[k] += v
        per_run.append({
            "run_id": ra.run_id,
            "n_candidates": len(ra.candidates),
            "n_roots": len(roots),
            "n_lineages": len(lins),
            "roots": roots,
        })
    return CensusReport(
        n_runs=len(run_dirs),
        n_candidates=n_candidates,
        n_lineages=n_lineages,
        n_unique_parents=len(parents),
        n_unique_trees=len(trees),
        tree_ids=sorted(trees),
        drops=drops_total,
        per_run=per_run,
    )


# --------------------------------------------------------------------------
# self test (no archive needed)
# --------------------------------------------------------------------------

def _write_fixture(root: Path, run_id: str, chain_len: int,
                   tasks: list[str]) -> Path:
    """A synthetic single-chain archive: root + chain_len-1 descendants."""
    run = root / run_id
    (run / "archive").mkdir(parents=True, exist_ok=True)
    cands = []
    prev = None
    for i in range(chain_len):
        cid = "gen0_seed" if i == 0 else "cand_{}".format(i)
        cands.append({
            "candidate_id": cid,
            "parent_id": prev,
            "iteration": i,
            "depth": i + 1,
            "mean_score": 0.1 * i,
            "pass_at_1": 0.0,
            "num_children": 0 if i == chain_len - 1 else 1,
            "temperature_used": 0.7,
            "total_tokens": 0,
            "per_task_scores": {t: 0.1 * i for t in tasks},
        })
        prev = cid
    (run / "archive" / "index.json").write_text(
        json.dumps({"size": len(cands), "candidates": cands}), encoding="utf-8")
    return run


def self_test() -> int:
    import tempfile
    tasks = ["t1", "t2", "t3"]
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        # 3 runs x 4-candidate chain -> 2 lineages each -> 6 total
        dirs = [_write_fixture(root, "run_{}".format(i), 4, tasks)
                for i in range(3)]
        rep = census(dirs)
        assert rep.n_runs == 3, rep.n_runs
        assert rep.n_unique_trees == 3, rep.n_unique_trees
        assert rep.n_lineages == 6, rep.n_lineages
        assert rep.n_unique_parents == 6, rep.n_unique_parents
        assert rep.window_pairs == 0, rep.window_pairs
        assert not rep.formal_data_ready

        # cohort mismatch must drop the lineage, not silently keep it
        bad = _write_fixture(root, "run_bad", 3, tasks)
        blob = json.loads((bad / "archive" / "index.json").read_text())
        blob["candidates"][-1]["per_task_scores"] = {"t1": 0.2}
        (bad / "archive" / "index.json").write_text(json.dumps(blob))
        rep2 = census([bad])
        assert rep2.n_lineages == 0, rep2.n_lineages
        assert rep2.drops["cohort_mismatch"] == 1, rep2.drops

        # tree ids must not collide across runs (the whole point of the tuple)
        ids = {t[1] for t in rep.tree_ids}
        assert ids == {"gen0_seed"}, ids
        assert rep.tree_ids == [("run_0", "gen0_seed"),
                                ("run_1", "gen0_seed"),
                                ("run_2", "gen0_seed")], rep.tree_ids

        # split is deterministic and disjoint
        many = [(("run_{}".format(i)), "gen0_seed") for i in range(64)]
        a1, b1 = fixed_hash_split(many)
        a2, b2 = fixed_hash_split(many)
        assert a1 == a2 and b1 == b2
        assert not (set(a1) & set(b1))
        assert len(a1) == 32 and len(b1) == 32
    print("census self-test OK")
    return 0


# --------------------------------------------------------------------------

def _find_runs(root: Path) -> list[Path]:
    return sorted(p for p in root.iterdir()
                  if p.is_dir() and (p / "archive" / "index.json").is_file())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dirs", nargs="*", type=Path)
    ap.add_argument("--root", type=Path,
                    help="parent directory containing run dirs")
    ap.add_argument("--json", type=Path, help="write the report as JSON here")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()

    dirs = list(args.run_dirs)
    if args.root:
        dirs += _find_runs(args.root)
    if not dirs:
        ap.error("give run dirs, --root, or --self-test")

    rep = census(dirs)
    d = rep.as_dict()
    print("runs                 : {}".format(d["n_runs"]))
    print("candidates           : {}".format(d["n_candidates"]))
    print("eligible lineages    : {}".format(d["n_lineages"]))
    print("unique parents       : {}".format(d["n_unique_parents"]))
    print("unique trees         : {}   (formal target: >= {} per window; "
          "readiness status only)".format(d["n_unique_trees"], N_TREE_MIN))
    print("complete window pairs: {}".format(d["n_window_pairs_at_min"]))
    print("split task/protect   : {} / {} (disjoint={})".format(
        d["split"]["task"], d["split"]["protect"], d["split"]["disjoint"]))
    print("drops                : {}".format(d["drops"]))
    if not rep.formal_data_ready:
        print("\nVERDICT: PILOT -- unique trees {} < {} for the full frozen "
              "task/protect pair.\n         This is a READINESS STATUS, not a "
              "training gate: do NOT pad by re-counting the same tree. Use the "
              "real counts and label the run 'pilot' (v4.2 §4.3).".format(
                  d["n_unique_trees"], N_RUNS_BUDGET))
    else:
        print("\nVERDICT: FORMAL-READY -- {} tree(s) per window available."
              .format(N_TREE_MIN))
    if args.json:
        args.json.write_text(json.dumps(d, indent=2), encoding="utf-8")
        print("wrote {}".format(args.json))
    return 0


if __name__ == "__main__":
    sys.exit(main())
