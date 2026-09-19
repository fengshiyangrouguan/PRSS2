"""Two-step REAL lineage extraction (task book v4.1 §2.5).

The future observations used by RPBE must be real: a parent candidate's child
and grandchild have to exist in the archive, evaluated on the same task cohort.
Nothing here is invented -- no zero-fill, no copying the child into the
grandchild slot, no synthetic EOS observation. A candidate without a real
grandchild simply produces no lineage, and therefore no cut (v4.1 §2.5).

    c_v  ->  c_{v+1}  ->  c_{v+2}
      Y_{v+1} = Phi(T_{d+1})     the CHILD's real trace observations
      Y_{v+2} = Phi(T_{d+2})     the GRANDCHILD's real trace observations
      R_v     = mean_score(c_{v+2}) - mean_score(c_v)          (§2.9)

`Y` is a trace observation, NOT InjectedCode.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from meta_n.core.meta_layer import Trace
from meta_n.rpbe.records import make_candidate_id, parse_candidate_id

_TRACE_FIELDS = {
    "task_id", "depth", "script", "stdout", "stderr", "exit_code", "success",
    "score", "reasoning", "duration_s", "error_summary", "eval_feedback",
    "failure_class", "terminated_by",
}


@dataclass(frozen=True)
class Lineage:
    """One real parent -> child -> grandchild occurrence."""

    run_id: str
    candidate_id: str            # c_v, run-qualified
    depth: int                   # recursion layer of c_v
    root_candidate_id: str       # lineage root (tree identity, bare id)
    child_id: str                # c_{v+1}, run-qualified
    grandchild_id: str           # c_{v+2}, run-qualified
    child_traces: Tuple[Trace, ...]        # Y_{v+1}
    grandchild_traces: Tuple[Trace, ...]   # Y_{v+2}
    r: float                     # two-step REAL return
    weight: float                # parent-normalised
    parent_score: float
    grandchild_score: float

    @property
    def tree_id(self) -> Tuple[str, str]:
        return (self.run_id, self.root_candidate_id)


# --------------------------------------------------------------------------
# archive loading
# --------------------------------------------------------------------------

def _candidate_dirs(run_dir: Path) -> List[Path]:
    arch = run_dir / "archive"
    if not arch.is_dir():
        raise FileNotFoundError("no archive/ under {}".format(run_dir))
    return sorted(p for p in arch.iterdir()
                  if p.is_dir() and (p / "summary.json").is_file())


def load_candidates(run_dir: Path) -> Dict[str, Dict[str, Any]]:
    """candidate_id -> summary dict (bare ids, as the orchestrator wrote them)."""
    out: Dict[str, Dict[str, Any]] = {}
    for d in _candidate_dirs(run_dir):
        with open(d / "summary.json", "r", encoding="utf-8") as f:
            s = json.load(f)
        out[str(s.get("candidate_id", d.name))] = s
    return out


def load_traces(run_dir: Path, bare_candidate_id: str) -> Tuple[Trace, ...]:
    """The candidate's real trace observations, in a deterministic order.

    Trace JSON on disk carries a few extra bookkeeping keys the Trace model
    does not declare, so the payload is filtered to the declared fields.
    """
    tdir = run_dir / "archive" / bare_candidate_id / "traces"
    if not tdir.is_dir():
        return ()
    out: List[Trace] = []
    for p in sorted(tdir.glob("*.json")):
        with open(p, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, list):
            items = raw
        else:
            items = [raw]
        for item in items:
            out.append(Trace(**{k: v for k, v in item.items()
                                if k in _TRACE_FIELDS}))
    return tuple(out)


# --------------------------------------------------------------------------
# extraction
# --------------------------------------------------------------------------

def extract_lineages(
    run_dir: os.PathLike | str,
    *,
    require_same_cohort: bool = True,
) -> List[Lineage]:
    """Every real c_v -> c_{v+1} -> c_{v+2} in one run's archive.

    `require_same_cohort` enforces §2.9: both endpoints of R_v must have been
    scored on the identical task cohort, and both means must be finite. A cut
    that fails either check is DROPPED (never recorded as a NaN return).
    """
    run_dir = Path(run_dir)
    run_id = run_dir.name
    cands = load_candidates(run_dir)

    children: Dict[str, List[str]] = {}
    for cid, s in cands.items():
        p = s.get("parent_id")
        if p:
            children.setdefault(str(p), []).append(str(cid))
    for k in children:
        children[k] = sorted(children[k])

    def root_of(cid: str) -> Optional[str]:
        seen = set()
        cur = cid
        while cur is not None and cur not in seen:
            seen.add(cur)
            s = cands.get(cur)
            if s is None:
                return None
            if not s.get("parent_id"):
                return cur
            cur = str(s["parent_id"])
        return None

    # First pass: enumerate every real two-step path, keyed by its parent.
    by_parent: Dict[str, List[Tuple[str, str]]] = {}
    for cid, s in sorted(cands.items()):
        for ch in children.get(cid, []):
            for gk in children.get(ch, []):
                sc, sg = cands.get(cid), cands.get(gk)
                if sc is None or sg is None:
                    continue
                mc = float(sc.get("mean_score", 0.0))
                mg = float(sg.get("mean_score", 0.0))
                if not (math.isfinite(mc) and math.isfinite(mg)):
                    continue
                if require_same_cohort and (
                        set(sc.get("per_task_scores") or {}) !=
                        set(sg.get("per_task_scores") or {})):
                    continue
                by_parent.setdefault(cid, []).append((ch, gk))

    out: List[Lineage] = []
    for cid in sorted(by_parent):
        paths = by_parent[cid]
        # §2.5: one occurrence per real path, weighted by PARENT so a parent
        # with many children is not amplified relative to a parent with one.
        w = 1.0 / float(len(paths))
        root = root_of(cid)
        if root is None:
            continue
        sc = cands[cid]
        depth = int(sc.get("depth", 1) or 1) - 1
        for (ch, gk) in paths:
            sg = cands[gk]
            out.append(Lineage(
                run_id=run_id,
                candidate_id=make_candidate_id(run_id, cid),
                depth=depth,
                root_candidate_id=root,
                child_id=make_candidate_id(run_id, ch),
                grandchild_id=make_candidate_id(run_id, gk),
                child_traces=load_traces(run_dir, ch),
                grandchild_traces=load_traces(run_dir, gk),
                r=float(sg.get("mean_score", 0.0)) -
                  float(sc.get("mean_score", 0.0)),
                weight=w,
                parent_score=float(sc.get("mean_score", 0.0)),
                grandchild_score=float(sg.get("mean_score", 0.0)),
            ))
    return out


def group_by_tree(lineages: Iterable[Lineage]) -> Dict[Tuple[str, str],
                                                      List[Lineage]]:
    """tree_id -> its lineages. Tree identity is (run_id, root), never bare."""
    out: Dict[Tuple[str, str], List[Lineage]] = {}
    for ln in lineages:
        out.setdefault(ln.tree_id, []).append(ln)
    return out


# --------------------------------------------------------------------------
# acceptance self-test
# --------------------------------------------------------------------------

def _write_run(root: Path, run_id: str, spec: Dict[str, Any],
               scores: Dict[str, float],
               cohort: Sequence[str] = ("t1", "t2")) -> Path:
    """Materialise a synthetic archive. `spec` maps id -> parent id or None."""
    run = root / run_id
    (run / "archive").mkdir(parents=True, exist_ok=True)
    for cid, parent in spec.items():
        d = run / "archive" / cid
        (d / "traces").mkdir(parents=True, exist_ok=True)
        s = {"candidate_id": cid, "parent_id": parent,
             "depth": 1 if parent is None else 2,
             "mean_score": scores.get(cid, 0.0),
             "per_task_scores": {t: scores.get(cid, 0.0) for t in cohort}}
        (d / "summary.json").write_text(json.dumps(s), encoding="utf-8")
        (d / "traces" / "t1.json").write_text(
            json.dumps({"task_id": "t1", "score": scores.get(cid, 0.0),
                        "success": scores.get(cid, 0.0) > 0.0}),
            encoding="utf-8")
    return run


def self_test() -> int:
    import tempfile
    print("lineage.py acceptance")
    print("=" * 68)
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        # The shape the REAL pilot produced: root with TWO children, only one
        # of which has a grandchild.
        spec = {"gen0_seed": None,
                "gen1_b0_k0": "gen0_seed",
                "gen2_b0_k0": "gen0_seed",
                "gen3_b0_k0": "gen1_b0_k0"}
        scores = {"gen0_seed": 0.68, "gen1_b0_k0": 0.88,
                  "gen2_b0_k0": 0.88, "gen3_b0_k0": 0.84}
        run = _write_run(root, "run_0", spec, scores)
        lins = extract_lineages(run)

        assert len(lins) == 1, [l.candidate_id for l in lins]
        ln = lins[0]
        assert ln.candidate_id == "run_0:gen0_seed"
        assert ln.child_id == "run_0:gen1_b0_k0"
        assert ln.grandchild_id == "run_0:gen3_b0_k0"
        assert ln.tree_id == ("run_0", "gen0_seed")
        assert abs(ln.r - (0.84 - 0.68)) < 1e-9, ln.r
        assert ln.weight == 1.0
        assert len(ln.child_traces) == 1 and len(ln.grandchild_traces) == 1
        print("OK  real lineage       root with 2 children -> only the path "
              "with a real grandchild survives (R_v={:.2f})".format(ln.r))

        # a parent with MULTIPLE complete paths: weights normalise to 1 total
        spec2 = {"gen0_seed": None, "a": "gen0_seed", "b": "gen0_seed",
                 "a1": "a", "b1": "b"}
        sc2 = {k: 0.1 for k in spec2}
        run2 = _write_run(root, "run_1", spec2, sc2)
        l2 = extract_lineages(run2)
        assert len(l2) == 2, len(l2)
        assert all(abs(x.weight - 0.5) < 1e-9 for x in l2), [x.weight for x in l2]
        print("OK  parent weighting   2 real paths from one parent -> weight "
              "0.5 each (sums to 1)")

        # no grandchild -> NO lineage (never zero-filled)
        spec3 = {"gen0_seed": None, "gen1_b0_k0": "gen0_seed"}
        run3 = _write_run(root, "run_2", spec3, {"gen0_seed": 0.5,
                                                 "gen1_b0_k0": 0.6})
        assert extract_lineages(run3) == []
        print("OK  no grandchild      a 2-node chain yields ZERO lineage "
              "(no zero-fill, no child copy)")

        # cohort mismatch -> dropped
        drift = root / "run_3" / "archive"
        _write_run(root, "run_3", spec, scores)
        p = drift / "gen3_b0_k0" / "summary.json"
        d = json.loads(p.read_text())
        d["per_task_scores"] = {"t1": 0.84}          # cohort shrank
        p.write_text(json.dumps(d), encoding="utf-8")
        assert extract_lineages(root / "run_3") == []
        print("OK  cohort check       mismatched task cohort drops the cut")

        # runs do not cross-contaminate: both have a bare 'gen0_seed'
        both = extract_lineages(run) + extract_lineages(run2)
        trees = {x.tree_id for x in both}
        assert trees == {("run_0", "gen0_seed"), ("run_1", "gen0_seed")}, trees
        print("OK  run isolation      bare 'gen0_seed' in 2 runs -> 2 distinct "
              "tree_ids")

        # deterministic
        assert [x.candidate_id for x in extract_lineages(run)] == \
               [x.candidate_id for x in extract_lineages(run)]
        print("OK  deterministic      repeated extraction is identical")

        # no paid requests
        paid = 0
        led = os.environ.get("META_N_REQUEST_LEDGER", "").strip()
        if led and os.path.isfile(led):
            from meta_n.rpbe.accounting import snapshot
            paid = snapshot()["backend_requests"]
        assert paid == 0
        print("OK  zero-cost          paid backend_requests = {}".format(paid))

    print()
    print("VERDICT: ALL OK")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(self_test())
