"""Canonical post-freeze transition audit (§2.2, §2.3, §6, §2.4).

WHY A SEPARATE AUDIT. Search-time scores cannot support the regression
estimator: consolidation inherits non-focus task traces, so a child's search
score may be partly its parent's. The estimator must be computed from FRESH
executions on canonical development instances, after search and archive freeze.

WHAT IT DOES (§6):

  1. read every recorded depth-2 -> depth-3 proposal edge from the SLOT LEDGER
     (not the archive -- rejected and failed children must be included whenever
     the child is executable);
  2. deduplicate candidate EXECUTIONS while preserving every edge identity;
  3. re-evaluate each executable parent and child on ALL cohort tasks with the
     same canonical development instances and repeat seeds;
  4. never use `precomputed_frozen`, per-task-best inheritance, gate-trace
     reuse, archive updates, or test data;
  5. persist raw task/repeat results BEFORE aggregation.

FAIL-CLOSED RULE. Every raw row records whether its score was freshly executed.
An inherited score is invalid input, and the audit refuses to aggregate rather
than quietly reporting a regression rate computed from a parent's own trace.

SEED DERIVATION (§6). The audit seed comes from
`(search_seed, task_id, repeat_index)` only -- never from method, candidate or
depth -- so Official and Predictive see matched instances. A backend that cannot
honour a per-request seed must say so in the manifest; the audit then relies on
matched frozen instances plus repeats and does NOT claim full CRN.
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from meta_n.sri.protocol import ProtocolError, sha256_of

AUDIT_SCHEMA_VERSION = 1

# §2.4: the primary stability statistic is the structural 2 -> 3 transition.
PRIMARY_TRANSITION: Tuple[int, int] = (2, 3)
# Secondary, pre-specified transitions using the identical definition.
SECONDARY_TRANSITIONS: Tuple[Tuple[int, int], ...] = ((1, 2), (3, 4), (4, 5))

# Comparison tolerance for `delta < -threshold`.
#
# The rule is a strict float comparison, so a delta that is *decimally* exactly
# -threshold can land a few ULPs below it (0.88 - 0.90 == -0.020000000000000018)
# and be counted as a regression. Without a tolerance the published rate would
# depend on the platform's arithmetic rather than on the protocol.
#
# This tolerance is therefore PART OF THE FROZEN DEFINITION, not a tunable:
# `regression <=> delta < -threshold - REGRESSION_EPS`. One nanounit is far below
# any score difference the estimators can resolve, so it cannot mask a real
# regression.
REGRESSION_EPS = 1e-9


class InheritedScore(ProtocolError):
    """A score arrived that was not freshly executed. Invalid audit input."""


# --------------------------------------------------------------------------
# audit seed
# --------------------------------------------------------------------------

def audit_seed(search_seed: int, task_id: str, repeat_index: int) -> int:
    """Deterministic, method/candidate/depth-independent seed (§6).

    Derived by hashing the tuple and folding to 31 bits, so two arms evaluating
    the same (search_seed, task, repeat) get the SAME seed regardless of which
    candidate they are scoring or how deep it sits.
    """
    h = sha256_of({"search_seed": int(search_seed), "task_id": str(task_id),
                   "repeat_index": int(repeat_index)})
    return int(h[:8], 16) & 0x7FFFFFFF


# --------------------------------------------------------------------------
# candidate material
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class CandidateMaterial:
    """Everything needed to EXECUTE one candidate on the whole cohort.

    A multi-task candidate stores one script per cohort task, so material is a
    task_id -> source map. This is the only handle the audit needs; it holds no
    scores, which is what keeps inherited numbers out of the estimator.
    """

    candidate_id: str
    structural_depth: int
    task_scripts: Dict[str, str]

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ProtocolError("material needs a candidate_id")
        if int(self.structural_depth) < 1:
            raise ProtocolError("structural_depth must be >= 1")
        if not isinstance(self.task_scripts, dict):
            raise ProtocolError("task_scripts must be a mapping")
        object.__setattr__(self, "structural_depth", int(self.structural_depth))
        object.__setattr__(self, "task_scripts",
                           {str(k): str(v) for k, v in self.task_scripts.items()})

    def sha256(self) -> str:
        return sha256_of({"candidate_id": self.candidate_id,
                          "structural_depth": self.structural_depth,
                          "task_scripts": self.task_scripts})

    def is_executable_on(self, cohort: Sequence[str]) -> bool:
        return all(t in self.task_scripts and self.task_scripts[t].strip()
                   for t in cohort)

    def missing_tasks(self, cohort: Sequence[str]) -> List[str]:
        return [t for t in cohort
                if t not in self.task_scripts or not self.task_scripts[t].strip()]


def load_material_from_dir(root, candidate_id: str, structural_depth: int,
                           cohort: Sequence[str]) -> Optional[CandidateMaterial]:
    """Read `traces/<task>.py`-style material from a candidate directory.

    Used for BOTH archive candidates and the out-of-archive `material_path`
    written for rejected/failed slots, so the two sources share one loader.
    """
    root = Path(root)
    tdir = root / "traces" if (root / "traces").is_dir() else root
    if not tdir.is_dir():
        return None
    scripts: Dict[str, str] = {}
    for t in cohort:
        p = tdir / (t + ".py")
        if p.is_file():
            scripts[t] = p.read_text(encoding="utf-8", errors="replace")
    if not scripts:
        return None
    return CandidateMaterial(candidate_id=candidate_id,
                             structural_depth=int(structural_depth),
                             task_scripts=scripts)


# --------------------------------------------------------------------------
# edges
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Edge:
    """One valid generated (parent, child) transition, with its slot identity.

    `slot_id` is carried so an edge can always be traced back to the nominal
    slot row that produced it -- including a slot whose child never entered the
    archive.
    """

    slot_id: str
    iteration: int
    parent: CandidateMaterial
    child: CandidateMaterial
    child_was_admitted: bool
    child_terminal_status: str

    @property
    def transition(self) -> Tuple[int, int]:
        return (self.parent.structural_depth, self.child.structural_depth)


def build_edges(ledger_rows: Iterable[Mapping[str, Any]],
                material_of) -> Tuple[List[Edge], List[Dict[str, Any]]]:
    """Depth-2 -> depth-3 edges from the ledger, plus a drop account.

    `material_of(row, side)` returns a `CandidateMaterial` or None. A row whose
    parent or child material is unavailable makes the edge INVALID, and invalidity
    is REPORTED (§13: "never silently dropped") -- it shows up in the drop
    account and, through it, in `F` / `R_worst`.
    """
    edges: List[Edge] = []
    drops: List[Dict[str, Any]] = []
    for r in ledger_rows:
        pd = r.get("parent_structural_depth")
        cd = r.get("proposed_child_depth")
        sid = r.get("slot_id")
        term = r.get("terminal_status") or "OPEN"
        if pd is None or cd is None:
            # §13: an edge we cannot even PLACE is reported, never silently
            # dropped -- a malformed row must not quietly shrink the
            # denominator.
            drops.append({"slot_id": sid, "reason": "missing_structural_depth",
                          "parent_structural_depth": pd,
                          "proposed_child_depth": cd,
                          "terminal_status": term})
            continue
        if (int(pd), int(cd)) not in (PRIMARY_TRANSITION,) + SECONDARY_TRANSITIONS:
            # A legitimate transition we are not auditing in this pass (e.g. a
            # 3->4 edge). Out of scope, not a failure.
            continue
        pm = material_of(r, "parent")
        cm = material_of(r, "child")
        if pm is None or cm is None:
            drops.append({"slot_id": sid, "reason": "missing_material",
                          "transition": "{}{}".format(int(pd), int(cd)),
                          "parent_material": pm is not None,
                          "child_material": cm is not None,
                          "terminal_status": term})
            continue
        edges.append(Edge(slot_id=sid, iteration=int(r.get("iteration") or 0),
                          parent=pm, child=cm,
                          child_was_admitted=bool(r.get("archive_admitted")),
                          child_terminal_status=term))
    return edges, drops


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class RawEval:
    """One task/repeat observation. `freshly_executed=False` is INVALID input."""

    score: float
    success: bool
    freshly_executed: bool
    detail: str = ""


def _check_raw(ev: RawEval, where: str) -> RawEval:
    if not ev.freshly_executed:
        raise InheritedScore(
            "{} returned a NON-fresh score (inherited/reused); the canonical "
            "audit only accepts freshly executed scores (§6)".format(where))
    if ev.score is None or not math.isfinite(float(ev.score)):
        raise InheritedScore(
            "{} returned a non-finite score ({}); non-finite scores are failed "
            "audit observations (§13)".format(where, ev.score))
    return ev


@dataclass
class AuditResult:
    raw_rows: List[Dict[str, Any]] = field(default_factory=list)
    edges: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    exact_depth: Dict[str, Any] = field(default_factory=dict)

    def write(self, out_dir) -> Dict[str, str]:
        d = Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)
        paths = {}
        for name, obj, as_lines in (
                ("raw_evaluations.jsonl", self.raw_rows, True),
                ("edges.jsonl", self.edges, True),
                ("metrics.json", self.metrics, False),
                ("exact_depth.json", self.exact_depth, False)):
            p = d / name
            if as_lines:
                with p.open("w", encoding="utf-8") as f:
                    for row in obj:
                        f.write(json.dumps(row, sort_keys=True) + "\n")
            else:
                p.write_text(json.dumps(obj, indent=2, sort_keys=True),
                             encoding="utf-8")
            paths[name] = str(p)
        return paths


def run_canonical_audit(edges: Sequence[Edge], *, cohort: Sequence[str],
                        evaluator, search_seed: int,
                        repeats: int = 3,
                        threshold: float = 0.02,
                        drops: Optional[Sequence[Mapping[str, Any]]] = None,
                        pairing_limitations: Sequence[str] = (),
                        ) -> AuditResult:
    """§6: re-evaluate every executable parent/child and compute the metrics.

    `evaluator.evaluate(task_id, source, seed=...) -> RawEval`. Dedup happens on
    the EXECUTION (candidate_id, task, repeat) while every edge keeps its own
    row, so a parent shared by many edges is not re-run many times but is still
    counted once per edge-task pair in the denominator.
    """
    if repeats < 1:
        raise ProtocolError("repeats must be >= 1")
    cohort = list(cohort)

    # ---- pass 1: dedup executions, keep a fresh-score cache ---------------
    cache: Dict[Tuple[str, str, int], RawEval] = {}
    rows: List[Dict[str, Any]] = []
    exec_count = 0

    def score_of(mat: CandidateMaterial, task_id: str) -> Optional[float]:
        nonlocal exec_count
        if task_id not in mat.task_scripts:
            return None
        vals = []
        for rep in range(int(repeats)):
            key = (mat.candidate_id, task_id, rep)
            if key not in cache:
                seed = audit_seed(search_seed, task_id, rep)
                ev = evaluator.evaluate(task_id, mat.task_scripts[task_id],
                                        seed=seed)
                ev = _check_raw(ev, "{}|{}|rep{}".format(
                    mat.candidate_id, task_id, rep))
                cache[key] = ev
                exec_count += 1
                rows.append({
                    "schema_version": AUDIT_SCHEMA_VERSION,
                    "candidate_id": mat.candidate_id,
                    "structural_depth": mat.structural_depth,
                    "material_sha256": mat.sha256(),
                    "task_id": task_id, "repeat_index": rep, "seed": seed,
                    "score": float(ev.score), "success": bool(ev.success),
                    "freshly_executed": True, "detail": ev.detail,
                })
            vals.append(float(cache[key].score))
        # median: an audit repeat is NESTED inside one edge-task observation and
        # must never enlarge the denominator (§2.2).
        return statistics.median(vals)

    # ---- pass 2: per-edge-task deltas -------------------------------------
    metric_rows: List[Dict[str, Any]] = []
    edge_rows: List[Dict[str, Any]] = []
    for e in edges:
        per_task: Dict[str, Dict[str, Optional[float]]] = {}
        for t in cohort:
            p = score_of(e.parent, t)
            c = score_of(e.child, t)
            if p is None or c is None:
                continue                      # task not covered by the material
            per_task[t] = {"parent": p, "child": c, "delta": c - p}
        edge_rows.append({
            "schema_version": AUDIT_SCHEMA_VERSION,
            "slot_id": e.slot_id, "iteration": e.iteration,
            "parent_id": e.parent.candidate_id,
            "child_id": e.child.candidate_id,
            "parent_depth": e.parent.structural_depth,
            "child_depth": e.child.structural_depth,
            "transition": "{}{}".format(*e.transition),
            "child_was_admitted": e.child_was_admitted,
            "child_terminal_status": e.child_terminal_status,
            "parent_material_sha256": e.parent.sha256(),
            "child_material_sha256": e.child.sha256(),
            "per_task": per_task,
        })

    # ---- aggregate per transition ----------------------------------------
    def agg(tr: Tuple[int, int]) -> Dict[str, Any]:
        eids = [r for r in edge_rows if r["transition"] == "{}{}".format(*tr)]
        regressions = 0
        pairs = 0
        deltas: List[float] = []
        per_task_reg: Dict[str, List[int]] = {t: [] for t in cohort}
        for r in eids:
            for t, d in r["per_task"].items():
                pairs += 1
                deltas.append(d["delta"])
                hit = 1 if d["delta"] < -float(threshold) - REGRESSION_EPS else 0
                regressions += hit
                per_task_reg.setdefault(t, []).append(hit)
        r_valid = (regressions / pairs) if pairs else None
        return {
            "transition": "{}{}".format(*tr),
            "threshold": float(threshold),
            "regression_eps": REGRESSION_EPS,
            "rule": "regression <=> delta < -threshold - regression_eps",
            "edge_count": len(eids),
            "valid_edge_task_pairs": pairs,
            "regressions": regressions,
            "R_valid": r_valid,
            "delta_mean": (statistics.fmean(deltas) if deltas else None),
            "delta_median": (statistics.median(deltas) if deltas else None),
            "per_task_regression_rate": {
                t: (sum(v) / len(v) if v else None)
                for t, v in per_task_reg.items()},
        }

    primary = agg(PRIMARY_TRANSITION)
    transitions = {"R_2to3": primary}
    for tr in SECONDARY_TRANSITIONS:
        transitions["R_{}to{}".format(*tr)] = agg(tr)

    # ---- §2.3 failure accounting, PER TRANSITION ---------------------------
    # The numerator and the denominator must describe the SAME transition.
    # The first revision divided the 2->3 regression count by an ALL-transition
    # attempt count (and added every transition's drops to the same numerator),
    # so `R_worst` was not the worst case of any quantity: adding unaudited
    # 3->4 edges to the denominator could pull the published rate DOWN.
    drops = list(drops or [])
    drops_by_tr: Dict[str, List[Dict[str, Any]]] = {}
    unplaceable: List[Dict[str, Any]] = []
    for d in drops:
        tr = d.get("transition")
        if tr is None:
            unplaceable.append(d)
        else:
            drops_by_tr.setdefault(str(tr), []).append(d)

    def account(tr: Tuple[int, int], rows_for_tr: List[Dict[str, Any]],
                regressions: int) -> Dict[str, Any]:
        dr = drops_by_tr.get("{}{}".format(*tr), [])
        attempted = (len(rows_for_tr) + len(dr)) * len(cohort)
        missing_task_pairs = sum(len(cohort) - len(r["per_task"])
                                 for r in rows_for_tr)
        failed = missing_task_pairs + len(dr) * len(cohort)
        return {
            "transition": "{}{}".format(*tr),
            "edges": len(rows_for_tr),
            "dropped_edges": len(dr),
            "attempted_edge_task_slots": attempted,
            "failed_edge_task_slots": failed,
            "F": (failed / attempted) if attempted else None,
            "R_worst": (((regressions or 0) + failed) / attempted
                        if attempted else None),
        }

    by_transition = {}
    for tr in (PRIMARY_TRANSITION,) + SECONDARY_TRANSITIONS:
        key = "R_{}to{}".format(*tr)
        rows_tr = [r for r in edge_rows if r["transition"] == "{}{}".format(*tr)]
        by_transition[key] = account(tr, rows_tr,
                                     transitions[key]["regressions"] or 0)

    primary_fa = by_transition["R_2to3"]
    # The overall view exists for the unplaceable rows, which belong to no single
    # transition. A row whose structural depth is missing is a real defect (the
    # ledger gate refuses it for an executed slot), so it is disclosed here
    # rather than folded into one transition's rate.
    all_rows = edge_rows
    all_drops = [d for dr in drops_by_tr.values() for d in dr]
    attempted_all = ((len(all_rows) + len(all_drops) + len(unplaceable))
                     * len(cohort))
    failed_all = (sum(len(cohort) - len(r["per_task"]) for r in all_rows)
                  + (len(all_drops) + len(unplaceable)) * len(cohort))
    overall = {
        "attempted_edge_task_slots": attempted_all,
        "failed_edge_task_slots": failed_all,
        "F": (failed_all / attempted_all) if attempted_all else None,
        "R_worst": None,          # a single-transition quantity; see by_transition
        "unplaceable_rows": len(unplaceable),
        "note": "overall F includes rows whose structural depth is missing; "
                "R_worst is reported PER TRANSITION because a 2->3 regression "
                "cannot be divided by a 3->4 denominator",
    }

    # ---- §2.4 depth reporting ---------------------------------------------
    best_at_depth: Dict[int, float] = {}
    counts_at_depth: Dict[int, int] = {}
    for e in edges:
        for mat in (e.parent, e.child):
            counts_at_depth[mat.structural_depth] = \
                counts_at_depth.get(mat.structural_depth, 0) + 1
    exact_depth = {
        "note": "exact_depth_best is over FINAL-ARCHIVE candidates by "
                "structural depth; populated by the caller which owns the "
                "frozen archive (see finalize_exact_depth).",
        "depths_present_in_edges": sorted(counts_at_depth),
        "edge_candidate_counts_by_depth": counts_at_depth,
        "iteration_archive_best": "reported separately by the runner (§2.4): "
                                  "`iteration_archive_best(j)`, never mixed "
                                  "with exact structural depth",
    }

    metrics = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "search_seed": int(search_seed),
        "cohort": cohort,
        "repeats": int(repeats),
        "threshold": float(threshold),
        "deduplicated_executions": exec_count,
        "raw_observations": len(rows),
        "edges": len(edge_rows),
        "invalid_edges": list(drops or []),
        "transitions": transitions,
        "failure_accounting": dict(
            primary_fa,
            by_transition=by_transition,
            overall=overall,
            unplaceable_rows=list(unplaceable),
            note="R_valid is the paper's primary statistic; F and R_worst "
                 "are mandatory robustness disclosures (§2.3) and are computed "
                 "per transition from that transition's own denominator."),
        "pairing_limitations": list(pairing_limitations),
    }
    return AuditResult(raw_rows=rows, edges=edge_rows, metrics=metrics,
                       exact_depth=exact_depth)


def assert_placeable(ledger_rows: Iterable[Mapping[str, Any]]) -> None:
    """Every EXECUTED slot must carry both structural depths (§2.4, §13).

    A row that ran but whose depth is unknown cannot be placed on the depth
    axis, so it silently leaves every transition denominator -- the run would
    look smaller rather than broken. This is checked at freeze time so the
    defect surfaces before the paywalled audit, not after.
    """
    from meta_n.sri.ledger import EXECUTABLE_STATUSES
    bad = []
    for r in ledger_rows:
        if (r.get("terminal_status") or "OPEN") not in EXECUTABLE_STATUSES:
            continue
        if r.get("parent_structural_depth") is None or \
                r.get("proposed_child_depth") is None:
            bad.append({"slot_id": r.get("slot_id"),
                        "terminal_status": r.get("terminal_status"),
                        "parent_structural_depth":
                            r.get("parent_structural_depth"),
                        "proposed_child_depth": r.get("proposed_child_depth")})
    if bad:
        raise ProtocolError(
            "{} executed slot(s) have no structural depth and could not be "
            "placed in any transition denominator: {}".format(len(bad), bad))


def finalize_exact_depth(audit: AuditResult,
                         archive_candidates: Sequence[Mapping[str, Any]],
                         dev_score_of) -> Dict[str, Any]:
    """§2.4: `exact_depth_best(d)` over the FINAL ARCHIVE, structurally.

    Kept separate from any iteration-based view so `depth` can never be
    confused with `iteration`. Synthesized candidates are excluded: `merge_oracle`
    is written with a FABRICATED depth and the oracle's own macro score, so it
    would otherwise be reported as the best candidate at whatever depth it
    claimed.
    """
    from meta_n.sri.protocol import is_synthesized_candidate
    best: Dict[int, Dict[str, Any]] = {}
    counts: Dict[int, int] = {}
    excluded: List[str] = []
    for c in archive_candidates:
        if is_synthesized_candidate(c):
            excluded.append(str(c.get("candidate_id")))
            continue
        d = int(c.get("depth") or 0)
        counts[d] = counts.get(d, 0) + 1
        s = dev_score_of(c)
        if s is None or not math.isfinite(float(s)):
            continue
        cur = best.get(d)
        if cur is None or float(s) > cur["dev_score"]:
            best[d] = {"candidate_id": c.get("candidate_id"),
                       "dev_score": float(s)}
    out = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "exact_depth_best": {str(d): v for d, v in sorted(best.items())},
        "exact_depth_counts": {str(d): n for d, n in sorted(counts.items())},
        "available_depths": sorted(counts),
        "synthesized_excluded": excluded,
        "note": "structural depth (candidate.depth); iteration-wise archive "
                "best is a SEPARATE series and must not be substituted. "
                "Synthesized (oracle/merge) candidates are excluded because "
                "their depth is fabricated by assembly.",
    }
    audit.exact_depth.update(out)
    return out


# --------------------------------------------------------------------------
# acceptance self-test (§14 items 4, 5, 6, 7, 8)
# --------------------------------------------------------------------------

class _StubEvaluator:
    """Deterministic offline evaluator: score = f(candidate, task).

    Optionally returns an inherited score to prove the audit rejects it.
    """

    def __init__(self, table, *, inherited=()):
        self.table = table
        self.inherited = set(inherited)
        self.calls = []

    def evaluate(self, task_id, source, *, seed):
        self.calls.append((task_id, seed))
        key = (source, task_id)
        if key in self.inherited:
            return RawEval(score=0.5, success=True, freshly_executed=False,
                           detail="reused gate trace")
        return RawEval(score=float(self.table.get(key, 0.0)), success=True,
                       freshly_executed=True)


def _mat(cid, depth, **scores):
    return CandidateMaterial(candidate_id=cid, structural_depth=depth,
                             task_scripts=scores)


def self_test() -> int:
    import tempfile
    print("transition_audit.py acceptance")
    print("=" * 68)

    cohort = ["T1", "T2"]
    # parent depth 2, two children depth 3: one regresses hard, one improves.
    parent = _mat("p2", 2, T1="src_p_T1", T2="src_p_T2")
    good = _mat("c3a", 3, T1="src_a_T1", T2="src_a_T2")
    bad = _mat("c3b", 3, T1="src_b_T1", T2="src_b_T2")

    table = {
        ("src_p_T1", "T1"): 0.90, ("src_p_T2", "T2"): 0.90,
        ("src_a_T1", "T1"): 0.95, ("src_a_T2", "T2"): 0.99,   # improves both
        ("src_b_T1", "T1"): 0.70, ("src_b_T2", "T2"): 0.89,   # T1 only regresses
    }
    ledger = [
        {"slot_id": "it0-p0-c0", "iteration": 0, "parent_structural_depth": 2,
         "proposed_child_depth": 3, "terminal_status": "evaluated_admitted",
         "archive_admitted": True, "parent_id": "p2"},
        {"slot_id": "it0-p0-c1", "iteration": 0, "parent_structural_depth": 2,
         "proposed_child_depth": 3, "terminal_status": "gate_rejected",
         "archive_admitted": False, "parent_id": "p2"},
        {"slot_id": "it0-p1-c0", "iteration": 0, "parent_structural_depth": 1,
         "proposed_child_depth": 2, "terminal_status": "evaluated_admitted",
         "archive_admitted": True, "parent_id": "gen0"},          # other depth
    ]
    mats = {"it0-p0-c0": (parent, good), "it0-p0-c1": (parent, bad)}

    def material_of(row, side):
        pair = mats.get(row["slot_id"])
        if pair is None:
            return _mat("gen0", 1, T1="g", T2="g") if side == "parent" else None
        return pair[0] if side == "parent" else pair[1]

    edges, drops = build_edges(ledger, material_of)
    assert len(edges) == 2, [e.slot_id for e in edges]
    print("OK  edge set           {} depth2->3 edges (the depth1->2 row is "
          "excluded by transition filter)".format(len(edges)))

    # §14.4: a GATE-REJECTED but executable child is still audited
    assert any(e.child_terminal_status == "gate_rejected" for e in edges)
    assert any(not e.child_was_admitted for e in edges)
    print("OK  rejected included  the gate_rejected child is in the edge set")

    ev = _StubEvaluator(table)
    res = run_canonical_audit(edges, cohort=cohort, evaluator=ev, search_seed=0,
                              repeats=3, threshold=0.02)
    m = res.metrics
    r23 = m["transitions"]["R_2to3"]
    assert r23["edge_count"] == 2
    assert r23["valid_edge_task_pairs"] == 4, r23
    # exactly ONE regression: bad|T1 = 0.70 - 0.90 = -0.20
    assert r23["regressions"] == 1, r23
    assert abs(r23["R_valid"] - 0.25) < 1e-12, r23["R_valid"]
    print("OK  R_2to3             1 regression / 4 edge-task pairs = {}"
          .format(r23["R_valid"]))

    # BOUNDARY: the rule is `delta < -threshold - REGRESSION_EPS`, so a delta
    # that is decimally exactly -threshold is NOT a regression. Pinned because
    # the same comparison without the tolerance flips on float representation
    # (0.88 - 0.90 == -0.020000000000000018), which would make the published
    # rate platform-dependent.
    bnd = [Edge(slot_id="b", iteration=0,
                parent=_mat("p", 2, T1="s_p"), child=_mat("c", 3, T1="s_c"),
                child_was_admitted=True, child_terminal_status="evaluated_admitted")]
    ev_b = _StubEvaluator({("s_p", "T1"): 0.90, ("s_c", "T1"): 0.88})
    rb = run_canonical_audit(bnd, cohort=["T1"], evaluator=ev_b, search_seed=0,
                             repeats=1, threshold=0.02)
    assert rb.metrics["transitions"]["R_2to3"]["regressions"] == 0
    # ...and a clearly-below delta still counts.
    ev_c = _StubEvaluator({("s_p", "T1"): 0.90, ("s_c", "T1"): 0.87})
    rc = run_canonical_audit(bnd, cohort=["T1"], evaluator=ev_c, search_seed=0,
                             repeats=1, threshold=0.02)
    assert rc.metrics["transitions"]["R_2to3"]["regressions"] == 1
    print("OK  threshold boundary delta == -threshold is NOT a regression; "
          "delta = -0.03 IS (tolerance {} is part of the frozen rule)"
          .format(REGRESSION_EPS))

    # §14.8: repeats are nested -> they cannot inflate the denominator.
    # 3 distinct candidates (p2, c3a, c3b) x 2 tasks x 3 repeats = 18 raw rows,
    # collapsing to 4 edge-task pairs.
    assert m["raw_observations"] == 3 * len(cohort) * 3, m["raw_observations"]
    assert r23["valid_edge_task_pairs"] == 4
    print("OK  repeats nested     {} raw observations collapse to 4 edge-task "
          "pairs (repeats never enlarge the denominator)"
          .format(m["raw_observations"]))

    # §14.6: every cohort task freshly evaluated, with DEDUPLICATED executions.
    # The dedup claim is about a candidate shared by several edges: p2 is the
    # parent of BOTH edges, so it must be executed once per (task, repeat) -- 6
    # times -- not once per edge (12).
    assert m["deduplicated_executions"] == 3 * len(cohort) * 3, m
    p2_rows = [r for r in res.raw_rows if r["candidate_id"] == "p2"]
    assert len(p2_rows) == len(cohort) * 3, len(p2_rows)
    print("OK  dedup              parent p2 shared by 2 edges executed {}x "
          "(once per task/repeat), not {:.0f}x".format(
              len(p2_rows), 2 * len(cohort) * 3))

    # §14.7: matched task/repeat seeds, method/candidate independent
    s1 = audit_seed(3, "T1", 0)
    assert s1 == audit_seed(3, "T1", 0)
    assert s1 != audit_seed(3, "T1", 1)
    assert s1 != audit_seed(3, "T2", 0)
    assert s1 != audit_seed(4, "T1", 0)
    print("OK  audit seed         derived only from (search_seed, task, repeat)")

    # §14.5: an INHERITED score is refused outright
    ev2 = _StubEvaluator(table, inherited={("src_a_T1", "T1")})
    try:
        run_canonical_audit(edges, cohort=cohort, evaluator=ev2, search_seed=0,
                            repeats=2)
        raise AssertionError("an inherited score was accepted")
    except InheritedScore as e:
        assert "NON-fresh" in str(e)
    print("OK  inherited rejected an inherited score fails the audit closed")

    # raw rows state freshness explicitly
    assert all(r["freshly_executed"] is True for r in res.raw_rows)
    print("OK  freshness recorded every raw row carries freshly_executed=True")

    # §2.3 failure accounting
    fa = m["failure_accounting"]
    assert fa["attempted_edge_task_slots"] == 4, fa
    assert fa["F"] == 0.0 and fa["R_worst"] == 0.25, fa
    print("OK  failure accounting F={} R_worst={} (no missing material here)"
          .format(fa["F"], fa["R_worst"]))

    # a missing-material edge is INVALID and visible, not dropped silently
    def material_of_drop(row, side):
        if row["slot_id"] == "it0-p0-c1":
            return None
        return material_of(row, side)
    edges2, drops2 = build_edges(ledger, material_of_drop)
    # TWO rows are dropped here: the gate-rejected edge (material suppressed by
    # the stub) and the depth1->2 row (its child material was never provided).
    assert len(edges2) == 1, len(edges2)
    assert len(drops2) == 2, drops2
    assert all(d["reason"] == "missing_material" for d in drops2)
    res2 = run_canonical_audit(edges2, cohort=cohort, evaluator=_StubEvaluator(table),
                               search_seed=0, repeats=1, drops=drops2)
    fa2 = res2.metrics["failure_accounting"]
    # PER TRANSITION: the surviving 2->3 edge plus the dropped 2->3 edge give
    # attempted = 2 * 2 = 4 and failed = 1 drop * 2 tasks = 2 -> F = 0.5. The
    # dropped 1->2 row belongs to a DIFFERENT transition and must not appear in
    # either the numerator or the denominator (mixing them reported F=4/6).
    assert fa2["attempted_edge_task_slots"] == 4, fa2
    assert fa2["failed_edge_task_slots"] == 2, fa2
    assert abs(fa2["F"] - 0.5) < 1e-12, fa2
    assert fa2["F"] <= 1.0
    assert fa2["by_transition"]["R_1to2"]["F"] == 1.0, fa2["by_transition"]
    print("OK  invalid visible    dropped edges raise the 2->3 F to {:.3f} "
          "(attempted counts them); the 1->2 drop stays in its own transition"
          .format(fa2["F"]))

    # ---- the mixing bug: a 3->4 edge must not move the 2->3 rate ----------
    # `R_worst` adds THIS transition's regressions to THIS transition's
    # failures. The first revision divided 2->3 regressions by an attempt count
    # that included every 3->4 edge, so a broken 3->4 arm pulled the published
    # 2->3 rate towards zero.
    wide = list(ledger) + [
        {"slot_id": "it2-p0-c0", "iteration": 2, "parent_structural_depth": 3,
         "proposed_child_depth": 4, "terminal_status": "execution_error",
         "archive_admitted": False, "parent_id": "c3a"}]
    mats_wide = dict(mats)
    mats_wide["it0-p0-c0"] = (parent, good)
    mats_wide["it0-p0-c1"] = (parent, bad)
    mats_wide["it2-p0-c0"] = (_mat("c3a", 3, T1="a", T2="a"),
                              _mat("c4", 4, T1="b", T2="b"))

    def material_of_wide(row, side):
        pair = mats_wide.get(row["slot_id"])
        if pair is None:
            return _mat("gen0", 1, T1="g", T2="g") if side == "parent" else None
        return pair[0] if side == "parent" else pair[1]

    edges_w, drops_w = build_edges(wide, material_of_wide)
    res_w = run_canonical_audit(edges_w, cohort=cohort,
                                evaluator=_StubEvaluator(table),
                                search_seed=0, repeats=1, drops=drops_w)
    faw = res_w.metrics["failure_accounting"]
    r23_w = res_w.metrics["transitions"]["R_2to3"]
    assert r23_w["regressions"] == r23["regressions"], r23_w
    assert faw["R_worst"] == fa["R_worst"], (faw, fa)
    assert faw["attempted_edge_task_slots"] == fa["attempted_edge_task_slots"]
    assert "R_3to4" in faw["by_transition"], sorted(faw["by_transition"])
    print("OK  no cross-talk      adding a 3->4 edge leaves R_2to3 and its "
          "F/R_worst unchanged (R_worst={})".format(faw["R_worst"]))

    # an EXECUTED row with no structural depth is refused at freeze time
    try:
        assert_placeable(ledger + [{"slot_id": "x", "terminal_status":
                                    "evaluated_admitted",
                                    "parent_structural_depth": None,
                                    "proposed_child_depth": 3}])
        raise AssertionError("an unplaceable executed row was accepted")
    except ProtocolError as e:
        assert "structural depth" in str(e)
    # ...but a FAILED row that never produced a child is fine
    assert_placeable(ledger + [{"slot_id": "y", "terminal_status":
                                "empty_injection",
                                "parent_structural_depth": 2,
                                "proposed_child_depth": None}])
    print("OK  placeability gate  an executed slot without a depth is refused; "
          "a failed slot without one is not")

    # exact-depth reporting is structural
    finalize_exact_depth(res, [
        {"candidate_id": "gen0", "depth": 1}, {"candidate_id": "p2", "depth": 2},
        {"candidate_id": "c3a", "depth": 3}, {"candidate_id": "c3b", "depth": 3},
    ], dev_score_of=lambda c: {"gen0": 0.5, "p2": 0.9, "c3a": 0.95,
                               "c3b": 0.7}[c["candidate_id"]])
    ed = res.exact_depth["exact_depth_best"]
    assert ed["3"]["candidate_id"] == "c3a" and ed["3"]["dev_score"] == 0.95
    assert res.exact_depth["exact_depth_counts"]["3"] == 2
    assert "iteration" in res.exact_depth["note"]
    print("OK  exact depth        best-at-depth is structural and separate from "
          "the iteration series")

    with tempfile.TemporaryDirectory() as td:
        paths = res.write(td)
        for name in ("raw_evaluations.jsonl", "edges.jsonl", "metrics.json",
                     "exact_depth.json"):
            assert Path(paths[name]).is_file(), name
        n_raw = sum(1 for _ in open(paths["raw_evaluations.jsonl"]))
        assert n_raw == len(res.raw_rows)
        print("OK  artifacts          wrote raw_evaluations.jsonl ({} rows), "
              "edges.jsonl, metrics.json, exact_depth.json".format(n_raw))

    print()
    print("VERDICT: ALL OK")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(self_test())
