"""Final candidate selection (§2.1), per-run metrics and cross-seed aggregation
(§11).

TWO THINGS THIS MODULE EXISTS TO STOP:

  * a **task-wise virtual oracle** becoming Final Score. The oracle picks a
    different candidate per task, so it is not a deployable state at all. It
    survives only as the explicitly named diagnostic `oracle_upper_bound_dev`,
    which is never sent to held-out test in formal mode.
  * the label **"best chain"**. Final Score is the held-out score of ONE
    dev-selected deployable candidate, and this module never emits a
    `chain_test_mean_score` key or a `best chain` label.

SELECTION IS DEV-ONLY (§2.1). `select_deployable` reads development scores and
nothing else; the held-out evaluation happens strictly afterwards, so test data
cannot influence the selected candidate, the archive, the threshold, or the
reported depth. Ties break deterministically and method-independently:
development score desc, then creation index asc, then candidate id lexicographic
asc.

The aggregator consumes ONLY frozen artifacts and refuses to mix schema
versions, cohorts, root hashes or profiles (§11).
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from meta_n.sri.protocol import (ProtocolError, is_synthesized_candidate,
                                 sha256_of)

METRICS_SCHEMA_VERSION = 1


# --------------------------------------------------------------------------
# §2.1 single deployable selection
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Deployable:
    """One candidate that could be shipped, with its DEV evidence only."""

    candidate_id: str
    dev_score: float
    structural_depth: int
    creation_index: int
    manifest_sha256: str = ""
    # Every candidate the selector refused, WITH its reason: the report has to
    # show that the winner was chosen against a stated pool, not an implicit one.
    excluded_from_pool: List[Dict[str, Any]] = field(default_factory=list)


def select_deployable(candidates: Sequence[Mapping[str, Any]],
                      dev_score_of,
                      creation_index_of=None,
                      is_executable_of=None,
                      allow_synthesized: bool = False) -> Deployable:
    """§2.1 `x_hat = argmax_{x in A_deploy} macro_dev(x)`, deterministic ties.

    `candidates` is the FROZEN final archive. The root stays eligible (it is a
    candidate like any other); a non-executable or unscored candidate is
    excluded with a reason rather than silently skipped.

    Two exclusions the first revision left out, both of which silently produced
    the WRONG Final Score:

      * `allow_synthesized=False` (the default) refuses to select a candidate
        meta-n assembled rather than bred. `merge_oracle` is the virtual oracle
        by construction and carries a fabricated depth; selecting it makes Final
        Score a diagnostic, which is exactly what §2.1 forbids.
      * `is_executable_of` (when supplied) refuses a candidate with no material
        to deploy. Without it the archive's highest-scoring entry could be one
        whose scripts do not exist on disk.
    """
    pool: List[Deployable] = []
    excluded: List[Dict[str, Any]] = []
    for i, c in enumerate(candidates):
        cid = str(c.get("candidate_id") or "")
        if not cid:
            excluded.append({"reason": "no_candidate_id", "index": i})
            continue
        if not allow_synthesized and is_synthesized_candidate(c):
            excluded.append({"candidate_id": cid,
                             "reason": "synthesized_oracle_not_deployable",
                             "dev_score": c.get("mean_score")})
            continue
        if is_executable_of is not None and not is_executable_of(c):
            excluded.append({"candidate_id": cid,
                             "reason": "no_executable_material"})
            continue
        s = dev_score_of(c)
        if s is None or not math.isfinite(float(s)):
            excluded.append({"candidate_id": cid, "reason": "no_finite_dev_score",
                             "dev_score": s})
            continue
        idx = (int(creation_index_of(c)) if creation_index_of is not None
               else int(c.get("creation_index", i)))
        pool.append(Deployable(candidate_id=cid, dev_score=float(s),
                               structural_depth=int(c.get("depth") or 0),
                               creation_index=idx))
    if not pool:
        raise ProtocolError(
            "no deployable candidate: the frozen archive has no candidate with "
            "a finite development score (excluded: {})".format(excluded))
    # deterministic, method-independent tie-break
    pool.sort(key=lambda d: (-d.dev_score, d.creation_index, d.candidate_id))
    win = pool[0]
    return Deployable(candidate_id=win.candidate_id, dev_score=win.dev_score,
                      structural_depth=win.structural_depth,
                      creation_index=win.creation_index,
                      manifest_sha256=win.manifest_sha256,
                      excluded_from_pool=excluded)


def assert_no_oracle_as_final(final_record: Mapping[str, Any]) -> None:
    """§2.1/§7: a task-wise virtual oracle may never appear as Final Score."""
    forbidden = ("oracle", "virtual_oracle", "per_task_oracle")
    for k in final_record:
        kl = str(k).lower()
        if any(f in kl for f in forbidden) and "upper_bound" not in kl:
            raise ProtocolError(
                "field {!r} looks like a virtual-oracle result in the Final "
                "Score record; the oracle is a diagnostic "
                "(`oracle_upper_bound_dev`) and is never sent to held-out test "
                "in formal mode".format(k))


def final_record(selected: Deployable, test_result: Mapping[str, Any]) -> Dict[str, Any]:
    """§7: the ONE primary held-out object."""
    rec = {
        "selected_candidate_id": selected.candidate_id,
        "selected_candidate_dev_score": selected.dev_score,
        "selected_candidate_depth": selected.structural_depth,
        "selected_candidate_manifest_sha256": selected.manifest_sha256,
        "final_test_scores": dict(test_result.get("test_scores") or {}),
        "final_test_macro_score": test_result.get("test_mean_score"),
    }
    assert_no_oracle_as_final(rec)
    return rec


def forbid_best_chain_label(record: Mapping[str, Any]) -> None:
    """Acceptance criterion 4: no key may call the selected candidate a chain."""
    for k in record:
        if "chain" in str(k).lower():
            raise ProtocolError(
                "output key {!r} uses the retired `best chain` framing; the "
                "primary object is a single dev-selected deployable candidate "
                "(§2.4/§7)".format(k))


# --------------------------------------------------------------------------
# §11 per-run metrics
# --------------------------------------------------------------------------

def per_run_metrics(*, selected: Deployable, test_result: Mapping[str, Any],
                    audit_metrics: Mapping[str, Any],
                    ledger_counts: Mapping[str, int],
                    resource_use: Mapping[str, Any],
                    archive_candidates: Sequence[Mapping[str, Any]],
                    dev_score_of,
                    iteration_archive_best: Optional[Sequence[Mapping[str, Any]]] = None,
                    ) -> Dict[str, Any]:
    """Everything §11 asks for at run level, from frozen artifacts only."""
    exact = {}
    counts = {}
    synthesized_excluded: List[str] = []
    for c in archive_candidates:
        if is_synthesized_candidate(c):
            # §2.4: `merge_oracle` is written with a FABRICATED depth (2) and the
            # oracle's own macro score, so letting it into the depth series
            # reports the virtual oracle as the best depth-2 candidate -- the
            # exact conflation the depth series exists to prevent.
            synthesized_excluded.append(str(c.get("candidate_id")))
            continue
        d = int(c.get("depth") or 0)
        counts[d] = counts.get(d, 0) + 1
        s = dev_score_of(c)
        if s is None or not math.isfinite(float(s)):
            continue
        if d not in exact or float(s) > exact[d]:
            exact[d] = float(s)

    out = {
        "schema_version": METRICS_SCHEMA_VERSION,
        "final": final_record(selected, test_result),
        "selected_pool": {
            "excluded": list(selected.excluded_from_pool),
            "note": "the pool Final Score was selected AGAINST; a synthesized "
                    "oracle candidate is excluded by construction (§2.1)",
        },
        "synthesized_excluded_from_depth": synthesized_excluded,
        "exact_depth_dev_best": {str(k): v for k, v in sorted(exact.items())},
        "exact_depth_counts": {str(k): v for k, v in sorted(counts.items())},
        "iteration_archive_best": list(iteration_archive_best or []),
        "transition_denominators": {
            tr: {"attempted": v.get("edge_count"),
                 "valid_edge_task_pairs": v.get("valid_edge_task_pairs"),
                 "regressions": v.get("regressions")}
            for tr, v in (audit_metrics.get("transitions") or {}).items()},
        "R_valid": {tr: v.get("R_valid")
                    for tr, v in (audit_metrics.get("transitions") or {}).items()},
        "failure_accounting": dict(audit_metrics.get("failure_accounting") or {}),
        "ledger_terminal_counts": dict(ledger_counts),
        "resource_use": dict(resource_use),
    }
    forbid_best_chain_label(out)
    # §2.4: depth and iteration are separate series and must not be conflated
    out["depth_vs_iteration_note"] = (
        "exact_depth_* is structural (candidate.depth); iteration_archive_best "
        "is iteration-indexed. They are different series.")
    return out


# --------------------------------------------------------------------------
# §11 cross-seed aggregation
# --------------------------------------------------------------------------

@dataclass
class SeedRun:
    """One independent search seed's frozen per-run artifacts."""

    search_seed: int
    arm: str                       # "official" | "predictive"
    profile_sha256: str
    cohort_id: str
    root_bundle_sha256: str
    # §11: results from different backbones are NOT the same experiment, and the
    # aggregator must say which one it pooled. Without this the merged report
    # carried no model identity at all.
    backbone: str = ""
    # §9: the machine/process identity the run happened on, hashed from the root
    # bundle's `environment_fingerprint`. Two arms of one seed MUST share it --
    # scores produced on different machines are not a pair, however similar the
    # numbers look.
    environment_sha256: str = ""
    schema_version: int = METRICS_SCHEMA_VERSION
    final_macro: Optional[float] = None
    per_transition: Dict[str, Optional[float]] = field(default_factory=dict)
    payload: Dict[str, Any] = field(default_factory=dict)


def _assert_homogeneous(runs: Sequence[SeedRun]) -> None:
    for key in ("schema_version", "profile_sha256", "cohort_id", "backbone"):
        vals = {getattr(r, key) for r in runs}
        if len(vals) != 1:
            raise ProtocolError(
                "aggregator refuses mixed {}: {}".format(key, sorted(map(str, vals))))
    # One (arm, seed) may appear ONCE. A duplicate would silently overwrite in
    # the seed-keyed dicts below, so the run that happened to be read last would
    # decide the pooled number.
    seen: Dict[Tuple[str, int], int] = {}
    for r in runs:
        k = (str(r.arm), int(r.search_seed))
        seen[k] = seen.get(k, 0) + 1
    dup = {k: n for k, n in seen.items() if n > 1}
    if dup:
        raise ProtocolError(
            "aggregator refuses duplicate (arm, search_seed) run(s): {} -- one "
            "run per arm per seed; repeats are nested inside a seed".format(dup))
    # root hashes must correspond one-to-one across arms per seed, which is a
    # stronger statement than "all equal": the arms share a root PER SEED.
    # The environment must match across a seed's arms for the same reason (§9):
    # a number produced on another machine is not a pair, whatever it looks like.
    by_seed: Dict[int, Dict[str, set]] = {}
    for r in runs:
        by_seed.setdefault(r.search_seed, {}).setdefault(r.arm, set()).add(
            r.root_bundle_sha256)
    for seed, arms in by_seed.items():
        envs = {r.environment_sha256 for r in runs if r.search_seed == seed}
        if len(envs) > 1:
            raise ProtocolError(
                "seed {} did not share one environment across arms: {} -- "
                "cross-machine scores must never form a pair (§9)".format(
                    seed, sorted(envs)))
        if len(arms) < 2:
            continue
        roots = set()
        for a in arms.values():
            roots |= a
        if len(roots) != 1:
            raise ProtocolError(
                "seed {} did not share one root across arms: {}".format(
                    seed, {k: sorted(v) for k, v in arms.items()}))


def aggregate(runs: Sequence[SeedRun]) -> Dict[str, Any]:
    """§11 cross-seed summary. Repeats are nested and never counted as seeds."""
    if not runs:
        raise ProtocolError("nothing to aggregate")
    _assert_homogeneous(runs)

    by_arm: Dict[str, List[SeedRun]] = {}
    for r in runs:
        by_arm.setdefault(r.arm, []).append(r)

    per_arm: Dict[str, Any] = {}
    for arm, rs in sorted(by_arm.items()):
        finals = [r.final_macro for r in rs if r.final_macro is not None]
        per_arm[arm] = {
            "n_search_seeds": len(rs),
            "seeds": sorted(r.search_seed for r in rs),
            "final_mean": statistics.fmean(finals) if finals else None,
            "final_std": (statistics.stdev(finals) if len(finals) > 1 else 0.0)
                         if finals else None,
            "per_seed_transition_rates": {
                tr: [r.per_transition.get(tr) for r in
                     sorted(rs, key=lambda x: x.search_seed)]
                for tr in sorted({k for r in rs for k in r.per_transition})},
        }
        # macro mean over seeds, then uncertainty across seeds (§11)
        for tr in per_arm[arm]["per_seed_transition_rates"]:
            vals = [v for v in per_arm[arm]["per_seed_transition_rates"][tr]
                    if v is not None]
            per_arm[arm].setdefault("transition_macro", {})[tr] = {
                "mean": statistics.fmean(vals) if vals else None,
                "std": (statistics.stdev(vals) if len(vals) > 1 else 0.0)
                       if vals else None,
                "n_seeds": len(vals),
            }

    # paired Official - Predictive, per seed (§11)
    paired: Dict[str, Any] = {}
    off = {r.search_seed: r for r in by_arm.get("official", [])}
    pre = {r.search_seed: r for r in by_arm.get("predictive", [])}
    common = sorted(set(off) & set(pre))
    if common:
        diffs = []
        for s in common:
            a, b = off[s].final_macro, pre[s].final_macro
            if a is not None and b is not None:
                diffs.append(b - a)
        paired = {
            "seeds": common,
            "final_delta_mean": statistics.fmean(diffs) if diffs else None,
            "final_delta_std": (statistics.stdev(diffs) if len(diffs) > 1 else 0.0)
                               if diffs else None,
            "n_pairs": len(diffs),
            "note": "paired under the shared root seed; repeats stay nested",
        }
        for tr in sorted({k for r in runs for k in r.per_transition}):
            d = [pre[s].per_transition.get(tr) - off[s].per_transition.get(tr)
                 for s in common
                 if pre[s].per_transition.get(tr) is not None
                 and off[s].per_transition.get(tr) is not None]
            paired.setdefault("transition_delta", {})[tr] = {
                "mean": statistics.fmean(d) if d else None,
                "std": (statistics.stdev(d) if len(d) > 1 else 0.0) if d else None,
                "n_pairs": len(d),
            }

    # transparency only: pooled counts, never an estimator (§11)
    pooled: Dict[str, Any] = {}
    for tr in sorted({k for r in runs for k in r.per_transition}):
        pooled[tr] = {"per_seed_rates": [r.per_transition.get(tr) for r in runs]}
    return {
        "schema_version": METRICS_SCHEMA_VERSION,
        "cohort_id": runs[0].cohort_id,
        "profile_sha256": runs[0].profile_sha256,
        "backbone": runs[0].backbone,
        "environments": sorted({r.environment_sha256 for r in runs}),
        "n_runs": len(runs),
        "per_arm": per_arm,
        "paired": paired,
        "pooled_transparency_only": pooled,
        "notes": [
            "evaluation repeats are nested inside a seed and are never counted "
            "as independent search seeds (§3.3/§11)",
            "R_valid is primary; F and R_worst are robustness disclosures (§2.3)",
            "no `best chain` estimator exists; Final Score is one deployable "
            "candidate (§2.4/§7)",
        ],
    }


def reject_best_chain_estimator(aggregate_payload: Mapping[str, Any]) -> None:
    """Acceptance criterion 4 applied to the aggregate payload."""
    def walk(o, path="$"):
        if isinstance(o, Mapping):
            for k, v in o.items():
                if "chain" in str(k).lower():
                    raise ProtocolError(
                        "aggregate key {}.{} uses the retired `best chain` "
                        "framing".format(path, k))
                walk(v, path + "." + str(k))
        elif isinstance(o, (list, tuple)):
            for i, v in enumerate(o):
                walk(v, "{}[{}]".format(path, i))
    walk(aggregate_payload)


# --------------------------------------------------------------------------
# acceptance self-test (§14 items 9, 10, 11, 12)
# --------------------------------------------------------------------------

def self_test() -> int:
    print("metrics.py acceptance")
    print("=" * 68)

    # -- §14.10: one deployable, deterministic tie-break --------------------
    cands = [
        {"candidate_id": "gen0_seed", "depth": 1, "creation_index": 0},
        {"candidate_id": "gen1_b0_k0", "depth": 2, "creation_index": 1},
        {"candidate_id": "gen2_b0_k0", "depth": 3, "creation_index": 2},
    ]
    dev = {"gen0_seed": 0.70, "gen1_b0_k0": 0.80, "gen2_b0_k0": 0.80}
    sel = select_deployable(cands, lambda c: dev[c["candidate_id"]])
    assert sel.candidate_id == "gen1_b0_k0", sel
    # ...and lexicographic as the last resort
    tie = [{"candidate_id": "b", "depth": 2, "creation_index": 5},
           {"candidate_id": "a", "depth": 2, "creation_index": 5}]
    sel2 = select_deployable(tie, lambda c: 0.5)
    assert sel2.candidate_id == "a", sel2
    print("OK  single deployable  argmax by DEV; ties break on creation index "
          "then candidate id (never on test)")

    # the root is eligible
    sel3 = select_deployable([{"candidate_id": "gen0_seed", "depth": 1,
                               "creation_index": 0}], lambda c: 0.9)
    assert sel3.candidate_id == "gen0_seed"
    print("OK  root eligible      the root competes like any other candidate")

    # ---- the synthesized oracle may NOT become Final Score ---------------
    # `merge_oracle` has the highest score in the archive because it IS the
    # virtual oracle, and a fabricated depth of 2. Selecting it would publish a
    # diagnostic as Final Score (§2.1/§2.4).
    arch_with_oracle = list(cands) + [
        {"candidate_id": "merge_oracle", "depth": 2, "creation_index": 9,
         "mean_score": 0.99}]
    dev_o = dict(dev, merge_oracle=0.99)      # the oracle IS the maximum
    sel_o = select_deployable(arch_with_oracle,
                              lambda c: dev_o[c["candidate_id"]])
    assert sel_o.candidate_id == "gen1_b0_k0", sel_o
    assert any(e.get("reason") == "synthesized_oracle_not_deployable"
               for e in sel_o.excluded_from_pool)
    print("OK  oracle excluded    merge_oracle (highest dev score) is refused as "
          "Final Score and the exclusion is recorded")
    # ...and the depth series must not contain it either
    sel_all = select_deployable(arch_with_oracle, lambda c: dev_o[c["candidate_id"]],
                                allow_synthesized=True)
    assert sel_all.candidate_id == "merge_oracle"      # the pool WAS the guard
    print("OK  override explicit  allow_synthesized=True selects it, which proves "
          "the default was doing the work")

    # a candidate with no deployable material is excluded
    sel_x = select_deployable(cands, lambda c: dev[c["candidate_id"]],
                              is_executable_of=lambda c: c["candidate_id"] != "gen2_b0_k0")
    assert sel_x.candidate_id == "gen1_b0_k0", sel_x
    assert any(e.get("reason") == "no_executable_material"
               for e in sel_x.excluded_from_pool)
    print("OK  executability     a candidate with no material on disk is "
          "excluded, not deployed")

    # an unscored candidate is excluded with a reason
    try:
        select_deployable([{"candidate_id": "x", "depth": 2}], lambda c: None)
        raise AssertionError("an all-unscored archive produced a selection")
    except ProtocolError as e:
        assert "no deployable candidate" in str(e)
    print("OK  empty pool         an all-unscored archive is refused, not "
          "silently empty")

    # -- §14.12: the oracle cannot become Final Score ------------------------
    rec = final_record(sel, {"test_mean_score": 0.77,
                             "test_scores": {"T1": 0.8, "T2": 0.74}})
    assert rec["final_test_macro_score"] == 0.77
    assert rec["selected_candidate_id"] == "gen1_b0_k0"
    try:
        assert_no_oracle_as_final({"oracle_test_mean": 0.99})
        raise AssertionError("an oracle field passed as Final Score")
    except ProtocolError as e:
        assert "virtual-oracle" in str(e)
    # the explicitly named diagnostic is allowed
    assert_no_oracle_as_final({"oracle_upper_bound_dev": 0.99})
    print("OK  oracle excluded    an oracle field cannot be Final Score; the "
          "`oracle_upper_bound_dev` diagnostic is allowed")

    # -- acceptance 4: no `best chain` label --------------------------------
    try:
        forbid_best_chain_label({"chain_test_mean_score": 0.7})
        raise AssertionError("a `best chain` key was accepted")
    except ProtocolError as e:
        assert "best chain" in str(e)
    print("OK  no best-chain      a chain_test_* key is refused outright")

    # -- §14.9: exact depth is structural, iteration is separate -------------
    audit = {"transitions": {"R_2to3": {"edge_count": 4,
                                       "valid_edge_task_pairs": 24,
                                       "regressions": 3, "R_valid": 0.125}},
             "failure_accounting": {"F": 0.0, "R_worst": 0.125}}
    pr = per_run_metrics(selected=sel, test_result={"test_mean_score": 0.77},
                         audit_metrics=audit, ledger_counts={"evaluated_admitted": 24},
                         resource_use={"outer_calls": 30},
                         archive_candidates=cands,
                         dev_score_of=lambda c: dev[c["candidate_id"]],
                         iteration_archive_best=[{"iteration": 0, "best_dev": 0.7},
                                                 {"iteration": 1, "best_dev": 0.8}])
    assert pr["exact_depth_dev_best"]["3"] == 0.80
    assert pr["exact_depth_dev_best"]["1"] == 0.70
    assert pr["iteration_archive_best"][-1]["best_dev"] == 0.8
    assert "structural" in pr["depth_vs_iteration_note"]
    assert pr["R_valid"]["R_2to3"] == 0.125
    print("OK  depth vs iteration exact_depth_dev_best and "
          "iteration_archive_best are separate series")

    # the synthesized oracle is kept OUT of the depth series
    pr_o = per_run_metrics(
        selected=sel, test_result={"test_mean_score": 0.77},
        audit_metrics=audit, ledger_counts={}, resource_use={},
        archive_candidates=list(cands) + [
            {"candidate_id": "merge_oracle", "depth": 2, "creation_index": 9}],
        dev_score_of=lambda c: dev_o.get(c["candidate_id"], 0.0))
    assert pr_o["exact_depth_dev_best"]["2"] == 0.80, pr_o["exact_depth_dev_best"]
    assert pr_o["synthesized_excluded_from_depth"] == ["merge_oracle"]
    print("OK  depth uncontaminated merge_oracle (0.99 at fabricated depth 2) "
          "does not become the depth-2 best")

    # -- aggregation ---------------------------------------------------------
    runs = []
    for s in (0, 1, 2):
        for arm, f in (("official", 0.75), ("predictive", 0.78)):
            runs.append(SeedRun(search_seed=s, arm=arm,
                                profile_sha256="P" * 64,
                                cohort_id="sri_primary6:6",
                                root_bundle_sha256=("R{}".format(s) + "x" * 62),
                                final_macro=f + 0.001 * s,
                                per_transition={"R_2to3": 0.10 + 0.01 * s}))
    agg = aggregate(runs)
    assert agg["per_arm"]["official"]["n_search_seeds"] == 3
    assert abs(agg["paired"]["final_delta_mean"] - 0.03) < 1e-9, agg["paired"]
    assert agg["paired"]["n_pairs"] == 3
    assert agg["paired"]["transition_delta"]["R_2to3"]["n_pairs"] == 3
    reject_best_chain_estimator(agg)
    print("OK  aggregation        3 seeds x 2 arms; paired final delta = "
          "{:+.4f} over {} pairs".format(agg["paired"]["final_delta_mean"],
                                         agg["paired"]["n_pairs"]))
    print("OK  no chain key       the aggregate payload passes the retired-label "
          "check")

    # mixed cohorts / profiles are refused
    bad = list(runs) + [SeedRun(search_seed=3, arm="official",
                                profile_sha256="P" * 64,
                                cohort_id="sri_extended10:10",
                                root_bundle_sha256="y" * 64)]
    try:
        aggregate(bad)
        raise AssertionError("a mixed cohort was aggregated")
    except ProtocolError as e:
        assert "mixed cohort_id" in str(e)
    bad2 = list(runs) + [SeedRun(search_seed=3, arm="official",
                                 profile_sha256="Q" * 64,
                                 cohort_id="sri_primary6:6",
                                 root_bundle_sha256="z" * 64)]
    try:
        aggregate(bad2)
        raise AssertionError("a mixed profile was aggregated")
    except ProtocolError as e:
        assert "mixed profile_sha256" in str(e)
    print("OK  aggregator guards  mixed cohort_id and profile_sha256 refused")

    # arms that did not share a root PER SEED are refused
    bad3 = []
    for s in (0, 1):
        bad3.append(SeedRun(search_seed=s, arm="official",
                            profile_sha256="P" * 64, cohort_id="sri_primary6:6",
                            root_bundle_sha256="A" * 64, final_macro=0.7))
        bad3.append(SeedRun(search_seed=s, arm="predictive",
                            profile_sha256="P" * 64, cohort_id="sri_primary6:6",
                            root_bundle_sha256="B" * 64, final_macro=0.8))
    try:
        aggregate(bad3)
        raise AssertionError("arms with different roots were aggregated")
    except ProtocolError as e:
        assert "did not share one root" in str(e)
    print("OK  root pairing       arms without a shared per-seed root refused")

    # a duplicate (arm, seed) run is refused rather than silently overwriting
    try:
        aggregate(list(runs) + [SeedRun(search_seed=0, arm="official",
                                        profile_sha256="P" * 64,
                                        cohort_id="sri_primary6:6",
                                        root_bundle_sha256="R0" + "x" * 62,
                                        final_macro=0.0)])
        raise AssertionError("a duplicate (arm, seed) run was aggregated")
    except ProtocolError as e:
        assert "duplicate" in str(e)
    print("OK  duplicate seed      two runs for the same (arm, seed) refused")

    # mixing backbones is refused: they are not the same experiment
    try:
        aggregate(list(runs) + [SeedRun(search_seed=3, arm="official",
                                        profile_sha256="P" * 64,
                                        cohort_id="sri_primary6:6",
                                        root_bundle_sha256="R3" + "x" * 62,
                                        backbone="grok-4.6", final_macro=0.9)])
        raise AssertionError("two backbones were aggregated")
    except ProtocolError as e:
        assert "mixed backbone" in str(e)
    print("OK  backbone recorded   the aggregate names its backbone and refuses "
          "to pool two of them")

    # §9: two arms of one seed produced on different machines cannot pair
    cross = []
    for s in (0,):
        cross.append(SeedRun(search_seed=s, arm="official",
                             profile_sha256="P" * 64, cohort_id="sri_primary6:6",
                             root_bundle_sha256="A" * 64,
                             environment_sha256="E1" + "0" * 62,
                             final_macro=0.7))
        cross.append(SeedRun(search_seed=s, arm="predictive",
                             profile_sha256="P" * 64, cohort_id="sri_primary6:6",
                             root_bundle_sha256="A" * 64,
                             environment_sha256="E2" + "0" * 62,
                             final_macro=0.8))
    try:
        aggregate(cross)
        raise AssertionError("cross-machine arms were paired")
    except ProtocolError as e:
        assert "cross-machine" in str(e)
    print("OK  cross-machine      arms of one seed from two environments are "
          "refused as a pair")

    print()
    print("VERDICT: ALL OK")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(self_test())
