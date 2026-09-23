"""(SUPERSEDED -- will NOT run against current HEAD.)

This dry-run targeted the epsilon-band + leave-one-task-out selector, which was
BUILT, tested (E/F/G all passed) and then REVERTED on purpose: the dry run on
this same archive showed it selected the SAME winner on both arms, so it fixed
nothing while adding method surface -- and changing the allocator and the final
selector together would have made any later improvement unattributable. It is
kept as the evidence for THAT decision, not as a runnable tool: it calls
`select_deployable_robust`, which no longer exists. The dev->held-out
mis-ranking it was meant to address (dev 0.9547 -> test 0.8390 vs dev 0.9468 ->
test 0.8775) is recorded as a DIAGNOSTIC of seed 0 and was NOT used to redesign
the rule.

---

DRY RUN of the new final selector on the OLD seed-0 archive (0 API).

Purpose is NOT to see whether it happens to land on the candidate we now know
scores 0.8775 on held-out -- that would be reverse-engineering the rule from the
answer, which the frozen protocol forbids. The purpose is to verify the rule
works and to show that the winner is decided by MULTI-TASK STABILITY rather than
by a 0.0079 dev gap.

Prints, per candidate: depth, generation, macro dev, gap to best, band
membership, the six leave-one-task-out means, the robust score, and whether it
was selected. Writes each arm's selection_manifest.json into a DRY-RUN
directory so the real `final/` artifacts are untouched.
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

OUT = Path("/root/autodl-tmp/sri_r3_seed0")
DRY = OUT / "dryrun_selection"
REPO = Path("/root/autodl-tmp/rpbe-sri/meta-n-main")
sys.path.insert(0, str(REPO))

from meta_n.sri.protocol import default_profile_path, load_profile  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "sri_runner", REPO / "scripts" / "run_sri_formal.py")
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)

PROFILE = load_profile(default_profile_path("sri_primary6"))
SLUGS, _ = R.cohort_ids(PROFILE)
EPS = float(PROFILE.regression_threshold)
DRY.mkdir(parents=True, exist_ok=True)

print("=" * 108)
print("FINAL SELECTOR DRY RUN -- epsilon=%.3f (profile.regression_threshold), "
      "rule=epsilon_band + leave-one-task-out min" % EPS)
print("=" * 108)

for arm in ("official", "predictive"):
    a = OUT / "arms" / arm
    idx = json.load(open(a / "archive" / "index.json", encoding="utf-8"))
    cands = list(idx.get("candidates") or [])
    for i, c in enumerate(cands):
        c["creation_index"] = i

    def executable(c, _a=a):
        m = R.load_material_from_dir(
            _a / "archive" / str(c.get("candidate_id")),
            str(c.get("candidate_id")), int(c.get("depth") or 1), SLUGS)
        return m is not None and m.is_executable_on(SLUGS)

    sel, man = R.select_deployable_robust(
        cands, epsilon=EPS, cohort=SLUGS,
        dev_score_of=lambda c: c.get("mean_score"),
        task_scores_of=lambda c: c.get("per_task_scores") or {},
        creation_index_of=lambda c: c["creation_index"],
        is_executable_of=executable)
    man["arm"] = arm
    man["epsilon_source"] = "profile.regression_threshold"
    (DRY / ("%s.selection_manifest.json" % arm)).write_text(
        json.dumps(man, indent=2, sort_keys=True), encoding="utf-8")

    print()
    print("  %s   (pool=%d deployable, band=%d)" %
          (arm, man["pool_size"], len(man["band_candidate_ids"])))
    print("    max_dev = %.4f   band: %s"
          % (man["max_dev"], ",".join(man["band_candidate_ids"])))
    print()
    print("    %-14s %5s %5s %9s %9s %5s  %s"
          % ("candidate", "depth", "gen", "macro_dev", "gap_best", "band",
             "  ".join("LOO_%d" % (i + 1) for i in range(len(SLUGS)))))
    print("    " + "-" * 100)
    rows = []
    for cid in sorted(man["per_candidate_dev"]):
        dev = man["per_candidate_dev"][cid]
        dep = man["per_candidate_depth"][cid]
        loo = man["per_candidate_loo_means"].get(cid) or {}
        rb = man["robust_score"].get(cid)
        gen = ""
        for c in cands:
            if str(c.get("candidate_id")) == cid:
                gen = str(c.get("candidate_id", "").split("_")[0])
                break
        rows.append((cid, dep, gen, dev, dev - man["max_dev"],
                     cid in man["band_candidate_ids"], loo, rb,
                     cid == man["selected_candidate_id"]))
    for cid, dep, gen, dev, gap, inb, loo, rb, is_sel in sorted(
            rows, key=lambda r: -r[3]):
        cells = "  ".join("%7.4f" % loo[t] if t in loo else "      -"
                          for t in SLUGS)
        print("    %-14s %5d %5s %9.4f %9.4f %5s  %s%s"
              % (cid, dep, gen, dev, gap, "YES" if inb else "no", cells,
                 "   <== SELECTED" if is_sel else ""))
    print()
    print("    selected = %s   robust = %.4f   tie_break = %s"
          % (man["selected_candidate_id"], man["selected_robust_score"] or -1,
             man["tie_break_reason"]))

print()
print("=" * 108)
print("  Manifests written to %s" % DRY)
print("  NOTE: 'gen' is the generation prefix of the candidate id; the selector")
print("  itself never reads it except as a determinism tie-break.")
print("  The rule is DEV-ONLY: no test score entered this decision.")
