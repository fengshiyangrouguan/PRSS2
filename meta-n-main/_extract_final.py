"""Zero-API extraction: is `capacitated_warehouse_location`'s per-repeat TEST
score already sitting in the existing final artifacts?

The question this answers, before any money is spent: the original Gemini run
did a FULL final (6 tasks x 3 repeats), so that task's test rows should already
exist. Re-evaluating them would be paying twice for a number already on disk.

IT ALSO CHECKS WHOSE FINAL THIS IS. `_ours_rerun.sh` re-ran the predictive arm
(swapping in the SiblingAllocator) and then continued through
freeze/audit/final, so `final/predictive.json` may have been OVERWRITTEN and no
longer describe the `_predictive_OLD_selector` archive that produced the
0.9547 / +0.1094 numbers. The tell is `oracle_upper_bound_dev`: it should equal
the arm archive's own best dev.

Usage: python _extract_final.py
"""
import json
from pathlib import Path

OUT = Path("/root/autodl-tmp/sri_r3_seed0")
TASK = "capacitated_warehouse_location"


def arm_best(arm):
    p = OUT / "arms" / arm / "archive" / "index.json"
    if not p.is_file():
        return None
    cs = json.loads(p.read_text()).get("candidates") or []
    return max((float(c["mean_score"]) for c in cs), default=None), len(cs)


print("=" * 80)
print("arm archive-best dev vs the final record's oracle_upper_bound_dev")
print("=" * 80)
for arm in ("official", "_predictive_OLD_selector", "predictive"):
    b = arm_best(arm)
    print("  %-26s archive_best=%s  n=%s" % (arm, b[0], b[1]))

print()
print("=" * 80)
print("final/<arm>.json : which archive does it describe?")
print("=" * 80)
for arm in ("official", "predictive"):
    p = OUT / "final" / ("%s.json" % arm)
    if not p.is_file():
        print("  %s: MISSING" % arm)
        continue
    d = json.loads(p.read_text())
    print("  %-12s oracle_upper_bound_dev = %s" % (arm, d.get("oracle_upper_bound_dev")))
    print("               selected_material_sha256 = %s" % str(
        d.get("selected_material_sha256"))[:24])
    fin = d.get("final")
    if isinstance(fin, dict):
        print("               final sub-keys: %s" % sorted(fin.keys())[:12])
        for k in ("per_task", "per_task_scores", "raw", "repeats", "seeds",
                  "test_macro", "macro"):
            if k in fin:
                print("               fin[%s] = %s" % (k, json.dumps(fin[k])[:400]))

print()
print("=" * 80)
print("raw rows for %s" % TASK)
print("=" * 80)
for arm in ("official", "predictive"):
    p = OUT / "final" / ("%s.raw.jsonl" % arm)
    if not p.is_file():
        print("  %s: no raw file" % arm)
        continue
    hits = []
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("task_id") == TASK:
            hits.append(r)
    print("  %-12s %d row(s)" % (arm, len(hits)))
    for r in sorted(hits, key=lambda x: x.get("repeat_index") or 0):
        print("      repeat=%s seed=%s score=%s success=%s fresh=%s"
              % (r.get("repeat_index"), r.get("seed"), r.get("score"),
                 r.get("success"), r.get("freshly_executed")))

print()
print("=" * 80)
print("does the old final exist anywhere else (backups)?")
print("=" * 80)
import subprocess
r = subprocess.run(
    ["bash", "-lc",
     "find /root/autodl-tmp -maxdepth 5 -path '*sri_r3*' -name '*.json' "
     "2>/dev/null | grep -iE 'final|aggregate' | head -30"],
    capture_output=True, text=True)
print(r.stdout or "  (none found)")
