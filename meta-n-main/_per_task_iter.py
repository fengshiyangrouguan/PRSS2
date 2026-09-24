"""Per-task dev, iteration by iteration, for BOTH arms of the original run.

Two questions, one table each:
  1. what does each TASK's dev look like at each iteration (so a task that moves
     can be told apart from one that is pinned from the start);
  2. the macro progression, which is what `_metrics_seed0.py` reads.

The iteration comes from the candidate id: `gen{N}_b{beam}_k{k}` -- N is the
generation the candidate was bred in, and `gen0_seed` is the shared root. The
row shown for iteration N is the ARCHIVE-BEST-so-far at that point (the same
quantity `convergence.json` records), so the per-task columns always describe
one real candidate rather than a stitched-together maximum.
"""
import json
import sys
from pathlib import Path

# argv[1] = run dir, argv[2] = the predictive arm's DIRECTORY NAME (it is
# `_predictive_OLD_selector` in the original run, plain `predictive` in the
# cross-backbone legs -- the arm's MEANING is the same, only the folder differs,
# and hardcoding one silently reports the other's numbers as "ours").
OUT = Path(sys.argv[1] if len(sys.argv) > 1
           else "/root/autodl-tmp/sri_r3_seed0")
PRED = sys.argv[2] if len(sys.argv) > 2 else "_predictive_OLD_selector"
ARMS = {"official": "official", "predictive": PRED}
TASKS = ["aircraft_landing", "assignment_problem", "assortment_problem",
         "bin_packing___one_dimensional", "capacitated_warehouse_location",
         "common_due_date_scheduling"]
SHORT = ["aircraft", "assign", "assortmnt", "binpack", "capacit", "commondue"]


def gen_of(cid):
    if cid.startswith("gen0"):
        return 0
    try:
        return int(cid.split("_")[0][3:])
    except Exception:                                          # noqa: BLE001
        return 99


for arm, sub in ARMS.items():
    p = OUT / "arms" / sub / "archive" / "index.json"
    if not p.is_file():
        print("%s: MISSING" % arm)
        continue
    cs = json.loads(p.read_text())["candidates"]
    for c in cs:
        c["_gen"] = gen_of(c["candidate_id"])

    print("=" * 118)
    print("ARM %s   (%d candidates)" % (arm, len(cs)))
    print("=" * 118)
    hdr = "  %-4s %-16s %-9s " % ("iter", "archive-best", "mean")
    hdr += " ".join("%-9s" % s for s in SHORT)
    print(hdr)

    best = None
    for g in sorted({c["_gen"] for c in cs}):
        pool = [c for c in cs if c["_gen"] == g]
        top = max(pool, key=lambda c: float(c["mean_score"]))
        if best is None or float(top["mean_score"]) > float(best["mean_score"]):
            best = top
        pt = best.get("per_task_scores") or {}
        row = "  %-4d %-16s %-9.4f " % (g, best["candidate_id"],
                                        float(best["mean_score"]))
        row += " ".join("%-9s" % ("%.4f" % float(pt.get(t, float("nan"))))
                        for t in TASKS)
        print(row + ("   <- new best" if top is best else ""))

    print()
    print("  per-task min..max across ALL candidates (where the task is alive):")
    for t, s in zip(TASKS, SHORT):
        vals = [float((c.get("per_task_scores") or {}).get(t, float("nan")))
                for c in cs]
        vals = [v for v in vals if v == v]
        if not vals:
            continue
        print("     %-12s min=%.4f  max=%.4f   n_distinct=%d"
              % (s, min(vals), max(vals), len({round(v, 4) for v in vals})))
    print()
