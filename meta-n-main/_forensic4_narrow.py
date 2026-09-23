"""FORENSIC #4 (narrow): only the rows that are NOT already known.

ALREADY KNOWN, not re-measured (from final/<arm>.json, the shipped winners):
    official gen6_b0_k1  d6  dev 0.9001  test 0.8338
    ours     gen6_b1_k1  d4  dev 0.9547  test 0.8390
and from the earlier run of the wide script:
    official gen4_b1_k0  d4  dev 0.8832  test 0.8244

WHAT IS STILL MISSING, and why exactly these:

  A. ours' depth-5 candidates vs their own depth-4 parents.
     The shipped winner is a d4 candidate with NO d5 descendant (it appeared at
     generation 6, the last iteration, so nothing extended it). So the user's
     original "run the d4 winner's d5 descendant" is impossible. The nearest
     available test is the d5 candidates that DO exist, against their parents:
        gen5_b0_k1 (d4, 0.9468) -> gen6_b0_k1 (d5, 0.9161)   big dev drop
        gen3_b0_k0 (d4, 0.9227) -> gen4_b0_k0 (d5, 0.9227)   IDENTICAL dev
     The second pair is a clean controlled comparison: same dev, one layer
     deeper. If its test is HIGHER, the extra depth generalises better while
     dev-max cannot see it -- winner's curse. If test is also lower, depth is
     not the problem.

  B. official's own depth curve, to compare like with like.
     gen3_b0_k1 (d4 ancestry, dev 0.8813) -> gen5_b1_k1 (d5, dev 0.8953) ->
     gen6_b0_k1 (d6, test 0.8338, known).

Every row is flushed to disk as soon as it is computed, so partial results are
usable and an interruption loses only the row in flight.
"""
import argparse
import importlib.util
import json
import math
import statistics
import sys
from pathlib import Path

OUT = Path("/root/autodl-tmp/sri_r3_seed0")
REPO = Path("/root/autodl-tmp/rpbe-sri/meta-n-main")
RESULT = OUT / "forensic4_narrow.jsonl"
sys.path.insert(0, str(REPO))

from meta_n.sri.protocol import (default_profile_path, load_profile,  # noqa: E402
                                 sha256_of)

spec = importlib.util.spec_from_file_location(
    "sri_runner", REPO / "scripts" / "run_sri_formal.py")
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)

PROFILE = load_profile(default_profile_path("sri_primary6"))
SLUGS, NAME_OF = R.cohort_ids(PROFILE)
ARGS = argparse.Namespace(evaluator="real",
                          data_dir="/root/autodl-tmp/meta-n-main/data/co_bench",
                          timeout=10, search_seed=0)
EV = R.make_evaluator(ARGS, PROFILE, split="test")

# (arm, candidate, depth, s_dev, why)
TODO = [
    ("predictive", "gen5_b0_k1", 4, 0.9468, "ours d4 parent of the d5 drop"),
    ("predictive", "gen6_b0_k1", 5, 0.9161, "ours d5 child  (dev -0.0307)"),
    ("predictive", "gen3_b0_k0", 4, 0.9227, "ours d4 of the IDENTICAL-dev pair"),
    ("predictive", "gen4_b0_k0", 5, 0.9227, "ours d5 of the IDENTICAL-dev pair"),
    ("official", "gen3_b0_k1", 4, 0.8813, "official d4 of the shipped ancestry"),
    ("official", "gen5_b1_k1", 5, 0.8953, "official d5 of the shipped ancestry"),
]

done = {}
if RESULT.is_file():
    for line in RESULT.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            done[(r["arm"], r["candidate"])] = r

print("=" * 96)
print("FORENSIC #4 NARROW -- %d rows to compute, %d already cached"
      % (len(TODO), len(done)))
print("=" * 96, flush=True)

for arm, cid, depth, sdev, why in TODO:
    if (arm, cid) in done:
        continue
    mat = R.load_material_from_dir(OUT / "arms" / arm / "archive" / cid,
                                   cid, depth, SLUGS)
    per = {}
    for t in SLUGS:
        src = (mat.task_scripts or {}).get(t) if mat is not None else None
        if not src:
            per[t] = None
            continue
        vals = []
        for rep in range(int(PROFILE.test_repeats)):
            seed = int(sha256_of({"s": ARGS.search_seed, "t": t,
                                  "r": rep})[:8], 16) & 0x7FFFFFFF
            ev = EV.evaluate_test(t, src, seed=seed)
            if ev.score is None or not math.isfinite(float(ev.score)):
                continue
            vals.append(float(ev.score))
        per[t] = statistics.median(vals) if vals else None
    got = [v for v in per.values() if v is not None]
    stest = sum(got) / len(got) if got else float("nan")
    rec = {"arm": arm, "candidate": cid, "depth": depth, "s_dev": sdev,
           "s_test": stest, "dev_minus_test": sdev - stest,
           "per_task": per, "why": why}
    with open(RESULT, "a", encoding="utf-8") as f:      # flush per row
        f.write(json.dumps(rec) + "\n")
        f.flush()
    print("  DONE %-11s %-13s d=%d  s_dev=%.4f  s_test=%.4f  (dev-test %+.4f)"
          % (arm, cid, depth, sdev, stest, sdev - stest), flush=True)

print()
print("=" * 96)
print("ALL ROWS (including the ones measured earlier / read from final/*.json)")
print("=" * 96)
KNOWN = [
    ("official", "gen4_b1_k0", 4, 0.8832, 0.8244, "official d4 winner (dev-max)"),
    ("official", "gen5_b1_k1", 5, 0.8953, None, "official d5 ancestry"),
    ("official", "gen6_b0_k1", 6, 0.9001, 0.8338, "official SHIPPED winner"),
    ("predictive", "gen5_b0_k1", 4, 0.9468, None, "ours d4 parent"),
    ("predictive", "gen6_b0_k1", 5, 0.9161, None, "ours d5 child"),
    ("predictive", "gen3_b0_k0", 4, 0.9227, None, "ours d4 identical-dev"),
    ("predictive", "gen4_b0_k0", 5, 0.9227, None, "ours d5 identical-dev"),
    ("predictive", "gen6_b1_k1", 4, 0.9547, 0.8390, "ours SHIPPED winner"),
]
rows = {}
for line in RESULT.read_text(encoding="utf-8").splitlines() if RESULT.is_file() else []:
    if line.strip():
        r = json.loads(line)
        rows[(r["arm"], r["candidate"])] = (r["s_dev"], r["s_test"], r["why"])
for arm, cid, d, sd, st, why in KNOWN:
    if (arm, cid) in rows:
        sd, st, _ = rows[(arm, cid)]
    print("  %-11s %-13s d=%d  s_dev=%.4f  s_test=%s  dev-test=%s  %s"
          % (arm, cid, d, sd,
             "%.4f" % st if st is not None else "  --  ",
             "%+.4f" % (sd - st) if st is not None else "   --  ", why))
