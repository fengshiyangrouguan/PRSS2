"""Which task actually carries the increment? Zero API, reads the archives.

The diagnostic task was assumed to be `capacitated_warehouse_location`. That
assumption was never checked, and it is checkable for free: both arms' archives
carry per-task dev scores, so the root -> selected-candidate increment can be
read off directly for every task.

Three views, because "the biggest increment" can mean different things:
  A. per-task dev delta of the DEV-SELECTED candidate vs the shared root,
     per arm  -- how much did the search move THIS task?
  B. per-task dev delta of the archive BEST vs the root -- the same question
     without the deployable-selection filter.
  C. per-task ARM GAP (predictive - official) at the selected candidate --
     where do the two arms actually differ on dev?
"""
import json
from pathlib import Path

OUT = Path("/root/autodl-tmp/sri_r3_seed0")
ARMS = {"official": "official", "predictive": "_predictive_OLD_selector"}


def cands(arm):
    p = OUT / "arms" / ARMS[arm] / "archive" / "index.json"
    return json.loads(p.read_text()).get("candidates") or []


def root_of(cs):
    r = [c for c in cs if int(c.get("depth") or 0) == 1]
    return r[0] if r else None


def best_of(cs):
    return max(cs, key=lambda c: float(c.get("mean_score") or -1))


def tasks_of(cs):
    t = set()
    for c in cs:
        t |= set((c.get("per_task_scores") or {}).keys())
    return sorted(t)


data = {}
for arm in ARMS:
    cs = cands(arm)
    data[arm] = {"cands": cs, "root": root_of(cs), "best": best_of(cs),
                 "tasks": tasks_of(cs)}

tasks = sorted(set(data["official"]["tasks"]) |
               set(data["predictive"]["tasks"]))

print("=" * 96)
print("A. DEV-SELECTED candidate minus shared root, per task")
print("=" * 96)
print("  %-36s %12s %12s %12s" % ("task", "official", "predictive", "arm gap"))
r_o = data["official"]["root"]["per_task_scores"]
r_p = data["predictive"]["root"]["per_task_scores"]
s_o = data["official"]["best"]["per_task_scores"]
s_p = data["predictive"]["best"]["per_task_scores"]
rows = []
for t in tasks:
    do = float(s_o.get(t, 0)) - float(r_o.get(t, 0))
    dp = float(s_p.get(t, 0)) - float(r_p.get(t, 0))
    rows.append((t, do, dp, dp - do))
for t, do, dp, gap in sorted(rows, key=lambda x: -max(abs(x[1]), abs(x[2]))):
    print("  %-36s %+12.4f %+12.4f %+12.4f" % (t, do, dp, gap))

print()
print("=" * 96)
print("B. dev values: root -> selected  (absolute, so the level is visible too)")
print("=" * 96)
print("  %-36s %24s %24s" % ("task", "official root->sel", "predictive root->sel"))
for t, do, dp, gap in sorted(rows, key=lambda x: -max(abs(x[1]), abs(x[2]))):
    print("  %-36s %10.4f -> %-10.4f %10.4f -> %-10.4f"
          % (t, float(r_o.get(t, 0)), float(s_o.get(t, 0)),
             float(r_p.get(t, 0)), float(s_p.get(t, 0))))

print()
print("=" * 96)
print("C. macro dev, for scale")
print("=" * 96)
for arm in ARMS:
    d = data[arm]
    print("  %-12s root=%.4f  selected=%.4f  archive_best=%.4f"
          % (arm, float(d["root"]["mean_score"]), float(d["best"]["mean_score"]),
             max(float(c["mean_score"]) for c in d["cands"])))
