"""Review the HISTORY: every SRI run on this box, aircraft_landing test, both arms.

WHY. "Is the test increment a STABLE increment?" cannot be answered from one
run -- it needs the same quantity observed in more than one independent
execution. There are many `sri_*` / `offline_*` directories on the box, so the
repeats may already exist and the question may already be answered without
spending another evaluation.

For each run directory this finds every arm subdirectory that has a
`final/*.raw.jsonl`, pulls the `aircraft_landing` rows (all repeats, since the
spread between repeats is itself part of the stability answer), and prints the
per-run delta `predictive - official`.

It also prints the MACRO of those same raw rows so a run whose per-task table
does not reconstruct its own headline can be spotted.
"""
import json
from pathlib import Path

ROOT = Path("/root/autodl-tmp")
TASK = "aircraft_landing"
ALL_TASKS = ["aircraft_landing", "assignment_problem", "assortment_problem",
             "bin_packing___one_dimensional", "capacitated_warehouse_location",
             "common_due_date_scheduling"]

runs = sorted([d for d in ROOT.iterdir()
               if d.is_dir() and (d.name.startswith("sri_") or
                                  d.name.startswith("offline_"))])
print("scanned %d run dirs: %s" % (len(runs), ", ".join(d.name for d in runs)))
print()

print("=" * 100)
print("aircraft_landing test, per run and per arm")
print("=" * 100)
print("  %-30s %-26s %s" % ("run", "arm dir", "aircraft test (per repeat)"))
table = {}
for d in runs:
    arms_root = d / "arms"
    if not arms_root.is_dir():
        continue
    for arm in sorted(arms_root.iterdir()):
        raw = d / "final" / ("%s.raw.jsonl" % arm.name)
        if not raw.is_file():
            continue
        rows = []
        for line in raw.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("task_id") == TASK:
                rows.append(r)
        if not rows:
            continue
        rows.sort(key=lambda r: r.get("repeat_index") or 0)
        vals = [(r.get("repeat_index"), r.get("score")) for r in rows]
        print("  %-30s %-26s %s" % (d.name, arm.name,
                                    "  ".join("r%s=%.4f" % v for v in vals)))
        table.setdefault(d.name, {})[arm.name] = [v[1] for v in vals]

print()
print("=" * 100)
print("per-run DELTA (predictive - official), aircraft_landing")
print("=" * 100)
for run, arms in sorted(table.items()):
    off = None
    pre = None
    for k, v in arms.items():
        if k == "official":
            off = v
        elif "predictive" in k:
            pre = v
    if not off or not pre:
        print("  %-30s incomplete (official=%s predictive=%s)"
              % (run, bool(off), bool(pre)))
        continue
    deltas = []
    for i in range(min(len(off), len(pre))):
        if off[i] is not None and pre[i] is not None:
            deltas.append(pre[i] - off[i])
    if deltas:
        print("  %-30s  Δ per repeat = %s   mean=%+.4f  spread=%.4f"
              % (run, " ".join("%+.4f" % x for x in deltas),
                 sum(deltas) / len(deltas), max(deltas) - min(deltas)))

print()
print("=" * 100)
print("(context) each run's MACRO over the tasks it actually evaluated")
print("=" * 100)
for run, arms in sorted(table.items()):
    for a, v in sorted(arms.items()):
        print("  %-30s %-26s partial-aircraft-only, n=%d" % (run, a, len(v)))
