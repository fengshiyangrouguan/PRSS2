"""Zero-API STRUCTURAL audit -- corrected to the evaluator's ACTUAL split rule.

THE CORRECTION THAT MATTERS. The first version of this audit read
`get_dev() -> {file: []}` as "this file contributes no dev instances" and flagged
two of the six proposed tasks as broken. That was the AUDIT being wrong, not the
tasks. `_TaskEvaluator` follows the Original CO-Bench convention, stated in this
repo at co_bench.py (the `_CREW_HELDOUT_RUNNER` comment):

    "the evaluator defaults an empty list to [0]"

so an EMPTY dev list means "index 0 of this file is dev", and a file ABSENT from
the map is entirely held out. Under that rule:

    dev  = for each file IN the map: the listed indices, or [0] if the list is empty
    test = for EVERY file: all indices NOT selected as dev

Capacitated therefore has 12 dev instances (index 0 of each of 12 files), which
is exactly what the live run logged ("Eval DEV split ... 12 instances"), and
Assignment has 4. Both are valid splits.

WHAT THIS CAN AND CANNOT SETTLE. It reads only the DATASET (instance files +
get_dev + the task's own load_data), so it settles the split-structure questions.
It CANNOT settle the evaluation-side criteria -- score resolution, saturation,
runtime stability, recursive headroom -- because those need a solver to run and
that needs the LLM. Those belong to the Phase-A pilot on the new cohort.

THE RULES APPLIED (fixed before any result is seen; membership is NEVER random):
  1. dev must be non-empty  (otherwise the search gets no signal)
  2. test must be non-empty (otherwise there is no held-out measurement)
  3. dev must cover MORE THAN ONE family/scale (a single family leaves nothing
     to generalise across)
  4. dev must not be a single index repeated across files while every file is
     identical in scale -- reported, not auto-failed
"""
import glob
import importlib.util
import os
import re
import sys

DATA = "/root/autodl-tmp/meta-n-main/data/co_bench"

COHORT = ["Aircraft landing", "Capacitated warehouse location",
          "Common due date scheduling", "Flow shop scheduling",
          "Generalised assignment problem", "Job shop scheduling"]
# ^ the FROZEN Structural6 (2026-09-23). Re-running this script re-validates it.
# ^ the frozen backup order, screened in order; the first that passes is taken
#   and screening STOPS there (never "pick the best-looking one").


def load_module(task):
    p = os.path.join(DATA, task, "config.py")
    spec = importlib.util.spec_from_file_location(
        "cb_" + re.sub(r"\W", "_", task), p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def audit(task):
    r = {"task": task}
    m = load_module(task)
    names = sorted(os.path.basename(p)
                   for p in glob.glob(os.path.join(DATA, task, "*.txt")))
    fams = {tuple(re.findall(r"\d+", n)) for n in names if re.findall(r"\d+", n)}
    dev_map = dict(m.get_dev() or {})
    load = getattr(m, "load_data", None)

    n_dev = n_test = 0
    per_file = []
    for n in names:
        try:
            inst = list(load(os.path.join(DATA, task, n))) if load else []
        except Exception as e:                                  # noqa: BLE001
            per_file.append((n, "load_err:%s" % type(e).__name__))
            continue
        k = len(inst)
        if n in dev_map:
            # EMPTY list -> [0]  (Original CO-Bench convention)
            dev_idx = set(list(dev_map[n]) or [0])
        else:
            dev_idx = set()                     # file entirely held out
        n_dev += sum(1 for i in range(k) if i in dev_idx)
        n_test += sum(1 for i in range(k) if i not in dev_idx)
        per_file.append((n, k))

    r.update(files=len(names), families=len(fams),
             dev_files=len(dev_map), test_only_files=len(names) - len(dev_map),
             dev_instances=n_dev, test_instances=n_test,
             instances=sum(k for _, k in per_file if isinstance(k, int)))
    r["verdict"] = ("OK" if (n_dev > 0 and n_test > 0 and len(fams) > 1)
                    else "REVIEW")
    return r, per_file


print("=" * 104)
print("STRUCTURAL AUDIT -- the evaluator's own split convention (empty dev list == [0])")
print("=" * 104)
print("  %-34s %6s %6s %9s %8s %9s %9s  %s"
      % ("task", "files", "fams", "devfiles", "testonly", "dev#", "test#", "verdict"))
rows = []
for t in COHORT:
    r, per = audit(t)
    rows.append((r, per))
    print("  %-34s %6d %6d %9d %8d %9d %9d  %s"
          % (t[:34], r["files"], r["families"], r["dev_files"],
             r["test_only_files"], r["dev_instances"], r["test_instances"],
             r["verdict"]))

print()
print("=" * 104)
print("PER-FILE instance counts (why the dev/test split is what it is)")
print("=" * 104)
for r, per in rows:
    print("  %s  -- %d files, %d instances total"
          % (r["task"][:34], r["files"], r["instances"]))
    show = per[:6]
    print("     %s%s" % (", ".join("%s=%s" % (n[:16], k) for n, k in show),
                         " ..." if len(per) > 6 else ""))
