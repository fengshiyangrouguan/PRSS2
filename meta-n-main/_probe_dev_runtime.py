"""Measure per-task DEV evaluation wall-clock, with a local stub solver (0 API).

WHY THIS IS A SCREENING CRITERION AND NOT A NICETY. A CO-Bench dev score is
"#instances finishing inside the 10s timeout / #dev instances". The evaluator
runs EVERY dev instance, so the dev split SIZE sets the wall-clock of every
single candidate evaluation, and the search performs ~4 evaluations per
iteration x T iterations x 2 arms of them. A task with a 675-instance dev split
therefore costs ~169x the wall-clock of one with 4 -- and because each instance
is capped by a WALL-CLOCK timeout, a heavy dev split also makes the score far
more sensitive to machine load, which is exactly the confound the profile pins
`instance_workers` to avoid.

THE STUB SPINS, IT DOES NOT FAIL FAST. A solver that raises immediately would
understate the cost completely, because most of the expense is instances that
run until the timeout. Measured on the live runs, timeouts dominate (e.g.
"12 instances, 9 errors (9 timeouts)"), so the stub burns a little under the
10s budget per instance -- the realistic upper bound.

NO LLM IS CALLED: the evaluator only RUNS the solver source it is given.
"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CODEBERT_PATH", "/root/autodl-tmp/models/codebert-base")
# MUST be a Path: `_TaskEvaluator.__init__` does `data_dir / task_name`,
# which raises `str / str` if this is a string.
DATA = Path("/root/autodl-tmp/meta-n-main/data/co_bench")
NAME_OF = {
    "aircraft_landing": "Aircraft landing",
    "assignment_problem": "Assignment problem",
    "capacitated_warehouse_location": "Capacitated warehouse location",
    "common_due_date_scheduling": "Common due date scheduling",
    "flow_shop_scheduling": "Flow shop scheduling",
    "generalised_assignment_problem": "Generalised assignment problem",
    "hybrid_reentrant_shop_scheduling": "Hybrid Reentrant Shop Scheduling",
    "job_shop_scheduling": "Job shop scheduling",
}

# Spins until just under the 10s instance budget, then returns something
# harmless. Any exception is fine too -- the point is the WALL-CLOCK.
STUB = '''
import time
def solve(**kwargs):
    t0 = time.time()
    while time.time() - t0 < 9.5:
        pass
    return {}
'''

sys.path.insert(0, "/root/autodl-tmp/rpbe-sri/meta-n-main")
from meta_n.integrations.co_bench import _TaskEvaluator          # noqa: E402

TASKS = sys.argv[1:] or [
    "hybrid_reentrant_shop_scheduling", "job_shop_scheduling",
    "flow_shop_scheduling", "generalised_assignment_problem",
]

print("=" * 92)
print("DEV-split wall-clock per candidate evaluation (stub spins ~9.5s/instance)")
print("=" * 92)
print("  %-40s %8s %10s %12s" % ("task", "dev#", "seconds", "s/instance"))
tot = 0.0
for slug in TASKS:
    name = NAME_OF.get(slug, slug)
    t0 = time.time()
    out = None
    try:
        ev = _TaskEvaluator(name, DATA, timeout=10, instance_workers=8)
        out = ev.evaluate(STUB)
    except Exception as e:                                      # noqa: BLE001
        # EXPECTED. The stub returns `{}`, so the task's own norm_score cannot
        # divide its fields and raises -- but `norm_score` runs AFTER every
        # instance has been executed, so the wall-clock measured below is still
        # the real evaluation cost. Failing fast (raising inside the solver)
        # would have measured nothing, which is why the stub spins first.
        why = "%s: %s" % (type(e).__name__, str(e)[:50])
    else:
        why = ""
    finally:
        dt = time.time() - t0
    tot += dt
    raw = len(out["raw_results"]) if isinstance(out, dict) and isinstance(
        out.get("raw_results"), (list, tuple, dict)) else None
    print("  %-40s %8s %10.1f %12s  %s"
          % (name[:40], raw if raw is not None else "?",
             dt, ("%.2f" % (dt / raw)) if raw else "?", why))
print()
print("  TOTAL measured: %.1f s = %.1f min" % (tot, tot / 60.0))
print("  A search evaluation covers ALL cohort tasks, so the per-evaluation cost")
print("  is the SUM over the cohort; the search does that ~4x/iteration.")
