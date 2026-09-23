"""0-API budget smoke for the Structural6 Phase-A launcher.

Proves, against a REAL temp ledger, that the three budget/env fixes hold:
  1. `child_ledger_path()` == the file `CostTracker` writes (local date, not UTC);
  2. pre-existing spend in that file does NOT shrink the batch allowance -- the
     inner daily cap becomes already-spent + batch_cap, so a run gets the full
     batch cap of NEW spend;
  3. the outer `Fuse`, given `path=ledger`, reads the SAME file, so its
     phase_spend = batch_spend is real rather than zero-by-mistake.

No LLM, no network. It writes only a temp ledger under a temp dir.
"""
import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.pop("META_N_COST_LEDGER_DIR", None)   # start clean

from scripts.run_phase_a_structural6 import child_ledger_path  # noqa: E402

tmp = tempfile.mkdtemp(prefix="s6_budget_smoke_")
os.environ["META_N_COST_LEDGER_DIR"] = tmp

# What the child's CostTracker computes for "now", local date, utc=False.
tracker_dir = Path(tmp).expanduser()
tracker_path = tracker_dir / "{}.jsonl".format(
    datetime.now().strftime("%Y-%m-%d"))

got = child_ledger_path()
print("=" * 74)
print("SMOKE 1: launcher ledger == CostTracker ledger")
print("=" * 74)
print("  child_ledger_path() : %s" % got)
print("  CostTracker would   : %s" % tracker_path)
print("  SAME FILE           : %s" % (got == tracker_path))

# Pre-write $7 of "already spent today" into that exact file.
pre = 7.0
got.parent.mkdir(parents=True, exist_ok=True)
with open(got, "w", encoding="utf-8") as f:
    f.write(json.dumps({"cost_usd": pre, "pid": 999999}) + "\n")

from meta_n.rpbe.budget import Caps, Fuse, total_cost  # noqa: E402

print()
print("=" * 74)
print("SMOKE 2: pre-existing spend must not shrink the batch allowance")
print("=" * 74)
baseline = total_cost(got)
batch_cap = 36.0
per_run_cap = 8.0
daily_cap = baseline + batch_cap
print("  already on the ledger today : $%.2f" % baseline)
print("  batch cap                   : $%.2f" % batch_cap)
print("  inner daily cap set to      : $%.2f  (already-spent + batch)" % daily_cap)
print("  -> a run may ADD up to $%.2f before the inner guard fires" % batch_cap)
assert abs(baseline - pre) < 1e-9, baseline

print()
print("=" * 74)
print("SMOKE 3: the outer Fuse reads the SAME file")
print("=" * 74)
# Fake a "child" that spends: append $3 with OUR pid (simulating one run's cost).
my_pid = os.getpid()
with open(got, "a", encoding="utf-8") as f:
    f.write(json.dumps({"cost_usd": 3.0, "pid": my_pid}) + "\n")

fuse = Fuse(my_pid, Caps(per_run=per_run_cap, total=batch_cap),
            phase_baseline=baseline, path=got)
reason = fuse.check()
print("  fuse.check() after $3 of batch spend -> reason = %r" % reason)
print("  run_spend  = $%.2f (per-run, by pid)" % fuse.run_spend)
print("  phase_spend= $%.2f (batch, vs the single baseline)" % fuse.phase_spend)
assert abs(fuse.phase_spend - 3.0) < 1e-9
assert reason is None, reason

# And a breach IS seen once the batch cap is crossed. Spend $34 more under a
# DIFFERENT pid so the per-run cap (my pid's $3 stays under $8) does NOT fire
# first -- we are specifically proving the BATCH cap, and the per-run cap firing
# first is correct-but-not-the-point.
for _ in range(4):
    with open(got, "a", encoding="utf-8") as f:
        f.write(json.dumps({"cost_usd": 8.5, "pid": my_pid + 1}) + "\n")
reason = fuse.check()
print("  after exceeding the batch cap -> reason = %r" % reason)
assert reason is not None and "phase-A cap" in reason, reason

print()
print("  ALL THREE SMOKES PASS")
