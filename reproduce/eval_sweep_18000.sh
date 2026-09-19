#!/bin/bash
# Rollout for the three sweep points. RUN IT AFTER TRAINING FINISHES.
#
# WHY AFTER, NOT DURING: tiered_eval.py reads dataset_statistics.json from the
# checkpoint's own directory, and this trainer only writes that file when the
# run completes. Evaluating a mid-run snapshot therefore dies with
# FileNotFoundError -- that is exactly what happened to us in Stage8, and the
# workaround then was to hand-copy the file from another run.
#
# The eval output is NOT piped through a grep: the "[demo_N] tier=X/3 ..."
# lines are the per-demo evidence and must reach the file (see
# ../results/PER_DEMO_RECOVERY.md for what losing them costs).
#
# Usage: RUN=gamma-rpbe_seed42_18k bash reproduce/eval_sweep_18000.sh
set -euo pipefail
: "${RUN:?set RUN to the run directory name}"
: "${OUT:=/root/autodl-tmp/outputs}"
D="$OUT/$RUN"
E="$OUT/eval_$RUN"
mkdir -p "$E"
[ -f "$D/dataset_statistics.json" ] || {
  echo "refusing: $D/dataset_statistics.json missing -- training has not finished"; exit 1; }

for tag in snapshot_12000 snapshot_15000 checkpoint; do
  CK="$D/$tag.pt"
  [ -f "$CK" ] || { echo "skip $tag (missing)"; continue; }
  echo "=== $RUN :: $tag ===" >> "$E/rollout_full.log"
  /root/env_eval/bin/python tiered_eval.py \
    --ckpt "$CK" \
    --demo-start 1 --n 20 --exec 8 --maxsteps 370 --seed 42 \
    >> "$E/rollout_full.log" 2>&1
  echo "--- done $tag ---" >> "$E/rollout_full.log"
done
echo "wrote $E/rollout_full.log"

# req 3: per-demo table -- how many times EACH demo completed the 3-cycle task,
# one column per checkpoint.
python "$(dirname "$0")/tier_table.py" "$E/rollout_full.log" "$E"

# req 2: train loss + val loss, one row per 1000 steps
python "$(dirname "$0")/extract_curves.py" "$D/train.log" "$E/curves.csv"
