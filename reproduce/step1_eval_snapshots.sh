#!/bin/bash
# STEP 1b -- find the first host snapshot that actually rolls out.
# Run this WHILE step 1 trains (or after).  Each eval is ~6 min.
#
# usage: RUN=host_full bash reproduce/step1_eval_snapshots.sh
set -euo pipefail
: "${RUN:?set RUN}"
: "${OUT:=/root/autodl-tmp/outputs}"
D="$OUT/$RUN"; E="$OUT/eval_$RUN"; mkdir -p "$E"
mkdir -p "$E"; cp -n "$D/dataset_statistics.json" "$E/" 2>/dev/null || true
for s in 5000 10000 15000 20000 25000 30000 35000; do
  CK="$D/snapshot_$s.pt"; [ -f "$CK" ] || continue
  grep -q "=== snapshot_$s ===" "$E/rollout.log" 2>/dev/null && continue
  echo "=== snapshot_$s ===" >> "$E/rollout.log"
  cp "$D/dataset_statistics.json" "$(dirname "$CK")/" 2>/dev/null || true
  /root/env_eval/bin/python tiered_eval.py --ckpt "$CK" \
    --demo-start 1 --n 20 --exec 8 --maxsteps 370 --seed 42 >> "$E/rollout.log" 2>&1
  echo "--- done snapshot_$s ---" >> "$E/rollout.log"
done
python "$(dirname "$0")/tier_table.py" "$E/rollout.log" "$E" || true
echo
echo "STOP RULE: take the FIRST snapshot whose weighted_success > 0 as your host."
