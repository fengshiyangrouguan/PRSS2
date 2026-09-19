#!/bin/bash
# The rollout metric: weighted_success = sum(tier)/(3*n), 20 demos.
# Note: the recovered command said --max-steps 15000 for s8h, but training ran
# with a /10000 denominator (absolute 15000 -> 25000); set it to what you want.
#
# *** DO NOT PIPE THIS THROUGH A SUMMARY grep. ***
# The Stage8 runs did, and that is exactly why no per-demo identity survived.
# Keep the full stdout: the "[demo_N] tier=X/3 ..." lines are the evidence.
set -e
: "${CKPT:?set CKPT}"
: "${STATS:?set STATS to the dataset_statistics.json from the same --data-root/--task-filter}"
cp "$STATS" "$(dirname "$CKPT")/"
/root/env_eval/bin/python tiered_eval.py \
  --ckpt "$CKPT" \
  --demo-start 1 --n 20 --exec 8 --maxsteps 370 --seed 42 \
  >> rollout_full.log 2>&1          # full output, unfiltered
tail -1 rollout_full.log
