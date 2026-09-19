#!/bin/bash
# Drive the multi-seed sweep: train every (arm, seed) pair, at most N at a time.
#
# Usage:
#   HOST=/path/host.pt SEEDS="42 7 123" ARMS="gamma-rpbe" GPUS="0 1" \
#     bash reproduce/sweep_launch.sh
#
# One process per GPU. After ALL training finishes, run eval_sweep_18000.sh per
# run (not concurrently -- see the note in that script about
# dataset_statistics.json).
set -uo pipefail
: "${HOST:?set HOST}"
: "${SEEDS:?set SEEDS, e.g. \"42 7 123\"}"
: "${ARMS:=gamma-rpbe}"
: "${GPUS:=0 1}"
HERE="$(cd "$(dirname "$0")" && pwd)"

read -ra G <<<"$GPUS"
i=0
for arm in $ARMS; do
  for seed in $SEEDS; do
    gpu="${G[$((i % ${#G[@]}))]}"
    echo "== $arm seed=$seed -> GPU $gpu"
    CUDA_VISIBLE_DEVICES="$gpu" HOST="$HOST" SEED="$seed" \
      nohup bash "$HERE/train_sweep_18000.sh" "$arm" \
      > "/root/autodl-tmp/outputs/${arm}_seed${seed}_18k.launch.log" 2>&1 &
    i=$((i+1))
    # keep at most ${#G[@]} training processes alive
    while [ "$(jobs -rp | wc -l)" -ge "${#G[@]}" ]; do sleep 30; done
  done
done
wait
echo "all trainings finished; now run eval_sweep_18000.sh for each run"
