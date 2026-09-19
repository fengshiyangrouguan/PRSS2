#!/bin/bash
# One seed of the multi-seed sweep: a SINGLE uninterrupted run to 18000 steps.
#
# Rollout points are 12000 / 15000 / 18000 (18000 = the endpoint, saved as
# checkpoint.pt; snapshots cover 12000 and 15000).
#
# Usage:
#   HOST=/path/host.pt SEED=42 bash reproduce/train_sweep_18000.sh [arm]
# arm defaults to gamma-rpbe (ours). Pass gamma-task for the matched control.
#
# Disk: ~9.3 GB per run (2 snapshots + best + latest + checkpoint, no fullstate).
# Time: ~1.56 s/step -> 18000 steps ~= 7.8 h.
set -euo pipefail
ARM="${1:-gamma-rpbe}"
: "${SEED:?set SEED}"
: "${HOST:?set HOST to the host checkpoint}"
: "${BASE:=/root/autodl-tmp/openvla-7b-prismatic/checkpoints/step-295000-epoch-40-loss=0.2200.pt}"
: "${DR:=/root/autodl-tmp/libero-mem-no-noops}"
: "${OUT:=/root/autodl-tmp/outputs}"
RUN="${ARM}_seed${SEED}_18k"
export RPBE_EMBODIED_PATH=/root/autodl-tmp

[ -e "$OUT/$RUN" ] && { echo "refusing: $OUT/$RUN already exists"; exit 1; }
mkdir -p "$OUT/$RUN"

/root/env_eval/bin/python train_libero_mem_rpbe.py \
  --pretrained-checkpoint "$BASE" \
  --data-root "$DR" \
  --run-dir "$OUT/$RUN" \
  --arm "$ARM" \
  --task-filter KITCHEN_SCENE1_3 \
  --reg-head 0 \
  --train-scope gamma-only \
  --max-steps 18000 --batch-size 4 --grad-accum 4 \
  --lr 1e-5 --gamma-lr 2e-5 --sched const --warmup-steps 0 \
  --mem-length 16 --repeated-diffusion-steps 4 --future-action-window-size 15 \
  --eval-every 1000 --checkpoint-every 1000 --log-every 50 \
  --snapshot-steps 12000,15000 --no-fullstate 1 \
  --rpbe-mode project --kappa 0.02 --proj-tau 3e-4 \
  --proj-iters 400 --proj-iters-max 1600 \
  --seed "$SEED" --image-aug 1 --dim-weight 0 \
  --kf-min-abs 64 --gamma-replay-batch-size 64 \
  --gamma-task-boundary-episodes 1 --rpbe-stats-episodes 4 \
  --init-from-weights "$HOST" \
  > "$OUT/$RUN/train.log" 2>&1
