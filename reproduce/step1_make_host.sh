#!/bin/bash
# STEP 1 of 2 -- build a host that can actually do the T3 task.
#
# Why this step exists: the base checkpoint does NOT contain MemoryVLA's DiT
# action head or memory banks (loading it prints "No ActionModel found ...
# Initializing a new one.").  So the base cannot do the task, and stage 2
# freezes the host -- there would be nothing to preserve.  This is where task
# ability is created, and the evidence says it comes from training the MEMORY
# + DiT, not LoRA: our historical host had param_version = 0 (LoRA never
# stepped) yet rolled out at 23.3%.
#
# `--arm avg` = plain average merging, i.e. the model WITHOUT RPBE.  That is
# what "host" means here.
#
# WAIT -- do not train blind.  `full` scope degrades a working model if run
# too long (measured: a 23.3% host pushed to 0.0-3.3% by the same recipe).
# Snapshots land every 5000 steps; evaluate them and stop at the first one
# that moves.
#
# Usage: HOSTOUT=/root/autodl-tmp/outputs/host_full bash reproduce/step1_make_host.sh
set -euo pipefail
: "${BASE:?set BASE to the openvla-7b-prismatic checkpoint}"
: "${DR:?set DR to the data root}"
: "${OUT:=/root/autodl-tmp/outputs}"
RUN="${RUN:-host_full}"
export RPBE_EMBODIED_PATH=/root/autodl-tmp

[ -e "$OUT/$RUN" ] && { echo "refusing: $OUT/$RUN exists"; exit 1; }
mkdir -p "$OUT/$RUN"

/root/env_eval/bin/python train_libero_mem_rpbe.py \
  --pretrained-checkpoint "$BASE" \
  --data-root "$DR" \
  --run-dir "$OUT/$RUN" \
  --arm avg \
  --task-filter KITCHEN_SCENE1_3 \
  --reg-head 0 \
  --train-scope full \
  --max-steps "${STEPS:-40000}" --batch-size 4 --grad-accum 4 \
  --lr 2e-5 --sched const --warmup-steps 0 \
  --mem-length 16 --repeated-diffusion-steps 4 --future-action-window-size 15 \
  --eval-every 1000 --checkpoint-every 1000 --log-every 50 \
  --snapshot-steps 5000,10000,15000,20000,25000,30000,35000 --no-fullstate 1 \
  --seed 42 --image-aug 1 --dim-weight 0 \
  > "$OUT/$RUN/train.log" 2>&1
