#!/bin/bash
# OURS, stage 1 of 2: the RPBE run itself, steps 0 -> 15000.
# This is the run whose continuation produced the 25.0% peak.
# Flags recovered verbatim from the session transcript; the s8g script was
# never committed (it lived only on the box).
#
# !!! INIT REQUIREMENT: the host checkpoint is LOST (see ../REPRODUCE.md).
#     Substitute your own host at HOST=... below.
set -e
: "${BASE:=/root/autodl-tmp/openvla-7b-prismatic/checkpoints/step-295000-epoch-40-loss=0.2200.pt}"
: "${DR:=/root/autodl-tmp/libero-mem-no-noops}"
: "${HOST:?set HOST to your Stage5-avg host checkpoint}"
export RPBE_EMBODIED_PATH=/root/autodl-tmp

/root/env_eval/bin/python train_libero_mem_rpbe.py \
  --pretrained-checkpoint "$BASE" \
  --data-root "$DR" \
  --run-dir /root/autodl-tmp/outputs/gamma-only_ours_s8g \
  --arm gamma-rpbe \
  --task-filter KITCHEN_SCENE1_3 \
  --reg-head 0 \
  --train-scope gamma-only \
  --max-steps 15000 --batch-size 4 --grad-accum 4 \
  --lr 1e-5 --gamma-lr 2e-5 --sched const --warmup-steps 0 \
  --mem-length 16 --repeated-diffusion-steps 4 --future-action-window-size 15 \
  --eval-every 500 --checkpoint-every 1000 --log-every 50 \
  --snapshot-steps 1000,3000,6000,10000 --no-fullstate 1 \
  --rpbe-mode project --kappa 0.02 --proj-tau 3e-4 \
  --proj-iters 400 --proj-iters-max 1600 \
  --seed 42 --image-aug 1 --dim-weight 0 \
  --kf-min-abs 64 --gamma-replay-batch-size 64 \
  --gamma-task-boundary-episodes 1 --rpbe-stats-episodes 4 \
  --init-from-weights "$HOST"
