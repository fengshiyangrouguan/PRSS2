#!/bin/bash
# OURS, stage 2 of 2: the weight-initialised continuation that produced the
# 25.0% peak at absolute step 17000 (= local 2000).
# Command recovered VERBATIM from the transcript.
#
# This is a WEIGHT warm start, not a bit-exact resume: stage 1 ran with
# --no-fullstate 1, so there is no optimizer/scheduler/RNG state to continue
# from. The optimiser state at the start of this stage is fresh.
set -e
: "${BASE:=/root/autodl-tmp/openvla-7b-prismatic/checkpoints/step-295000-epoch-40-loss=0.2200.pt}"
: "${DR:=/root/autodl-tmp/libero-mem-no-noops}"
: "${S8G:?set S8G to the stage-1 checkpoint (…/gamma-only_ours_s8g/checkpoint.pt)}"
export RPBE_EMBODIED_PATH=/root/autodl-tmp

/root/env_eval/bin/python train_libero_mem_rpbe.py \
  --pretrained-checkpoint "$BASE" \
  --data-root "$DR" \
  --run-dir /root/autodl-tmp/outputs/gamma-only_ours_s8h \
  --arm gamma-rpbe --task-filter KITCHEN_SCENE1_3 --reg-head 0 \
  --train-scope gamma-only \
  --max-steps 10000 --batch-size 4 --grad-accum 4 \
  --lr 1e-5 --gamma-lr 2e-5 --sched const --warmup-steps 0 \
  --eval-every 500 --checkpoint-every 1000 --log-every 50 \
  --snapshot-steps 3000,6000,10000 --no-fullstate 1 \
  --rpbe-mode project --kappa 0.02 --proj-tau 3e-4 \
  --proj-iters 400 --proj-iters-max 1600 \
  --init-from-weights "$S8G"
