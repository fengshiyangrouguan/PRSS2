#!/bin/bash
# T1 · RPBE arm — our method (Gamma merge + task replay + TPG-isomorphic
# multi-halfspace feasibility projection).
# Recipe identical to the avg baseline (t1_avg_seed42) except --arm.
cd /root/autodl-tmp/PRSS2/third_party/memoryvla || exit 1
export PYTHONPATH=/root/autodl-tmp/PRSS2/src
export RPBE_EMBODIED_PATH=/root/autodl-tmp/PRSS2/src
export LLAMA2_LOCAL_PATH=/root/autodl-tmp/Llama-2-7b-hf
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ulimit -n 65536
PY=/root/autodl-tmp/env_memvla/bin/python
RUN=/root/autodl-tmp/runs/t1_rpbe_seed42

mkdir -p "$RUN"
echo "=== T1 RPBE LAUNCH $(date -Is) ===" | tee -a "$RUN/run.log"
echo "arm = gamma-rpbe (Gamma merge + task replay + RPBE projection)" | tee -a "$RUN/run.log"

$PY -u train_libero_mem_rpbe.py \
  --pretrained-checkpoint /root/autodl-tmp/openvla-7b-prismatic/checkpoints/step-295000-epoch-40-loss=0.2200.pt \
  --data-root /root/autodl-tmp/datasets/LIBERO-Mem \
  --task-filter KITCHEN_SCENE1_1 \
  --arm gamma-rpbe \
  --run-dir "$RUN" \
  --max-steps 20000 \
  --eval-every 2000 \
  --checkpoint-every 1000 \
  --snapshot-steps "10000,15000,20000" \
  --no-fullstate 1 \
  --seed 42 \
  2>&1 | tee -a "$RUN/run.log"
echo "=== T1 RPBE EXIT rc=${PIPESTATUS[0]} $(date -Is) ===" | tee -a "$RUN/run.log"
