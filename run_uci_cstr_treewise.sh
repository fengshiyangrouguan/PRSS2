#!/bin/bash
# UCI constrained RPBE -- FINAL SPEC: tree-wise multi-half-space QP.
#
# Algorithm (replaces the rejected "aggregate" plan-B projection):
#   * every per-(tree, interface) RPBE direction g_{i,tau} is kept SEPARATE
#     (the aggregate form flattens them into one half space, where tree-level
#     conflicts g_1 + g_2 ~ 0 cancel and the projection never fires);
#   * group close solves, with t the aggregate task gradient on the
#     correction scope,
#         min_d 1/2 ||d - t||^2   s.t.  g_{i,tau}^T d >= -kappa ||g_{i,tau}|| ||t||
#     via its dual by FISTA (accelerated projected gradient, spectral step);
#   * the correction is restricted to Gamma/compressor params only -- every
#     other host param keeps the plain task gradient.
#
# kappa = 0.05 chosen from the tree-wise cosine probe (tw_probe5, seed0,
# 3 epochs, baseline trajectory): frac(cos < -kappa) = 0.099 / 0.058 / 0.023
# over epochs 0-2, i.e. it constrains the worst 2-10% of trees; kappa = 0.10
# would constrain < 1%.  cos_mean ~ 0 with ~50% of trees at cos < 0 is the
# cancellation the aggregate form could not see.
#
# Budget: 80 epochs, patience 80, lambda = 3hop calibrated 0.00937.
# Usage: bash run_uci_cstr_treewise.sh [seed ...]   (default: 1 2)
set -u
cd /root/autodl-tmp/PRSS2_uci_v2 || exit 1
export PYTHONPATH=/root/autodl-tmp/PRSS2_uci_v2:/root/autodl-tmp/benchtemp/experimental_codes/tgn-jodie-dyrep
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/root/miniconda3/bin/python
ROOT=/root/autodl-tmp/PRSS2_uci_v2/outputs/uci_formal_v2
DATA_DIR=/root/autodl-tmp/benchtemp/data_uci
LR=1e-4
REPR_LR=3e-4
EPOCHS=80
PAT=80
LAM=0.00937285625636343
KAPPA=0.05
MODE=treewise
SCOPE=gamma
SEEDS="${*:-1 2}"
mkdir -p "$ROOT/logs"

gpu_used_mb() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1
}

wait_gpu() {
  local limit="$1"
  while :; do
    local used
    used=$(gpu_used_mb)
    if [ -n "$used" ] && [ "$used" -lt "$limit" ]; then return 0; fi
    sleep 20
  done
}

run_ours_cstr_treewise() {
  local seed="$1"
  local out="$ROOT/seed${seed}_TGN_3hop/ours_cstr_${MODE}_e80"
  if [ -f "$out/summary.json" ] && grep -q '"stop_reason": "budget"' "$out/summary.json"; then
    echo "SKIP seed=$seed (already complete)" >> "$ROOT/run_treewise.log"; return 0
  fi
  if pgrep -f "train_uci_link.*ours_cstr_${MODE}_e80.*--seed $seed" >/dev/null 2>&1; then
    echo "BUSY seed=$seed" >> "$ROOT/run_treewise.log"; return 0
  fi
  mkdir -p "$out"
  echo "START seed=$seed mode=$MODE scope=$SCOPE kappa=$KAPPA epochs=$EPOCHS $(date '+%H:%M:%S') commit=$(git rev-parse --short HEAD)" \
    > "$ROOT/logs/seed${seed}_ours_cstr_${MODE}_e80.log"
  $PY -m scripts.train_uci_link \
    --arm 2obs_aligned --lambda-kf "$LAM" \
    --rpbe-constrain --rpbe-constrain-mode "$MODE" \
    --rpbe-constrain-scope "$SCOPE" --rpbe-kappa "$KAPPA" \
    --data-dir "$DATA_DIR" --gpu 0 \
    --epochs "$EPOCHS" --budget-cap "$EPOCHS" --patience "$PAT" \
    --kf-group-batches 40 --n-neighbors 10 --n-layers 3 \
    --lr "$LR" --repr-lr "$REPR_LR" \
    --seed "$seed" --output "$out" >> "$ROOT/logs/seed${seed}_ours_cstr_${MODE}_e80.log" 2>&1
  echo "DONE seed=$seed rc=$? $(date '+%H:%M:%S')" >> "$ROOT/logs/seed${seed}_ours_cstr_${MODE}_e80.log"
}

for s in $SEEDS; do
  wait_gpu 6000
  run_ours_cstr_treewise "$s" &
  sleep 25
done
wait
echo "ALL_DONE_TREEWISE seeds=$SEEDS $(date '+%H:%M:%S')" >> "$ROOT/run_treewise.log"
