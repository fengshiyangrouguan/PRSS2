#!/bin/bash
# develop_UCI — plan-B formal: constrained RPBE ours (kappa=0.05) first,
# 3 seeds x 3 parallel, then taskonly (matched code baseline) 3 seeds.
# 50 epochs run to the full budget (patience=50), repr-lr compensated,
# lambda = 3hop calibrated 0.00937 (the projection is lambda-invariant,
# the additive surrogate keeps its value for diagnostics).
set -u
cd /root/autodl-tmp/PRSS2_uci_v2 || exit 1
export PYTHONPATH=/root/autodl-tmp/PRSS2_uci_v2:/root/autodl-tmp/benchtemp/experimental_codes/tgn-jodie-dyrep
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/root/miniconda3/bin/python
ROOT=/root/autodl-tmp/PRSS2_uci_v2/outputs/uci_formal_v2
DATA_DIR=/root/autodl-tmp/benchtemp/data_uci
LR=1e-4
REPR_LR=3e-4
EPOCHS=50
PAT=50
LAM=0.00937285625636343
KAPPA=0.05
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

confirm_ramp() {
  local pid="$1" before="$2"
  for _ in $(seq 1 24); do
    if ! kill -0 "$pid" 2>/dev/null; then return 0; fi
    sleep 10
    local used
    used=$(gpu_used_mb)
    if [ -n "$used" ] && [ "$used" -ge $((before + 3000)) ]; then return 0; fi
  done
  return 0
}

skip_or_busy() {
  local out="$1" pat="$2" label="$3"
  if [ -f "$out/summary.json" ] && grep -q '"stop_reason": "budget"' "$out/summary.json"; then
    echo "SKIP $label" >> "$ROOT/run.log"; return 0
  fi
  if [ -f "$out/summary.json" ] && ! grep -q '"stop_reason": "budget"' "$out/summary.json"; then
    rm -f "$out/summary.json"
    echo "RESET $label (non-budget residue)" >> "$ROOT/run.log"
  fi
  if pgrep -f "$pat" >/dev/null 2>&1; then
    echo "BUSY $label" >> "$ROOT/run.log"; return 0
  fi
  return 1
}

run_ours_cstr() {
  local seed="$1"
  local out="$ROOT/seed${seed}_TGN_3hop/ours_cstr_e50"
  skip_or_busy "$out" "train_uci_link.*seed${seed}_TGN_3hop/ours_cstr_e50" "ours_cstr s$seed" && return 0
  mkdir -p "$out"
  echo "START ours_cstr seed=$seed kappa=$KAPPA $(date '+%H:%M:%S') commit=$(git rev-parse --short HEAD)" \
    > "$ROOT/logs/seed${seed}_ours_cstr_e50.log"
  $PY -m scripts.train_uci_link \
    --arm 2obs_aligned --lambda-kf "$LAM" \
    --rpbe-constrain --rpbe-kappa "$KAPPA" \
    --data-dir "$DATA_DIR" --gpu 0 \
    --epochs "$EPOCHS" --budget-cap "$EPOCHS" --patience "$PAT" \
    --kf-group-batches 40 --n-neighbors 10 --n-layers 3 \
    --lr "$LR" --repr-lr "$REPR_LR" \
    --seed "$seed" --output "$out" >> "$ROOT/logs/seed${seed}_ours_cstr_e50.log" 2>&1
  echo "DONE_ours_cstr_s${seed} rc=$? $(date '+%H:%M:%S')" >> "$ROOT/logs/seed${seed}_ours_cstr_e50.log"
}

run_taskonly() {
  local seed="$1"
  local out="$ROOT/seed${seed}_TGN_3hop/taskonly_cstr_e50"
  skip_or_busy "$out" "train_uci_link.*seed${seed}_TGN_3hop/taskonly_cstr_e50" "taskonly_cstr s$seed" && return 0
  mkdir -p "$out"
  echo "START taskonly_cstr seed=$seed $(date '+%H:%M:%S') commit=$(git rev-parse --short HEAD)" \
    > "$ROOT/logs/seed${seed}_taskonly_cstr_e50.log"
  $PY -m scripts.train_uci_link \
    --config P0 \
    --data-dir "$DATA_DIR" --gpu 0 \
    --epochs "$EPOCHS" --budget-cap "$EPOCHS" --patience "$PAT" \
    --kf-group-batches 40 --n-neighbors 10 --n-layers 3 \
    --lr "$LR" --repr-lr "$REPR_LR" \
    --seed "$seed" --output "$out" >> "$ROOT/logs/seed${seed}_taskonly_cstr_e50.log" 2>&1
  echo "DONE_taskonly_cstr_s${seed} rc=$? $(date '+%H:%M:%S')" >> "$ROOT/logs/seed${seed}_taskonly_cstr_e50.log"
}

# ---- phase 1: ours (constrained), 3 seeds x 3 parallel ----
for s in 0 1 2; do
  wait_gpu 6000
  before=$(gpu_used_mb)
  run_ours_cstr "$s" & pid=$!
  confirm_ramp "$pid" "$before"
done
wait
echo "ALL_DONE_CSTR_OURS $(date '+%H:%M:%S')" >> "$ROOT/run.log"

# ---- phase 2: taskonly (matched baseline), 3 seeds x 3 parallel ----
for s in 0 1 2; do
  wait_gpu 6000
  before=$(gpu_used_mb)
  run_taskonly "$s" & pid=$!
  confirm_ramp "$pid" "$before"
done
wait
echo "ALL_DONE_CSTR_TASKONLY $(date '+%H:%M:%S')" >> "$ROOT/run.log"
