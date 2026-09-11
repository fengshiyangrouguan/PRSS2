#!/bin/bash
# develop_UCI — e50 extension: ours + taskonly, 3hop AND 2hop, 5 seeds each,
# run to the FULL 50 epochs (patience=50 == no early stop; selection = best
# of 50).  Lambda is NOT recalibrated — the per-hop calibrated values are
# reused (3hop 0.00937 / 2hop 0.01559; taskonly has no aux term).
# Outputs to {ours,taskonly}_e50/ so the 30-epoch results stay intact.
#
# Phase A: 2hop (fast, ~40 min/seed) then Phase B: 3hop (~3 h/seed).
# 2 parallel per seed round; memory gate + SKIP/BUSY idempotent.
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
LAM_3HOP=0.00937285625636343
LAM_2HOP=0.015585656981550107
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

# run_one HOP SEED ARM-NAME [extra args...]
run_one() {
  local hop="$1" seed="$2" armname="$3"; shift 3
  local out="$ROOT/seed${seed}_TGN_${hop}/${armname}_e50"
  skip_or_busy "$out" "train_uci_link.*seed${seed}_TGN_${hop}/${armname}_e50" "${armname}_e50 ${hop} s$seed" && return 0
  mkdir -p "$out"
  echo "START ${armname}_e50 hop=$hop seed=$seed $(date '+%H:%M:%S') commit=$(git rev-parse --short HEAD)" \
    > "$ROOT/logs/seed${seed}_${armname}_e50_${hop}.log"
  $PY -m scripts.train_uci_link \
    --data-dir "$DATA_DIR" --gpu 0 \
    --epochs "$EPOCHS" --budget-cap "$EPOCHS" --patience "$PAT" \
    --kf-group-batches 40 --n-neighbors 10 \
    --lr "$LR" --repr-lr "$REPR_LR" \
    --seed "$seed" --output "$out" "$@" \
    >> "$ROOT/logs/seed${seed}_${armname}_e50_${hop}.log" 2>&1
  echo "DONE_${armname}_e50_${hop}_s${seed} rc=$? $(date '+%H:%M:%S')" >> "$ROOT/logs/seed${seed}_${armname}_e50_${hop}.log"
}

# per-seed round: ours + taskonly in parallel (same hop)
run_round() {
  local hop="$1" seed="$2" lam="$3"
  echo "=== e50 ${hop} seed $seed round $(date '+%H:%M:%S') ===" >> "$ROOT/run.log"
  wait_gpu 8000
  before=$(gpu_used_mb)
  run_one "$hop" "$seed" ours --arm 2obs_aligned --lambda-kf "$lam" --n-layers "$([ "$hop" = 2hop ] && echo 2 || echo 3)" &
  pid1=$!
  confirm_ramp "$pid1" "$before"
  b2=$(gpu_used_mb)
  run_one "$hop" "$seed" taskonly --config P0 --n-layers "$([ "$hop" = 2hop ] && echo 2 || echo 3)" &
  pid2=$!
  confirm_ramp "$pid2" "$b2"
  wait
  echo "e50 ${hop} seed $seed round done $(date '+%H:%M:%S')" >> "$ROOT/run.log"
}

# ---- Phase A: 2hop ----
for s in 0 1 2 3 4; do
  run_round 2hop "$s" "$LAM_2HOP"
done
echo "ALL_DONE_E50_2HOP $(date '+%H:%M:%S')" >> "$ROOT/run.log"

# ---- Phase B: 3hop ----
for s in 0 1 2 3 4; do
  run_round 3hop "$s" "$LAM_3HOP"
done
echo "ALL_DONE_E50_3HOP $(date '+%H:%M:%S')" >> "$ROOT/run.log"
