#!/bin/bash
# develop_UCI — e50 extension, 3hop ONLY: ours + taskonly, 5 seeds each,
# run to the FULL 50 epochs (patience=50 == no early stop; selection = best
# of 50).  Lambda reuses the 3hop calibrated value 0.00937 (no recalib —
# the 30-epoch 3hop line used the same structure).  Outputs to
# {ours,taskonly}_e50/ so the 30-epoch results stay intact.
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

run_one() {
  local seed="$1" armname="$2"; shift 2
  local out="$ROOT/seed${seed}_TGN_3hop/${armname}_e50"
  skip_or_busy "$out" "train_uci_link.*seed${seed}_TGN_3hop/${armname}_e50" "${armname}_e50 s$seed" && return 0
  mkdir -p "$out"
  echo "START ${armname}_e50 3hop seed=$seed $(date '+%H:%M:%S') commit=$(git rev-parse --short HEAD)" \
    > "$ROOT/logs/seed${seed}_${armname}_e50_3hop.log"
  $PY -m scripts.train_uci_link \
    --data-dir "$DATA_DIR" --gpu 0 \
    --epochs "$EPOCHS" --budget-cap "$EPOCHS" --patience "$PAT" \
    --kf-group-batches 40 --n-neighbors 10 --n-layers 3 \
    --lr "$LR" --repr-lr "$REPR_LR" \
    --seed "$seed" --output "$out" "$@" \
    >> "$ROOT/logs/seed${seed}_${armname}_e50_3hop.log" 2>&1
  echo "DONE_${armname}_e50_3hop_s${seed} rc=$? $(date '+%H:%M:%S')" >> "$ROOT/logs/seed${seed}_${armname}_e50_3hop.log"
}

for s in 0 1 2; do
  echo "=== e50 3hop seed $s round $(date '+%H:%M:%S') ===" >> "$ROOT/run.log"
  wait_gpu 8000
  before=$(gpu_used_mb)
  run_one "$s" ours --arm 2obs_aligned --lambda-kf "$LAM_3HOP" &
  pid1=$!
  confirm_ramp "$pid1" "$before"
  b2=$(gpu_used_mb)
  run_one "$s" taskonly --config P0 &
  pid2=$!
  confirm_ramp "$pid2" "$b2"
  wait
  echo "e50 3hop seed $s round done $(date '+%H:%M:%S')" >> "$ROOT/run.log"
done
echo "ALL_DONE_E50_3HOP $(date '+%H:%M:%S')" >> "$ROOT/run.log"
