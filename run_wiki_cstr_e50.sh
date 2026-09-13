#!/bin/bash
# Wikipedia constrained-RPBE rerun (reviewer formulation, 2026-09-13):
# per-tree separate directions + row-normalized QP (cosine Gram) + FISTA
# budget ladder + hard feasibility certificate (skip on failure).
# 6 seeds x 2 structures, 50 epochs run to budget, ALL parallel (wiki runs
# take ~2.6GB each).  Output to seed{N}_TGN_WIKI_{3L5N,2L10N}/ours_cstr_e50/
# (the pre-fix runs stay untouched in ours/).
set -u
cd /root/autodl-tmp/PRSS2 || exit 1
export PYTHONPATH=/root/autodl-tmp/PRSS2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/root/miniconda3/bin/python
ROOT=/root/autodl-tmp/PRSS2/outputs/wiki_formal
DATA_DIR=old/processed_tgn_data
PRETRAIN_3L=/root/autodl-tmp/wikipedia_assets/outputs/t2_pretrain/best.pt
PRETRAIN_2L=/root/autodl-tmp/PRSS2/outputs/t2_pretrain_2L10N/best.pt
EPOCHS=50
PAT=50
LAM_3L5N=5
LAM_2L10N=2
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
    if [ -n "$used" ] && [ "$used" -ge $((before + 2000)) ]; then return 0; fi
  done
  return 0
}

run_wiki() {
  local tag="$1" seed="$2" nl="$3" nd="$4" lam="$5"
  local out="$ROOT/seed${seed}_TGN_WIKI_${tag}/ours_cstr_e50"
  local pretrain="$PRETRAIN_3L"
  if [ "$nl" = "2" ]; then pretrain="$PRETRAIN_2L"; fi
  if [ -f "$out/summary.json" ] && grep -q '"test"' "$out/summary.json"; then
    echo "SKIP cstr $tag seed=$seed" >> "$ROOT/run_wiki_cstr.log"; return 0
  fi
  if pgrep -f "train_jodie.*seed${seed}_TGN_WIKI_${tag}/ours_cstr_e50" >/dev/null 2>&1; then
    echo "BUSY cstr $tag seed=$seed" >> "$ROOT/run_wiki_cstr.log"; return 0
  fi
  mkdir -p "$out"
  echo "START cstr $tag seed=$seed lam=$lam $(date '+%H:%M:%S') commit=$(git rev-parse --short HEAD)" \
    > "$ROOT/logs/seed${seed}_${tag}_cstr.log"
  $PY -m scripts.train_jodie -d wikipedia --data-dir "$DATA_DIR" \
    --pretrained-checkpoint "$pretrain" \
    --output "$out" \
    --n-epoch "$EPOCHS" --n-layer "$nl" --n-degree "$nd" \
    --seed "$seed" --bs 200 --patience "$PAT" --gpu 0 \
    --rpbe --kf-lambda "$lam" --kf-group-batches 56 --repr-lr 1e-3 \
    --rpbe-constrain --rpbe-kappa "$KAPPA" \
    >> "$ROOT/logs/seed${seed}_${tag}_cstr.log" 2>&1
  echo "DONE cstr $tag seed=$seed rc=$? $(date '+%H:%M:%S')" >> "$ROOT/logs/seed${seed}_${tag}_cstr.log"
}

for s in 0 1 2 3 4 5; do
  wait_gpu 24000
  before=$(gpu_used_mb)
  run_wiki 3L5N "$s" 3 5 "$LAM_3L5N" & pid=$!
  confirm_ramp "$pid" "$before"
  b2=$(gpu_used_mb)
  run_wiki 2L10N "$s" 2 10 "$LAM_2L10N" & pid2=$!
  confirm_ramp "$pid2" "$b2"
done
wait
while pgrep -f "train_jodie.*TGN_WIKI_3L5N" >/dev/null 2>&1; do sleep 120; done
echo "ALL_DONE_WIKI_CSTR_3L5N $(date '+%H:%M:%S')" >> "$ROOT/run_wiki_cstr.log"
while pgrep -f "train_jodie.*TGN_WIKI_2L10N" >/dev/null 2>&1; do sleep 120; done
echo "ALL_DONE_WIKI_CSTR_2L10N $(date '+%H:%M:%S')" >> "$ROOT/run_wiki_cstr.log"
