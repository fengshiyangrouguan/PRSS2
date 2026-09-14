#!/bin/bash
# G3 (21954): P1 / P2 / R0  × 3 seeds（接管 G4 的原队列）
# G4（35360）当前被 additive s1 插队占用，additive 跑完后 G4 的
# P1/P2/R0 行会因本组已完成的 summary 触发 SKIP，不重复。
ROOT=/root/autodl-tmp/PRSS2_uci_v2/outputs/uci_formal_v2
DATA_DIR=/root/autodl-tmp/benchtemp/data_uci
LAM=0.00668
EP=80
PAT=80
PY=/root/miniconda3/bin/python
mkdir -p "$ROOT/logs"

run_one() {
  local abl="$1" seed="$2"; shift 2
  local out="$ROOT/seed${seed}_TGN_UCI_3L10N/${abl}"
  if [ -f "$out/summary.json" ] && grep -q '"stop_reason": "budget"' "$out/summary.json"; then
    echo "SKIP $abl s$seed" >> "$ROOT/run_abl_G3.log"; return 0
  fi
  if [ -f "$out/summary.json" ] && ! grep -q '"stop_reason": "budget"' "$out/summary.json"; then
    rm -f "$out/summary.json"; echo "RESET $abl s$seed" >> "$ROOT/run_abl_G3.log"
  fi
  if pgrep -f "train_uci_link.*seed${seed}_TGN_UCI_3L10N/${abl}" >/dev/null 2>&1; then
    echo "BUSY $abl s$seed" >> "$ROOT/run_abl_G3.log"; return 0
  fi
  mkdir -p "$out"
  echo "START $abl seed=$seed $(date '+%H:%M:%S')" >> "$ROOT/run_abl_G3.log"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -m scripts.train_uci_link \
    --data-dir "$DATA_DIR" --gpu 0 \
    --epochs "$EP" --budget-cap "$EP" --patience "$PAT" \
    --kf-group-batches 40 --n-neighbors 10 --n-layers 3 \
    --lr 1e-4 --repr-lr 3e-4 \
    --seed "$seed" --output "$out" "$@" \
    > "$ROOT/logs/seed${seed}_${abl}_abl.log" 2>&1
  echo "DONE $abl s$seed rc=$? $(date '+%H:%M:%S')" >> "$ROOT/run_abl_G3.log"
}

for s in 0 1 2; do
  run_one P1 "$s" --config P1 --rpbe-constrain --rpbe-constrain-mode treewise --rpbe-kappa 0.05
  run_one P2 "$s" --config P2 --rpbe-constrain --rpbe-constrain-mode treewise --rpbe-kappa 0.05
  run_one R0 "$s" --arm 2obs_aligned --lambda-kf "$LAM" --rpbe-constrain --rpbe-constrain-mode treewise --rpbe-kappa 0.05
done
echo "ALL_DONE_ABL_G3 $(date '+%H:%M:%S')" >> "$ROOT/run_abl_G3.log"
