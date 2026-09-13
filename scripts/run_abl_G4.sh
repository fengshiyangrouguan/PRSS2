#!/bin/bash
# G4 (35360): P1 / P2 / R0  × 3 seeds
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
    echo "SKIP $abl s$seed" >> "$ROOT/run_abl_G4.log"; return 0
  fi
  if [ -f "$out/summary.json" ] && ! grep -q '"stop_reason": "budget"' "$out/summary.json"; then
    rm -f "$out/summary.json"; echo "RESET $abl s$seed" >> "$ROOT/run_abl_G4.log"
  fi
  if pgrep -f "train_uci_link.*seed${seed}_TGN_UCI_3L10N/${abl}" >/dev/null 2>&1; then
    echo "BUSY $abl s$seed" >> "$ROOT/run_abl_G4.log"; return 0
  fi
  mkdir -p "$out"
  echo "START $abl seed=$seed $(date '+%H:%M:%S')" >> "$ROOT/run_abl_G4.log"
  $PY -m scripts.train_uci_link \
    --data-dir "$DATA_DIR" --gpu 0 \
    --epochs "$EP" --budget-cap "$EP" --patience "$PAT" \
    --kf-group-batches 40 --n-neighbors 10 --n-layers 3 \
    --lr 1e-4 --repr-lr 3e-4 \
    --seed "$seed" --output "$out" "$@" \
    > "$ROOT/logs/seed${seed}_${abl}_abl.log" 2>&1
  echo "DONE $abl s$seed rc=$? $(date '+%H:%M:%S')" >> "$ROOT/run_abl_G4.log"
}

# 补跑：修复前（supervised 树数门禁 raise）崩溃的 s0 格；run_one 自带
# SKIP 幂等保护，已完成的格不会重跑。
run_one P1 0 --config P1 --rpbe-constrain --rpbe-constrain-mode treewise --rpbe-kappa 0.05
run_one P2 0 --config P2 --rpbe-constrain --rpbe-constrain-mode treewise --rpbe-kappa 0.05

for s in 0 1 2; do
  run_one P1 "$s" --config P1 --rpbe-constrain --rpbe-constrain-mode treewise --rpbe-kappa 0.05
  run_one P2 "$s" --config P2 --rpbe-constrain --rpbe-constrain-mode treewise --rpbe-kappa 0.05
  run_one R0 "$s" --arm 2obs_aligned --lambda-kf "$LAM" --rpbe-constrain --rpbe-constrain-mode treewise --rpbe-kappa 0.05
done
echo "ALL_DONE_ABL_G4 $(date '+%H:%M:%S')" >> "$ROOT/run_abl_G4.log"
