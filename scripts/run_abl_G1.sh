#!/bin/bash
# G1 (26389): Per-tree Est. 消融 × 3 seeds（审阅定义的新格）
# 正式方法 = window-level pooled estimation；本格 = 每棵树只用自己的
# 行独立估计中心化/白化/协方差/伴随。其余全部与 R0 相同（同样的树数、
# targets、interface-wise 约束、κ=0.05、一个 window 一次更新）。
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
    echo "SKIP $abl s$seed" >> "$ROOT/run_abl_G1.log"; return 0
  fi
  if [ -f "$out/summary.json" ] && ! grep -q '"stop_reason": "budget"' "$out/summary.json"; then
    rm -f "$out/summary.json"; echo "RESET $abl s$seed" >> "$ROOT/run_abl_G1.log"
  fi
  if pgrep -f "train_uci_link.*seed${seed}_TGN_UCI_3L10N/${abl}" >/dev/null 2>&1; then
    echo "BUSY $abl s$seed" >> "$ROOT/run_abl_G1.log"; return 0
  fi
  mkdir -p "$out"
  echo "START $abl seed=$seed $(date '+%H:%M:%S')" >> "$ROOT/run_abl_G1.log"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -m scripts.train_uci_link \
    --data-dir "$DATA_DIR" --gpu 0 \
    --epochs "$EP" --budget-cap "$EP" --patience "$PAT" \
    --kf-group-batches 40 --n-neighbors 10 --n-layers 3 \
    --lr 1e-4 --repr-lr 3e-4 \
    --seed "$seed" --output "$out" "$@" \
    > "$ROOT/logs/seed${seed}_${abl}_abl.log" 2>&1
  echo "DONE $abl s$seed rc=$? $(date '+%H:%M:%S')" >> "$ROOT/run_abl_G1.log"
}

for s in 0 1 2; do
  run_one PerTree "$s" --arm 2obs_aligned --lambda-kf "$LAM" --rpbe-constrain --rpbe-constrain-mode treewise --rpbe-kappa 0.05 --est-mode per_tree
done
echo "ALL_DONE_ABL_G1 $(date '+%H:%M:%S')" >> "$ROOT/run_abl_G1.log"
