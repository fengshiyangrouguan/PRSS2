#!/bin/bash
# Table 2 A7 (constraint strength) + A3/A4/A5 (appendix axes):
# k0 / C1 / B1 / E1   (canonical lives on the A1 card)
# Output: seed{N}_TGN_UCI_3L10N/{k0,C1,B1,E1}/
set -u
cd /root/autodl-tmp/PRSS2_uci_v2 || exit 1
export PYTHONPATH=/root/autodl-tmp/PRSS2_uci_v2:/root/autodl-tmp/benchtemp/experimental_codes/tgn-jodie-dyrep
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/root/miniconda3/bin/python
ROOT=/root/autodl-tmp/PRSS2_uci_v2/outputs/uci_formal_v2
DATA_DIR=/root/autodl-tmp/benchtemp/data_uci
LAM=LAM_PLACEHOLDER
EP=80
PAT=80
mkdir -p "$ROOT/logs"

run_one() {
  local abl="$1" seed="$2"; shift 2
  local out="$ROOT/seed${seed}_TGN_UCI_3L10N/${abl}"
  if [ -f "$out/summary.json" ] && grep -q '"stop_reason": "budget"' "$out/summary.json"; then
    echo "SKIP $abl s$seed" >> "$ROOT/run_abl_A3A7.log"; return 0
  fi
  if [ -f "$out/summary.json" ] && ! grep -q '"stop_reason": "budget"' "$out/summary.json"; then
    rm -f "$out/summary.json"; echo "RESET $abl s$seed" >> "$ROOT/run_abl_A3A7.log"
  fi
  if pgrep -f "train_uci_link.*seed${seed}_TGN_UCI_3L10N/${abl}" >/dev/null 2>&1; then
    echo "BUSY $abl s$seed" >> "$ROOT/run_abl_A3A7.log"; return 0
  fi
  mkdir -p "$out"
  echo "START $abl seed=$seed $(date '+%H:%M:%S')" >> "$ROOT/run_abl_A3A7.log"
  $PY -m scripts.train_uci_link \
    --data-dir "$DATA_DIR" --gpu 0 \
    --epochs "$EP" --budget-cap "$EP" --patience "$PAT" \
    --kf-group-batches 40 --n-neighbors 10 --n-layers 3 \
    --lr 1e-4 --repr-lr 3e-4 \
    --seed "$seed" --output "$out" "$@" \
    > "$ROOT/logs/seed${seed}_${abl}_abl.log" 2>&1
  echo "DONE $abl s$seed rc=$? $(date '+%H:%M:%S')" >> "$ROOT/run_abl_A3A7.log"
}

for s in 0 1 2; do
  run_one k0 "$s" --arm 2obs_aligned --lambda-kf "$LAM" --rpbe-constrain --rpbe-constrain-mode treewise --rpbe-kappa 0
  run_one C1 "$s" --config C1 --rpbe-constrain --rpbe-constrain-mode treewise --rpbe-kappa 0.05
  run_one B1 "$s" --config B1 --rpbe-constrain --rpbe-constrain-mode treewise --rpbe-kappa 0.05
  run_one E1 "$s" --config E1 --rpbe-constrain --rpbe-constrain-mode treewise --rpbe-kappa 0.05
done
echo "ALL_DONE_ABL_A3A7 $(date '+%H:%M:%S')" >> "$ROOT/run_abl_A3A7.log"
