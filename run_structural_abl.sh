#!/bin/bash
# Structural ablation, three arms (paper Part III).
#
#   1obs            supervision_mode=1obs
#   2obs_aligned    supervision_mode=2obs_aligned
#   2obs_mispaired  supervision_mode=2obs_mispaired
#
# Every arm starts from the SAME stage-1 checkpoint (t2_pretrain), same seed,
# same Gamma/rank/cuts/task-loss, same macro-group cadence and same exact
# replay estimator — the ONLY difference is how the second future observation
# is used (mispaired = the horizon-2 event is deranged inside a
# (tau, role2, coarse-time) bucket, keeping its marginal).  Arm names below.
#
# A 1obs arm must use the SAME Y1+Y2-valid cut intersection as the 2obs arms
# (enforced inside the builder: supervision_mode != production always keeps
# only cuts with both futures), and per-tree total weight is 1 in all arms.
#
# Naming: outputs/<outroot>/seed{N}/<arm>  (outroot defaults to struct_abl).
# Idempotent: a finished arm (its _SUCCESS.json exists) is skipped.
#
# Usage:
#   bash run_structural_abl.sh [--seeds 0 1 2 3 4] [--outroot struct_abl] [--parallel 3]

cd /root/autodl-tmp/PRSS2

SEEDS="0 1 2 3 4"
OUTROOT=struct_abl
PAR=3

while [ $# -gt 0 ]; do
  case "$1" in
    --seeds) shift; SEEDS="$@"; break ;;
    --outroot) OUTROOT="$2"; shift 2 ;;
    --parallel) PAR="$2"; shift 2 ;;
    *) echo "unknown arg $1"; exit 2 ;;
  esac
done

run_one() {
  seed=$1; arm=$2
  out=outputs/${OUTROOT}/seed${seed}/${arm}
  if [ -f $out/_SUCCESS.json ]; then echo "SKIP $out"; return; fi
  if pgrep -f "train_jodie.*${out}" >/dev/null 2>&1; then echo "BUSY $out"; return; fi
  mkdir -p $out
  /root/miniconda3/bin/python -m scripts.train_jodie \
    --rpbe --kf-lambda 0.088 --kf-estimator exact_replay \
    --supervision-mode $arm --n-observations 2 \
    --kf-group-batches 56 --kf-min-abs 896 \
    --repr-lr 1e-3 --ridge-eps 1e-3 --sketch-dim 64 \
    -d wikipedia --data-dir old/processed_tgn_data \
    --pretrained-checkpoint outputs/t2_pretrain/best.pt \
    --bs 200 --n-layer 3 --n-degree 5 --n-epoch 20 --patience 10 \
    --seed $seed --output $out > $out.log 2>&1
  echo "DONE $out"
}
export -f run_one
export OUTROOT

echo "== structural ablation: seeds=$SEEDS arms=1obs,2obs_aligned,2obs_mispaired par=$PAR =="

for seed in $SEEDS; do
  echo "$seed 1obs
$seed 2obs_aligned
$seed 2obs_mispaired"
done | xargs -P $PAR -n 2 bash -c 'run_one $0 $1'

echo STRUCT_ABL_DONE
