#!/bin/bash
# tgbl-wiki recursive-closure structural ablation (4 arms x seeds).
#
# Arms (identical host/Gamma/task/optimizer; only the auxiliary supervision
# differs):
#   gamma_task_only  host+Gamma on L_link only (aux off)
#   1obs             child-future marginal (m_p=0)
#   2obs_aligned     child + true parent future (m_p=1)
#   2obs_mispaired   child + deranged parent future (m_p=1)
#
# One process per GPU slot; finished arms (their _SUCCESS.json exists) are
# skipped; interrupted runs resume idempotently.  Run from the repo root.
#
# Usage:
#   bash run_tgbl_wiki_recursive_abl.sh \
#       --seeds 0 1 2 3 4 --arms gamma_task_only 1obs 2obs_aligned 2obs_mispaired \
#       --gpus 0 --epochs 30

cd /root/autodl-tmp/PRSS2

SEEDS="0 1 2 3 4"
ARMS="gamma_task_only 1obs 2obs_aligned 2obs_mispaired"
GPUS="0"
EPOCHS=30
OUTROOT=tgbl_wiki_recursive_abl
K=2
PAR=1

while [ $# -gt 0 ]; do
  case "$1" in
    --seeds) shift; SEEDS=""; while [ $# -gt 0 ] && [[ "$1" != --* ]]; do SEEDS="$SEEDS $1"; shift; done;;
    --arms) shift; ARMS=""; while [ $# -gt 0 ] && [[ "$1" != --* ]]; do ARMS="$ARMS $1"; shift; done;;
    --gpus) GPUS="$2"; shift 2;;
    --epochs) EPOCHS="$2"; shift 2;;
    --outroot) OUTROOT="$2"; shift 2;;
    --k) K="$2"; shift 2;;
    --par) PAR="$2"; shift 2;;
    *) echo "unknown arg $1"; exit 2;;
  esac
done
SEEDS=$(echo $SEEDS)
ARMS=$(echo $ARMS)
GPU_ARR=($GPUS)

echo "== tgbl-wiki recursive abl: seeds=$SEEDS arms=$ARMS gpus=$GPUS epochs=$EPOCHS =="

run_one() {
  seed=$1; arm=$2; gpu=$3
  out=outputs/${OUTROOT}/seed${seed}/${arm}
  if [ -f $out/_SUCCESS.json ]; then echo "SKIP seed$seed/$arm"; return; fi
  if pgrep -f "train_tgb_link.*${out}" >/dev/null 2>&1; then echo "BUSY seed$seed/$arm"; return; fi
  mkdir -p $out
  OMP_NUM_THREADS=1 PYTHONIOENCODING=utf-8 \
    /root/miniconda3/bin/python -m scripts.train_tgb_link \
      --arm $arm --seed $seed --data-dir datasets --gpu $gpu \
      --output $out --epochs $EPOCHS --bs 200 --lr 1e-4 \
      --n-neighbors 10 --n-layers 3 --trace-roots 32 \
      --trace-pairs-per-parent ${K} --kf-group-batches 56 --kf-min-trees 896 \
      --lambda-kf 0.088 --ridge-eps 1e-3 --sketch-dim 64 --width-D 128 \
      > $out.log 2>&1
  echo "DONE seed$seed/$arm (rc=$?)"
}
export -f run_one
export OUTROOT EPOCHS K

n_gpu=${#GPU_ARR[@]}
jobs=0
for seed in $SEEDS; do
  for arm in $ARMS; do
    gpu=${GPU_ARR[$((jobs % n_gpu))]}
    run_one $seed $arm $gpu &
    jobs=$((jobs+1))
    # throttle to PAR concurrent
    while [ $(jobs -pr | wc -l) -ge $PAR ]; do
      wait -n 2>/dev/null || sleep 2
    done
  done
done
wait
echo "== tgbl-wiki recursive abl ALL DONE =="
