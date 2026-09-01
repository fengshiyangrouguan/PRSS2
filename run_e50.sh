#!/bin/bash
# From-scratch 50-epoch pass (patience 10) for the 8 bottleneck runs:
# seed 1-4 x {task_only, exact_replay}.  Fresh output dirs {arm}_e50;
# the 20-epoch originals stay untouched for comparison.  4-way parallel.
cd /root/autodl-tmp/PRSS2
for seed in 1 2 3 4; do
  for arm in task_only exact_replay; do
    echo "$seed $arm"
  done
done | xargs -P 4 -n 2 bash -c '
  seed=$0; arm=$1
  out=outputs/seed${seed}_TGN/${arm}_e50
  if [ -f $out/_SUCCESS.json ]; then echo "SKIP $out"; exit 0; fi
  mkdir -p $out
  common="-d wikipedia --data-dir old/processed_tgn_data --pretrained-checkpoint outputs/t2_pretrain/best.pt --bs 200 --n-layer 3 --n-degree 5 --n-epoch 50 --patience 10 --seed $seed"
  if [ "$arm" = "task_only" ]; then
    /root/miniconda3/bin/python -m scripts.train_jodie \
      --rpbe --kf-lambda 0 --kf-estimator exact_replay \
      --kf-group-batches 56 --repr-lr 1e-3 --ridge-eps 1e-3 --sketch-dim 64 \
      $common --output $out > $out.log 2>&1
  else
    /root/miniconda3/bin/python -m scripts.train_jodie \
      --rpbe --kf-lambda 0.088 --kf-estimator exact_replay \
      --kf-group-batches 56 --repr-lr 1e-3 --ridge-eps 1e-3 --sketch-dim 64 \
      $common --output $out > $out.log 2>&1
  fi
  echo "DONE $out"
'
echo E50_DONE
