#!/bin/bash
# Reddit multi-seed TGN experiment: 5 seeds x 3 arms, 6-way parallel
# (two seeds per round).  Naming: outputs/reddit_seed{N}/{arm}.
# Config: stage1 checkpoint reddit_pretrain/best.pt, ridge=1e-3, m=64,
# kf_group_batches=56, repr_lr=1e-3, lambda calibrated 0.088.
cd /root/autodl-tmp/PRSS2

run_seed() {
  seed=$1
  out=outputs/reddit_seed${seed}
  if [ -f $out/vanilla/_SUCCESS.json ] \
     && [ -f $out/task_only/_SUCCESS.json ] \
     && [ -f $out/exact_replay/_SUCCESS.json ]; then
    echo "SKIP seed$seed"; return
  fi
  mkdir -p $out
  common="-d reddit --data-dir old/processed_tgn_data --pretrained-checkpoint outputs/reddit_pretrain/best.pt --bs 200 --n-layer 3 --n-degree 5 --n-epoch 20 --patience 10 --seed $seed"
  nohup /root/miniconda3/bin/python -m scripts.train_jodie \
    $common --output $out/vanilla > $out/vanilla.log 2>&1 &
  nohup /root/miniconda3/bin/python -m scripts.train_jodie \
    --rpbe --kf-lambda 0 --kf-estimator exact_replay \
    --kf-group-batches 56 --repr-lr 1e-3 --ridge-eps 1e-3 --sketch-dim 64 \
    $common --output $out/task_only > $out/task_only.log 2>&1 &
  nohup /root/miniconda3/bin/python -m scripts.train_jodie \
    --rpbe --kf-lambda 0.088 --kf-estimator exact_replay \
    --kf-group-batches 56 --repr-lr 1e-3 --ridge-eps 1e-3 --sketch-dim 64 \
    $common --output $out/exact_replay > $out/exact_replay.log 2>&1 &
  echo "START seed$seed"
}

for pair in "0 1" "2 3" "4"; do
  for seed in $pair; do
    run_seed $seed
  done
  wait
  echo "ROUND_DONE: $pair"
done
echo REDDIT_ALL_SEEDS_DONE
