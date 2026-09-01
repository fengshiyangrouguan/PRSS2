#!/bin/bash
# Reddit formal paired arms at the smoke32 cadence: task_only@144 and
# exact_replay@m32, 5 seeds each (10 runs).  Two rounds of 6+4 (GPU
# ~4GB/run).  task_only shares the exact cadence (review requirement:
# same group_batches and initialization for the matched control).
cd /root/autodl-tmp/PRSS2

run_arm() {
  seed=$1; arm=$2
  out=outputs/reddit_seed${seed}/${arm}_smoke32
  if [ -f $out/_SUCCESS.json ]; then echo "SKIP $out"; return; fi
  mkdir -p $out
  common="-d reddit --data-dir old/processed_tgn_data --pretrained-checkpoint outputs/reddit_pretrain/best.pt --bs 200 --n-layer 3 --n-degree 5 --n-epoch 20 --patience 10 --seed $seed --kf-group-batches 144 --trace-roots 100 --kf-min-abs 64"
  if [ "$arm" = "task_only" ]; then
    nohup /root/miniconda3/bin/python -m scripts.train_jodie \
      --rpbe --kf-lambda 0 --kf-estimator exact_replay \
      --repr-lr 1e-3 --ridge-eps 1e-3 --sketch-dim 32 \
      $common --output $out > $out.log 2>&1 &
  else
    nohup /root/miniconda3/bin/python -m scripts.train_jodie \
      --rpbe --kf-lambda 0.088 --kf-estimator exact_replay \
      --repr-lr 1e-3 --ridge-eps 1e-3 --sketch-dim 32 \
      $common --output $out > $out.log 2>&1 &
  fi
  echo "START $out"
}

for seed in 0 1 2 3 4; do
  run_arm $seed task_only
  run_arm $seed exact_replay
done
wait
echo REDDIT_EXACT_DONE
