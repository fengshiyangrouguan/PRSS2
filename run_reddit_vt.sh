#!/bin/bash
# Reddit vanilla + task_only pass (exact arm paused pending the
# strict-future review).  Seeds 0-1 already done; this runs seeds
# 2,3,4 x {vanilla, task_only} = 6 parallel jobs in one round.
# Idempotent per arm.
cd /root/autodl-tmp/PRSS2

run_arm() {
  seed=$1; arm=$2
  out=outputs/reddit_seed${seed}/${arm}
  if [ -f $out/_SUCCESS.json ]; then echo "SKIP $out"; return; fi
  mkdir -p $out
  common="-d reddit --data-dir old/processed_tgn_data --pretrained-checkpoint outputs/reddit_pretrain/best.pt --bs 200 --n-layer 3 --n-degree 5 --n-epoch 20 --patience 10 --seed $seed"
  if [ "$arm" = "vanilla" ]; then
    nohup /root/miniconda3/bin/python -m scripts.train_jodie \
      $common --output $out > $out.log 2>&1 &
  else
    nohup /root/miniconda3/bin/python -m scripts.train_jodie \
      --rpbe --kf-lambda 0 --kf-estimator exact_replay \
      --kf-group-batches 56 --repr-lr 1e-3 --ridge-eps 1e-3 --sketch-dim 64 \
      $common --output $out > $out.log 2>&1 &
  fi
  echo "START $out"
}

for seed in 2 3 4; do
  run_arm $seed vanilla
  run_arm $seed task_only
done
wait
echo VT_DONE
