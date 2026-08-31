#!/bin/bash
# Formal multi-seed TGN experiment (review step 6): 5 seeds x 3 arms.
# Naming: outputs/seed{N}_TGN/{vanilla, task_only, exact_replay}.
# One seed's three arms run in parallel; seeds run sequentially.
# Config (review decision): ridge=1e-3, m=64, kf_group_batches=56,
# repr_lr=1e-3, lambda calibrated to r_eff~0.1 (0.088).
cd /root/autodl-tmp/PRSS2
for seed in 0 1 2 3 4; do
  out=outputs/seed${seed}_TGN
  mkdir -p $out
  common="-d wikipedia --data-dir old/processed_tgn_data --pretrained-checkpoint outputs/t2_pretrain/best.pt --bs 200 --n-layer 3 --n-degree 5 --n-epoch 20 --patience 10 --seed $seed"
  # Arm 1: vanilla official protocol (frozen host, decoder only).
  nohup /root/miniconda3/bin/python -m scripts.train_jodie \
    $common --output $out/vanilla > $out/vanilla.log 2>&1 &
  # Arm 2: TGN + Gamma, task-only grouped (matched control, lambda=0).
  nohup /root/miniconda3/bin/python -m scripts.train_jodie \
    --rpbe --kf-lambda 0 --kf-estimator exact_replay \
    --kf-group-batches 56 --repr-lr 1e-3 --ridge-eps 1e-3 --sketch-dim 64 \
    $common --output $out/task_only > $out/task_only.log 2>&1 &
  # Arm 3: exact replay with calibrated lambda.
  nohup /root/miniconda3/bin/python -m scripts.train_jodie \
    --rpbe --kf-lambda 0.088 --kf-estimator exact_replay \
    --kf-group-batches 56 --repr-lr 1e-3 --ridge-eps 1e-3 --sketch-dim 64 \
    $common --output $out/exact_replay > $out/exact_replay.log 2>&1 &
  wait
  echo "SEED${seed}_DONE"
done
echo ALL_SEEDS_DONE
