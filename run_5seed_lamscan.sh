#!/bin/bash
# 5-seed lambda scan (seeds 1-4; seed 0 lives in outputs/s0_lambda_scan).
# 4 variants per seed: vanilla (official stage-2), lam0 (matched task-only
# control, same macro cadence), lam2, lam5.  lambda=10 was rejected by the
# s0 scan (0.8655 < vanilla 0.8730).
# Group length 56: 32 roots/batch x 64% valid-root rate -> ~1146 trees >= 1024.
cd /root/autodl-tmp/PRSS2
for seed in 1 2 3 4; do
  for lam in vanilla 0 2 5; do
    echo "$seed $lam"
  done
done | xargs -P 3 -n 2 bash -c '
  seed=$0; lam=$1
  out=outputs/s${seed}_lambda_scan/lam${lam}
  if [ -f $out/_SUCCESS.json ]; then echo "SKIP $out"; exit 0; fi
  mkdir -p $out
  common="-d wikipedia --data-dir old/processed_tgn_data --pretrained-checkpoint outputs/t2_pretrain/best.pt --output $out --n-epoch 20 --patience 10 --bs 200 --n-layer 3 --n-degree 5 --seed $seed"
  if [ "$lam" = "vanilla" ]; then
    /root/miniconda3/bin/python -m scripts.train_jodie $common > $out.log 2>&1
  else
    /root/miniconda3/bin/python -m scripts.train_jodie --rpbe --kf-lambda $lam --kf-group-batches 56 --repr-lr 1e-3 $common > $out.log 2>&1
  fi
  echo "DONE $out"
'
echo ALL_SEEDS_DONE
