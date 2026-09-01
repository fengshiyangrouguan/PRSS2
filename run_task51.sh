#!/bin/bash
# Task 51: frozen 9-config sweep (ridge x m) on the s0 checkpoint.
# 2-way parallel (each diagnose_loss run uses ~11 GB GPU memory).
cd /root/autodl-tmp/PRSS2
for ridge in 1e-4 1e-3 1e-2; do
  for m in 16 32 64; do
    echo "$ridge $m"
  done
done | xargs -P 2 -n 2 bash -c '
  ridge=$0; m=$1
  tag=ridge_${ridge}_m_${m}
  if [ -f outputs/diag51/${tag}.json ]; then echo "SKIP $tag"; exit 0; fi
  /root/miniconda3/bin/python -m scripts.diagnose_loss \
    --run-dir outputs/s0_lambda_scan/lam0 \
    --data-dir old/processed_tgn_data --audit-split train \
    --group-batches 56 --groups 8 --permutations 30 --window-sweep 56 \
    --ridge-override $ridge --m-override $m \
    --out-json outputs/diag51/${tag}.json > outputs/diag51/${tag}.log 2>&1
  echo "DONE $tag"
'
echo TASK51_DONE
