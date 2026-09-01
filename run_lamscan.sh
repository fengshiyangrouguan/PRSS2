#!/bin/bash
# DEPRECATED (LOSS_DIAGNOSIS): this script predates the macro-group K
# unit fix; its lambda values are in the old (K-shrunk) units and its
# results must NOT be mixed with post-fix runs.  Refuse to start.
echo "run_lamscan.sh is deprecated: pre-K-cancellation lambda units." \
     "Use run_5seed_lamscan.sh with the post-fix training code instead." >&2
exit 1
cd /root/autodl-tmp/PRSS2
# Wait for the vanilla pretrain checkpoint (shared by all lambda runs).
while [ ! -f outputs/t2_pretrain/_SUCCESS.json ] 2>/dev/null \
    && ! grep -q Traceback outputs/t2_pretrain.log 2>/dev/null; do
  sleep 60
done
if grep -q Traceback outputs/t2_pretrain.log; then
  echo T2_PRE_FAILED >> /root/autodl-tmp/matrix_status.txt
  exit 1
fi
for lam in 0.01 0.015 0.02 0.03 0.05; do
  tag=$(echo "$lam" | tr -d '.')
  nohup /root/miniconda3/bin/python -m scripts.train_jodie \
    --rpbe --kf-lambda "$lam" -d wikipedia \
    --data-dir old/processed_tgn_data \
    --pretrained-checkpoint outputs/t2_pretrain/best.pt \
    --output "outputs/lamscan_$tag" \
    --n-epoch 20 --patience 10 --bs 200 --n-layer 3 --n-degree 5 \
    --seed 0 > "outputs/lamscan_$tag.log" 2>&1 &
  echo "LAM_$tag_STARTED" >> /root/autodl-tmp/matrix_status.txt
done
# Wait for all five.
for lam in 0.01 0.015 0.02 0.03 0.05; do
  tag=$(echo "$lam" | tr -d '.')
  while [ ! -f "outputs/lamscan_$tag/_SUCCESS.json" ] 2>/dev/null \
      && ! grep -q Traceback "outputs/lamscan_$tag.log" 2>/dev/null; do
    sleep 300
  done
done
echo LAMSCAN_DONE >> /root/autodl-tmp/matrix_status.txt
