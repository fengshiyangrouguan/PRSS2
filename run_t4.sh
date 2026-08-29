#!/bin/bash
cd /root/autodl-tmp/PRSS2
wait_for() {
  while [ ! -f "$1/_SUCCESS.json" ] 2>/dev/null && ! grep -q Traceback "$1.log" 2>/dev/null; do
    sleep 60
  done
}
wait_for outputs/t4_pretrain
if grep -q Traceback outputs/t4_pretrain.log; then
  echo T4_PRE_FAILED >> /root/autodl-tmp/matrix_status.txt
  exit 1
fi
nohup /root/miniconda3/bin/python -m scripts.train_jodie --rpbe --kf-lambda 0.0013 -d wikipedia \
  --data-dir old/processed_tgn_data \
  --pretrained-checkpoint outputs/t4_pretrain/best.pt \
  --output outputs/t4_stage2 --n-epoch 10 --bs 200 --n-layer 4 --n-degree 3 \
  > outputs/t4_stage2.log 2>&1 &
echo T4_STAGE2_STARTED >> /root/autodl-tmp/matrix_status.txt
wait_for outputs/t4_stage2
echo T4_DONE >> /root/autodl-tmp/matrix_status.txt
