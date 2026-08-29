#!/bin/bash
cd /root/autodl-tmp/PRSS2
# 等 t3_stage2 跑完 10 epoch，用其 best.pt 续训 10 epoch
while [ ! -f outputs/t3_stage2/_SUCCESS.json ] 2>/dev/null && ! grep -q Traceback outputs/t3_stage2.log 2>/dev/null; do
  sleep 120
done
if grep -q Traceback outputs/t3_stage2.log; then
  echo T3_S2_FAILED >> /root/autodl-tmp/matrix_status.txt
  exit 1
fi
nohup /root/miniconda3/bin/python -m scripts.train_jodie --rpbe --kf-lambda 0.013 -d wikipedia \
  --data-dir old/processed_tgn_data \
  --pretrained-checkpoint outputs/t3_stage2/best.pt \
  --output outputs/t3_stage2_ext --n-epoch 10 --bs 200 --n-layer 4 --n-degree 3 \
  > outputs/t3_stage2_ext.log 2>&1 &
echo T3_S2_EXT_STARTED >> /root/autodl-tmp/matrix_status.txt
while [ ! -f outputs/t3_stage2_ext/_SUCCESS.json ] 2>/dev/null && ! grep -q Traceback outputs/t3_stage2_ext.log 2>/dev/null; do
  sleep 120
done
echo T3_S2_EXT_DONE >> /root/autodl-tmp/matrix_status.txt
