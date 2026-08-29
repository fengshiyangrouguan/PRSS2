#!/bin/bash
cd /root/autodl-tmp/PRSS2
wait_for() {
  while ! grep -qE '_SUCCESS|Traceback' "$1" 2>/dev/null; do sleep 60; done
}
# 任务2 阶段二立即启动（vanilla checkpoint 已保留）
nohup /root/miniconda3/bin/python -m scripts.train_jodie --rpbe --kf-lambda 0.01 -d wikipedia \
  --data-dir old/processed_tgn_data \
  --pretrained-checkpoint outputs/t2_pretrain/best.pt \
  --output outputs/t2_stage2 --n-epoch 10 --bs 200 --n-layer 3 --n-degree 5 \
  > outputs/t2_stage2.log 2>&1 &
echo T2_STAGE2_STARTED >> /root/autodl-tmp/matrix_status.txt
wait_for outputs/t1_pretrain.log
if grep -q Traceback outputs/t1_pretrain.log; then
  echo T1_PRE_FAILED >> /root/autodl-tmp/matrix_status.txt
else
  nohup /root/miniconda3/bin/python -m scripts.train_jodie --rpbe --kf-lambda 0.01 -d wikipedia \
    --data-dir old/processed_tgn_data \
    --pretrained-checkpoint outputs/t1_pretrain/best.pt \
    --output outputs/t1_stage2 --n-epoch 10 --bs 200 --n-layer 3 --n-degree 5 \
    > outputs/t1_stage2.log 2>&1 &
  echo T1_STAGE2_STARTED >> /root/autodl-tmp/matrix_status.txt
fi
wait_for outputs/t2_stage2.log
wait_for outputs/t1_stage2.log
echo MATRIX_DONE >> /root/autodl-tmp/matrix_status.txt
