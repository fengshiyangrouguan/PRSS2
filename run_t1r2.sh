#!/bin/bash
PY=/root/miniconda3/bin/python
cd /root/autodl-tmp/PRSS2
echo T1R2_START $(date +%H:%M:%S)
$PY -m scripts.train_pretrain -d wikipedia --data-dir old/processed_tgn_data \
    --output outputs/pretrained/wiki_s0_s1r_formal --stage1-rpbe \
    --resume-from outputs/pretrained/wiki_s0_s1r_formal/rolling_step.pt \
    --n-epoch 50 --n-layer 3 --n-degree 5 --seed 0 --bs 100 --gpu 0 \
    > outputs/formal_s1r_v3.log 2>&1 && echo T1R2_S1_OK || echo T1R2_S1_FAIL
$PY -m scripts.train_jodie --rpbe -d wikipedia --data-dir old/processed_tgn_data \
    --pretrained-checkpoint outputs/pretrained/wiki_s0_s1r_formal/best.pt \
    --output outputs/formal/wiki_s0_rpbe \
    --n-epoch 10 --n-layer 3 --n-degree 5 --seed 0 --bs 100 --gpu 0 \
    > outputs/formal_s2r_v3.log 2>&1 && echo T1R2_S2_OK || echo T1R2_S2_FAIL
echo T1R2_ALL_DONE $(date +%H:%M:%S)
