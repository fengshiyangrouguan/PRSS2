#!/bin/bash
PY=/root/miniconda3/bin/python
cd /root/autodl-tmp/PRSS2
echo T2S2_START $(date +%H:%M:%S)
$PY -m scripts.train_jodie --rpbe -d wikipedia --data-dir old/processed_tgn_data \
    --pretrained-checkpoint outputs/pretrained/wiki_s0_s1v_formal/best.pt \
    --output outputs/formal/wiki_s0_rpbe_s1v \
    --n-epoch 10 --n-layer 3 --n-degree 5 --seed 0 --bs 100 --gpu 0 \
    > outputs/formal_s2r_s1v_v2.log 2>&1 && echo T2S2_OK || echo T2S2_FAIL
echo T2S2_DONE $(date +%H:%M:%S)
