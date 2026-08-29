#!/bin/bash
# Baseline replica of task 2: same stage-1 host (wiki_s0_s1v_formal/best.pt),
# stage 2 WITHOUT RPBE (frozen host + decoder only) — official TGN baseline.
PY=/root/miniconda3/bin/python
cd /root/autodl-tmp/PRSS2
echo BASELINE_START $(date +%H:%M:%S)
$PY -m scripts.train_jodie -d wikipedia --data-dir old/processed_tgn_data \
    --pretrained-checkpoint outputs/pretrained/wiki_s0_s1v_formal/best.pt \
    --output outputs/formal/wiki_s0_vanilla_s1v \
    --n-epoch 10 --n-layer 3 --n-degree 5 --seed 0 --bs 100 --gpu 0 \
    > outputs/formal_baseline_s1v.log 2>&1 && echo BASELINE_OK || echo BASELINE_FAIL
echo BASELINE_DONE $(date +%H:%M:%S)
