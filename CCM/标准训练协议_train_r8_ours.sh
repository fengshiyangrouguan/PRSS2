#!/bin/bash
# ============================================================
# R8 ours 训练标准命令（DailyDialog，官方协议复刻 + RPBE）
# 参照：/root/autodl-tmp 环境；训练产物存 outputs/ccm_pilot/
# 2026-09-15 记录，配合 DailyDialog评估协议排查记录.md
# ============================================================
cd /root/autodl-tmp/third_party/ccm

SEED=0                 # 0 / 1 / 2
ARM=ours               # ours（Γ+RPBE）/ task_only
LR=3e-5                # 对话线微调协议（官方初始训练 3e-4 不用于续训）
LAMBDA=0.0223          # R8 主校准（λ 扫描可用 0.0669）
OUTPUT=/root/autodl-tmp/outputs/ccm_pilot/seed${SEED}_R8_ours_l01_e300

/root/miniconda3/bin/python /root/autodl-tmp/scripts/train_ccm.py \
  --arm ${ARM} \
  --seed ${SEED} \
  --lr ${LR} \
  --kf_lambda ${LAMBDA} \
  --max_steps 300 \
  --save_every 50 \
  --official_host \
  --foundation /root/autodl-tmp/result/dialog/llama-7b-no \
  --official_adapter /root/autodl-tmp/result/dialog/llama-7b-no-online-merge_recur-ntok2 \
  --model_name_or_path /root/autodl-tmp/llama-7b-hf \
  --output ${OUTPUT}
