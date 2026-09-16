#!/bin/bash
# ============================================================
# CCM-LLM RPBE 标准命令集（R8 / R10）
# 环境：/root/autodl-tmp（AutoDL，80GB 卡，miniconda python）
# 2026-09-16 整理。协议定义与全部结果见同目录
# CCM_RPBE完整记录_R8到R10.md
# ============================================================
BASE=/root/autodl-tmp
PY=/root/miniconda3/bin/python
FD=$BASE/result/dialog/llama-7b-no
AD=$BASE/result/dialog/llama-7b-no-online-merge_recur-ntok2
MODEL=$BASE/llama-7b-hf
DIALOG=$BASE/dailydialog_mirror/ijcnlp_dailydialog
cd $BASE/third_party/ccm

# ---------- R8 训练（aggregate，λ=0.0223，300 窗口，每 50 checkpoint）----------
# 说明：R8 = 官方协议复刻 + single-cut 2Obs RPBE。λ 为 R8 时代校准值，
# 当前 frozen_method.json 已更新为 R10 的 0.005936691082765761，
# 要复现 R8 训练需先把 frozen 里 lambda_kf 改回 0.02229659292991447。
train_r8() {
  SEED=$1
  $PY $BASE/scripts/train_ccm.py \
    --arm ours --seed $SEED --lr 3e-5 \
    --kf-lambda 0.02229659292991447 \
    --max-steps 300 --save-every 50 \
    --official-host --foundation $FD --official-adapter $AD \
    --model-name-or-path $MODEL --output $BASE/outputs/ccm_pilot/seed${SEED}_R8_ours_e300
}

# ---------- R10 训练（当前正式协议）----------
# chain-wise 多 cut + gamma-only + P0 dropout 修复 + rpbe-lr 5e-5 + λ=0.00594
# 50 窗口封顶（续训 60-70 出现微退化），每 10 checkpoint。
train_r10() {
  SEED=$1
  $PY $BASE/scripts/train_ccm.py \
    --arm ours --seed $SEED --lr 3e-5 --rpbe-lr 5e-5 \
    --kf-lambda 0.005936691082765761 \
    --sketch-dim 32 --ridge-eps 0.001 --z-dim 128 --gamma-hidden 64 \
    --lora-r 8 --kf-min-cuts 128 --grad-clip 1.0 --rpbe-seed 0 \
    --max-steps 300 --schedule-total-steps 1000 \
    --rpbe-constrain-mode aggregate --rpbe-gamma-only \
    --max-windows 50 --checkpoint-every 10 \
    --official-host --foundation $FD --official-adapter $AD \
    --model-name-or-path $MODEL \
    --dialog-mirror $DIALOG \
    --output $BASE/outputs/ccm_pilot/seed${SEED}_R10_ours_e50
}

# ---------- 协议 C：时间步截断（官方 Table 25 同概念，主口径）----------
# test 集、L=压缩次数 ∈ {1,2,4,8,13}（脚本内 turns=L+2）、token-weighted、EOS 排除。
# CKPT=ours checkpoint；官方 merge 对照见 ccm_r8_results 文档（official_truncate_v2.log）。
eval_C() {
  CKPT=$1; OUT=$2
  OFFICIAL_HOST_EVAL=1 FOUNDATION=$FD OFFICIAL_ADAPTER=$AD \
  $PY $BASE/scripts/eval_ccm_depth.py --mode ccm --truncate \
    --ours-ckpt $CKPT --out $OUT
}

# ---------- 协议 B：pooled 五桶（官方 evaluate_perp）----------
# 官方 DialogueDataset.eval_dataset（val 集、自然长度分桶 turn_3/4/6/10/14，
# turn_14 桶用 15 轮）、官方 _loglikelihood_clm（EOS 排除）、
# 报 perplexity 字段 = token-weighted。
eval_B() {
  CKPT=$1; OUT=$2
  OUR_CKPT=$CKPT FOUNDATION=$FD OFFICIAL_ADAPTER=$AD EVAL_OUT=$OUT \
  $PY $BASE/scripts/eval_ccm_official.py \
    +dialog=llama-7b model.model_name_or_path=$MODEL \
    training.comp.num_comp_tokens=2 training.comp.attn_type=merge_recur \
    training.do_train=false wandb.log=false
}

# ---------- 协议 A：5L record-mean（深度曲线，51 条长对话）----------
# clean val 集 len>=15 的 51 条对话；每 L 取尾端窗口
# dialog[13-L:13] + [dialog[13] 上下文] + [dialog[14] 目标]（目标固定）；
# 对话内 token 平均 + 对话间等权平均；EOS 排除。merge/ours 跑法：
eval_A_ours() {
  CKPT=$1
  CKPT=$CKPT OUT=$BASE/pooled_r8/eval_5L_ours.json \
  $PY $BASE/scripts/eval_official_host_ckpt.py
}
# merge 与 no_ctx/full_ctx 的协议 A 跑法（内联脚本，官方权重）：
#   merge:    见 eval_5L_merge_ours.log 的启动命令（build_official_merge
#             不加载 ours ckpt 直接 evaluate_arm）
#   no_ctx:   neg_control collator（comp_type='neg_control'），同 51 对话同截断
#   full_ctx: online=False collator（纯拼接无 comp token），同 51 对话同截断
#   （三者完整命令见 ccm_r8_results 与本目录 CCM_RPBE完整记录_R8到R10.md）

# ---------- 官方 merge 对照（协议 C 时间步）----------
eval_C_merge() {
  OUT=$1
  OFFICIAL_HOST_EVAL=1 FOUNDATION=$FD OFFICIAL_ADAPTER=$AD \
  $PY $BASE/scripts/eval_ccm_depth.py --mode ccm --truncate \
    --ours-ckpt "" --out $OUT
}
# 注：官方 merge 的时间步数字用 ref/官方加载方式另见文档；
# 实际历史跑法见 official_truncate_v2.log 的启动命令。
