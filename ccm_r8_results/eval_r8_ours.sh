#!/bin/bash
# ============================================================
# R8 ours 推理/评估标准命令（DailyDialog 两个外部口径）
# 关键：必须带 num_comp_tokens=2 + attn_type=merge_recur！
#       truncate 的步数 = 压缩次数 L（轮数 = L+2）。
# 2026-09-15 记录，配合 DailyDialog评估协议排查记录.md
# ============================================================
cd /root/autodl-tmp/third_party/ccm

CKPT=/root/autodl-tmp/outputs/ccm_pilot/seed0_R8_ours_e300/checkpoint_step50.pt
FD=/root/autodl-tmp/result/dialog/llama-7b-no
AD=/root/autodl-tmp/result/dialog/llama-7b-no-online-merge_recur-ntok2
OUTDIR=/root/autodl-tmp/pooled_r8

# ---------- 协议 C：时间步截断（官方 Table 25 同概念，主口径）----------
# L = 压缩次数 ∈ {1,2,4,8,13}；脚本内部转为 L+2 轮截断
OFFICIAL_HOST_EVAL=1 FOUNDATION=$FD OFFICIAL_ADAPTER=$AD \
/root/miniconda3/bin/python /root/autodl-tmp/scripts/eval_ccm_depth.py \
  --mode ccm --truncate \
  --ours-ckpt $CKPT \
  --out ${OUTDIR}/trunc_seedX.json

# ---------- 协议 B：pooled 五桶（官方 evaluate_perp）----------
OUR_CKPT=$CKPT FOUNDATION=$FD OFFICIAL_ADAPTER=$AD \
EVAL_OUT=${OUTDIR}/pooled_seedX.json \
/root/miniconda3/bin/python /root/autodl-tmp/scripts/eval_ccm_official.py \
  +dialog=llama-7b model.model_name_or_path=/root/autodl-tmp/llama-7b-hf \
  training.comp.num_comp_tokens=2 training.comp.attn_type=merge_recur \
  training.do_train=false wandb.log=false

# ---------- 官方 merge 对照（两个协议）----------
# pooled:
# /root/miniconda3/bin/python /root/autodl-tmp/scripts/eval_ccm_official.py \
#   +dialog=llama-7b model.model_name_or_path=/root/autodl-tmp/llama-7b-hf \
#   training.eval_path=$AD \
#   training.comp.num_comp_tokens=2 training.comp.attn_type=merge_recur \
#   training.do_train=false wandb.log=false
