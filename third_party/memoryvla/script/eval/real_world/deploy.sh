#!/bin/bash
set -euo pipefail

ckpt_path=/PATH/TO/YOUR/MODEL/CKPT
unnorm_key=custom_finetuning
port=2345
action_chunking_window=10
# you can also use action ensemble, but action chunking is faster for real world deployment

CUDA_VISIBLE_DEVICES=0 \
python deploy.py \
    --saved_model_path ${ckpt_path} \
    --unnorm_key ${unnorm_key} \
    --adaptive_ensemble_alpha 0.1 \
    --cfg_scale 1.5 \
    --port ${port} \
    --action_chunking \
    --action_chunking_window ${action_chunking_window}
