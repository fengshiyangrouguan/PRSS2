#!/usr/bin/env bash
set -euo pipefail
METHOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "${METHOD_DIR}/.." && pwd)"
TGN_DIR="${TGN_DIR:-${METHOD_DIR}/official_tgn/source}"
DATASET="${DATASET:-wikipedia}"
DATA_DIR="${DATA_DIR:-${WORKSPACE_DIR}/processed_tgn_data}"
N_LAYER="${N_LAYER:-2}"
N_DEGREE="${N_DEGREE:-10}"
BS="${BS:-100}"
EPOCHS="${EPOCHS:-10}"
PATIENCE="${PATIENCE:-3}"
SEED="${SEED:-0}"
CANDIDATE_DIM="${CANDIDATE_DIM:-256}"
TRACE_ROOTS="${TRACE_ROOTS:-8}"
MONITOR_EVERY="${MONITOR_EVERY:-50}"
SPECTRAL_STEP_SIZE="${SPECTRAL_STEP_SIZE:-0.25}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-50}"
RESUME_FROM="${RESUME_FROM:-}"
FINETUNE_HOST="${FINETUNE_HOST:-0}"
OUT="${OUTPUT_ROOT:-${METHOD_DIR}/outputs/node_classification/${DATASET}/seed_${SEED}_l${N_LAYER}}"
mkdir -p "$OUT/logs"
export TGN_DIR

if [[ -z "${PRETRAINED_CHECKPOINT:-}" ]]; then
  candidates=(
    "/root/autodl-tmp/PRSS/outputs_tgn_diagnostics/baselines/memory_l${N_LAYER}_seed0/saved_models/tgn-${DATASET}.pth"
    "/root/autodl-tmp/PRSS/outputs_tgn_diagnostics/baselines/memory_l${N_LAYER}_seed0/saved_models/tgn-wikipedia.pth"
    "/root/autodl-tmp/PRSS/outputs_tgn_diagnostics/baselines/memory_l${N_LAYER}_seed0/saved_checkpoints/tgn-${DATASET}-8.pth"
    "${TGN_DIR}/saved_models/tgn-attn-${DATASET}.pth"
    "${TGN_DIR}/saved_models/tgn-${DATASET}.pth"
  )
  for f in "${candidates[@]}"; do
    if [[ -f "$f" ]]; then PRETRAINED_CHECKPOINT="$f"; break; fi
  done
fi
if [[ -z "${PRETRAINED_CHECKPOINT:-}" || ! -f "$PRETRAINED_CHECKPOINT" ]]; then
  echo "ERROR: no pretrained checkpoint found; set PRETRAINED_CHECKPOINT=/path/to/model" >&2
  exit 2
fi

if [[ -z "$RESUME_FROM" && -d "$OUT/prss_matched" && ! -f "$OUT/prss_matched/_SUCCESS.json" ]]; then
  mv "$OUT/prss_matched" "$OUT/prss_matched_failed_$(date +%Y%m%d_%H%M%S)"
fi
HOST_FLAG=()
[[ "$FINETUNE_HOST" == "1" ]] && HOST_FLAG=(--finetune-host)
RESUME_FLAG=()
[[ -n "$RESUME_FROM" ]] && RESUME_FLAG=(--resume-from "$RESUME_FROM")
GPU_COUNT=$(python - <<'PY'
import torch; print(torch.cuda.device_count())
PY
)
PHYSICAL_GPU=1
[[ "$GPU_COUNT" -lt 2 ]] && PHYSICAL_GPU=0

echo "PRSS-only retry: physical GPU=$PHYSICAL_GPU N_LAYER=$N_LAYER checkpoint=$PRETRAINED_CHECKPOINT spectral_step=$SPECTRAL_STEP_SIZE resume=${RESUME_FROM:-none}"
CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU" python -u "$METHOD_DIR/experiments/train_supervised_prss_switch.py" \
  --mode prss -d "$DATASET" --data-dir "$DATA_DIR" --pretrained-checkpoint "$PRETRAINED_CHECKPOINT" \
  --output "$OUT/prss_matched" --bs "$BS" --n-degree "$N_DEGREE" --n-layer "$N_LAYER" \
  --n-epoch "$EPOCHS" --patience "$PATIENCE" --use-memory --seed "$SEED" --selection-metric auc \
  --monitor-every "$MONITOR_EVERY" --candidate-dim "$CANDIDATE_DIM" --trace-roots "$TRACE_ROOTS" \
  --spectral-step-size "$SPECTRAL_STEP_SIZE" --checkpoint-every "$CHECKPOINT_EVERY" \
  "${RESUME_FLAG[@]}" "${HOST_FLAG[@]}" --gpu 0 2>&1 | tee -a "$OUT/logs/gpu1_prss.log"
