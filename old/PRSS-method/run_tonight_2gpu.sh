#!/usr/bin/env bash
set -euo pipefail

METHOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "${METHOD_DIR}/.." && pwd)"
TGN_DIR="${TGN_DIR:-${METHOD_DIR}/official_tgn/source}"
DATASET="${DATASET:-wikipedia}"
DATA_DIR="${DATA_DIR:-${WORKSPACE_DIR}/processed_tgn_data}"

# N_LAYER=2 is the recommended PRSS recursion experiment. Set N_LAYER=1 to reproduce the
# upstream node-classification paper default architecture exactly.
N_LAYER="${N_LAYER:-2}"
N_DEGREE="${N_DEGREE:-10}"
BS="${BS:-100}"
EPOCHS="${EPOCHS:-10}"
PATIENCE="${PATIENCE:-3}"
SEED="${SEED:-0}"
CANDIDATE_DIM="${CANDIDATE_DIM:-256}"
TRACE_ROOTS="${TRACE_ROOTS:-8}"
MONITOR_EVERY="${MONITOR_EVERY:-50}"
FINETUNE_HOST="${FINETUNE_HOST:-0}"   # 0 = closest to official frozen-host protocol
RUN_OFFICIAL_REFERENCE="${RUN_OFFICIAL_REFERENCE:-1}"
RUN_TESTS="${RUN_TESTS:-0}"
OUT="${OUTPUT_ROOT:-${METHOD_DIR}/outputs/node_classification/${DATASET}/seed_${SEED}_l${N_LAYER}}"
mkdir -p "$OUT/logs"
export TGN_DIR

if [[ "$RUN_TESTS" == "1" ]]; then
  if python - <<'PY' >/dev/null 2>&1
import pytest
PY
  then
    (cd "$METHOD_DIR" && python -m pytest -q)
  else
    echo "WARNING: pytest not installed; skipping optional development tests. Formal training does not depend on pytest."
  fi
fi

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
  echo "ERROR: no compatible pretrained self-supervised TGN checkpoint found for N_LAYER=${N_LAYER}." >&2
  echo "Set PRETRAINED_CHECKPOINT=/path/to/tgn-${DATASET}.pth" >&2
  exit 2
fi

cat <<EOF
=== Official-mother TGN + PRSS experiment ===
METHOD_DIR=$METHOD_DIR
TGN_DIR=$TGN_DIR
DATA_DIR=$DATA_DIR
PRETRAINED_CHECKPOINT=$PRETRAINED_CHECKPOINT
DATASET=$DATASET N_LAYER=$N_LAYER N_DEGREE=$N_DEGREE BS=$BS EPOCHS=$EPOCHS
FINETUNE_HOST=$FINETUNE_HOST (0 = official-style frozen host; 1 = matched host fine-tuning)
OUT=$OUT
EOF

unset CUDA_VISIBLE_DEVICES || true
python - <<'PY'
import torch
print('torch=', torch.__version__, 'cuda=', torch.cuda.is_available(), 'visible_devices=', torch.cuda.device_count())
for i in range(torch.cuda.device_count()): print(i, torch.cuda.get_device_name(i))
PY
GPU_COUNT=$(python - <<'PY'
import torch; print(torch.cuda.device_count())
PY
)
if [[ "$GPU_COUNT" -lt 2 ]]; then
  echo "WARNING: only $GPU_COUNT PyTorch-visible GPU(s). Falling back to sequential execution." >&2
fi

HOST_FLAG=()
if [[ "$FINETUNE_HOST" == "1" ]]; then HOST_FLAG=(--finetune-host); fi
COMMON=(
  -d "$DATASET" --data-dir "$DATA_DIR" --pretrained-checkpoint "$PRETRAINED_CHECKPOINT"
  --bs "$BS" --n-degree "$N_DEGREE" --n-layer "$N_LAYER" --n-epoch "$EPOCHS"
  --patience "$PATIENCE" --use-memory --seed "$SEED" --selection-metric auc
  --monitor-every "$MONITOR_EVERY"
  "${HOST_FLAG[@]}"
)

archive_incomplete() {
  local d="$1"
  if [[ -d "$d" && ! -f "$d/_SUCCESS.json" ]]; then
    local ts; ts=$(date +%Y%m%d_%H%M%S)
    mv "$d" "${d}_incomplete_${ts}"
  fi
}
archive_incomplete "$OUT/vanilla_matched"
archive_incomplete "$OUT/prss_matched"
archive_incomplete "$OUT/official_reference"

run_vanilla() {
  if [[ -f "$OUT/vanilla_matched/_SUCCESS.json" ]]; then
    echo "vanilla_matched already complete; skip" | tee "$OUT/logs/gpu0_vanilla.log"
    return
  fi
  CUDA_VISIBLE_DEVICES=0 python -u "$METHOD_DIR/experiments/train_supervised_prss_switch.py" \
    --mode vanilla --output "$OUT/vanilla_matched" "${COMMON[@]}" --gpu 0 \
    > "$OUT/logs/gpu0_vanilla.log" 2>&1
}

run_prss() {
  if [[ -f "$OUT/prss_matched/_SUCCESS.json" ]]; then
    echo "prss_matched already complete; skip" | tee "$OUT/logs/gpu1_prss.log"
    return
  fi
  local physical_gpu=1
  [[ "$GPU_COUNT" -lt 2 ]] && physical_gpu=0
  CUDA_VISIBLE_DEVICES="$physical_gpu" python -u "$METHOD_DIR/experiments/train_supervised_prss_switch.py" \
    --mode prss --output "$OUT/prss_matched" "${COMMON[@]}" --gpu 0 \
    --candidate-dim "$CANDIDATE_DIM" --trace-roots "$TRACE_ROOTS" \
    > "$OUT/logs/gpu1_prss.log" 2>&1
}

run_official_reference() {
  [[ "$RUN_OFFICIAL_REFERENCE" != "1" ]] && return
  if [[ -f "$OUT/official_reference/_SUCCESS.json" ]]; then
    echo "official_reference already complete; skip" >> "$OUT/logs/gpu0_official.log"
    return
  fi
  local OFF_WORK="$OUT/official_reference/work"
  rm -rf "$OFF_WORK"
  mkdir -p "$OFF_WORK/log" "$OFF_WORK/results" "$OFF_WORK/saved_models" "$OFF_WORK/saved_checkpoints"
  ln -s "$DATA_DIR" "$OFF_WORK/data"
  local PREFIX="official_l${N_LAYER}"
  ln -s "$PRETRAINED_CHECKPOINT" "$OFF_WORK/saved_models/${PREFIX}-${DATASET}.pth"
  (
    cd "$OFF_WORK"
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$TGN_DIR" python -u "$METHOD_DIR/official_tgn/train_supervised.py" \
      -d "$DATASET" --bs "$BS" --prefix "$PREFIX" --n_degree "$N_DEGREE" --n_layer "$N_LAYER" \
      --n_epoch "$EPOCHS" --gpu 0 --use_memory
  ) > "$OUT/logs/gpu0_official.log" 2>&1
  python "$METHOD_DIR/experiments/summarize_official_reference.py" --workdir "$OFF_WORK" \
    --output "$OUT/official_reference" --prefix "$PREFIX" >> "$OUT/logs/gpu0_official.log" 2>&1
}

if [[ "$GPU_COUNT" -ge 2 ]]; then
  echo "=== start matched runs in parallel: GPU0 vanilla | GPU1 PRSS ==="
  run_vanilla & PID_V=$!
  run_prss & PID_P=$!
  fail=0
  if ! wait "$PID_V"; then
    echo "ERROR: matched vanilla failed"; tail -200 "$OUT/logs/gpu0_vanilla.log" || true; fail=1
  fi
  # Once GPU0 is free, run the byte-for-byte upstream anchor while PRSS can continue on GPU1.
  if [[ "$fail" -eq 0 ]]; then
    echo "=== GPU0 vanilla done; run exact upstream reference on GPU0 ==="
    if ! run_official_reference; then
      echo "ERROR: exact official reference failed"; tail -200 "$OUT/logs/gpu0_official.log" || true; fail=1
    fi
  fi
  if ! wait "$PID_P"; then
    echo "ERROR: PRSS failed"; tail -240 "$OUT/logs/gpu1_prss.log" || true; fail=1
  fi
  [[ "$fail" -ne 0 ]] && exit 4
else
  run_vanilla
  run_prss
  run_official_reference
fi

python "$METHOD_DIR/experiments/compare_nodecls.py" "$OUT" | tee "$OUT/comparison.txt"
echo "DONE: $OUT"
