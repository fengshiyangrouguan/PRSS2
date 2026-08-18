#!/usr/bin/env bash
set -euo pipefail

# First-look PRSS experiment on TGN dynamic node classification.
# We deliberately avoid DDP inside a temporal-memory trajectory.  Instead:
#   GPU0: vanilla -> response_only control
#   GPU1: full PRSS
# This both uses the two cards and separates deep-supervision gains from quotient-selection gains.

PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "${PACKAGE_DIR}/.." && pwd)"
export TGN_DIR="${TGN_DIR:-${WORKSPACE_DIR}/tgn}"
DATASET="${DATASET:-wikipedia}"
PROCESSED_DIR="${PROCESSED_DIR:-${WORKSPACE_DIR}/processed_tgn_data}"
RAW_CSV="${RAW_CSV:-${WORKSPACE_DIR}/${DATASET}.csv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PACKAGE_DIR}/outputs/node_classification}"
EPOCHS="${EPOCHS:-15}"
LAYERS="${LAYERS:-2}"
DEGREE="${DEGREE:-10}"
BATCH_SIZE="${BATCH_SIZE:-200}"
CANDIDATE_DIM="${CANDIDATE_DIM:-256}"
SEED="${SEED:-0}"
MAX_AUX_ROOTS="${MAX_AUX_ROOTS:-4}"
MAX_AUX_NODES="${MAX_AUX_NODES:-8}"
LAMBDA_RESP="${LAMBDA_RESP:-0.5}"
LAMBDA_SPEC="${LAMBDA_SPEC:-0.05}"
SELECTION_METRIC="${SELECTION_METRIC:-ap}"
PATIENCE="${PATIENCE:-3}"
RUN_RESPONSE_ONLY="${RUN_RESPONSE_ONLY:-1}"
RUN_TESTS="${RUN_TESTS:-0}"
RUN_SYNTHETIC="${RUN_SYNTHETIC:-0}"
AUTO_DOWNLOAD="${AUTO_DOWNLOAD:-1}"
USE_ALL_VISIBLE_GPUS="${USE_ALL_VISIBLE_GPUS:-1}"

cd "${PACKAGE_DIR}"

if [[ "${USE_ALL_VISIBLE_GPUS}" == "1" ]]; then
  unset CUDA_VISIBLE_DEVICES || true
fi

need_processed=0
for path in "${PROCESSED_DIR}/ml_${DATASET}.csv" \
            "${PROCESSED_DIR}/ml_${DATASET}.npy" \
            "${PROCESSED_DIR}/ml_${DATASET}_node.npy"; do
  [[ -f "${path}" ]] || need_processed=1
done
if [[ "${need_processed}" == "1" ]]; then
  if [[ ! -f "${RAW_CSV}" ]]; then
    if [[ "${AUTO_DOWNLOAD}" != "1" ]]; then
      echo "Missing ${RAW_CSV}. Enable AUTO_DOWNLOAD=1 or provide RAW_CSV." >&2
      exit 2
    fi
    "${PACKAGE_DIR}/download_jodie_dataset.sh" "${DATASET}" "${RAW_CSV}"
  fi
  python experiments/preprocess_jodie.py --data "${DATASET}" --bipartite \
    --input "${RAW_CSV}" --output-dir "${PROCESSED_DIR}"
fi

if [[ "${RUN_TESTS}" == "1" ]]; then
  if python - <<'PYTEST' >/dev/null 2>&1
import pytest
PYTEST
  then
    python -m pytest -q
  else
    echo "WARNING: pytest is not installed; skipping optional unit tests. The experiment itself does not depend on pytest." >&2
  fi
fi
if [[ "${RUN_SYNTHETIC}" == "1" ]]; then
  mkdir -p "${OUTPUT_ROOT}"
  python experiments/synthetic_tree.py --steps 300 --output "${OUTPUT_ROOT}/synthetic_tree_quick.json"
fi

GPU_COUNT="$(python - <<'PY'
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
)"
echo "PyTorch-visible CUDA devices: ${GPU_COUNT}"
if (( GPU_COUNT == 0 )); then
  echo "WARNING: CUDA is not visible; full TGN runs will be slow on CPU." >&2
fi

BASE_DIR="${OUTPUT_ROOT}/${DATASET}/seed_${SEED}"
VANILLA_DIR="${BASE_DIR}/vanilla_tgn"
RESPONSE_DIR="${BASE_DIR}/prss_response_only"
PRSS_DIR="${BASE_DIR}/prss_full"
mkdir -p "${BASE_DIR}/logs"

common=(
  --data "${DATASET}"
  --data-dir "${PROCESSED_DIR}"
  --epochs "${EPOCHS}"
  --n-layer "${LAYERS}"
  --n-degree "${DEGREE}"
  --batch-size "${BATCH_SIZE}"
  --seed "${SEED}"
  --selection-metric "${SELECTION_METRIC}"
  --patience "${PATIENCE}"
  --gpu 0
)

run_vanilla() {
  if [[ -f "${VANILLA_DIR}/_SUCCESS.json" ]]; then
    echo "skip completed vanilla: ${VANILLA_DIR}"
    return 0
  fi
  rm -rf "${VANILLA_DIR}"; mkdir -p "${VANILLA_DIR}"
  python -u experiments/train_tgn_node_classification.py \
    "${common[@]}" --model vanilla --output "${VANILLA_DIR}"
}

run_prss_variant() {
  local variant="$1" out="$2"
  if [[ -f "${out}/_SUCCESS.json" ]]; then
    echo "skip completed ${variant}: ${out}"
    return 0
  fi
  rm -rf "${out}"; mkdir -p "${out}"
  python -u experiments/train_tgn_node_classification.py \
    "${common[@]}" --model prss --prss-variant "${variant}" --output "${out}" \
    --candidate-dim "${CANDIDATE_DIM}" \
    --max-aux-roots "${MAX_AUX_ROOTS}" \
    --max-aux-nodes "${MAX_AUX_NODES}" \
    --lambda-resp "${LAMBDA_RESP}" \
    --lambda-spec "${LAMBDA_SPEC}"
}

if (( GPU_COUNT >= 2 )); then
  if [[ "${RUN_RESPONSE_ONLY}" == "1" ]]; then
    echo "GPU0: matched vanilla TGN -> response-only control | GPU1: full PRSS-TGN"
  else
    echo "GPU0: matched vanilla TGN | GPU1: full PRSS-TGN"
  fi
  echo "Logs: ${BASE_DIR}/logs/gpu0_vanilla.log, ${BASE_DIR}/logs/gpu0_response_only.log, ${BASE_DIR}/logs/gpu1_prss_full.log"
  set +e
  (
    export CUDA_VISIBLE_DEVICES=0
    run_vanilla >"${BASE_DIR}/logs/gpu0_vanilla.log" 2>&1
    S_V=$?
    if (( S_V != 0 )); then exit "${S_V}"; fi
    if [[ "${RUN_RESPONSE_ONLY}" == "1" ]]; then
      run_prss_variant response_only "${RESPONSE_DIR}" >"${BASE_DIR}/logs/gpu0_response_only.log" 2>&1
    fi
  ) &
  PID0=$!
  ( export CUDA_VISIBLE_DEVICES=1; run_prss_variant full "${PRSS_DIR}" ) \
    >"${BASE_DIR}/logs/gpu1_prss_full.log" 2>&1 &
  PID1=$!
  wait "${PID0}"; S0=$?
  wait "${PID1}"; S1=$?
  set -e
  if (( S0 != 0 )); then
    echo "GPU0 vanilla/response pipeline failed (exit ${S0}):" >&2
    tail -n 240 "${BASE_DIR}/logs/gpu0_vanilla.log" >&2 || true
    tail -n 240 "${BASE_DIR}/logs/gpu0_response_only.log" >&2 || true
    exit "${S0}"
  fi
  if (( S1 != 0 )); then
    echo "GPU1 full PRSS failed (exit ${S1}):" >&2
    tail -n 300 "${BASE_DIR}/logs/gpu1_prss_full.log" >&2 || true
    exit "${S1}"
  fi
else
  echo "Fewer than two CUDA devices visible; running vanilla/full sequentially."
  run_vanilla 2>&1 | tee "${BASE_DIR}/logs/vanilla.log"
  run_prss_variant full "${PRSS_DIR}" 2>&1 | tee "${BASE_DIR}/logs/prss_full.log"
  if [[ "${RUN_RESPONSE_ONLY}" == "1" ]]; then
    run_prss_variant response_only "${RESPONSE_DIR}" 2>&1 | tee "${BASE_DIR}/logs/response_only.log"
  fi
fi

if [[ "${RUN_RESPONSE_ONLY}" == "1" ]]; then
  python experiments/compare_node_classification.py \
    --vanilla "${VANILLA_DIR}" --response-only "${RESPONSE_DIR}" --prss "${PRSS_DIR}" \
    --output "${BASE_DIR}/comparison.json"
else
  python experiments/compare_node_classification.py \
    --vanilla "${VANILLA_DIR}" --prss "${PRSS_DIR}" \
    --output "${BASE_DIR}/comparison.json"
fi

echo "Done. Key outputs:"
echo "  ${VANILLA_DIR}/results.json"
if [[ "${RUN_RESPONSE_ONLY}" == "1" ]]; then echo "  ${RESPONSE_DIR}/results.json"; fi
echo "  ${PRSS_DIR}/results.json"
echo "  ${BASE_DIR}/comparison.json"
