#!/usr/bin/env bash
set -euo pipefail

PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "${PACKAGE_DIR}/.." && pwd)"
export TGN_DIR="${TGN_DIR:-${WORKSPACE_DIR}/tgn}"
DATASET="${DATASET:-wikipedia}"
RAW_CSV="${RAW_CSV:-${WORKSPACE_DIR}/${DATASET}.csv}"
PROCESSED_DIR="${PROCESSED_DIR:-${WORKSPACE_DIR}/processed_tgn_data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PACKAGE_DIR}/outputs/autodl}"
VARIANT="${VARIANT:-full}"
EPOCHS="${EPOCHS:-50}"
CANDIDATE_DIM="${CANDIDATE_DIM:-256}"
MONITOR_EVERY="${MONITOR_EVERY:-50}"

cd "${PACKAGE_DIR}"
python -m pytest -q
python experiments/synthetic_tree.py \
  --output "${OUTPUT_ROOT}/synthetic_tree.json"

if [[ ! -f "${PROCESSED_DIR}/ml_${DATASET}.csv" || \
      ! -f "${PROCESSED_DIR}/ml_${DATASET}.npy" || \
      ! -f "${PROCESSED_DIR}/ml_${DATASET}_node.npy" ]]; then
  if [[ ! -f "${RAW_CSV}" ]]; then
    echo "Raw CSV not found: ${RAW_CSV}" >&2
    exit 2
  fi
  python experiments/preprocess_jodie.py \
    --data "${DATASET}" --bipartite --input "${RAW_CSV}" --output-dir "${PROCESSED_DIR}"
fi

DESTINATION="${OUTPUT_ROOT}/${DATASET}/${VARIANT}"
if [[ -f "${DESTINATION}/_SUCCESS.json" ]]; then
  echo "skip completed ${VARIANT} run"
  if [[ ! -f "${DESTINATION}/monitor/mechanism_dashboard.png" ]]; then
    python experiments/render_monitor_report.py --run-dir "${DESTINATION}"
  fi
  exit 0
fi
if [[ -d "${DESTINATION}" && ! -f "${DESTINATION}/_SUCCESS.json" ]]; then
  archived="${DESTINATION}_incomplete_$(date +%Y%m%d_%H%M%S)"
  echo "archive incomplete run to ${archived}"
  mv "${DESTINATION}" "${archived}"
fi
python experiments/train_tgn_prss.py \
  --data "${DATASET}" \
  --data-dir "${PROCESSED_DIR}" \
  --output "${DESTINATION}" \
  --variant "${VARIANT}" \
  --candidate-dim "${CANDIDATE_DIM}" \
  --monitor-every "${MONITOR_EVERY}" \
  --use-memory \
  --epochs "${EPOCHS}"
python experiments/render_monitor_report.py --run-dir "${DESTINATION}"
