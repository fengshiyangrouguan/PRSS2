#!/usr/bin/env bash
set -euo pipefail

PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "${PACKAGE_DIR}/.." && pwd)"
export TGN_DIR="${TGN_DIR:-${WORKSPACE_DIR}/tgn}"
DATASET="${DATASET:-wikipedia}"
PROCESSED_DIR="${PROCESSED_DIR:-${WORKSPACE_DIR}/processed_tgn_data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PACKAGE_DIR}/outputs/lambda_spec}"
EPOCHS="${EPOCHS:-50}"
CANDIDATE_DIM="${CANDIDATE_DIM:-256}"
MONITOR_EVERY="${MONITOR_EVERY:-50}"
SEEDS="${SEEDS:-0 1 2}"
LAMBDA_SPECS="${LAMBDA_SPECS:-0 0.01 0.1 0.5 1.0}"

cd "${PACKAGE_DIR}"
for seed in ${SEEDS}; do
  for lambda_spec in ${LAMBDA_SPECS}; do
    destination="${OUTPUT_ROOT}/${DATASET}/lambda_${lambda_spec}/seed_${seed}"
    if [[ -f "${destination}/_SUCCESS.json" ]]; then
      echo "skip completed lambda_spec=${lambda_spec} seed=${seed}"
      if [[ ! -f "${destination}/monitor/mechanism_dashboard.png" ]]; then
        python experiments/render_monitor_report.py --run-dir "${destination}"
      fi
      continue
    fi
    if [[ -d "${destination}" ]]; then
      archived="${destination}_incomplete_$(date +%Y%m%d_%H%M%S)"
      echo "archive incomplete run to ${archived}"
      mv "${destination}" "${archived}"
    fi
    python experiments/train_tgn_prss.py \
      --data "${DATASET}" \
      --data-dir "${PROCESSED_DIR}" \
      --output "${destination}" \
      --variant full \
      --candidate-dim "${CANDIDATE_DIM}" \
      --lambda-spec "${lambda_spec}" \
      --monitor-every "${MONITOR_EVERY}" \
      --use-memory \
      --epochs "${EPOCHS}" \
      --seed "${seed}"
    python experiments/render_monitor_report.py --run-dir "${destination}"
  done
done
