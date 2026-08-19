#!/usr/bin/env bash
set -euo pipefail

PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "${PACKAGE_DIR}/.." && pwd)"
export TGN_DIR="${TGN_DIR:-${WORKSPACE_DIR}/tgn}"
DATASET="${DATASET:-wikipedia}"
PROCESSED_DIR="${PROCESSED_DIR:-${WORKSPACE_DIR}/processed_tgn_data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PACKAGE_DIR}/outputs/ablations}"
EPOCHS="${EPOCHS:-50}"
CANDIDATE_DIM="${CANDIDATE_DIM:-256}"
MONITOR_EVERY="${MONITOR_EVERY:-50}"
SEEDS="${SEEDS:-0 1 2}"
VARIANTS="${VARIANTS:-fixed_random pca direct linear_reader_svd no_nonlinear_lift neural_svd_no_spec full}"

cd "${PACKAGE_DIR}"
for seed in ${SEEDS}; do
  for variant in ${VARIANTS}; do
    destination="${OUTPUT_ROOT}/${DATASET}/${variant}/seed_${seed}"
    if [[ -f "${destination}/_SUCCESS.json" ]]; then
      echo "skip completed ${variant} seed=${seed}"
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
      --variant "${variant}" \
      --candidate-dim "${CANDIDATE_DIM}" \
      --monitor-every "${MONITOR_EVERY}" \
      --use-memory \
      --epochs "${EPOCHS}" \
      --seed "${seed}"
    python experiments/render_monitor_report.py --run-dir "${destination}"
  done
done
python experiments/compare_prss_runs.py \
  --runs-root "${OUTPUT_ROOT}/${DATASET}" \
  --output "${OUTPUT_ROOT}/${DATASET}/evidence_report.json"
