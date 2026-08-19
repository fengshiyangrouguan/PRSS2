#!/usr/bin/env bash
set -euo pipefail

TGN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${TGN_DIR}/.." && pwd)"
RAW_CSV="${RAW_CSV:-${PROJECT_DIR}/wikipedia.csv}"
DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/processed_tgn_data}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs_tgn_diagnostics}"
N_EPOCH="${N_EPOCH:-50}"
SEEDS="${SEEDS:-0 1 2}"
GPU="${GPU:-0}"

cd "${TGN_DIR}"
mkdir -p "${DATA_DIR}" "${OUTPUT_DIR}/baselines"

if [[ ! -f "${DATA_DIR}/ml_wikipedia.csv" || ! -f "${DATA_DIR}/ml_wikipedia.npy" || ! -f "${DATA_DIR}/ml_wikipedia_node.npy" ]]; then
  python -u utils/preprocess_data.py \
    --data wikipedia --bipartite --input "${RAW_CSV}" --output_dir "${DATA_DIR}"
fi
python -u diagnostics/verify_real_data.py \
  --raw-csv "${RAW_CSV}" --data-dir "${DATA_DIR}" \
  --output "${OUTPUT_DIR}/real_data_verification.json"

for memory in memory no_memory; do
  for layer in 1 2 3; do
    for seed in ${SEEDS}; do
      artifact="${OUTPUT_DIR}/baselines/${memory}_l${layer}_seed${seed}"
      memory_flag=()
      if [[ "${memory}" == "memory" ]]; then
        memory_flag=(--use_memory)
      fi
      if [[ ! -f "${artifact}/saved_models/tgn-wikipedia.pth" || ! -f "${artifact}/saved_models/tgn-wikipedia.json" || ! -f "${artifact}/results/tgn.pkl" ]]; then
        python -u train_self_supervised.py \
          --data wikipedia --data_dir "${DATA_DIR}" --artifacts_dir "${artifact}" \
          --prefix tgn --n_layer "${layer}" --n_degree 10 --n_epoch "${N_EPOCH}" \
          --bs 200 --patience 5 --seed "${seed}" --gpu "${GPU}" "${memory_flag[@]}"
      fi
    done
  done
done

python -u diagnostics/summarize_baselines.py \
  --artifacts-root "${OUTPUT_DIR}/baselines" --output "${OUTPUT_DIR}/depth_support"

# The primary theoretical test uses the same frozen L=1 state while exposing K=1,2,3 history.
for memory in memory no_memory; do
  artifact="${OUTPUT_DIR}/baselines/${memory}_l1_seed0"
  cuts="${OUTPUT_DIR}/${memory}/cuts"
  probes="${OUTPUT_DIR}/${memory}/conditional"
  if [[ ! -f "${cuts}/_SUCCESS.json" ]]; then
    python -u diagnostics/extract_cuts.py \
      --data-dir "${DATA_DIR}" \
      --checkpoint "${artifact}/saved_models/tgn-wikipedia.pth" \
      --config "${artifact}/saved_models/tgn-wikipedia.json" \
      --output "${cuts}" --max-hops 3 --batch-size 200 --gpu "${GPU}"
  fi
  if [[ ! -f "${probes}/_SUCCESS.json" ]]; then
    python -u diagnostics/conditional_residual.py \
      --cuts "${cuts}" --output "${probes}" --hops 1,2,3 \
      --variants structure,structure_edge,all --gpu "${GPU}"
  fi
  if [[ ! -f "${OUTPUT_DIR}/${memory}/collision/_SUCCESS.json" ]]; then
    python -u diagnostics/collision_analysis.py \
      --cuts "${cuts}" --probe "${probes}/probe_all_k3.pt" \
      --output "${OUTPUT_DIR}/${memory}/collision" --variant all --hop 3 --jobs -1 --gpu "${GPU}"
  fi
  if [[ ! -f "${OUTPUT_DIR}/${memory}/predictive_rank/_SUCCESS.json" ]]; then
    python -u diagnostics/predictive_rank.py \
      --cuts "${cuts}" --probe "${probes}/probe_all_k3.pt" \
      --output "${OUTPUT_DIR}/${memory}/predictive_rank" --variant all --hop 3 --gpu "${GPU}"
  fi
done

echo "All outputs are under ${OUTPUT_DIR}"
