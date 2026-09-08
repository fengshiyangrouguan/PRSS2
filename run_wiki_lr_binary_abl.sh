#!/bin/bash
# Wiki-LR-Binary 9-config launcher (spec §9).  Runs exactly the nine unique
# IDs on the shared negatives manifest + group plan; never re-samples.
#
# SMOKE=1 (default) caps each run at two macro groups (--max-batches 16) and one
# epoch, purely to verify the pipeline (n_closed>0, aux active, no crash).
# Full runs: SMOKE=0 with EPOCHS/... as desired (9 arms are meant to run in one
# batch, not one-by-one).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"; cd "$ROOT"
DATA="${DATA_DIR:-datasets}"
NEG="${NEG:-datasets/wiki_lr_binary_negatives.json}"
PLAN="${PLAN:-datasets/wiki_lr_binary_group_plan.json}"
SEED="${SEED:-0}"
GPU="${GPU:-0}"
EPOCHS="${EPOCHS:-1}"
SMOKE_BATCHES="${SMOKE_BATCHES:-16}"     # 16 = two 8-batch groups (smoke)
LAMBDA="${LAMBDA:-0.088}"
OUT="${OUT:-outputs/wiki_lr_binary/seed${SEED}}"
mkdir -p "$OUT"
ARGS=(
  --seed "$SEED" --gpu "$GPU"
  --data-dir "$DATA" --negatives "$NEG" --group-plan "$PLAN"
  --epochs "$EPOCHS"
  --bs 200 --n-neighbors 5 --n-layers 3
  --trace-roots 160 --trace-pairs-per-parent 2
  --kf-group-batches 8 --kf-min-trees 896
  --lambda-kf "$LAMBDA" --ridge-eps 1e-3 --sketch-dim 64 --width-D 128
)
if [ "${SMOKE:-1}" = "1" ]; then ARGS+=(--max-batches "$SMOKE_BATCHES"); fi
for ID in R0 P0 P1 P2 S1 S2 C1 B1 E1; do
  echo "=== $ID ==="
  mkdir -p "$OUT/$ID"
  python -m scripts.train_tgb_link --config "$ID" "${ARGS[@]}" \
      --output "$OUT/$ID" > "$OUT/$ID.log" 2>&1 \
    || { echo "FAILED $ID"; tail -40 "$OUT/$ID.log"; exit 1; }
  echo "--- $ID tail ---"; tail -3 "$OUT/$ID.log"
done
echo "ALL 9 DONE -> $OUT"
