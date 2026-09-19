#!/usr/bin/env bash
# ============================================================================
# Agentic Stage 2 — CO-Bench
#
# Mirrors baseline full_cobench_v5.
# Adds: --use-agentic --agentic-max-turns 8 --agentic-token-budget 200000
# Bumps: --parallel 1 → 2
# ============================================================================

set -euo pipefail

# Fail fast if the API key isn't exported — the LLM calls would otherwise 401
# at the first iteration and the user would only notice after a long wait.
: "${OPENROUTER_API_KEY:?OPENROUTER_API_KEY not set — run 'source .env' from the repo root before this script.}"

MODEL="google/gemma-4-31b-it"
BASE_URL="https://openrouter.ai/api/v1"
OUTPUT_DIR="./experiments/agentic"
LOG_DIR="./logs/agentic_runs"
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

EXP_NAME="full_cobench_agentic"
LOG_FILE="$LOG_DIR/${EXP_NAME}.log"

if [ -f "${OUTPUT_DIR}/${EXP_NAME}/summary.json" ]; then
  echo "[SKIP] $EXP_NAME — already completed"
  exit 0
fi

echo "============================================"
echo "Agentic Stage 2 — CO-Bench"
echo "Started: $(date)"
echo "============================================"

python -m meta_n.main --benchmark-config none \
  --benchmark co_bench \
  --model "$MODEL" \
  --base-url "$BASE_URL" \
  --max-tokens 16384 \
  --epsilon 0.02 \
  --max-depth 10 \
  --parallel 2 \
  --n-few-shot 5 \
  --max-val 50 \
  --max-retries 2 \
  --retry-threshold 0.1 \
  --beam-width 2 \
  --beam-candidates 2 \
  --max-iterations 8 \
  --patience 4 \
  --gate-tasks 3 \
  --novelty-alpha 0.3 \
  --seed 42 \
  --use-archive \
  --use-agentic \
  --agentic-max-turns 8 \
  --agentic-token-budget 200000 \
  --output-dir "$OUTPUT_DIR" \
  --exp-name "$EXP_NAME" \
  > "$LOG_FILE" 2>&1

rc=$?
echo "Stage 2 finished: $(date), exit=$rc"
exit $rc
