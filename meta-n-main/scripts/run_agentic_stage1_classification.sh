#!/usr/bin/env bash
# ============================================================================
# Agentic Stage 1 — Classification (S2D + LawBench)
#
# Mirrors baselines:
#   - full_symptom2disease_v5
#   - full_lawbench_v5
# Adds: --use-agentic --agentic-max-turns 5 --agentic-token-budget 200000
# Bumps: --parallel 1 → 2 (per agentic full-scale plan)
#
# Usage:
#   source .env && bash scripts/run_agentic_stage1_classification.sh
#
# Resumes by skipping completed runs (presence of summary.json).
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

COMMON=(
  --benchmark-config none  # pin: this script encodes its own historical flag set; the bundled YAML must not rewrite it
  --model "$MODEL"
  --base-url "$BASE_URL"
  --output-dir "$OUTPUT_DIR"
  --max-tokens 16384
  --epsilon 0.02
  --max-depth 10
  --seed 42
  --use-archive
  --use-agentic
  --agentic-max-turns 5
  --agentic-token-budget 200000
  --novelty-alpha 0.3
)

run_experiment() {
  local exp_name="$1"; shift
  local log_file="$LOG_DIR/${exp_name}.log"

  if [ -f "${OUTPUT_DIR}/${exp_name}/summary.json" ]; then
    echo "[SKIP] $exp_name — already completed"
    return 0
  fi

  echo "[START] $exp_name — $(date '+%H:%M:%S')"
  python -m meta_n.main "${COMMON[@]}" --exp-name "$exp_name" "$@" \
    > "$log_file" 2>&1
  local rc=$?

  if [ $rc -eq 0 ]; then
    echo "[DONE]  $exp_name — $(date '+%H:%M:%S')"
  else
    echo "[FAIL]  $exp_name — exit $rc — see $log_file"
  fi
  return $rc
}

# --- Mirror full_symptom2disease_v5 ---
s2d() {
  run_experiment full_s2d_agentic \
    --benchmark symptom2disease \
    --parallel 2 \
    --n-few-shot 10 \
    --max-val 50 \
    --max-retries 3 \
    --retry-threshold 0.1 \
    --beam-width 1 \
    --beam-candidates 3 \
    --max-iterations 20 \
    --patience 999 \
    --gate-tasks 0
}

# --- Mirror full_lawbench_v5 ---
lawbench() {
  run_experiment full_lawbench_agentic \
    --benchmark lawbench_charge \
    --parallel 2 \
    --n-few-shot 10 \
    --max-val 50 \
    --max-retries 3 \
    --retry-threshold 0.1 \
    --beam-width 1 \
    --beam-candidates 3 \
    --max-iterations 20 \
    --patience 999 \
    --gate-tasks 0
}

echo "============================================"
echo "Agentic Stage 1 — Classification"
echo "Started: $(date)"
echo "============================================"

s2d &
PID1=$!
lawbench &
PID2=$!

echo "PIDs: s2d=$PID1, lawbench=$PID2"
echo "Monitor: tail -f $LOG_DIR/full_*_agentic.log"

FAILED=0
wait $PID1 || FAILED=$((FAILED + 1))
wait $PID2 || FAILED=$((FAILED + 1))

echo ""
echo "Stage 1 finished: $(date), failed=$FAILED"
exit $FAILED
