#!/usr/bin/env bash
# ============================================================================
# Agentic Stage 3b — Symbolic Regression (4 sub-benchmarks)
#
# Mirrors baselines:
#   - full_sr_bio_pop_growth_v2
#   - full_sr_chem_react
#   - full_sr_matsci
#   - full_sr_phys_osc
# Adds: --use-agentic --agentic-max-turns 8 --agentic-token-budget 200000
# Parallelism: 4 concurrent processes, each --parallel 2 (8 worker streams)
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
  --benchmark symbolic_regression
  --model "$MODEL"
  --base-url "$BASE_URL"
  --output-dir "$OUTPUT_DIR"
  --max-tokens 16384
  --epsilon 0.02
  --max-depth 10
  --parallel 2
  --n-few-shot 5
  --max-val 50
  --max-retries 1
  --retry-threshold 0.5
  --beam-width 2
  --beam-candidates 2
  --max-iterations 6
  --patience 10
  --gate-tasks 3
  --novelty-alpha 0.3
  --seed 42
  --use-archive
  --use-agentic
  --agentic-max-turns 8
  --agentic-token-budget 200000
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

bio_pop()    { run_experiment full_sr_bio_pop_growth_agentic --bench-tasks bio_pop_growth; }
chem_react() { run_experiment full_sr_chem_react_agentic     --bench-tasks chem_react; }
matsci()     { run_experiment full_sr_matsci_agentic         --bench-tasks matsci; }
phys_osc()   { run_experiment full_sr_phys_osc_agentic       --bench-tasks phys_osc; }

echo "============================================"
echo "Agentic Stage 3b — SR (4 sub-benchmarks)"
echo "Started: $(date)"
echo "============================================"

bio_pop &    PID1=$!
chem_react & PID2=$!
matsci &     PID3=$!
phys_osc &   PID4=$!

echo "PIDs: bio_pop=$PID1, chem_react=$PID2, matsci=$PID3, phys_osc=$PID4"

FAILED=0
wait $PID1 || FAILED=$((FAILED + 1))
wait $PID2 || FAILED=$((FAILED + 1))
wait $PID3 || FAILED=$((FAILED + 1))
wait $PID4 || FAILED=$((FAILED + 1))

echo ""
echo "Stage 3b finished: $(date), failed=$FAILED"
exit $FAILED
