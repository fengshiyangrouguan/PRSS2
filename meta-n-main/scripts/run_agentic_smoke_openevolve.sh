#!/usr/bin/env bash
# ============================================================================
# Openevolve smoke — 1 task each from alphaevolve, symreg, algotune
#
# Validates that the code-first agentic prompt works for solver_language=openevolve.
# Each smoke runs --bench-limit 1, --max-iterations 1, --beam 1x1, max_turns=8.
# Pass criterion: agentic loop emits a non-empty <code> block (i.e. trace.script
# is non-empty), regardless of score.
# ============================================================================

set -euo pipefail

# Fail fast if the API key isn't exported — the LLM calls would otherwise 401
# at the first iteration and the user would only notice after a long wait.
: "${OPENROUTER_API_KEY:?OPENROUTER_API_KEY not set — run 'source .env' from the repo root before this script.}"

MODEL="google/gemma-4-31b-it"
BASE_URL="https://openrouter.ai/api/v1"
OUTPUT_DIR="./experiments/agentic_smoke"
LOG_DIR="./logs/agentic_smoke"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

COMMON=(
  --benchmark-config none  # pin: this script encodes its own historical flag set; the bundled YAML must not rewrite it
  --model "$MODEL"
  --base-url "$BASE_URL"
  --output-dir "$OUTPUT_DIR"
  --max-tokens 16384
  --epsilon 0.02
  --max-depth 10
  --parallel 1
  --bench-limit 1
  --n-few-shot 5
  --max-val 5
  --max-retries 1
  --retry-threshold 0.3
  --beam-width 1
  --beam-candidates 1
  --max-iterations 1
  --patience 999
  --gate-tasks 0
  --novelty-alpha 0.3
  --seed 42
  --use-archive
  --use-agentic
  --agentic-max-turns 8
  --agentic-token-budget 200000
)

run_smoke() {
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

echo "============================================"
echo "Openevolve smoke"
echo "Started: $(date)"
echo "============================================"

# Run sequentially so we don't compete with running stages
run_smoke smoke_alphaevolve --benchmark alphaevolve_math || true
run_smoke smoke_algotune    --benchmark algotune        || true
run_smoke smoke_symreg      --benchmark symbolic_regression --bench-tasks bio_pop_growth || true

echo ""
echo "============================================"
echo "=== Smoke verdicts (script non-empty == prompt OK) ==="
for exp in smoke_alphaevolve smoke_algotune smoke_symreg; do
  python3 - <<PYEOF
import json, glob, os
exp = "$exp"
trace_files = glob.glob(f"$OUTPUT_DIR/{exp}/archive/gen0_seed/traces/*.json")
if not trace_files:
    print(f"{exp}: NO TRACE FOUND")
else:
    t = json.load(open(trace_files[0]))
    script_len = len(t.get('script') or '')
    score = t.get('score', 0.0)
    err = t.get('error_summary') or ''
    verdict = "PASS" if script_len > 50 else "FAIL"
    print(f"{exp}: {verdict} — script_len={script_len}, score={score:.3f}, err={err[:80]!r}")
PYEOF
done
echo "============================================"
echo "Smoke finished: $(date)"
