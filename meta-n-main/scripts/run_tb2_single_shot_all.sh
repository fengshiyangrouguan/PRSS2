#!/usr/bin/env bash
# ============================================================================
# TB2 Single-Shot Category Experiments — 3-Slot Concurrent Runner
#
# Launches all 13 experiment runs across 3 parallel slots.
# Each slot runs its assigned experiments sequentially.
# The 3 slots run concurrently.
#
# Solver: single-shot (default Layer1Solver, NO --use-agentic).
# Output: ./experiments/single_shot/tb2_<category>/
#
# Usage:
#   source .env && bash scripts/run_tb2_single_shot_all.sh
#
# Prerequisites:
#   - Docker running
#   - OPENROUTER_API_KEY exported
#   - TB2 tasks cached at ~/.cache/harbor/tasks/packages/terminal-bench/
#
# To resume a crashed run, re-run this script — completed experiments
# (with summary.json) are automatically skipped.
# ============================================================================

set -euo pipefail

# Fail fast if the API key isn't exported — the LLM calls would otherwise 401
# at the first iteration and the user would only notice after a long wait.
: "${OPENROUTER_API_KEY:?OPENROUTER_API_KEY not set — run 'source .env' from the repo root before this script.}"

# --- Configuration ---
MODEL="google/gemma-4-31b-it"
BASE_URL="https://openrouter.ai/api/v1"
EXCLUDE="together"
MAX_TOKENS=16384
BENCH_DIR="$HOME/.cache/harbor/tasks/packages/terminal-bench"
OUTPUT_DIR="./experiments/single_shot"
LOG_DIR="./logs/tb2_runs"

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

# --- Shared args ---
COMMON=(
  --benchmark-config none  # pin: this script encodes its own historical flag set; the bundled YAML must not rewrite it
  --benchmark terminal_bench
  --bench-data-dir "$BENCH_DIR"
  --model "$MODEL"
  --base-url "$BASE_URL"
  --exclude-providers "$EXCLUDE"
  --max-tokens "$MAX_TOKENS"
  --output-dir "$OUTPUT_DIR"
  --use-archive
  --beam-width 1
  --beam-candidates 2
  --max-retries 1
  --retry-threshold 0.5
  --seed 42
)

# --- Helper: run one experiment (skip if already done) ---
run_experiment() {
  local exp_name="$1"; shift
  local log_file="$LOG_DIR/${exp_name}.log"

  if [ -f "${OUTPUT_DIR}/${exp_name}/summary.json" ]; then
    echo "[SKIP] $exp_name — already completed (summary.json exists)"
    return 0
  fi

  if [ -f "${OUTPUT_DIR}/${exp_name}/run.log" ] && pgrep -f "exp-name ${exp_name}" > /dev/null 2>&1; then
    echo "[SKIP] $exp_name — already running"
    return 0
  fi

  echo "[START] $exp_name — $(date '+%H:%M:%S')"
  python -m meta_n.main "${COMMON[@]}" --exp-name "$exp_name" "$@" \
    > "$log_file" 2>&1
  local rc=$?

  if [ $rc -eq 0 ]; then
    echo "[DONE]  $exp_name — $(date '+%H:%M:%S') (exit 0)"
  else
    echo "[FAIL]  $exp_name — $(date '+%H:%M:%S') (exit $rc) — see $log_file"
  fi
  return $rc
}

# ============================================================================
# Slot 1 (~300 min): F → D → A → B
# ============================================================================
slot1() {
  echo "=== Slot 1 started at $(date '+%H:%M:%S') ==="

  # Run F — data-science (8 tasks, 6M+2H)
  run_experiment tb2_data_science \
    --bench-tasks hf-model-inference mcmc-sampling-stan mteb-leaderboard mteb-retrieve query-optimize reshard-c4-data rstan-to-pystan sam-cell-seg \
    --epsilon 0.03 --max-depth 4 --parallel 4 \
    --max-iterations 8 --patience 3 --gate-tasks 2

  # Run D — scientific-computing (8 tasks, 5M+3H)
  run_experiment tb2_scientific_computing \
    --bench-tasks adaptive-rejection-sampler bn-fit-modify dna-assembly dna-insert modernize-scientific-stack protein-assembly raman-fitting tune-mjcf \
    --epsilon 0.03 --max-depth 4 --parallel 4 \
    --max-iterations 8 --patience 3 --gate-tasks 2

  # Run A — debugging (5 tasks, 1E+4M)
  run_experiment tb2_debugging \
    --bench-tasks overfull-hbox build-cython-ext custom-memory-heap-crash merge-diff-arc-agi-task sqlite-db-truncate \
    --epsilon 0.03 --max-depth 4 --parallel 4 \
    --max-iterations 8 --patience 3 --gate-tasks 0

  # Run B — data-processing (4 tasks, 4M)
  run_experiment tb2_data_processing \
    --bench-tasks financial-document-processor log-summary-date-ranges multi-source-data-merger regex-log \
    --epsilon 0.03 --max-depth 4 --parallel 4 \
    --max-iterations 8 --patience 3 --gate-tasks 0

  echo "=== Slot 1 finished at $(date '+%H:%M:%S') ==="
}

# ============================================================================
# Slot 2 (~295 min): I → C → M → L
# ============================================================================
slot2() {
  echo "=== Slot 2 started at $(date '+%H:%M:%S') ==="

  # Run I — software-eng easy+medium (13 tasks, 3E+10M)
  run_experiment tb2_software_eng_easy_medium \
    --bench-tasks cobol-modernization fix-git prove-plus-comm build-pmars build-pov-ray code-from-image git-leak-recovery headless-terminal kv-store-grpc polyglot-c-py pypi-server schemelike-metacircular-eval winning-avg-corewars \
    --epsilon 0.03 --max-depth 4 --parallel 4 \
    --max-iterations 10 --patience 4 --gate-tasks 2

  # Run C — security (8 tasks, 6M+2H)
  run_experiment tb2_security \
    --bench-tasks break-filter-js-from-html crack-7z-hash filter-js-from-html fix-code-vulnerability openssl-selfsigned-cert password-recovery sanitize-git-repo vulnerable-secret \
    --epsilon 0.03 --max-depth 4 --parallel 4 \
    --max-iterations 8 --patience 3 --gate-tasks 2

  # Run M — misc/singletons (5 tasks, 3M+2H)
  run_experiment tb2_misc \
    --bench-tasks chess-best-move constraints-scheduling portfolio-optimization sparql-university video-processing \
    --epsilon 0.03 --max-depth 4 --parallel 4 \
    --max-iterations 6 --patience 3 --gate-tasks 0

  # Run L — machine-learning (3 tasks, 2M+1H)
  run_experiment tb2_machine_learning \
    --bench-tasks caffe-cifar-10 distribution-search llm-inference-batching-scheduler \
    --epsilon 0.03 --max-depth 4 --parallel 3 \
    --max-iterations 6 --patience 3 --gate-tasks 0

  echo "=== Slot 2 finished at $(date '+%H:%M:%S') ==="
}

# ============================================================================
# Slot 3 (~315 min): J → E → H → G → K
# ============================================================================
slot3() {
  echo "=== Slot 3 started at $(date '+%H:%M:%S') ==="

  # Run J — software-eng hard (13 tasks, 13H)
  run_experiment tb2_software_eng_hard \
    --bench-tasks cancel-async-tasks circuit-fibsqrt fix-ocaml-gc gpt2-codegolf make-doom-for-mips make-mips-interpreter path-tracing path-tracing-reverse polyglot-rust-c regex-chess torch-pipeline-parallelism torch-tensor-parallelism write-compressor \
    --epsilon 0.03 --max-depth 4 --parallel 4 \
    --max-iterations 6 --patience 3 --gate-tasks 2

  # Run E — system-administration (9 tasks, 7M+2H)
  run_experiment tb2_system_administration \
    --bench-tasks compile-compcert configure-git-webserver git-multibranch install-windows-3-11 mailman nginx-request-logging qemu-alpine-ssh qemu-startup sqlite-with-gcov \
    --epsilon 0.03 --max-depth 4 --parallel 4 \
    --max-iterations 8 --patience 3 --gate-tasks 2

  # Run H — model-training (4 tasks, 3M+1H)
  run_experiment tb2_model_training \
    --bench-tasks count-dataset-tokens pytorch-model-cli pytorch-model-recovery train-fasttext \
    --epsilon 0.03 --max-depth 4 --parallel 4 \
    --max-iterations 8 --patience 3 --gate-tasks 0

  # Run G — file-operations (5 tasks, 4M+1H)
  run_experiment tb2_file_operations \
    --bench-tasks db-wal-recovery extract-elf extract-moves-from-video gcode-to-text large-scale-text-editing \
    --epsilon 0.03 --max-depth 4 --parallel 4 \
    --max-iterations 8 --patience 3 --gate-tasks 0

  # Run K — mathematics (4 tasks, 1M+3H)
  run_experiment tb2_mathematics \
    --bench-tasks feal-differential-cryptanalysis feal-linear-cryptanalysis largest-eigenval model-extraction-relu-logits \
    --epsilon 0.03 --max-depth 4 --parallel 4 \
    --max-iterations 6 --patience 3 --gate-tasks 0

  echo "=== Slot 3 finished at $(date '+%H:%M:%S') ==="
}

# ============================================================================
# Launch all 3 slots concurrently
# ============================================================================
echo "============================================"
echo "TB2 Single-Shot Full Experiment — 13 runs, 3 slots"
echo "Started: $(date)"
echo "Logs:    $LOG_DIR/"
echo "Results: $OUTPUT_DIR/tb2_*/"
echo "============================================"
echo ""

slot1 &
PID1=$!
slot2 &
PID2=$!
slot3 &
PID3=$!

echo "Slot PIDs: $PID1, $PID2, $PID3"
echo "Monitor: tail -f $LOG_DIR/*.log"
echo ""

# Wait for all slots
FAILED=0
wait $PID1 || FAILED=$((FAILED + 1))
wait $PID2 || FAILED=$((FAILED + 1))
wait $PID3 || FAILED=$((FAILED + 1))

echo ""
echo "============================================"
echo "All slots finished: $(date)"
echo "Failed slots: $FAILED"
echo ""

# Summary
echo "=== Results Summary ==="
for d in "$OUTPUT_DIR"/tb2_*/; do
  exp=$(basename "$d")
  if [ -f "$d/summary.json" ]; then
    score=$(python3 -c "import json; d=json.load(open('$d/summary.json')); print(f'{d.get(\"best_mean_score\", d.get(\"mean_scores\", {}).get(str(max(int(k) for k in d.get(\"mean_scores\", {\"1\": 0}).keys())), 0)):.3f}')" 2>/dev/null || echo "???")
    echo "  $exp: best=$score"
  else
    echo "  $exp: NO RESULTS"
  fi
done

echo ""
echo "Done. Detailed logs in $LOG_DIR/"
