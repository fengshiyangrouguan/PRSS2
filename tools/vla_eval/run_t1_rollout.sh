#!/bin/bash
# T1 LIBERO-Mem rollout driver.
#
# Two separate python envs on purpose:
#   policy server (deploy.py)  -> env_memvla   (torch 2.8 + the VLA code)
#   simulator   (eval_libero_t1)-> env_libero  (libero/robosuite/mujoco + tf)
# deploy.py binds its port only AFTER the weights are loaded, so the evaluator
# polls the TCP port instead of sleeping a hardcoded 1800 s.
#
# Usage:
#   bash run_t1_rollout.sh <CHECKPOINT.pt> [EXTRA EVAL ARGS...]
#   bash run_t1_rollout.sh --dry-run <CHECKPOINT.pt>
set -uo pipefail

CKPT="${1:-}"; shift || true
DRY=""
if [ "$CKPT" = "--dry-run" ]; then DRY="--dry-run"; CKPT="${1:-}"; shift || true; fi
if [ -z "$CKPT" ]; then echo "usage: $0 [--dry-run] <checkpoint.pt> [eval args...]"; exit 2; fi
if [ ! -f "$CKPT" ]; then echo "!! checkpoint not found: $CKPT"; exit 2; fi

MEMVLA=/root/autodl-tmp/PRSS2/third_party/memoryvla
PRSS2SRC=/root/autodl-tmp/PRSS2/src
LIBERO=/root/autodl-tmp/libero-mem
POLICY_PY=/root/autodl-tmp/env_memvla/bin/python
SIM_PY=/root/autodl-tmp/env_libero/bin/python
T1EVAL=/root/autodl-tmp/t1_eval
RUN_DIR="$(dirname "$(readlink -f "$CKPT")")"
CKPT_TAG="$(basename "$CKPT" .pt)"

# deploy.py resolves config.yaml via dirname(dirname(saved_model_path)) and
# load_vla asserts checkpoint_pt.parent.name == "checkpoints". So the server
# must be pointed at <run_dir>/checkpoints/<ckpt>.pt, which finalize_run_dir.py
# creates as a symlink to the real checkpoint.
if [ "$(basename "$RUN_DIR")" = "checkpoints" ]; then
  DEPLOY_CKPT="$CKPT"
  RUN_DIR="$(dirname "$RUN_DIR")"
else
  DEPLOY_CKPT="$RUN_DIR/checkpoints/$(basename "$CKPT")"
fi

PORT="${PORT:-6877}"
# Official LIBERO-Mem protocol (scripts/mem_6_run_evaluation_env_pred.py):
#   num_trials_per_task = 20, num_steps_wait = 20, seed = 7, resolution = 256
TRIALS="${TRIALS:-20}"
WAIT="${WAIT:-20}"
MAXSTEPS="${MAXSTEPS:-600}"
SERVER_TIMEOUT="${SERVER_TIMEOUT:-900}"
OUT_DIR="${OUT_DIR:-$RUN_DIR/rollout_t1_$CKPT_TAG}"

echo "=== T1 rollout ==="
echo "  checkpoint : $CKPT"
echo "  deploy ckpt: $DEPLOY_CKPT"
echo "  run dir    : $RUN_DIR"
echo "  out dir    : $OUT_DIR"
echo "  port       : $PORT   trials: $TRIALS   max-steps: $MAXSTEPS"
echo "  server tout: ${SERVER_TIMEOUT}s"
echo

# ---- preflight: the run dir must look deployable -------------------------
for f in config.json dataset_statistics.json config.yaml; do
  if [ ! -f "$RUN_DIR/$f" ]; then
    echo "!! $RUN_DIR/$f missing."
    echo "   Run:  $POLICY_PY $T1EVAL/finalize_run_dir.py --run-dir $RUN_DIR"
    exit 3
  fi
done
if [ ! -e "$DEPLOY_CKPT" ]; then
  echo "!! $DEPLOY_CKPT missing (load_vla requires the checkpoint to sit in a"
  echo "   'checkpoints/' dir). Re-run finalize_run_dir.py."
  exit 3
fi
echo "preflight: config.json + dataset_statistics.json + config.yaml + checkpoints/ present"

UNNORM_KEY="${UNNORM_KEY:-libero_mem_no_noops}"

# ---- dry run: resolve everything, start nothing --------------------------
if [ -n "$DRY" ]; then
  echo
  echo "--- DRY RUN ---"
  cd "$MEMVLA" || exit 1
  PYTHONPATH="$MEMVLA:$MEMVLA/evaluation/libero:$LIBERO" \
  MUJOCO_GL=osmesa \
    "$SIM_PY" "$T1EVAL/eval_libero_t1.py" \
      --dry-run \
      --out-dir "$OUT_DIR" \
      --ckpt "$CKPT" \
      --unnorm-key "$UNNORM_KEY" \
      --config-path "$RUN_DIR/config.json" \
      --port "$PORT" \
      --num-trials "$TRIALS" \
      --max-steps "$MAXSTEPS" \
      "$@"
  rc=$?
  echo "dry-run rc=$rc"
  exit $rc
fi

# ---- policy server -------------------------------------------------------
cd "$MEMVLA" || exit 1
POLICY_LOG="$OUT_DIR.policy.log"
mkdir -p "$OUT_DIR"
BASE_CKPT="${BASE_CKPT:-/root/autodl-tmp/openvla-7b-prismatic/checkpoints/step-295000-epoch-40-loss=0.2200.pt}"
if [ ! -f "$BASE_CKPT" ]; then echo "!! base checkpoint missing: $BASE_CKPT"; exit 3; fi

echo "starting policy server (two-stage restore) -> $POLICY_LOG"
echo "  base  : $BASE_CKPT"
echo "  delta : $CKPT"
PYTHONPATH="$MEMVLA:$PRSS2SRC" \
LLAMA2_LOCAL_PATH=/root/autodl-tmp/Llama-2-7b-hf \
T1_BASE_CKPT="$BASE_CKPT" \
T1_DELTA_CKPT="$CKPT" \
OMP_NUM_THREADS=8 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$POLICY_PY" -u "$T1EVAL/deploy_t1.py" \
    --saved_model_path "$DEPLOY_CKPT" \
    --unnorm_key "$UNNORM_KEY" \
    --adaptive_ensemble_alpha 0.1 \
    --cfg_scale 1.5 \
    --port "$PORT" \
    --use_bf16 \
    --action_chunking \
    --action_chunking_window 8 \
    > "$POLICY_LOG" 2>&1 &
POLICY_PID=$!
echo "policy pid=$POLICY_PID"

cleanup() {
  if kill -0 "$POLICY_PID" 2>/dev/null; then
    echo "stopping policy server pid=$POLICY_PID"
    kill "$POLICY_PID" 2>/dev/null
    wait "$POLICY_PID" 2>/dev/null
  fi
}
trap cleanup EXIT

# ---- evaluator (polls the port itself; bounded wait) ---------------------
echo "running T1 evaluator (waits up to ${SERVER_TIMEOUT}s for the server)"
PYTHONPATH="$MEMVLA:$MEMVLA/evaluation/libero:$LIBERO" \
MUJOCO_GL=osmesa \
  "$SIM_PY" -u "$T1EVAL/eval_libero_t1.py" \
    --out-dir "$OUT_DIR" \
    --ckpt "$CKPT" \
    --unnorm-key "$UNNORM_KEY" \
    --config-path "$RUN_DIR/config.json" \
    --port "$PORT" \
    --server-timeout "$SERVER_TIMEOUT" \
    --num-trials "$TRIALS" \
    --num-steps-wait "$WAIT" \
    --max-steps "$MAXSTEPS" \
    "$@"
rc=$?
echo "evaluator rc=$rc"
exit $rc
