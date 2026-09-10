#!/bin/bash
# develop_UCI — UCI formal 3-arm x 5 seeds, 3 parallel (one arm per slot,
# seed rounds serial), with the repr-lr compensation restored from the wiki
# line (Eighth-review follow-up: repr steps ~35x less often than head).
#
# Phase 0: lambda calibration (once; skips if lambda_kyfan.json exists).
# Phase 1: per seed, ours + taskonly + vanilla launch in sequence under a
#   memory gate (32GB card: round starts at <8000MiB, each launch confirmed
#   by a +3GB ramp or process exit).  Idempotent: complete summary.json
#   SKIPs, a matching pgrep marks BUSY, failed runs rerun on a second pass.
set -u
cd /root/autodl-tmp/PRSS2_uci_v2 || exit 1
export PYTHONPATH=/root/autodl-tmp/PRSS2_uci_v2:/root/autodl-tmp/benchtemp/experimental_codes/tgn-jodie-dyrep
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/root/miniconda3/bin/python
ROOT=/root/autodl-tmp/PRSS2_uci_v2/outputs/uci_formal_v2
DATA_DIR=/root/autodl-tmp/benchtemp/data_uci
OFFPY=/root/autodl-tmp/PRSS2_uci_v2/scripts/official_uci_vanilla.py
LR=1e-4
REPR_LR=3e-4          # wiki-line ratio (repr-lr 1e-3 @ lr 3e-4 ~= 3.3x)
CALIB=outputs/lambda_calib_v2/lambda_kyfan.json
mkdir -p "$ROOT/logs"

gpu_used_mb() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1
}

wait_gpu() {  # wait until used < $1 MiB
  local limit="$1"
  while :; do
    local used
    used=$(gpu_used_mb)
    if [ -n "$used" ] && [ "$used" -lt "$limit" ]; then return 0; fi
    sleep 20
  done
}

confirm_ramp() {  # started pid shows +3GB growth or exits
  local pid="$1" before="$2"
  for _ in $(seq 1 24); do
    if ! kill -0 "$pid" 2>/dev/null; then return 0; fi
    sleep 10
    local used
    used=$(gpu_used_mb)
    if [ -n "$used" ] && [ "$used" -ge $((before + 3000)) ]; then return 0; fi
  done
  return 0
}

skip_or_busy() {  # "$out" "$proc_pat" "$label" -> 0 = skip/busy, 1 = run
  local out="$1" pat="$2" label="$3"
  if [ -f "$out/summary.json" ] && grep -q '"status": "complete"' "$out/summary.json"; then
    echo "SKIP $label" >> "$ROOT/run.log"; return 0
  fi
  if [ -f "$out/summary.json" ] && ! grep -q '"status": "complete"' "$out/summary.json"; then
    rm -f "$out/summary.json"   # failed-residue must not masquerade as done
    echo "RESET $label (failed residue)" >> "$ROOT/run.log"
  fi
  if pgrep -f "$pat" >/dev/null 2>&1; then
    echo "BUSY $label" >> "$ROOT/run.log"; return 0
  fi
  return 1
}

# ---- phase 0: lambda calibration (once) ----
if [ ! -f "$CALIB" ]; then
  wait_gpu 8000
  echo "CALIB_START $(date '+%H:%M:%S')" > outputs/lambda_calib_v2.log
  mkdir -p outputs/lambda_calib_v2
  $PY -m scripts.train_uci_link \
    --arm 2obs_aligned --data-dir "$DATA_DIR" --gpu 0 \
    --calibrate --calib-groups 8 --kf-group-batches 40 \
    --n-neighbors 10 --n-layers 3 --seed 0 \
    --output outputs/lambda_calib_v2 >> outputs/lambda_calib_v2.log 2>&1
  echo "CALIB_DONE rc=$? $(date '+%H:%M:%S')" >> outputs/lambda_calib_v2.log
fi
LAM0=$($PY -c "import json;print(json.load(open('$CALIB'))['lambda0'])")
LAM=$($PY -c "print(0.088 * 0.15 * float('$LAM0'))")
echo "LAMBDA=$LAM (lambda0=$LAM0) $(date '+%H:%M:%S')" >> "$ROOT/run.log"

run_ours() {
  local seed="$1" out="$ROOT/seed${seed}_TGN_3hop/ours"
  skip_or_busy "$out" "train_uci_link.*seed${seed}_TGN_3hop/ours" "ours s$seed" && return 0
  mkdir -p "$out"
  echo "START ours seed=$seed lam=$LAM $(date '+%H:%M:%S') commit=$(git rev-parse --short HEAD)" \
    > "$ROOT/logs/seed${seed}_ours.log"
  $PY -m scripts.train_uci_link \
    --arm 2obs_aligned --lambda-kf "$LAM" \
    --data-dir "$DATA_DIR" --gpu 0 \
    --epochs 30 --budget-cap 30 --patience 8 --kf-group-batches 40 \
    --n-neighbors 10 --n-layers 3 \
    --lr "$LR" --repr-lr "$REPR_LR" \
    --seed "$seed" --output "$out" >> "$ROOT/logs/seed${seed}_ours.log" 2>&1
  echo "DONE_ours_s${seed} rc=$? $(date '+%H:%M:%S')" >> "$ROOT/logs/seed${seed}_ours.log"
}

run_taskonly() {
  local seed="$1" out="$ROOT/seed${seed}_TGN_3hop/taskonly"
  skip_or_busy "$out" "train_uci_link.*seed${seed}_TGN_3hop/taskonly" "taskonly s$seed" && return 0
  mkdir -p "$out"
  echo "START taskonly seed=$seed $(date '+%H:%M:%S') commit=$(git rev-parse --short HEAD)" \
    > "$ROOT/logs/seed${seed}_taskonly.log"
  $PY -m scripts.train_uci_link \
    --config P0 \
    --data-dir "$DATA_DIR" --gpu 0 \
    --epochs 30 --budget-cap 30 --patience 8 --kf-group-batches 40 \
    --n-neighbors 10 --n-layers 3 \
    --lr "$LR" --repr-lr "$REPR_LR" \
    --seed "$seed" --output "$out" >> "$ROOT/logs/seed${seed}_taskonly.log" 2>&1
  echo "DONE_taskonly_s${seed} rc=$? $(date '+%H:%M:%S')" >> "$ROOT/logs/seed${seed}_taskonly.log"
}

run_vanilla() {
  local seed="$1" out="$ROOT/seed${seed}_TGN_3hop/vanilla"
  skip_or_busy "$out" "official_uci_vanilla.*seed ${seed}\b" "vanilla s$seed" && return 0
  mkdir -p "$out/log" "$out/results" "$out/saved_checkpoints" "$out/saved_models"
  local prefix="uci-vanilla-s${seed}"
  echo "START vanilla seed=$seed $(date '+%H:%M:%S') commit=$(git rev-parse --short HEAD)" \
    > "$ROOT/logs/seed${seed}_vanilla.log"
  ( cd "$out" && PYTHONPATH="/root/autodl-tmp:$PYTHONPATH" $PY "$OFFPY" \
      -d uci --prefix "$prefix" --seed "$seed" --data-dir "$DATA_DIR/" \
      --bs 200 --n_degree 10 --n_layer 3 --lr 1e-4 --n_epoch 50 --patience 3 \
      --use_memory --gpu 0 >> "$ROOT/logs/seed${seed}_vanilla.log" 2>&1 )
  local rc=$?
  $PY - "$seed" "$rc" "$out" <<'EOF'
import glob, json, os, pickle, re, sys
seed, rc, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
summary = {"status": "complete" if rc == 0 else "failed", "seed": seed,
           "rc": rc, "protocol": "official-tgn", "n_layer": 3, "n_degree": 10}
pks = sorted(glob.glob(os.path.join(out, "results", "*.pkl")))
pk = pks[0] if pks else ""
if pk and os.path.exists(pk):
    with open(pk, "rb") as f:
        r = pickle.load(f)
    vals = list(r.get("val_aps", []))
    best_ep = int(max(range(len(vals)), key=vals.__getitem__)) if vals else -1
    summary.update({
        "best_epoch": best_ep,
        "best_val_ap": float(vals[best_ep]) if vals else None,
        "test_ap": float(r.get("test_ap")),
        "n_epochs_run": len(vals),
    })
logs = sorted(glob.glob(os.path.join(out, "log", "*-uci-LP-*.log")),
              reverse=True)  # newest first (failed retries leave older logs)
for lg in logs:
    with open(lg) as f:
        txt = f.read()
    m = re.search(r"Test statistics: Transductive: Old  nodes -- auc: ([\d.eE+-]+), ap: ([\d.eE+-]+)", txt)
    if m:
        summary["test_auc"] = float(m.group(1))
        break
json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=2)
EOF
  echo "DONE_vanilla_s${seed} rc=$rc $(date '+%H:%M:%S')" >> "$ROOT/logs/seed${seed}_vanilla.log"
}

for s in 0 1 2 3 4; do
  echo "=== seed $s round $(date '+%H:%M:%S') ===" >> "$ROOT/run.log"
  wait_gpu 8000
  before=$(gpu_used_mb)
  run_ours "$s" & pid1=$!
  confirm_ramp "$pid1" "$before"
  b2=$(gpu_used_mb)
  run_taskonly "$s" & pid2=$!
  confirm_ramp "$pid2" "$b2"
  b3=$(gpu_used_mb)
  run_vanilla "$s" & pid3=$!
  confirm_ramp "$pid3" "$b3"
  wait
  echo "seed $s round done $(date '+%H:%M:%S')" >> "$ROOT/run.log"
done
echo "ALL_DONE $(date '+%H:%M:%S')" >> "$ROOT/run.log"
