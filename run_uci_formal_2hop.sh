#!/bin/bash
# develop_UCI — UCI formal 3-arm x 5 seeds, 2-HOP variant (n_layers=2,
# n_neighbors=10), 3 parallel, repr-lr compensated (same as the 3hop line).
#
# Phase 0: lambda calibration at n_layers=2 (once; separate calib dir from
#   the 3hop calibration — the layer count changes the cut spectrum and
#   therefore r_eff).  The memory gate makes this orchestrator wait for any
#   3hop processes to finish before the calib starts.
# Phase 1: per seed, ours + taskonly + vanilla, parent seed{N}_TGN_2hop.
set -u
cd /root/autodl-tmp/PRSS2_uci_v2 || exit 1
export PYTHONPATH=/root/autodl-tmp/PRSS2_uci_v2:/root/autodl-tmp/benchtemp/experimental_codes/tgn-jodie-dyrep
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/root/miniconda3/bin/python
ROOT=/root/autodl-tmp/PRSS2_uci_v2/outputs/uci_formal_v2
DATA_DIR=/root/autodl-tmp/benchtemp/data_uci
OFFPY=/root/autodl-tmp/PRSS2_uci_v2/scripts/official_uci_vanilla.py
LR=1e-4
REPR_LR=3e-4
NL=2                 # 2-hop: n_layers=2, n_neighbors=10
CALIB=outputs/lambda_calib_2hop/lambda_kyfan.json
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

skip_or_busy() {
  local out="$1" pat="$2" label="$3"
  if [ -f "$out/summary.json" ] && grep -q '"status": "complete"' "$out/summary.json"; then
    echo "SKIP $label" >> "$ROOT/run.log"; return 0
  fi
  if [ -f "$out/summary.json" ] && ! grep -q '"status": "complete"' "$out/summary.json"; then
    rm -f "$out/summary.json"
    echo "RESET $label (failed residue)" >> "$ROOT/run.log"
  fi
  if pgrep -f "$pat" >/dev/null 2>&1; then
    echo "BUSY $label" >> "$ROOT/run.log"; return 0
  fi
  return 1
}

# ---- phase 0: 2hop lambda calibration (once) ----
if [ ! -f "$CALIB" ]; then
  wait_gpu 8000
  echo "CALIB2_START $(date '+%H:%M:%S')" > outputs/lambda_calib_2hop.log
  mkdir -p outputs/lambda_calib_2hop
  $PY -m scripts.train_uci_link \
    --arm 2obs_aligned --data-dir "$DATA_DIR" --gpu 0 \
    --calibrate --calib-groups 8 --kf-group-batches 40 \
    --n-neighbors 10 --n-layers "$NL" --seed 0 \
    --output outputs/lambda_calib_2hop >> outputs/lambda_calib_2hop.log 2>&1
  echo "CALIB2_DONE rc=$? $(date '+%H:%M:%S')" >> outputs/lambda_calib_2hop.log
fi
LAM0=$($PY -c "import json;print(json.load(open('$CALIB'))['lambda0'])")
LAM=$($PY -c "print(0.088 * 0.15 * float('$LAM0'))")
echo "LAMBDA2=$LAM (lambda0=$LAM0, n_layers=$NL) $(date '+%H:%M:%S')" >> "$ROOT/run.log"

run_ours() {
  local seed="$1"
  local out="$ROOT/seed${seed}_TGN_2hop/ours"
  skip_or_busy "$out" "train_uci_link.*seed${seed}_TGN_2hop/ours" "ours2 s$seed" && return 0
  mkdir -p "$out"
  echo "START ours2 seed=$seed lam=$LAM nl=$NL $(date '+%H:%M:%S') commit=$(git rev-parse --short HEAD)" \
    > "$ROOT/logs/seed${seed}_ours2.log"
  $PY -m scripts.train_uci_link \
    --arm 2obs_aligned --lambda-kf "$LAM" \
    --data-dir "$DATA_DIR" --gpu 0 \
    --epochs 30 --budget-cap 30 --patience 8 --kf-group-batches 40 \
    --n-neighbors 10 --n-layers "$NL" \
    --lr "$LR" --repr-lr "$REPR_LR" \
    --seed "$seed" --output "$out" >> "$ROOT/logs/seed${seed}_ours2.log" 2>&1
  echo "DONE_ours2_s${seed} rc=$? $(date '+%H:%M:%S')" >> "$ROOT/logs/seed${seed}_ours2.log"
}

run_taskonly() {
  local seed="$1"
  local out="$ROOT/seed${seed}_TGN_2hop/taskonly"
  skip_or_busy "$out" "train_uci_link.*seed${seed}_TGN_2hop/taskonly" "taskonly2 s$seed" && return 0
  mkdir -p "$out"
  echo "START taskonly2 seed=$seed nl=$NL $(date '+%H:%M:%S') commit=$(git rev-parse --short HEAD)" \
    > "$ROOT/logs/seed${seed}_taskonly2.log"
  $PY -m scripts.train_uci_link \
    --config P0 \
    --data-dir "$DATA_DIR" --gpu 0 \
    --epochs 30 --budget-cap 30 --patience 8 --kf-group-batches 40 \
    --n-neighbors 10 --n-layers "$NL" \
    --lr "$LR" --repr-lr "$REPR_LR" \
    --seed "$seed" --output "$out" >> "$ROOT/logs/seed${seed}_taskonly2.log" 2>&1
  echo "DONE_taskonly2_s${seed} rc=$? $(date '+%H:%M:%S')" >> "$ROOT/logs/seed${seed}_taskonly2.log"
}

run_vanilla() {
  local seed="$1"
  local out="$ROOT/seed${seed}_TGN_2hop/vanilla"
  skip_or_busy "$out" "official_uci_vanilla.*seed ${seed}\b" "vanilla2 s$seed" && return 0
  mkdir -p "$out/log" "$out/results" "$out/saved_checkpoints" "$out/saved_models"
  local prefix="uci-vanilla2-s${seed}"
  echo "START vanilla2 seed=$seed nl=$NL $(date '+%H:%M:%S') commit=$(git rev-parse --short HEAD)" \
    > "$ROOT/logs/seed${seed}_vanilla2.log"
  ( cd "$out" && PYTHONPATH="/root/autodl-tmp:$PYTHONPATH" $PY "$OFFPY" \
      -d uci --prefix "$prefix" --seed "$seed" --data-dir "$DATA_DIR/" \
      --bs 200 --n_degree 10 --n_layer "$NL" --lr 1e-4 --n_epoch 50 --patience 3 \
      --use_memory --gpu 0 >> "$ROOT/logs/seed${seed}_vanilla2.log" 2>&1 )
  local rc=$?
  $PY - "$seed" "$rc" "$out" <<'EOF'
import glob, json, os, pickle, re, sys
seed, rc, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
summary = {"status": "complete" if rc == 0 else "failed", "seed": seed,
           "rc": rc, "protocol": "official-tgn", "n_layer": 2, "n_degree": 10}
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
  echo "DONE_vanilla2_s${seed} rc=$rc $(date '+%H:%M:%S')" >> "$ROOT/logs/seed${seed}_vanilla2.log"
}

for s in 0 1 2 3 4; do
  echo "=== 2hop seed $s round $(date '+%H:%M:%S') ===" >> "$ROOT/run.log"
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
  echo "2hop seed $s round done $(date '+%H:%M:%S')" >> "$ROOT/run.log"
done
echo "ALL_DONE_2HOP $(date '+%H:%M:%S')" >> "$ROOT/run.log"
