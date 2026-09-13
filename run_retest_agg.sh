#!/bin/bash
# Dual-aggregation retest orchestrator: waits for ALL training processes to
# exit (the 32GB card has no headroom while a treewise run is up), then runs
# the full-sweep retest (idempotent: existing retest_agg.json entries are
# skipped on re-runs).
set -u
cd /root/autodl-tmp/PRSS2_uci_v2 || exit 1
export PYTHONPATH=/root/autodl-tmp/PRSS2_uci_v2:/root/autodl-tmp/benchtemp/experimental_codes/tgn-jodie-dyrep
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/root/miniconda3/bin/python
ROOT=/root/autodl-tmp/PRSS2_uci_v2/outputs/uci_formal_v2
DATA_DIR=/root/autodl-tmp/benchtemp/data_uci
mkdir -p "$ROOT/logs"

while pgrep -f "[t]rain_uci_link" >/dev/null 2>&1; do
  echo "WAIT_TRAIN $(date '+%H:%M:%S')" >> "$ROOT/retest_agg.log"
  sleep 300
done
echo "RETEST_START $(date '+%H:%M:%S')" >> "$ROOT/retest_agg.log"
$PY scripts/retest_uci_agg.py \
  --root "$ROOT" --data-dir "$DATA_DIR" --gpu 0 \
  >> "$ROOT/retest_agg.log" 2>&1
echo "RETEST_DONE rc=$? $(date '+%H:%M:%S')" >> "$ROOT/retest_agg.log"
