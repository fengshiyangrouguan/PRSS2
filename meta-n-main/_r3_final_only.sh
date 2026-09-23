#!/bin/bash
# final -> aggregate for seed 0, alone (0 concurrency with the audit: a CO-Bench
# score counts instances finishing inside a 10s timeout, so a second heavy job
# would move the number we are measuring).
#
# `stage_final` depends on FREEZE only -- verified in the code
# (`frz = _assert_frozen(out)`, inputs={"freeze": ...}), never on the audit. So
# running final without the audit is a REORDERING, not a missing input.
set -uo pipefail

REPO=/root/autodl-tmp/rpbe-sri/meta-n-main
OUT=/root/autodl-tmp/sri_r3_seed0
cd "$REPO" || exit 1
say() { echo "[$(date '+%F %T')] $*"; }

set -a; . /root/autodl-tmp/meta-n-main/.env; set +a
export LLM_BACKEND=relay ALLOW_PAID_API=YES_I_ACCEPT_REAL_COST
export META_N_EXTRA_HEADERS_JSON='{"Accept-Encoding": "identity"}'
export CODEBERT_PATH=/root/autodl-tmp/models/codebert-base
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HUB_OFFLINE=1

CMD=(/root/miniconda3/bin/python -u scripts/run_sri_formal.py
     --profile sri_primary6 --backbone gemini-3.1-pro --search-seed 0
     --out "$OUT"
     --data-dir /root/autodl-tmp/meta-n-main/data/co_bench
     --gamma-checkpoint runs/_gamma_mt_gem_traceonly_clean_phaseb.pt
     --evaluator real)

say "SEED0_FINAL_START (audit deliberately skipped; stage order changed, not inputs)"
for st in final aggregate; do
  say "=== STAGE $st ==="
  if ! "${CMD[@]}" --stage "$st" --execute; then
    say "STAGE $st FAILED"
    exit 1
  fi
  say "stage $st OK"
done
say "FINAL_AGGREGATE_DONE"
