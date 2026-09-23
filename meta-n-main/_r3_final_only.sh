#!/bin/bash
# SUPERSEDED / DISABLED BY DESIGN -- do NOT use for a formal run.
#
# This launcher ran `freeze -> final -> aggregate` with the canonical transition
# audit DELIBERATELY SKIPPED. It produced the seed-0 numbers before the protocol
# dependency was enforced. `stage_final` now REQUIRES the audit stage, so this
# script will fail at the final stage -- which is the intended outcome, and the
# reason it is kept rather than deleted: it is the concrete instance that showed
# "the normal launcher runs audit" was an operator habit, not a guarantee.
#
# Use _ours_rerun.sh (freeze -> audit -> final -> aggregate).
#
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
