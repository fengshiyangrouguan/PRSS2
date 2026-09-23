#!/bin/bash
# Re-run ONLY the predictive arm from the frozen seed-0 root, then the LOCAL
# stages. Same root bytes, same official archive, same budget -- the only change
# is the sibling-diverse allocator on the predictive path.
set -uo pipefail

REPO=/root/autodl-tmp/rpbe-sri/meta-n-main
OUT=/root/autodl-tmp/sri_r3_seed0
LOG=/root/autodl-tmp/sri_ours_rerun.log
cd "$REPO" || exit 1
say() { echo "[$(date '+%F %T')] $*"; }

set -a; . /root/autodl-tmp/meta-n-main/.env; set +a
export LLM_BACKEND=relay ALLOW_PAID_API=YES_I_ACCEPT_REAL_COST
export META_N_EXTRA_HEADERS_JSON='{"Accept-Encoding": "identity"}'
export CODEBERT_PATH=/root/autodl-tmp/models/codebert-base
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HUB_OFFLINE=1
export META_N_REDUCTION_TRACE="$OUT/reduction_trace.jsonl"

CMD=(/root/miniconda3/bin/python -u scripts/run_sri_formal.py
     --profile sri_primary6 --backbone gemini-3.1-pro --search-seed 0
     --out "$OUT"
     --data-dir /root/autodl-tmp/meta-n-main/data/co_bench
     --gamma-checkpoint runs/_gamma_mt_gem_traceonly_clean_phaseb.pt
     --evaluator real)

say "OURS_RERUN_START  (same frozen root; official archive reused)"
say "reduction trace -> $META_N_REDUCTION_TRACE"
rm -f "$META_N_REDUCTION_TRACE"

# PAID: the predictive arm only.
say "=== STAGE predictive ==="
"${CMD[@]}" --stage predictive --execute || { say "predictive FAILED"; exit 1; }
say "stage predictive OK"

# LOCAL from here on: no LLM. freeze re-checks parity against the reused
# official archive; final re-measures BOTH arms' selected candidates so the two
# held-out numbers come from one session; aggregate pools them.
for st in freeze final aggregate; do
  say "=== STAGE $st ==="
  "${CMD[@]}" --stage "$st" --execute || { say "stage $st FAILED"; exit 1; }
  say "stage $st OK"
done
say "OURS_RERUN_DONE"
