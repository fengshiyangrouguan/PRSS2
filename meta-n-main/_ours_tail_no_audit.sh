#!/bin/bash
# Tail stages WITHOUT the audit: freeze -> final -> aggregate. 0 API, 0 LLM.
#
# WHY THERE IS NO AUDIT HERE. `stage_final` used to require it (added in
# 7ee939d); that requirement was reverted at the operator's request because the
# audit is local-but-slow (every parent/child re-evaluated on 6 tasks x 3
# repeats -- hours of wall clock) while its metric (R_2->3) is not in the paper
# and the held-out number does not depend on it (`final` reads the test split,
# which the audit never touches).
#
# CONSEQUENCE, STATED PLAINLY: the run this produces is NOT protocol-complete.
# Say so in the provenance. Nothing here is a substitute for the audit.
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
export META_N_REDUCTION_TRACE="$OUT/reduction_trace.jsonl"   # never deleted: it is the WAL

CMD=(/root/miniconda3/bin/python -u scripts/run_sri_formal.py
     --profile sri_primary6 --backbone gemini-3.1-pro --search-seed 0
     --out "$OUT"
     --data-dir /root/autodl-tmp/meta-n-main/data/co_bench
     --gamma-checkpoint runs/_gamma_mt_gem_traceonly_clean_phaseb.pt
     --evaluator real)

say "TAIL_START (no audit; NOT protocol-complete)"
for st in freeze final aggregate; do
  say "=== STAGE $st ==="
  "${CMD[@]}" --stage "$st" --execute || { say "stage $st FAILED"; exit 1; }
  say "stage $st OK"
done
say "TAIL_DONE"
