#!/bin/bash
# Re-run ONLY the predictive arm from the frozen seed-0 root, then the LOCAL
# stages. Same root bytes, same official archive, same budget -- the only change
# is the sibling-diverse allocator on the predictive path.
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
# The allocator REPLAYS this file on start to keep the sibling variant sequence
# intact across --resume, so it is a write-ahead log, not scratch output. It is
# therefore NEVER deleted here: deleting it on a resumed run would silently
# restart every sibling group at variant 0. A genuinely fresh start is the one
# case where it must be cleared, and that is done by hand, not by this script.
export META_N_REDUCTION_TRACE="$OUT/reduction_trace.jsonl"

CMD=(/root/miniconda3/bin/python -u scripts/run_sri_formal.py
     --profile sri_primary6 --backbone gemini-3.1-pro --search-seed 0
     --out "$OUT"
     --data-dir /root/autodl-tmp/meta-n-main/data/co_bench
     --gamma-checkpoint runs/_gamma_mt_gem_traceonly_clean_phaseb.pt
     --evaluator real)

say "OURS_RERUN_START  (same frozen root; official archive reused)"
say "reduction trace (WAL, kept across resume) -> $META_N_REDUCTION_TRACE"
say "rows already in the trace: $(wc -l < "$META_N_REDUCTION_TRACE" 2>/dev/null || echo 0)"

# BEFORE ANY PAID ACTION. The rerun setup prunes the stage manifest to
# preflight/root/fork, which also removes the `official` record -- and
# `stage_freeze` refuses when an arm stage is missing, so the predictive arm
# would be PAID FOR and then fail. Depending on an operator to remember this is
# not acceptable on a chain that spends real money, so the rebuild (which
# validates the official artifacts first and refuses on any inconsistency) runs
# here, unconditionally, every time.
say "=== preflight: validate + rebuild the official stage record (0 API) ==="
/root/miniconda3/bin/python _rebuild_official_record.py \
  || { say "official record rebuild VALIDATION FAILED -- refusing to spend"; exit 1; }
say "official record ready"

# PAID: the predictive arm only. `official` is NOT re-run -- the allocator never
# touched its path, and its stage record was rebuilt from the on-disk artifacts.
say "=== STAGE predictive ==="
"${CMD[@]}" --stage predictive --execute || { say "predictive FAILED"; exit 1; }
say "stage predictive OK"

# LOCAL from here on (no LLM). audit is part of the frozen protocol and is the
# independent re-evaluation gate; it was missing from an earlier version of this
# script, which would have sent a new archive straight to final.
for st in freeze audit final aggregate; do
  say "=== STAGE $st ==="
  "${CMD[@]}" --stage "$st" --execute || { say "stage $st FAILED"; exit 1; }
  say "stage $st OK"
done
say "OURS_RERUN_DONE"
