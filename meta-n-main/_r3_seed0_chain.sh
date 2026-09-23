#!/bin/bash
# seed-0 formal pair, unattended, straight through the chain.
#
# root -> fork -> official -> predictive -> freeze -> audit -> final -> aggregate
#
# The three launch facts this run depends on, checked BY READING THE PLAN, not
# assumed: backbone=gemini-3.1-pro, the Gamma file SHA, and a fresh --out that is
# not sri_matched_v2. The runner's own preflight re-checks the Gamma identity
# against the FORK stage before the first paid call, so a swap cannot slip in.
#
# Re-running this script RESUMES: `--stage all --execute` skips any stage whose
# manifest already exists, so the retry loop continues instead of restarting.
set -uo pipefail

REPO=/root/autodl-tmp/rpbe-sri/meta-n-main
OUT=/root/autodl-tmp/sri_r3_seed0
LOG=/root/autodl-tmp/sri_r3_seed0.log
cd "$REPO" || exit 1

say() { echo "[$(date '+%F %T')] $*"; }

set -a; . /root/autodl-tmp/meta-n-main/.env; set +a
export LLM_BACKEND=relay ALLOW_PAID_API=YES_I_ACCEPT_REAL_COST
export META_N_EXTRA_HEADERS_JSON='{"Accept-Encoding": "identity"}'
export CODEBERT_PATH=/root/autodl-tmp/models/codebert-base
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HUB_OFFLINE=1
# The request ledger is still NOT set here: the runner derives it per arm from
# the sri context, and a global override would sit on top of the accounting the
# ledger audit reads.
#
# META_N_REDUCTION_TRACE IS now set. Without it the 4-of-N choice is persisted
# nowhere -- `proposal_slots.jsonl` carries slot_id/parent_id/depth but no trace
# ids -- so the seed-0 analysis could not see any selection below depth 2 at all.
# The reducer writes one JSONL row per reduction: pool identity + hash, Gamma's
# raw scores and rank, the allocator variant, and the chosen trace ids, with a
# monotonic reduction_seq that joins back to proposal_slots.jsonl by order.
export META_N_REDUCTION_TRACE="$OUT/reduction_trace.jsonl"

CMD=(/root/miniconda3/bin/python -u scripts/run_sri_formal.py
     --profile sri_primary6 --backbone gemini-3.1-pro --search-seed 0
     --out "$OUT"
     --data-dir /root/autodl-tmp/meta-n-main/data/co_bench
     --gamma-checkpoint runs/_gamma_mt_gem_traceonly_clean_phaseb.pt
     --evaluator real --stage all --execute)

say "SEED0_CHAIN_START out=$OUT"
say "gamma sha256 = $(sha256sum runs/_gamma_mt_gem_traceonly_clean_phaseb.pt | cut -d' ' -f1)"
say "backbone     = gemini-3.1-pro   backend=$LLM_BACKEND"

for attempt in 1 2 3; do
  say "=== ATTEMPT $attempt: stage all ==="
  if "${CMD[@]}"; then
    say "ALL_DONE"
    exit 0
  fi
  say "attempt $attempt failed (rc=$?) -- resuming from the last good stage"
  sleep 30
done

say "CHAIN_GAVE_UP after 3 attempts"
exit 1
