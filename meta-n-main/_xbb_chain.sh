#!/bin/bash
# Cross-backbone diagnostic: one leg per backbone. Usage: _xbb_chain.sh <model>
#
# WHAT THIS IS. The frozen SRI experiment is a shared-root paired comparison:
# one root, forked into an `official` arm (native 4-trace rule) and a
# `predictive` arm (the FROZEN Gemini-trained Gamma picks the 4 traces). The
# headline is the d1->d2 Shared-root Selection Gain and its gap. This runs that
# same frozen protocol on a DIFFERENT backbone.
#
# The root is regenerated, and that is fine -- each leg compares PAIRED WITHIN
# itself (both arms inherit this leg's root). What must NOT be done is comparing
# the absolute numbers against the Gemini leg's 0.9547 / +0.1094: the roots
# differ, so only the SIGN and DIRECTION of the gap are comparable.
#
# THE CODE COMES FROM A WORKTREE AT 5d2efb3, deliberately. HEAD carries a
# SiblingAllocator that hands the siblings of one parent DIFFERENT trace
# subsets; the frozen design requires every sibling of a parent to receive the
# SAME 4 traces. The allocator is a module-level singleton with no env switch,
# so it cannot be disabled without editing -- hence the worktree, which also
# leaves the current HEAD and its 5 allocator commits untouched.
#
# EACH LEG GETS ITS OWN WORKTREE. Two legs sharing one would fight over
# `__pycache__` and the runner's scratch under the same cwd; the code is
# read-only during a run, so separate checkouts of the same commit cost nothing
# but a directory.
#
# STOPS AT `predictive`. freeze/audit/final are the full pipeline and are not
# needed for the selection gain. Each stage retries 3x because this relay
# throws 502/503 in bursts.
set -uo pipefail

MODEL="${1:?usage: _xbb_chain.sh <model>}"
SAFE="$(echo "$MODEL" | tr './' '__')"
REPO=/root/autodl-tmp/rpbe-sri
WT="$REPO/../wt_$SAFE"
OUT="/root/autodl-tmp/sri_xbb_$SAFE"
LOG="/root/autodl-tmp/sri_xbb_$SAFE.log"
say() { echo "[$(date '+%F %T')] $*"; }

cd "$REPO" || exit 1
if [ ! -d "$WT" ]; then
  git worktree add --detach "$WT" 5d2efb3 >/dev/null 2>&1 || { say "WORKTREE_ADD_FAILED"; exit 1; }
fi
cd "$WT/meta-n-main" || exit 1

set -a; . /root/autodl-tmp/meta-n-main/.env; set +a
export LLM_BACKEND=relay ALLOW_PAID_API=YES_I_ACCEPT_REAL_COST
export META_N_EXTRA_HEADERS_JSON='{"Accept-Encoding": "identity"}'
export CODEBERT_PATH=/root/autodl-tmp/models/codebert-base
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HUB_OFFLINE=1
export META_N_REDUCTION_TRACE="$OUT/reduction_trace.jsonl"
mkdir -p "$OUT"

# The Gamma is NOT tracked by git, so it does not exist inside the worktree --
# point at the real file by absolute path and record its hash, because a
# worktree checkout is exactly where a silent "no checkpoint" would hide.
GAMMA=/root/autodl-tmp/rpbe-sri/meta-n-main/runs/_gamma_mt_gem_traceonly_clean_phaseb.pt

CMD=(/root/miniconda3/bin/python -u scripts/run_sri_formal.py
     --profile sri_primary6 --backbone "$MODEL" --search-seed 0
     --out "$OUT"
     --data-dir /root/autodl-tmp/meta-n-main/data/co_bench
     --gamma-checkpoint "$GAMMA"
     --evaluator real)

say "XBB_START model=$MODEL out=$OUT"
say "code HEAD     = $(git rev-parse --short HEAD)"
say "allocator     = $(grep -c SiblingAllocator meta_n/rpbe/context_reduction.py) occurrences (must be 0)"
say "gamma sha256  = $(sha256sum "$GAMMA" | cut -d' ' -f1)"

for st in preflight root fork official predictive; do
  for attempt in 1 2 3; do
    say "=== stage $st (attempt $attempt) ==="
    "${CMD[@]}" --stage "$st" --execute
    rc=$?
    if [ "$rc" -eq 0 ]; then say "stage $st OK"; break; fi
    say "stage $st FAILED rc=$rc"
    if [ "$attempt" -eq 3 ]; then say "GIVING_UP stage=$st"; exit 1; fi
    sleep 60
  done
done

say "XBB_DONE model=$MODEL -- now run the selection-gain metric on $OUT"
