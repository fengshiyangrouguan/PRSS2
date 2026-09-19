#!/bin/bash
# Phase A launcher WITH the budget fuse. Never run a paid batch without it.
#
# Policy (2026-09-17): no batch Phase A until a single pilot's REAL ledger cost
# is known. Defaults below are deliberately tiny.
#
# Usage:
#   scripts/run_phase_a.sh <benchmark> <out_dir> [extra meta-n args...]
#
# Env:
#   PER_RUN_CAP   per-run spend cap in USD   (default 0.25)
#   TOTAL_CAP     whole-Phase-A cap in USD   (default 0.50)
#   PY            python interpreter          (default /root/miniconda3/bin/python)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

PER_RUN_CAP="${PER_RUN_CAP:-0.25}"
TOTAL_CAP="${TOTAL_CAP:-0.50}"
PY="${PY:-/root/miniconda3/bin/python}"

BENCH="${1:?usage: run_phase_a.sh <benchmark> <out_dir> [extra args...]}"
OUT_DIR="${2:?usage: run_phase_a.sh <benchmark> <out_dir> [extra args...]}"
shift 2

set -a; . ./.env; set +a
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# --- 1. refuse to start unless the balance covers the cap plus a buffer ------
"$PY" -m meta_n.rpbe.budget --check-balance --total "$TOTAL_CAP"

# --- 2. baseline the ledger BEFORE any spend --------------------------------
BASELINE="$("$PY" -c "import sys; sys.path.insert(0,'.'); from meta_n.rpbe.budget import total_cost; print('%.6f' % total_cost())")"
echo "[fuse] ledger baseline \$${BASELINE}  per-run cap \$${PER_RUN_CAP}  total cap \$${TOTAL_CAP}"

# --- 3. Meta^n's own daily guard is the first line (it counts per-process) --
export META_N_DAILY_BUDGET_USD="$PER_RUN_CAP"

LOG="$(dirname "$OUT_DIR")/$(basename "$OUT_DIR").log"
mkdir -p "$(dirname "$LOG")"

setsid "$PY" -m meta_n.main --benchmark "$BENCH" --use-archive \
  --output-dir "$OUT_DIR" "$@" > "$LOG" 2>&1 &
RUN_PID=$!
echo "[fuse] meta-n pid $RUN_PID -> $LOG"

# --- 4. external kill switch ------------------------------------------------
"$PY" -m meta_n.rpbe.budget --watch "$RUN_PID" \
  --per-run "$PER_RUN_CAP" --total "$TOTAL_CAP" --baseline "$BASELINE"
FUSE_RC=$?

wait "$RUN_PID" 2>/dev/null || true
echo "[fuse] done (fuse rc=$FUSE_RC). ledger now \$$("$PY" -c "import sys; sys.path.insert(0,'.'); from meta_n.rpbe.budget import total_cost; print('%.4f' % total_cost())")"
echo "[fuse] REAL spend for this run: \$$("$PY" -c "
import sys; sys.path.insert(0,'.')
from meta_n.rpbe.budget import total_cost
print('%.4f' % (total_cost() - float('$BASELINE')))
")"
exit "$FUSE_RC"
