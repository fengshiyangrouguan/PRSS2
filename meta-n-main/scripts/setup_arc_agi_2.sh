#!/bin/bash
# Setup script for ARC-AGI-2 benchmark data.
#
# Clones the official arcprize/ARC-AGI-2 repo (Apache-2.0) and validates
# task counts. The repo layout is:
#   data/training/  -> 1000 task JSON files (one per task)
#   data/evaluation/ -> 120 task JSON files (one per task)
#
# Each JSON has {"train": [{input, output}, ...], "test": [{input, output}, ...]}.
#
# Usage:
#   bash scripts/setup_arc_agi_2.sh [data_dir]
#
# Default data_dir: ./data/arc_agi_2
#
# Idempotent: re-running pulls latest if a clone exists.

set -e

DATA_DIR="${1:-./data/arc_agi_2}"
REPO_URL="https://github.com/arcprize/ARC-AGI-2.git"

echo "=== ARC-AGI-2 Benchmark Setup ==="
echo "Data directory: $DATA_DIR"

if [ -d "$DATA_DIR/.git" ]; then
    echo "Existing clone found — fetching updates..."
    git -C "$DATA_DIR" pull --ff-only
else
    if [ -d "$DATA_DIR" ]; then
        echo "ERROR: $DATA_DIR exists but is not a git checkout. Move it aside or pass a different path."
        exit 1
    fi
    echo "Cloning ARC-AGI-2 repository..."
    git clone --depth 1 "$REPO_URL" "$DATA_DIR"
fi

# --- Validate counts ---
TRAIN_COUNT=$(ls "$DATA_DIR/data/training" 2>/dev/null | wc -l | tr -d ' ')
EVAL_COUNT=$(ls "$DATA_DIR/data/evaluation" 2>/dev/null | wc -l | tr -d ' ')

echo "Training tasks: $TRAIN_COUNT (expected 1000)"
echo "Evaluation tasks: $EVAL_COUNT (expected 120)"

if [ "$TRAIN_COUNT" -ne 1000 ] || [ "$EVAL_COUNT" -ne 120 ]; then
    echo "ERROR: task counts do not match expected (1000 train, 120 eval)."
    echo "       The upstream repo may have changed shape — verify https://github.com/arcprize/ARC-AGI-2"
    exit 1
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Usage:"
echo "  python -m meta_n.main --benchmark arc_agi_2 --bench-limit 5 --use-archive ..."
echo ""
echo "Pilot guidance: at 20 evolve iters × 120 tasks, full sweeps cost ~\$30+ per baseline run."
echo "For pilot/smoke runs, restrict to a subset:"
echo "  --bench-limit 5                                    (Meta^n)"
echo "  META_N_ARC_AGI_2_TASK_FILTER='id1|id2|id3|...'    (Gödel + OpenEvolve baselines)"
