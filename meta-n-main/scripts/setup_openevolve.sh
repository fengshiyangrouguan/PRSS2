#!/bin/bash
# Setup script for OpenEvolve benchmark data.
#
# Downloads the OpenEvolve repository and prepares data for three domains:
#   - AlphaEvolve Math (self-contained, ready to use)
#   - Symbolic Regression (requires running data_api.py to generate problems)
#   - AlgoTune (optional — requires heavy dependencies)
#
# Usage:
#   bash scripts/setup_openevolve.sh [data_dir]
#
# Default data_dir: ./data/openevolve

set -e

DATA_DIR="${1:-./data/openevolve}"
REPO_URL="https://github.com/algorithmicsuperintelligence/openevolve.git"

echo "=== OpenEvolve Benchmark Setup ==="
echo "Data directory: $DATA_DIR"

# --- Clone repo ---
if [ -d "$DATA_DIR" ]; then
    echo "Directory $DATA_DIR already exists. Skipping clone."
else
    echo "Cloning OpenEvolve repository..."
    git clone --depth 1 "$REPO_URL" "$DATA_DIR"
fi

# --- AlphaEvolve Math ---
MATH_DIR="$DATA_DIR/examples/alphaevolve_math_problems"
if [ -d "$MATH_DIR" ]; then
    MATH_COUNT=$(find "$MATH_DIR" -name "evaluator.py" | wc -l | tr -d ' ')
    echo "AlphaEvolve Math: $MATH_COUNT problems ready at $MATH_DIR"
else
    echo "WARNING: AlphaEvolve math problems not found at $MATH_DIR"
fi

# --- Symbolic Regression ---
SR_DIR="$DATA_DIR/examples/symbolic_regression"
if [ -d "$SR_DIR" ]; then
    if [ -d "$SR_DIR/problems" ]; then
        SR_COUNT=$(find "$SR_DIR/problems" -name "evaluator.py" | wc -l | tr -d ' ')
        echo "Symbolic Regression: $SR_COUNT problems already generated at $SR_DIR/problems"
    else
        echo "Generating symbolic regression problems..."
        cd "$SR_DIR"
        pip install -r requirements.txt 2>/dev/null || echo "  (some deps may be missing)"
        python data_api.py 2>/dev/null && echo "  Generated successfully." || \
            echo "  WARNING: data_api.py failed. You may need the bench/ submodule from LLM-SRBench."
        cd - > /dev/null
    fi
else
    echo "WARNING: Symbolic regression directory not found at $SR_DIR"
fi

# --- AlgoTune ---
AT_DIR="$DATA_DIR/examples/algotune"
if [ -d "$AT_DIR" ]; then
    AT_COUNT=$(find "$AT_DIR" -maxdepth 2 -name "evaluator.py" | wc -l | tr -d ' ')
    echo "AlgoTune: $AT_COUNT tasks at $AT_DIR (install deps separately: pip install -r $AT_DIR/requirements.txt)"
else
    echo "WARNING: AlgoTune directory not found at $AT_DIR"
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Usage:"
echo "  python -m meta_n.main --benchmark alphaevolve_math --bench-data-dir $MATH_DIR"
echo "  python -m meta_n.main --benchmark symbolic_regression --bench-data-dir $SR_DIR/problems"
echo "  python -m meta_n.main --benchmark algotune --bench-data-dir $AT_DIR"
