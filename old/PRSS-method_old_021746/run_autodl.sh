#!/usr/bin/env bash
set -euo pipefail
# Tonight's default first-look experiment is the standard TGN dynamic node-classification task,
# not the near-ceiling Wikipedia random-negative link-prediction task.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/run_node_classification_2gpu.sh" "$@"
