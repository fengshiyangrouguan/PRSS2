#!/bin/bash
# Package the small (non-checkpoint) experiment result files into results/
# so they can be version-controlled (outputs/ is gitignored, *.pt excluded).
set -u
cd /root/autodl-tmp/PRSS2_uci_v2 || exit 1
rm -rf results
mkdir -p results

find outputs -type f \( \
    -name val_history.jsonl -o \
    -name metrics.jsonl -o \
    -name summary.json -o \
    -name config.json -o \
    -name window_diag.jsonl -o \
    -name comparison_audit.json \
  \) \
  ! -path '*/uci_formal/*' \
  ! -path '*/constrain_smoke/*' \
  ! -path '*/grad_align*' \
  ! -path '*/lambda_calib*' \
  ! -path '*/window_spectrum_diag/*' \
  | while read -r f; do
      # strip the leading "outputs/" component: the .gitignore rule
      # "outputs/" matches at ANY depth, so results/outputs/... would still
      # be ignored.
      rel="${f#outputs/}"
      mkdir -p "results/$(dirname "$rel")"
      cp "$f" "results/$rel"
    done

echo "files: $(find results -type f | wc -l)"
du -sh results
find results -type f -name '*.json*' -printf '%s\n' | awk '{s+=$1} END {printf "%.2f MB of json\n", s/1048576}'
