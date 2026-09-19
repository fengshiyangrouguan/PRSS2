#!/bin/bash
# Verify a checkout is running EXACTLY the code that produced the Stage8 numbers.
#
# The fingerprint is the git tree hash of the three code roots, so it is stable
# across machines, checkout order, line endings and file mtimes: only a real
# content change moves it.
#
# Usage:  bash reproduce/check_code_identity.sh [<ref>]      (default: HEAD)
set -e
REF="${1:-HEAD}"
EXPECTED="634c18f53b59099f5b0c09af2b29c639cc34f94ea5c6031f352beebb98b03926"
PATHS="third_party/memoryvla src/rpbe_embodied scripts/tiered_eval.py"

HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"

GOT=$(git ls-tree -r "$REF" -- $PATHS | sha256sum | cut -d' ' -f1)
N=$(git ls-tree -r "$REF" -- $PATHS | wc -l)

echo "ref        : $REF ($(git rev-parse --short "$REF"))"
echo "code files : $N"
echo "fingerprint: $GOT"
echo "expected   : $EXPECTED"

if [ "$GOT" = "$EXPECTED" ]; then
  echo "RESULT: MATCH -- this is the frozen Stage8 code."
  echo "        (frozen at c05e4fb8b6fd89f0bccd99ed4c3b9ef3448aed36)"
else
  echo "RESULT: MISMATCH -- code differs from what produced the numbers."
  echo "        diff vs c05e4fb:"
  git diff --stat c05e4fb "$REF" -- $PATHS | tail -20
  exit 1
fi
