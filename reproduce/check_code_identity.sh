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

# ---- deployed-file level check -------------------------------------------
# The 5 files that were actually on the training box, md5'd there and compared
# against local HEAD c05e4fb during the run (server had: 38751ce7 / 5f50ad9c /
# f826bcf1 / 17fed5d0 / c05e25f2 -- and they matched).  Those recorded values
# are the CRLF form, because the box got the Windows working-tree copy over
# sftp.  To be independent of line endings this check strips CR before hashing,
# so it gives the canonical LF values below and works on either platform.
echo
echo "deployed-file check (CR-stripped, so line endings do not matter):"
declare -A WANT=(
  [src/rpbe_embodied/loss.py]=9a8cdde57a7f0d7d1353bada37683e8c
  [src/rpbe_embodied/boundary.py]=5f50ad9cf0d5a030641360d3ee76c011
  [src/rpbe_embodied/resume.py]=f826bcf1566d66b9f2b4c13cdcb9c8c5
  [src/rpbe_embodied/test_projection.py]=45760bbd5f167da28c0b608782cf3134
  [third_party/memoryvla/train_libero_mem_rpbe.py]=c05e25f20f4dddded7aac1acec0539b7
)
bad=0
for f in "${!WANT[@]}"; do
  got=$(git show "c05e4fb:$f" | tr -d '' | md5sum | cut -d' ' -f1)
  if [ "$got" = "${WANT[$f]}" ]; then
    echo "  ok   $f"
  else
    echo "  BAD  $f  got=$got want=${WANT[$f]}"
    bad=1
  fi
done
[ "$bad" = 0 ] && echo "RESULT: all 5 deployed files verified against the frozen record."   || { echo "RESULT: deployed-file MISMATCH"; exit 1; }
