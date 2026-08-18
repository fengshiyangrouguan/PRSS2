#!/usr/bin/env bash
set -euo pipefail
DATASET="${1:-reddit}"
OUTPUT="${2:-${DATASET}.csv}"
case "${DATASET}" in
  wikipedia|reddit|mooc|lastfm) ;;
  *) echo "Supported JODIE datasets: wikipedia reddit mooc lastfm" >&2; exit 2 ;;
esac
URL="https://snap.stanford.edu/jodie/${DATASET}.csv"
echo "Downloading ${URL} -> ${OUTPUT}"
mkdir -p "$(dirname "${OUTPUT}")"
if command -v curl >/dev/null 2>&1; then
  curl -fL --retry 5 --retry-delay 3 "${URL}" -o "${OUTPUT}"
elif command -v wget >/dev/null 2>&1; then
  wget -O "${OUTPUT}" "${URL}"
else
  echo "Need curl or wget" >&2
  exit 2
fi
