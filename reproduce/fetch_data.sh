#!/bin/bash
# Download everything this experiment needs. NOTHING large is in the git repo.
#
# The HF mirror is used by default because direct HF access is usually blocked
# on these boxes; override HF_ENDPOINT if you have direct access.
#
# Usage:  ROOT=/root/autodl-tmp bash reproduce/fetch_data.sh
set -euo pipefail
: "${ROOT:=/root/autodl-tmp}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

DS="$ROOT/datasets/LIBERO-Mem"          # raw task data
T3="KITCHEN_SCENE1_3_lift_the_bowl_and_place_it_back_on_the_plate_3_times_demo.hdf5"

echo "### 1/5  T3 task data  (the '3 times' task -- 18.4 GB + metainfo 0.7 GB)"
mkdir -p "$DS"
#   dataset repo: huggingface.co/datasets/libero-mem/LIBERO-Mem
for f in "$T3" metainfo.json; do
  [ -s "$DS/$f" ] && { echo "  have $f"; continue; }
  echo "  fetching $f"
  curl -L -C - --retry 5 -o "$DS/$f" \
    "$HF_ENDPOINT/datasets/libero-mem/LIBERO-Mem/resolve/main/$f"
done
echo "  metainfo.json is MANDATORY: the loader reads task_description from it and"
echo "  raises ValueError: task ... missing from metainfo.json without it."

echo "### 2/5  no-op-filtered dataset"
echo "  The trainer's --data-root expects <ROOT>/libero-mem-no-noops, a build of"
echo "  the file above with no-op transitions removed (~9% of frames).  That"
echo "  filter script was never committed, so you must re-derive it."
echo "  IMPORTANT: for a multi-seed comparison the filter only has to be FIXED"
echo "  across your seeds -- it does not have to match ours."
echo "  Starting point: scripts/probe_libero_mem_hdf5.py"
echo "  If you skip it, point --data-root at the raw directory instead, but then"
echo "  list it explicitly in the write-up: it is a different dataset build."

echo "### 3/5  base VLA checkpoint  (30 GB)"
echo "  repo: huggingface.co/openvla/openvla-7b-prismatic"
echo "  file: checkpoints/step-295000-epoch-40-loss=0.2200.pt"
python -m pip install -q -U huggingface_hub || true
python - <<PY
from huggingface_hub import snapshot_download
snapshot_download(repo_id="openvla/openvla-7b-prismatic",
                  local_dir="$ROOT/openvla-7b-prismatic")
print("OPENVLA_DONE")
PY

echo "### 4/5  Llama-2-7b-hf  (13.5 GB, the LLM backbone)"
python - <<PY
from huggingface_hub import snapshot_download
snapshot_download(repo_id="NousResearch/Llama-2-7b-hf",
                  local_dir="$ROOT/Llama-2-7b-hf")
print("LLAMA2_DONE")
PY

echo "### 5/5  the LIBERO-Mem simulator repo (the env used by tiered_eval.py)"
[ -d "$ROOT/libero-mem" ] || git clone --depth 1 \
  https://github.com/libero-mem/libero-mem.git "$ROOT/libero-mem"

cat <<'NOTE'

Disk needed: openvla 30G + Llama-2 13.5G + T3 data 18.4G + metainfo 0.7G
             + checkpoints ~9.3G per run  ->  budget 200 GB+.

The 3-cycle task is KITCHEN_SCENE1_3 -- that is what "T3" means here, and it
is the task the Stage8 numbers were measured on.
NOTE
