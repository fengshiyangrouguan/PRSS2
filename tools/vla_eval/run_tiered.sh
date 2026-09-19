#!/bin/bash
# Tiered eval launcher with the fd-limit fix.
#
# The default `ulimit -n` on this container is 1024, and wandb's temp dirs
# (/tmp/tmp*wandb-artifacts) leak descriptors as the model loads. By demo 20
# the limit is exhausted, robosuite can no longer open its mesh .obj files and
# the demo dies with "resource not found" -- which earlier also manifested as
# an infinite reset-retry spin with zero I/O.
set -u
ulimit -n 65536 || echo "warning: could not raise ulimit"
echo "ulimit -n = $(ulimit -n)"

CKPT="${1:?usage: run_tiered.sh <ckpt> <out-dir> [extra args...]}"
OUT="${2:?usage: run_tiered.sh <ckpt> <out-dir> [extra args...]}"
shift 2

LIBERO_MEM_REPO="${LIBERO_MEM_REPO:-/root/autodl-tmp/libero-mem}"
MEMVLA_DIR="${MEMVLA_DIR:-/root/autodl-tmp/PRSS2/third_party/memoryvla}"
T1EVAL="${T1EVAL:-/root/autodl-tmp/t1_eval}"

cd "$LIBERO_MEM_REPO" || exit 1
export LIBERO_MEM_REPO MEMVLA_DIR
export PYTHONPATH="$MEMVLA_DIR:$LIBERO_MEM_REPO"
export MUJOCO_GL=egl
PY="${RPBE_PYTHON:-/root/autodl-tmp/env_memvla/bin/python}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$PY" "$HERE/tiered_eval_t1.py" \
  --ckpt "$CKPT" --out-dir "$OUT" "$@"
