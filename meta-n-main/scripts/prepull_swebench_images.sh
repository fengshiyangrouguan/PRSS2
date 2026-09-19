#!/usr/bin/env bash
# Pre-pull SWE-bench Verified prebuilt Docker images for a given set of tasks.
#
# Pulls images BEFORE meta-n runs so:
#   (1) LLM wall-clock isn't burned waiting for 1-2 GB image pulls,
#   (2) --parallel > 1 workers don't race on shared image layers,
#   (3) pull failures surface up-front, not 30 min into the run.
#
# The image name is read from each task's environment/Dockerfile FROM line
# (which uses SWE-bench's _1776_ encoding for __ in instance_ids — so we
# never have to encode it ourselves).
#
# Usage:
#   # Pull every image referenced by task dirs under <root>
#   bash scripts/prepull_swebench_images.sh <task-cache-root>
#
#   # Pull only specified task subdirs under <root>
#   bash scripts/prepull_swebench_images.sh <task-cache-root> <task_id> [<task_id> ...]
#
# Examples:
#   bash scripts/prepull_swebench_images.sh ./data/swe_bench_verified
#   bash scripts/prepull_swebench_images.sh ./data/swe_bench_verified \
#     django__django-13741 sympy__sympy-13798
#
set -uo pipefail

if [[ $# -lt 1 ]]; then
  cat >&2 <<EOF
usage: $0 <task-cache-root> [<task_id> ...]

Reads each task's environment/Dockerfile to discover the FROM image, then
pulls it. With no <task_id> args, pulls images for every subdir of the
cache root that contains environment/Dockerfile.
EOF
  exit 2
fi

CACHE_ROOT="$1"
shift
PICKED_IDS=("$@")

if [[ ! -d "$CACHE_ROOT" ]]; then
  echo "ERROR: cache root does not exist: $CACHE_ROOT" >&2
  exit 1
fi

# Pre-flight: Docker daemon
if ! docker info >/dev/null 2>&1; then
  echo "ERROR: Docker daemon not reachable. Start Docker and retry." >&2
  exit 1
fi

# Pre-flight: disk (approximate; macOS Docker Desktop reports its own VM disk)
ROOT_DIR="$(docker info --format '{{.DockerRootDir}}' 2>/dev/null || echo /)"
if [[ -d "$ROOT_DIR" ]]; then
  AVAIL_GB="$(df -k "$ROOT_DIR" 2>/dev/null | awk 'NR==2 {printf "%.0f", $4/1024/1024}')"
else
  AVAIL_GB="$(df -k / | awk 'NR==2 {printf "%.0f", $4/1024/1024}')"
fi

# Collect image names by reading Dockerfile FROM lines
declare -a IMAGES=()
declare -a SKIPPED=()

_image_from_dockerfile() {
  local df_path="$1"
  awk '/^[[:space:]]*FROM[[:space:]]+/ { print $2; exit }' "$df_path"
}

_walk_dirs() {
  if [[ ${#PICKED_IDS[@]} -gt 0 ]]; then
    for tid in "${PICKED_IDS[@]}"; do
      printf '%s\n' "$CACHE_ROOT/$tid"
    done
  else
    # All immediate subdirs of cache root. The -L flag follows symlinks
    # so the SWEBenchVerifiedAdapter staging dir (which contains symlinks
    # to harbor's content-addressed cache, not real subdirectories) is
    # walked correctly. Without -L, `-type d` filters out symlinks → "no
    # images discovered" on the no-picks path.
    find -L "$CACHE_ROOT" -mindepth 1 -maxdepth 1 -type d | sort
  fi
}

while IFS= read -r task_dir; do
  [[ -z "$task_dir" ]] && continue
  df_path="$task_dir/environment/Dockerfile"
  if [[ ! -f "$df_path" ]]; then
    # Maybe it's the harbor org/hash layout: <root>/<task>/<hash>/environment/Dockerfile
    nested=$(find "$task_dir" -mindepth 2 -maxdepth 3 -name Dockerfile -path '*/environment/*' 2>/dev/null | head -1)
    if [[ -n "$nested" ]]; then
      df_path="$nested"
    else
      SKIPPED+=("$(basename "$task_dir") [no Dockerfile]")
      continue
    fi
  fi
  img="$(_image_from_dockerfile "$df_path")"
  if [[ -z "$img" ]]; then
    SKIPPED+=("$(basename "$task_dir") [no FROM line]")
    continue
  fi
  IMAGES+=("$img")
done < <(_walk_dirs)

if [[ ${#IMAGES[@]} -eq 0 ]]; then
  echo "ERROR: no SWE-bench images discovered. Did you run --download first?" >&2
  [[ ${#SKIPPED[@]} -gt 0 ]] && printf '  skipped: %s\n' "${SKIPPED[@]}" >&2
  exit 1
fi

# Deduplicate (sort -u)
UNIQUE_IMAGES=()
while IFS= read -r img; do
  UNIQUE_IMAGES+=("$img")
done < <(printf '%s\n' "${IMAGES[@]}" | sort -u)

NEED_GB=$(( ${#UNIQUE_IMAGES[@]} * 2 ))  # ~2 GB per image worst case
echo "==> ${#UNIQUE_IMAGES[@]} unique image(s) to pull (~${NEED_GB} GB worst case, ${AVAIL_GB} GB free)"
if [[ "$AVAIL_GB" -lt "$NEED_GB" ]]; then
  echo "    WARNING: tight on disk — overlapping layers may keep actual usage lower." >&2
fi
[[ ${#SKIPPED[@]} -gt 0 ]] && printf '    skipped: %s\n' "${SKIPPED[@]}" >&2

FAIL=()
for img in "${UNIQUE_IMAGES[@]}"; do
  echo "==> Pulling $img"
  if ! docker pull "$img"; then
    echo "    FAILED: $img" >&2
    FAIL+=("$img")
  fi
done

echo
if [[ ${#FAIL[@]} -gt 0 ]]; then
  echo "ERROR: failed to pull ${#FAIL[@]} image(s):" >&2
  printf '  - %s\n' "${FAIL[@]}" >&2
  exit 1
fi

echo "OK: pulled ${#UNIQUE_IMAGES[@]} image(s)."
