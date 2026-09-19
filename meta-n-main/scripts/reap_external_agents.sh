#!/bin/sh
# ============================================================================
# reap_external_agents.sh — crash-leak reaper for the external-agents path
# (plan docs/external_agents_integration/06-concurrency-resource-model.md §6.2.1)
#
# The in-process DockerRunGuard.lease() finally + shutdown_sweep() only cover
# *graceful* teardown. On a `kill -9` (no finally fires) the run can leak four
# classes of resource that all carry the per-lease compose-project prefix
# `ext-{task_id}-{uuid}` (== EnvLease.session, parity terminal_bench.py:764):
#
#   1. Docker containers   (name=ext-…-main-1, …)
#   2. Docker networks      (Compose-prefixed with the project name)
#   3. Docker volumes       (Compose-prefixed with the project name)
#   4. Host scratch dirs    (mkdtemp prefix="ext_agent_", parity :765)
#
# This script is the out-of-band `kill -9` backstop, NOT the primary path. It
# force-kills/removes any leftover `ext-*` Docker objects and removes orphaned
# `ext_agent_*` scratch dirs under the scratch root. It is fully idempotent and
# safe to run at any time (including while live runs are in flight — see the
# scratch-dir mtime guard below, which spares freshly created dirs). It prints
# exactly what it reaps and exits 0 even when there is nothing to do.
#
# Usage:
#   scripts/reap_external_agents.sh [--scratch-root DIR] [--dry-run] [-h|--help]
#
# Environment:
#   SCRATCH_ROOT   Root dir to scan for orphaned ext_agent_* scratch dirs.
#                  Falls back to $TMPDIR, then /tmp. Overridden by --scratch-root.
#   DRY_RUN        If set to a non-empty value, print actions without executing
#                  (same as passing --dry-run).
#
# Exit status:
#   0  Always on a normal (idempotent) run, even if nothing was reaped or the
#      Docker daemon is unreachable. Non-zero only on usage errors.
# ============================================================================

# POSIX sh: -e would abort on the first `xargs`/`docker rm` that hits an
# already-gone object during a concurrent reap, which defeats idempotency.
# We instead guard every step individually, so only -u is enabled.
set -u

PROG=$(basename "$0")

# ----------------------------------------------------------------------------
# Argument parsing.
# ----------------------------------------------------------------------------
SCRATCH_ROOT_ARG=""
DRY_RUN="${DRY_RUN:-}"

usage() {
    cat <<EOF
$PROG — reap leaked external-agent Docker objects and scratch dirs.

Usage:
  $PROG [--scratch-root DIR] [--dry-run] [-h|--help]

Options:
  --scratch-root DIR   Root to scan for orphaned ext_agent_* dirs
                       (default: \$SCRATCH_ROOT, then \$TMPDIR, then /tmp).
  --dry-run            Print what would be reaped without removing anything.
  -h, --help           Show this help and exit.

Reaps (all idempotent, all matched by the per-lease 'ext-' compose prefix):
  * docker containers   --filter name=ext-   (kill -9 + rm)
  * docker networks      --filter name=ext-   (rm)
  * docker volumes       --filter name=ext-   (rm)
  * host scratch dirs    ext_agent_*  older than 0 days under the scratch root
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --scratch-root)
            if [ $# -lt 2 ]; then
                printf '%s: --scratch-root requires a directory argument\n' "$PROG" >&2
                exit 2
            fi
            SCRATCH_ROOT_ARG="$2"
            shift 2
            ;;
        --scratch-root=*)
            SCRATCH_ROOT_ARG="${1#--scratch-root=}"
            shift
            ;;
        --dry-run)
            DRY_RUN="1"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            printf '%s: unknown argument: %s\n' "$PROG" "$1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

# ----------------------------------------------------------------------------
# Resolve the scratch root (flag > env > $TMPDIR > /tmp).
# ----------------------------------------------------------------------------
if [ -n "$SCRATCH_ROOT_ARG" ]; then
    SCRATCH_ROOT="$SCRATCH_ROOT_ARG"
elif [ -n "${SCRATCH_ROOT:-}" ]; then
    SCRATCH_ROOT="$SCRATCH_ROOT"
elif [ -n "${TMPDIR:-}" ]; then
    SCRATCH_ROOT="$TMPDIR"
else
    SCRATCH_ROOT="/tmp"
fi

# Counters for the final summary line.
N_CONTAINERS=0
N_NETWORKS=0
N_VOLUMES=0
N_DIRS=0

log() {
    printf '%s\n' "$*"
}

# Echo whether we are in dry-run mode, prefixed so output is unambiguous.
if [ -n "$DRY_RUN" ]; then
    log "[$PROG] DRY RUN — no objects will be removed."
fi
log "[$PROG] scratch root: $SCRATCH_ROOT"

# ----------------------------------------------------------------------------
# Docker reaping. Only attempt if a docker CLI exists and the daemon answers;
# otherwise skip cleanly (the scratch-dir sweep still runs).
# ----------------------------------------------------------------------------
docker_available() {
    command -v docker >/dev/null 2>&1 || return 1
    docker info >/dev/null 2>&1 || return 1
    return 0
}

# reap_docker_objects KIND LIST_CMD REMOVE_CMD
#   KIND        human label (container|network|volume)
#   LIST_CMD    command that prints one object id per line (filtered to ext-)
#   REMOVE_CMD  command run with the ids appended (e.g. "docker rm -f")
# Sets the matching N_* counter via the global named in $4.
reap_docker_objects() {
    kind="$1"
    list_cmd="$2"
    remove_cmd="$3"
    counter_var="$4"

    # Collect ids (newline-separated). `|| true` keeps a transient docker error
    # from aborting the whole reaper.
    ids=$(eval "$list_cmd" 2>/dev/null || true)

    if [ -z "$ids" ]; then
        log "[$PROG] no leaked ext-* ${kind}s"
        return 0
    fi

    count=$(printf '%s\n' "$ids" | grep -c .)
    log "[$PROG] reaping $count leaked ext-* ${kind}(s):"
    # Print each id we are about to act on (so the user sees what was reaped).
    printf '%s\n' "$ids" | sed "s/^/[$PROG]   ${kind}: /"

    if [ -z "$DRY_RUN" ]; then
        # xargs -r: do nothing on empty input. Removal failures (object already
        # gone in a concurrent reap) are non-fatal by design.
        printf '%s\n' "$ids" | xargs $remove_cmd >/dev/null 2>&1 || true
    fi

    # Assign the counter through the indirection variable name.
    eval "$counter_var=$count"
    return 0
}

if docker_available; then
    # 1. Containers — force kill (-9) AND remove. `docker rm -f` SIGKILLs a
    #    running container then removes it, satisfying the kill -9 requirement.
    reap_docker_objects \
        "container" \
        "docker ps -aq --filter name=ext-" \
        "docker rm -f" \
        N_CONTAINERS

    # 2. Networks — remove (must come after containers are gone so the network
    #    has no active endpoints).
    reap_docker_objects \
        "network" \
        "docker network ls -q --filter name=ext-" \
        "docker network rm" \
        N_NETWORKS

    # 3. Volumes — remove (containers using them are already force-removed).
    reap_docker_objects \
        "volume" \
        "docker volume ls -q --filter name=ext-" \
        "docker volume rm" \
        N_VOLUMES
else
    log "[$PROG] docker CLI/daemon unavailable — skipping container/network/volume reap"
fi

# ----------------------------------------------------------------------------
# Host scratch-dir reaping. Orphaned mkdtemp(prefix="ext_agent_") dirs left by a
# kill -9 that prevented the lease `finally` (rmtree) from firing.
#
# The `-mtime +0` guard (modified more than 24h ago — i.e. older than 1 day)
# spares scratch dirs from currently in-flight runs, so this is safe to run
# while other runs are active. Use --scratch-root / SCRATCH_ROOT for a custom
# scratch location.
# ----------------------------------------------------------------------------
if [ -d "$SCRATCH_ROOT" ]; then
    # Gather orphaned dirs first so we can report and count them, then remove.
    # -maxdepth 1: only top-level scratch dirs (mkdtemp does not nest).
    orphans=$(find "$SCRATCH_ROOT" -maxdepth 1 -type d -name 'ext_agent_*' -mtime +0 2>/dev/null || true)

    if [ -z "$orphans" ]; then
        log "[$PROG] no orphaned ext_agent_* scratch dirs under $SCRATCH_ROOT"
    else
        N_DIRS=$(printf '%s\n' "$orphans" | grep -c .)
        log "[$PROG] reaping $N_DIRS orphaned ext_agent_* scratch dir(s):"
        printf '%s\n' "$orphans" | sed "s/^/[$PROG]   dir: /"
        if [ -z "$DRY_RUN" ]; then
            # rm -rf per dir; ignore individual failures to stay idempotent.
            printf '%s\n' "$orphans" | while IFS= read -r d; do
                [ -n "$d" ] || continue
                rm -rf -- "$d" 2>/dev/null || true
            done
        fi
    fi
else
    log "[$PROG] scratch root $SCRATCH_ROOT does not exist — skipping scratch reap"
fi

# ----------------------------------------------------------------------------
# Summary.
# ----------------------------------------------------------------------------
log "[$PROG] done — reaped: ${N_CONTAINERS} container(s), ${N_NETWORKS} network(s), ${N_VOLUMES} volume(s), ${N_DIRS} scratch dir(s)"

exit 0
