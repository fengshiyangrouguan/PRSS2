#!/usr/bin/env bash
# ============================================================================
# install_external_agents.sh — one persisted venv + committed lockfiles for the
# external-agents path (OpenHands + Terminus 2).
#
# Plan: docs/external_agents_integration/04-openhands-integration.md §4.2
#       docs/external_agents_integration/05-terminus-2-integration.md §5.1, §5.10
#       docs/external_agents_integration/10-phased-rollout-...md (Phase 1 / Phase 2)
#
# WHY THIS EXISTS
# ---------------
# Today *neither* external agent is importable:
#   - `import openhands`        -> ModuleNotFoundError  (OH SDK declared but NOT installed)
#   - `import terminal_bench`   -> ModuleNotFoundError  (TB checked out but NOT installed)
# Both BLOCKING spikes (OH spike §4.2, T2 import-smoke T2-S0 §5.1) need the
# packages importable *in the same interpreter that runs meta-n*. A throwaway
# venv is NOT sufficient — Phase 3 imports the SDK at run time — so we install
# into a single DEDICATED, PERSISTED venv (`.venv_external_agents`) at the repo
# root (the interpreter the bridge hard-codes) and commit the resulting
# `pip freeze` lockfiles under `constraints/`.
#
# The loose `==1.31.0` triple is NOT the source of truth: the pinned transitive
# deps (especially `litellm`, which shapes `.call()`/usage objects and
# `accumulated_cost`) ARE. That is exactly what the committed lockfiles capture.
#
# WHAT THIS SCRIPT DOES (idempotent — safe to re-run any number of times)
# ----------------------------------------------------------------------
#   1. Create `.venv_external_agents` at the repo root if it does not exist.
#   2. Install the OpenHands triple, pinned:
#        openhands-sdk==1.31.0 openhands-agent-server==1.31.0 openhands-tools==1.31.0
#      then freeze -> constraints/openhands-1.31.0.lock.txt
#   3. Install Terminus 2 editable:
#        pip install -e baselines/terminal-bench
#      then freeze -> constraints/terminal-bench.lock.txt
#   4. Smoke-import both packages so a broken install fails LOUDLY here, not
#      later inside a spike.
#   5. Print the next steps (run the two spike scripts).
#
# WHAT THIS SCRIPT DOES NOT DO
# ----------------------------
#   - It does NOT touch Docker, pull images, or run any benchmark / agent task.
#   - It does NOT modify meta-n's own environment (the project venv / system
#     Python). Everything lands inside `.venv_external_agents`.
#   - It does NOT edit pyproject.toml. The `openhands` / `terminus2` extras there
#     install against these committed lockfiles as constraints; this script is
#     what PRODUCES those lockfiles.
#
# REQUIREMENTS
# ------------
#   - Python 3.13.x (OpenHands's litellm dep requires >=3.12,<3.14; TB >=3.12).
#   - Network access to PyPI.
#   - `baselines/terminal-bench/` checked out (it is, in this repo).
#
# USAGE
# -----
#   scripts/install_external_agents.sh                 # install / refresh everything
#   PYTHON=python3.13 scripts/install_external_agents.sh
#   FORCE_RECREATE=1 scripts/install_external_agents.sh # blow away & rebuild .venv_external_agents
# ============================================================================

# Strict mode: stop on first error, unset-var use, or failed pipe stage. A bad
# install must NOT silently produce a half-written lockfile.
set -euo pipefail

# ----------------------------------------------------------------------------
# 0. Locate the repo root from this script's own path, so the script works no
#    matter the caller's cwd. All paths below are absolute / repo-relative.
# ----------------------------------------------------------------------------
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"
cd -- "${REPO_ROOT}"

# Tunables (overridable from the environment).
PYTHON="${PYTHON:-python3.13}"              # interpreter used to build the venv
VENV_DIR="${REPO_ROOT}/.venv_external_agents"  # the persisted, dedicated venv (§4.2)
CONSTRAINTS_DIR="${REPO_ROOT}/constraints"  # committed lockfiles live here
TB_DIR="${REPO_ROOT}/baselines/terminal-bench"
FORCE_RECREATE="${FORCE_RECREATE:-}"

# Pinned OpenHands triple (§4.2). Keep these three in lockstep at one version.
# Bumped 1.28.0 -> 1.31.0 (2026-07-05); 1.31.0 needs litellm>=1.84 which caps at
# Python <3.14, so build with python3.13 (the dead 3.12.13 base was retired).
OH_VERSION="1.31.0"
OH_LOCK="${CONSTRAINTS_DIR}/openhands-${OH_VERSION}.lock.txt"
TB_LOCK="${CONSTRAINTS_DIR}/terminal-bench.lock.txt"

VENV_PY="${VENV_DIR}/bin/python"
VENV_PIP="${VENV_DIR}/bin/pip"

# ----------------------------------------------------------------------------
# Small helpers for readable, sectioned output.
# ----------------------------------------------------------------------------
log()  { printf '\n\033[1;34m==>\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m  ok\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m  !!\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

# ----------------------------------------------------------------------------
# 1. Preflight: interpreter present and the right major.minor.
# ----------------------------------------------------------------------------
log "Preflight checks"

command -v "${PYTHON}" >/dev/null 2>&1 \
  || die "Python interpreter '${PYTHON}' not found. Set PYTHON=python3.13 (or a path)."

PY_VER="$("${PYTHON}" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
case "${PY_VER}" in
  3.12|3.13) ok "Using ${PYTHON} (Python ${PY_VER})" ;;
  *) die "Python ${PY_VER} unsupported. OpenHands needs >=3.12,<3.14 and TB needs >=3.12. \
Install Python 3.13 and re-run with PYTHON=python3.13." ;;
esac

[ -d "${TB_DIR}" ] \
  || die "terminal-bench source not found at ${TB_DIR}. Expected it checked out under baselines/."

mkdir -p "${CONSTRAINTS_DIR}"

# ----------------------------------------------------------------------------
# 2. Create (or reuse) the persisted venv. Idempotent: an existing, healthy
#    venv is reused so a re-run is fast; FORCE_RECREATE=1 rebuilds from scratch.
# ----------------------------------------------------------------------------
log "Persisted venv at ${VENV_DIR}"

if [ -n "${FORCE_RECREATE}" ] && [ -d "${VENV_DIR}" ]; then
  warn "FORCE_RECREATE set — removing existing ${VENV_DIR}"
  rm -rf -- "${VENV_DIR}"
fi

if [ -x "${VENV_PY}" ]; then
  ok "Reusing existing venv ($("${VENV_PY}" --version 2>&1))"
else
  "${PYTHON}" -m venv "${VENV_DIR}"
  ok "Created venv"
fi

# Always refresh the bootstrap toolchain. A current pip is required for the
# editable (PEP 660) TB install and for clean dependency resolution.
log "Upgrading pip / setuptools / wheel inside the venv"
"${VENV_PY}" -m pip install --upgrade pip setuptools wheel
ok "Build toolchain current"

# ----------------------------------------------------------------------------
# 3. Install the OpenHands triple (pinned) and freeze its lockfile.
#    Re-running upgrades to the exact pins and rewrites the lockfile — both are
#    idempotent given the same pins.
# ----------------------------------------------------------------------------
log "Installing OpenHands ${OH_VERSION} (sdk + agent-server + tools)"
"${VENV_PIP}" install \
  "openhands-sdk==${OH_VERSION}" \
  "openhands-agent-server==${OH_VERSION}" \
  "openhands-tools==${OH_VERSION}"
ok "OpenHands installed"

# Smoke-import so a broken install fails HERE, not inside the spike (§4.2 one-line
# `import openhands` smoke; the backend itself degrades to RuntimeError if absent).
log "Smoke-importing openhands"
"${VENV_PY}" -c 'import openhands; print("openhands import OK")' \
  || die "openhands installed but not importable — investigate before freezing."
ok "openhands importable"

# Freeze the *full* resolved dependency set (the real source of truth, §4.2).
log "Freezing -> ${OH_LOCK}"
"${VENV_PIP}" freeze > "${OH_LOCK}"
ok "Wrote $(wc -l < "${OH_LOCK}" | tr -d ' ') pinned lines to ${OH_LOCK#${REPO_ROOT}/}"

# ----------------------------------------------------------------------------
# 4. Install Terminus 2 (terminal-bench) editable in the SAME interpreter, then
#    freeze its lockfile (§5.1 / §5.10). Editable so local TB edits are picked
#    up without reinstalling.
# ----------------------------------------------------------------------------
log "Installing terminal-bench (editable) from ${TB_DIR#${REPO_ROOT}/}"
"${VENV_PIP}" install -e "${TB_DIR}"
ok "terminal-bench installed (editable)"

# Smoke-import including the exact symbol the T2 backend lazy-imports (§5.1, T2-S0).
log "Smoke-importing terminal_bench + Terminus2"
"${VENV_PY}" -c \
  'import terminal_bench; from terminal_bench.agents.terminus_2.terminus_2 import Terminus2; print("terminal_bench import OK")' \
  || die "terminal_bench installed but Terminus2 not importable — investigate before freezing."
ok "terminal_bench + Terminus2 importable"

# Freeze the TB lockfile. This is taken AFTER the OH install, so it reflects the
# fully-resolved combined environment (both agents share one interpreter, §5.1).
log "Freezing -> ${TB_LOCK}"
"${VENV_PIP}" freeze > "${TB_LOCK}"
ok "Wrote $(wc -l < "${TB_LOCK}" | tr -d ' ') pinned lines to ${TB_LOCK#${REPO_ROOT}/}"

# ----------------------------------------------------------------------------
# 5. Done — print next steps. The two BLOCKING spikes come next (Phase 1 / 2).
# ----------------------------------------------------------------------------
cat <<EOF

============================================================================
 Install complete. One persisted venv + two committed lockfiles:

   venv     : ${VENV_DIR#${REPO_ROOT}/}
   OH lock  : ${OH_LOCK#${REPO_ROOT}/}
   TB lock  : ${TB_LOCK#${REPO_ROOT}/}

 Commit the two lockfiles under constraints/ — they are the source of truth
 for the 'openhands' / 'terminus2' pyproject extras.

 NEXT STEPS — run the two BLOCKING spikes (do these before any full run):

   1. OpenHands spike (Phase 1, plan §4.2 — BLOCKING gate Checkpoint B):
        scripts/spikes/spike_openhands.sh
      Resolves spike #1 (in-process vs REST) and #2-#9 (ctors, metrics
      accessor, staging, command stream, provider env var, budget semantics,
      agent-server lifecycle + conversation DELETE).

   2. Terminus 2 import-smoke + micro-spikes (Phase 2, plan §5.1 T2-S0/S1/S2):
        scripts/spikes/spike_terminus2.sh
      Confirms the Terminus2 ctor / LiteLLM.call / AgentResult.failure_mode /
      episode-dir signatures, tmux presence, '{session}-main-1' container-name
      resolution, and that 'docker cp' lands helpers/ at /app/helpers/.

 Both spikes use this same '${VENV_DIR#${REPO_ROOT}/}/bin/python' interpreter.
============================================================================
EOF
