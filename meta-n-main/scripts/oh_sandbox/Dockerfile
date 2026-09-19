# OpenHands runner sandbox image (plan: Docker-sandboxed OH execution).
#
# Runs scripts/oh_runner.py INSIDE a container so the OH agent's terminal tool
# shells out against the container's filesystem, never the bare host. This
# removes the host-shell-out exfiltration risk that the host path mitigates only
# via an env-var allowlist.
#
# requirements.txt pins the same openhands-sdk 1.31.0 triple as the host venv,
# but resolved by pip INSIDE python:3.13-slim (linux) — a raw macOS host freeze
# is not portable here (macOS-only pyobjc arrives via browser-use->screeninfo,
# and linux needs a different screeninfo backend). See requirements.txt header.
# Python 3.13-slim tracks the host venv's 3.13.14 interpreter.
#
# The LLM endpoint is reached at build/run time via host.docker.internal (the
# backend rewrites 127.0.0.1 -> host.docker.internal and passes
# --add-host=host.docker.internal:host-gateway), since a container cannot reach
# the host loopback. The runner script itself is bind-mounted at run time (not
# COPYed) so edits to oh_runner.py take effect without a rebuild.
FROM python:3.13-slim

# git is pulled in by some openhands tool deps at import time; ca-certificates
# for litellm/httpx TLS to the (future) remote providers; tmux for the OH
# terminal tool (without it the tool warns and falls back to a less-stable
# subprocess terminal — still works, but tmux is the recommended backend).
RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates tmux \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/oh

COPY requirements.txt /opt/oh/requirements.txt
RUN pip install --no-cache-dir -r /opt/oh/requirements.txt

# Keep the SDK banner off stdout (the runner parses a clean result-file, but a
# stray banner in container logs is noise).
ENV OPENHANDS_SUPPRESS_BANNER=1 \
    PYTHONUNBUFFERED=1

# The workspace is bind-mounted at run time; default to a sane CWD.
WORKDIR /workspace
