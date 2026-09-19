"""Cross-process-safe durable append for JSONL ledgers and logs.

Single source for the flock-guarded append block shared by
``CostTracker.record`` / ``CostTracker.record_usd`` and ``LLMIOLogger.log``
(and, through LLMIOLogger, the external-agent telemetry writers).

O_APPEND on POSIX gives atomic appends only for writes < PIPE_BUF
(typically 4 KiB); ledger and LLM I/O lines are often larger, so an
exclusive ``fcntl.flock`` serialises cross-process appends and a
short-write loop guarantees the whole line lands (POSIX permits partial
``os.write``; without the loop, a partial line corrupts both the current
and the next process's append — the next write begins on the same line,
no intervening ``\\n``).
"""

from __future__ import annotations

import fcntl
import os
from pathlib import Path


def flock_append_bytes(path: str | Path, data: bytes) -> None:
    """Append ``data`` to ``path`` under an exclusive flock, looping on
    short writes until every byte lands (or ``os.write`` raises).

    Cross-process safety only: callers that share one file object across
    threads must supply their own thread lock around this call (both
    CostTracker and LLMIOLogger hold a ``threading.Lock``).
    """
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            offset = 0
            n = len(data)
            while offset < n:
                written = os.write(fd, data[offset:])
                if written <= 0:
                    # 0 only happens on closed/full disk; treat as fatal
                    # to avoid an infinite loop.
                    raise OSError(
                        f"os.write returned {written} after {offset}/{n} bytes"
                    )
                offset += written
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)
