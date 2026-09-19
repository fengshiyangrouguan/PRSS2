"""Backend-request accounting with a hard pre-send cap.

Task book context: a 3-iteration pilot billed 3324 requests where only 2341 were
log-visible. The 983 difference was `complete_with_breakdown` recursing under
`_suppress_io_log=True` for the empty-content escalation: BILLED but invisible.
So the money fuse must count what actually goes on the wire, not what the
semantic layer thinks it called.

Two counters, never conflated:

    logical_calls     one user/solver-level llm() invocation
    backend_requests  one HTTP request about to be sent  <-- the money gate

`reserve()` MUST be called immediately BEFORE the request is issued: count and
check first, then send. Counting after the send would let request N+1 spend
before the cap is noticed.

Cross-process safe: workers are separate processes, so the counter lives in a
file guarded by fcntl.flock (a threading fallback covers Windows dev boxes).
"""

from __future__ import annotations

import json
import os
import time
import uuid
from typing import Optional

REQUEST_LEDGER_ENV = "META_N_REQUEST_LEDGER"
CAP_ENV = "META_N_MAX_BACKEND_REQUESTS"
CAP_ENV_LEGACY = "META_N_MAX_LLM_CALLS"

PRIMARY = "primary"
EMPTY_ESCALATION = "empty_escalation"
TRANSPORT_RETRY = "transport_retry"
MOCK = "mock"                      # offline backend: counted, never billed
LOCAL = "local"                    # local model: real generation, still free
OFFLINE_KINDS = frozenset({MOCK, LOCAL})


class LLMCallBudgetExceeded(RuntimeError):
    """Raised BEFORE the request is sent, so no money is spent by the call
    that would have breached the cap."""


def _ledger_path() -> Optional[str]:
    p = os.environ.get(REQUEST_LEDGER_ENV, "").strip()
    return p or None


def _counter_path() -> Optional[str]:
    p = _ledger_path()
    return (p + ".count") if p else None


def backend_request_cap() -> int:
    """0 disables the cap. The explicit name wins over the legacy one."""
    for var in (CAP_ENV, CAP_ENV_LEGACY):
        raw = os.environ.get(var, "").strip()
        if raw:
            try:
                return int(raw)
            except ValueError:
                continue
    return 0


def new_logical_call_id() -> str:
    return uuid.uuid4().hex[:12]


# --------------------------------------------------------------------------
# the fuse itself
# --------------------------------------------------------------------------

def reserve(kind: str = PRIMARY, *, logical_call_id: Optional[str] = None,
            parent_request_id: Optional[int] = None) -> int:
    """Atomically take the next backend-request slot, then check the cap.

    Returns the 1-based backend_request_id. Raises LLMCallBudgetExceeded when
    the cap is exceeded -- BEFORE the caller sends anything.

    A no-op (returns 0) when no request ledger is configured, so existing runs
    keep their byte-identical behaviour.
    """
    path = _ledger_path()
    counter = _counter_path()
    if not path or not counter:
        return 0

    n = _bump(counter)

    rec = {
        "backend_request_id": n,
        "logical_call_id": logical_call_id or "unknown",
        "request_kind": kind,
        "parent_request_id": parent_request_id,
        "pid": os.getpid(),
        "ts": time.time(),
    }
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    except OSError:
        pass

    cap = backend_request_cap()
    if cap and n > cap:
        raise LLMCallBudgetExceeded(
            "backend request {} > cap {} ({}). HARD STOP before sending."
            .format(n, cap, CAP_ENV))
    return n


def _bump(counter_path: str) -> int:
    """Read-increment-write under an exclusive lock."""
    try:
        import fcntl
    except ImportError:                                    # Windows dev box
        fcntl = None
    with open(counter_path, "a+", encoding="utf-8") as f:
        if fcntl is not None:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.seek(0)
            raw = f.read().strip()
            n = (int(raw) + 1) if raw else 1
            f.seek(0)
            f.truncate()
            f.write(str(n))
            f.flush()
            os.fsync(f.fileno())
            return n
        finally:
            if fcntl is not None:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def record_offline(kind: str = MOCK,
                   logical_call_id: Optional[str] = None) -> None:
    """Record an offline-backend call (`mock` or `local`).

    Counted for diagnostics, never billed: no HTTP request leaves the box, so
    the fuse must not consume budget for it.
    """
    path = _ledger_path()
    if not path:
        return
    rec = {
        "backend_request_id": 0,
        "logical_call_id": logical_call_id or "unknown",
        "request_kind": kind,
        "parent_request_id": None,
        "pid": os.getpid(),
        "ts": time.time(),
    }
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    except OSError:
        pass


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def snapshot() -> dict:
    """Count backend requests by kind from the request ledger.

    ``backend_requests`` EXCLUDES mock calls -- it is the number the money
    fuse gates on. ``mock_calls`` is reported separately for unit tests.
    """
    by_kind: dict[str, int] = {}
    logical = set()
    path = _ledger_path()
    if path and os.path.isfile(path):
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                k = r.get("request_kind", "unknown")
                by_kind[k] = by_kind.get(k, 0) + 1
                logical.add(r.get("logical_call_id"))
    off = sum(v for k, v in by_kind.items() if k in OFFLINE_KINDS)
    return {"backend_requests": sum(v for k, v in by_kind.items()
                                    if k not in OFFLINE_KINDS),
            "offline_calls": off,
            "mock_calls": by_kind.get(MOCK, 0),
            "by_kind": by_kind,
            "logical_calls": len(logical)}


def format_accounting() -> str:
    s = snapshot()
    cap = backend_request_cap()
    lines = ["LLM ACCOUNTING",
             "logical_calls:        {}".format(s["logical_calls"]),
             "backend_requests:     {}".format(s["backend_requests"])]
    for k in (PRIMARY, EMPTY_ESCALATION, TRANSPORT_RETRY):
        lines.append("  {:<20}{}".format(k + ":", s["by_kind"].get(k, 0)))
    for k, v in sorted(s["by_kind"].items()):
        if k not in (PRIMARY, EMPTY_ESCALATION, TRANSPORT_RETRY):
            lines.append("  {:<20}{}".format(k + ":", v))
    lines.append("")
    lines.append("backend request cap:  {}".format(cap or "disabled"))
    return "\n".join(lines)
