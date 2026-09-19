"""Shared subprocess-evaluation helpers for the integration adapters.

Single home for the pieces co_bench / text_classification / openevolve /
arc_agi previously copy-pasted:

* the canonical inner-LLM usage dict helpers (``_empty_usage``,
  ``_tracker_usage``, ``_coerce_usage``, ``_add_usage``),
* ``_kill_process_tree`` (killpg fast path + psutil fallback),
* ``_snapshot_log_offset`` / ``_usage_from_log_since`` (snapshot the inner-log
  size before a child starts, then reconstruct that killed child's partial
  inner-LLM usage from the JSONL records it appended after the snapshot),
* ``detach_process_group`` (child-side setsid preamble), and
* ``run_process_with_timeout`` (the mp.Process + drain-queue-then-join +
  kill-ladder skeleton).

Contract: each integration module re-exports the names it historically
exposed, so ``from meta_n.integrations.co_bench import _kill_process_tree``
keeps working. Monkeypatching an integration module's re-exported global
intercepts DIRECT calls from that module (e.g. the hard_kill seam), but NOT
the timeout kill-ladder inside ``run_process_with_timeout`` — that resolves
through THIS module's globals; patch ``_subprocess_utils._kill_process_tree``
to intercept it. This module imports only stdlib (+ optional psutil), so
importing an integration no longer drags in sibling integrations transitively.
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
import signal
import time
from queue import Empty

try:
    import psutil  # for recursive child-process cleanup
except ImportError:
    psutil = None

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Canonical inner-LLM usage dict helpers
# ---------------------------------------------------------------------------

def _empty_usage() -> dict[str, int]:
    return {"total": 0, "prompt": 0, "completion": 0, "calls": 0}


def _tracker_usage(tracker) -> dict[str, int]:
    """Snapshot an LLMUsageTracker into the canonical usage dict."""
    return {
        "total": int(getattr(tracker, "total_tokens", 0) or 0),
        "prompt": int(getattr(tracker, "prompt_tokens", 0) or 0),
        "completion": int(getattr(tracker, "completion_tokens", 0) or 0),
        "calls": int(getattr(tracker, "call_count", 0) or 0),
    }


def _coerce_usage(value) -> dict[str, int]:
    """Normalise a usage value (possibly None / missing keys) to the canonical
    {total, prompt, completion, calls} shape with int values."""
    if not isinstance(value, dict):
        return _empty_usage()
    return {
        "total": int(value.get("total", 0) or 0),
        "prompt": int(value.get("prompt", 0) or 0),
        "completion": int(value.get("completion", 0) or 0),
        "calls": int(value.get("calls", 0) or 0),
    }


def _add_usage(a: dict[str, int], b: dict[str, int]) -> dict[str, int]:
    return {k: int(a.get(k, 0) or 0) + int(b.get(k, 0) or 0) for k in a}


# ---------------------------------------------------------------------------
# Process-tree cleanup
# ---------------------------------------------------------------------------

def _kill_process_tree(pid: int):
    """Kill a process and all its descendants.

    Prefers POSIX killpg() — atomic and race-free once the child has called
    setsid(). Falls back to psutil's recursive walk if killpg is unavailable
    or the child failed to detach (pgid != pid).
    """
    # Fast path: if the child detached into its own group, killpg reaps the
    # whole tree in one syscall, including any grandchildren spawned after
    # we'd snapshot psutil.children().
    if hasattr(os, "killpg") and hasattr(os, "getpgid"):
        try:
            pgid = os.getpgid(pid)
            if pgid == pid:  # confirms setsid succeeded for this child
                os.killpg(pgid, signal.SIGKILL)
                return
        except (OSError, ProcessLookupError):
            return  # child already gone

    # Fallback: walk descendants explicitly (Windows, or setsid failed)
    if psutil is None:
        return
    try:
        parent = psutil.Process(pid)
        for child in parent.children(recursive=True):
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        parent.kill()
    except psutil.NoSuchProcess:
        pass


def detach_process_group() -> None:
    """Child-side preamble: become our own session/process-group leader so the
    parent can reap the entire subtree atomically with killpg() (see
    :func:`_kill_process_tree`). Call this first in every subprocess target —
    necessary when the evaluated code spawns its own subprocesses or blocks in
    uninterruptible I/O (e.g. an inline llm() helper on a rate-limited call).
    """
    if hasattr(os, "setsid"):
        try:
            os.setsid()
        except OSError:
            pass  # already a session leader, or unsupported


# ---------------------------------------------------------------------------
# Inner-log usage reconstruction
# ---------------------------------------------------------------------------

def _snapshot_log_offset(inner_log_path: str | None) -> int:
    """Capture the inner-log byte size before a solver subprocess starts.

    The mate of :func:`_usage_from_log_since`: this offset is the
    ``start_offset`` that function reconstructs partial usage from, so records
    already appended by sibling chunks sharing the log file are not
    double-counted. A missing / unstattable log yields ``0``.
    """
    log_start_offset = 0
    if inner_log_path:
        try:
            log_start_offset = os.path.getsize(inner_log_path)
        except OSError:
            log_start_offset = 0
    return log_start_offset


def _usage_from_log_since(
    inner_log_path: str | None, start_offset: int, pid: int | None = None
) -> dict[str, int]:
    """Reconstruct partial inner-LLM usage from the JSONL inner-log.

    When a solver subprocess is killed (wall-clock timeout) or dies before it
    can put a usage dict on the queue, its in-process tracker is lost with the
    child — the only durable record of the inner-LLM calls it already made is
    the inner-log: every llm()/llm_batch() call appends one JSONL record with
    prompt/completion/total tokens (see ``LLMIOLogger``). We sum the records
    appended at/after ``start_offset`` (the log size captured before this
    subprocess started) so already-recorded history from sibling chunks that
    share the same log file is not double-counted. Best-effort: any missing
    log / read / parse error yields zeros rather than raising.

    ``pid``: when given, records stamped with a DIFFERENT ``extra.pid`` are
    excluded — concurrent sibling subprocesses append to the same log file, so
    the offset alone cannot attribute records to THIS child. Records without a
    pid stamp (legacy/plain) are always counted.
    """
    usage = _empty_usage()
    if not inner_log_path:
        return usage
    try:
        with open(inner_log_path, "rb") as f:
            f.seek(start_offset)
            for raw in f:
                line = raw.decode("utf-8", "replace").strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if not isinstance(rec, dict):
                    continue
                if pid is not None:
                    extra = rec.get("extra")
                    rec_pid = extra.get("pid") if isinstance(extra, dict) else None
                    if rec_pid is not None:
                        try:
                            if int(rec_pid) != int(pid):
                                continue  # a concurrent sibling's record
                        except (TypeError, ValueError):
                            pass  # unparseable stamp — treat as unstamped
                usage["total"] += int(rec.get("total_tokens", 0) or 0)
                usage["prompt"] += int(rec.get("prompt_tokens", 0) or 0)
                usage["completion"] += int(rec.get("completion_tokens", 0) or 0)
                usage["calls"] += 1
    except OSError:
        return _empty_usage()
    return usage


# ---------------------------------------------------------------------------
# Subprocess-with-timeout harness
# ---------------------------------------------------------------------------

_NO_RESULT = object()


def run_process_with_timeout(
    target,
    args: tuple,
    timeout: int,
    *,
    queue: mp.Queue,
    on_timeout,
    on_no_result,
) -> tuple:
    """Run ``target(*args)`` in an mp.Process with a hard wall-clock timeout.

    The shared skeleton of the four integration eval harnesses (co_bench,
    text_classification, openevolve, arc_agi). ``queue`` must be the same
    mp.Queue embedded in ``args`` — each target takes the queue at its own
    positional slot, so the caller builds both. On a clean finish the child's
    queued tuple is returned verbatim, preserving each integration's payload
    arity (2-tuple or 3-tuple).

    The parent DRAINS the queue before joining: a child whose queued pickle
    exceeds the OS pipe buffer (~64KiB) blocks at exit in mp.Queue's feeder
    thread until the parent reads, so a join-before-get skeleton deadlocks on
    large payloads and misreports the successful run as a timeout.

    ``on_timeout(elapsed_s: float, child_pid: int) -> tuple`` and
    ``on_no_result(exc: Exception, child_pid: int) -> tuple`` build each
    integration's failure tuple; the child pid lets callers attribute
    inner-log records to THIS child (see :func:`_usage_from_log_since`).
    """
    p = mp.Process(target=target, args=args)
    start = time.time()
    p.start()

    def _reap():
        # Recursively kill child processes (solvers spawn subprocesses).
        # Tree first, while the child is still alive: getpgid on a SIGTERM'd
        # zombie fails (macOS), which would skip the killpg fast path and
        # orphan grandchildren.
        _kill_process_tree(p.pid)
        p.terminate()
        p.join(1)
        if p.is_alive():
            p.kill()
            p.join(1)
            if p.is_alive():
                # Benign: the process group was already killpg'd with SIGKILL
                # (uncatchable, no orphaned grandchildren); a leader still
                # showing alive here is reaped by the next mp spawn's _cleanup()
                # / the atexit handler. debug (not warning) — it's log noise, not
                # a leak. (Verified: fires within ~175ms, nothing was wedged.)
                logger.debug("Process %d not yet reaped after SIGKILL", p.pid)

    try:
        deadline = start + timeout + 1
        result = _NO_RESULT
        get_error: Exception | None = None
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            try:
                result = queue.get(timeout=min(remaining, 0.2))
                break
            except Empty:
                if p.is_alive():
                    continue
                try:
                    # 1.0s grace lets mp.Queue's feeder thread flush a result
                    # the subprocess put just before exiting. get_nowait()
                    # previously raced the feeder and discarded successful runs
                    # whose result arrived in the last few ms before exit.
                    result = queue.get(timeout=1.0)
                except Exception as e:
                    get_error = e
                break
            except Exception as e:
                get_error = e
                break

        if result is not _NO_RESULT:
            p.join(1)
            if p.is_alive():
                _reap()
            return result

        if p.is_alive():
            elapsed = time.time() - start
            _reap()
            if get_error is not None:
                return on_no_result(get_error, p.pid)
            return on_timeout(elapsed, p.pid)

        p.join(1)
        return on_no_result(get_error if get_error is not None else Empty(), p.pid)
    finally:
        # Clean up queue resources to avoid leaks
        try:
            queue.close()
            queue.join_thread()
        except Exception:
            pass
        try:
            p.close()
        except ValueError:
            pass  # process still alive after kill — already logged above
