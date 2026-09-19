"""Append-only JSONL logger for raw LLM I/O.

Used by every LLM call site in meta-n + baselines so post-hoc analysis can
reconstruct exactly what was sent and received. One JSON object per line:

    {ts, source, model, messages, response, prompt_tokens, completion_tokens,
     total_tokens, extra}

Thread- and process-safe via fcntl.flock — the inner-LLM path runs solve()
in mp.Process workers, which may share a single output file when multiple
workers evaluate the same candidate (text_classification's
_run_solve_code_parallel chunks by case index).

Disabled by default. A logger is created only when a runner sets up its
output_dir/llm_io/ tree; absent a logger, every call site is a no-op.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

from meta_n.utils.flock_append import flock_append_bytes


class LLMIOLogger:
    """Append-only, thread-and-process-safe JSONL logger.

    Multiple LLMIOLogger instances may target the same path (one per
    subprocess); fcntl.flock serialises writes across processes. Within a
    single process, `_lock` serialises across threads so concurrent
    asyncio tasks can't interleave half-written lines.
    """

    def __init__(self, path: str | Path, source: str = "llm"):
        self.path = Path(path)
        self.source = source
        self._lock = threading.Lock()
        # Caller may have created the parent directory already, but doing
        # it here makes LLMIOLogger usable in tests / smoke checks without
        # ceremony.
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        *,
        messages: list | None = None,
        response: str = "",
        model: str = "",
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        extra: dict | None = None,
    ) -> None:
        """Append one JSONL record. Failures are swallowed with a stderr
        warning — losing one log line should never sink a run."""
        record: dict[str, Any] = {
            "ts": time.time(),
            "source": self.source,
            "model": model,
            # Normalise messages so pydantic ChatCompletionMessage instances
            # (which the OpenAI SDK injects when the caller reuses an
            # assistant turn from a prior response) survive json.dumps with
            # full structure. Without this, json.dumps falls back to
            # ``default=str`` and emits the object's repr instead of a
            # parseable {role, content, tool_calls} dict — breaking any
            # downstream analysis that wants to walk tool_calls.
            "messages": _coerce_messages(messages),
            "response": response or "",
            "prompt_tokens": int(prompt_tokens or 0),
            "completion_tokens": int(completion_tokens or 0),
            "total_tokens": int(total_tokens or 0),
        }
        if extra:
            record["extra"] = extra
        try:
            line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
            data = line.encode("utf-8")
            with self._lock:
                # LLM I/O lines are much larger than PIPE_BUF, so a plain
                # O_APPEND write is not atomic — flock_append_bytes holds an
                # exclusive flock and loops on short writes (see its docstring).
                flock_append_bytes(self.path, data)
        except Exception as e:  # noqa: BLE001 — never propagate a logging error
            import sys
            print(f"[LLMIOLogger] failed to append to {self.path}: {e!r}",
                  file=sys.stderr, flush=True)


def _coerce_messages(messages) -> list:
    """Convert a heterogeneous messages list into JSON-friendly dicts.

    Accepts dicts (passed through), pydantic message objects (model_dump
    preferred), and unknown types (wrapped). Returns [] for None/empty.
    Defensive: a single bad message must not corrupt the whole record.
    """
    if not messages:
        return []
    out: list = []
    for m in messages:
        if isinstance(m, dict):
            out.append(m)
            continue
        dump = getattr(m, "model_dump", None)
        if callable(dump):
            try:
                out.append(dump())
                continue
            except Exception:
                pass
        # Last resort — keep the raw repr so the record is still parseable
        # JSON. Wrapped under a sentinel key so consumers can spot it.
        out.append({"_raw": str(m)})
    return out


def make_logger(path: str | Path | None, source: str) -> LLMIOLogger | None:
    """Convenience: returns None when path is None/empty so callers stay terse.

    Note: ``bool(Path(""))`` is True in Python, so a naive truthiness check
    on a Path would create a logger pointing at the current directory.
    Stringify and check emptiness explicitly.
    """
    if path is None:
        return None
    if isinstance(path, Path):
        if str(path) == "" or str(path) == ".":
            return None
    elif not path:  # empty string / 0
        return None
    return LLMIOLogger(path, source=source)
