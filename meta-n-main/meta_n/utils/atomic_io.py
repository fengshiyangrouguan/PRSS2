"""Atomic JSON persistence shared by orchestrator and adapter cache writers.

Single home for the write-to-tmp-fsync-then-rename idiom (F193): callers get
crash-safe replacement without each hand-rolling the tmp-file dance.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

__all__ = ["atomic_json_dump"]


def atomic_json_dump(path: "Path | str", obj, *, indent: int | None = 2) -> None:
    """Write JSON to ``<path>.tmp``, fsync, then rename into place (atomic on POSIX).

    On ANY failure the tmp file is removed and the exception re-raised, so
    callers keep their own warn/raise policy. The payload is byte-identical to
    ``json.dump(obj, f, indent=indent)`` — this helper changes only the crash
    window: readers never observe a partially-written file, and the tmp file's
    data blocks are fsynced before the rename so an OS crash / power loss
    cannot leave a truncated ``path`` behind a durable rename. Durability of
    the rename itself (parent-directory fsync) is best-effort only — skipped
    where directories cannot be opened or fsynced.
    """
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "w") as f:
            json.dump(obj, f, indent=indent)
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(path)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
    try:
        dir_fd = os.open(path.parent, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(dir_fd)
    except OSError:
        pass
    finally:
        os.close(dir_fd)
