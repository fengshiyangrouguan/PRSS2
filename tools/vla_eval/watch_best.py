"""Preserve T1 checkpoints from outside the training process.

Why this is needed: `--checkpoint-every 1000` overwrites `latest.pt` in place,
`best.pt` is overwritten on every val improvement, and `--snapshot-steps`
defaults to "30000,35000,40000" which never fires in a 20000-step run. Without
this watcher the run leaves only two usable files at the end.

Watches BOTH:
  best.pt   -> snapshot_best_<step>.pt    (the val-improvement trajectory)
  latest.pt -> snapshot_latest_<step>.pt  (every periodic point)

On any mtime change the file is COPIED ASIDE FIRST (so a later in-place
overwrite can never race us), then its own `step` field is read from the copy
and the copy is renamed accordingly. Pure CPU + disk; never touches the
training process or the GPU.
"""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import torch

RUN = Path("/root/autodl-tmp/runs/t1_avg_seed42")


def snapshot(src: Path, dest_dir: Path, kind: str) -> int | None:
    """Copy <src> -> snapshot_<kind>_<step>.pt. None if already present."""
    tmp = dest_dir / f".inflight_{kind}.pt"
    shutil.copy2(src, tmp)
    payload = torch.load(tmp, map_location="cpu", weights_only=False)
    step = int(payload.get("step", -1))
    val = payload.get("best_val")
    out = dest_dir / f"snapshot_{step}.pt"
    if out.exists():
        tmp.unlink()
        return None
    tmp.replace(out)
    sz = out.stat().st_size / 1e9
    print(f"[snapshot] step={step:<6d} (from {kind}) best_val={val} -> "
          f"{out.name} ({sz:.2f} GB)", flush=True)
    return step


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=str(RUN))
    ap.add_argument("--interval", type=float, default=15.0)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    targets = {"best": run_dir / "best.pt", "latest": run_dir / "latest.pt"}
    log = run_dir / "run.log"

    if args.once:
        for kind, p in targets.items():
            if p.exists():
                snapshot(p, run_dir, kind)
        return 0

    print(f"[snapshot] watching {run_dir} every {args.interval}s", flush=True)
    seen: dict[str, float] = {}
    while True:
        try:
            for kind, p in targets.items():
                if not p.exists():
                    continue
                m = p.stat().st_mtime
                if seen.get(kind) != m:
                    seen[kind] = m
                    snapshot(p, run_dir, kind)
            if log.exists():
                txt = log.read_text(errors="replace")
                if "T1 EXIT" in txt:
                    print("[snapshot] training done; final pass", flush=True)
                    time.sleep(20)
                    for kind, p in targets.items():
                        if p.exists():
                            snapshot(p, run_dir, kind)
                    return 0
        except FileNotFoundError:
            pass
        except Exception as e:                      # noqa: BLE001
            print(f"[snapshot] warning: {type(e).__name__}: {e}", flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
