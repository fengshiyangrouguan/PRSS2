"""Snapshot ONLY a requested set of optimizer steps.

Same mechanism as watch2.py (copy aside FIRST so an in-place overwrite can't
race us, then read the checkpoint's own `step` field), but it persists only the
steps listed in --steps. Everything else is ignored, so the run dir does not
accumulate 13 near-duplicate 1.85 GB files.
"""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import torch

RUN = Path("/root/autodl-tmp/runs/t1_rpbe_seed42")


def snapshot(src: Path, dest_dir: Path, kind: str, want: set[int]) -> int | None:
    tmp = dest_dir / f".inflight_{kind}.pt"
    shutil.copy2(src, tmp)
    payload = torch.load(tmp, map_location="cpu", weights_only=False)
    step = int(payload.get("step", -1))
    if step not in want:
        tmp.unlink()
        return None
    out = dest_dir / f"snapshot_{step}.pt"
    if out.exists():
        tmp.unlink()
        return None
    tmp.replace(out)
    print(f"[snapshot] step={step:<6d} (from {kind}) best_val="
          f"{payload.get('best_val')} -> {out.name} "
          f"({out.stat().st_size / 1e9:.2f} GB)", flush=True)
    return step


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=str(RUN))
    ap.add_argument("--steps", required=True,
                    help="comma-separated optimizer steps to keep, e.g. "
                         "'10000,12000,15000,18000,20000'")
    ap.add_argument("--interval", type=float, default=15.0)
    ap.add_argument("--max-scan", type=float, default=0.0,
                    help="seconds to keep polling (0 = until TIERED/R2 exit)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    want = {int(x) for x in args.steps.split(",") if x.strip()}
    targets = {"best": run_dir / "best.pt", "latest": run_dir / "latest.pt"}
    log = run_dir / "run.log"
    print(f"[snapshot] keeping only steps {sorted(want)}", flush=True)
    print(f"[snapshot] watching {run_dir} every {args.interval}s", flush=True)

    seen: dict[str, float] = {}
    got: set[int] = set()
    t0 = time.time()
    while True:
        try:
            for kind, p in targets.items():
                if not p.exists():
                    continue
                m = p.stat().st_mtime
                if seen.get(kind) != m:
                    seen[kind] = m
                    s = snapshot(p, run_dir, kind, want)
                    if s is not None:
                        got.add(s)
            if log.exists() and "EXIT rc=" in log.read_text(errors="replace"):
                if got >= want:
                    print("[snapshot] all requested steps captured; done",
                          flush=True)
                    return 0
                print(f"[snapshot] training finished; requested={sorted(want)} "
                      f"got={sorted(got)} missing={sorted(want - got)}",
                      flush=True)
                return 0
            if args.max_scan and (time.time() - t0) > args.max_scan:
                print("[snapshot] max-scan reached", flush=True)
                return 0
        except Exception as e:                      # noqa: BLE001
            print(f"[snapshot] warning: {type(e).__name__}: {e}", flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
