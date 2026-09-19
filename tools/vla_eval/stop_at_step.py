"""Stop the T1 RPBE run once it reaches a target optimizer step.

Waits for BOTH
  * the run log to show `opt <target>/`,
  * the requested `snapshot_<target>.pt` to exist (written by watch_steps.py),
and any eval at that step to have been logged, then terminates the TRAINING
process (the python one, not the shell wrapper) so the wrapper still writes its
own EXIT line. Checkpoints are left untouched.
"""
from __future__ import annotations

import argparse
import os
import re
import signal
import time
from pathlib import Path

RUN = Path("/root/autodl-tmp/runs/t1_rpbe_seed42")


def training_pids() -> list[int]:
    out = []
    me = os.getpid()
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        pid = int(d.name)
        if pid == me:
            continue
        try:
            cmd = (d / "cmdline").read_bytes().decode("utf-8", "replace")
        except OSError:
            continue
        if "train_libero_mem_rpbe.py" in cmd:
            out.append(pid)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=str(RUN))
    ap.add_argument("--target", type=int, required=True)
    ap.add_argument("--settle", type=float, default=45.0,
                    help="seconds to wait after the snapshot appears, so the "
                         "eval at that step lands too")
    ap.add_argument("--poll", type=float, default=20.0)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    log = run_dir / "run.log"
    snap = run_dir / f"snapshot_{args.target}.pt"
    print(f"[stopat] waiting for step {args.target} in {log}", flush=True)

    reached = False
    t0 = time.time()
    while not reached:
        try:
            txt = log.read_text(errors="replace")
        except OSError:
            txt = ""
        if re.search(rf"^opt {args.target}/", txt, re.M) and snap.exists():
            reached = True
            break
        if "EXIT rc=" in txt:
            print("[stopat] training already ended on its own", flush=True)
            return 0
        if not training_pids():
            print("[stopat] training process is gone", flush=True)
            return 0
        if time.time() - t0 > 4 * 3600:
            print("[stopat] gave up after 4h", flush=True)
            return 1
        time.sleep(args.poll)

    print(f"[stopat] step {args.target} reached and {snap.name} exists "
          f"({snap.stat().st_size / 1e9:.2f} GB)", flush=True)
    print(f"[stopat] settling {args.settle:.0f}s for the eval at this step",
          flush=True)
    time.sleep(args.settle)

    pids = training_pids()
    print(f"[stopat] training pids: {pids}", flush=True)
    for p in pids:
        try:
            os.kill(p, signal.SIGTERM)
            print(f"[stopat] sent SIGTERM to {p}", flush=True)
        except OSError as e:
            print(f"[stopat] could not signal {p}: {e}", flush=True)
    time.sleep(5)
    left = training_pids()
    for p in left:
        try:
            os.kill(p, signal.SIGKILL)
            print(f"[stopat] SIGKILL to {p}", flush=True)
        except OSError:
            pass
    print(f"[stopat] done; remaining training pids: {training_pids()}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
