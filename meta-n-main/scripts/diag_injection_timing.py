"""Time the injected helper against the evaluator's timeout.

The gate SIGKILLed the child's evaluation. The helper carries a
`time_limit=15.0` wall-clock budget, so the question is how long one call
actually takes versus whatever timeout the CO-Bench evaluator enforces.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

FENCE = "`" * 3
TASK = "Bin packing - one-dimensional"


def main() -> int:
    # --- what timeout does the CO-Bench path use? ---
    from meta_n.integrations import co_bench
    sig = inspect.signature(co_bench.COBenchConfig.__init__) \
        if hasattr(co_bench, "COBenchConfig") else None
    print("=== CO-Bench timeout settings ===")
    for name, obj in vars(co_bench).items():
        if inspect.isclass(obj) and "Config" in name:
            try:
                src = inspect.getsource(obj)
            except OSError:
                continue
            for line in src.splitlines():
                if "timeout" in line.lower():
                    print("  %s: %s" % (name, line.strip()))
    print()

    # --- load the injected helper from the run ---
    run = sorted(Path("runs/kimi_p5").glob("*/"))[-1]
    outer = [json.loads(l)
             for l in (run / "llm_io" / "outer.jsonl").open() if l.strip()]
    s = str(outer[1]["response"])
    i = s.find("solver_lib:solve_one_dimensional_bin_packing")
    body = s[i:].split("\n", 1)[1]
    e = body.rfind(FENCE)
    body = body[:e] if e > 0 else body
    ns: dict = {}
    exec(body, ns)                                              # noqa: S102
    helper = ns["solve_one_dimensional_bin_packing"]
    print("helper signature:", inspect.signature(helper))
    print()

    # --- time it on real instances ---
    task_dir = Path("data/co_bench") / TASK
    spec = importlib.util.spec_from_file_location(
        "cb", str(task_dir / "config.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    print("=== per-instance wall clock (binpack1) ===")
    insts = mod.load_data(str(task_dir / "binpack1.txt"))
    total = 0.0
    for i, inst in enumerate(insts[:6]):
        t = time.time()
        r = helper(bin_capacity=inst["bin_capacity"], items=inst["items"],
                   num_items=inst["num_items"])
        dt = time.time() - t
        total += dt
        print("  inst %d: %6.2fs -> %s bins (best_known=%s)" % (
            i, dt, r.get("num_bins"), inst.get("best_known")))
    print("  ---- %.1fs for 6 instances ----" % total)

    print()
    print("=== across all 8 dev files (the set the gate evaluates) ===")
    grand = 0.0
    for f in sorted(task_dir.glob("binpack*.txt")):
        insts = mod.load_data(str(f))
        t = time.time()
        for inst in insts[:4]:
            helper(bin_capacity=inst["bin_capacity"], items=inst["items"],
                   num_items=inst["num_items"])
        dt = time.time() - t
        grand += dt
        print("  %-14s %d insts x4 -> %6.2fs" % (f.name, len(insts), dt))
    print("  ---- total %.1fs ----" % grand)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
