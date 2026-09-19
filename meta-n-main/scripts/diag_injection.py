"""Offline diagnosis: why does the Omega injection score 0.000?

The Kimi pilot's child solver was a 4-line wrapper that CORRECTLY called the
injected helper, so the injection plumbing works. The child still scored 0.000
against a parent at 0.494. That means the HELPER itself fails.

This replays the exact injection from the run's outer.jsonl against the real
CO-Bench instances, with no LLM calls and no cost.

Reads:
    <run>/llm_io/outer.jsonl   the raw Omega response (the injection)
    data/co_bench/<task>/      config.py (load_data) + case .txt files
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
import traceback
from pathlib import Path

sys.path.insert(0, ".")

FENCE = "`" * 3
TASK = "Bin packing - one-dimensional"
TASK_DIR_NAME = "Bin packing - one-dimensional"


def extract_blocks(text: str) -> dict:
    """Same fenced-block contract as OmegaEngine._parse_response."""
    out = {}
    for name in ("pre_process", "rationale"):
        m = re.search(FENCE + re.escape(name) + r"\s*\n(.*?)" + FENCE,
                      text, re.DOTALL)
        out[name] = m.group(1).strip() if m else None
    libs = {}
    for m in re.finditer(FENCE + r"solver_lib:(\w+)\s*\n(.*?)" + FENCE,
                         text, re.DOTALL):
        libs[m.group(1)] = m.group(2).strip()
    out["solver_libs"] = libs
    return out


def main() -> int:
    run_dirs = sorted(Path("runs/kimi_p5").glob("*/"))
    if not run_dirs:
        print("no run dir found")
        return 1
    run = run_dirs[-1]
    print("run:", run.name)

    outer = [json.loads(l) for l in (run / "llm_io" / "outer.jsonl").open()
             if l.strip()]
    print("outer records:", len(outer))

    omega = extract_blocks(str(outer[1]["response"]))
    print()
    print("=== Omega injection ===")
    print("  pre_process chars :", len(omega["pre_process"] or ""))
    print("  solver_libs       :", list(omega["solver_libs"]))
    print("  rationale head    :", (omega["rationale"] or "")[:120])

    # --- load the real CO-Bench task ---
    task_dir = Path("data/co_bench") / TASK_DIR_NAME
    spec = importlib.util.spec_from_file_location(
        "cobench_binpack", str(task_dir / "config.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    case = sorted(p for p in task_dir.iterdir()
                  if p.suffix == ".txt")[0]
    print()
    print("case file:", case.name)
    instances = mod.load_data(str(case))
    print("instances:", len(instances))
    inst = instances[0]
    print("instance keys:", sorted(inst.keys()) if isinstance(inst, dict)
          else type(inst).__name__)

    # --- build the helper namespace exactly as the child solver would ---
    ns: dict = {}
    for name, body in omega["solver_libs"].items():
        exec(body, ns)          # noqa: S102 - local diagnostic

    helper = ns.get("solve_one_dimensional_bin_packing")
    print("helper loaded:", helper is not None)

    # --- run the helper directly on a few instances ---
    print()
    print("=== running the injected helper directly ===")
    ok = fail = 0
    for i, inst in enumerate(instances[:8]):
        kw = dict(inst) if isinstance(inst, dict) else inst
        try:
            res = helper(bin_capacity=kw.get("bin_capacity"),
                         items=kw.get("items"),
                         num_items=kw.get("num_items"))
            n = res.get("num_bins") if isinstance(res, dict) else res
            print("  inst %d: returned %r" % (i, n))
            ok += 1
        except Exception as e:                                  # noqa: BLE001
            print("  inst %d: EXC %s: %s" % (i, type(e).__name__, e))
            if fail == 0:
                traceback.print_exc()
            fail += 1
    print("ok=%d fail=%d" % (ok, fail))

    # --- and the child wrapper as a whole ---
    print()
    print("=== running the child wrapper ===")
    child_ns = dict(ns)
    child_src = str(outer[2]["response"])
    m = re.search(FENCE + r"python\s*\n(.*?)" + FENCE, child_src, re.DOTALL)
    src = m.group(1) if m else child_src
    exec(src, child_ns)                                          # noqa: S102
    solve = child_ns.get("solve")
    for i, inst in enumerate(instances[:3]):
        kw = dict(inst) if isinstance(inst, dict) else inst
        try:
            r = solve(**kw)
            print("  inst %d -> %r" % (i, r if not isinstance(r, dict)
                                       else r.get("num_bins")))
        except Exception as e:                                  # noqa: BLE001
            print("  inst %d: EXC %s: %s" % (i, type(e).__name__, e))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
