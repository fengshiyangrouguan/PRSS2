"""OFFLINE COUNTERFACTUAL: same child, only its time budget changes.

Full evidence chain already established:
  * the seed solver has an unbounded local-search loop -> the LARGER instances
    (binpack5-8) blow the evaluator's per-instance timeout and score 0.000,
    which is exactly the 0.4936 average;
  * Omega read that pattern correctly from the trace;
  * Omega's injected helper returns the OPTIMAL bin count offline
    (== best_known on every instance tried);
  * yet the real child scored 0.000 and failed the gate.

Cause: `COBenchAdapter` runs each instance in a subprocess with
``timeout: int = 10`` seconds, while the helper's own budget is
``time_limit: float = 15.0`` -- 1.5x over. The subprocess is SIGKILLed, and
CO-Bench counts a timeout as a 0.0 score, so the mean collapses.

This script changes EXACTLY ONE thing -- the helper's internal wall-clock
budget -- and re-runs the REAL evaluator. Nothing else moves: same seed, same
Omega output, same child wrapper, same data, same 10 s evaluator timeout,
same scoring. Zero LLM calls.

Reads the child straight out of the run's `llm_io/outer.jsonl`, so it is
byte-identical to what the gate actually saw.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

FENCE = "`" * 3
TASK = "Bin packing - one-dimensional"
RUN_GLOB = "runs/kimi_p5"
BUDGETS = (15.0, 5.0, 2.0)
EVAL_TIMEOUT = 10          # the REAL evaluator timeout; never changed here


def _blocks(text: str) -> dict:
    libs = {}
    for m in re.finditer(FENCE + r"solver_lib:(\w+)\s*\n(.*?)" + FENCE,
                         text, re.DOTALL):
        libs[m.group(1)] = m.group(2).strip()
    return libs


def _py_block(text: str) -> str:
    m = re.search(FENCE + r"python\s*\n(.*?)" + FENCE, text, re.DOTALL)
    return (m.group(1) if m else text).strip()


def _retime(helper_src: str, budget: float) -> str:
    """Change ONLY the default of `time_limit`."""
    new, n = re.subn(r"(time_limit\s*:\s*float\s*=\s*)[0-9.]+",
                     r"\g<1>%s" % budget, helper_src)
    if n != 1:
        raise SystemExit("expected exactly one time_limit default, got %d" % n)
    return new


def main() -> int:
    from meta_n.integrations.co_bench import _TaskEvaluator

    run = sorted(Path(RUN_GLOB).glob("*/"))[-1]
    outer = [json.loads(l)
             for l in (run / "llm_io" / "outer.jsonl").open() if l.strip()]
    omega = _blocks(str(outer[1]["response"]))
    helper_name, helper_src = next(iter(omega.items()))
    wrapper = _py_block(str(outer[2]["response"]))
    seed_src = _py_block(str(outer[0]["response"]))

    print("run                :", run.name)
    print("helper             :", helper_name)
    print("evaluator timeout  : %ds (UNCHANGED)" % EVAL_TIMEOUT)
    print("instance_workers   : 1 (sequential, deterministic)")
    print()

    data_dir = Path("data/co_bench")
    ev = _TaskEvaluator(TASK, data_dir, timeout=EVAL_TIMEOUT,
                        instance_workers=1)

    def run_case(label, solve_source, budget=None):
        t = time.time()
        res = ev.evaluate(solve_source)
        dt = time.time() - t
        score = res["dev_score"]
        fb = res["dev_feedback"]
        tos = fb.count("Timeout")
        zeros = 0
        for line in fb.splitlines():
            if "-> Scores:" in line:
                zeros += line.count("'0.000'") + line.count("0.000,")
        print("  %-22s score=%.4f  timeouts=%-3d  (%.0fs)" % (
            label, score, tos, dt))
        return score, tos

    print("=" * 70)
    print("OFFLINE COUNTERFACTUAL -- only the helper's time budget changes")
    print("=" * 70)
    seed_score, _ = run_case("seed (control)", seed_src)
    print("      seed 0.000-count files are the timeout signature:")
    print("      " + " | ".join(
        l for l in ev.evaluate(seed_src)["dev_feedback"].splitlines()
        if "-> Scores" in l)[:150])
    print()

    rows = []
    for b in BUDGETS:
        src = _retime(helper_src, b) + "\n\n" + wrapper
        s, t = run_case("child budget=%.0fs" % b, src)
        rows.append((b, s, t))

    print()
    print("=" * 70)
    print("%-10s %-14s %-12s %s" % ("budget", "timeouts", "mean_score",
                                    "vs seed"))
    print("%-10s %-14s %-12.4f %s" % ("seed", "-", seed_score, "(baseline)"))
    for b, s, t in rows:
        verdict = "PASS (>= seed)" if s >= seed_score else "below seed"
        print("%-10s %-14d %-12.4f %s" % ("%.0fs" % b, t, s, verdict))
    print()
    ok = any(s >= seed_score and t == 0 for _, s, t in rows)
    print("VERDICT:", ("ROOT CAUSE CONFIRMED -- the SAME child clears the "
                       "evaluator/gate once its own budget fits inside the "
                       "10 s limit") if ok else
          "not confirmed by these budgets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
