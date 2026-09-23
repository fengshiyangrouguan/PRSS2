"""Structural6 Phase-A batch: 6 paid runs, params INHERITED field-by-field.

WHY THIS EXISTS AND WHY IT DOES NOT HAND-WRITE ITS ARGUMENTS. Phase-A's job is to
produce the archives a Gamma is trained on, so its protocol must match the one
the PREVIOUS clean Gamma was trained under -- otherwise the new archives describe
a different distribution and the retrain silently measures nothing. Earlier
Phase-A material in this repo (Grok-era, single-task, max_tokens 1024) is NOT
that protocol; `mtgem_run1..6` is. So this launcher READS a real clean-Gamma
Phase-A `config.json` and renders the command from it through the project's own
CONFIG_KEY_TO_CLI table, changing ONLY:

    bench_tasks   -> Structural6
    seed          -> the per-run seed
    output dir    -> per run

Everything else -- B=1, K=1, T=4, max_depth=10, eval_repeats=1, gate_tasks=3,
temperatures, the backbone, the token budget, the ablation flags -- comes from
the reference, not from a human typing flags. A reference that does not look
like the clean-Gamma protocol is REFUSED rather than partially trusted.

THE BUDGET BUG THIS ALSO FIXES. `run_phase_a.sh` takes its ledger baseline inside
the per-run script, so looping it makes the "phase" cap cover ONE run: six runs
means six resets and no batch ceiling. Worse, it exports
META_N_DAILY_BUDGET_USD=$PER_RUN_CAP, and the cost ledger is a FILE SHARED
ACROSS PROCESSES (`<dir>/<YYYY-MM-DD>.jsonl`, re-scanned by today_total_usd), so
run 2 already sees run 1's spend against that cap and is killed early.

Here: ONE baseline for the whole batch, a batch-wide total cap, a per-run cap
that stays per-run, and the inner daily guard set to the BATCH allowance.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from meta_n.rpbe.budget import Caps, Fuse, total_cost            # noqa: E402
from meta_n.sri.protocol import (CONFIG_KEY_TO_CLI,              # noqa: E402
                                 STRUCTURAL6, render_pinned_flags)

#: The clean-Gamma Phase-A protocol. A reference config that disagrees on ANY of
#: these is refused: it would mean the archives describe a different distribution
#: than the Gamma they are meant to train.
EXPECT = {"beam_width": 1, "beam_candidates": 1, "max_iterations": 4,
          "max_depth": 10, "eval_repeats": 1, "gate_tasks": 3,
          "model": "gemini-3.1-pro"}

DEFAULT_REF_GLOB = ("/root/autodl-tmp/meta-n-main/runs/mtgem_run*/"
                    "*co_bench_gemini-3.1-pro/config.json")


def load_reference(path: str | None) -> Dict[str, Any]:
    p = path
    if not p:
        hits = sorted(glob.glob(DEFAULT_REF_GLOB))
        if not hits:
            raise SystemExit(
                "no reference Phase-A config found (looked in %s). Pass "
                "--reference-config explicitly." % DEFAULT_REF_GLOB)
        p = hits[0]
    cfg = json.loads(Path(p).read_text(encoding="utf-8"))
    bad = {k: {"reference": cfg.get(k), "expected": v}
           for k, v in EXPECT.items() if cfg.get(k) != v}
    if bad:
        raise SystemExit(
            "REFUSING {} as the Phase-A reference: it does not carry the "
            "clean-Gamma protocol. Disagreements: {}. Inheriting from it would "
            "produce archives from a different distribution than the Gamma they "
            "are meant to train.".format(p, bad))
    return cfg


def build_argv(reference: Dict[str, Any], seed: int, out_dir: Path) -> List[str]:
    """Field-by-field inheritance. Only cohort / seed / output differ."""
    pinned = {k: v for k, v in reference.items() if k in CONFIG_KEY_TO_CLI}
    pinned["bench_tasks"] = list(STRUCTURAL6)
    pinned["seed"] = int(seed)
    argv = render_pinned_flags(pinned)
    # `use_archive` is RENDER_ONLY (meta-n records `orchestrator: evolutionary`
    # instead of the flag), so it is not in the reference config and has to be
    # said explicitly -- without it the run is the linear path, not the archive
    # orchestrator, and the archives would be unusable.
    argv.append("--use-archive")
    argv += ["--output-dir", str(out_dir)]
    return [str(x) for x in argv]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference-config", default=None,
                    help="a real clean-Gamma Phase-A config.json (default: the "
                         "first mtgem_run* gemini-3.1-pro run)")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--seeds", default="101,102,103,104,105,106",
                    help="the seed family the previous Phase-A used")
    ap.add_argument("--per-run-cap", type=float, default=8.0)
    ap.add_argument("--batch-total-cap", type=float, default=36.0)
    ap.add_argument("--execute", action="store_true",
                    help="without it, only PLANS (no API)")
    args = ap.parse_args()

    ref_path = args.reference_config or sorted(
        glob.glob(DEFAULT_REF_GLOB))[0]
    reference = load_reference(args.reference_config)
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    out_root = Path(args.out_root)

    print("=" * 78)
    print("Structural6 PHASE-A BATCH  (%s)" % ("EXECUTE" if args.execute else "PLAN"))
    print("=" * 78)
    print("  reference config : %s" % ref_path)
    print("  cohort           : %s" % ", ".join(STRUCTURAL6))
    print("  seeds            : %s" % seeds)
    print("  inherited (unchanged from the reference):")
    for k in sorted(EXPECT):
        print("      %-20s %s" % (k, reference.get(k)))
    print()

    plans = []
    for s in seeds:
        d = out_root / ("s%d" % s)
        argv = build_argv(reference, s, d)
        plans.append((s, d, argv))
        print("  [plan] %s" % " ".join(argv))
    print()

    if not args.execute:
        print("  PLAN ONLY. Re-run with --execute to spend.")
        return 0

    # ONE baseline for the WHOLE batch -- taken here, once, before any run.
    baseline = total_cost()
    caps = Caps(per_run=float(args.per_run_cap), total=float(args.batch_total_cap))
    # The inner guard's ledger is SHARED ACROSS PROCESSES, so it must see the
    # batch allowance. Setting it to the per-run cap would make run 2 start
    # already "over budget" from run 1's spend.
    os.environ["META_N_DAILY_BUDGET_USD"] = str(float(args.batch_total_cap))
    print("  batch baseline $%.4f   per-run cap $%.2f   batch total cap $%.2f"
          % (baseline, caps.per_run, caps.total))
    print("  META_N_DAILY_BUDGET_USD=$%s (the BATCH allowance, not per-run)"
          % os.environ["META_N_DAILY_BUDGET_USD"])
    print()

    rc = 0
    for s, d, argv in plans:
        d.mkdir(parents=True, exist_ok=True)
        log = d.parent / ("s%d.log" % s)
        print("[%s] run seed=%d -> %s" % (time.strftime("%F %T"), s, log))
        with open(log, "w", encoding="utf-8") as f:
            p = subprocess.Popen([sys.executable, "-u", "-m", "meta_n.main"] + argv,
                                 cwd=str(REPO), stdout=f,
                                 stderr=subprocess.STDOUT, env=dict(os.environ),
                                 start_new_session=True)
        # The SAME baseline for every run: phase_spend = total - baseline is then
        # cumulative over the batch, which is what a batch cap means.
        reason = Fuse(p.pid, caps, phase_baseline=baseline).watch()
        if reason not in ("process exited",):
            print("  !! FUSE FIRED: %s -- killing the run" % reason)
            try:
                os.killpg(os.getpgid(p.pid), 15)
            except OSError:
                pass
            p.wait()
            rc = 1
            break
        p.wait()
        print("[%s] seed=%d done rc=%s   batch spend $%.4f"
              % (time.strftime("%F %T"), s, p.returncode,
                 total_cost() - baseline))
        if p.returncode != 0:
            rc = p.returncode
            break

    print()
    print("  REAL batch spend: $%.4f" % (total_cost() - baseline))
    return rc


if __name__ == "__main__":
    sys.exit(main())
