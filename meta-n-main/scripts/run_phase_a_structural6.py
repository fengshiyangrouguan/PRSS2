"""Structural6 Phase-A batch: 6 paid runs, params INHERITED field-by-field.

WHY THIS EXISTS AND WHY IT DOES NOT HAND-WRITE ITS ARGUMENTS. Phase-A's job is to
produce the archives a Gamma is trained on, so its protocol must match the one
the PREVIOUS clean Gamma was trained under -- otherwise the new archives describe
a different distribution and the retrain silently measures nothing. Earlier
Phase-A material in this repo (Grok-era, single-task, max_tokens 1024) is NOT
that protocol; `mtgem_run1..6` is. So this launcher READS real clean-Gamma
`config.json` files and renders the command from them, changing ONLY:

    bench_tasks   -> Structural6
    seed          -> the per-run seed
    output dir    -> per run

FOUR THINGS THAT WERE WRONG HERE AND ARE FIXED BELOW, each of which would only
have shown up after money was spent:

  1. INHERITANCE WAS INCOMPLETE. `CONFIG_KEY_TO_CLI` is the SRI profile's
     mapping; a real `config.json` carries result-relevant keys it does not
     cover, and those fell back to argparse defaults. `base_url` is the worst of
     them -- it is the RELAY ENDPOINT, so a run that lost it would not even talk
     to the same service. `EXTRA_KEY_TO_CLI` now carries them, and a
     `--reduction-mode official` is stated explicitly rather than inherited by
     luck.
  2. NO `.env`. The old bash launcher sourced it before every run; this one
     inherited the ambient shell, so a PLAN looked green and only the paid
     EXECUTE would fail on a missing key.
  3. `Fuse.watch()` CAN HANG FOREVER. It decides liveness with `os.kill(pid, 0)`,
     and on Linux a child that has EXITED but not yet been reaped is a ZOMBIE,
     for which that call still SUCCEEDS. The first Phase-A would finish and the
     launcher would block on it, never starting the second. The loop is now
     driven by `Popen.poll()`, which reaps.
  4. THE REFERENCE WAS ONE CONFIG. `hits[0]` is only sound if all six agree, so
     that is now CHECKED rather than assumed -- and it holds.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from meta_n.sri.protocol import (CONFIG_KEY_TO_CLI,              # noqa: E402
                                 STRUCTURAL6, render_pinned_flags)

# `meta_n.rpbe.budget` is imported INSIDE main(), AFTER .env is sourced. It
# freezes `LEDGER_DIR = Path(os.environ.get("META_N_COST_LEDGER_DIR", ...))` at
# IMPORT time, so importing it at module level would pin the ledger to the
# pre-.env default while the child processes write to the .env one -- the
# external fuse would then watch a file nobody writes and report zero spend.

#: The clean-Gamma Phase-A protocol. A reference config that disagrees on ANY of
#: these is refused: it would mean the archives describe a different distribution
#: than the Gamma they are meant to train.
EXPECT = {"beam_width": 1, "beam_candidates": 1, "max_iterations": 4,
          "max_depth": 10, "eval_repeats": 1, "gate_tasks": 3,
          "model": "gemini-3.1-pro"}

#: Result-relevant keys `main.py` records that `CONFIG_KEY_TO_CLI` does not
#: cover. Dropping them silently reverts each to its argparse default. Measured
#: on the six clean-Gamma configs, every one of these AGREES across all six, so
#: inheriting them cannot smuggle a per-run difference in.
EXTRA_KEY_TO_CLI: Dict[str, str] = {
    "base_url": "--base-url",
    "omega_context_budget": "--omega-context-budget",
    "exclude_providers": "--exclude-providers",
    "request_timeout": "--request-timeout",
    "retry_threshold": "--retry-threshold",
    "epsilon": "--epsilon",
}

#: Fields expected to differ per run; everything ELSE must agree across the set.
PER_RUN_FIELDS = ("seed", "timestamp", "benchmark_config_applied")

DEFAULT_REF_GLOB = ("/root/autodl-tmp/meta-n-main/runs/mtgem_run*/"
                    "*co_bench_gemini-3.1-pro/config.json")
DEFAULT_ENV_FILE = "/root/autodl-tmp/meta-n-main/.env"


def load_env_file(path: str) -> int:
    """`set -a; . <file>; set +a`, done by ACTUAL BASH.

    Hand-parsing is not equivalent to `source`: it cannot do `export K=v`, does
    not expand `K="$OTHER/xxx"`, and mishandles quoting/continuations. This file
    supplies the relay key, the pricing override JSON and the cost ledger
    directory -- so imitating the shell grammar here is exactly the kind of
    quiet divergence that turns into a wrong provider or a wrong ledger. Running
    bash and reading the resulting environment back makes the launcher's
    environment BYTE-FOR-BYTE the one the old bash launcher produced.
    """
    p = Path(path)
    if not p.is_file():
        raise SystemExit(
            "env file {} not found. Pass --env-file. The previous Phase-A "
            "launcher sourced this before every run; skipping it turns a "
            "missing relay key into a failure that only appears once the "
            "first PAID call is attempted.".format(path))
    before = set(os.environ)
    r = subprocess.run(["bash", "-c", 'set -a; . "$1"; set +a; env -0',
                        "_", str(p)], capture_output=True)
    if r.returncode != 0:
        raise SystemExit("sourcing {} failed: {}".format(
            p, r.stderr.decode("utf-8", "replace")[-400:]))
    for chunk in r.stdout.split(b"\x00"):
        if not chunk:
            continue
        k, _, v = chunk.decode("utf-8", "replace").partition("=")
        if k:
            os.environ[k] = v
    return len(set(os.environ) - before)


def child_ledger_path() -> Path:
    """The EXACT file the child's `CostTracker` will write to.

    `CostTracker` keys its ledger by the MACHINE-LOCAL date (`utc=False`) under
    `META_N_COST_LEDGER_DIR` (default `~/.meta_n_costs`), while `budget.py`
    freezes that env var at import and keys by UTC. When local != UTC those are
    two different files for part of every day -- not only across a UTC midnight
    -- and an external fuse watching the wrong one sees no spend at all. This
    reproduces the child's convention rather than approximating it.
    """
    d = os.environ.get("META_N_COST_LEDGER_DIR", "").strip()
    base = Path(d).expanduser() if d else Path.home() / ".meta_n_costs"
    import datetime as _dt
    return base / "{}.jsonl".format(_dt.datetime.now().strftime("%Y-%m-%d"))


def load_reference_set(pattern: str) -> List[Dict[str, Any]]:
    hits = sorted(glob.glob(pattern))
    if not hits:
        raise SystemExit("no reference Phase-A configs matched %s" % pattern)
    return [json.loads(Path(p).read_text(encoding="utf-8")) for p in hits]


def assert_consistent(refs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """One config is only a valid reference if ALL of them agree.

    Otherwise `hits[0]` silently picks one run's protocol for the whole cohort
    and nothing records that a choice was made.
    """
    keys = set().union(*[set(r) for r in refs])
    bad = {}
    for k in sorted(keys):
        if k in PER_RUN_FIELDS:
            continue
        vals = {json.dumps(r.get(k), sort_keys=True) for r in refs}
        if len(vals) > 1:
            bad[k] = sorted(vals)[:3]
    if bad:
        raise SystemExit(
            "the %d reference configs DISAGREE on result-relevant field(s) %s, "
            "so one of them cannot stand in for the protocol. Resolve that "
            "before spending." % (len(refs), sorted(bad)))
    return dict(refs[0])


def check_reference(reference: Dict[str, Any], where: str) -> None:
    bad = {k: {"reference": reference.get(k), "expected": v}
           for k, v in EXPECT.items() if reference.get(k) != v}
    if bad:
        raise SystemExit(
            "REFUSING {} as the Phase-A reference: it does not carry the "
            "clean-Gamma protocol. Disagreements: {}. Inheriting from it would "
            "produce archives from a different distribution than the Gamma they "
            "are meant to train.".format(where, bad))


def render_extra(reference: Dict[str, Any]) -> List[str]:
    """The keys `CONFIG_KEY_TO_CLI` does not cover. None means 'omit'."""
    argv: List[str] = []
    for key, flag in sorted(EXTRA_KEY_TO_CLI.items()):
        if key not in reference:
            continue
        v = reference[key]
        if v is None:
            continue                     # the flag's default IS None: omit it
        if isinstance(v, (list, tuple)):
            argv.append(flag)
            argv.extend(str(x) for x in v)
        else:
            argv += [flag, str(v)]
    return argv


def build_argv(reference: Dict[str, Any], seed: int, out_dir: Path) -> List[str]:
    """Field-by-field inheritance. Only cohort / seed / output differ."""
    pinned = {k: v for k, v in reference.items() if k in CONFIG_KEY_TO_CLI}
    pinned["bench_tasks"] = list(STRUCTURAL6)
    pinned["seed"] = int(seed)
    argv = render_pinned_flags(pinned)
    argv += render_extra(reference)
    # `use_archive` is RENDER_ONLY (meta-n records `orchestrator: evolutionary`
    # instead of the flag), so it is not in the reference config and has to be
    # said explicitly -- without it the run is the linear path, not the archive
    # orchestrator, and its archives could not feed build_records.
    argv.append("--use-archive")
    # Phase-A is the NATIVE rule by definition. Stated rather than inherited: the
    # argparse default happens to be `official` today, and a Phase-A whose
    # archives were produced under the adapter would train Gamma on its own
    # output.
    argv += ["--reduction-mode", "official"]
    argv += ["--output-dir", str(out_dir)]
    return [str(x) for x in argv]


def wait_with_fuse(p: subprocess.Popen, fuse: Fuse, poll: float = 2.0):
    """Poll the CHILD, not `kill(pid, 0)`.

    A child that has exited but not been reaped is a zombie, and `os.kill(pid, 0)`
    still succeeds for one -- which is how a `Fuse.watch()` loop blocks forever
    after a run finishes NORMALLY. `Popen.poll()` reaps, so it reports the exit.
    """
    while p.poll() is None:
        reason = fuse.check()
        if reason is not None:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            except OSError:
                pass
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                except OSError:
                    pass
                p.wait()
            return reason
        time.sleep(poll)
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference-glob", default=DEFAULT_REF_GLOB)
    ap.add_argument("--env-file", default=DEFAULT_ENV_FILE)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--seeds", default="101,102,103,104,105,106",
                    help="the seed family the previous Phase-A used")
    ap.add_argument("--per-run-cap", type=float, default=8.0)
    ap.add_argument("--batch-total-cap", type=float, default=36.0)
    ap.add_argument("--execute", action="store_true",
                    help="without it, only PLANS (no API)")
    args = ap.parse_args()

    refs = load_reference_set(args.reference_glob)
    reference = assert_consistent(refs)
    check_reference(reference, "%d configs matching %s"
                    % (len(refs), args.reference_glob))
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    out_root = Path(args.out_root)

    print("=" * 78)
    print("Structural6 PHASE-A BATCH  (%s)" % ("EXECUTE" if args.execute else "PLAN"))
    print("=" * 78)
    print("  reference set : %d configs, all agreeing on result-relevant fields"
          % len(refs))
    print("  cohort        : %s" % ", ".join(STRUCTURAL6))
    print("  seeds         : %s" % seeds)
    print("  inherited (unchanged from the reference):")
    for k in sorted(EXPECT):
        print("      %-20s %s" % (k, reference.get(k)))
    for k in sorted(EXTRA_KEY_TO_CLI):
        print("      %-20s %s" % (k, reference.get(k)))
    print()

    plans = [(s, out_root / ("s%d" % s), None) for s in seeds]
    for s, d, _ in plans:
        print("  [plan] s%d -> %s" % (s, d))
    print()
    for s, d, _ in plans[:1]:
        print("  [cmd] %s" % " ".join(build_argv(reference, s, d)))
    print()

    if not args.execute:
        print("  PLAN ONLY. Re-run with --execute to spend.")
        return 0

    n = load_env_file(args.env_file)
    print("  sourced %s -> %d new env key(s)" % (args.env_file, n))

    # AFTER the sourcing, or budget.py pins the pre-.env ledger dir.
    from meta_n.rpbe.budget import Caps, Fuse, total_cost

    ledger = child_ledger_path()
    # Spend ALREADY on the books for the child's own day-key. The inner guard
    # stops a run when the DAY's total reaches its cap, and it has no notion of a
    # batch baseline -- so setting the cap to the batch allowance alone would
    # kill a run partway through whenever earlier spend existed (measured
    # concern: $7 already spent + a $36 allowance leaves only $29 of headroom,
    # and the run dies at $36 of DAILY spend, not $36 of BATCH spend).
    #
    # The intent is "the batch may ADD up to batch_total_cap", so the inner cap
    # has to be today_before_batch + batch_total_cap; the outer fuse below then
    # enforces the part that actually matters, batch_spend <= batch_total_cap.
    baseline = total_cost(ledger)
    daily_cap = baseline + float(args.batch_total_cap)
    os.environ["META_N_DAILY_BUDGET_USD"] = str(daily_cap)
    caps = Caps(per_run=float(args.per_run_cap), total=float(args.batch_total_cap))
    print("  ledger        : %s" % ledger)
    print("  spend already on it today : $%.4f" % baseline)
    print("  inner daily cap           : $%.4f  (already-spent + batch cap)"
          % daily_cap)
    print("  batch cap (outer fuse)    : $%.2f   per-run cap $%.2f"
          % (caps.total, caps.per_run))
    print()

    rc = 0
    for s, d, _ in plans:
        if d.exists() and any(d.glob("*co_bench*")):
            print("[%s] seed=%d SKIPPED: %s already holds a run -- re-running "
                  "would spend again and add another tree" % (
                      time.strftime("%F %T"), s, d))
            continue
        d.mkdir(parents=True, exist_ok=True)
        log = d.parent / ("s%d.log" % s)
        argv = build_argv(reference, s, d)
        print("[%s] run seed=%d -> %s" % (time.strftime("%F %T"), s, log))
        with open(log, "w", encoding="utf-8") as f:
            p = subprocess.Popen([sys.executable, "-u", "-m", "meta_n.main"] + argv,
                                 cwd=str(REPO), stdout=f,
                                 stderr=subprocess.STDOUT, env=dict(os.environ),
                                 start_new_session=True)
        # The SAME baseline for every run: phase_spend = total - baseline is then
        # cumulative over the batch, which is what a batch cap means.
        # SAME path the child writes to, so phase_spend is the batch's own
        # new spend and not zero-by-mistake.
        reason = wait_with_fuse(
            p, Fuse(p.pid, caps, phase_baseline=baseline, path=ledger))
        if reason is not None:
            print("  !! FUSE FIRED: %s -- run killed" % reason)
            rc = 1
            break
        print("[%s] seed=%d done rc=%s   batch spend $%.4f"
              % (time.strftime("%F %T"), s, p.returncode,
                 total_cost(ledger) - baseline))
        if p.returncode != 0:
            rc = p.returncode
            break

    print()
    print("  REAL batch spend: $%.4f" % (total_cost(ledger) - baseline))
    return rc


if __name__ == "__main__":
    sys.exit(main())
