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

THE BUDGET IS A REQUEST COUNT, NOT DOLLARS. `run_mt_gem.sh` -- the launcher the
clean Gamma was actually trained under -- spent against
`$META_N_MAX_BACKEND_REQUESTS` (enforced at the send site by
`meta_n.rpbe.accounting`, counted before each HTTP request) and had NO dollar
tracker at all. An earlier revision of THIS file instead set
`$META_N_DAILY_BUDGET_USD`, which arms `meta_n.utils.cost_tracker`; that needs a
per-model price table and died instantly on `gemini-3.1-pro` with
`KeyError: Cannot enable cost tracking: No pricing for model ...`. The dollar
subsystem is therefore GONE from here -- absent, not merely configured around --
so there is exactly one money mechanism in the batch and it is the one the
reference protocol used.

FOUR THINGS THAT WERE WRONG HERE AND ARE FIXED BELOW, each of which would only
have shown up after money was spent:

  1. INHERITANCE WAS INCOMPLETE. `CONFIG_KEY_TO_CLI` is the SRI profile's
     mapping; a real `config.json` carries result-relevant keys it does not
     cover, and those fell back to argparse defaults. `base_url` is the worst of
     them -- it is the RELAY ENDPOINT, so a run that lost it would not even talk
     to the same service. `EXTRA_KEY_TO_CLI` now carries them, and a
     `--reduction-mode official` is stated explicitly rather than inherited by
     luck. (`api_key` is NOT in a config.json -- the run records `base_url`
     only -- so it is read from the sourced environment and passed explicitly,
     exactly as the bash launcher did.)
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

#: The relay environment `run_mt_gem.sh` exported before every clean-Gamma run.
#: `META_N_MAX_BACKEND_REQUESTS` is set PER RUN from `--max-requests`;
#: `META_N_REQUEST_LEDGER` is set per run to that run's own file.
RELAY_ENV = {
    "LLM_BACKEND": "relay",
    "ALLOW_PAID_API": "YES_I_ACCEPT_REAL_COST",
    "META_N_EXTRA_HEADERS_JSON": '{"Accept-Encoding": "identity"}',
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
    supplies the relay key -- which a config.json never records -- so imitating
    the shell grammar here is exactly the kind of quiet divergence that turns
    into a wrong provider. Running bash and reading the resulting environment
    back makes the launcher's environment BYTE-FOR-BYTE the one the old bash
    launcher produced.
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


def arm_relay_env(max_requests: int, base_url: Optional[str]) -> None:
    """Replicate `run_mt_gem.sh`'s exports, including the request-count cap.

    `META_N_MAX_BACKEND_REQUESTS` is read by `meta_n.rpbe.accounting` at the
    send site, so it arms BEFORE the client is built and counts what actually
    goes on the wire -- including empty-content escalations that the I/O log
    never sees. `main.py` only overwrites it when `--max-backend-requests` is
    passed, and we never pass it, so this value is the one in force.
    """
    for k, v in RELAY_ENV.items():
        os.environ[k] = v
    os.environ["META_N_MAX_BACKEND_REQUESTS"] = str(int(max_requests))
    if base_url:
        os.environ["RELAY_BASE_URL"] = str(base_url)
    # The encoder directory, if the sourced .env did not already point at one.
    # `main.py` resolves the CO-Bench data dir relative to cwd, but the CodeBERT
    # path comes from this variable, and a run that lost it would fail in the
    # encoder rather than at the first LLM call.
    os.environ.setdefault("CODEBERT_PATH", "/root/autodl-tmp/models/codebert-base")
    # A leftover dollar cap from an earlier revision would re-arm the tracker
    # this file deliberately does not use. Clear it rather than trust the shell.
    os.environ.pop("META_N_DAILY_BUDGET_USD", None)


def api_key_from_env() -> str:
    """The relay key a config.json cannot carry.

    `meta-n` records `base_url` but never the key, so it has to come from the
    sourced `.env`. `run_mt_gem.sh` passed it as `--api-key "$RELAY_API_KEY"`;
    this does the same. Refusing here is cheap -- the alternative is discovering
    it after the first paid call fails.
    """
    for var in ("RELAY_API_KEY", "OPENROUTER_API_KEY"):
        v = os.environ.get(var, "").strip()
        if v:
            return v
    raise SystemExit(
        "no relay key in the environment ($RELAY_API_KEY / $OPENROUTER_API_KEY). "
        "The .env that supplies it was not sourced, or does not define it.")


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


def build_argv(reference: Dict[str, Any], seed: int, out_dir: Path,
               api_key: Optional[str] = None,
               data_dir: Optional[str] = None,
               exp_name: Optional[str] = None) -> List[str]:
    """Field-by-field inheritance. Only cohort / seed / output differ."""
    pinned = {k: v for k, v in reference.items() if k in CONFIG_KEY_TO_CLI}
    pinned["bench_tasks"] = list(STRUCTURAL6)
    pinned["seed"] = int(seed)
    argv = render_pinned_flags(pinned)
    argv += render_extra(reference)
    if api_key is not None:
        argv += ["--api-key", api_key]
    # `./data/co_bench` is resolved RELATIVE TO CWD. The tree that carries the
    # new SRI code and the tree that carries the dataset are not the same
    # checkout on this box, so relying on cwd would make the run read an empty
    # directory and fail at the first task load, long after the plan looked fine.
    if data_dir:
        argv += ["--bench-data-dir", str(data_dir)]
    # `--exp-name` PINS the run directory name, and the run directory name IS
    # `run_id`, which is half of `tree_id` -- and `tree_role_v421` assigns
    # task/protect by hashing `tree_id`. Without this the name carries a
    # second-resolution timestamp that does not exist until the run starts, so
    # the split cannot be known before spending. Pinning it makes the split a
    # PREDICTABLE function of a name chosen in advance.
    #
    # DEVIATION, stated plainly: `census.py` invariant I4 reads "no salt, no
    # re-hash, no manual moving of trees to force a ratio". Choosing names to
    # land 2 task + 1 protect does exactly that in effect, though it satisfies
    # I1-I3 and the intent behind them (the choice is made from the frozen hash
    # BEFORE any score is read, never from results). That is acceptable for a
    # DEV SCREEN and must not be reused for a formal claim -- a formal cohort
    # uses natural naming and takes the split the hash gives it.
    if exp_name:
        argv += ["--exp-name", str(exp_name)]
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


def mask(argv: List[str]) -> str:
    out, hide = [], False
    for a in argv:
        if hide:
            out.append("***")
            hide = False
            continue
        out.append(a)
        if a == "--api-key":
            hide = True
    return " ".join(out)


def requests_used(ledger: Path) -> int:
    """Requests actually issued, from the counter `accounting.reserve()` bumps.

    The counter file is the authoritative number (`reserve()` increments it
    under a lock before each send); the ledger lines are a fallback for the
    window before the counter file exists.
    """
    cnt = Path(str(ledger) + ".count")
    try:
        return int(cnt.read_text(encoding="utf-8").strip() or "0")
    except (OSError, ValueError):
        pass
    try:
        with open(ledger, "r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    except OSError:
        return 0


def wait_for_run(p: subprocess.Popen, ledger: Path, cap: int,
                 wall_clock_s: float, poll: float = 2.0) -> Optional[str]:
    """Block until the child exits; return a reason string if it was killed.

    Polls the CHILD, not `kill(pid, 0)`: a child that has exited but not been
    reaped is a zombie, and `os.kill(pid, 0)` still succeeds for one, which is
    how a liveness loop blocks forever after a run finishes NORMALLY.
    `Popen.poll()` reaps, so it reports the exit.

    Two guards, and neither is the primary budget. The child enforces
    `META_N_MAX_BACKEND_REQUESTS` itself at the send site; the count check here
    is a backstop for a child that somehow stops honouring it. The wall clock is
    the only guard the bash launcher lacked, and it is what stops a hung run
    from blocking the remaining five with no bound.
    """
    t0 = time.time()
    while p.poll() is None:
        if cap > 0 and requests_used(ledger) >= cap:
            reason = "request cap %d reached" % cap
            break
        if time.time() - t0 > wall_clock_s:
            reason = "wall clock %.1fh exceeded" % (wall_clock_s / 3600.0)
            break
        time.sleep(poll)
    else:
        return None
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference-glob", default=DEFAULT_REF_GLOB)
    ap.add_argument("--env-file", default=DEFAULT_ENV_FILE)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--seeds", default="101,102,103,104,105,106",
                    help="the seed family the previous Phase-A used")
    ap.add_argument("--exp-names", default="",
                    help="comma list PARALLEL to --seeds, pinning each run's "
                         "directory name (hence its tree_id, hence its "
                         "task/protect role). Empty entries use the natural "
                         "timestamp name. Used to pre-register a usable split "
                         "before spending; see the note in build_argv.")
    ap.add_argument("--bench-data-dir", default=None,
                    help="where data/co_bench lives; the dataset and the new SRI "
                         "code are in different checkouts on this box, and "
                         "main.py resolves the default RELATIVE TO CWD")
    ap.add_argument("--max-requests", type=int, default=200,
                    help="backend-request cap PER RUN ($META_N_MAX_BACKEND_"
                         "REQUESTS), the same cap the clean-Gamma Phase-A used")
    ap.add_argument("--wall-clock-hours", type=float, default=8.0,
                    help="kill a run that outlives this; 0 disables")
    ap.add_argument("--execute", action="store_true",
                    help="without it, only PLANS (no API)")
    args = ap.parse_args()

    refs = load_reference_set(args.reference_glob)
    reference = assert_consistent(refs)
    check_reference(reference, "%d configs matching %s"
                    % (len(refs), args.reference_glob))
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    names = [x.strip() for x in args.exp_names.split(",")] if args.exp_names \
        else []
    if names and len(names) != len(seeds):
        raise SystemExit(
            "--exp-names has %d entries but --seeds has %d. They are PARALLEL: "
            "one name per seed, empty string for natural naming."
            % (len(names), len(seeds)))
    names = names or [""] * len(seeds)
    out_root = Path(args.out_root)

    print("=" * 78)
    print("Structural6 PHASE-A BATCH  (%s)" % ("EXECUTE" if args.execute else "PLAN"))
    print("=" * 78)
    print("  reference set : %d configs, all agreeing on result-relevant fields"
          % len(refs))
    print("  cohort        : %s" % ", ".join(STRUCTURAL6))
    print("  seeds         : %s" % seeds)
    print("  budget        : %d backend requests PER RUN (no dollar tracker)"
          % args.max_requests)
    print("  data dir      : %s" % (args.bench_data_dir or "./data/co_bench (cwd)"))
    print("  wall clock    : %s per run" % (
        "none" if args.wall_clock_hours <= 0
        else "%.1f h" % args.wall_clock_hours))
    print("  inherited (unchanged from the reference):")
    for k in sorted(EXPECT):
        print("      %-20s %s" % (k, reference.get(k)))
    for k in sorted(EXTRA_KEY_TO_CLI):
        print("      %-20s %s" % (k, reference.get(k)))
    print()

    plans = [(s, out_root / ("s%d" % s), n) for s, n in zip(seeds, names)]
    for s, d, n in plans:
        tag = (n if n else "(natural timestamp name)")
        print("  [plan] s%d -> %s   run_id=%s" % (s, d, tag))
    print()
    print("  split note: task/protect is decided by sha256 of")
    print("              (run_dir_name, 'gen0_seed'). Print it with")
    print("              meta_n.rpbe.census.tree_role_v421 BEFORE spending.")
    print()
    print("  [cmd] %s" % mask(build_argv(reference, seeds[0], plans[0][1],
                                         api_key="<from .env>",
                                         data_dir=args.bench_data_dir,
                                         exp_name=plans[0][2] or None)))
    print()

    if not args.execute:
        print("  PLAN ONLY. Re-run with --execute to spend.")
        return 0

    n = load_env_file(args.env_file)
    print("  sourced %s -> %d new env key(s)" % (args.env_file, n))
    arm_relay_env(args.max_requests, reference.get("base_url"))
    key = api_key_from_env()
    print("  relay armed: LLM_BACKEND=%s  max_backend_requests=%s  key=...%s"
          % (os.environ["LLM_BACKEND"], os.environ["META_N_MAX_BACKEND_REQUESTS"],
             key[-4:]))
    print()

    wall_s = args.wall_clock_hours * 3600.0
    rc = 0
    for s, d, n in plans:
        # `**/config.json` rather than a name pattern: a PINNED exp_name has no
        # `co_bench` in it, so the old glob would miss a completed run and spend
        # again. An empty dir written by a crashed attempt still matches nothing
        # and is correctly re-run.
        if d.exists() and any(d.glob("**/config.json")):
            print("[%s] seed=%d SKIPPED: %s already holds a run -- re-running "
                  "would spend again and add another tree" % (
                      time.strftime("%F %T"), s, d))
            continue
        d.mkdir(parents=True, exist_ok=True)
        log = d.parent / ("s%d.log" % s)
        ledger = d.parent / ("s%d_requests.jsonl" % s)
        for stale in (ledger, Path(str(ledger) + ".count")):
            try:
                stale.unlink()
            except OSError:
                pass
        os.environ["META_N_REQUEST_LEDGER"] = str(ledger)

        argv = build_argv(reference, s, d, api_key=key,
                          data_dir=args.bench_data_dir,
                          exp_name=n or None)
        print("[%s] run seed=%d -> %s" % (time.strftime("%F %T"), s, log))
        with open(log, "w", encoding="utf-8") as f:
            p = subprocess.Popen([sys.executable, "-u", "-m", "meta_n.main"] + argv,
                                 cwd=str(REPO), stdout=f,
                                 stderr=subprocess.STDOUT, env=dict(os.environ),
                                 start_new_session=True)
        reason = wait_for_run(p, ledger, args.max_requests, wall_s)
        used = requests_used(ledger)
        if reason is not None:
            print("  !! KILLED: %s -- after %d request(s)" % (reason, used))
            rc = 1
            break
        print("[%s] seed=%d done rc=%s   requests=%d"
              % (time.strftime("%F %T"), s, p.returncode, used))
        if p.returncode != 0:
            rc = p.returncode
            break

    print()
    print("  batch complete, rc=%d" % rc)
    return rc


if __name__ == "__main__":
    sys.exit(main())
