"""Structural6 budget-separation probe: FORMAL-shape Official SEARCH, same root.

THE QUESTION THIS ANSWERS. `sri_fast_s6_s0` ran Structural6 at B=1 K=1 T=4 and
the search went flat after one iteration (root 0.8772 -> best 0.8913, +0.0141),
so the dev score never opened a gap between the arms. Two explanations fit:

  A. the Structural6 cohort has no search headroom (tasks already near ceiling)
  B. the FAST profile cut the search budget so hard there was nothing to find

They are confounded because the fast run changed BOTH the cohort and the budget
relative to the PRIMARY6 formal run. This probe holds the ROOT fixed and moves
ONLY the budget, which is the one design that separates them:

    same Structural6 root (0.8772)  +  B=2 K=2 T=6 depth=6 repeats=3

Measured budget gap being closed: the formal arm explores ~7 iters x 2 x 2 ~ 24
candidates, the fast arm ~5 x 1 x 1 = 4.

WHY IT DOES NOT GO THROUGH `run_sri_formal.py`. That runner pins ONE profile for
the whole run and `require_stage(out, "root", inputs={"profile_sha256": ...})`
compares the whole profile hash, so attaching a formal-budget arm to a
root generated under the fast profile is REFUSED. The guard is coarser than its
own dependency, and that is checkable rather than assumed:

  * the root is generated with `max_iterations = 0` (see `build_root_cmd`), so
    beam_width / beam_candidates / max_iterations / search_eval_repeats never
    enter it;
  * `diff sri_structural6.yaml sri_structural6_fast.yaml` changes only budget,
    repeat counts, seed list and the request cap -- cohort, model, max_tokens
    and temperatures are identical, and those are what the root does depend on.

So this script builds the arm command with the repo's OWN renderers
(`pinned_for` + `render_pinned_flags`, exactly as `build_arm_cmd` does) and
launches it against a COPY of the frozen root. Nothing frozen is edited, and
the probe is deliberately NOT recorded as a formal SRI stage -- it is a probe.

SEARCH ONLY. `--no-test-eval` drops the test split, and no audit/freeze/final
stage exists here at all.

Usage:
    python _probe_formal_search.py            # PLAN: print the command only
    python _probe_formal_search.py --execute  # spend
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path("/root/autodl-tmp/rpbe-sri/meta-n-main")
sys.path.insert(0, str(REPO))

FAST = Path("/root/autodl-tmp/sri_fast_s6_s0")
PROBE = Path("/root/autodl-tmp/sri_s6_probe")
PROFILE = REPO / "meta_n/configs/sri_structural6.yaml"
DATA = "/root/autodl-tmp/meta-n-main/data/co_bench"
SEED = 0


class A:                       # the little namespace `pinned_for` wants
    backbone = "gemini-3.1-pro"
    search_seed = SEED
    data_dir = DATA
    evaluator = "real"
    gamma_checkpoint = None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()

    from meta_n.sri.protocol import load_profile, render_pinned_flags

    sys.path.insert(0, str(REPO / "scripts"))
    from run_sri_formal import (ARM_REDUCTION_MODE, launch_endpoint,
                                pinned_for, subprocess_env)

    profile = load_profile(str(PROFILE))
    pinned = pinned_for(A, profile)

    root_src = FAST / "root" / "phaseH_root"
    arm_dir = PROBE / "arms" / "official"
    ctx = PROBE / "context" / "official.json"

    print("=" * 78)
    print("PROBE: Structural6 FORMAL-shape Official SEARCH on the SAME root")
    print("=" * 78)
    print("  profile       : %s" % PROFILE.name)
    print("  profile sha   : %s" % profile.sha256()[:24])
    print("  B / K / T     : %s / %s / %s"
          % (profile.beam_width, profile.beam_candidates,
             profile.max_iterations))
    print("  max_depth     : %s" % profile.max_depth)
    print("  eval_repeats  : %s" % profile.search_eval_repeats)
    print("  cohort        : %s" % ", ".join(profile.cohort))
    print("  root (copied) : %s" % root_src)
    print("  arm dir       : %s" % arm_dir)
    print("  search only   : --no-test-eval, and no audit/freeze/final stage")
    print()

    ep = launch_endpoint()
    cmd = ([sys.executable, "-m", "meta_n.main"]
           + render_pinned_flags(pinned)
           + ["--no-test-eval", "--bench-data-dir", DATA,
              "--resume", "--base-url", ep["base_url"],
              "--reduction-mode", ARM_REDUCTION_MODE["official"],
              "--output-dir", str(PROBE / "arms"),
              "--exp-name", "official", "--sri-context", str(ctx)])
    print("  [cmd] %s" % " ".join(cmd))
    print()

    if not root_src.is_dir():
        print("REFUSING: no frozen root at %s" % root_src)
        return 2
    if not (root_src / "checkpoint.json").is_file():
        print("REFUSING: the frozen root has no checkpoint.json, so --resume "
              "would silently start a fresh run instead of continuing it.")
        return 2

    if not args.execute:
        print("PLAN ONLY. Re-run with --execute to spend.")
        return 0

    # RESUME IN PLACE when this arm already has a checkpoint. That is what
    # makes a transient relay 502 cheap: the first attempt was killed mid
    # iteration 2 of 6, and re-copying the root here would silently restart
    # from iteration 0 and re-spend everything it had already bought.
    resuming = (arm_dir / "checkpoint.json").is_file()
    if resuming:
        print("  RESUME in place from %s (checkpoint present; the root is NOT "
              "re-copied)" % arm_dir)
    else:
        if arm_dir.exists():
            # Only ever clear a path proven to sit inside PROBE, so a mistyped
            # constant cannot delete something else.
            if PROBE not in arm_dir.parents:
                print("REFUSING: %s is not under %s" % (arm_dir, PROBE))
                return 2
            print("  clearing stale %s" % arm_dir)
            shutil.rmtree(arm_dir)
        arm_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(root_src, arm_dir)
        ctx.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(FAST / "context" / "official.json", ctx)

        idx = arm_dir / "archive" / "index.json"
        import json
        n0 = len(json.loads(idx.read_text())["candidates"]) if idx.is_file() else 0
        print("  copied root: %d candidate(s) in the inherited archive" % n0)
        if n0 == 0:
            print("REFUSING: the copied archive is empty; --resume would start "
                  "fresh and the shared-root design would collapse.")
            return 2
    print("  launching...")
    log = PROBE / "official.log"
    # `subprocess_env()` maps the relay key onto OPENROUTER_API_KEY. Passing
    # `--api-key` instead would put the key in argv, where `ps` shows it to
    # every other tenant on the box -- that is why the runner does it this way.
    with log.open("w", encoding="utf-8") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                            cwd=str(REPO), env=subprocess_env()).returncode
    print("  rc=%d   log=%s" % (rc, log))
    return rc


if __name__ == "__main__":
    sys.exit(main())
