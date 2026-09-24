"""One frozen candidate, ONE task, ONE held-out evaluation. Per arm.

WHY NOT `--stage final`. That stage evaluates the WHOLE cohort x
`profile.test_repeats` -- six tasks x three repeats by default -- and it also
requires the `freeze` stage. The diagnostic being run here is a single
stable-increment task, evaluated once, so a full-cohort final would spend ~18x
the evaluations on five tasks nobody reads.

THE ORDERING THIS PRESERVES, and it is the entire point:

    DEV selects  ->  freeze  ->  TEST once

`select_deployable` ranks on `mean_score`, which IS the dev macro; the test
split is touched only AFTER the winner is fixed, and the winner is never
re-chosen using test. If a test score ever fed back into selection, the
held-out number would silently become a training number. The code below calls
the runner's OWN `select_deployable` and `make_evaluator` rather than
re-implementing them, so this cannot drift from what `stage_final` would have
done -- it just does it for one task and one repeat.

WHAT IS DELIBERATELY SKIPPED: `freeze` and `audit`. Freeze snapshots hashes and
audit re-derives the canonical transition table; NEITHER selects the candidate.
The archive on disk after the arms finish is already the complete search
result, so selecting from it is the same decision `stage_final` would make.

Usage:
  python _single_task_test.py --out /root/autodl-tmp/sri_xbb_gemini-3_7-flash \
      [--task capacitated_warehouse_location] [--code <worktree>/meta-n-main]
"""
import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path


def load_runner(code_root: Path):
    spec = importlib.util.spec_from_file_location(
        "sri_runner", code_root / "scripts" / "run_sri_formal.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--code",
                    default="/root/autodl-tmp/wt_gemini-3_7-flash/meta-n-main")
    ap.add_argument("--task", default="capacitated_warehouse_location")
    ap.add_argument("--data-dir",
                    default="/root/autodl-tmp/meta-n-main/data/co_bench")
    ap.add_argument("--search-seed", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=10)
    ap.add_argument("--execute", action="store_true",
                    help="without it: PLANS only, no evaluator is run")
    args = ap.parse_args()

    code_root = Path(args.code)
    sys.path.insert(0, str(code_root))
    R = load_runner(code_root)
    from meta_n.sri.metrics import select_deployable
    from meta_n.sri.protocol import (default_profile_path, load_profile,
                                     sha256_of)
    from meta_n.sri.transition_audit import load_material_from_dir

    profile = load_profile(default_profile_path("sri_primary6"))
    slugs, _name_of = R.cohort_ids(profile)
    if args.task not in slugs:
        raise SystemExit("task %r is not in the cohort %s"
                         % (args.task, list(slugs)))
    ns = argparse.Namespace(evaluator="real", data_dir=args.data_dir,
                            timeout=args.timeout, search_seed=args.search_seed)
    out = Path(args.out)

    print("=" * 84)
    print("SINGLE-TASK HELD-OUT TEST   task=%s   repeats=1" % args.task)
    print("  DEV selects -> freeze -> TEST once; test never feeds selection")
    print("  out = %s" % out)
    print("=" * 84)

    evaluator = None
    if args.execute:
        evaluator = R.make_evaluator(ns, profile, split="test")

    rows = []
    for arm in R.arm_order_for_seed(args.search_seed):
        a = R.arm_dir(out, arm)
        idx = R.read_json(a / "archive" / "index.json")
        if idx is None:
            raise SystemExit("arm %s has no archive index under %s" % (arm, a))
        cands = list(idx.get("candidates") or [])
        for i, c in enumerate(cands):
            c["creation_index"] = i

        def executable(c, _a=a):
            material = load_material_from_dir(
                _a / "archive" / str(c.get("candidate_id")),
                str(c.get("candidate_id")), int(c.get("depth") or 1), slugs)
            return material is not None and material.is_executable_on(slugs)

        sel = select_deployable(cands, lambda c: c.get("mean_score"),
                                is_executable_of=executable)
        mat = R._selected_material(a, sel, slugs)
        if mat is None:
            raise SystemExit("arm %s: selected %s has no material on disk"
                             % (arm, sel.candidate_id))
        src = mat.task_scripts.get(args.task)

        # Same seed derivation `stage_final` uses, with r=0 because we run the
        # single repeat. Reusing it is what makes the number comparable to a
        # future full final on the same task.
        seed = int(sha256_of({"s": args.search_seed, "t": args.task,
                              "r": 0})[:8], 16) & 0x7FFFFFFF
        print()
        print("arm %-11s dev-selected candidate = %s  (depth=%s, dev macro=%.4f)"
              % (arm, sel.candidate_id, sel.structural_depth,
                 float(sel.dev_score)))
        print("             archive=%d candidates   task script present=%s"
              % (len(cands), bool(src)))
        if not src:
            print("             REFUSING to score: the selected candidate has "
                  "no script for %s" % args.task)
            continue
        if not args.execute:
            print("             [plan] would evaluate_test(%s, seed=%d) ONCE"
                  % (args.task, seed))
            rows.append((arm, sel, None))
            continue
        ev = evaluator.evaluate_test(args.task, src, seed=seed)
        sc = ev.score
        print("             TEST  %s = %s  (freshly_executed=%s)"
              % (args.task, ("%.6f" % float(sc)) if (sc is not None and
                 math.isfinite(float(sc))) else sc, ev.freshly_executed))
        rows.append((arm, sel, sc))

    if args.execute and len(rows) >= 2:
        o = [r for r in rows if r[0] == "official"][0]
        p = [r for r in rows if r[0] == "predictive"][0]
        if o[2] is not None and p[2] is not None:
            print()
            print("-" * 84)
            print("  Official  : dev=%.4f  test=%.6f"
                  % (float(o[1].dev_score), float(o[2])))
            print("  Predictive: dev=%.4f  test=%.6f"
                  % (float(p[1].dev_score), float(p[2])))
            print("  TEST delta (ours - official) = %+.6f"
                  % (float(p[2]) - float(o[2])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
