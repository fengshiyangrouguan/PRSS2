"""Fix verification: helper-merge unit tests + offline gen2 revival.

Part 1 -- the four unit tests the fix must satisfy:
    1. old valid + new INVALID same name -> keep the old valid
    2. old valid + new VALID   same name -> new overrides
    3. only an INVALID helper            -> not injected
    4. bad pre_process                   -> does NOT affect an independent,
                                            valid code_library

Part 2 -- offline revival of the exact gen2 that died. Same artefacts, same
CO-Bench evaluator, NO new model calls:
    depth1/depth2/depth3 InjectedCode from the run's outer.jsonl
      -> FIXED merge
      -> the SAME gen2 solver source
      -> real _TaskEvaluator (10 s, unchanged)
Before the fix this NameErrored on 160/160 instances (score 0.000).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
FENCE = "`" * 3

VALID = "def foo(a, b):\n    return a + b\n"
INVALID = "import sys\ndef foo(a, b):\n    return a + b\n"   # blocked import


def _ic(depth, libs=None, pre=None):
    from meta_n.core.meta_layer import InjectedCode
    return InjectedCode(pre_process=pre, rationale="",
                        code_library=dict(libs or {}), source_depth=depth)


def part1() -> bool:
    from meta_n.core.meta_layer import merge_code_libraries
    from meta_n.core.code_library import validate_library_function
    from meta_n.utils.safety import validate_code

    print("=" * 68)
    print("PART 1 -- helper-merge unit tests")
    print("=" * 68)

    # sanity: the fixture really is invalid under the EXISTING predicate
    assert not validate_library_function("foo", INVALID, executor=None)
    assert validate_library_function("foo", VALID, executor=None)
    print("  fixture: VALID passes / INVALID (import sys) fails "
          "validate_library_function")

    ok = True

    # 1. old valid + new invalid same name -> keep old valid
    m, _ = merge_code_libraries([_ic(2, {"foo": VALID}),
                                 _ic(3, {"foo": INVALID})])
    t1 = m.get("foo") == VALID
    ok &= t1
    print("  %s 1. old valid + new INVALID same name -> old KEPT"
          % ("OK " if t1 else "** "))

    # 2. old valid + new valid same name -> new overrides
    new_valid = "def foo(a, b):\n    return a * b\n"
    m, _ = merge_code_libraries([_ic(2, {"foo": VALID}),
                                 _ic(3, {"foo": new_valid})])
    t2 = m.get("foo") == new_valid
    ok &= t2
    print("  %s 2. old valid + new VALID same name -> new OVERRIDES"
          % ("OK " if t2 else "** "))

    # 3. only invalid helper -> not injected
    m, _ = merge_code_libraries([_ic(2, {"foo": INVALID})])
    t3 = "foo" not in m
    ok &= t3
    print("  %s 3. only INVALID helper -> NOT injected"
          % ("OK " if t3 else "** "))

    # 4. bad pre_process does not affect an independent valid code_library
    bad_pre = 'x = (\n    "unterminated\n'
    assert not validate_code(bad_pre)[0]
    m, _ = merge_code_libraries([_ic(2, {"foo": VALID}, pre=bad_pre)])
    t4 = m.get("foo") == VALID
    ok &= t4
    print("  %s 4. bad pre_process -> independent valid code_library intact "
          "(merge never reads pre_process)" % ("OK " if t4 else "** "))
    return ok


def _parse(text, depth):
    from meta_n.core.meta_layer import InjectedCode
    s = str(text)

    def block(tag):
        k = s.find(FENCE + tag)
        if k < 0:
            return None
        nl = s.find("\n", k)
        e = s.find(FENCE, nl)
        return s[nl + 1:e].strip()

    libs = {}
    i = 0
    while True:
        k = s.find(FENCE + "solver_lib:", i)
        if k < 0:
            break
        nl = s.find("\n", k)
        name = s[k + len(FENCE) + len("solver_lib:"):nl].strip()
        e = s.find(FENCE, nl)
        libs[name] = s[nl + 1:e].strip()
        i = e + len(FENCE)
    return InjectedCode(pre_process=block("pre_process"),
                        rationale=block("rationale") or "",
                        code_library=libs, source_depth=depth)


def _py(text):
    s = str(text)
    k = s.find(FENCE + "python")
    if k < 0:
        return s
    nl = s.find("\n", k)
    e = s.find(FENCE, nl)
    return s[nl + 1:e].strip()


def part2() -> bool:
    from meta_n.core.code_library import prepend_python_library
    from meta_n.core.meta_layer import merge_code_libraries

    print()
    print("=" * 68)
    print("PART 2 -- offline revival of the SAME gen2 (no model calls)")
    print("=" * 68)
    run = sorted(Path("runs/val_ds").glob("*/"))[-1]
    recs = [json.loads(l)
            for l in (run / "llm_io" / "outer.jsonl").open() if l.strip()]
    ic2 = _parse(recs[3]["response"], 2)
    ic3 = _parse(recs[7]["response"], 3)
    script3 = _py(recs[8]["response"])

    merged, _ = merge_code_libraries([ic2, ic3])
    print("  merged library names:", sorted(merged))
    src_is_d2 = merged.get("solve_bin_packing") == ic2.code_library.get(
        "solve_bin_packing")
    print("  resolve: kept depth-2's valid helper: %s" % src_is_d2)

    composed = prepend_python_library(script3, merged, None)
    print("  composed chars: %d (was 548 = EMPTY library before the fix)"
          % len(composed))

    from meta_n.integrations.co_bench import _TaskEvaluator
    ev = _TaskEvaluator("Bin packing - one-dimensional",
                        Path("data/co_bench"), timeout=10, instance_workers=1)
    res = ev.evaluate(composed)
    score = res["dev_score"]
    fb = res["dev_feedback"]
    print("  dev score       : %.4f" % score)
    print("  timeouts/errors : %d / %d" % (fb.count("Timeout"),
                                           fb.count("Error")))
    print("  feedback head   : %s" % fb.splitlines()[0][:90])
    print()
    ok = score >= 0.9432
    print("  %s gen2 revived: %.4f %s 0.9432 (parent baseline)"
          % ("OK " if ok else "** ", score, ">=" if ok else "<"))
    return ok


def main() -> int:
    a = part1()
    b = part2()
    print()
    print("VERDICT:", "ALL OK" if (a and b) else
          "unit=%s replay=%s" % (a, b))
    return 0 if (a and b) else 1


if __name__ == "__main__":
    raise SystemExit(main())
