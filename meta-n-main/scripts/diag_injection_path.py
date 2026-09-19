"""Reproduce the REAL MetaLayer injection path for the depth-3 child.

The depth-2 pre_process was skipped for a syntax error, yet the depth-3 solver
NameErrored on every instance. `merge_code_libraries` does NOT consult
pre_process validity, and `prepend_python_library` validates each HELPER
independently -- so the coupling must be somewhere in this path.

This rebuilds the exact InjectedCode stack from the run's outer.jsonl and walks
the real helpers: merge -> validate_library_function -> prepend -> exec.
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, ".")
FENCE = "`" * 3


def parse_injected(text, depth):
    """Mirror OmegaEngine._parse_response."""
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

    libs_b = {}
    i = 0
    while True:
        k = s.find(FENCE + "solver_lib_bash:", i)
        if k < 0:
            break
        nl = s.find("\n", k)
        name = s[k + len(FENCE) + len("solver_lib_bash:"):nl].strip()
        e = s.find(FENCE, nl)
        libs_b[name] = s[nl + 1:e].strip()
        i = e + len(FENCE)
    return InjectedCode(pre_process=block("pre_process"),
                        rationale=block("rationale") or "",
                        code_library=libs, code_library_bash=libs_b,
                        source_depth=depth)


def py_block(text):
    s = str(text)
    k = s.find(FENCE + "python")
    if k < 0:
        return s
    nl = s.find("\n", k)
    e = s.find(FENCE, nl)
    return s[nl + 1:e].strip()


def main() -> int:
    from meta_n.core.code_library import (
        prepend_python_library, validate_library_function)
    from meta_n.core.meta_layer import merge_code_libraries
    from meta_n.utils.safety import validate_code

    run = sorted(Path("runs/val_ds").glob("*/"))[-1]
    recs = [json.loads(l)
            for l in (run / "llm_io" / "outer.jsonl").open() if l.strip()]

    ic2 = parse_injected(recs[3]["response"], 2)
    ic3 = parse_injected(recs[7]["response"], 3)
    script3 = py_block(recs[8]["response"])

    print("=== the two injected layers ===")
    for ic in (ic2, ic3):
        ok, err = validate_code(ic.pre_process) if ic.pre_process else (True, "")
        print("  d=%d pre_process valid=%s %s | libs=%s" % (
            ic.source_depth, ok, ("" if ok else err[:60]), list(ic.code_library)))

    merged_py, _ = merge_code_libraries([ic2, ic3])
    print()
    print("=== merged library (later layer wins by name) ===")
    for n in sorted(merged_py):
        v = validate_library_function(n, merged_py[n], executor=None)
        print("  %-24s validate_library_function = %s" % (n, v))

    print()
    print("=== which helper does the depth-3 solver actually call? ===")
    calls = re.findall(r"\b([A-Za-z_]\w*)\s*\(", script3)
    called = sorted(set(calls) - {"solve", "int", "len", "range", "list",
                                  "set", "dict", "sum", "min", "max", "sorted"})
    print("  called names:", called)
    for c in called:
        print("    %-24s in merged lib: %s" % (c, c in merged_py))

    print()
    print("=== prepend + exec exactly as MetaLayer does ===")
    composed = prepend_python_library(script3, merged_py, None)
    print("  composed chars:", len(composed))
    print("  composed head :", composed[:160].replace("\n", " | "))
    ns: dict = {}
    try:
        exec(composed, ns)                                       # noqa: S102
        print("  exec: OK")
    except Exception as e:                                       # noqa: BLE001
        print("  exec RAISED %s: %s" % (type(e).__name__, e))
        return 0

    import importlib.util
    task_dir = Path("data/co_bench/Bin packing - one-dimensional")
    spec = importlib.util.spec_from_file_location("cb", str(task_dir / "config.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    inst = mod.load_data(str(task_dir / "binpack1.txt"))[0]
    solve = ns.get("solve")
    if solve is None:
        print("  NO solve() in namespace")
        return 0
    try:
        r = solve(**inst)
        print("  solve(inst) ->", str(r)[:100])
    except Exception as e:                                       # noqa: BLE001
        print("  solve RAISED %s: %s" % (type(e).__name__, e))
        import traceback
        traceback.print_exc()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
