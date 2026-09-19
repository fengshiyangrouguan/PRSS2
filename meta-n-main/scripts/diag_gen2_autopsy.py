"""gen2 autopsy: why did depth-3 crash on EVERY instance in 1.0s?

Log evidence: "160 instances, 160 errors (0 timeouts)" -- not a timeout, an
immediate exception in every instance. This dumps the depth-3 child's solver
and the Omega injection that produced it, then replays the crash locally.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
FENCE = "`" * 3


def main() -> int:
    run = sorted(Path("runs/val_ds").glob("*/"))[-1]
    recs = [json.loads(l)
            for l in (run / "llm_io" / "outer.jsonl").open() if l.strip()]

    def blocks(text):
        s = str(text)
        out = {}
        for tag in ("rationale", "pre_process"):
            k = s.find(FENCE + tag)
            if k >= 0:
                e = s.find(FENCE, k + len(FENCE) + len(tag))
                out[tag] = s[k + len(FENCE) + len(tag):e].strip()
        libs = {}
        i = 0
        while True:
            k = s.find(FENCE + "solver_lib:", i)
            if k < 0:
                break
            name_end = s.find("\n", k)
            name = s[k + len(FENCE) + len("solver_lib:"):name_end].strip()
            e = s.find(FENCE, name_end)
            libs[name] = s[name_end:e].strip()
            i = e + len(FENCE)
        out["libs"] = libs
        return out

    print("=== Omega injection #2 (depth 3) ===")
    om = blocks(recs[7]["response"])
    print("  pre_process chars:", len(om.get("pre_process") or ""))
    print("  libs            :", list(om["libs"]))
    print("  rationale head  :", (om.get("rationale") or "")[:200])
    print()
    print("  pre_process body:")
    print("   ", (om.get("pre_process") or "").replace("\n", "\n    ")[:700])

    print()
    print("=== depth-3 child solver ===")
    src = str(recs[8]["response"])
    k = src.find(FENCE + "python")
    if k >= 0:
        e = src.find(FENCE, k + len(FENCE) + 6)
        src = src[k + len(FENCE) + 6:e]
    print(src[:1400])

    print()
    print("=== static scan of the child solver ===")
    for pat in ("solve_", "import ", "def ", "llm("):
        print("  %-10s x%d" % (pat, src.count(pat)))

    # --- replay: which names does it reference? ---
    import ast
    try:
        tree = ast.parse(src)
    except SyntaxError as exc:
        print("  SYNTAX ERROR:", exc)
        return 0
    defined = {n.name for n in ast.walk(tree)
               if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    defined |= {t.id for n in ast.walk(tree) if isinstance(n, ast.Assign)
                for t in n.targets if isinstance(t, ast.Name)}
    print("  defined names:", sorted(defined)[:12])

    # --- run it against one real instance to see the actual exception ---
    import importlib.util
    task_dir = Path("data/co_bench/Bin packing - one-dimensional")
    spec = importlib.util.spec_from_file_location("cb", str(task_dir / "config.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    inst = mod.load_data(str(task_dir / "binpack1.txt"))[0]

    ns = {}
    for name, body in om["libs"].items():
        try:
            exec(body, ns)                                       # noqa: S102
            print("  helper %s: loaded OK" % name)
        except Exception as e:                                   # noqa: BLE001
            print("  helper %s: LOAD FAILED %s: %s" % (name,
                                                       type(e).__name__, e))
    try:
        exec(src, ns)                                            # noqa: S102
        solve = ns.get("solve")
        print("  solver: loaded OK" if solve else "  solver: NO solve()")
    except Exception as e:                                       # noqa: BLE001
        print("  solver: LOAD FAILED %s: %s" % (type(e).__name__, e))
        return 0

    if solve:
        try:
            r = solve(**inst)
            print("  solve(inst) ->", str(r)[:120])
        except Exception as e:                                   # noqa: BLE001
            print("  solve(inst) RAISED %s: %s" % (type(e).__name__, e))
            import traceback
            traceback.print_exc()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
