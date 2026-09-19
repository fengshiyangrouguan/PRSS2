"""Confirm: BOTH layer helpers contain the blocked `import sys`."""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, ".")
FENCE = "`" * 3


def libs(text):
    s = str(text)
    out = {}
    i = 0
    while True:
        k = s.find(FENCE + "solver_lib:", i)
        if k < 0:
            break
        nl = s.find("\n", k)
        name = s[k + len(FENCE) + len("solver_lib:"):nl].strip()
        e = s.find(FENCE, nl)
        out[name] = s[nl + 1:e].strip()
        i = e + len(FENCE)
    return out


def main() -> int:
    run = sorted(Path("runs/val_ds").glob("*/"))[-1]
    recs = [json.loads(l)
            for l in (run / "llm_io" / "outer.jsonl").open() if l.strip()]

    for label, idx, depth in (("depth-2 (Omega iter 1)", 3, 2),
                              ("depth-3 (Omega iter 2)", 7, 3)):
        bl = libs(recs[idx]["response"])
        print("=== %s ===" % label)
        for name, body in bl.items():
            imports = re.findall(r"^\s*(?:import|from)\s+(\w+)", body, re.M)
            print("  helper '%s': %d chars, imports=%s" % (
                name, len(body), imports))
            print("    head:", body.splitlines()[0][:80] if body else "")
            for ln in body.splitlines():
                if ln.strip().startswith(("import ", "from ")):
                    print("      >>", ln.strip())
        print()

    # what does the depth-3 pre_process TELL the solver to do?
    s = str(recs[7]["response"])
    k = s.find(FENCE + "pre_process")
    nl = s.find("\n", k)
    e = s.find(FENCE, nl)
    pre = s[nl + 1:e]
    print("=== depth-3 pre_process advertises ===")
    for ln in pre.splitlines():
        if "solve_bin_packing" in ln:
            print("  ", ln.strip()[:110])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
