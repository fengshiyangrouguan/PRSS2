"""Pin the gen2 crash: the depth-2 pre_process has a syntax error.

Log: "pre_process (d=2) failed validation: Syntax error: unterminated string
literal (detected at line 19) - skipping". The depth-3 solver calls the
depth-2 layer's helper; if that layer is skipped the name is undefined on every
instance -> 160 errors, 0 timeouts.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
FENCE = "`" * 3


def blocks(text):
    s = str(text)
    out = {}
    for tag in ("rationale", "pre_process"):
        k = s.find(FENCE + tag)
        if k < 0:
            continue
        nl = s.find("\n", k)
        e = s.find(FENCE, nl)
        out[tag] = s[nl + 1:e]
    libs = {}
    i = 0
    while True:
        k = s.find(FENCE + "solver_lib:", i)
        if k < 0:
            break
        nl = s.find("\n", k)
        name = s[k + len(FENCE) + len("solver_lib:"):nl].strip()
        e = s.find(FENCE, nl)
        libs[name] = s[nl + 1:e]
        i = e + len(FENCE)
    out["libs"] = libs
    return out


def validate(label, src):
    import ast
    try:
        ast.parse(src)
        print("  %-22s VALID (%d chars)" % (label, len(src)))
        return True
    except SyntaxError as e:
        print("  %-22s SYNTAX ERROR: %s (line %s)" % (label, e.msg, e.lineno))
        return False


def main() -> int:
    run = sorted(Path("runs/val_ds").glob("*/"))[-1]
    recs = [json.loads(l)
            for l in (run / "llm_io" / "outer.jsonl").open() if l.strip()]

    d2 = blocks(recs[3]["response"])      # Omega @ depth 2
    d3 = blocks(recs[7]["response"])      # Omega @ depth 3

    print("=== validate Omega's pre_process blocks ===")
    validate("depth-2 pre_process", d2.get("pre_process") or "")
    validate("depth-3 pre_process", d3.get("pre_process") or "")
    print()
    print("  depth-2 solver_libs:", list(d2["libs"]))
    print("  depth-3 solver_libs:", list(d3["libs"]))

    print()
    print("=== depth-2 pre_process as emitted (around line 19) ===")
    src = d2.get("pre_process") or ""
    for i, line in enumerate(src.splitlines(), 1):
        mark = " <<<" if 15 <= i <= 22 else ""
        print("  %2d| %s%s" % (i, line[:96], mark))

    # --- what does the depth-3 solver actually reference? ---
    print()
    print("=== depth-3 solver's helper references ===")
    csrc = str(recs[8]["response"])
    for name in list(d2["libs"]) + list(d3["libs"]):
        n = csrc.count(name)
        if n:
            print("  %-24s referenced x%d  (defined at %s)" % (
                name, n,
                "d2" if name in d2["libs"] else "-"
                + ("d3" if name in d3["libs"] else "")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
