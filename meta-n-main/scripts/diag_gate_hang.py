"""Why is the gate hanging? Inspect the child's generated solver."""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, ".")
FENCE = "`" * 3


def main() -> int:
    run = sorted(Path("runs/val_kimi").glob("*/"))[-1]
    recs = [json.loads(l)
            for l in (run / "llm_io" / "outer.jsonl").open() if l.strip()]
    print("outer records:", len(recs))
    for i, r in enumerate(recs):
        head = str(r.get("response", ""))[:60].replace("\n", " ")
        print("  #%d %s" % (i, head))

    if len(recs) < 3:
        print("child script not written yet")
        return 0

    s = str(recs[2]["response"])
    m = re.search(FENCE + r"python\s*\n(.*?)" + FENCE, s, re.DOTALL)
    src = m.group(1) if m else s
    print()
    print("=== child solver (%d lines) ===" % src.count("\n"))
    print(src[:1600])

    print()
    print("=== blocking-call scan ===")
    for pat in ("input(", "sys.stdin", "stdin.read", "while True",
                "signal.alarm", "sleep(", "time.sleep"):
        n = src.count(pat)
        if n:
            print("  %-14s x%d   <-- SUSPECT" % (pat, n))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
