#!/usr/bin/env python3
"""Per-demo tier table: how many times each demo completed the 3-cycle task.

Reads an UNFILTERED tiered_eval log (the one with `[demo_N] tier=X/3 ...`
lines) and writes two things:

    tier_by_demo.csv   demo, then one column per checkpoint
    tier_summary.csv   per checkpoint: n0/n1/n2/n3, >=1, >=2, strict, weighted

This is the table whose absence cost us every Stage8 per-demo identity --
see ../results/PER_DEMO_RECOVERY.md.

Usage:  python tier_table.py <rollout_full.log> [<outdir>]
"""
import csv
import os
import re
import sys
from collections import defaultdict

SEC = re.compile(r"^===\s*(.+?)\s*::\s*(\S+)\s*===\s*$")
DEMO = re.compile(r"\[demo_(\d+)\]\s*tier=(\d)/3")
STRICT = re.compile(r"tier=(\d)/3.*strict=(\w+)")


def main():
    path = sys.argv[1]
    outdir = sys.argv[2] if len(sys.argv) > 2 else "."
    per = defaultdict(dict)          # tag -> {demo: tier}
    strict = defaultdict(dict)       # tag -> {demo: bool}
    tag = None
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = SEC.match(line.strip())
            if m:
                tag = m.group(2)
                continue
            if tag is None:
                continue
            for m in DEMO.finditer(line):
                per[tag][int(m.group(1))] = int(m.group(2))
            for m in STRICT.finditer(line):
                strict[tag][int(m.group(1))] = (m.group(2) == "True")

    if not per:
        sys.exit(f"no per-demo lines in {path} -- was the log filtered by a grep?")

    tags = list(per)
    demos = sorted({d for v in per.values() for d in v})
    with open(os.path.join(outdir, "tier_by_demo.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["demo"] + tags)
        for d in demos:
            w.writerow([d] + [per[t].get(d, "") for t in tags])

    with open(os.path.join(outdir, "tier_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["checkpoint", "n_demos", "n_0", "n_1", "n_2", "n_3",
                    "ge1", "ge2", "strict", "sum_tier", "weighted_success_pct"])
        for t in tags:
            v = per[t]
            n = len(v)
            c = [sum(1 for x in v.values() if x == k) for k in range(4)]
            st = sum(1 for d, s in strict[t].items() if s and v.get(d) == 3)
            wsum = c[1] + 2 * c[2] + 3 * c[3]
            w.writerow([t, n] + c + [c[1] + c[2] + c[3], c[2] + c[3], st,
                                     wsum, "%.1f" % (100.0 * wsum / (3 * n))])
    print("wrote tier_by_demo.csv and tier_summary.csv")
    for t in tags:
        v = per[t]
        c = [sum(1 for x in v.values() if x == k) for k in range(4)]
        print("  %-22s n=%d  0/1/2/3 = %d/%d/%d/%d  weighted=%.1f%%"
              % (t, len(v), *c, 100.0 * (c[1] + 2*c[2] + 3*c[3]) / (3 * len(v))))


if __name__ == "__main__":
    main()
