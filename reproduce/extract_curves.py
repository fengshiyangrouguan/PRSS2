#!/usr/bin/env python3
"""Turn a trainer log into a step-indexed curve file: train loss + val loss.

Writes `curves.csv` with one row per 1000 training steps:

    step,train_loss,val_loss,n_train_points

`train_loss` is the MEAN of every logged training loss inside that 1000-step
block (the trainer logs every --log-every steps, i.e. 20 samples per block at
--log-every 50); `val_loss` is the `[eval @ opt N] val action loss X` line at
that step, blank if that step has no eval.

Usage:  python extract_curves.py <train.log> [<out.csv>] [--bin 1000]
"""
import re
import sys
from collections import defaultdict

TRAIN = re.compile(r"(?:opt|step)\s+(\d+)/\d+[^|]*\|\s*loss\s+([0-9.]+)")
VAL = re.compile(r"\[eval @ opt (\d+)\]\s*val action loss\s+([0-9.]+)")


def parse(path):
    train, val = [], {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = TRAIN.search(line)
            if m:
                train.append((int(m.group(1)), float(m.group(2))))
            m = VAL.search(line)
            if m:
                val[int(m.group(1))] = float(m.group(2))
    return train, val


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    path = args[0]
    out = args[1] if len(args) > 1 else "curves.csv"
    binw = 1000
    if "--bin" in sys.argv:
        binw = int(sys.argv[sys.argv.index("--bin") + 1])

    train, val = parse(path)
    if not train:
        sys.exit(f"no training-loss lines matched in {path}")
    bins = defaultdict(list)
    for step, loss in train:
        bins[(step // binw) * binw].append(loss)

    steps = sorted(set(bins) | {s for s in val if s % binw == 0})
    with open(out, "w", newline="\n") as f:
        f.write("step,train_loss,val_loss,n_train_points\n")
        for s in steps:
            b = bins.get(s, [])
            f.write("%d,%s,%s,%d\n" % (
                s,
                ("%.6f" % (sum(b) / len(b))) if b else "",
                ("%.6f" % val[s]) if s in val else "",
                len(b)))

    print(f"train points: {len(train)}   val points: {len(val)}   rows: {len(steps)}")
    print(f"wrote {out}")
    for s in steps[:3] + steps[-3:]:
        b = bins.get(s, [])
        print("  %6d  train=%-9s val=%s" % (
            s, ("%.4f" % (sum(b)/len(b))) if b else "-",
            ("%.4f" % val[s]) if s in val else "-"))


if __name__ == "__main__":
    main()
