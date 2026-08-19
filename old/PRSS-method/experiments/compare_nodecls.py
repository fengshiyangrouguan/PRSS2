#!/usr/bin/env python3
import json, sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for name in ("official_reference", "vanilla_matched", "prss_matched"):
    p = root / name / "summary.json"
    if p.exists():
        d = json.load(open(p))
        t = d.get("test", {})
        rows.append((name, t.get("auc"), t.get("ap"), t.get("nll"), d.get("best_epoch")))
print("model\ttest_auc\ttest_ap\ttest_nll\tbest_epoch")
for r in rows:
    print("%s\t%s\t%s\t%s\t%s" % r)
if (root / "vanilla_matched" / "summary.json").exists() and (root / "prss_matched" / "summary.json").exists():
    v = json.load(open(root / "vanilla_matched" / "summary.json"))["test"]
    p = json.load(open(root / "prss_matched" / "summary.json"))["test"]
    print("\nPRSS - vanilla:")
    print("delta_auc=%.8f" % (p["auc"] - v["auc"]))
    print("delta_ap=%.8f" % (p["ap"] - v["ap"]))
    print("delta_nll=%.8f (negative is better)" % (p["nll"] - v["nll"]))
