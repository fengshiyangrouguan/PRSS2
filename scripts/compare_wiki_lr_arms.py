#!/usr/bin/env python3
"""Comparison audit for the 9 Wiki-LR-Binary configs (spec §7).

Reads each config's ``config.json`` and ``summary.json`` and asserts that every
config differs from R0 ONLY on its allowed axis (aux_kind / use_parent /
mispaired / context_mode / variant); all task/architecture/plan fields
(bs, n_neighbors, n_layers, kf group, threshold, data, negatives manifest) must
be identical to R0.  Writes ``comparison_audit.json`` in the run root.

Usage:
    python -m scripts.compare_wiki_lr_arms --out outputs/wiki_lr_binary/seed0
"""

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

ALLOWED = {
    "P0": {"aux_kind"}, "P1": {"aux_kind"}, "P2": {"aux_kind"},
    "S1": {"use_parent"}, "S2": {"mispaired"},
    "C1": {"context_mode"}, "B1": {"variant"}, "E1": {"variant"},
}
FIXED = ["data", "bs", "n_neighbors", "n_layers", "kf_group_batches",
         "kf_min_trees", "lambda_kf", "group_plan_sha", "maps_sha"]
CLI_FIXED = ["negatives", "group_plan", "bs", "n_neighbors", "n_layers",
             "kf_group_batches", "kf_min_trees", "rpbe_seed"]


def main():
    p = argparse.ArgumentParser("compare 9 arms")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    root = Path(args.out)
    r0 = json.load(open(root / "R0" / "config.json"))

    rows = []
    ok = True
    for cid in ["R0", "P0", "P1", "P2", "S1", "S2", "C1", "B1", "E1"]:
        cfg = json.load(open(root / cid / "config.json"))
        if cid == "R0":
            rows.append({"config": "R0", "role": "canonical", "ok": True})
            continue
        changed = {k for k in ("aux_kind", "use_parent", "mispaired",
                               "context_mode", "variant")
                   if cfg.get(k) != r0.get(k)}
        fixed_bad = [k for k in FIXED if cfg.get(k) != r0.get(k)]
        cli0, clic = r0.get("cli", {}), cfg.get("cli", {})
        fixed_bad += ["cli." + k for k in CLI_FIXED
                      if clic.get(k) != cli0.get(k)]
        allowed = ALLOWED[cid]
        good = changed == allowed and not fixed_bad
        ok = ok and good
        sum_path = root / cid / "summary.json"
        summ = json.load(open(sum_path)) if sum_path.exists() else {}
        rows.append({
            "config": cid,
            "changed_vs_R0": sorted(changed),
            "allowed": sorted(allowed),
            "fixed_field_diffs": fixed_bad,
            "ok": good,
            "best_ap_all": summ.get("best_ap_all"),
            "stop_reason": summ.get("stop_reason"),
        })
    audit = {"ok": ok, "rows": rows,
             "note": "each arm differs from R0 only on its allowed axis"}
    (root / "comparison_audit.json").write_text(
        json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
