#!/usr/bin/env python3
"""Generate the shared Wiki-LR-Binary compact negative manifest once.

Usage:
    python -m scripts.build_wiki_binary_negatives \
        --data-dir datasets \
        --out datasets/wiki_lr_binary_negatives.json \
        --protocol-seed 20260908

Reads the official tgbl-wiki val/test negative pools, selects one stable,
history-priority negative per positive (spec §2.2), applies the collision
gate, and writes the manifest + report.  All arms and model seeds later read
this same file; it is never regenerated per run.
"""

import argparse
import json
import os
import sys
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rpbe.data.tgb_link import TGBLinkDataset
from rpbe.data.wiki_binary_negatives import build_compact_negatives, \
    manifest_sha256


def main():
    p = argparse.ArgumentParser("build Wiki-LR-Binary compact negatives")
    p.add_argument("--data-dir", default="datasets")
    p.add_argument("--out", default="datasets/wiki_lr_binary_negatives.json")
    p.add_argument("--protocol-seed", type=int, default=20260908)
    args = p.parse_args()

    ds = TGBLinkDataset(root=args.data_dir)
    payload = build_compact_negatives(ds, protocol_seed=args.protocol_seed)

    sha = manifest_sha256(payload)
    counts = payload["_meta"]["counts"]
    payload["_meta"]["sha256"] = sha
    payload["_meta"]["positive_counts"] = {
        "val": int(len(ds.val.sources)),
        "test": int(len(ds.test.sources)),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(payload, f)
    print(json.dumps(payload["_meta"], indent=2))
    print("manifest sha256:", sha)
    print("wrote", out, "rows_val=%d rows_test=%d" %
          (len(payload["val"]), len(payload["test"])))

    # real-data gate (spec §8): history coverage and zero collisions
    assert counts["val_hist"] >= 5000, \
        "val historical-negative coverage {} < 5000".format(
            counts["val_hist"])
    assert counts["test_hist"] >= 5000, \
        "test historical-negative coverage {} < 5000".format(
            counts["test_hist"])
    assert counts["collisions"] == 0, \
        "collision gate failed: {}".format(counts["collisions"])
    print("gates OK (hist>=5000, collisions=0)")


if __name__ == "__main__":
    main()
