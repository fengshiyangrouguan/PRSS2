#!/usr/bin/env python3
"""Generate fixed, time-stratified query-id sets shared by all arms/seeds.

One file of val query indices (default 1000, equal quantiles over time) and
one of test query indices (default 2000).  All arms and seeds use the SAME
indices so their ``sampled_query_*_mrr`` are directly comparable.

Usage:
    python -m scripts.make_query_set --data-dir datasets \
        --n-val 1000 --n-test 2000 --out datasets/tgb_wiki_query_sets.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rpbe.data.tgb_link import TGBLinkDataset


def parse_args():
    p = argparse.ArgumentParser("query-id set generator")
    p.add_argument("--data-dir", default="datasets")
    p.add_argument("--n-val", type=int, default=1000)
    p.add_argument("--n-test", type=int, default=2000)
    p.add_argument("--out", default="datasets/tgb_wiki_query_sets.json")
    return p.parse_args()


def stratified(ids_sorted_by_time, n):
    """Pick ``n`` indices spread evenly over the chronological stream."""
    m = len(ids_sorted_by_time)
    if n >= m:
        return list(ids_sorted_by_time)
    step = m / float(n)
    return [int(ids_sorted_by_time[int(round(j * step))])
            for j in range(n)]


def main():
    args = parse_args()
    ds = TGBLinkDataset(root=args.data_dir)
    # split indices are already chronological (row order == time order within
    # each split), so stratified sampling over row order is time-stratified.
    _, _, val, test = ds.splits()
    val_ids = stratified(list(range(len(val.sources))), args.n_val)
    test_ids = stratified(list(range(len(test.sources))), args.n_test)
    out = {"n_val_total": len(val.sources),
           "n_test_total": len(test.sources),
           "val_query_ids": val_ids,
           "test_query_ids": test_ids}
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f)
    print("wrote {} (val {} test {})".format(path, len(val_ids),
                                             len(test_ids)))


if __name__ == "__main__":
    main()
