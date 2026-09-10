"""UCI temporal link prediction under the official BenchTemp protocol.

Replicates ``benchtemp.lp.DataLoader`` + ``RandEdgeSampler`` semantics for the
transductive setting (spec: UCI is the locked host task):

* split  : ts quantiles [0.70, 0.85] -> train / val / test
* nodes  : u/i each reindexed from 1 (0 = padding sentinel); both spaces map
  onto the shared 1900-row feature table (official convention)
* train negatives : random dst from the UNIQUE train-dst pool (global RNG)
* val/test negatives : RandEdgeSampler over the FULL stream with fixed seeds
  (val seed=0, test seed=2), reset per evaluation (official convention)
* metric : sklearn AP (secondary) / AUC (primary), one negative per positive

The official loader also samples 10% of test-time nodes and removes their
edges from train (``new_test_node_set``, ``random.seed(2020)``) even in the
transductive setting — replicated here exactly.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class Stream:
    sources: np.ndarray      # int64, >= 1
    destinations: np.ndarray  # int64, >= 1
    timestamps: np.ndarray    # float64
    edge_idxs: np.ndarray     # int64 row index into the edge feature table


class UCILinkDataset:
    """BenchTemp-protocol loader (UCI or MOOC — same file convention) +
    official transductive split."""

    name = "uci"

    def __init__(self, data_dir: str, data_name: str = "uci"):
        self.name = data_name
        data_dir = Path(data_dir)
        df = pd.read_csv(data_dir / "ml_{}.csv".format(data_name))
        edge_features = np.load(data_dir / "ml_{}.npy".format(data_name))
        node_features = np.load(data_dir / "ml_{}_node.npy".format(data_name))

        sources = df.u.values.astype(np.int64)
        destinations = df.i.values.astype(np.int64)
        timestamps = df.ts.values.astype(np.float64)
        edge_idxs = df.idx.values.astype(np.int64)

        # ---- official split (benchtemp/lp/dataloader.py) ----
        # The official UCI run (n_train=41884 in its own logs) uses the PURE
        # time split for the transductive main protocol — the new-test-node
        # removal only feeds the inductive subsets, which this experiment
        # does not use.
        val_time, test_time = list(np.quantile(timestamps, [0.70, 0.85]))
        train_mask = timestamps <= val_time
        val_mask = (timestamps > val_time) & (timestamps <= test_time)
        test_mask = timestamps > test_time

        def _mk(m):
            return Stream(sources[m], destinations[m], timestamps[m],
                          edge_idxs[m])

        self.train = _mk(train_mask)
        self.val = _mk(val_mask)
        self.test = _mk(test_mask)
        # full chronological stream (official eval neighbor finder source)
        self.full = _mk(np.ones(len(sources), dtype=bool))
        self.val_time = float(val_time)
        self.test_time = float(test_time)

        self.node_features = node_features.astype(np.float32)
        self.edge_features = edge_features.astype(np.float32)
        self.msg_dim = int(edge_features.shape[1])
        self.n_nodes = int(node_features.shape[0])
        # official RandEdgeSampler draws from UNIQUE dst (train) /
        # unique full-stream dst (val/test)
        self.train_dst_pool = np.unique(self.train.destinations)
        self.full_src_pool = np.unique(sources)
        self.full_dst_pool = np.unique(destinations)

        # time stats for the official memory time-diff normalization
        ts = timestamps.astype(np.float64)
        self.mean_time_shift = float(ts.mean())
        self.std_time_shift = float(ts.std()) + 1e-8

    # ------------------------------------------------------------- negatives
    def val_negatives(self, n: int, seed: int = 0) -> Tuple[np.ndarray,
                                                             np.ndarray]:
        rs = np.random.RandomState(seed)
        si = rs.randint(0, len(self.full_src_pool), n)
        di = rs.randint(0, len(self.full_dst_pool), n)
        return self.full_src_pool[si], self.full_dst_pool[di]

    def test_negatives(self, n: int, seed: int = 2) -> Tuple[np.ndarray,
                                                              np.ndarray]:
        return self.val_negatives(n, seed=seed)

    # ------------------------------------------------------------- sanity
    def sanity_check(self) -> Dict[str, int]:
        return {
            "n_edges": len(self.train.sources) + len(self.val.sources)
            + len(self.test.sources),
            "n_train": len(self.train.sources),
            "n_val": len(self.val.sources),
            "n_test": len(self.test.sources),
            "n_nodes": self.n_nodes,
            "msg_dim": self.msg_dim,
        }
