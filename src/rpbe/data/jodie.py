"""JODIE node-classification dataset wrapper.

Reads ``ml_{name}.csv`` / ``ml_{name}.npy`` / ``ml_{name}_node.npy`` from an
explicit ``data_dir``.  The quantile split is line-for-line equivalent to the
upstream ``utils/data_processing.get_data_node_classification`` (which
hardcodes ``./data/`` and is therefore not vendored)::

    val_time, test_time = np.quantile(ts, [0.70, 0.85])
    train: ts <= val_time
    val:   val_time < ts <= test_time
    test:  ts > test_time

``use_validation=False`` keeps the upstream provenance semantics (val == test
for official-anchor reproduction only; the paper-facing protocol always uses
``use_validation=True``).
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class JodieData:
    """Mirror of the upstream ``Data`` class, for replay/eval iteration."""

    sources: np.ndarray
    destinations: np.ndarray
    timestamps: np.ndarray
    edge_idxs: np.ndarray
    labels: np.ndarray

    n_interactions: int = field(init=False)
    unique_nodes: set = field(init=False)
    n_unique_nodes: int = field(init=False)

    def __post_init__(self):
        self.n_interactions = len(self.sources)
        self.unique_nodes = set(self.sources) | set(self.destinations)
        self.n_unique_nodes = len(self.unique_nodes)

    def slice(self, start: int, stop: Optional[int] = None):
        """Sub-stream by index range; used for smoke-test caps."""
        if stop is None:
            stop = start
            start = 0
        return JodieData(self.sources[start:stop], self.destinations[start:stop],
                         self.timestamps[start:stop], self.edge_idxs[start:stop],
                         self.labels[start:stop])


def compute_time_statistics(sources, destinations, timestamps):
    """Per-stream time-shift statistics, identical to the upstream helper."""
    last_timestamp_sources = dict()
    last_timestamp_dst = dict()
    all_timediffs_src = []
    all_timediffs_dst = []
    for k in range(len(sources)):
        source_id = sources[k]
        dest_id = destinations[k]
        c_timestamp = timestamps[k]
        if source_id not in last_timestamp_sources.keys():
            last_timestamp_sources[source_id] = 0
        if dest_id not in last_timestamp_dst.keys():
            last_timestamp_dst[dest_id] = 0
        all_timediffs_src.append(c_timestamp - last_timestamp_sources[source_id])
        all_timediffs_dst.append(c_timestamp - last_timestamp_dst[dest_id])
        last_timestamp_sources[source_id] = c_timestamp
        last_timestamp_dst[dest_id] = c_timestamp
    assert len(all_timediffs_src) == len(sources)
    assert len(all_timediffs_dst) == len(sources)
    return (float(np.mean(all_timediffs_src)), float(np.std(all_timediffs_src)),
            float(np.mean(all_timediffs_dst)), float(np.std(all_timediffs_dst)))


class JodieDataset:
    """One JODIE node-classification dataset with quantile train/val/test splits."""

    def __init__(self, name: str = "wikipedia", data_dir: Optional[str] = None,
                 use_validation: bool = True, seed: int = 2020):
        self.name = name
        self.data_dir = data_dir
        self.use_validation = use_validation
        self.seed = seed
        csv_path = _join(data_dir, "ml_{}.csv".format(name))
        graph_df = pd.read_csv(csv_path)
        self.edge_features = np.load(_join(data_dir, "ml_{}.npy".format(name)))
        self.node_features = np.load(_join(data_dir, "ml_{}_node.npy".format(name)))

        val_time, test_time = list(np.quantile(graph_df.ts, [0.70, 0.85]))
        self.val_time = float(val_time)
        self.test_time = float(test_time)

        sources = graph_df.u.values
        destinations = graph_df.i.values
        edge_idxs = graph_df.idx.values
        labels = graph_df.label.values
        timestamps = graph_df.ts.values

        self.full = JodieData(sources, destinations, timestamps, edge_idxs, labels)

        train_mask = timestamps <= val_time if use_validation else timestamps <= test_time
        test_mask = timestamps > test_time
        val_mask = np.logical_and(timestamps <= test_time, timestamps > val_time) \
            if use_validation else test_mask

        self.train = JodieData(
            sources[train_mask], destinations[train_mask], timestamps[train_mask],
            edge_idxs[train_mask], labels[train_mask])
        self.val = JodieData(
            sources[val_mask], destinations[val_mask], timestamps[val_mask],
            edge_idxs[val_mask], labels[val_mask])
        self.test = JodieData(
            sources[test_mask], destinations[test_mask], timestamps[test_mask],
            edge_idxs[test_mask], labels[test_mask])

        # Deterministic split tie-breaking only (unused by the protocol, kept
        # for upstream parity).
        np.random.seed(seed)

    def splits(self) -> Tuple[JodieData, JodieData, JodieData, JodieData]:
        """Return (full, train, val, test)."""
        return self.full, self.train, self.val, self.test

    def time_stats(self) -> Tuple[float, float, float, float]:
        """(mean_src, std_src, mean_dst, std_dst) over the full stream."""
        return compute_time_statistics(
            self.full.sources, self.full.destinations, self.full.timestamps)

    def sanity_check(self) -> Dict:
        checks = {
            "name": self.name,
            "data_dir": self.data_dir,
            "use_validation": self.use_validation,
            "val_time": self.val_time,
            "test_time": self.test_time,
            "n_interactions": self.full.n_interactions,
            "n_unique_nodes": self.full.n_unique_nodes,
            "node_feature_dim": int(self.node_features.shape[1]),
            "edge_feature_dim": int(self.edge_features.shape[1]),
            "node_feature_rows": int(self.node_features.shape[0]),
            "edge_feature_rows": int(self.edge_features.shape[0]),
            "train_n": self.train.n_interactions,
            "val_n": self.val.n_interactions,
            "test_n": self.test.n_interactions,
            "train_positives": int((self.train.labels > 0.5).sum()),
            "val_positives": int((self.val.labels > 0.5).sum()),
            "test_positives": int((self.test.labels > 0.5).sum()),
            "splits_cover_full": (self.train.n_interactions + self.val.n_interactions
                                  + self.test.n_interactions) == self.full.n_interactions,
            "timestamps_sorted": bool(
                (np.diff(self.full.timestamps) >= 0).all()),
            "edge_idxs_one_based": bool((self.full.edge_idxs >= 1).all()),
        }
        return checks


def _join(data_dir, filename):
    if data_dir is None:
        return filename
    import os
    return os.path.join(data_dir, filename)
