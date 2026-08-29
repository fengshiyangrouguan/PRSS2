"""JODIE data loading: quantile split boundary semantics + real-data anchors.

The split rules are line-for-line equivalent to the upstream
``get_data_node_classification``::

    train: ts <= val_time ; val: val_time < ts <= test_time ; test: ts > test_time

Boundary test uses ts = arange(101): np.quantile(ts, 0.70) == 70.0 and
np.quantile(ts, 0.85) == 85.0 exactly, so every comparison operator is
exercised on a member row.
"""

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from rpbe.data.jodie import JodieDataset, compute_time_statistics

REAL_DATA_DIR = Path(__file__).resolve().parents[1] / "old" / "processed_tgn_data"


def _write_synth(data_dir, n=101, seed=0):
    """Write a synthetic ml_synth.csv/.npy/_node.npy trio."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(seed)
    ts = np.arange(n, dtype=np.float64)
    u = rng.randint(1, 20, size=n).astype(np.int64)
    i = rng.randint(1, 20, size=n).astype(np.int64)
    labels = rng.randint(0, 2, size=n).astype(np.float64)
    idx = np.arange(1, n + 1, dtype=np.int64)
    pd.DataFrame({",": np.arange(n), "u": u, "i": i, "ts": ts,
                  "label": labels, "idx": idx}).to_csv(
        data_dir / "ml_synth.csv", index=False)
    np.save(data_dir / "ml_synth.npy", rng.randn(n + 1, 6).astype(np.float32))
    np.save(data_dir / "ml_synth_node.npy", np.zeros((30, 6), dtype=np.float32))
    return u, i, ts, labels, idx


class TestQuantileSplit(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        _write_synth(self._tmp.name)
        self.ds = JodieDataset("synth", data_dir=self._tmp.name)

    def test_split_sizes_at_exact_boundaries(self):
        # ts = 0..100, val_time = 70.0, test_time = 85.0
        self.assertEqual(self.ds.val_time, 70.0)
        self.assertEqual(self.ds.test_time, 85.0)
        full, train, val, test = self.ds.splits()
        self.assertEqual(full.n_interactions, 101)
        self.assertEqual(train.n_interactions, 71)   # ts <= 70
        self.assertEqual(val.n_interactions, 15)     # 70 < ts <= 85
        self.assertEqual(test.n_interactions, 15)    # ts > 85
        self.assertEqual(train.n_interactions + val.n_interactions
                         + test.n_interactions, 101)

    def test_boundary_row_ownership(self):
        # Row with ts == 70 must be in train; ts == 85 in val; ts == 86 in test.
        _, _, ts, _, idx = _write_synth(self._tmp.name)
        _, train, val, test = self.ds.splits()
        self.assertIn(71, set(train.edge_idxs))    # idx of the ts=70 row (0-based 70)
        self.assertIn(86, set(val.edge_idxs))      # idx of the ts=85 row
        self.assertIn(87, set(test.edge_idxs))     # idx of the ts=86 row

    def test_features_and_sanity(self):
        self.assertEqual(self.ds.node_features.shape, (30, 6))
        self.assertEqual(self.ds.edge_features.shape, (101 + 1, 6))
        self.assertTrue((self.ds.node_features == 0).all())  # JODIE node feats are zeros
        sc = self.ds.sanity_check()
        self.assertTrue(sc["splits_cover_full"])
        self.assertTrue(sc["timestamps_sorted"])
        self.assertTrue(sc["edge_idxs_one_based"])
        self.assertEqual(sc["node_feature_dim"], 6)
        self.assertEqual(sc["edge_feature_dim"], 6)

    def test_use_validation_false_makes_val_equals_test(self):
        ds = JodieDataset("synth", data_dir=self._tmp.name, use_validation=False)
        full, train, val, test = ds.splits()
        # Upstream: val_mask = test_mask when use_validation=False.
        self.assertEqual(val.n_interactions, test.n_interactions)
        self.assertTrue(np.array_equal(val.timestamps, test.timestamps))
        # train covers everything up to test_time.
        self.assertEqual(train.n_interactions + test.n_interactions,
                         full.n_interactions)


class TestTimeStatistics(unittest.TestCase):
    def test_known_stream(self):
        # src deltas: node1 [10-0, 20-10], node2 [5-0] -> [10, 10, 5].
        # dst deltas: node3 [10-0, 5-10], node4 [20-0] -> [10, -5, 20].
        # Negative deltas are kept as-is (upstream does not clamp).
        sources = np.array([1, 1, 2])
        destinations = np.array([3, 4, 3])
        timestamps = np.array([10.0, 20.0, 5.0])
        ms, ss, md, sd = compute_time_statistics(sources, destinations, timestamps)
        self.assertAlmostEqual(ms, 25.0 / 3)
        self.assertAlmostEqual(md, 25.0 / 3)  # (10 + 20 + (-5)) / 3
        self.assertAlmostEqual(ss, float(np.std([10, 10, 5])))
        self.assertAlmostEqual(sd, float(np.std([10, -5, 20])))


@unittest.skipUnless(
    (REAL_DATA_DIR / "ml_wikipedia.csv").exists(),
    "real JODIE wikipedia data not present (old/processed_tgn_data)")
class TestRealWikipedia(unittest.TestCase):
    """Hard anchor: split counts and positive counts must match the v1 anchors."""

    def test_anchor_counts(self):
        ds = JodieDataset("wikipedia", data_dir=str(REAL_DATA_DIR))
        full, train, val, test = ds.splits()
        self.assertEqual(full.n_interactions, 157474)
        self.assertEqual(train.n_interactions, 110232)
        self.assertEqual(val.n_interactions, 23621)
        self.assertEqual(test.n_interactions, 23621)
        # Positive (state-changing) rows are very sparse.
        self.assertEqual(int((train.labels > 0.5).sum()), 156)
        self.assertEqual(int((val.labels > 0.5).sum()), 17)
        self.assertEqual(int((test.labels > 0.5).sum()), 44)

    def test_anchor_dimensions(self):
        ds = JodieDataset("wikipedia", data_dir=str(REAL_DATA_DIR))
        self.assertEqual(ds.node_features.shape, (9228, 172))   # 0 = padding
        self.assertEqual(ds.edge_features.shape, (157474 + 1, 172))
        # Node feature rows past 9227 are the padding row only.
        self.assertEqual(ds.full.n_unique_nodes, 9227)

    def test_time_stats_finite(self):
        ds = JodieDataset("wikipedia", data_dir=str(REAL_DATA_DIR))
        ms, ss, md, sd = ds.time_stats()
        for v in (ms, ss, md, sd):
            self.assertTrue(np.isfinite(v))
            self.assertGreater(v, 0.0)


if __name__ == "__main__":
    unittest.main()
