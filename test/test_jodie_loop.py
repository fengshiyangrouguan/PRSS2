"""JODIE loop contracts: row selection, metric bundle, tiny end-to-end smoke."""

import unittest

import numpy as np
import torch

from rpbe.compressor import RecursiveCompressor
from rpbe.config import RPBConfig
from rpbe.data.jodie import JodieData
from rpbe.maps import FixedMaps
from rpbe.records import JodieCutBuilder, NODE_CLASS
from rpbe.training.jodie_loop import (JodieNodeClassificationLoop,
                                      metric_bundle, select_trace_rows)

from test_jodie_adapter import install_adapter, make_tiny_tgn
from test_jodie_vendor import REQUIRES_NUMPY_BRIDGE


class _FakeMonitor:
    def validate_losses(self, losses, step):
        for k, v in losses.items():
            assert np.isfinite(v), (k, v)

    def validate_kf(self, kf_by_tau, dims, step):
        for tau, j in kf_by_tau.items():
            assert np.isfinite(j), (tau, j)
            assert 0.0 <= j <= dims[tau] + 1e-4, (tau, j, dims[tau])

    def alert(self, severity, code, message, **meta):
        pass  # warnings are not errors in the smoke test


class TestSelectTraceRows(unittest.TestCase):
    def test_positives_first_uses_all_positives(self):
        labels = np.array([0, 1, 0, 1, 1, 0])
        rows = select_trace_rows(labels, 2, seed=0, batch_index=0)
        self.assertEqual(rows, [1, 3])

    def test_positives_first_backfills_negatives(self):
        labels = np.array([0, 1, 0, 0, 0])
        rows = select_trace_rows(labels, 3, seed=0, batch_index=0)
        self.assertEqual(len(rows), 3)
        self.assertIn(1, rows)  # the only positive is always included
        self.assertEqual(rows, sorted(rows))

    def test_positive_first_deterministic_per_batch(self):
        labels = np.zeros(20)
        a = select_trace_rows(labels, 5, seed=7, batch_index=3)
        b = select_trace_rows(labels, 5, seed=7, batch_index=3)
        self.assertEqual(a, b)

    def test_evenly_spaced(self):
        labels = np.zeros(10)
        rows = select_trace_rows(labels, 4, seed=0, batch_index=0,
                                 mode="evenly_spaced")
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows, sorted(rows))

    def test_off_returns_empty(self):
        self.assertEqual(select_trace_rows(np.ones(5), 4, 0, 0, "off"), [])

    def test_max_roots_zero_returns_empty(self):
        self.assertEqual(select_trace_rows(np.ones(5), 0, 0, 0,
                                           "positive_first"), [])

    def test_unknown_mode_raises(self):
        with self.assertRaises(ValueError):
            select_trace_rows(np.zeros(3), 2, 0, 0, "bogus")


class TestMetricBundle(unittest.TestCase):
    def test_perfect_separation(self):
        labels = np.array([0, 0, 1, 1])
        probs = np.array([0.01, 0.02, 0.99, 0.98])
        m = metric_bundle(labels, probs)
        self.assertGreater(m["auc"], 0.99)
        self.assertGreater(m["ap"], 0.99)
        self.assertAlmostEqual(m["positives"], 2)
        self.assertAlmostEqual(m["pairs"], 4)
        self.assertAlmostEqual(m["positive_rate"], 0.5)

    def test_reversed_scores_auc_below_half(self):
        labels = np.array([0, 0, 1, 1])
        probs = np.array([0.99, 0.98, 0.01, 0.02])
        self.assertLess(metric_bundle(labels, probs)["auc"], 0.5)

    def test_single_class_auc_is_nan(self):
        m = metric_bundle(np.zeros(5), np.full(5, 0.5))
        self.assertTrue(np.isnan(m["auc"]))
        self.assertEqual(m["ap"], 0.0)


@REQUIRES_NUMPY_BRIDGE
class TestLoopSmoke(unittest.TestCase):
    """End-to-end: one train epoch, eval, and replay on tiny data, with and
    without the RPBE component attached."""

    def setUp(self):
        tgn, device, stream = make_tiny_tgn()
        sources, destinations, timestamps, edge_idxs, labels = stream
        self.tgn = tgn
        self.device = device
        self.train = JodieData(sources[:60], destinations[:60],
                               timestamps[:60], edge_idxs[:60], labels[:60])
        self.val = JodieData(sources[60:80], destinations[60:80],
                             timestamps[60:80], edge_idxs[60:80], labels[60:80])
        self.test = JodieData(sources[80:], destinations[80:],
                              timestamps[80:], edge_idxs[80:], labels[80:])
        self.stream_times = timestamps

    def _make_loop(self, rpbe=False):
        tgn = self.tgn
        from rpbe.hosts.official_tgn import MLP
        decoder = MLP(dim=8, drop=0.1).to(self.device)
        main_params = list(decoder.parameters()) + [
            p for p in tgn.parameters() if p.requires_grad]
        adapter = cut_builder = fixed_maps = rpbe_cfg = None
        if rpbe:
            cfg = RPBConfig(
                state_dims={"tjo:layer0": 8, "tjo:layer1": 8, "tjo:layer2": 8},
                own_dims={"tjo:layer0": 8, "tjo:layer1": 8, "tjo:layer2": 8},
                width_D=16, m=64, kf_min_abs=4, rpbe_seed=0,
                delta_t_scale=1.0)
            rpbe_cfg = cfg
            compressor = RecursiveCompressor(cfg).to(self.device)
            main_params += [p for p in compressor.parameters()]
            adapter = install_adapter(tgn)
            adapter.compressor = compressor
            fixed_maps = FixedMaps(cfg).to(self.device)
            endpoints = {int(e): (int(s), int(d))
                         for s, d, e in zip(self.train.sources,
                                            self.train.destinations,
                                            self.train.edge_idxs)}
            labels_tbl = {int(e): float(y)
                          for e, y in zip(self.train.edge_idxs,
                                          self.train.labels)}
            adapter.edge_tables = (endpoints, labels_tbl)
            adapter._endpoints = endpoints
            cut_builder = JodieCutBuilder((endpoints, labels_tbl),
                                          stage=NODE_CLASS, seed=0)
        seen = set()
        main_params = [p for p in main_params
                       if not (id(p) in seen or seen.add(id(p)))]
        optimizer = torch.optim.Adam(main_params, lr=3e-4)
        return JodieNodeClassificationLoop(
            tgn=tgn, decoder=decoder, optimizer=optimizer,
            device=self.device, batch_size=8, n_neighbors=4, grad_clip=5.0,
            monitor=_FakeMonitor(), seed=0, finetune_host=True,
            adapter=adapter, cut_builder=cut_builder, fixed_maps=fixed_maps,
            rpbe_cfg=rpbe_cfg, trace_roots=4)

    def test_pure_host_train_eval_replay(self):
        loop = self._make_loop(rpbe=False)
        row = loop.train_epoch(0, 0, self.train)
        self.assertIn("train_task_loss", row)
        self.assertTrue(np.isfinite(row["train_task_loss"]))
        val_row = loop.evaluate_split(self.val, reset=False)
        self.assertIn("auc", val_row)
        self.assertEqual(val_row["embedding_dims_observed"], {"source": 8})
        loop.reset_memory()
        loop.replay_split(self.train)
        loop.replay_split(self.val)
        test_row = loop.evaluate_split(self.test, reset=False)
        self.assertIn("auc", test_row)

    def test_rpbe_train_epoch_finite_kf(self):
        loop = self._make_loop(rpbe=True)
        row = loop.train_epoch(0, 0, self.train)
        self.assertIn("train_kf_loss", row)
        self.assertTrue(np.isfinite(row["train_kf_loss"]))
        self.assertGreater(row["n_batches"], 0)
        val_row = loop.evaluate_split(self.val, reset=False)
        self.assertIn("auc", val_row)
        # Evaluation must not have built any trace.
        self.assertIsNone(loop.adapter.trace)


if __name__ == "__main__":
    unittest.main()
