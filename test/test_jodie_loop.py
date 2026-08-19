"""JODIE loop: root selection, metric bundle, and an end-to-end smoke run.

Root selection and metric math are pure numpy and run locally; the loop smoke
test drives the vendored TGN host and needs the torch<->numpy bridge, so it
runs on the GPU box (skipped locally, same as test_jodie_vendor/adapter).
"""

import unittest

import numpy as np
import torch

from prss.data.jodie import JodieData
from prss.hosts.jodie_bridge import JodieNodeClassificationBridge
from prss.training.jodie_loop import (JodieNodeClassificationLoop,
                                      metric_bundle, select_trace_rows)

from test_jodie_adapter import (make_tiny_prss, make_tiny_tgn,
                                install_adapter)
from test_jodie_vendor import REQUIRES_NUMPY_BRIDGE


class _FakeMonitor:
    """Minimal MonitorWriter stand-in for the smoke test."""

    def validate_losses(self, losses, step):
        for v in losses.values():
            if not np.isfinite(v):
                raise AssertionError(f"non-finite loss at step {step}: {losses}")


class TestSelectTraceRows(unittest.TestCase):
    """B1 hook semantics: positives first, then deterministic negatives."""

    def test_positives_first_uses_all_positives(self):
        labels = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
        rows = select_trace_rows(labels, max_roots=8, seed=0, batch_index=0,
                                 mode="positive_first")
        # All 3 positives chosen first, then negatives fill up to max_roots=8
        # (final list is row-sorted, which is what the adapter consumes).
        self.assertEqual(rows, [0, 1, 2, 3, 4])
        self.assertEqual(set(rows), {0, 1, 2, 3, 4})
        self.assertEqual(set(rows) & {0, 2, 4}, {0, 2, 4})  # all positives kept

    def test_positives_first_backfills_negatives(self):
        labels = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        rows = select_trace_rows(labels, max_roots=3, seed=0, batch_index=0,
                                 mode="positive_first")
        self.assertEqual(rows[0], 0)
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(0 <= r < 5 for r in rows))
        self.assertEqual(rows, sorted(rows))

    def test_positive_first_deterministic_per_batch(self):
        labels = np.zeros(10)
        a = select_trace_rows(labels, 4, seed=0, batch_index=3,
                              mode="positive_first")
        b = select_trace_rows(labels, 4, seed=0, batch_index=3,
                              mode="positive_first")
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

    def test_nll_and_components(self):
        labels = np.array([0, 1])
        probs = np.array([0.75, 0.25])
        m = metric_bundle(labels, probs)
        expected = -(0 * np.log(0.75) + 1 * np.log(0.25)) / 2 - (1 * np.log(0.25) + 0 * np.log(0.75)) / 2
        self.assertAlmostEqual(m["nll"], float(expected))
        self.assertAlmostEqual(m["positive_nll"], float(-np.log(0.25)))
        self.assertAlmostEqual(m["negative_nll"], float(-np.log(0.25)))

    def test_single_class_auc_is_nan(self):
        m = metric_bundle(np.zeros(5), np.full(5, 0.5))
        self.assertTrue(np.isnan(m["auc"]))
        self.assertEqual(m["ap"], 0.0)


@REQUIRES_NUMPY_BRIDGE
class TestLoopSmoke(unittest.TestCase):
    """End-to-end: one train epoch, eval, replay, and audits on tiny data."""

    def setUp(self):
        tgn, device, stream = make_tiny_tgn()
        sources, destinations, timestamps, edge_idxs, labels = stream
        self.tgn = tgn
        self.device = device
        # Train = rows 0..59, val = 60..79, test = 80..99: timestamps are
        # sorted, so val/test continue the train stream (upstream semantics),
        # never rewinding into the past (which trips the memory monotonicity
        # assert). The test split lets replay(train)+replay(val) flow straight
        # into evaluate(test) without replaying anything twice.
        self.train = JodieData(sources[:60], destinations[:60],
                               timestamps[:60], edge_idxs[:60], labels[:60])
        self.val = JodieData(sources[60:80], destinations[60:80],
                             timestamps[60:80], edge_idxs[60:80], labels[60:80])
        self.test = JodieData(sources[80:], destinations[80:],
                              timestamps[80:], edge_idxs[80:], labels[80:])
        config, prss = make_tiny_prss(variant="spectral")
        adapter = install_adapter(tgn, prss)
        logt = np.log1p(timestamps.astype(np.float64))
        bridge = JodieNodeClassificationBridge(
            adapter, prss, log_time_mean=float(logt.mean()),
            log_time_std=float(logt.std() + 1e-8))
        from prss.hosts.official_tgn import MLP
        decoder = MLP(dim=8, drop=0.1).to(device)
        main_params = list(decoder.parameters()) + list(prss.parameters())
        unrestricted = list(prss.unrestricted.parameters())
        seen = {id(p) for p in unrestricted}
        main_params = [p for p in main_params if id(p) not in seen]
        optimizer = torch.optim.Adam(main_params, lr=3e-4)
        self.loop = JodieNodeClassificationLoop(
            tgn=tgn, decoder=decoder, adapter=adapter, bridge=bridge,
            prss_core=prss, optimizer=optimizer,
            unrestricted_optimizer=torch.optim.Adam(unrestricted, lr=3e-4),
            device=device, batch_size=8, n_neighbors=4, grad_clip=5.0,
            lambda_resp=1.0, lambda_spec=0.1, trace_roots=4,
            trace_mode="positive_first", spectral_warmup=2,
            spectral_interval=2, monitor=_FakeMonitor(), seed=0)

    def test_train_epoch_runs_and_returns_metrics(self):
        row = self.loop.train_epoch(0, 0, self.train)
        for key in ("train_task_loss", "train_response_loss",
                    "train_spectral_loss", "train_unrestricted_loss",
                    "n_batches", "global_step"):
            self.assertIn(key, row)
        self.assertTrue(np.isfinite(row["train_task_loss"]))
        self.assertGreater(row["n_batches"], 0)
        self.assertGreater(row["global_step"], 0)
        self.assertIn("auc", row["train"])
        self.assertIn("ap", row["train"])

    def test_evaluate_and_audits(self):
        self.loop.train_epoch(0, 0, self.train)
        before_counts, before_r = self.loop.audit_before()
        val_row = self.loop.evaluate_split(self.val, reset=False)
        self.loop.audit_after(before_counts, before_r, "validation")
        self.loop.reenable_spectral()
        self.assertIn("auc", val_row)
        self.assertEqual(val_row["embedding_dims_observed"], {"source": 8})

    def test_replay_then_test_with_isolation(self):
        self.loop.train_epoch(0, 0, self.train)
        self.loop.reset_memory()
        self.loop.replay_split(self.train)
        self.loop.replay_split(self.val)
        before_counts, before_r = self.loop.audit_before()
        test_row = self.loop.evaluate_split(self.test, reset=False)
        self.loop.audit_after(before_counts, before_r, "test")
        self.loop.reenable_spectral()
        self.assertIn("auc", test_row)

    def test_vanilla_mode_no_prss_components(self):
        """vanilla loop: no adapter/bridge/core, task loss only."""
        tgn, device, stream = make_tiny_tgn()
        sources, destinations, timestamps, edge_idxs, labels = stream
        train = JodieData(sources, destinations, timestamps, edge_idxs, labels)
        from prss.hosts.official_tgn import MLP
        decoder = MLP(dim=8, drop=0.1).to(device)
        loop = JodieNodeClassificationLoop(
            tgn=tgn, decoder=decoder, adapter=None, bridge=None,
            prss_core=None, optimizer=torch.optim.Adam(decoder.parameters(),
                                                       lr=3e-4),
            unrestricted_optimizer=None, device=device, batch_size=8,
            n_neighbors=4, grad_clip=5.0, lambda_resp=1.0, lambda_spec=0.1,
            trace_roots=4, trace_mode="positive_first", spectral_warmup=2,
            spectral_interval=2, monitor=_FakeMonitor(), seed=0)
        row = loop.train_epoch(0, 0, train)
        self.assertTrue(np.isfinite(row["train_task_loss"]))
        self.assertEqual(row["train_response_loss"], 0.0)
        self.assertEqual(row["train_spectral_loss"], 0.0)
        before_counts, before_r = loop.audit_before()
        val_row = loop.evaluate_split(train, reset=True)
        loop.audit_after(before_counts, before_r, "validation")
        self.assertIn("auc", val_row)


if __name__ == "__main__":
    unittest.main()
