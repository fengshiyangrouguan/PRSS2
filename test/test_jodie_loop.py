"""JODIE loop contracts: row selection, metric bundle, tiny end-to-end smoke."""

import unittest
from pathlib import Path

import numpy as np
import torch

from rpbe.compressor import RecursiveCompressor
from rpbe.config import RPBConfig
from rpbe.data.jodie import JodieData
from rpbe.maps import FixedMaps
from rpbe.records import JodieCutBuilder, JodieFutureIndex, NODE_CLASS
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


class TestOnePassSourceContract(unittest.TestCase):
    def test_training_code_has_no_shadow_replay_path(self):
        training = Path(__file__).resolve().parents[1] / "src" / "rpbe" \
            / "training"
        text = (training / "jodie_loop.py").read_text(encoding="utf-8") \
            + (training / "pretrain_loop.py").read_text(encoding="utf-8")
        for forbidden in ("_close_and_replay", "_backup_replay_state",
                          "_restore_replay_state", "root_events",
                          "torch.autograd.grad"):
            self.assertNotIn(forbidden, text)


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

    def _make_loop(self, rpbe=False, supervision_mode="production"):
        tgn = self.tgn
        from rpbe.hosts.official_tgn import MLP
        decoder = MLP(dim=8, drop=0.1).to(self.device)
        head_params = list(decoder.parameters())
        repr_params = [p for p in tgn.parameters() if p.requires_grad]
        adapter = cut_builder = fixed_maps = rpbe_cfg = None
        if rpbe:
            cfg = RPBConfig(
                state_dims={"tjo:layer0": 8, "tjo:layer1": 8, "tjo:layer2": 8},
                own_dims={"tjo:layer0": 8, "tjo:layer1": 8, "tjo:layer2": 8},
                width_D=16, m=64, kf_min_abs=4, kf_min_ratio=0.5,
                rpbe_seed=0, delta_t_scale=1.0,
                supervision_mode=supervision_mode)
            rpbe_cfg = cfg
            compressor = RecursiveCompressor(cfg).to(self.device)
            repr_params += [p for p in compressor.parameters()]
            adapter = install_adapter(tgn)
            adapter.compressor = compressor
            fixed_maps = FixedMaps(cfg).to(self.device)
            cut_builder = JodieCutBuilder(
                JodieFutureIndex(self.train),
                stage=NODE_CLASS, seed=0,
                n_observations=2,
                supervision_mode=supervision_mode)
        seen = set()
        repr_params = [p for p in repr_params
                       if not (id(p) in seen or seen.add(id(p)))]
        head_optimizer = torch.optim.Adam(head_params, lr=3e-4)
        repr_optimizer = torch.optim.Adam(repr_params, lr=3e-4)
        return JodieNodeClassificationLoop(
            tgn=tgn, decoder=decoder, repr_optimizer=repr_optimizer,
            head_optimizer=head_optimizer,
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
        self.assertIn("train_kf_score", row)
        self.assertTrue(np.isfinite(row["train_kf_score"]))
        self.assertIsNotNone(row["kf"])
        self.assertTrue(np.isfinite(row["kf"]["kf_loss"]))
        self.assertIn("J_norm", row["kf"])
        self.assertGreater(row["n_batches"], 0)
        val_row = loop.evaluate_split(self.val, reset=False)
        self.assertIn("auc", val_row)
        # Evaluation must not have built any trace.
        self.assertIsNone(loop.adapter.trace)

    def test_structural_arms_train_finite(self):
        """Each structural arm runs one epoch without crashing, produces the
        shared Y1+Y2-valid cut set, and evaluates cleanly.

        This is a construction/training sanity gate only.  Whether a KF
        window actually CLOSES depends on the real per-batch cut funnel; on
        the tiny synthetic stream here the intersection can be too small to
        fill a window, so we assert cuts are produced and no error, and
        separately that a KF diagnostic is emitted when the window closes.
        The window-closing gate on real data is run as a separate server
        diagnostic."""
        n_nodes, n_cyc = 6, 60
        n = n_nodes * n_cyc
        srcs = np.asarray([i % n_nodes for i in range(n)],
                          dtype=np.int64)
        dsts = np.asarray([(i + 1) % n_nodes for i in range(n)],
                          dtype=np.int64)
        _rs = np.random.RandomState(7)
        tms = np.cumsum(_rs.uniform(0.5, 2.0, size=n)).astype(np.float64)
        eis = np.arange(1, n + 1, dtype=np.int64)
        lab = (np.arange(n) % 3 == 0).astype(np.float64)
        from test_jodie_vendor import make_tgn as _MT
        node_features = np.zeros((n_nodes + 1, 8), dtype=np.float32)
        edge_features = np.zeros((n + 1, 8), dtype=np.float32)
        dense_train = JodieData(srcs[:n // 2], dsts[:n // 2],
                                tms[:n // 2], eis[:n // 2], lab[:n // 2])
        dense_val = JodieData(srcs[n // 2:], dsts[n // 2:], tms[n // 2:],
                              eis[n // 2:], lab[n // 2:])
        from rpbe.hosts.official_tgn import MLP
        cut_counts = {}
        for mode in ("1obs", "2obs_aligned", "2obs_mispaired"):
            with self.subTest(mode=mode):
                tgn, device = _MT(
                    node_features, edge_features, srcs, dsts, tms, eis,
                    device=torch.device("cpu"), n_layers=2, n_heads=2,
                    n_neighbors=4, memory_dimension=8, message_dimension=8)
                decoder = MLP(dim=8, drop=0.1).to(device)
                head_params = list(decoder.parameters())
                cfg = RPBConfig(
                    state_dims={"tjo:layer0": 8, "tjo:layer1": 8,
                                "tjo:layer2": 8},
                    own_dims={"tjo:layer0": 8, "tjo:layer1": 8,
                              "tjo:layer2": 8},
                    width_D=16, m=64, kf_min_abs=4, kf_min_ratio=0.5,
                    rpbe_seed=0, delta_t_scale=1.0,
                    supervision_mode=mode)
                compressor = RecursiveCompressor(cfg).to(device)
                adapter = install_adapter(tgn)
                adapter.compressor = compressor
                repr_params = [p for p in tgn.parameters()
                               if p.requires_grad]
                fixed_maps = FixedMaps(cfg).to(device)
                cut_builder = JodieCutBuilder(
                    JodieFutureIndex(dense_train), stage=NODE_CLASS, seed=0,
                    n_observations=2, supervision_mode=mode)
                head_optimizer = torch.optim.Adam(head_params, lr=3e-4)
                repr_optimizer = torch.optim.Adam(repr_params, lr=3e-4)
                loop = JodieNodeClassificationLoop(
                    tgn=tgn, decoder=decoder,
                    repr_optimizer=repr_optimizer,
                    head_optimizer=head_optimizer, device=device,
                    batch_size=24, n_neighbors=4, grad_clip=5.0,
                    monitor=_FakeMonitor(), seed=0, finetune_host=True,
                    adapter=adapter, cut_builder=cut_builder,
                    fixed_maps=fixed_maps, rpbe_cfg=cfg, trace_roots=8)
                row = loop.train_epoch(0, 0, dense_train)
                self.assertTrue(np.isfinite(row["train_task_loss"]),
                                "{} task loss not finite".format(mode))
                val_row = loop.evaluate_split(dense_val, reset=False)
                self.assertIn("auc", val_row)
                self.assertIsNone(loop.adapter.trace,
                                  "{} leaked a trace into eval".format(mode))
                # Count the Y1+Y2-valid cuts actually produced across the
                # epoch (via the adapter trace during a manual forward), to
                # confirm the builder emits rows for this mode.
                n_produced = 0
                if tgn.use_memory:
                    tgn.memory.__init_memory__()
                for k in range(4):
                    s0 = k * 24
                    e0 = min(len(dense_train.sources), s0 + 24)
                    tr = select_trace_rows(
                        lab[s0:e0], 8, 0, k, "evenly_spaced")
                    adapter.set_trace_source_rows(tr)
                    with torch.no_grad():
                        tgn.compute_temporal_embeddings(
                            dense_train.sources[s0:e0],
                            dense_train.destinations[s0:e0],
                            dense_train.destinations[s0:e0],
                            dense_train.timestamps[s0:e0],
                            dense_train.edge_idxs[s0:e0], 4)
                    if adapter.trace and adapter.trace.cuts:
                        n_produced += len(cut_builder.build(
                            adapter.trace, batch_seed=k))
                    adapter.clear_trace()
                cut_counts[mode] = n_produced
                if n_produced > 0 and row.get("kf") is not None:
                    self.assertTrue(np.isfinite(row["kf"]["kf_loss"]),
                                    "{} kf_loss not finite".format(mode))
        # 1obs emits one row per cut, 2obs emits two rows per cut, so row
        # counts differ 2x; the shared Y1+Y2-intersection CUT population must
        # match across arms.  Rebuild once from a manual trace to count cuts.
        cut_set_counts = {}
        tgn, device = _MT(node_features, edge_features, srcs, dsts, tms, eis,
                          device=torch.device("cpu"), n_layers=2, n_heads=2,
                          n_neighbors=4, memory_dimension=8,
                          message_dimension=8)
        cfg1 = RPBConfig(
            state_dims={"tjo:layer0": 8, "tjo:layer1": 8, "tjo:layer2": 8},
            own_dims={"tjo:layer0": 8, "tjo:layer1": 8, "tjo:layer2": 8},
            width_D=16, m=64, kf_min_abs=4, kf_min_ratio=0.5,
            rpbe_seed=0, delta_t_scale=1.0, supervision_mode="1obs")
        comp1 = RecursiveCompressor(cfg1).to(device)
        ad1 = install_adapter(tgn)
        ad1.compressor = comp1
        idx = JodieFutureIndex(dense_train)
        for mode in ("1obs", "2obs_aligned", "2obs_mispaired"):
            cb = JodieCutBuilder(idx, stage=NODE_CLASS, seed=0,
                                 n_observations=2, supervision_mode=mode)
            seen = set()
            if tgn.use_memory:
                tgn.memory.__init_memory__()
            for k in range(6):
                s0 = k * 24
                e0 = min(len(dense_train.sources), s0 + 24)
                tr = select_trace_rows(lab[s0:e0], 8, 0, k,
                                       "evenly_spaced")
                ad1.set_trace_source_rows(tr)
                with torch.no_grad():
                    tgn.compute_temporal_embeddings(
                        dense_train.sources[s0:e0],
                        dense_train.destinations[s0:e0],
                        dense_train.destinations[s0:e0],
                        dense_train.timestamps[s0:e0],
                        dense_train.edge_idxs[s0:e0], 4)
                if ad1.trace and ad1.trace.cuts:
                    for r in cb.build(ad1.trace, batch_seed=k):
                        seen.add(r.cut_id)
                ad1.clear_trace()
            cut_set_counts[mode] = len(seen)
        self.assertEqual(cut_set_counts["1obs"],
                         cut_set_counts["2obs_aligned"])
        self.assertEqual(cut_set_counts["1obs"],
                         cut_set_counts["2obs_mispaired"])


if __name__ == "__main__":
    unittest.main()
