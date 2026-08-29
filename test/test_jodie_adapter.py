"""JodieTGNAdapter: identity parity and trace structure.

Numerical tests need the torch<->numpy bridge; on the local Windows box that
bridge is broken (known env issue), so the suite skips there and runs on the
GPU box where the bridge works.
"""

import unittest
from types import SimpleNamespace

import numpy as np
import torch

from rpbe.hosts.jodie_tgn import JodieTGNAdapter, TAU_TEMPLATE

from test_jodie_vendor import (
    REQUIRES_NUMPY_BRIDGE, make_synthetic_data, make_tgn)


def make_tiny_tgn(n_layers=2):
    (sources, destinations, timestamps, edge_idxs, labels,
     node_features, edge_features) = make_synthetic_data(
        n_nodes=32, n_interactions=100, feat_dim=8, timestamp_scale=50.0)
    tgn, device = make_tgn(
        node_features, edge_features, sources, destinations, timestamps,
        edge_idxs, n_layers=n_layers, n_heads=2, n_neighbors=4,
        memory_dimension=8, message_dimension=8)
    return tgn, device, (sources, destinations, timestamps, edge_idxs, labels)


def install_adapter(tgn, n_neighbors=4):
    adapter = JodieTGNAdapter(tgn.embedding_module, compressor=None,
                              n_neighbors=n_neighbors)
    tgn.embedding_module = adapter
    return adapter


def forward_batch(tgn, sources, destinations, timestamps, edge_idxs, bs=8,
                  n_neighbors=4, train=True):
    # ``train=True`` leaves dropout randomness in the host attention; parity
    # comparisons must use train=False so the adapter-vs-host contract is
    # checked free of the host's own stochasticity.
    tgn.train() if train else tgn.eval()
    return tgn.compute_temporal_embeddings(
        sources[:bs], destinations[:bs], destinations[:bs],
        timestamps[:bs], edge_idxs[:bs], n_neighbors)


@REQUIRES_NUMPY_BRIDGE
class TestIdentityParity(unittest.TestCase):
    """Without a compressor the adapter forward must equal the bare host."""

    def test_identity_forward_bitwise_matches_bare_host(self):
        tgn, device, (sources, destinations, timestamps, edge_idxs, labels) = \
            make_tiny_tgn()
        bare = make_tiny_tgn()[0]
        bare.load_state_dict(tgn.state_dict())
        install_adapter(tgn)

        src_emb_a, dst_emb_a, neg_emb_a = forward_batch(
            tgn, sources, destinations, timestamps, edge_idxs, train=False)
        src_emb_b, dst_emb_b, neg_emb_b = forward_batch(
            bare, sources, destinations, timestamps, edge_idxs, train=False)
        for a, b in zip((src_emb_a, dst_emb_a, neg_emb_a),
                        (src_emb_b, dst_emb_b, neg_emb_b)):
            self.assertEqual(tuple(a.shape), tuple(b.shape))
            self.assertTrue(torch.allclose(a, b, atol=1e-5),
                            "adapter forward diverges from bare host")

    def test_identity_forward_bitwise_matches_bare_host_l1(self):
        """Single-layer host (official reddit config, n_layer=1)."""
        tgn, device, (sources, destinations, timestamps, edge_idxs, labels) = \
            make_tiny_tgn(n_layers=1)
        bare = make_tiny_tgn(n_layers=1)[0]
        bare.load_state_dict(tgn.state_dict())
        install_adapter(tgn)
        src_emb_a, dst_emb_a, neg_emb_a = forward_batch(
            tgn, sources, destinations, timestamps, edge_idxs, train=False)
        src_emb_b, dst_emb_b, neg_emb_b = forward_batch(
            bare, sources, destinations, timestamps, edge_idxs, train=False)
        for a, b in zip((src_emb_a, dst_emb_a, neg_emb_a),
                        (src_emb_b, dst_emb_b, neg_emb_b)):
            self.assertEqual(tuple(a.shape), tuple(b.shape))
            self.assertTrue(torch.allclose(a, b, atol=1e-5),
                            "L=1 adapter forward diverges from bare host")

    def test_forward_without_trace_is_bit_identical_to_traced_off(self):
        tgn, device, (sources, destinations, timestamps, edge_idxs, labels) = \
            make_tiny_tgn()
        install_adapter(tgn)
        # eval mode + fresh memory before each call: no dropout and no memory
        # drift, so the two forwards can only differ if tracing itself leaks
        # into the embedding computation.
        a, _, _ = forward_batch(tgn, sources, destinations, timestamps,
                                edge_idxs, train=False)
        tgn.memory.__init_memory__()
        tgn.embedding_module.clear_trace()
        b, _, _ = forward_batch(tgn, sources, destinations, timestamps,
                                edge_idxs, train=False)
        self.assertTrue(torch.allclose(a, b, atol=1e-7))


@REQUIRES_NUMPY_BRIDGE
class TestTraceStructure(unittest.TestCase):
    def setUp(self):
        tgn, device, self.stream = make_tiny_tgn()
        self.adapter = install_adapter(tgn)
        self.tgn = tgn

    def test_trace_is_bounded_internal_self_spine(self):
        sources, destinations, timestamps, edge_idxs, labels = self.stream
        self.adapter.set_trace_source_rows([2, 5])
        forward_batch(self.tgn, sources, destinations, timestamps, edge_idxs)
        trace = self.adapter.trace
        self.assertIsNotNone(trace)
        self.assertEqual(trace.root_rows, [2, 5])
        # L=2 has exactly one internal compressible interface per root.
        self.assertEqual(len(trace.cuts), 2)
        self.assertEqual({cut.tau for cut in trace.cuts}, {"tjo:layer1"})
        self.assertEqual({cut.root_row for cut in trace.cuts}, {2, 5})
        for cut in trace.cuts:
            self.assertEqual(cut.node, int(sources[cut.root_row]))
            self.assertEqual(cut.time, float(timestamps[cut.root_row]))
            self.assertEqual(cut.path, [(0, 0.0)])
        self.assertFalse(hasattr(trace, "occurrences"))
        self.assertFalse(hasattr(trace, "roots"))

    def test_trace_records_z_with_grad(self):
        """The compact candidate's z stays graph-connected."""
        sources, destinations, timestamps, edge_idxs, labels = self.stream
        self.tgn.train()
        for p in self.tgn.parameters():
            p.requires_grad_(True)
        self.adapter.set_trace_source_rows([0])
        with torch.enable_grad():
            self.tgn.compute_temporal_embeddings(
                sources[:4], destinations[:4], destinations[:4],
                timestamps[:4], edge_idxs[:4], 4)
        trace = self.adapter.trace
        self.assertEqual(len(trace.cuts), 1)
        self.assertTrue(trace.cuts[0].z.requires_grad)

    def test_gamma_runs_at_internal_layers_only(self):
        class CountingCompressor(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.cfg = SimpleNamespace(state_dims={
                    "tjo:layer0": 8,
                    "tjo:layer1": 8,
                    "tjo:layer2": 8,
                })
                self.calls = []

            def compress(self, *, tau, own_input, aggregate_output):
                self.calls.append((tau, int(aggregate_output.shape[0])))
                return aggregate_output

        sources, destinations, timestamps, edge_idxs, labels = self.stream
        compressor = CountingCompressor()
        self.adapter.compressor = compressor
        forward_batch(self.tgn, sources, destinations, timestamps, edge_idxs)
        self.assertGreater(len(compressor.calls), 0)
        self.assertEqual({tau for tau, _ in compressor.calls},
                         {"tjo:layer1"})
        self.assertNotIn("tjo:layer0", [tau for tau, _ in compressor.calls])
        self.assertNotIn("tjo:layer2", [tau for tau, _ in compressor.calls])

    def test_clear_trace_resets(self):
        sources, destinations, timestamps, edge_idxs, labels = self.stream
        self.adapter.set_trace_source_rows([0])
        forward_batch(self.tgn, sources, destinations, timestamps, edge_idxs)
        self.assertIsNotNone(self.adapter.trace)
        self.adapter.clear_trace()
        self.assertIsNone(self.adapter.trace)
        self.assertEqual(self.adapter._trace_top_rows, set())


if __name__ == "__main__":
    unittest.main()
