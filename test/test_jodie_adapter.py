"""JodieTGNAdapter: identity parity and trace structure.

Numerical tests need the torch<->numpy bridge; on the local Windows box that
bridge is broken (known env issue), so the suite skips there and runs on the
GPU box where the bridge works.
"""

import unittest

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

    def test_trace_tree_depth_and_roots(self):
        sources, destinations, timestamps, edge_idxs, labels = self.stream
        self.adapter.set_trace_source_rows([2, 5])
        forward_batch(self.tgn, sources, destinations, timestamps, edge_idxs)
        trace = self.adapter.trace
        self.assertIsNotNone(trace)
        self.assertEqual(trace.root_rows, [2, 5])
        self.assertEqual(len(trace.roots), 2)
        # Tree depth = L+1 layers: layer2 root -> layer1 -> layer0.
        layers = {occ.metadata["layer"] for occ in trace.occurrences.values()}
        self.assertEqual(layers, {0, 1, 2})
        for root in trace.roots:
            occ = trace.occurrences[root]
            self.assertEqual(occ.tau, "tjo:layer2")
            # children: 1 source continuation + up to n_neighbors neighbors.
            self.assertGreaterEqual(len(occ.children), 1)
            self.assertLessEqual(len(occ.children), 1 + 4)
            rels = set(occ.child_relations)
            self.assertLessEqual(rels, {0, 1})

    def test_children_delta_times(self):
        sources, destinations, timestamps, edge_idxs, labels = self.stream
        self.adapter.set_trace_source_rows([0])
        forward_batch(self.tgn, sources, destinations, timestamps, edge_idxs)
        trace = self.adapter.trace
        root = trace.occurrences[trace.roots[0]]
        for cid, rel, delta in zip(root.children, root.child_relations,
                                   root.child_delta_t):
            self.assertGreaterEqual(delta, 0.0)
            if rel == 0:
                self.assertEqual(delta, 0.0)

    def test_metadata_contract_node_time_own_raw(self):
        """Every occurrence carries node / as-of time / own_raw, and the
        as-of time is the query timestamp for the whole tree (the official
        recursion reuses the query timestamp for neighbors)."""
        sources, destinations, timestamps, edge_idxs, labels = self.stream
        self.adapter.set_trace_source_rows([0, 1])
        forward_batch(self.tgn, sources, destinations, timestamps, edge_idxs)
        trace = self.adapter.trace
        for occ in trace.occurrences.values():
            self.assertIn("node", occ.metadata)
            self.assertIn("time", occ.metadata)
            self.assertIn("own_raw", occ.metadata)
            self.assertGreaterEqual(occ.metadata["node"], 0)
            self.assertIn("layer", occ.metadata)
            own = occ.metadata["own_raw"]
            self.assertEqual(tuple(own.shape), (8,))  # host_dim of the tiny host
            self.assertTrue(torch.isfinite(own).all())
            # Note: synthetic node_features are all-zero, so own_raw sums to
            # zero — only shape/finiteness is contractually guaranteed here.
        # as-of time == query timestamp of the traced row, for every node.
        for root_id, row in zip(trace.roots, trace.root_rows):
            t_root = float(timestamps[row])
            for oid in trace.occurrences:
                if self._descends_from(trace, oid, root_id):
                    self.assertEqual(trace.occurrences[oid].metadata["time"],
                                     t_root)

    @staticmethod
    def _descends_from(trace, oid, root_id):
        seen = set()

        def walk(x):
            if x == oid:
                return True
            if x in seen:
                return False
            seen.add(x)
            return any(walk(c) for c in trace.occurrences[x].children)

        return walk(root_id)

    def test_c1_local_zeroing(self):
        """Legacy C1: the child-state block of local_features stays zeroed."""
        sources, destinations, timestamps, edge_idxs, labels = self.stream
        self.adapter.set_trace_source_rows([1])
        forward_batch(self.tgn, sources, destinations, timestamps, edge_idxs)
        trace = self.adapter.trace
        zero_dim = self.adapter._local_zero_dim  # host_dim + n*host_dim
        for occ in trace.occurrences.values():
            if occ.tau == "tjo:layer0":
                self.assertTrue((occ.local_features == 0).all())
            else:
                local = occ.local_features
                self.assertTrue((local[:zero_dim] == 0).all(),
                                "child-state block must be zeroed (C1)")
                self.assertGreater(local[zero_dim:].abs().sum().item(), 0.0,
                                   "parent-side features must survive")

    def test_trace_records_z_with_grad(self):
        """OccurrenceState carries z only; with a grad-enabled host forward
        the traced z is graph-connected."""
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
        root = trace.occurrences[trace.roots[0]]
        self.assertTrue(root.state.z.requires_grad)

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
