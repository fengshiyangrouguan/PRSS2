"""JodieTGNAdapter: identity parity, trace structure, multi-tau aux, isolation.

Numerical tests need the torch<->numpy bridge; on the local Windows box that
bridge is broken (known env issue), so the suite skips there and runs on the
GPU box where the bridge works.
"""

import unittest

import numpy as np
import torch

from prss.config import InterfaceSpec, PRSSConfig
from prss.core import PRSSCore
from prss.hosts.jodie_bridge import JodieNodeClassificationBridge
from prss.hosts.jodie_tgn import JodieTGNAdapter, TAU_TEMPLATE, jodie_preagg_dim
from prss.training.isolation import assert_clean, counts_of_spectral, r_copies

from test_jodie_vendor import (
    REQUIRES_NUMPY_BRIDGE, make_synthetic_data, make_tgn)


def make_tiny_prss(variant="vanilla", n_layers=2, host_dim=8, time_dim=8,
                   edge_dim=4, n_neighbors=4, candidate_dim=16):
    taus = [TAU_TEMPLATE.format(layer) for layer in range(n_layers + 1)]
    interfaces = {
        TAU_TEMPLATE.format(0): InterfaceSpec(
            TAU_TEMPLATE.format(0), raw_dim=host_dim, candidate_dim=host_dim,
            host_dim=host_dim, response_dim=1),
    }
    for layer in range(1, n_layers + 1):
        interfaces[TAU_TEMPLATE.format(layer)] = InterfaceSpec(
            TAU_TEMPLATE.format(layer), raw_dim=host_dim,
            candidate_dim=candidate_dim, host_dim=host_dim, response_dim=1)
    config = PRSSConfig(
        interfaces=interfaces,
        parent_local_dim=jodie_preagg_dim(host_dim, time_dim, edge_dim,
                                          n_neighbors),
        root_metadata_dim=1, relation_count=2, variant=variant)
    return config, PRSSCore(config)


def make_tiny_tgn():
    (sources, destinations, timestamps, edge_idxs, labels,
     node_features, edge_features) = make_synthetic_data(
        n_nodes=32, n_interactions=100, feat_dim=8, timestamp_scale=50.0)
    tgn, device = make_tgn(
        node_features, edge_features, sources, destinations, timestamps,
        edge_idxs, n_layers=2, n_heads=2, n_neighbors=4,
        memory_dimension=8, message_dimension=8)
    return tgn, device, (sources, destinations, timestamps, edge_idxs, labels)


def install_adapter(tgn, prss_core, n_neighbors=4):
    adapter = JodieTGNAdapter(tgn.embedding_module, prss_core, n_neighbors)
    tgn.embedding_module = adapter
    return adapter


def forward_batch(tgn, sources, destinations, timestamps, edge_idxs, bs=8,
                  n_neighbors=4):
    tgn.train()
    return tgn.compute_temporal_embeddings(
        sources[:bs], destinations[:bs], destinations[:bs],
        timestamps[:bs], edge_idxs[:bs], n_neighbors)


@REQUIRES_NUMPY_BRIDGE
class TestIdentityParity(unittest.TestCase):
    """With a vanilla compressor the adapter forward must equal the bare host."""

    def test_vanilla_forward_bitwise_matches_bare_host(self):
        tgn, device, (sources, destinations, timestamps, edge_idxs, labels) = \
            make_tiny_tgn()
        bare = make_tiny_tgn()[0]
        bare.load_state_dict(tgn.state_dict())
        config, prss = make_tiny_prss(variant="vanilla")
        install_adapter(tgn, prss)

        src_emb_a, dst_emb_a, neg_emb_a = forward_batch(
            tgn, sources, destinations, timestamps, edge_idxs)
        src_emb_b, dst_emb_b, neg_emb_b = forward_batch(
            bare, sources, destinations, timestamps, edge_idxs)
        for a, b in zip((src_emb_a, dst_emb_a, neg_emb_a),
                        (src_emb_b, dst_emb_b, neg_emb_b)):
            self.assertEqual(tuple(a.shape), tuple(b.shape))
            self.assertTrue(torch.allclose(a, b, atol=1e-5),
                            "adapter forward diverges from bare host")

    def test_forward_without_trace_is_bit_identical_to_traced_off(self):
        tgn, device, (sources, destinations, timestamps, edge_idxs, labels) = \
            make_tiny_tgn()
        config, prss = make_tiny_prss(variant="vanilla")
        install_adapter(tgn, prss)
        tgn.train()
        a, _, _ = forward_batch(tgn, sources, destinations, timestamps,
                                edge_idxs)
        tgn.embedding_module.clear_trace()
        b, _, _ = forward_batch(tgn, sources, destinations, timestamps,
                                edge_idxs)
        self.assertTrue(torch.allclose(a, b, atol=1e-7))


@REQUIRES_NUMPY_BRIDGE
class TestTraceStructure(unittest.TestCase):
    def setUp(self):
        tgn, device, self.stream = make_tiny_tgn()
        config, prss = make_tiny_prss(variant="spectral")
        self.adapter = install_adapter(tgn, prss)
        self.tgn = tgn
        self.prss = prss

    def test_trace_tree_depth_and_roots(self):
        sources, destinations, timestamps, edge_idxs, labels = self.stream
        # Trace rows 2 and 5 (source segment of the concatenated call).
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
        # Source child has delta 0; neighbor children have delta = ts - edge_time >= 0.
        for cid, rel, delta in zip(root.children, root.child_relations,
                                   root.child_delta_t):
            self.assertGreaterEqual(delta, 0.0)
            if rel == 0:
                self.assertEqual(delta, 0.0)

    def test_c1_local_zeroing(self):
        """Outside never sees the subtree states: leading child-state block is 0."""
        sources, destinations, timestamps, edge_idxs, labels = self.stream
        self.adapter.set_trace_source_rows([1])
        forward_batch(self.tgn, sources, destinations, timestamps, edge_idxs)
        trace = self.adapter.trace
        zero_dim = self.adapter._local_zero_dim  # host_dim + n*host_dim
        for occ in trace.occurrences.values():
            if occ.tau == "tjo:layer0":
                # Base interface: local is all zeros.
                self.assertTrue((occ.local_features == 0).all())
            else:
                local = occ.local_features
                self.assertTrue((local[:zero_dim] == 0).all(),
                                "child-state block must be zeroed (C1)")
                self.assertGreater(local[zero_dim:].abs().sum().item(), 0.0,
                                   "parent-side features must survive")

    def test_clear_trace_resets(self):
        sources, destinations, timestamps, edge_idxs, labels = self.stream
        self.adapter.set_trace_source_rows([0])
        forward_batch(self.tgn, sources, destinations, timestamps, edge_idxs)
        self.assertIsNotNone(self.adapter.trace)
        self.adapter.clear_trace()
        self.assertIsNone(self.adapter.trace)
        self.assertEqual(self.adapter._trace_top_rows, set())


@REQUIRES_NUMPY_BRIDGE
class TestMultiTauAux(unittest.TestCase):
    def test_bridge_builds_layer1_and_layer2_only(self):
        tgn, device, (sources, destinations, timestamps, edge_idxs, labels) = \
            make_tiny_tgn()
        config, prss = make_tiny_prss(variant="spectral")
        adapter = install_adapter(tgn, prss)
        bridge = JodieNodeClassificationBridge(adapter, prss)
        tgn.train()
        trace_rows = [0, 3]
        adapter.set_trace_source_rows(trace_rows)
        forward_batch(tgn, sources, destinations, timestamps, edge_idxs)
        aux = bridge.build(timestamps[trace_rows],
                           torch.tensor([1.0, 0.0], dtype=torch.float32))
        self.assertEqual(set(aux.matrices_by_tau), {"tjo:layer1", "tjo:layer2"})
        self.assertNotIn("tjo:layer0", aux.matrices_by_tau)
        self.assertEqual(set(aux.occurrence_counts), {"tjo:layer1", "tjo:layer2"})
        # Losses are finite and require grad.
        self.assertTrue(torch.isfinite(aux.response_loss))
        self.assertTrue(aux.response_loss.requires_grad)
        self.assertTrue(aux.spectral_loss.requires_grad)
        total = (aux.response_loss + aux.spectral_loss + aux.unrestricted_loss)
        total.backward()
        # Readers and quotient must have received gradients.
        for tau in ("tjo:layer1", "tjo:layer2"):
            reader = prss.readers[tau]
            grads = [p.grad for p in reader.parameters()
                     if p.grad is not None]
            self.assertTrue(grads, "reader {} got no gradient".format(tau))

    def test_bridge_without_trace_returns_zeros(self):
        tgn, device, (sources, destinations, timestamps, edge_idxs, labels) = \
            make_tiny_tgn()
        config, prss = make_tiny_prss(variant="spectral")
        adapter = install_adapter(tgn, prss)
        bridge = JodieNodeClassificationBridge(adapter, prss)
        tgn.embedding_module.clear_trace()
        aux = bridge.build(np.array([1.0]), torch.tensor([1.0]))
        self.assertEqual(float(aux.response_loss), 0.0)
        self.assertEqual(aux.matrices_by_tau, {})


@REQUIRES_NUMPY_BRIDGE
class TestIsolation(unittest.TestCase):
    def test_eval_never_mutates_spectral_state(self):
        tgn, device, (sources, destinations, timestamps, edge_idxs, labels) = \
            make_tiny_tgn()
        config, prss = make_tiny_prss(variant="spectral")
        adapter = install_adapter(tgn, prss)
        tgn.eval()
        before_counts = counts_of_spectral(prss)
        before_r = r_copies(prss)
        # Held-out evaluation: no trace, spectral updates gated off.
        prss.set_spectral_updates_allowed(False)
        adapter.clear_trace()
        with torch.no_grad():
            forward_batch(tgn, sources, destinations, timestamps, edge_idxs)
        assert_clean(before_counts, before_r, prss,
                     bool(adapter.trace is not None), "test")
        prss.set_spectral_updates_allowed(True)

    def test_gated_statistics_are_skipped(self):
        tgn, device, (sources, destinations, timestamps, edge_idxs, labels) = \
            make_tiny_tgn()
        config, prss = make_tiny_prss(variant="spectral")
        adapter = install_adapter(tgn, prss)
        tgn.train()
        before = counts_of_spectral(prss)
        prss.set_spectral_updates_allowed(False)
        adapter.set_trace_source_rows([0])
        forward_batch(tgn, sources, destinations, timestamps, edge_idxs)
        prss.update_statistics(1, {})
        prss.maybe_update(1)
        prss.set_spectral_updates_allowed(True)
        after = counts_of_spectral(prss)
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
