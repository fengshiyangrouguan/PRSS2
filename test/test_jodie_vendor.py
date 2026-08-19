"""Vendored upstream TGN: forward smoke, state_dict round-trip, message chain.

The vendor package must stay a faithful copy of twitter-research/tgn
(commit d55bbe678, imports-only rewrites, see README.md).  These tests pin
the behavioural contract the JODIE line relies on: the full recursive
``compute_temporal_embeddings`` call surface, memory-message flow, and
``backup_memory``/``restore_memory``/``detach_memory`` round-trips.
"""

import tempfile
import unittest

import numpy as np
import torch

from prss.hosts.official_tgn import TGN, MLP, NeighborFinder, get_neighbor_finder


def _numpy_bridge_ok():
    """torch.from_numpy fails on this Windows box (known env issue); run
    the numerical tests on the GPU box instead of failing locally."""
    try:
        torch.from_numpy(np.zeros(1, dtype=np.float32))
        return True
    except Exception:
        return False


REQUIRES_NUMPY_BRIDGE = unittest.skipUnless(
    _numpy_bridge_ok(),
    "torch numpy bridge broken locally; run on the GPU box")


def make_synthetic_data(n_nodes=40, n_interactions=120, feat_dim=8,
                        seed=0, timestamp_scale=10.0):
    """Deterministic synthetic stream: sources, destinations, times, idx."""
    rng = np.random.RandomState(seed)
    sources = rng.randint(0, n_nodes, size=n_interactions)
    destinations = rng.randint(0, n_nodes, size=n_interactions)
    timestamps = np.sort(rng.rand(n_interactions) * timestamp_scale)
    edge_idxs = np.arange(1, n_interactions + 1)
    labels = rng.randint(0, 2, size=n_interactions).astype(np.float64)
    node_features = np.zeros((n_nodes + 1, feat_dim), dtype=np.float32)
    edge_features = rng.randn(n_interactions + 1, feat_dim).astype(np.float32)
    return (sources, destinations, timestamps, edge_idxs, labels,
            node_features, edge_features)


def make_tgn(node_features, edge_features, sources, destinations,
             timestamps, edge_idxs=None, device=None, n_layers=2,
             n_heads=2, n_neighbors=4, memory_dimension=8,
             message_dimension=8, use_memory=True,
             time_shifts=(0.0, 1.0, 0.0, 1.0)):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    max_node_idx = int(max(sources.max(), destinations.max()))
    if edge_idxs is None:
        edge_idxs = np.arange(1, len(sources) + 1)
    adj_list = [[] for _ in range(max_node_idx + 1)]
    for s, d, e, t in zip(sources, destinations, edge_idxs, timestamps):
        adj_list[s].append((d, e, t))
        adj_list[d].append((s, e, t))
    finder = NeighborFinder(adj_list, uniform=False)
    ms, ss, md, sd = time_shifts
    return TGN(
        neighbor_finder=finder, node_features=node_features,
        edge_features=edge_features, device=device, n_layers=n_layers,
        n_heads=n_heads, dropout=0.1, use_memory=use_memory,
        memory_update_at_start=True, message_dimension=message_dimension,
        memory_dimension=memory_dimension,
        embedding_module_type="graph_attention", message_function="identity",
        mean_time_shift_src=ms, std_time_shift_src=ss,
        mean_time_shift_dst=md, std_time_shift_dst=sd,
        n_neighbors=n_neighbors, aggregator_type="last").to(device), device


@REQUIRES_NUMPY_BRIDGE
class TestVendorTGNForward(unittest.TestCase):
    def setUp(self):
        (self.sources, self.destinations, self.timestamps, self.edge_idxs,
         self.labels, self.node_features, self.edge_features) = make_synthetic_data()
        self.tgn, self.device = make_tgn(
            self.node_features, self.edge_features, self.sources,
            self.destinations, self.timestamps)
        self.batch = self.sources[:8], self.destinations[:8], self.timestamps[:8]

    def test_compute_temporal_embeddings_shapes_and_nan(self):
        tgn = self.tgn
        tgn.train()
        src, dst, ts = self.batch
        src_emb, dst_emb, neg_emb = tgn.compute_temporal_embeddings(
            src, dst, dst, ts, self.edge_idxs[:8], n_neighbors=4)
        for emb in (src_emb, dst_emb, neg_emb):
            self.assertEqual(tuple(emb.shape), (8, 8))
            self.assertFalse(torch.isnan(emb).any())
            self.assertTrue(torch.isfinite(emb).all())

    def test_memory_advances_after_batch(self):
        """Official message loop: batch N's messages are consumed by batch
        N+1 (memory_update_at_start), so a single first batch cannot move the
        memory — two batches can."""
        tgn = self.tgn
        tgn.train()
        src, dst, ts = self.batch
        # First forward only registers messages (nothing to consume yet).
        tgn.compute_temporal_embeddings(src, dst, dst, ts,
                                        self.edge_idxs[:8], n_neighbors=4)
        before = tgn.memory.memory.data.clone()
        # Second forward consumes the first batch's messages: memory moves.
        tgn.compute_temporal_embeddings(src, dst, dst, ts,
                                        self.edge_idxs[:8], n_neighbors=4)
        after = tgn.memory.memory.data.clone()
        self.assertFalse(torch.equal(before, after))

    def test_layer0_equals_raw_source_features(self):
        """n_layers=0 returns memory+node-features unchanged (upstream contract)."""
        tgn = self.tgn
        tgn.train()
        memory, _ = tgn.get_updated_memory(list(range(tgn.n_nodes)),
                                          tgn.memory.messages)
        src = torch.from_numpy(self.batch[0]).long().to(self.device)
        expected = tgn.node_raw_features[src] + memory[self.batch[0]]
        actual = tgn.embedding_module.compute_embedding(
            memory=memory, source_nodes=self.batch[0], timestamps=self.batch[2],
            n_layers=0, n_neighbors=4)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))

    def test_mlp_decoder_shape(self):
        mlp = MLP(dim=8, drop=0.1)
        out = mlp(torch.randn(5, 8))
        self.assertEqual(tuple(out.shape), (5,))

    def test_get_neighbor_finder(self):
        # Pure upstream helper: adjacency built from a Data-like object.
        class _D:
            def __init__(self):
                self.sources = self.sources0 = np.arange(10, dtype=np.int64)
                self.destinations = np.arange(10, dtype=np.int64) + 1
                self.edge_idxs = np.arange(1, 11, dtype=np.int64)
                self.timestamps = np.arange(10, dtype=np.float64) + 1.0
        finder = get_neighbor_finder(_D(), uniform=False)
        self.assertIsInstance(finder, NeighborFinder)
        nbrs, eidx, etimes = finder.get_temporal_neighbor(
            np.array([2]), np.array([6.0]), n_neighbors=2)
        self.assertEqual(nbrs.shape, (1, 2))


@REQUIRES_NUMPY_BRIDGE
class TestStateDict(unittest.TestCase):
    def test_roundtrip_state_dict(self):
        (sources, destinations, timestamps, edge_idxs, labels,
         node_features, edge_features) = make_synthetic_data()
        tgn, device = make_tgn(node_features, edge_features, sources,
                               destinations, timestamps)
        sd = tgn.state_dict()
        keys = set(sd)
        # The memory state must be in the dict (upstream keeps it as nn.Parameter).
        self.assertIn("memory.memory", keys)
        self.assertIn("memory.last_update", keys)
        # Fresh copy loads the same dict exactly.
        tgn2, _ = make_tgn(node_features, edge_features, sources,
                           destinations, timestamps)
        tgn2.load_state_dict(sd)
        for k in keys:
            self.assertTrue(torch.equal(sd[k], tgn2.state_dict()[k]))

    def test_strict_load_contract(self):
        """Pretrained-key validator contract: missing/unexpected must be empty."""
        (sources, destinations, timestamps, edge_idxs, labels,
         node_features, edge_features) = make_synthetic_data()
        tgn, device = make_tgn(node_features, edge_features, sources,
                               destinations, timestamps)
        sd = tgn.state_dict()
        tgn2, _ = make_tgn(node_features, edge_features, sources,
                           destinations, timestamps)
        missing, unexpected = tgn2.load_state_dict(sd, strict=False)
        self.assertEqual(sorted(missing), [])
        self.assertEqual(sorted(unexpected), [])


@REQUIRES_NUMPY_BRIDGE
class TestMessageChain(unittest.TestCase):
    def test_store_aggregate_update(self):
        (sources, destinations, timestamps, edge_idxs, labels,
         node_features, edge_features) = make_synthetic_data()
        tgn, device = make_tgn(node_features, edge_features, sources,
                               destinations, timestamps)
        tgn.train()
        mem = tgn.memory
        node = int(sources[0])
        # Manually store one message per node and flush via update_memory.
        msg = torch.randn(1, mem.message_dimension, device=device)
        mem.store_raw_messages([node], {node: [(msg[0], torch.tensor(
            float(timestamps[0]), device=device))]})
        before = mem.get_memory([node]).clone()
        tgn.update_memory([node], mem.messages)
        after = mem.get_memory([node])
        self.assertFalse(torch.allclose(before, after))
        # last_update advanced.
        self.assertGreater(mem.last_update[node].item(), 0.0)

    def test_backup_restore_detach(self):
        (sources, destinations, timestamps, edge_idxs, labels,
         node_features, edge_features) = make_synthetic_data()
        tgn, device = make_tgn(node_features, edge_features, sources,
                               destinations, timestamps)
        tgn.train()
        src, dst, ts = self_batch = (sources[:8], destinations[:8], timestamps[:8])
        tgn.compute_temporal_embeddings(src, dst, dst, ts, edge_idxs[:8],
                                        n_neighbors=4)
        backup = tgn.memory.backup_memory()
        live_memory = tgn.memory.memory.data.clone()
        live_messages = dict(tgn.memory.messages)
        self.assertGreater(len(live_messages), 0)
        # Corrupt, then restore.
        tgn.memory.memory.data.zero_()
        tgn.memory.messages.clear()
        tgn.memory.restore_memory(backup)
        self.assertTrue(torch.equal(tgn.memory.memory.data, live_memory))
        self.assertEqual(set(tgn.memory.messages), set(live_messages))
        # detach keeps values, drops autograd graph.
        tgn.memory.detach_memory()
        self.assertTrue(tgn.memory.memory.requires_grad is False)

    def test_clear_messages_after_update(self):
        """Upstream message loop: a forward registers messages at its end;
        the following forward consumes and clears them, then registers a fresh
        batch. Observable as: register -> consume (memory moves) -> re-register."""
        (sources, destinations, timestamps, edge_idxs, labels,
         node_features, edge_features) = make_synthetic_data()
        tgn, device = make_tgn(node_features, edge_features, sources,
                               destinations, timestamps)
        tgn.train()
        src, dst, ts = (sources[:8], destinations[:8], timestamps[:8])
        # First forward: nothing to consume, only registration happens.
        tgn.compute_temporal_embeddings(src, dst, dst, ts, edge_idxs[:8],
                                        n_neighbors=4)
        self.assertTrue(
            any(len(v) > 0 for v in tgn.memory.messages.values()))
        # Second forward: consumes the first batch (memory moves) and
        # re-registers fresh messages for the next batch.
        before = tgn.memory.memory.data.clone()
        tgn.compute_temporal_embeddings(src, dst, dst, ts, edge_idxs[:8],
                                        n_neighbors=4)
        self.assertFalse(torch.equal(before, tgn.memory.memory.data.clone()))
        self.assertTrue(
            any(len(v) > 0 for v in tgn.memory.messages.values()))


if __name__ == "__main__":
    unittest.main()
