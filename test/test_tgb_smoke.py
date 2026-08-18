"""TGB host smoke tests: identity parity, trace/bridge, isolation.

Requires the ``tgb`` extra (torch_geometric + tgb); skipped otherwise.
Dataset-dependent tests skip unless ``TGB_ROOT`` / network access is available.
"""

import os
import unittest

import numpy as np
import torch

torch_geometric = None
try:
    import torch_geometric  # noqa: F401
except ImportError:
    torch_geometric = None


@unittest.skipIf(torch_geometric is None, "torch_geometric not installed")
class TestPyGHost(unittest.TestCase):
    def _make(self, mem_dim=8, time_dim=4, msg_dim=6, emb_dim=8, n_nodes=16):
        from prss.hosts.pyg_models.decoder import LinkPredictor
        from prss.hosts.pyg_models.emb_module import GraphAttentionEmbedding
        from prss.hosts.pyg_models.memory_module import TGNMemory
        from prss.hosts.pyg_models.msg_agg import LastAggregator
        from prss.hosts.pyg_models.msg_func import IdentityMessage
        from prss.hosts.pyg_models.neighbor_loader import LastNeighborLoader

        memory = TGNMemory(n_nodes, msg_dim, mem_dim, time_dim,
                           message_module=IdentityMessage(msg_dim, mem_dim, time_dim),
                           aggregator_module=LastAggregator())
        gnn = GraphAttentionEmbedding(mem_dim, emb_dim, msg_dim, time_enc=memory.time_enc)
        link_pred = LinkPredictor(emb_dim)
        loader = LastNeighborLoader(n_nodes, size=4)
        return memory, gnn, link_pred, loader, mem_dim, time_dim, msg_dim, emb_dim

    def _config(self, mem_dim, time_dim, msg_dim, emb_dim, candidate_dim=16):
        from prss.config import InterfaceSpec, PRSSConfig
        from prss.hosts.tgn_pyg import TAU, pyg_preagg_dim
        return PRSSConfig(
            interfaces={TAU: InterfaceSpec(TAU, raw_dim=emb_dim,
                                           candidate_dim=candidate_dim,
                                           host_dim=emb_dim)},
            context_dim=8, root_metadata_dim=emb_dim + 2,
            parent_local_dim=pyg_preagg_dim(mem_dim, time_dim, msg_dim, 4),
            relation_count=4, relation_dim=8,
            reader_hidden_dim=16, candidate_hidden_dim=16,
        )

    def _stream(self, loader, memory, src, dst, t, msg):
        memory.update_state(src, dst, t, msg)
        loader.insert(src, dst)
        memory.detach()

    def test_identity_initialization_matches_vanilla_host(self):
        torch.manual_seed(0)
        np.random.seed(0)
        from prss.core import PRSSCore
        from prss.hosts.tgn_pyg import PyGTGNAdapter

        (memory_a, gnn_a, _, loader_a, mem_d, time_d, msg_d, emb_d) = self._make()
        (memory_b, gnn_b, _, loader_b, *_) = self._make()
        gnn_b.load_state_dict(gnn_a.state_dict())
        memory_b.load_state_dict(memory_a.state_dict())
        gnn_a.eval(); memory_a.eval(); gnn_b.eval(); memory_b.eval()

        # Advance both streams identically with a few events.
        src = torch.tensor([1, 2, 3]); dst = torch.tensor([4, 5, 6])
        t = torch.tensor([1, 2, 3])
        msg = torch.randn(3, msg_d)
        for loader, memory in ((loader_a, memory_a), (loader_b, memory_b)):
            for i in range(3):
                self._stream(loader, memory, src[i:i+1], dst[i:i+1], t[i:i+1], msg[i:i+1])

        query = torch.tensor([1, 4, 2])
        n_id_a, eidx_a, eid_a = loader_a(query)
        n_id_b, eidx_b, eid_b = loader_b(query)
        with torch.no_grad():
            z_a, lu_a = memory_a(n_id_a)
            vanilla = gnn_a(z_a, lu_a, eidx_a, t[eid_a], msg[eid_a])
            core = PRSSCore(self._config(mem_d, time_d, msg_d, emb_d),
                            variant="spectral")
            adapter = PyGTGNAdapter(memory_b, gnn_b, core, n_neighbors=4,
                                    mem_dim=mem_d, time_dim=time_d, msg_dim=msg_d,
                                    emb_dim=emb_d)
            adapter.eval()
            z = adapter.embed(n_id_b, eidx_b, t[eid_b], msg[eid_b])
        self.assertTrue(torch.allclose(z, vanilla, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.equal(memory_a.memory, memory_b.memory))
        self.assertTrue(torch.equal(memory_a.last_update, memory_b.last_update))

    def test_trace_and_bridge_produce_auxiliary(self):
        torch.manual_seed(1)
        np.random.seed(1)
        from prss.core import PRSSCore
        from prss.hosts.tgn_pyg import PyGTGNAdapter
        from prss.hosts.tgn_pyg_bridge import TGBLinkOutsideBridge

        (memory, gnn, _, loader, mem_d, time_d, msg_d, emb_d) = self._make()
        src = torch.tensor([1, 2, 3]); dst = torch.tensor([4, 5, 6])
        t = torch.tensor([1, 2, 3])
        msg = torch.randn(3, msg_d)
        for i in range(3):
            self._stream(loader, memory, src[i:i+1], dst[i:i+1], t[i:i+1], msg[i:i+1])

        core = PRSSCore(self._config(mem_d, time_d, msg_d, emb_d), variant="spectral")
        adapter = PyGTGNAdapter(memory, gnn, core, n_neighbors=4, mem_dim=mem_d,
                                time_dim=time_d, msg_dim=msg_d, emb_dim=emb_d)
        bridge = TGBLinkOutsideBridge(adapter, core, time_mean=0.0, time_std=1.0)

        query_src = torch.tensor([1, 2])
        query_dst = torch.tensor([4, 5])
        query_t = torch.tensor([4.0, 5.0])
        neg = torch.tensor([8, 9])
        n_id = torch.cat([query_src, query_dst, neg]).unique()
        n_id, eidx, eid = loader(n_id)
        assoc = torch.empty(16, dtype=torch.long)
        assoc[n_id] = torch.arange(n_id.numel())
        root_locals = torch.cat([assoc[query_src], assoc[query_dst], assoc[neg]])
        root_times = query_t.repeat(3)
        z = adapter.embed(n_id, eidx, t[eid], msg[eid], root_locals, root_times)
        self.assertIsNotNone(adapter.trace)
        self.assertGreater(len(adapter.trace.occurrences), 0)

        aux = bridge.build(z, query_t, assoc[query_src], assoc[query_dst], assoc[neg],
                           trace_rows=[0, 1])
        from prss.hosts.tgn_pyg import TAU
        self.assertIn(TAU, aux.matrices_by_tau)
        self.assertGreater(aux.matrices_by_tau[TAU].shape[0], 0)
        self.assertTrue(aux.response_loss.requires_grad)
        aux.response_loss.backward()
        reader = core.readers[TAU]
        self.assertTrue(any(p.grad is not None for p in reader.parameters()))

    def test_trace_is_finite_and_bfs_safe(self):
        """Regression: mutual neighbors (u<->v) must not create infinite BFS.

        The sampled subgraph is undirected, so both directions of an edge can
        appear, and when both endpoints are traced roots their occurrences can
        reference each other.  The bridge/auxiliary BFS must still terminate
        (this was a real OOM bug from an infinite outside loop)."""
        torch.manual_seed(3)
        np.random.seed(3)
        from prss.core import PRSSCore
        from prss.hosts.tgn_pyg import PyGTGNAdapter

        (memory, gnn, _, loader, mem_d, time_d, msg_d, emb_d) = self._make()
        # Mutual pair: 1 -> 4 and 4 -> 1.
        self._stream(loader, memory, torch.tensor([1]), torch.tensor([4]),
                     torch.tensor([1]), torch.randn(1, msg_d))
        self._stream(loader, memory, torch.tensor([4]), torch.tensor([1]),
                     torch.tensor([2]), torch.randn(1, msg_d))

        core = PRSSCore(self._config(mem_d, time_d, msg_d, emb_d), variant="spectral")
        adapter = PyGTGNAdapter(memory, gnn, core, n_neighbors=4, mem_dim=mem_d,
                                time_dim=time_d, msg_dim=msg_d, emb_dim=emb_d)

        n_id = torch.tensor([1, 4])
        n_id, eidx, eid = loader(n_id)
        root_locals = torch.tensor([0, 1])  # both 1 and 4 are roots
        root_times = torch.tensor([3.0, 3.0])
        with torch.no_grad():
            adapter.embed(n_id, eidx, torch.tensor([1, 2])[eid], torch.randn(2, msg_d)[eid],
                          root_locals, root_times)
        trace = adapter.trace
        self.assertIsNotNone(trace)
        self.assertLessEqual(len(trace.occurrences), 8)  # finite, no explosion
        # BFS over the children graph (mirroring the bridge) must terminate and
        # visit exactly the recorded occurrences.
        seen = set()
        for root in trace.roots:
            stack = [root]
            while stack:
                oid = stack.pop()
                if oid in seen:
                    continue
                seen.add(oid)
                stack.extend(trace.occurrences[oid].children)
        self.assertEqual(seen, set(trace.occurrences.keys()))

    def test_evaluator_input_format_gives_correct_mrr(self):
        """Regression: the TGB Evaluator needs per-edge (1,)-pos plus (K,)-neg.

        An extra list wrapper around the negatives silently corrupts the ranking
        and produced MRR ~ 0.007 (below random) in the first smoke run."""
        from tgb.linkproppred.evaluate import Evaluator

        from prss.training.event_loop import _metric_bundle

        evaluator = Evaluator(name="tgbl-wiki")
        # All negatives score below the positive -> rank 1 -> MRR 1.0.
        mrr = _metric_bundle(torch.tensor(1.0),
                             torch.tensor([0.5, 0.4, 0.3, 0.2]), evaluator, "mrr")
        self.assertAlmostEqual(mrr, 1.0, places=4)
        # One negative scores above the positive -> rank 2 -> MRR 0.5.
        mrr2 = _metric_bundle(torch.tensor(0.5),
                              torch.tensor([0.9, 0.1, 0.1, 0.1]), evaluator, "mrr")
        self.assertAlmostEqual(mrr2, 0.5, places=4)

    def test_inference_without_trace_does_not_touch_spectral_state(self):
        torch.manual_seed(2)
        np.random.seed(2)
        from prss.core import PRSSCore
        from prss.hosts.tgn_pyg import PyGTGNAdapter

        (memory, gnn, _, loader, mem_d, time_d, msg_d, emb_d) = self._make()
        core = PRSSCore(self._config(mem_d, time_d, msg_d, emb_d), variant="spectral")
        adapter = PyGTGNAdapter(memory, gnn, core, n_neighbors=4, mem_dim=mem_d,
                                time_dim=time_d, msg_dim=msg_d, emb_dim=emb_d)
        before_r = {t: q.projection().clone() for t, q in core.quotients.items()}
        before_g = {t: q.snapshot()["reader_gram_updates"] for t, q in core.quotients.items()}
        adapter.clear_trace()
        core.set_spectral_updates_allowed(False)
        with torch.no_grad():
            z = adapter.embed(torch.tensor([1, 2]), torch.zeros(2, 0, dtype=torch.long),
                              torch.zeros(0, dtype=torch.long),
                              torch.zeros(0, msg_d))
        self.assertIsNone(adapter.trace)
        for t, q in core.quotients.items():
            self.assertTrue(torch.equal(before_r[t], q.projection()))
            self.assertEqual(before_g[t], q.snapshot()["reader_gram_updates"])


if __name__ == "__main__":
    unittest.main()
