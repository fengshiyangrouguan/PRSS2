import os
import sys
import unittest
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
TGN = ROOT / 'official_tgn' / 'source'
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(TGN)); sys.path.insert(0, str(ROOT/'experiments'))

from prss.spectral import SpectralQuotient, projector
from prss import PRSSCore, PRSSTGNEmbeddingAdapter
from prss.candidate import ExactPreAggregationCandidate
from model.tgn import TGN
from utils.data_processing import Data
from utils.utils import get_neighbor_finder, MLP
from train_supervised_prss_switch import full_official_embedding_call


class TestSpectral(unittest.TestCase):
    def test_operator_bank_svd_equivalence(self):
        torch.manual_seed(0)
        d, k = 6, 2
        q, _ = torch.linalg.qr(torch.randn(d, k), mode='reduced')
        true_rows = q.T
        coeff = torch.randn(40, k)
        stack = coeff @ true_rows
        B = stack[:, None, :]
        s = SpectralQuotient('x', k, d, gram_ema=1.0)
        s.accumulate(B)
        self.assertTrue(s.update(100))
        self.assertLess(torch.linalg.norm(projector(true_rows)-projector(s.R)).item(), 1e-4)
        self.assertAlmostEqual(s.snapshot()['energy_at_k'], 1.0, places=5)

    def test_streaming_gram_matches_explicit_stacked_right_singular_subspace(self):
        torch.manual_seed(4)
        d, k = 9, 3
        banks = [torch.randn(7, 2, d), torch.randn(5, 2, d)]
        # Use gram_ema=1 and one combined accumulation so this is exact rather than EMA-weighted.
        B = torch.cat(banks, dim=0)
        s = SpectralQuotient('x', k, d, gram_ema=1.0)
        s.accumulate(B)
        self.assertTrue(s.update(100))
        flat = B.reshape(-1, d)
        _, _, vh = torch.linalg.svd(flat, full_matrices=False)
        p_svd = projector(vh[:k])
        p_stream = projector(s.R)
        self.assertLess(torch.linalg.norm(p_svd-p_stream).item(), 5e-4)


class TestCandidate(unittest.TestCase):
    def test_candidate_consumes_all_official_aggregate_inputs(self):
        b, n, h, t, e, d = 2, 3, 4, 4, 5, 7
        c = ExactPreAggregationCandidate(h, e, t, n, d, hidden_dim=8)
        src = torch.randn(b, h)
        src_time = torch.randn(b, 1, t)
        nbr = torch.randn(b, n, h)
        et = torch.randn(b, n, t)
        ef = torch.randn(b, n, e)
        mask = torch.zeros(b, n, dtype=torch.bool)
        x = c.exact_preagg(src, src_time, nbr, et, ef, mask)
        self.assertEqual(x.shape[-1], h+t+n*h+n*t+n*e+n)
        vanilla = torch.randn(b, h)
        out = c(vanilla, src, src_time, nbr, et, ef, mask)
        self.assertTrue(torch.equal(out[:, :h], vanilla))


class TestTGNAdapter(unittest.TestCase):
    def _make(self, use_memory=False, n_layers=2, n_neighbors=2):
        torch.manual_seed(1); np.random.seed(1)
        n_nodes, dim = 8, 4
        node = np.random.randn(n_nodes, dim).astype(np.float32)
        edge = np.random.randn(10, dim).astype(np.float32); edge[0] = 0
        data = Data(
            np.array([1,2,1,3,2]), np.array([2,3,4,4,5]),
            np.array([1.,2.,3.,4.,5.]), np.array([1,2,3,4,5]), np.zeros(5))
        finder = get_neighbor_finder(data, uniform=False, max_node_idx=n_nodes-1)
        tgn = TGN(finder, node, edge, torch.device('cpu'), n_layers=n_layers, n_heads=2,
                  dropout=0.0, use_memory=use_memory, memory_dimension=dim,
                  embedding_module_type='graph_attention', message_function='identity',
                  aggregator_type='last', n_neighbors=n_neighbors)
        return tgn, dim

    def test_identity_initialization_matches_official_recursive_forward(self):
        tgn, dim = self._make(False, 2, 2)
        tgn.eval()
        src = np.array([1,3]); ts = np.array([6.,6.])
        with torch.no_grad():
            official = tgn.embedding_module.compute_embedding(None, src, ts, n_layers=2, n_neighbors=2)
        host = tgn.embedding_module
        core = PRSSCore(dim, dim, dim, 2, 2, candidate_dim=6, candidate_hidden=8,
                        context_dim=8, reader_hidden=8)
        adapter = PRSSTGNEmbeddingAdapter(host, core)
        adapter.eval()
        with torch.no_grad():
            got = adapter.compute_embedding(None, src, ts, n_layers=2, n_neighbors=2)
        self.assertTrue(torch.allclose(official, got, atol=1e-6, rtol=1e-6),
                        (official-got).abs().max().item())

    def test_full_tgn_call_and_memory_semantics_match_at_identity_initialization(self):
        vanilla, dim = self._make(True, 1, 2)
        wrapped, _ = self._make(True, 1, 2)
        wrapped.load_state_dict(vanilla.state_dict())
        core = PRSSCore(dim, dim, dim, 2, 1, candidate_dim=6, candidate_hidden=8,
                        context_dim=8, reader_hidden=8)
        wrapped.embedding_module = PRSSTGNEmbeddingAdapter(wrapped.embedding_module, core)
        vanilla.eval(); wrapped.eval()
        s=np.array([1,2]); d=np.array([2,3]); t=np.array([6.,7.]); idx=np.array([1,2])
        with torch.no_grad():
            a = vanilla.compute_temporal_embeddings(s,d,d,t,idx,2)
            b = wrapped.compute_temporal_embeddings(s,d,d,t,idx,2)
        for x,y in zip(a,b):
            self.assertTrue(torch.allclose(x,y,atol=1e-6,rtol=1e-6), (x-y).abs().max().item())
        self.assertTrue(torch.allclose(vanilla.memory.memory, wrapped.memory.memory, atol=1e-6))
        self.assertTrue(torch.allclose(vanilla.memory.last_update, wrapped.memory.last_update, atol=1e-6))
        keys=set(vanilla.memory.messages)|set(wrapped.memory.messages)
        for key in keys:
            va=vanilla.memory.messages[key]; vb=wrapped.memory.messages[key]
            self.assertEqual(len(va),len(vb))
            for (ma,ta),(mb,tb) in zip(va,vb):
                self.assertTrue(torch.allclose(ma,mb,atol=1e-6,rtol=1e-6))
                self.assertTrue(torch.allclose(ta,tb,atol=1e-6,rtol=1e-6))

    def test_trace_only_selected_top_source(self):
        tgn, dim = self._make(False, 1, 2)
        core = PRSSCore(dim,dim,dim,2,1,candidate_dim=6,candidate_hidden=8,context_dim=8,reader_hidden=8)
        adapter = PRSSTGNEmbeddingAdapter(tgn.embedding_module,core)
        adapter.set_trace_source_rows([0])
        adapter.compute_embedding(None,np.array([1,2,3]),np.array([6.,6.,6.]),1,2)
        self.assertEqual(adapter.trace.root_rows,[0])
        self.assertEqual(len(adapter.trace.roots),1)

    def test_inference_without_trace_does_not_touch_spectral_state(self):
        tgn, dim = self._make(False, 1, 2)
        core = PRSSCore(dim,dim,dim,2,1,candidate_dim=6,candidate_hidden=8,context_dim=8,reader_hidden=8)
        adapter = PRSSTGNEmbeddingAdapter(tgn.embedding_module,core)
        before_R=core.quotients['1'].R.clone(); before_G=core.quotients['1'].G.clone()
        adapter.clear_trace()
        with torch.no_grad():
            adapter.compute_embedding(None,np.array([1,2]),np.array([6.,6.]),1,2)
        self.assertIsNone(adapter.trace)
        self.assertTrue(torch.equal(before_R,core.quotients['1'].R))
        self.assertTrue(torch.equal(before_G,core.quotients['1'].G))


class TestMotherDerivedVanilla(unittest.TestCase):
    def test_one_frozen_decoder_step_matches_upstream_block(self):
        torch.manual_seed(11); np.random.seed(11)
        n_nodes, dim = 9, 4
        node=np.random.randn(n_nodes,dim).astype(np.float32)
        edge=np.random.randn(10,dim).astype(np.float32); edge[0]=0
        data=Data(np.array([1,2,1,3]),np.array([2,3,4,5]),np.array([1.,2.,3.,4.]),
                  np.array([1,2,3,4]),np.array([0.,1.,0.,0.]))
        finder=get_neighbor_finder(data,False,max_node_idx=n_nodes-1)
        def make():
            return TGN(finder,node,edge,torch.device('cpu'),n_layers=1,n_heads=2,dropout=0.0,
                       use_memory=True,memory_dimension=dim,embedding_module_type='graph_attention',
                       message_function='identity',aggregator_type='last',n_neighbors=2)
        a=make(); b=make(); b.load_state_dict(a.state_dict()); a.eval(); b.eval()
        da=MLP(dim,drop=0.0); db=MLP(dim,drop=0.0); db.load_state_dict(da.state_dict())
        oa=torch.optim.Adam(da.parameters(),lr=3e-4); ob=torch.optim.Adam(db.parameters(),lr=3e-4)
        src=np.array([1,2]); dst=np.array([2,3]); ts=np.array([5.,6.]); idx=np.array([1,2]); y=torch.tensor([0.,1.])
        oa.zero_grad()
        with torch.no_grad():
            sa,_,_=a.compute_temporal_embeddings(src,dst,dst,ts,idx,2)
        pa=da(sa).sigmoid(); la=torch.nn.BCELoss()(pa,y); la.backward(); oa.step()
        ob.zero_grad()
        sb,_,_=full_official_embedding_call(b,src,dst,ts,idx,2,grad_enabled=False)
        pb=db(sb).sigmoid(); lb=torch.nn.BCELoss()(pb,y); lb.backward(); ob.step()
        self.assertTrue(torch.equal(sa,sb))
        self.assertTrue(torch.equal(pa,pb))
        self.assertEqual(float(la.detach()),float(lb.detach()))
        for xa,xb in zip(da.parameters(),db.parameters()): self.assertTrue(torch.equal(xa,xb))
        self.assertTrue(torch.equal(a.memory.memory,b.memory.memory))
        self.assertTrue(torch.equal(a.memory.last_update,b.memory.last_update))


class TestPackaging(unittest.TestCase):
    def test_no_alternate_reduction_runtime_import(self):
        paths = list((ROOT/'prss').glob('*.py')) + [ROOT/'experiments'/'train_supervised_prss_switch.py']
        text = '\n'.join(p.read_text().lower() for p in paths)
        for token in ['sklearn.decomposition', 'principalcomponentanalysis']:
            self.assertNotIn(token, text)


if __name__ == '__main__':
    unittest.main()


def test_layer0_is_not_a_predictive_reader_interface():
    """The leaf/base state has d0==k0; PRSS starts at recursive aggregate layers >=1."""
    from prss.core import PRSSCore
    core = PRSSCore(host_dim=8, edge_dim=3, time_dim=4, n_neighbors=2, n_layers=2,
                    candidate_dim=12, context_dim=6, reader_hidden=8)
    assert "0" not in core.readers
    assert "0" not in core.unrestricted
    assert "1" in core.readers and "2" in core.readers
    assert core.quotients["0"].dimensional_compression is False


def test_spectral_loss_waits_for_first_solve_and_tiny_reader_gradient_is_finite():
    import torch
    from prss.spectral import SpectralQuotient

    q = SpectralQuotient('x', host_dim=2, candidate_dim=4, gram_ema=0.05)
    B0 = torch.randn(3, 1, 4, requires_grad=True)
    loss0 = q.spectral_loss(B0)
    assert float(loss0.detach()) == 0.0
    loss0.backward()
    assert torch.isfinite(B0.grad).all()

    # Establish a data-driven quotient first.
    with torch.no_grad():
        q.accumulate(torch.randn(16, 1, 4))
        assert q.update(1)

    # Near-zero B was the unstable corner in the old ratio gradient.
    B = (torch.randn(8, 1, 4) * 1e-10).requires_grad_()
    loss = q.spectral_loss(B)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(B.grad).all()


def test_damped_spectral_deployment_is_finite_and_energy_monotone():
    torch.manual_seed(123)
    d, k = 32, 16
    s = SpectralQuotient('damped', k, d, gram_ema=1.0, spectral_step_size=0.25)
    for step in range(1, 21):
        B = torch.randn(64, 1, d)
        s.accumulate(B)
        old = s.R.detach().clone()
        g = s.G.detach().double()
        old_score = torch.trace(projector(old.double()) @ g)
        assert s.update(step)
        new_score = torch.trace(projector(s.R.detach().double()) @ g)
        assert torch.isfinite(s.R).all()
        assert torch.isfinite(s.G).all()
        assert new_score + 1e-9 >= old_score
        assert 0.0 <= s.snapshot()['accepted_spectral_step'] <= 0.25


def test_rejected_solve_does_not_activate_spectral_loss_against_identity_initialization():
    import torch
    from prss.spectral import SpectralQuotient
    q = SpectralQuotient('gate', host_dim=2, candidate_dim=4, gram_ema=1.0, spectral_step_size=0.25)
    # Counting a solve attempt must not by itself make the arbitrary R=[I,0] a spec target.
    q.spectral_updates_t.fill_(3)
    B = torch.randn(5, 1, 4, requires_grad=True)
    loss = q.spectral_loss(B)
    assert float(loss.detach()) == 0.0


def test_damped_spectral_update_makes_nonzero_energy_improving_move_on_misaligned_gram():
    import torch
    from prss.spectral import SpectralQuotient, projector
    torch.manual_seed(77)
    d, k = 8, 3
    # Predictive bank lives mostly in the last coordinates, deliberately misaligned with R0=[I,0].
    B = torch.zeros(128, 1, d)
    B[:, 0, 5:] = torch.randn(128, 3)
    q = SpectralQuotient('move', k, d, gram_ema=1.0, spectral_step_size=0.25)
    q.accumulate(B)
    old = q.R.clone()
    g = q.G.double()
    before = torch.trace(projector(old.double()) @ g)
    assert q.update(200)
    after = torch.trace(projector(q.R.double()) @ g)
    assert q.snapshot()['accepted_spectral_step'] > 0.0
    assert q.snapshot()['projector_distance'] > 0.0
    assert after > before
    assert q._has_deployed_data_driven_quotient()
