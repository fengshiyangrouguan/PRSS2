"""Indexing tests for the CONDITIONAL 2Obs construction (review round 5).

The restored design (plan Test D):

    obs1: C = u_{v+1},              Y = u_{v+2}         (both in prompt)
    obs2: C = (u_{v+1}, u_{v+2}, one-update), Y = u_k (labels, EOS out)

    p_h = CountSketch([1; chi(C_h)] (x) phi(Y_h))

These tests pin the MECHANICAL contract: which tokens each measurement
consumes, dimension/shape invariants, frozenness, and that p_2 really
depends on the task target u_k (the bug the round-5 review caught).
Pure torch; no GPU, no model.
"""

import unittest

import torch

from rpbe.llm.dialogue_records import Llmmaps
from rpbe.llm.utterance_embed import UtteranceEmbed


class FakeEmbed(torch.nn.Module):
    """Deterministic token -> vector embedding for indexing tests."""

    def __init__(self, vocab: int, dim: int):
        super().__init__()
        self.vocab = vocab
        self.dim = dim
        w = torch.arange(vocab * dim, dtype=torch.float32).reshape(
            vocab, dim) / (vocab * dim)
        self.weight = torch.nn.Parameter(w, requires_grad=False)

    def forward(self, ids):
        return self.weight[ids]


class TestConditional2Obs(unittest.TestCase):
    def setUp(self):
        self.hidden = 32
        self.embed = FakeEmbed(64, self.hidden)
        self.chi_fn = UtteranceEmbed(self.hidden, d_chi=64, seed=7,
                                     combine_dim=1)
        self.phi_fn = UtteranceEmbed(self.hidden, d_chi=32, seed=107)
        self.maps = Llmmaps(d_chi=64, d_phi=32, m=64, seed=7)
        # three utterances with disjoint token id ranges
        self.u_a = torch.tensor([[1, 2, 3, 4]])
        self.u_b = torch.tensor([[10, 11, 12]])
        self.u_k = torch.tensor([[20, 21, 22, 23, 24]])

    def _p2(self, u_a=None, u_b=None, u_k=None):
        u_a = self.u_a if u_a is None else u_a
        u_b = self.u_b if u_b is None else u_b
        u_k = self.u_k if u_k is None else u_k
        chi2 = self.chi_fn.combine(self.embed, u_a, u_b, tag=1)
        phi2 = self.phi_fn(self.embed, u_k, tag=1)
        return self.maps.pv(chi2[0], phi2[0])

    def test_shapes(self):
        chi1 = self.chi_fn(self.embed, self.u_a, tag=0)
        phi1 = self.phi_fn(self.embed, self.u_b, tag=0)
        chi2 = self.chi_fn.combine(self.embed, self.u_a, self.u_b, tag=1)
        phi2 = self.phi_fn(self.embed, self.u_k, tag=1)
        self.assertEqual(chi1.shape, (1, 64))
        self.assertEqual(phi1.shape, (1, 32))
        self.assertEqual(chi2.shape, (1, 64))
        self.assertEqual(phi2.shape, (1, 32))
        p1 = self.maps.pv(chi1[0], phi1[0])
        p2 = self.maps.pv(chi2[0], phi2[0])
        self.assertEqual(p1.shape, (64,))
        self.assertEqual(p2.shape, (64,))

    def test_p2_consumes_task_target(self):
        """THE round-5 bug contract: phi_2 (and thus p_2) must depend on
        the task target u_k taken from labels."""
        base = self._p2()
        changed = self._p2(u_k=torch.tensor([[30, 31]]))
        self.assertTrue(
            (base - changed).abs().sum() > 1e-6,
            "p_2 does not change when u_k changes")

    def test_p2_consumes_context_pair(self):
        """chi_2 must depend on BOTH u_{v+1} and u_{v+2} (plus tag)."""
        base = self._p2()
        changed_a = self._p2(u_a=torch.tensor([[5, 6]]))
        changed_b = self._p2(u_b=torch.tensor([[13, 14]]))
        self.assertTrue((base - changed_a).abs().sum() > 1e-6)
        self.assertTrue((base - changed_b).abs().sum() > 1e-6)

    def test_p1_consumes_only_its_pair(self):
        """obs1: chi = u_{v+1}, phi = u_{v+2}; u_k must NOT enter p_1."""
        chi1 = self.chi_fn(self.embed, self.u_a, tag=0)
        phi1a = self.phi_fn(self.embed, self.u_b, tag=0)
        phi1b = self.phi_fn(self.embed, self.u_k, tag=0)
        p1_a = self.maps.pv(chi1[0], phi1a[0])
        p1_b = self.maps.pv(chi1[0], phi1b[0])
        self.assertTrue((p1_a - p1_b).abs().sum() > 1e-6)

    def test_frozen(self):
        """chi/phi/p are constants: zero grad, no requires_grad leak."""
        for fn in (self.chi_fn, self.phi_fn, self.maps):
            for p in fn.parameters():
                self.assertFalse(p.requires_grad)
        p2 = self._p2()
        self.assertFalse(p2.requires_grad)

    def test_combine_tag_matters(self):
        """The one-update tag inside chi_2 must change the measurement."""
        c_tag1 = self.chi_fn.combine(self.embed, self.u_a, self.u_b, tag=1)
        c_tag0 = self.chi_fn.combine(self.embed, self.u_a, self.u_b, tag=0)
        self.assertTrue((c_tag1 - c_tag0).abs().sum() > 1e-6)


if __name__ == "__main__":
    unittest.main()
