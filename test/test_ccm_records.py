"""L4 acceptance tests: DialogueCutBuilder + Llmmaps + window close
(plan v2 L4: Test D rows / Test E measurement / Test F window path).

One cut -> two horizon rows sharing cut_id and weight 0.5/0.5; both rows
enter the SAME Ky Fan window and the clustered correction runs through
the existing WeightedWelford / latent_z_adjoint path.
"""

import sys
import unittest
from pathlib import Path

_CCM_DIR = Path(__file__).resolve().parents[1] / "third_party" / "ccm"
if str(_CCM_DIR) not in sys.path:
    sys.path.insert(0, str(_CCM_DIR))

import torch

from rpbe.llm.dialogue_records import (DialogueCutBuilder, DialogueMeta,
                                       HORIZON_WEIGHTS, Llmmaps, MEM_TAU)
from rpbe.llm.utterance_embed import UtteranceEmbed
from rpbe.loss import KFMomentWindow


class TestDialogueCutBuilder(unittest.TestCase):
    """Test D: row structure, weights, shared cut_id, k gating."""

    def setUp(self):
        torch.manual_seed(0)
        self.maps = Llmmaps(d_chi=8, d_phi=8, m=4, seed=3)
        self.builder = DialogueCutBuilder(self.maps, z_dim=8)

    def _meta(self, k, sample_id=7):
        return DialogueMeta(sample_id=sample_id, k=k,
                            sum_positions=[(0, 1)] * max(k, 1),
                            utterance_spans=[(0, 1)] * max(k, 1))

    def test_two_rows_share_cut(self):
        z = torch.randn(8)
        chi1 = torch.randn(8)
        chi2 = torch.randn(8)
        rows = self.builder.build(self._meta(6), z, chi1, chi2)
        self.assertEqual(len(rows), 2)
        r1, r2 = rows
        self.assertEqual(r1.cut_id, r2.cut_id)
        self.assertEqual(r1.row_id, r1.cut_id + (1,))
        self.assertEqual(r2.row_id, r2.cut_id + (2,))
        self.assertEqual(r1.tree_id, 7)
        self.assertEqual(r2.tree_id, 7)
        self.assertEqual(r1.tau, MEM_TAU)
        self.assertEqual(r2.tau, MEM_TAU)
        self.assertEqual(r1.weight, HORIZON_WEIGHTS[0])
        self.assertEqual(r2.weight, HORIZON_WEIGHTS[1])
        self.assertEqual(r1.time, 3.0)  # v = k - 3 = 3
        self.assertEqual(r2.context["cut_turn"], 3)
        self.assertTrue(torch.equal(r1.z, z))
        self.assertTrue(torch.equal(r2.z, z))
        self.assertIsNotNone(r1.p_override)
        self.assertIsNotNone(r2.p_override)

    def test_k_gating(self):
        z = torch.randn(8)
        self.assertEqual(self.builder.build(self._meta(3), z, None, None), [])
        rows = self.builder.build(self._meta(4), z,
                                  torch.randn(8), torch.randn(8))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0].time, 1.0)  # v = 1

    def test_occurrence_counter(self):
        z = torch.randn(8)
        r1 = self.builder.build(self._meta(5), z, torch.randn(8),
                                torch.randn(8))
        r2 = self.builder.build(self._meta(5), z, torch.randn(8),
                                torch.randn(8))
        self.assertEqual(r1[0].occurrence_id, 0)
        self.assertEqual(r2[0].occurrence_id, 1)
        self.assertEqual(self.builder.next_oid, 2)

    def test_z_keeps_gradient(self):
        z = torch.randn(8, requires_grad=True)
        rows = self.builder.build(self._meta(5), z, torch.randn(8),
                                  torch.randn(8))
        (rows[0].z.sum() + rows[1].z.sum()).backward()
        self.assertIsNotNone(z.grad)
        self.assertFalse(rows[0].p_override.requires_grad)


class TestLlmmaps(unittest.TestCase):
    """Test E: the fixed measurement p = Sketch([1; chi] (x) phi_h)."""

    def test_pv_manual_reference(self):
        torch.manual_seed(1)
        maps = Llmmaps(d_chi=6, d_phi=5, m=7, seed=9)
        chi = torch.randn(3, 6)
        for h in (1, 2):
            p = maps.pv(chi, h)
            self.assertEqual(tuple(p.shape), (3, 7))
            body = torch.cat([torch.ones(3, 1), chi], dim=1)
            prod = torch.einsum("bd,f->bdf", body,
                                maps.phi_table[h - 1].float()).reshape(3, -1)
            ref = torch.zeros(3, 7)
            ref.index_add_(1, maps.sketch_cols,
                           prod[:, maps.sketch_rows] * maps.sketch_signs)
            ref = ref * maps.scale
            self.assertTrue(torch.allclose(p, ref, rtol=0, atol=0))
        self.assertFalse(torch.equal(maps.phi_table[0], maps.phi_table[1]))

    def test_chi_tag_changes_p2(self):
        torch.manual_seed(2)
        emb = torch.nn.Embedding(64, 16)
        ue = UtteranceEmbed(hidden_dim=16, d_chi=8, seed=5)
        ids = torch.tensor([[3, 5, 9]])
        chi0 = ue(emb, ids, tag=0)
        chi1 = ue(emb, ids, tag=1)
        self.assertFalse(torch.equal(chi0, chi1))  # one-update marker enters


class TestWindowClose(unittest.TestCase):
    """Test F: both horizons of a cut close through WeightedWelford and
    the cut-level adjoint merges them into ONE replay gradient."""

    def test_close_replay_merges_horizons(self):
        torch.manual_seed(11)
        maps = Llmmaps(d_chi=8, d_phi=8, m=4, seed=3)
        builder = DialogueCutBuilder(maps, z_dim=8)
        win = KFMomentWindow({"mem": 8}, min_ratio=2.0, min_abs=4,
                             eps=1e-4, fixed_maps=maps, strict=True,
                             autoclose=False)
        zs = torch.randn(10, 8, dtype=torch.float64)
        for s in range(10):
            meta = DialogueMeta(sample_id=s, k=5,
                                sum_positions=[(0, 1)] * 5,
                                utterance_spans=[(0, 1)] * 5)
            rows = builder.build(meta, zs[s].clone(), torch.randn(8),
                                 torch.randn(8))
            win.add(rows)
        closed, plan, diag = win.close_replay()
        self.assertIn("mem", closed)
        self.assertEqual(diag["mem"]["M_unique"], 10)
        self.assertEqual(diag["mem"]["M_unique_trees"], 10)
        # Cluster degrees of freedom: 10 cuts, each weight 1.0 -> D = 9.
        self.assertAlmostEqual(diag["mem"]["D"], 9.0, places=3)
        # The replay plan emits ONE gradient per cut (merged horizons).
        by_batch = plan["mem"]["by_batch"]
        n_entries = sum(len(b) for b in by_batch)
        self.assertEqual(n_entries, 10)
        # Every cut appears once with a z-dim gradient.
        self.assertTrue(all(g.shape == (8,) for b in by_batch
                            for _, g in b))


class TestSurrogateGradient(unittest.TestCase):
    """The surrogate is numerically zero with the exact window gradient."""

    def test_surrogate_matches_direct(self):
        torch.manual_seed(12)
        maps = Llmmaps(d_chi=8, d_phi=8, m=4, seed=3)
        win = KFMomentWindow({"mem": 8}, min_ratio=2.0, min_abs=4,
                             eps=1e-4, fixed_maps=maps, strict=True,
                             autoclose=False)
        n_cuts = 8
        z_rows = torch.randn(n_cuts, 8, dtype=torch.float64)
        p_rows = torch.stack([maps.pv(torch.randn(8), 1 + (i % 2))
                              for i in range(n_cuts)]).double()
        w = torch.ones(n_cuts, dtype=torch.float64) * 0.5
        cut_ids = [(s, s, "mem") for s in range(n_cuts)]
        # Direct gradient via latent_z_adjoint (window truth).
        from rpbe.loss import latent_z_adjoint, WeightedWelford
        wf = WeightedWelford(8, 4)
        wf.add(z_rows, p_rows, w, cut_ids)
        r = wf.result()
        j, g_by_cut, diag = latent_z_adjoint(
            z_rows, p_rows, w, cut_ids, r["mu_z"], r["mu_p"], r["D"], 1e-4)
        self.assertIsNotNone(j)
        # Surrogate path: numerically zero, gradient = g_by_cut per row.
        z_live = z_rows.clone().requires_grad_(True)
        terms = [(g_by_cut[cid].detach() * z_live[i]).sum()
                 - (g_by_cut[cid].detach() * z_live[i].detach()).sum()
                 for i, cid in enumerate(cut_ids)]
        surr = sum(terms)
        self.assertAlmostEqual(float(surr.detach()), 0.0, places=9)
        surr.backward()
        for i, cid in enumerate(cut_ids):
            self.assertTrue(torch.allclose(z_live.grad[i],
                                           g_by_cut[cid], rtol=1e-6,
                                           atol=1e-8))


if __name__ == "__main__":
    unittest.main()
