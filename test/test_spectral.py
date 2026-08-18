"""Spectral quotient invariants: SVD equivalence, damping, gates, covariance path."""

import unittest

import torch

from prss.spectral import (
    SpectralQuotient,
    centered_candidate_gram,
    projector,
    procrustes_align_rows,
    random_semi_orthogonal,
    row_orthonormalize,
)


class TestSVDEquivalence(unittest.TestCase):
    def test_operator_bank_svd_equivalence(self):
        torch.manual_seed(0)
        d, k = 6, 2
        q, _ = torch.linalg.qr(torch.randn(d, k), mode="reduced")
        true_rows = q.T
        coeff = torch.randn(40, k)
        stack = coeff @ true_rows
        B = stack[:, None, :]
        s = SpectralQuotient("x", k, d, gram_ema=1.0)
        s.accumulate(B)
        self.assertTrue(s.update(100))
        self.assertLess(torch.linalg.norm(projector(true_rows) - projector(s.R)).item(), 1e-4)
        self.assertAlmostEqual(s.snapshot()["energy_at_k"], 1.0, places=5)

    def test_streaming_gram_matches_explicit_stacked_right_singular_subspace(self):
        torch.manual_seed(4)
        d, k = 9, 3
        banks = [torch.randn(7, 2, d), torch.randn(5, 2, d)]
        B = torch.cat(banks, dim=0)
        s = SpectralQuotient("x", k, d, gram_ema=1.0)
        s.accumulate(B)
        self.assertTrue(s.update(100))
        flat = B.reshape(-1, d)
        _, _, vh = torch.linalg.svd(flat, full_matrices=False)
        p_svd = projector(vh[:k])
        p_stream = projector(s.R)
        self.assertLess(torch.linalg.norm(p_svd - p_stream).item(), 5e-4)


class TestGeometryTools(unittest.TestCase):
    def test_procrustes_aligns_rotated_basis(self):
        torch.manual_seed(7)
        k, d = 3, 8
        rows = random_semi_orthogonal(k, d)
        q = torch.linalg.qr(torch.randn(k, k)).Q
        rotated = q @ rows
        aligned = procrustes_align_rows(rotated, rows)
        self.assertLess(torch.linalg.norm(aligned - rows).item(), 1e-5)

    def test_random_semi_orthogonal_has_orthonormal_rows(self):
        torch.manual_seed(3)
        r = random_semi_orthogonal(4, 10)
        self.assertLess(torch.linalg.norm(r @ r.T - torch.eye(4)).item(), 1e-5)

    def test_row_orthonormalize_is_idempotent(self):
        torch.manual_seed(9)
        r = row_orthonormalize(torch.randn(5, 12))
        again = row_orthonormalize(r)
        self.assertLess(torch.linalg.norm(r - again).item(), 1e-5)

    def test_centered_candidate_gram_is_mean_shift_invariant(self):
        torch.manual_seed(11)
        x = torch.randn(64, 16)
        g1 = centered_candidate_gram(x)
        g2 = centered_candidate_gram(x + 7.3)
        self.assertLess(torch.linalg.norm(g1 - g2).item() / g1.norm().item(), 1e-5)


class TestRIsBuffer(unittest.TestCase):
    def test_projection_is_buffer_not_parameter(self):
        s = SpectralQuotient("x", 2, 6)
        self.assertIsInstance(s.R, torch.Tensor)
        self.assertNotIn("R", dict(s.named_parameters()))

    def test_gradient_does_not_reach_projection(self):
        s = SpectralQuotient("x", 2, 6, gram_ema=1.0)
        with torch.no_grad():
            s.accumulate(torch.randn(32, 1, 6))
            s.update(10)
        h = torch.randn(4, 6, requires_grad=True)
        z = s.project(h)
        z.sum().backward()
        self.assertIsNotNone(h.grad)
        self.assertFalse(s.R.requires_grad)


class TestDampedDeployment(unittest.TestCase):
    def test_damped_update_is_finite_and_energy_monotone(self):
        torch.manual_seed(123)
        d, k = 32, 16
        s = SpectralQuotient("damped", k, d, gram_ema=1.0, spectral_step_size=0.25)
        for step in range(1, 21):
            B = torch.randn(64, 1, d)
            s.accumulate(B)
            old = s.R.detach().clone()
            g = s.G.detach().double()
            old_score = torch.trace(projector(old.double()) @ g)
            self.assertTrue(s.update(step))
            new_score = torch.trace(projector(s.R.detach().double()) @ g)
            self.assertTrue(torch.isfinite(s.R).all())
            self.assertTrue(torch.isfinite(s.G).all())
            self.assertGreaterEqual(new_score.item(), old_score.item() - 1e-9)
            self.assertGreaterEqual(s.snapshot()["accepted_spectral_step"], 0.0)
            self.assertLessEqual(s.snapshot()["accepted_spectral_step"], 0.25)

    def test_exact_deploy_at_step_size_one(self):
        torch.manual_seed(5)
        d, k = 12, 4
        s = SpectralQuotient("exact", k, d, gram_ema=1.0, spectral_step_size=1.0)
        B = torch.randn(200, 1, d)
        s.accumulate(B)
        self.assertTrue(s.update(1))
        flat = B.reshape(-1, d)
        _, _, vh = torch.linalg.svd(flat, full_matrices=False)
        self.assertLess(torch.linalg.norm(projector(s.R) - projector(vh[:k])).item(), 1e-4)


class TestSpectralLossGate(unittest.TestCase):
    def test_spectral_loss_is_zero_before_deployment(self):
        q = SpectralQuotient("gate", host_dim=2, candidate_dim=4, gram_ema=0.05)
        B = torch.randn(3, 1, 4, requires_grad=True)
        loss = q.spectral_loss(B)
        self.assertEqual(float(loss.detach()), 0.0)
        loss.backward()
        self.assertTrue(torch.isfinite(B.grad).all())

    def test_rejected_solve_does_not_activate_loss(self):
        q = SpectralQuotient("gate2", host_dim=2, candidate_dim=4, gram_ema=1.0,
                             spectral_step_size=0.25)
        # Counting a solve attempt must not by itself make R=[I,0] a spec target.
        q.spectral_updates_t.fill_(3)
        B = torch.randn(5, 1, 4, requires_grad=True)
        self.assertEqual(float(q.spectral_loss(B).detach()), 0.0)

    def test_near_zero_reader_gradient_is_finite_after_deployment(self):
        q = SpectralQuotient("tiny", host_dim=2, candidate_dim=4, gram_ema=1.0)
        with torch.no_grad():
            q.accumulate(torch.randn(16, 1, 4))
            q.update(1)
        B = (torch.randn(8, 1, 4) * 1e-10).requires_grad_()
        loss = q.spectral_loss(B)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(B.grad).all())


class TestCovariancePath(unittest.TestCase):
    def test_accumulate_covariance_updates_same_state_machine(self):
        torch.manual_seed(0)
        d, k = 8, 3
        q = SpectralQuotient("cov", host_dim=k, candidate_dim=d, gram_ema=1.0)
        # Low-rank candidate stream: covariance lives in a known k-dim subspace.
        basis = random_semi_orthogonal(k, d)
        x = torch.randn(256, k) @ basis + 0.01 * torch.randn(256, d)
        q.accumulate_covariance(centered_candidate_gram(x))
        self.assertEqual(int(q.reader_gram_updates_t.item()), 1)
        self.assertTrue(q.update(50))
        cov = centered_candidate_gram(x)
        vals = torch.linalg.eigvalsh(cov + 1e-8 * torch.eye(d)).clamp_min(0.0)
        top = torch.sort(vals, descending=True).values[:k]
        expected = float(top.sum().item() / max(vals.sum().item(), 1e-12))
        snap = q.snapshot()
        self.assertGreater(snap["energy_at_k"], 0.9)
        self.assertAlmostEqual(snap["energy_at_k"], expected, places=4)

    def test_covariance_shape_mismatch_raises(self):
        q = SpectralQuotient("badcov", host_dim=3, candidate_dim=8)
        with self.assertRaises(ValueError):
            q.accumulate_covariance(torch.eye(5))


if __name__ == "__main__":
    unittest.main()
