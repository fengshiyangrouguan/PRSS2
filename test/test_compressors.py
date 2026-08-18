"""Compressor variant behavior contracts and red lines."""

import unittest

import torch
from torch import nn

from prss.compressors import VARIANT_REGISTRY, InterfaceData, build_compressor
from prss.config import InterfaceSpec, PRSSConfig
from prss.core import PRSSCore
from prss.spectral import projector, random_semi_orthogonal


def spec(d=8, k=4):
    return InterfaceSpec("conv", raw_dim=4, candidate_dim=d, host_dim=k)


def config(**overrides):
    kwargs = dict(
        interfaces={"conv": spec()},
        context_dim=8,
        root_metadata_dim=2,
        parent_local_dim=6,
        relation_count=4,
        relation_dim=8,
        candidate_hidden_dim=16,
        reader_hidden_dim=16,
        outside_layers=2,
    )
    kwargs.update(overrides)
    return PRSSConfig(**kwargs)


class TestRegistry(unittest.TestCase):
    def test_all_five_variants_registered(self):
        self.assertEqual(set(VARIANT_REGISTRY.keys()),
                         {"vanilla", "random", "pca", "direct", "spectral"})

    def test_unknown_variant_raises(self):
        with self.assertRaises(ValueError):
            build_compressor("nope", spec(), config())

    def test_duplicate_registration_raises(self):
        from prss.compressors.base import register_variant, Compressor

        class Duplicate(Compressor):
            name = "vanilla"

        with self.assertRaises(ValueError):
            register_variant(Duplicate)


class TestVanilla(unittest.TestCase):
    def test_identity_like_projection(self):
        c = build_compressor("vanilla", spec(d=8, k=4), config())
        h = torch.randn(3, 8)
        z = c.project(h)
        self.assertTrue(torch.equal(z, h[:, :4]))

    def test_trainable_raises(self):
        c = build_compressor("vanilla", spec(), config())
        with self.assertRaises(RuntimeError):
            c.set_projection_trainable(True)


class TestRandom(unittest.TestCase):
    def test_random_semi_orthogonal_and_frozen(self):
        torch.manual_seed(1)
        c = build_compressor("random", spec(), config())
        rows = c.projection()
        self.assertLess(torch.linalg.norm(rows @ rows.T - torch.eye(4)).item(), 1e-5)
        # Not the identity-like initialization.
        init = torch.zeros(4, 8)
        init[:, :4] = torch.eye(4)
        self.assertGreater((rows - init).abs().max().item(), 0.1)
        # Frozen: updates are no-ops and nothing is trainable.
        self.assertFalse(any(p.requires_grad for p in c.parameters()))
        self.assertFalse(c.maybe_update(100))
        self.assertIsNone(c.update_statistics(100, InterfaceData()))

    def test_trainable_raises(self):
        c = build_compressor("random", spec(), config())
        with self.assertRaises(RuntimeError):
            c.set_projection_trainable(True)


class TestPCA(unittest.TestCase):
    def test_exact_deploy_of_covariance_subspace(self):
        torch.manual_seed(3)
        c = build_compressor("pca", spec(d=8, k=4), config())
        basis = random_semi_orthogonal(4, 8)
        x = torch.randn(512, 4) @ basis
        c.update_statistics(0, InterfaceData(candidates=x))
        self.assertTrue(c.maybe_update(10))
        snap = c.snapshot()
        self.assertEqual(snap["accepted_spectral_step"], 1.0)
        self.assertLess(torch.linalg.norm(projector(c.projection()) -
                                          projector(basis)).item(), 1e-4)

    def test_mean_shift_invariance(self):
        torch.manual_seed(4)
        x = torch.randn(256, 8)
        c1 = build_compressor("pca", spec(d=8, k=4), config(gram_ema_rho=1.0))
        c2 = build_compressor("pca", spec(d=8, k=4), config(gram_ema_rho=1.0))
        c1.update_statistics(0, InterfaceData(candidates=x))
        c2.update_statistics(0, InterfaceData(candidates=x + 7.3))
        self.assertTrue(c1.maybe_update(10))
        self.assertTrue(c2.maybe_update(10))
        self.assertLess(torch.linalg.norm(projector(c1.projection()) -
                                          projector(c2.projection())).item(), 1e-4)

    def test_trainable_raises(self):
        c = build_compressor("pca", spec(), config())
        with self.assertRaises(RuntimeError):
            c.set_projection_trainable(True)


class TestDirect(unittest.TestCase):
    def test_projection_is_parameter(self):
        c = build_compressor("direct", spec(), config())
        self.assertIn("R", dict(c.named_parameters()))

    def test_gradient_step_moves_projection(self):
        torch.manual_seed(5)
        c = build_compressor("direct", spec(), config())
        opt = torch.optim.SGD(c.parameters(), lr=0.1)
        h = torch.randn(16, 8)
        loss = c.project(h).square().mean()
        before = c.projection().detach().clone()
        opt.zero_grad()
        loss.backward()
        opt.step()
        self.assertGreater((c.projection().detach() - before).abs().max().item(), 0.0)

    def test_trainable_flag_contract(self):
        c = build_compressor("direct", spec(), config())
        c.set_projection_trainable(True)  # idempotent
        with self.assertRaises(RuntimeError):
            c.set_projection_trainable(False)

    def test_snapshot_flags_unconstrained(self):
        c = build_compressor("direct", spec(), config())
        snap = c.snapshot()
        self.assertFalse(snap["projection_expected_orthogonal"])


class TestSpectral(unittest.TestCase):
    def test_trainable_raises(self):
        c = build_compressor("spectral", spec(), config())
        with self.assertRaises(RuntimeError):
            c.set_projection_trainable(True)

    def test_update_statistics_uses_reader_gram(self):
        torch.manual_seed(6)
        c = build_compressor("spectral", spec(), config(gram_ema_rho=1.0))
        B = torch.randn(64, 1, 8)
        c.update_statistics(0, InterfaceData(reader_matrices=B))
        self.assertTrue(c.maybe_update(10))
        self.assertGreater(c.snapshot()["gram_trace"], 0.0)


class TestCoreVariantInjection(unittest.TestCase):
    def test_variant_contracts(self):
        contracts = {
            "vanilla": (False, False),
            "random": (True, False),
            "pca": (True, False),
            "direct": (True, False),
            "spectral": (True, True),
        }
        for variant, expected in contracts.items():
            core = PRSSCore(config(), variant=variant)
            self.assertEqual(core.aux_contract(), expected, variant)

    def test_direct_core_trains_projection_parameters(self):
        core = PRSSCore(config(), variant="direct")
        params = [p for n, p in core.quotients["conv"].named_parameters()]
        self.assertTrue(params)
        self.assertTrue(all(p.requires_grad for p in params))


if __name__ == "__main__":
    unittest.main()
