"""PRSSCore contract: tau indexing, host-agnosticism, identity-like parity."""

import unittest
from pathlib import Path

import torch

from prss.config import InterfaceSpec, PRSSConfig
from prss.core import PRSSCore

SRC_DIR = Path(__file__).resolve().parents[1] / "src"


def make_config():
    return PRSSConfig(
        interfaces={
            "leaf": InterfaceSpec("leaf", raw_dim=4, candidate_dim=4, host_dim=4),
            "conv": InterfaceSpec("conv", raw_dim=4, candidate_dim=8, host_dim=4),
        },
        context_dim=8,
        root_metadata_dim=2,
        parent_local_dim=6,
        relation_count=4,
        relation_dim=8,
        outside_layers=2,
    )


class TestInterfaceSpec(unittest.TestCase):
    def test_candidate_below_host_raises(self):
        with self.assertRaises(ValueError):
            InterfaceSpec("x", raw_dim=2, candidate_dim=2, host_dim=4)

    def test_nonpositive_dims_raise(self):
        with self.assertRaises(ValueError):
            InterfaceSpec("x", raw_dim=0, candidate_dim=4, host_dim=2)

    def test_compression_properties(self):
        comp = InterfaceSpec("x", raw_dim=8, candidate_dim=8, host_dim=4)
        base = InterfaceSpec("y", raw_dim=4, candidate_dim=4, host_dim=4)
        self.assertTrue(comp.dimensional_compression)
        self.assertFalse(base.dimensional_compression)
        self.assertEqual(comp.compression_ratio, 2.0)

    def test_config_key_must_match_name(self):
        with self.assertRaises(ValueError):
            PRSSConfig(interfaces={"a": InterfaceSpec("b", 4, 4, 4)})


class TestHostAgnosticSource(unittest.TestCase):
    def test_core_has_no_host_hardcoding(self):
        text = (SRC_DIR / "prss" / "core.py").read_text()
        for token in ["tgn_layer_", "tgw:", "tgp:"]:
            self.assertNotIn(token, text)

    def test_config_has_no_host_hardcoding(self):
        text = (SRC_DIR / "prss" / "config.py").read_text()
        self.assertNotIn("tgn_layer_", text)


class TestTauIndexing(unittest.TestCase):
    def setUp(self):
        self.core = PRSSCore(make_config(), variant="spectral")

    def test_quotients_keyed_by_tau(self):
        self.assertEqual(set(self.core.quotients.keys()), {"leaf", "conv"})

    def test_base_interface_has_no_reader_or_builder(self):
        self.assertNotIn("leaf", self.core.readers)
        self.assertNotIn("leaf", self.core.builders)
        self.assertIn("conv", self.core.readers)
        self.assertIn("conv", self.core.builders)

    def test_base_interface_gets_vanilla_quotient(self):
        q = self.core.quotients["leaf"]
        self.assertEqual(q.name, "vanilla")

    def test_base_interface_project_is_identity(self):
        h = torch.randn(5, 4)
        z = self.core.project("leaf", h)
        self.assertTrue(torch.equal(z, h))

    def test_make_candidate_passthrough_for_base_interface(self):
        h = torch.randn(5, 4)
        self.assertTrue(torch.equal(self.core.make_candidate("leaf", h), h))


class TestRawEqualsCandidate(unittest.TestCase):
    def test_raw_equals_candidate_has_reader_but_no_builder(self):
        # Synthetic-tree style interface: raw IS the candidate, yet compression
        # (candidate > host) still exists, so a reader must supervise it.
        config = PRSSConfig(
            interfaces={"branch": InterfaceSpec("branch", raw_dim=12,
                                                candidate_dim=12, host_dim=3)},
            context_dim=8, root_metadata_dim=2, parent_local_dim=6,
        )
        core = PRSSCore(config, variant="spectral")
        self.assertNotIn("branch", core.builders)
        self.assertIn("branch", core.readers)
        raw = torch.randn(4, 12)
        self.assertTrue(torch.equal(core.make_candidate("branch", raw), raw))


class TestIdentityParity(unittest.TestCase):
    def test_identity_like_initialization_reproduces_vanilla(self):
        core = PRSSCore(make_config(), variant="spectral")
        vanilla = torch.randn(3, 4)
        flat_preagg = torch.randn(3, 6)
        candidate = core.make_candidate("conv", vanilla, flat_preagg)
        self.assertEqual(candidate.shape, (3, 8))
        self.assertTrue(torch.equal(candidate[:, :4], vanilla))
        z = core.project("conv", candidate)
        self.assertTrue(torch.allclose(z, vanilla, atol=1e-6, rtol=1e-6))

    def test_candidate_width_mismatch_raises(self):
        core = PRSSCore(make_config(), variant="spectral")
        with self.assertRaises(ValueError):
            core.make_candidate("conv", torch.randn(3, 4), torch.randn(3, 5))


class TestVariantAndGates(unittest.TestCase):
    def test_unknown_variant_raises(self):
        with self.assertRaises(ValueError):
            PRSSCore(make_config(), variant="no_such_variant")

    def test_spectral_updates_gate(self):
        core = PRSSCore(make_config(), variant="spectral")
        core.set_spectral_updates_allowed(False)
        result = core.maybe_update(step=100)
        self.assertFalse(any(result.values()))

    def test_aux_contract_matches_variant(self):
        core = PRSSCore(make_config(), variant="spectral")
        use_resp, use_spec = core.aux_contract()
        self.assertTrue(use_resp)
        self.assertTrue(use_spec)


if __name__ == "__main__":
    unittest.main()
