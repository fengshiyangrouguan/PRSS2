"""FixedMaps contracts: determinism, joint sensitivity, no gradient."""

import unittest

import torch

from rpbe.config import RPBConfig
from rpbe.maps import FixedMaps


def make_cfg(seed=0, **kw):
    return RPBConfig(
        interfaces={"tjo:layer0": 8, "tjo:layer1": 8, "tjo:layer2": 8},
        own_dims={"tjo:layer0": 8, "tjo:layer1": 8, "tjo:layer2": 8},
        d_c=32, d_f=32, m=256, rpbe_seed=seed, **kw)


def base_context(**kw):
    ctx = {"delta_t": 12345.0, "counterpart": 77, "role": 0,
           "query_type": 1}
    ctx.update(kw)
    return ctx


class TestDeterminism(unittest.TestCase):
    def test_same_seed_same_output(self):
        a = FixedMaps(make_cfg(seed=5))
        b = FixedMaps(make_cfg(seed=5))
        ctx = base_context()
        for y in (0.0, 1.0):
            self.assertTrue(torch.equal(a.psi(ctx, y), b.psi(ctx, y)))

    def test_different_seed_different_output(self):
        a = FixedMaps(make_cfg(seed=5))
        b = FixedMaps(make_cfg(seed=6))
        ctx = base_context()
        self.assertFalse(torch.equal(a.psi(ctx, 1.0), b.psi(ctx, 1.0)))

    def test_fingerprint_stable_and_seed_sensitive(self):
        a = FixedMaps(make_cfg(seed=5))
        b = FixedMaps(make_cfg(seed=5))
        c = FixedMaps(make_cfg(seed=6))
        self.assertEqual(a.isolation_fingerprint(), b.isolation_fingerprint())
        self.assertNotEqual(a.isolation_fingerprint(), c.isolation_fingerprint())

    def test_fingerprint_detects_buffer_and_scale_changes(self):
        a = FixedMaps(make_cfg(seed=5))
        fp0 = a.isolation_fingerprint()
        # Flip one entry deep in a buffer (past the old 4096-entry window).
        with torch.no_grad():
            a.categorical_c[-1, -1] *= -1
        self.assertNotEqual(a.isolation_fingerprint(), fp0,
                            "buffer change must change the fingerprint")
        # A different measurement scale must change it too.
        b = FixedMaps(make_cfg(seed=5, delta_t_scale=2.5))
        self.assertNotEqual(b.isolation_fingerprint(), fp0)


class TestPsiBehavior(unittest.TestCase):
    def setUp(self):
        self.maps = FixedMaps(make_cfg(seed=1))

    def test_shape_and_finiteness(self):
        out = self.maps.psi(base_context(), 1.0)
        self.assertEqual(tuple(out.shape), (256,))
        self.assertTrue(torch.isfinite(out).all())
        self.assertGreater(out.abs().sum().item(), 0.0)

    def test_joint_sensitivity(self):
        ctx = base_context()
        p0 = self.maps.psi(ctx, 0.0)
        p1 = self.maps.psi(ctx, 1.0)
        p2 = self.maps.psi(base_context(counterpart=78), 1.0)
        p3 = self.maps.psi(base_context(delta_t=99999.0), 1.0)
        p4 = self.maps.psi(base_context(query_type=0), 1.0)
        self.assertFalse(torch.equal(p0, p1), "y must change the sketch")
        self.assertFalse(torch.equal(p1, p2), "counterpart must change it")
        self.assertFalse(torch.equal(p1, p3), "delta_t must change it")
        self.assertFalse(torch.equal(p1, p4), "query_type must change it")

    def test_no_gradient(self):
        out = self.maps.psi(base_context(), 0.0)
        self.assertIsNone(out.grad_fn)
        c = self.maps.context_vector(base_context())
        f = self.maps.future_vector(1.0)
        self.assertIsNone(c.grad_fn)
        self.assertIsNone(f.grad_fn)

    def test_category_hash_collapses_large_ids(self):
        # Counterpart ids far beyond the bin count still hash deterministically.
        a = self.maps.psi(base_context(counterpart=77), 1.0)
        b = self.maps.psi(base_context(counterpart=77 + 4096), 1.0)
        self.assertTrue(torch.equal(a, b))

    def test_pv_batch_matches_pv_per_row(self):
        contexts = [base_context(counterpart=i, delta_t=1000.0 * (i + 1),
                                 role=i % 2, query_type=i % 2)
                    for i in range(6)]
        outcomes = [0.0, 1.0, 1.0, 0.0, 1.0, 0.0]
        batch = self.maps.pv_batch(contexts, outcomes)
        for i, (c, y) in enumerate(zip(contexts, outcomes)):
            self.assertTrue(torch.equal(batch[i], self.maps.psi(c, y)),
                            "pv_batch row {} diverges from pv".format(i))


if __name__ == "__main__":
    unittest.main()
