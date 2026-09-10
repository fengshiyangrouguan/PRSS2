"""Joint child-parent boundary fixed-map gates (plan step 4, gate C).

Verifies:
* 1Obs (use_parent=0) and 2Obs (use_parent=1) output the SAME p dimension;
* fingerprint reproducible and seed/buffer-sensitive;
* buffers carry no gradient;
* replacing the parent future changes p_v clearly;
* two distinct real events (same outcome but different counterpart / message /
  time) yield different event signatures and p_v;
* C_v (context) is supplied separately and does not include donor future
  fields (the map only takes a ctx vector — caller responsibility verified by
  construction of the ctx vector in the loss integration test).
"""

import unittest

import numpy as np
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rpbe.pair_maps import BoundaryMaps


def _maps(d_event=96, d_ctx=32, m=128, d_msg=172, seed=0):
    return BoundaryMaps(d_ctx=d_ctx, d_event=d_event, m=m, d_msg=d_msg,
                        delta_t_scale=1.0, msg_scale=1e-3, seed=seed)


def _ev(counterpart, role, delta_t, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"counterpart": counterpart, "role": role, "delta_t": delta_t,
            "message": torch.randn(172, generator=g)}


class TestBoundaryMaps(unittest.TestCase):
    def test_arms_same_output_dim(self):
        fm = _maps()
        ctx = torch.randn(32)
        child = _ev(5, 0, 10.0)
        parent = _ev(7, 1, 20.0)
        p1 = fm.pv(ctx, child, parent, use_parent=0)
        p2 = fm.pv(ctx, child, parent, use_parent=1)
        self.assertEqual(p1.shape, (128,))
        self.assertEqual(p2.shape, (128,))
        self.assertEqual(p1.shape, p2.shape)

    def test_parent_replacement_changes_p(self):
        fm = _maps()
        ctx = torch.randn(32)
        child = _ev(5, 0, 10.0)
        parent_a = _ev(7, 1, 20.0)
        parent_b = _ev(99, 0, 90.0)   # different counterpart/time/role
        pa = fm.pv(ctx, child, parent_a, use_parent=1)
        pb = fm.pv(ctx, child, parent_b, use_parent=1)
        self.assertFalse(torch.equal(pa, pb),
                         "replacing parent future must change p_v")

    def test_1obs_insensitive_to_parent(self):
        """1Obs (use_parent=0) ignores the parent future entirely."""
        fm = _maps()
        ctx = torch.randn(32)
        child = _ev(5, 0, 10.0)
        parent_a = _ev(7, 1, 20.0)
        parent_b = _ev(99, 0, 90.0)
        pa = fm.pv(ctx, child, parent_a, use_parent=0)
        pb = fm.pv(ctx, child, parent_b, use_parent=0)
        self.assertTrue(torch.equal(pa, pb),
                        "1Obs p must be invariant to the parent future")

    def test_event_signature_distinguishes_events(self):
        fm = _maps()
        e1 = fm.event_vector(5, 0, 10.0, torch.randn(172))
        e2 = fm.event_vector(6, 0, 10.0, torch.randn(172))
        self.assertEqual(e1.shape, (fm.d_event,))
        self.assertFalse(torch.equal(e1, e2))
        # same event -> same signature (deterministic)
        msg = torch.randn(172)
        a = fm.event_vector(5, 0, 10.0, msg)
        b = fm.event_vector(5, 0, 10.0, msg)
        self.assertTrue(torch.equal(a, b))

    def test_fingerprint_reproducible_and_seed_sensitive(self):
        a = _maps(seed=1)
        b = _maps(seed=1)
        self.assertEqual(a.isolation_fingerprint(),
                         b.isolation_fingerprint())
        c = _maps(seed=2)
        self.assertNotEqual(a.isolation_fingerprint()["sha256"],
                            c.isolation_fingerprint()["sha256"])

    def test_no_gradient_through_buffers(self):
        fm = _maps()
        ctx = torch.randn(32, requires_grad=True)
        child = _ev(5, 0, 10.0)
        parent = _ev(7, 1, 20.0)
        p = fm.pv(ctx, child, parent, use_parent=1)
        self.assertTrue(torch.isfinite(p).all())
        for name in ("event_hash_table", "rff_w", "rff_b", "msg_proj"):
            self.assertFalse(getattr(fm, name).requires_grad,
                             "{} must be a frozen buffer".format(name))

    def test_mispaired_changes_2obs_but_not_1obs(self):
        """The mispaired control swaps the parent event: 2Obs must move, the
        child marginal (1Obs) must be bit-identical."""
        fm = _maps()
        ctx = torch.randn(32)
        child = _ev(3, 1, 5.0)
        palign = _ev(11, 0, 15.0)
        pmisp = _ev(22, 1, 45.0)
        a2 = fm.pv(ctx, child, palign, use_parent=1)
        b2 = fm.pv(ctx, child, pmisp, use_parent=1)
        a1 = fm.pv(ctx, child, palign, use_parent=0)
        b1 = fm.pv(ctx, child, pmisp, use_parent=0)
        self.assertFalse(torch.equal(a2, b2))
        self.assertTrue(torch.equal(a1, b1))


if __name__ == "__main__":
    unittest.main()
