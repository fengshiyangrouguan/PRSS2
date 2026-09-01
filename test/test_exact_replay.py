"""Exact-replay implementation contracts (review P2, gates 1-2).

Gate 2 (replay gradient == direct backprop) is locked at the surrogate
assembly level here: the pass-2 auxiliary must produce EXACTLY the
cut-level z gradient recorded in the replay plan (times the -lambda*K
and rank coefficients).  Gate 5 (state restore) is locked for buffers,
the occurrence counter and the memory backup/restore round-trip.  The
full two-pass forward equivalence (gate 1) needs the GPU host and is
checked by the sprint/cloud validation.
"""

import unittest

import torch

from rpbe.training.jodie_loop import JodieNodeClassificationLoop


class _Cut:
    tau = "t"
    occurrence_id = 7

    def __init__(self, z):
        self.z = z


class TestExactReplaySurrogate(unittest.TestCase):
    def _loop(self):
        loop = object.__new__(JodieNodeClassificationLoop)
        loop.device = torch.device("cpu")
        loop._tau_coeff = {"t": 0.5}
        loop.lambda_kf = 2.0
        return loop

    def test_surrogate_numerically_zero_with_plan_gradient(self):
        loop = self._loop()
        z = torch.randn(4, requires_grad=True)
        g = torch.randn(4)
        plan = {"t": {"by_batch": [[(7, g)]]}}
        auxiliary, n_terms, align = loop._batch_surrogate_exact(
            [_Cut(z)], plan, group_k=5, batch_offset=0)
        self.assertEqual(n_terms, 1)
        self.assertEqual(align, (1, 1))
        self.assertAlmostEqual(float(auxiliary.detach()), 0.0, places=6)
        auxiliary.backward()
        expected = -loop.lambda_kf * 5.0 * 0.5 * g
        self.assertTrue(torch.allclose(z.grad, expected, atol=1e-6))

    def test_surrogate_skips_missing_occurrence(self):
        loop = self._loop()
        z = torch.randn(4, requires_grad=True)
        plan = {"t": {"by_batch": [[(99, torch.randn(4))]]}}
        auxiliary, n_terms, align = loop._batch_surrogate_exact(
            [_Cut(z)], plan, group_k=5, batch_offset=0)
        self.assertEqual(n_terms, 0)
        self.assertEqual(align, (1, 0))  # planned but not matched
        self.assertEqual(float(auxiliary.detach()), 0.0)

    def test_surrogate_accumulates_per_tau_coeff(self):
        loop = self._loop()
        loop._tau_coeff = {"t": 0.5, "u": 0.25}
        z = torch.randn(4, requires_grad=True)
        plan = {"t": {"by_batch": [[(7, torch.randn(4))]]},
                "u": {"by_batch": [[(8, torch.randn(4))]]}}
        auxiliary, n_terms, align = loop._batch_surrogate_exact(
            [_Cut(z)], plan, group_k=5, batch_offset=0)
        self.assertEqual(n_terms, 1)  # occurrence 8 not present
        self.assertEqual(align, (2, 1))
        self.assertAlmostEqual(float(auxiliary.detach()), 0.0, places=6)


class TestGroupStateRoundtrip(unittest.TestCase):
    def _fixtures(self):
        class _BufModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("run", torch.tensor([1.0, 2.0]))

        class _Memory:
            def backup_memory(self):
                return ("mem",)

            def restore_memory(self, backup):
                self.got = backup

        class _TGN(torch.nn.Module):
            use_memory = True

            def __init__(self):
                super().__init__()
                self.memory = _Memory()

        adapter = _BufModule()
        adapter.compressor = _BufModule()
        adapter._next_oid = 42
        tgn = _TGN()
        loop = object.__new__(JodieNodeClassificationLoop)
        loop.tgn = tgn
        loop.adapter = adapter
        return loop, adapter, tgn

    def test_buffers_oid_and_memory_restore(self):
        loop, adapter, tgn = self._fixtures()
        state = loop._save_group_state()
        adapter.run.add_(5.0)
        adapter.compressor.run.mul_(0.0)
        adapter._next_oid = 99
        loop._restore_group_state(state)
        self.assertTrue(torch.allclose(
            adapter.run, torch.tensor([1.0, 2.0])))
        self.assertTrue(torch.allclose(
            adapter.compressor.run, torch.tensor([1.0, 2.0])))
        self.assertEqual(adapter._next_oid, 42)
        self.assertEqual(tgn.memory.got, ("mem",))


class TestEstimatorSwitch(unittest.TestCase):
    def test_unknown_estimator_rejected(self):
        class _FakeOpt:
            param_groups = [{"params": []}]

        with self.assertRaises(ValueError):
            JodieNodeClassificationLoop(
                tgn=None, decoder=None, repr_optimizer=_FakeOpt(),
                head_optimizer=_FakeOpt(), device=torch.device("cpu"),
                batch_size=8, n_neighbors=4, grad_clip=5.0, monitor=None,
                seed=0, kf_estimator="bogus")


if __name__ == "__main__":
    unittest.main()
