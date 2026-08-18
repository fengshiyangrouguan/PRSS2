"""build_auxiliary contract on a toy recursive trace."""

import unittest

import torch

from prss.auxiliary import build_auxiliary
from prss.config import InterfaceSpec, PRSSConfig
from prss.core import PRSSCore
from prss.state import QuotientState, RecursiveOccurrence, RecursiveTrace


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


def make_trace():
    """Root (conv) with two leaf children; one root row."""
    trace = RecursiveTrace()
    root_id = 0
    leaf_1 = 1
    leaf_2 = 2
    trace.add(RecursiveOccurrence(
        occurrence_id=root_id, tau="conv",
        state=QuotientState("conv", raw=torch.zeros(8), candidate=torch.randn(8),
                            quotient=torch.randn(4)),
        local_features=torch.randn(6),
        children=[leaf_1, leaf_2],
        child_relations=[0, 1],
        child_delta_t=[0.0, 3.0],
    ))
    trace.add(RecursiveOccurrence(
        occurrence_id=leaf_1, tau="leaf",
        state=QuotientState("leaf", raw=torch.zeros(4), candidate=torch.randn(4),
                            quotient=torch.randn(4)),
        local_features=torch.randn(6),
    ))
    trace.add(RecursiveOccurrence(
        occurrence_id=leaf_2, tau="leaf",
        state=QuotientState("leaf", raw=torch.zeros(4), candidate=torch.randn(4),
                            quotient=torch.randn(4)),
        local_features=torch.randn(6),
    ))
    trace.roots = [root_id]
    trace.root_rows = [0]
    return trace


class TestBuildAuxiliary(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.core = PRSSCore(make_config(), variant="spectral")
        self.trace = make_trace()
        self.aux = build_auxiliary(
            self.core, self.trace,
            root_metadata=torch.tensor([[0.5, 0.0]]),
            root_labels=torch.tensor([1.0]),
        )

    def test_only_compressive_interface_supervised(self):
        self.assertEqual(set(self.aux.occurrence_counts.keys()), {"conv"})
        self.assertEqual(self.aux.occurrence_counts["conv"], 1)

    def test_matrix_shapes(self):
        B = self.aux.matrices_by_tau["conv"]
        self.assertEqual(B.shape, (1, 1, 8))

    def test_matrix_snapshot_is_detached(self):
        self.assertFalse(self.aux.matrices_by_tau["conv"].requires_grad)

    def test_context_shape(self):
        self.assertEqual(self.aux.contexts_by_tau["conv"].shape, (1, 8))

    def test_losses_are_graph_connected(self):
        self.assertTrue(self.aux.response_loss.requires_grad)
        # Spectral loss is gated: R has not left [I,0].
        self.assertEqual(float(self.aux.spectral_loss.detach()), 0.0)
        self.assertTrue(self.aux.unrestricted_loss.requires_grad)

    def test_backward_reaches_readers_and_candidates(self):
        loss = self.aux.response_loss + 0.1 * self.aux.spectral_loss
        loss.backward()
        reader = self.core.readers["conv"]
        grads = [p.grad for p in reader.parameters() if p.grad is not None]
        self.assertTrue(grads)
        self.assertTrue(all(torch.isfinite(g).all() for g in grads))

    def test_logits_align_with_targets(self):
        self.assertEqual(self.aux.structured_logits.shape, self.aux.targets.shape)
        self.assertEqual(int(self.aux.targets.numel()), 1)

    def test_empty_trace_returns_zero_batch(self):
        empty = RecursiveTrace()
        out = build_auxiliary(self.core, empty,
                              root_metadata=torch.zeros(0, 2),
                              root_labels=torch.zeros(0))
        self.assertEqual(out.targets.numel(), 0)
        self.assertEqual(float(out.response_loss.detach()), 0.0)

    def test_root_metadata_width_mismatch_raises(self):
        with self.assertRaises(ValueError):
            build_auxiliary(self.core, self.trace,
                            root_metadata=torch.zeros(1, 3),
                            root_labels=torch.tensor([1.0]))


if __name__ == "__main__":
    unittest.main()
