"""Known-subspace identification gate (spec Phase 2): D, E -> A -> Y.

D and E are marginally uninformative about Y; their interaction through a known
k-dimensional subspace determines Y.  PRSS (spectral) must recover that subspace;
the random/pca/direct task baselines provide the reference frame.

This is a method-identification test, not natural-data evidence.
"""

import unittest

import torch
from torch import nn
from torch.nn import functional as F

from prss.compressors import InterfaceData
from prss.config import InterfaceSpec, PRSSConfig
from prss.core import PRSSCore
from prss.spectral import principal_angles, random_semi_orthogonal

D = 12
K = 3


def make_true_rows(device):
    return torch.linalg.qr(torch.randn(D, K, device=device), mode="reduced").Q.T


def sample_batch(batch_size, true_rows):
    device = true_rows.device
    left = torch.randn(batch_size, D, device=device)
    right_signal = torch.randn(batch_size, K, device=device)
    right_noise = torch.randn(batch_size, D, device=device)
    p = true_rows.T @ true_rows
    right_noise = right_noise @ (torch.eye(D, device=device) - p)
    right = right_signal @ true_rows + right_noise
    target = ((left @ true_rows.T) * right_signal).sum(dim=-1) / (K ** 0.5)
    return left, right, target


def subspace_coverage(rows, true_rows):
    return float((rows @ true_rows.T).square().sum() / true_rows.shape[0])


def make_config():
    return PRSSConfig(
        interfaces={
            "branch": InterfaceSpec("branch", raw_dim=D, candidate_dim=D, host_dim=K),
            "parent": InterfaceSpec("parent", raw_dim=4, candidate_dim=4, host_dim=4),
        },
        context_dim=32,
        root_metadata_dim=1,
        parent_local_dim=2,
        relation_count=3,
        relation_dim=8,
        reader_hidden_dim=64,
        lambda_spec=0.05,
        gram_ema_rho=0.1,
        spectral_update_interval=20,
        spectral_warmup_steps=40,
        ridge_eps=1e-6,
    )


class SyntheticPRSS(nn.Module):
    def __init__(self, variant="spectral"):
        super().__init__()
        self.prss = PRSSCore(make_config(), variant=variant)
        self.aggregator = nn.Sequential(
            nn.Linear(2 * K + 2, 32), nn.GELU(), nn.Linear(32, 4))
        self.readout = nn.Linear(4, 1)

    def forward_components(self, left, right):
        batch_size = len(left)
        left_cand = self.prss.make_candidate("branch", left)
        right_cand = self.prss.make_candidate("branch", right)
        left_z = self.prss.project("branch", left_cand)
        right_z = self.prss.project("branch", right_cand)
        parent_local = torch.zeros(batch_size, 2, device=left.device)
        parent_raw = self.aggregator(torch.cat([left_z, right_z, parent_local], dim=-1))
        parent_cand = self.prss.make_candidate("parent", parent_raw)
        task_prediction = self.readout(self.prss.project("parent", parent_cand)).squeeze(-1)

        root_metadata = torch.zeros(batch_size, 1, device=left.device)
        parent_context = self.prss.outside.root_context(root_metadata, "parent")
        right_summary = self.prss.outside.summarize_siblings(
            {"branch": right_cand.unsqueeze(-2)}, parent_context)
        left_summary = self.prss.outside.summarize_siblings(
            {"branch": left_cand.unsqueeze(-2)}, parent_context)
        left_context = self.prss.outside.child_context(
            parent_context, parent_local, 1, 1.0, right_summary, "branch")
        right_context = self.prss.outside.child_context(
            parent_context, parent_local, 2, 1.0, left_summary, "branch")

        left_logits, left_matrix, _ = self.prss.structured_read(
            "branch", left_context, left_cand)
        right_logits, right_matrix, _ = self.prss.structured_read(
            "branch", right_context, right_cand)
        unrestricted = torch.cat([
            self.prss.unrestricted_read("branch", left_context.detach(), left_cand.detach()),
            self.prss.unrestricted_read("branch", right_context.detach(), right_cand.detach()),
        ], dim=0)
        return {
            "task": task_prediction,
            "structured": torch.cat([left_logits, right_logits], dim=0).squeeze(-1),
            "unrestricted": unrestricted.squeeze(-1),
            "branch_readers": torch.cat([left_matrix, right_matrix], dim=0),
            "branch_candidates": torch.cat([left_cand, right_cand], dim=0),
        }


def train_task_baseline(kind, true_rows, steps, batch_size, learning_rate):
    d = true_rows.shape[1]
    k = true_rows.shape[0]
    device = true_rows.device
    if kind == "direct":
        projection = nn.Linear(d, k, bias=False).to(device)
        parameters = list(projection.parameters())
        project = projection
    else:
        if kind == "random":
            rows = random_semi_orthogonal(k, d, device=device)
        elif kind == "pca":
            left, right, _ = sample_batch(8192, true_rows)
            data = torch.cat([left, right], dim=0)
            _, _, vh = torch.linalg.svd(data - data.mean(dim=0), full_matrices=False)
            rows = vh[:k]
        else:
            raise ValueError(kind)
        project = lambda values: F.linear(values, rows)
        parameters = []
    predictor = nn.Sequential(nn.Linear(2 * k, 32), nn.GELU(), nn.Linear(32, 1)).to(device)
    optimizer = torch.optim.Adam(parameters + list(predictor.parameters()), lr=learning_rate)
    for _ in range(steps):
        left, right, target = sample_batch(batch_size, true_rows)
        prediction = predictor(torch.cat([project(left), project(right)], dim=-1)).squeeze(-1)
        loss = F.mse_loss(prediction, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    left, right, target = sample_batch(4096, true_rows)
    with torch.no_grad():
        prediction = predictor(torch.cat([project(left), project(right)], dim=-1)).squeeze(-1)
        mse = F.mse_loss(prediction, target)
        learned_rows = projection.weight if kind == "direct" else rows
        learned_rows = torch.linalg.qr(learned_rows.T, mode="reduced").Q.T
    return {"test_mse": float(mse), "true_subspace_coverage":
            subspace_coverage(learned_rows, true_rows)}


class TestSyntheticTree(unittest.TestCase):
    def test_spectral_recovers_known_subspace(self):
        torch.manual_seed(7)
        device = torch.device("cpu")
        true_rows = make_true_rows(device)
        model = SyntheticPRSS(variant="spectral").to(device)
        main_parameters = [p for n, p in model.named_parameters()
                           if "unrestricted" not in n]
        diagnostic_parameters = list(model.prss.unrestricted.parameters())
        optimizer = torch.optim.Adam(main_parameters, lr=2e-3, weight_decay=1e-5)
        diagnostic_optimizer = torch.optim.Adam(diagnostic_parameters, lr=2e-3)

        steps = 800
        batch_size = 256
        for step in range(steps):
            model.train()
            model.prss.set_spectral_updates_allowed(True)
            left, right, target = sample_batch(batch_size, true_rows)
            output = model.forward_components(left, right)
            repeated_target = target.repeat(2)
            task_loss = F.mse_loss(output["task"], target)
            response = F.mse_loss(output["structured"], repeated_target)
            spectral = 0.5 * (
                model.prss.spectral_loss("branch",
                                         output["branch_readers"][:batch_size]) +
                model.prss.spectral_loss("branch",
                                         output["branch_readers"][batch_size:]))
            total = task_loss + response + 0.05 * spectral
            optimizer.zero_grad()
            total.backward()
            optimizer.step()

            diagnostic = F.mse_loss(output["unrestricted"], repeated_target)
            diagnostic_optimizer.zero_grad()
            diagnostic.backward()
            diagnostic_optimizer.step()

            model.prss.update_statistics(step, {
                "branch": InterfaceData(candidates=output["branch_candidates"],
                                        reader_matrices=output["branch_readers"])})
            model.prss.maybe_update(step)

        model.eval()
        model.prss.set_spectral_updates_allowed(False)
        left, right, target = sample_batch(4096, true_rows)
        with torch.no_grad():
            output = model.forward_components(left, right)
            task_mse = float(F.mse_loss(output["task"], target))
            structured_mse = float(F.mse_loss(output["structured"], target.repeat(2)))
            unrestricted_mse = float(F.mse_loss(output["unrestricted"], target.repeat(2)))
        final_rows = model.prss.quotients["branch"].projection()
        angles = principal_angles(final_rows, true_rows)
        max_angle = float(angles.max())
        coverage = subspace_coverage(final_rows, true_rows)

        # Gate: the known subspace must be recovered within 0.20 rad.
        self.assertLess(max_angle, 0.20, f"max principal angle {max_angle:.4f}")
        self.assertGreater(coverage, 0.9)
        # Structured reader should not be far behind the unrestricted comparator.
        self.assertLess(structured_mse, unrestricted_mse + 0.5)
        # A real task signal must exist: task MSE on known target below chance level.
        self.assertLess(task_mse, 0.1)

    def test_task_baselines_are_reproducible(self):
        torch.manual_seed(11)
        true_rows = make_true_rows(torch.device("cpu"))
        baselines = {
            kind: train_task_baseline(kind, true_rows, steps=300, batch_size=256,
                                      learning_rate=2e-3)
            for kind in ("random", "pca", "direct")
        }
        # On this isotropic-Gaussian toy task PCA has no variance structure to find
        # and behaves like random; the learned (direct) projection must beat both.
        self.assertLess(baselines["direct"]["test_mse"], baselines["random"]["test_mse"])
        self.assertGreater(baselines["direct"]["true_subspace_coverage"], 0.5)


if __name__ == "__main__":
    unittest.main()
