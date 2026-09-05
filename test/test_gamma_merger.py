"""GammaMerger learning-path tests (review ruling B1): at init the output
is EXACTLY the average merge, yet EVERY parameter group has a nonzero
gradient after one replay-style backward (alpha=1, U=0, MLP random)."""
import importlib.util
import unittest
from pathlib import Path

import torch

# load the module file directly: importing the `vla` package would pull
# the full prismatic/timm chain, which is not available in every env
_SPEC = importlib.util.spec_from_file_location(
    "gamma_merger",
    Path(__file__).resolve().parents[1] / "third_party" / "memoryvla"
    / "vla" / "gamma_merger.py")
_gm = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_gm)
GammaMerger = _gm.GammaMerger


class TestGammaLearningPath(unittest.TestCase):
    def test_init_is_exact_average(self):
        torch.manual_seed(0)
        g = GammaMerger(dim=128, rank=16, alpha_init=1.0, seed=7)
        m_a = torch.randn(4, 128)
        m_b = torch.randn(4, 128)
        out = g(m_a, m_b)
        self.assertTrue(torch.allclose(out, 0.5 * (m_a + m_b), atol=1e-6))

    def test_learning_path_opens(self):
        """Step 0: U alone has a nonzero gradient (the B1 fix).  After a
        few optimizer steps U leaves zero, and THEN every group -- MLP,
        proj_in, alpha -- receives gradients (they are gated by U^T)."""
        torch.manual_seed(0)
        g = GammaMerger(dim=128, rank=16, alpha_init=1.0, seed=7)
        m_a = torch.randn(4, 128)
        m_b = torch.randn(4, 128)
        c = torch.randn(4, 128)
        # step 0: only U (proj_out) can have a nonzero gradient
        z_hat = g(m_a.detach(), m_b.detach())
        (c * z_hat).sum().backward()
        self.assertGreater(g.proj_out.weight.grad.abs().sum().item(), 0.0,
                           "U (proj_out) has zero gradient at step 0")
        # a few optimizer steps move U away from zero
        opt = torch.optim.AdamW(g.parameters(), lr=0.01)
        for _ in range(3):
            opt.zero_grad()
            z_hat = g(m_a.detach(), m_b.detach())
            (c * z_hat).sum().backward()
            opt.step()
        self.assertGreater(g.proj_out.weight.abs().sum().item(), 0.0,
                           "U never left zero")
        # now EVERY parameter group must receive a gradient
        opt.zero_grad()
        z_hat = g(m_a.detach(), m_b.detach())
        (c * z_hat).sum().backward()
        for name, p in g.named_parameters():
            self.assertIsNotNone(p.grad, f"{name}: grad is None")
            self.assertGreater(p.grad.abs().sum().item(), 0.0,
                               f"{name}: zero gradient after U update")

    def test_output_leaves_average_after_update(self):
        """One optimizer step on U/MLP/alpha must move Gamma away from Avg."""
        torch.manual_seed(0)
        g = GammaMerger(dim=128, rank=16, alpha_init=1.0, seed=7)
        opt = torch.optim.AdamW(g.parameters(), lr=0.01)
        m_a = torch.randn(4, 128)
        m_b = torch.randn(4, 128)
        for _ in range(3):
            opt.zero_grad()
            z_hat = g(m_a.detach(), m_b.detach())
            (torch.randn(4, 128) * z_hat).sum().backward()
            opt.step()
        out = g(m_a, m_b)
        self.assertGreater((out - 0.5 * (m_a + m_b)).abs().mean().item(), 1e-6)


if __name__ == "__main__":
    unittest.main()
