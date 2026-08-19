import torch
from torch import nn

from prss.ablations import VARIANTS, configure_ablation
from prss.config import InterfaceSpec, PRSSConfig
from prss.reader import LinearConditionalMatrixReader
from prss.spectral import principal_angles
from prss.system import PRSSSystem


def make_system(d=8, k=3):
  config = PRSSConfig(
    interfaces={"tau": InterfaceSpec("tau", d, d, k)},
    context_dim=5,
    gram_ema_rho=1.0,
    spectral_update_interval=1,
    spectral_warmup_steps=0,
  )
  return PRSSSystem(config)


def test_every_named_ablation_builds_an_executable_policy():
  for variant in VARIANTS:
    system = make_system()
    policy = configure_ablation(system, variant)
    assert policy.name == variant
    state = system.make_state("tau", torch.randn(4, 8))
    assert state.quotient.shape == (4, 3)


def test_direct_ablation_really_optimizes_projection():
  system = make_system()
  configure_ablation(system, "direct")
  projection = system.quotients.state_for("tau").R
  assert isinstance(projection, nn.Parameter)
  before = projection.detach().clone()
  optimizer = torch.optim.SGD(system.parameters(), lr=0.1)
  loss = system.make_state("tau", torch.randn(16, 8)).quotient.square().mean()
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  assert not torch.allclose(before, projection.detach())


def test_pca_ablation_updates_from_candidates_not_reader_matrices():
  torch.manual_seed(17)
  system = make_system(d=9, k=3)
  policy = configure_ablation(system, "pca")
  true_rows = torch.linalg.qr(torch.randn(9, 3), mode="reduced").Q.T
  dominant = 5.0 * torch.randn(2000, 3) @ true_rows
  noise = 0.01 * torch.randn(2000, 9)
  policy.update_statistics(system, {}, {"tau": dominant + noise})
  assert policy.maybe_update(system, 0)
  learned = system.quotients.state_for("tau").R
  assert float(principal_angles(learned, true_rows).max()) < 0.01


def test_fixed_random_never_changes_statistics_or_projection():
  system = make_system()
  policy = configure_ablation(system, "fixed_random")
  state = system.quotients.state_for("tau")
  old_r, old_g = state.R.clone(), state.G_ema.clone()
  policy.update_statistics(system, {}, {"tau": torch.randn(20, 8)})
  assert not policy.maybe_update(system, 0)
  assert torch.equal(old_r, state.R)
  assert torch.equal(old_g, state.G_ema)


def test_linear_reader_ablation_replaces_conditional_mlp():
  system = make_system()
  configure_ablation(system, "linear_reader_svd")
  assert isinstance(system.readers[system._key("tau")], LinearConditionalMatrixReader)


def test_fixed_random_really_replaces_identity_like_projection():
  config = PRSSConfig(
    interfaces={"tau": InterfaceSpec("tau", 6, 6, 3)},
    context_dim=5, initialization="identity_like")
  system = PRSSSystem(config)
  before = system.quotients.state_for("tau").R.clone()
  configure_ablation(system, "fixed_random")
  after = system.quotients.state_for("tau").R
  assert not torch.allclose(before, after)
  assert torch.allclose(after @ after.T, torch.eye(3), atol=1e-5)


def test_pca_ablation_is_invariant_to_large_mean_shift():
  torch.manual_seed(123)
  true_rows = torch.linalg.qr(torch.randn(8, 2), mode="reduced").Q.T
  centered = 4.0 * torch.randn(3000, 2) @ true_rows + 0.01 * torch.randn(3000, 8)
  shift = torch.arange(8, dtype=centered.dtype) * 100.0
  learned = []
  for values in (centered, centered + shift):
    system = make_system(d=8, k=2)
    policy = configure_ablation(system, "pca")
    policy.update_statistics(system, {}, {"tau": values})
    assert policy.maybe_update(system, 0)
    learned.append(system.quotients.state_for("tau").R.clone())
  assert float(principal_angles(learned[0], learned[1]).max()) < 1e-3
