import pytest
import torch

from prss.config import InterfaceSpec, PRSSConfig
from prss.losses import response_loss, spectral_tail_loss
from prss.system import PRSSSystem


def make_system():
  config = PRSSConfig(
    interfaces={
      "small": InterfaceSpec("small", raw_dim=6, candidate_dim=9, host_dim=3),
      "wide": InterfaceSpec("wide", raw_dim=5, candidate_dim=8, host_dim=5),
    },
    context_dim=7,
    root_metadata_dim=4,
    parent_local_dim=3,
    reader_hidden_dim=12,
    spectral_update_interval=2,
    spectral_warmup_steps=0,
  )
  return PRSSSystem(config)


def test_flexible_interface_shapes_are_never_hardcoded():
  system = make_system()
  small = system.make_state("small", torch.randn(4, 6))
  wide = system.make_state("wide", torch.randn(4, 5))
  assert small.candidate.shape == (4, 9)
  assert small.quotient.shape == (4, 3)
  assert wide.candidate.shape == (4, 8)
  assert wide.quotient.shape == (4, 5)
  assert system.quotients.state_for("small").R.shape == (3, 9)
  assert system.quotients.state_for("wide").R.shape == (5, 8)


def test_response_and_spectral_losses_reach_lift_reader_not_R():
  system = make_system()
  raw = torch.randn(16, 6)
  context = torch.randn(16, 7)
  target = torch.randint(0, 2, (16,), dtype=torch.float32)
  state = system.make_state("small", raw)
  logits, matrix, _ = system.structured_read("small", context, state.candidate)
  loss = response_loss(logits, target) + 0.2 * system.spectral_loss("small", matrix)
  loss.backward()
  lift_grads = [parameter.grad for parameter in system.lifts.parameters()
                if parameter.requires_grad]
  reader_grads = [parameter.grad for parameter in system.readers.parameters()
                  if parameter.requires_grad]
  assert any(gradient is not None and gradient.abs().sum() > 0 for gradient in lift_grads)
  assert any(gradient is not None and gradient.abs().sum() > 0 for gradient in reader_grads)
  assert system.quotients.state_for("small").R.grad is None


def test_spectral_tail_is_zero_on_retained_rowspace_and_one_on_orthogonal_space():
  rows = torch.tensor([[1.0, 0.0, 0.0]])
  retained = torch.tensor([[[2.0, 0.0, 0.0]]])
  removed = torch.tensor([[[0.0, 3.0, 0.0]]])
  assert float(spectral_tail_loss(retained, rows)) < 1e-7
  assert float(spectral_tail_loss(removed, rows)) > 0.999


def test_eval_mode_cannot_update_gram_or_svd():
  system = make_system().eval()
  readers = {"small": torch.randn(2, 1, 9)}
  with pytest.raises(RuntimeError):
    system.update_spectral_statistics(readers)
  before = system.quotients.state_for("small").R.clone()
  assert system.maybe_spectral_update(1000) is False
  assert torch.equal(before, system.quotients.state_for("small").R)



def test_state_spectral_loss_is_exact_operator_tail_and_detaches_only_R():
  system = make_system()
  candidate = torch.randn(8, 9, requires_grad=True)
  reader_matrix = torch.randn(8, 1, 9, requires_grad=True)
  loss = system.state_spectral_loss("small", reader_matrix, candidate)
  loss.backward()
  # PRSS spec: L_spec is ||B(I-P_R)||/||B||. It organizes the future reader; Phi is
  # trained by L_resp/main-task, not by inventing a different state-tail objective.
  assert reader_matrix.grad is not None and float(reader_matrix.grad.abs().sum()) > 0
  assert candidate.grad is None
  assert system.quotients.state_for("small").R.grad is None


def test_spectral_warmup_updates_after_exact_requested_number_of_batches():
  system = make_system().train()
  system.config.spectral_warmup_steps = 3
  system.config.spectral_update_interval = 2
  readers = {"small": torch.randn(4, 1, 9)}
  # accumulate evidence every batch, as the real trainer does
  for step in range(2):
    system.update_spectral_statistics(readers)
    assert system.maybe_spectral_update(step) is False
  system.update_spectral_statistics(readers)
  assert system.maybe_spectral_update(2) is True  # exactly after batch 3
