import torch

from prss.spectral import (PredictiveQuotientState, orthogonal_procrustes_align,
                          principal_angles, reader_gram)


def orthonormal_rows(rows, atol=1e-5):
  identity = torch.eye(rows.shape[0], device=rows.device, dtype=rows.dtype)
  return torch.allclose(rows @ rows.T, identity, atol=atol)


def test_reader_gram_shape_and_value():
  matrix = torch.randn(11, 3, 7)
  gram = reader_gram(matrix)
  expected = sum(item.T @ item for item in matrix) / len(matrix)
  assert gram.shape == (7, 7)
  assert torch.allclose(gram, expected)


def test_eigh_recovers_known_top_k_subspace():
  torch.manual_seed(4)
  d, k, p, n = 9, 3, 2, 200
  true_rows = torch.linalg.qr(torch.randn(d, k), mode="reduced").Q.T
  coefficients = torch.randn(n, p, k)
  readers = coefficients @ true_rows
  state = PredictiveQuotientState("known", k, d, ema_rho=1.0, ridge_eps=1e-7)
  state.update_from_readers(readers)
  state.spectral_update()
  angles = principal_angles(state.R, true_rows)
  assert state.R.shape == (k, d)
  assert orthonormal_rows(state.R)
  assert float(angles.max()) < 1e-3


def test_procrustes_removes_basis_rotation():
  torch.manual_seed(8)
  d, k = 10, 4
  old = torch.linalg.qr(torch.randn(d, k), mode="reduced").Q.T
  rotation = torch.linalg.qr(torch.randn(k, k)).Q
  new = rotation @ old
  aligned = orthogonal_procrustes_align(new, old)
  assert torch.allclose(aligned, old, atol=1e-5)


def test_projection_is_buffer_and_backpropagates_only_to_candidate():
  state = PredictiveQuotientState("tau", 3, 7)
  candidate = torch.randn(5, 7, requires_grad=True)
  state(candidate).square().sum().backward()
  assert candidate.grad is not None
  assert state.R.requires_grad is False
  assert all(parameter is not state.R for parameter in state.parameters())


def test_equal_dimensions_are_compatibility_not_compression():
  state = PredictiveQuotientState("square", 5, 5, initialization="identity_like")
  assert not state.dimensional_compression
  assert torch.allclose(state.R, torch.eye(5))



def test_zero_predictive_gram_keeps_projection_exactly_unchanged():
  state = PredictiveQuotientState(
    "zero", 3, 7, initialization="identity_like", ridge_eps=1e-7, step_size=0.25)
  before = state.R.clone()
  state.update_gram(torch.zeros(7, 7))
  state.spectral_update()
  assert torch.equal(before, state.R)
  assert int(state.last_effective_predictive_rank) == 0
  assert float(state.last_accepted_step) == 0.0


def test_rank_deficient_gram_preserves_old_nullspace_as_tiebreaker():
  d, k = 6, 3
  state = PredictiveQuotientState(
    "rank1", k, d, initialization="identity_like", ridge_eps=1e-8,
    step_size=1.0, eigen_floor_ratio=1e-6)
  old = state.R.clone().double()
  direction = torch.zeros(d, dtype=torch.double)
  direction[-1] = 1.0
  gram = torch.outer(direction, direction).float()
  state.update_gram(gram)
  state.spectral_update()
  rows = state.R.double()
  # The one evidence-supported direction must be retained.
  assert float((rows @ direction).square().sum()) > 0.999
  # The two unconstrained dimensions should come from the previous quotient rowspace, not arbitrary
  # numerical eigenvectors in the Gram nullspace.
  old_projector = old.T @ old
  overlap = torch.trace(rows @ old_projector @ rows.T)
  assert float(overlap) > 1.999


def test_damped_spectral_step_never_decreases_current_predictive_energy():
  torch.manual_seed(99)
  state = PredictiveQuotientState(
    "damped", 3, 8, initialization="identity_like", ridge_eps=1e-8,
    step_size=0.25, eigen_floor_ratio=1e-6)
  readers = torch.randn(50, 1, 8)
  gram = reader_gram(readers)
  before = state.R.clone().double()
  before_energy = torch.trace(before @ gram.double() @ before.T)
  state.update_gram(gram)
  state.spectral_update()
  after = state.R.double()
  after_energy = torch.trace(after @ gram.double() @ after.T)
  assert float(after_energy + 1e-9) >= float(before_energy)
  assert orthonormal_rows(state.R)


def test_streaming_future_operator_gram_matches_explicit_stacked_svd_subspace():
  """The implementation's G=mean(B^T B) is exactly the right-singular subspace of stacked B."""
  torch.manual_seed(2026)
  n, p, d, k = 37, 2, 11, 4
  bank = torch.randn(n, p, d)
  gram = reader_gram(bank)
  _, eigvecs = torch.linalg.eigh(gram.double())
  gram_rows = eigvecs[:, -k:].T
  stacked = bank.reshape(n * p, d).double()
  _, _, vh = torch.linalg.svd(stacked, full_matrices=False)
  svd_rows = vh[:k]
  angles = principal_angles(gram_rows, svd_rows)
  assert float(angles.max()) < 1e-6
