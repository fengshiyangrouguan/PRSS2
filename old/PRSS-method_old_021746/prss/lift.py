"""Small nonlinear candidate lifts Phi_omega."""

import torch
from torch import nn


class CandidateLift(nn.Module):
  def __init__(self, raw_dim, candidate_dim, hidden_dim=None, layers=2,
               activation="gelu", layer_norm=True):
    super().__init__()
    if layers < 1:
      raise ValueError("Candidate lift must have at least one layer")
    hidden_dim = hidden_dim or candidate_dim
    activation_factory = {"gelu": nn.GELU, "relu": nn.ReLU}.get(activation)
    if activation_factory is None:
      raise ValueError("Unsupported activation: {}".format(activation))
    modules = []
    in_dim = raw_dim
    for _ in range(layers - 1):
      modules.extend([nn.Linear(in_dim, hidden_dim), activation_factory()])
      in_dim = hidden_dim
    modules.append(nn.Linear(in_dim, candidate_dim))
    if layer_norm:
      modules.append(nn.LayerNorm(candidate_dim))
    self.network = nn.Sequential(*modules)
    self.residual = nn.Identity() if raw_dim == candidate_dim else nn.Linear(raw_dim, candidate_dim,
                                                                             bias=False)
    self.residual_scale = nn.Parameter(torch.tensor(0.1))
    self.raw_dim = raw_dim
    self.candidate_dim = candidate_dim

  def forward(self, raw):
    if raw.shape[-1] != self.raw_dim:
      raise ValueError("Expected raw width {}, got {}".format(self.raw_dim, raw.shape[-1]))
    return self.network(raw) + self.residual_scale * self.residual(raw)


class IdentityCandidateLift(nn.Module):
  """Used by the no-lift ablation and known-subspace sanity tests."""
  def __init__(self, dimension):
    super().__init__()
    self.raw_dim = dimension
    self.candidate_dim = dimension

  def forward(self, raw):
    if raw.shape[-1] != self.raw_dim:
      raise ValueError("Identity lift width mismatch")
    return raw


class LinearCandidateLift(nn.Module):
  """No-nonlinear-lift ablation: a single affine expansion into d_tau."""
  def __init__(self, raw_dim, candidate_dim):
    super().__init__()
    self.raw_dim = raw_dim
    self.candidate_dim = candidate_dim
    self.linear = nn.Linear(raw_dim, candidate_dim)

  def forward(self, raw):
    if raw.shape[-1] != self.raw_dim:
      raise ValueError("Linear lift width mismatch")
    return self.linear(raw)
