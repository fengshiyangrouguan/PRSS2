"""Conditional future-reading matrices B_tau(C) and unrestricted diagnostic readers."""

import torch
from torch import nn


class ConditionalMatrixReader(nn.Module):
  def __init__(self, context_dim, candidate_dim, response_dim=1, hidden_dim=128):
    super().__init__()
    self.context_dim = context_dim
    self.candidate_dim = candidate_dim
    self.response_dim = response_dim
    self.trunk = nn.Sequential(
      nn.Linear(context_dim, hidden_dim),
      nn.GELU(),
      nn.LayerNorm(hidden_dim),
    )
    self.matrix_head = nn.Linear(hidden_dim, response_dim * candidate_dim)
    self.bias_head = nn.Linear(hidden_dim, response_dim)

  def forward(self, context):
    if context.shape[-1] != self.context_dim:
      raise ValueError("Context width mismatch")
    hidden = self.trunk(context)
    matrix = self.matrix_head(hidden).reshape(
      *context.shape[:-1], self.response_dim, self.candidate_dim)
    bias = self.bias_head(hidden)
    return matrix, bias

  @staticmethod
  def logits(matrix, bias, candidate):
    if matrix.shape[:-2] != candidate.shape[:-1]:
      raise ValueError("Reader/candidate batch shapes do not match")
    return bias + torch.einsum("...pd,...d->...p", matrix, candidate)


class UnrestrictedReader(nn.Module):
  """Diagnostic comparator MLP(h, C); never used to construct the quotient Gram."""
  def __init__(self, context_dim, candidate_dim, response_dim=1, hidden_dim=128):
    super().__init__()
    self.context_dim = context_dim
    self.candidate_dim = candidate_dim
    self.network = nn.Sequential(
      nn.Linear(context_dim + candidate_dim, hidden_dim),
      nn.GELU(),
      nn.LayerNorm(hidden_dim),
      nn.Linear(hidden_dim, hidden_dim),
      nn.GELU(),
      nn.Linear(hidden_dim, response_dim),
    )

  def forward(self, candidate, context):
    if candidate.shape[:-1] != context.shape[:-1]:
      raise ValueError("Unrestricted reader batch shapes do not match")
    return self.network(torch.cat([candidate, context], dim=-1))


class LinearConditionalMatrixReader(nn.Module):
  """Ridge/linear future-reader ablation with the same global B(C) semantics."""
  def __init__(self, context_dim, candidate_dim, response_dim=1):
    super().__init__()
    self.context_dim = context_dim
    self.candidate_dim = candidate_dim
    self.response_dim = response_dim
    self.output = nn.Linear(context_dim, response_dim * candidate_dim + response_dim)

  def forward(self, context):
    values = self.output(context)
    split = self.response_dim * self.candidate_dim
    matrix = values[..., :split].reshape(
      *context.shape[:-1], self.response_dim, self.candidate_dim)
    bias = values[..., split:]
    return matrix, bias

  @staticmethod
  def logits(matrix, bias, candidate):
    return bias + torch.einsum("...pd,...d->...p", matrix, candidate)
