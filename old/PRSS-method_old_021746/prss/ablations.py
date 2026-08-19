"""Explicit, named ablations required by the PRSS method specification."""

from dataclasses import dataclass

import torch
from torch import nn

from prss.lift import LinearCandidateLift
from prss.reader import LinearConditionalMatrixReader
from prss.spectral import random_semi_orthogonal


VARIANTS = (
  "response_only",
  "fixed_random",
  "pca",
  "direct",
  "linear_reader_svd",
  "no_nonlinear_lift",
  "neural_svd_no_spec",
  "full",
)


@dataclass
class AblationPolicy:
  name: str
  statistic: str
  use_response_loss: bool
  use_spectral_loss: bool
  update_projection: bool

  @torch.no_grad()
  def update_statistics(self, system, readers_by_tau, candidates_by_tau):
    if self.statistic == "reader":
      system.update_spectral_statistics(readers_by_tau)
    elif self.statistic == "pca":
      # PCA must use a centered covariance rather than the uncentered second moment E[h h^T].
      system.quotients.update_candidate_covariances(candidates_by_tau)
    elif self.statistic == "none":
      return
    else:
      raise ValueError(self.statistic)

  @torch.no_grad()
  def maybe_update(self, system, step):
    if not self.update_projection:
      return False
    return system.maybe_spectral_update(step)


def _make_projection_trainable(system):
  for state in system.quotients.states.values():
    # d_tau == k_tau is compatibility mode, not a quotient-selection problem.  Training an
    # arbitrary square map there adds unrelated model capacity and makes the direct baseline unfair.
    if not state.dimensional_compression:
      continue
    value = state.R.detach().clone()
    del state._buffers["R"]
    state.register_parameter("R", nn.Parameter(value))


def _replace_with_fixed_random_projection(system):
  """Make the fixed-random ablation actually random.

  TGN installs PRSS with identity-compatible R=[I,0] for a fair main-method initialization.  Leaving
  that projection untouched made the historical ``fixed_random`` ablation a mislabeled fixed-identity
  model.  Randomize it explicitly and then freeze it.
  """
  for state in system.quotients.states.values():
    if not state.dimensional_compression:
      continue
    state.R.copy_(random_semi_orthogonal(
      state.host_dim, state.candidate_dim, device=state.R.device, dtype=state.R.dtype))


def configure_ablation(system, variant, integration_adapter=None):
  if variant not in VARIANTS:
    raise ValueError("Unknown PRSS variant: {}".format(variant))
  device = next(system.parameters()).device
  if variant == "no_nonlinear_lift":
    if integration_adapter is not None and hasattr(integration_adapter, "use_linear_candidate_lift"):
      integration_adapter.use_linear_candidate_lift()
    else:
      for tau, spec in system.config.interfaces.items():
        system.lifts[system._key(tau)] = LinearCandidateLift(
          spec.raw_dim, spec.candidate_dim).to(device)
  if variant == "linear_reader_svd":
    for tau, spec in system.config.interfaces.items():
      system.readers[system._key(tau)] = LinearConditionalMatrixReader(
        system.config.context_dim, spec.candidate_dim, spec.response_dim).to(device)
  if variant == "direct":
    _make_projection_trainable(system)
  if variant == "fixed_random":
    _replace_with_fixed_random_projection(system)
  if variant == "pca":
    # PCA is a static covariance baseline, not the damped alternating PRSS update.  Use the exact
    # principal subspace so the ablation means what its name says.
    for state in system.quotients.states.values():
      state.step_size = 1.0
      state.eigen_floor_ratio = 0.0

  policies = {
    # Same auxiliary continuation supervision as full PRSS, but R remains [I,0].  This is the
    # crucial deep-supervision control for end-to-end node classification.
    "response_only": AblationPolicy(variant, "none", True, False, False),
    # Keep future-response supervision identical so the comparison isolates how R is chosen.
    "fixed_random": AblationPolicy(variant, "none", True, False, False),
    "pca": AblationPolicy(variant, "pca", True, False, True),
    "direct": AblationPolicy(variant, "none", True, False, False),
    "linear_reader_svd": AblationPolicy(variant, "reader", True, True, True),
    "no_nonlinear_lift": AblationPolicy(variant, "reader", True, True, True),
    "neural_svd_no_spec": AblationPolicy(variant, "reader", True, False, True),
    "full": AblationPolicy(variant, "reader", True, True, True),
  }
  return policies[variant]
