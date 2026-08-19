"""Combined neural and block-coordinate spectral PRSS system."""

from collections import defaultdict

import torch
from torch import nn

from prss.lift import CandidateLift, IdentityCandidateLift
from prss.losses import spectral_tail_loss
from prss.outside_context import OutsideContextEncoder
from prss.reader import ConditionalMatrixReader, UnrestrictedReader
from prss.spectral import PredictiveQuotientBank
from prss.state import QuotientState


class PRSSSystem(nn.Module):
  def __init__(self, config, no_lift_types=None):
    super().__init__()
    self.config = config
    no_lift_types = set(no_lift_types or [])
    self._tau_to_key = {tau: "type_{:04d}".format(index)
                        for index, tau in enumerate(config.interfaces)}
    lifts = {}
    readers = {}
    unrestricted = {}
    for tau, spec in config.interfaces.items():
      key = self._tau_to_key[tau]
      if tau in no_lift_types:
        if spec.raw_dim != spec.candidate_dim:
          raise ValueError("No-lift ablation requires raw_dim == candidate_dim")
        lifts[key] = IdentityCandidateLift(spec.raw_dim)
      else:
        lifts[key] = CandidateLift(
          spec.raw_dim, spec.candidate_dim,
          hidden_dim=config.lift_hidden_dim or spec.candidate_dim,
          layers=config.lift_layers)
      readers[key] = ConditionalMatrixReader(
        config.context_dim, spec.candidate_dim, spec.response_dim,
        config.reader_hidden_dim)
      unrestricted[key] = UnrestrictedReader(
        config.context_dim, spec.candidate_dim, spec.response_dim,
        config.reader_hidden_dim)
    self.lifts = nn.ModuleDict(lifts)
    self.readers = nn.ModuleDict(readers)
    self.unrestricted_readers = nn.ModuleDict(unrestricted)
    self.outside = OutsideContextEncoder(
      config.interfaces, config.root_metadata_dim, config.parent_local_dim,
      context_dim=config.context_dim, relation_count=config.relation_count,
      relation_dim=config.relation_dim, layers=config.outside_layers,
      detach_siblings=config.detach_siblings)
    self.quotients = PredictiveQuotientBank(
      config.interfaces, ema_rho=config.gram_ema_rho, ridge_eps=config.ridge_eps,
      initialization=config.initialization, align=config.procrustes_align,
      step_size=config.spectral_step_size,
      eigen_floor_ratio=config.spectral_eigen_floor_ratio)
    self.spectral_updates_allowed = True

  def _key(self, tau):
    try:
      return self._tau_to_key[tau]
    except KeyError as error:
      raise KeyError("Unknown interface type: {}".format(tau)) from error

  def make_state(self, tau, raw):
    spec = self.config.interface(tau)
    if raw.shape[-1] != spec.raw_dim:
      raise ValueError("{} raw width {}, host contract says {}".format(
        tau, raw.shape[-1], spec.raw_dim))
    candidate = self.lifts[self._key(tau)](raw)
    quotient = self.quotients(tau, candidate)
    if candidate.shape[-1] != spec.candidate_dim:
      raise RuntimeError("Candidate lift violated d_tau")
    if quotient.shape[-1] != spec.host_dim:
      raise RuntimeError("PRSS projection violated k_tau")
    return QuotientState(tau=tau, raw=raw, candidate=candidate, quotient=quotient)

  def make_state_from_candidate(self, tau, raw, candidate):
    """Project an integration-provided rich candidate into the host interface budget.

    This path is used when the host adapter can construct a richer state *before* the
    host-width bottleneck (e.g. from TGN pre-aggregation tensors).  It prevents the
    invalid pattern ``k -> d -> k`` after information has already been compressed to k.
    """
    spec = self.config.interface(tau)
    if raw.shape[-1] != spec.raw_dim:
      raise ValueError("{} raw width {}, host contract says {}".format(
        tau, raw.shape[-1], spec.raw_dim))
    if candidate.shape[-1] != spec.candidate_dim:
      raise ValueError("{} candidate width {}, expected {}".format(
        tau, candidate.shape[-1], spec.candidate_dim))
    quotient = self.quotients(tau, candidate)
    if quotient.shape[-1] != spec.host_dim:
      raise RuntimeError("PRSS projection violated k_tau")
    return QuotientState(tau=tau, raw=raw, candidate=candidate, quotient=quotient)

  def structured_read(self, tau, context, candidate):
    reader = self.readers[self._key(tau)]
    matrix, bias = reader(context)
    logits = reader.logits(matrix, bias, candidate)
    return logits, matrix, bias

  def unrestricted_read(self, tau, context, candidate):
    return self.unrestricted_readers[self._key(tau)](candidate, context)

  def spectral_loss(self, tau, reader_matrix):
    """Operator tail diagnostic; do not use this to train the reader itself."""
    return spectral_tail_loss(reader_matrix, self.quotients.state_for(tau).R)

  def state_spectral_loss(self, tau, reader_matrix, candidate):
    """Exact operator-tail loss from the PRSS specification.

    The candidate argument is accepted for integration API compatibility, but the spectral
    objective is defined on the future-reading operator itself:
        ||B(C)(I-P_R)||_F^2 / (||B(C)||_F^2 + eps).
    R is a detached spectral block variable; gradients reach B(C)/the outside reader, while
    Phi is trained by the proper response loss and the main task loss.
    """
    del candidate
    return spectral_tail_loss(reader_matrix, self.quotients.state_for(tau).R)

  def reader_norms(self, readers_by_tau):
    return {tau: float(values.detach().square().sum(dim=(-2, -1)).sqrt().mean())
            for tau, values in readers_by_tau.items() if values.numel()}

  def set_spectral_updates_allowed(self, allowed):
    self.spectral_updates_allowed = bool(allowed)

  @torch.no_grad()
  def update_spectral_statistics(self, readers_by_tau):
    if not self.training:
      raise RuntimeError("Validation/test cannot update PRSS Gram statistics")
    if not self.spectral_updates_allowed:
      raise RuntimeError("Spectral statistics are frozen")
    self.quotients.update_grams(readers_by_tau)

  @torch.no_grad()
  def maybe_spectral_update(self, step):
    if not self.training or not self.spectral_updates_allowed:
      return False
    completed = int(step) + 1
    warmup = int(self.config.spectral_warmup_steps)
    interval = int(self.config.spectral_update_interval)
    if completed < warmup:
      return False
    # If warmup=100, the first analytic quotient update occurs after exactly 100 batches,
    # then every `interval` batches.  The previous condition accidentally delayed it to 200.
    if (completed - warmup) % interval != 0:
      return False
    updated = self.quotients.spectral_update_all()
    return bool(updated)

  def spectral_diagnostics(self):
    return self.quotients.diagnostics()
