"""EMA predictive statistics and stable analytic rank-k quotient updates.

The spectral state is deliberately a non-gradient block variable.  Two stability rules are important:

1. If the empirical predictive Gram has rank < host width, its nullspace is non-identifiable.  We
   keep the old quotient directions in that unconstrained subspace instead of accepting arbitrary
   eigenvectors from ``eigh``.
2. Spectral changes are damped and accepted only when they do not decrease captured predictive
   energy on the current Gram.  This avoids a large discontinuous basis jump inside a recursive host.
"""

import torch
from torch import nn
from torch.nn import functional as F


def random_semi_orthogonal(host_dim, candidate_dim, device=None, dtype=None):
  if candidate_dim < host_dim:
    raise ValueError("candidate_dim must be >= host_dim")
  matrix = torch.randn(candidate_dim, host_dim, device=device, dtype=dtype)
  q, _ = torch.linalg.qr(matrix, mode="reduced")
  return q.transpose(0, 1).contiguous()


def identity_like_projection(host_dim, candidate_dim, device=None, dtype=None):
  result = torch.zeros(host_dim, candidate_dim, device=device, dtype=dtype)
  result[:, :host_dim] = torch.eye(host_dim, device=device, dtype=dtype)
  return result


def orthogonal_procrustes_align(new_rows, old_rows):
  """Left-align an orthonormal row basis to the previous coordinate system."""
  if new_rows.shape != old_rows.shape:
    raise ValueError("Procrustes bases must have the same shape")
  cross = old_rows @ new_rows.transpose(-1, -2)
  u, _, vh = torch.linalg.svd(cross, full_matrices=False)
  rotation = u @ vh
  return rotation @ new_rows


def principal_angles(rows_a, rows_b):
  if rows_a.shape != rows_b.shape:
    raise ValueError("Principal-angle bases must have the same shape")
  singular_values = torch.linalg.svdvals(rows_a @ rows_b.transpose(-1, -2))
  return torch.arccos(singular_values.clamp(-1.0, 1.0))


def projector_distance(rows_a, rows_b):
  projector_a = rows_a.transpose(-1, -2) @ rows_a
  projector_b = rows_b.transpose(-1, -2) @ rows_b
  return torch.linalg.matrix_norm(projector_a - projector_b, ord="fro")


def reader_gram(reader_matrix):
  """Compute (1/N) sum_v B_v^T B_v; response rows are summed within each occurrence."""
  if reader_matrix.ndim < 3:
    raise ValueError("B must have shape [..., response_dim, candidate_dim]")
  candidate_dim = reader_matrix.shape[-1]
  flattened = reader_matrix.reshape(-1, reader_matrix.shape[-2], candidate_dim)
  if len(flattened) == 0:
    raise ValueError("Cannot compute a Gram from an empty reader bank")
  return torch.einsum("npd,npe->de", flattened, flattened) / float(len(flattened))


def centered_candidate_gram(candidates):
  """Centered candidate covariance used by the PCA ablation.

  The old implementation used E[h h^T], which is a second moment rather than PCA covariance and can
  spend principal directions on the mean vector.  This helper intentionally centers each batch.
  """
  if candidates.ndim < 2:
    raise ValueError("Candidates must have shape [..., candidate_dim]")
  flat = candidates.reshape(-1, candidates.shape[-1])
  if len(flat) == 0:
    raise ValueError("Cannot compute covariance from an empty candidate bank")
  centered = flat - flat.mean(dim=0, keepdim=True)
  denom = float(max(len(flat) - 1, 1))
  return centered.transpose(0, 1) @ centered / denom


def _orthonormal_rows(values, target_rows=None, eps=1e-10):
  """Return an orthonormal basis for the row span of ``values``."""
  if values.numel() == 0:
    return values.new_zeros((0, values.shape[-1]))
  # QR on the transpose creates orthonormal columns = orthonormal rows after transposing back.
  q, r = torch.linalg.qr(values.transpose(0, 1), mode="reduced")
  diag = torch.abs(torch.diagonal(r, 0)) if r.numel() else values.new_zeros((0,))
  rank = int((diag > eps).sum().item())
  if target_rows is not None:
    rank = min(rank, int(target_rows))
  return q[:, :rank].transpose(0, 1).contiguous()


def _complete_predictive_rows(predictive_rows, old_rows, host_dim, eps=1e-10):
  """Fill an underdetermined predictive subspace with the previous quotient directions.

  When rank(G) < k, every k-dimensional subspace containing range(G) is spectrally optimal.  Taking
  arbitrary nullspace eigenvectors creates meaningless quotient rotations.  We use the old rowspace
  as the tie-breaker, and only fall back to canonical coordinates if old rows do not provide enough
  independent directions.
  """
  d = old_rows.shape[-1]
  pieces = []
  if predictive_rows.numel():
    predictive_rows = _orthonormal_rows(predictive_rows, target_rows=host_dim, eps=eps)
    pieces.append(predictive_rows)
  current = predictive_rows if predictive_rows.numel() else old_rows.new_zeros((0, d))
  need = host_dim - current.shape[0]
  if need > 0:
    if current.shape[0]:
      projector = current.transpose(0, 1) @ current
      residual_old = old_rows - old_rows @ projector
    else:
      residual_old = old_rows
    old_fill = _orthonormal_rows(residual_old, target_rows=need, eps=eps)
    if old_fill.numel():
      pieces.append(old_fill)
      current = torch.cat([current, old_fill], dim=0)
      need = host_dim - current.shape[0]
  if need > 0:
    eye = torch.eye(d, device=old_rows.device, dtype=old_rows.dtype)
    if current.shape[0]:
      projector = current.transpose(0, 1) @ current
      residual_eye = eye - eye @ projector
    else:
      residual_eye = eye
    canonical = _orthonormal_rows(residual_eye, target_rows=need, eps=eps)
    pieces.append(canonical)
  result = torch.cat(pieces, dim=0) if pieces else old_rows.clone()
  if result.shape[0] != host_dim:
    raise RuntimeError("Could not construct a full host-width quotient basis")
  # One final QR removes accumulated floating-point error.
  result = _orthonormal_rows(result, target_rows=host_dim, eps=eps)
  if result.shape[0] != host_dim:
    raise RuntimeError("Completed quotient basis lost rank")
  return result


def _captured_energy(rows, gram):
  return torch.trace(rows @ gram @ rows.transpose(0, 1))


def _damped_orthogonal_step(old_rows, target_rows, gram, step_size, tolerance=1e-12):
  """Take a stable row-space step and reject any update that loses current Gram energy."""
  target_rows = orthogonal_procrustes_align(target_rows, old_rows)
  # Do not report a fictitious accepted step when the evidence leaves the quotient unchanged.
  if torch.allclose(target_rows, old_rows, atol=1e-10, rtol=1e-8):
    return old_rows.clone(), 0.0, 0.0
  old_score = _captured_energy(old_rows, gram)
  if step_size >= 1.0:
    candidates = [1.0]
  else:
    candidates = []
    alpha = float(step_size)
    while alpha >= 1e-3:
      candidates.append(alpha)
      alpha *= 0.5
  for alpha in candidates:
    mixed = (1.0 - alpha) * old_rows + alpha * target_rows
    proposal = _orthonormal_rows(mixed, target_rows=old_rows.shape[0])
    if proposal.shape != old_rows.shape:
      continue
    score = _captured_energy(proposal, gram)
    scale = torch.abs(old_score).clamp_min(1.0)
    if score + tolerance * scale >= old_score:
      return proposal, float(alpha), float((score - old_score).detach().cpu())
  return old_rows.clone(), 0.0, 0.0


class PredictiveQuotientState(nn.Module):
  """Non-trainable spectral state for one host interface type tau."""
  def __init__(self, interface_name, host_dim, candidate_dim, ema_rho=0.05,
               ridge_eps=1e-5, initialization="random", align=True,
               step_size=1.0, eigen_floor_ratio=1e-4):
    super().__init__()
    if candidate_dim < host_dim:
      raise ValueError("candidate_dim must be >= host_dim")
    if not 0 < step_size <= 1:
      raise ValueError("step_size must be in (0,1]")
    if not 0 <= eigen_floor_ratio < 1:
      raise ValueError("eigen_floor_ratio must be in [0,1)")
    self.interface_name = interface_name
    self.host_dim = int(host_dim)
    self.candidate_dim = int(candidate_dim)
    self.ema_rho = float(ema_rho)
    self.ridge_eps = float(ridge_eps)
    self.align = bool(align)
    self.step_size = float(step_size)
    self.eigen_floor_ratio = float(eigen_floor_ratio)
    if initialization == "random":
      projection = random_semi_orthogonal(host_dim, candidate_dim)
    elif initialization == "identity_like":
      projection = identity_like_projection(host_dim, candidate_dim)
    else:
      raise ValueError("Unknown initialization: {}".format(initialization))
    self.register_buffer("R", projection)
    self.register_buffer("G_ema", ridge_eps * torch.eye(candidate_dim))
    self.register_buffer("eigenvalues", torch.zeros(candidate_dim))
    self.register_buffer("gram_updates", torch.zeros((), dtype=torch.long))
    self.register_buffer("gram_updates_since_spectral", torch.zeros((), dtype=torch.long))
    self.register_buffer("spectral_updates", torch.zeros((), dtype=torch.long))
    self.register_buffer("last_projector_distance", torch.zeros(()))
    self.register_buffer("last_principal_angles", torch.zeros(host_dim))
    self.register_buffer("last_effective_predictive_rank", torch.zeros((), dtype=torch.long))
    self.register_buffer("last_accepted_step", torch.zeros(()))
    self.register_buffer("last_objective_gain", torch.zeros(()))

  @property
  def dimensional_compression(self):
    return self.candidate_dim > self.host_dim

  def forward(self, candidate):
    if candidate.shape[-1] != self.candidate_dim:
      raise ValueError("{} candidate width {}, expected {}".format(
        self.interface_name, candidate.shape[-1], self.candidate_dim))
    quotient = F.linear(candidate, self.R)
    if quotient.shape[-1] != self.host_dim:
      raise RuntimeError("Quotient violated host interface width")
    return quotient

  def projector(self, detach=True):
    rows = self.R.detach() if detach else self.R
    return rows.transpose(-1, -2) @ rows

  @torch.no_grad()
  def update_gram(self, batch_gram):
    if batch_gram.shape != self.G_ema.shape:
      raise ValueError("Gram shape mismatch for {}".format(self.interface_name))
    if not torch.isfinite(batch_gram).all():
      raise FloatingPointError("Non-finite predictive Gram for {}".format(self.interface_name))
    batch_gram = 0.5 * (batch_gram + batch_gram.transpose(-1, -2))
    if int(self.gram_updates_since_spectral.item()) == 0:
      self.G_ema.copy_(batch_gram)
    else:
      self.G_ema.mul_(1.0 - self.ema_rho).add_(batch_gram, alpha=self.ema_rho)
    self.gram_updates.add_(1)
    self.gram_updates_since_spectral.add_(1)

  @torch.no_grad()
  def update_from_readers(self, reader_matrix):
    self.update_gram(reader_gram(reader_matrix.detach()))

  @torch.no_grad()
  def spectral_update(self):
    symmetric = 0.5 * (self.G_ema + self.G_ema.transpose(-1, -2))
    if not torch.isfinite(symmetric).all():
      raise FloatingPointError("Non-finite EMA Gram before spectral update for {}".format(
        self.interface_name))
    work = symmetric.double()
    regularized = work + self.ridge_eps * torch.eye(
      self.candidate_dim, device=work.device, dtype=work.dtype)
    eigenvalues_reg, eigenvectors = torch.linalg.eigh(regularized)
    eigenvalues = (eigenvalues_reg - self.ridge_eps).clamp_min(0)
    max_eigenvalue = eigenvalues[-1] if len(eigenvalues) else work.new_tensor(0.0)
    threshold = torch.maximum(
      work.new_tensor(self.ridge_eps * 10.0),
      max_eigenvalue * self.eigen_floor_ratio)
    effective_rank = int((eigenvalues > threshold).sum().item())
    retained_predictive_rank = min(effective_rank, self.host_dim)

    old_rows = self.R.detach().double().clone()
    if retained_predictive_rank > 0:
      predictive_rows = eigenvectors[:, -retained_predictive_rank:].transpose(0, 1).contiguous()
      target_rows = _complete_predictive_rows(
        predictive_rows, old_rows, self.host_dim, eps=max(self.ridge_eps * 0.1, 1e-12))
    else:
      # No evidence-supported direction: the quotient is intentionally unchanged.
      target_rows = old_rows.clone()

    if self.align:
      target_rows = orthogonal_procrustes_align(target_rows, old_rows)
    proposal, accepted_step, objective_gain = _damped_orthogonal_step(
      old_rows, target_rows, work, self.step_size)
    distance = projector_distance(proposal, old_rows)
    angles = principal_angles(proposal, old_rows)
    self.last_projector_distance.copy_(distance.to(self.last_projector_distance.dtype))
    self.last_principal_angles.copy_(angles.to(self.last_principal_angles.dtype))
    self.last_effective_predictive_rank.fill_(retained_predictive_rank)
    self.last_accepted_step.fill_(accepted_step)
    self.last_objective_gain.fill_(objective_gain)
    self.R.copy_(proposal.to(self.R.dtype))
    self.eigenvalues.copy_(eigenvalues.to(self.eigenvalues.dtype))
    self.spectral_updates.add_(1)
    self.G_ema.copy_(self.ridge_eps * torch.eye(
      self.candidate_dim, device=self.G_ema.device, dtype=self.G_ema.dtype))
    self.gram_updates_since_spectral.zero_()
    return self.R

  def spectral_diagnostics(self):
    eigenvalues = self.eigenvalues.detach().double().clamp_min(0)
    descending = torch.flip(eigenvalues, dims=[0])
    total = descending.sum().clamp_min(torch.finfo(descending.dtype).eps)
    ranks = sorted(set([
      max(1, self.host_dim // 4),
      max(1, self.host_dim // 2),
      self.host_dim,
      min(2 * self.host_dim, self.candidate_dim),
    ]))
    energy = {rank: float(descending[:rank].sum() / total) for rank in ranks}
    return {
      "interface": self.interface_name,
      "host_dim": self.host_dim,
      "candidate_dim": self.candidate_dim,
      "dimensional_compression": self.dimensional_compression,
      "compression_ratio": (float(self.candidate_dim) / self.host_dim
                            if self.dimensional_compression else 1.0),
      "eigenvalues_descending": descending.cpu().tolist(),
      "energy": {"energy@{}".format(rank): value for rank, value in energy.items()},
      "tail": {"tail@{}".format(rank): 1.0 - value for rank, value in energy.items()},
      "projector_distance": float(self.last_projector_distance),
      "principal_angles_radians": self.last_principal_angles.detach().cpu().tolist(),
      "effective_predictive_rank": int(self.last_effective_predictive_rank),
      "accepted_spectral_step": float(self.last_accepted_step),
      "captured_energy_gain": float(self.last_objective_gain),
      "reader_gram_updates": int(self.gram_updates),
      "gram_updates_since_spectral": int(self.gram_updates_since_spectral),
      "spectral_updates": int(self.spectral_updates),
    }


class PredictiveQuotientBank(nn.Module):
  def __init__(self, interface_specs, ema_rho=0.05, ridge_eps=1e-5,
               initialization="random", align=True, step_size=1.0,
               eigen_floor_ratio=1e-4):
    super().__init__()
    self._tau_to_key = {}
    modules = {}
    for index, (tau, spec) in enumerate(interface_specs.items()):
      key = "interface_{:04d}".format(index)
      self._tau_to_key[tau] = key
      modules[key] = PredictiveQuotientState(
        tau, spec.host_dim, spec.candidate_dim, ema_rho=ema_rho,
        ridge_eps=ridge_eps, initialization=initialization, align=align,
        step_size=step_size, eigen_floor_ratio=eigen_floor_ratio)
    self.states = nn.ModuleDict(modules)

  def state_for(self, tau):
    try:
      return self.states[self._tau_to_key[tau]]
    except KeyError as error:
      raise KeyError("Unknown quotient interface: {}".format(tau)) from error

  def forward(self, tau, candidate):
    return self.state_for(tau)(candidate)

  @torch.no_grad()
  def update_grams(self, readers_by_tau):
    for tau, readers in readers_by_tau.items():
      if readers is not None and readers.numel() > 0:
        self.state_for(tau).update_from_readers(readers)

  @torch.no_grad()
  def update_candidate_covariances(self, candidates_by_tau):
    for tau, candidates in candidates_by_tau.items():
      if candidates is not None and candidates.numel() > 0:
        self.state_for(tau).update_gram(centered_candidate_gram(candidates.detach()))

  @torch.no_grad()
  def spectral_update_all(self):
    """Update only evidence-supported *compressive* interfaces.

    Interfaces with d_tau == k_tau have no dimensional quotient to learn: every full-rank
    orthogonal row-space is the whole candidate space, so rotating R only changes host coordinates
    and can destabilize the recursive model without reducing any predictive tail.  Likewise, an
    interface that received no fresh Gram observations since its previous update must not be marked
    as spectrally updated.
    """
    updated = {}
    for tau in self._tau_to_key:
      state = self.state_for(tau)
      if not state.dimensional_compression:
        continue
      if int(state.gram_updates_since_spectral.item()) <= 0:
        continue
      updated[tau] = state.spectral_update()
    return updated

  def diagnostics(self):
    return {tau: self.state_for(tau).spectral_diagnostics() for tau in self._tau_to_key}
