"""Validated per-interface dimensions and training configuration."""

from dataclasses import dataclass
from typing import Mapping, Optional


@dataclass(frozen=True)
class InterfaceSpec:
  """One host recursive interface type.

  ``host_dim`` is always supplied by the host adapter. PRSS has no global quotient-width default.
  """

  name: str
  raw_dim: int
  candidate_dim: int
  host_dim: int
  response_dim: int = 1

  def __post_init__(self):
    if not self.name:
      raise ValueError("Interface name cannot be empty")
    for field_name in ("raw_dim", "candidate_dim", "host_dim", "response_dim"):
      if getattr(self, field_name) <= 0:
        raise ValueError("{} must be positive for {}".format(field_name, self.name))
    if self.candidate_dim < self.host_dim:
      raise ValueError(
        "candidate_dim ({}) must be >= host_dim ({}) for {}".format(
          self.candidate_dim, self.host_dim, self.name))

  @property
  def compression_ratio(self):
    return float(self.candidate_dim) / float(self.host_dim)

  @property
  def dimensional_compression(self):
    return self.candidate_dim > self.host_dim


@dataclass
class PRSSConfig:
  interfaces: Mapping[str, InterfaceSpec]
  context_dim: int = 64
  root_metadata_dim: int = 1
  parent_local_dim: int = 1
  relation_count: int = 4
  relation_dim: int = 16
  outside_layers: int = 2
  reader_hidden_dim: int = 128
  lift_layers: int = 2
  lift_hidden_dim: Optional[int] = None
  lambda_task: float = 1.0
  lambda_resp: float = 1.0
  lambda_spec: float = 0.1
  gram_ema_rho: float = 0.05
  spectral_update_interval: int = 200
  spectral_warmup_steps: int = 200
  ridge_eps: float = 1e-5
  procrustes_align: bool = True
  spectral_step_size: float = 0.25
  spectral_eigen_floor_ratio: float = 1e-4
  detach_siblings: bool = True
  initialization: str = "random"

  def __post_init__(self):
    self.interfaces = dict(self.interfaces)
    if not self.interfaces:
      raise ValueError("At least one interface specification is required")
    for name, spec in self.interfaces.items():
      if name != spec.name:
        raise ValueError("Interface mapping key must equal InterfaceSpec.name")
    for field_name in ("context_dim", "root_metadata_dim", "parent_local_dim",
                       "relation_count", "relation_dim", "outside_layers",
                       "reader_hidden_dim", "lift_layers", "spectral_update_interval"):
      if getattr(self, field_name) <= 0:
        raise ValueError("{} must be positive".format(field_name))
    if self.spectral_warmup_steps < 0:
      raise ValueError("spectral_warmup_steps cannot be negative")
    if not 0 < self.gram_ema_rho <= 1:
      raise ValueError("gram_ema_rho must be in (0, 1]")
    if self.ridge_eps <= 0:
      raise ValueError("ridge_eps must be positive")
    if not 0 < self.spectral_step_size <= 1:
      raise ValueError("spectral_step_size must be in (0, 1]")
    if not 0 <= self.spectral_eigen_floor_ratio < 1:
      raise ValueError("spectral_eigen_floor_ratio must be in [0, 1)")
    if self.initialization not in ("random", "identity_like"):
      raise ValueError("initialization must be random or identity_like")

  @classmethod
  def from_host_dimensions(cls, host_dimensions: Mapping[str, int],
                           raw_dimensions: Optional[Mapping[str, int]] = None,
                           candidate_dimensions: Optional[Mapping[str, int]] = None,
                           response_dimensions: Optional[Mapping[str, int]] = None,
                           **kwargs):
    """Build specs while making the host the sole source of every k_tau."""
    raw_dimensions = raw_dimensions or host_dimensions
    candidate_dimensions = candidate_dimensions or host_dimensions
    response_dimensions = response_dimensions or {}
    specs = {}
    for name, host_dim in host_dimensions.items():
      specs[name] = InterfaceSpec(
        name=name,
        raw_dim=int(raw_dimensions[name]),
        candidate_dim=int(candidate_dimensions[name]),
        host_dim=int(host_dim),
        response_dim=int(response_dimensions.get(name, 1)),
      )
    return cls(interfaces=specs, **kwargs)

  def interface(self, name):
    try:
      return self.interfaces[name]
    except KeyError as error:
      raise KeyError("Unknown PRSS interface type: {}".format(name)) from error

  def as_dict(self):
    return {
      "interfaces": {
        name: {
          "raw_dim": spec.raw_dim,
          "candidate_dim": spec.candidate_dim,
          "host_dim": spec.host_dim,
          "response_dim": spec.response_dim,
          "dimensional_compression": spec.dimensional_compression,
          "compression_ratio": spec.compression_ratio if spec.dimensional_compression else 1.0,
        }
        for name, spec in self.interfaces.items()
      },
      "context_dim": self.context_dim,
      "lambda_task": self.lambda_task,
      "lambda_resp": self.lambda_resp,
      "lambda_spec": self.lambda_spec,
      "gram_ema_rho": self.gram_ema_rho,
      "spectral_update_interval": self.spectral_update_interval,
      "spectral_warmup_steps": self.spectral_warmup_steps,
      "ridge_eps": self.ridge_eps,
      "procrustes_align": self.procrustes_align,
      "spectral_step_size": self.spectral_step_size,
      "spectral_eigen_floor_ratio": self.spectral_eigen_floor_ratio,
      "no_vanilla_pretrain": True,
    }
