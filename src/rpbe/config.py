"""RPBE configuration.

Interface widths ``r_tau`` come from the host alone (the hard propagation
budget); everything else configures the fixed measurement and the Ky Fan
training term.
"""

from dataclasses import dataclass, field
from typing import Dict, Mapping


@dataclass
class RPBConfig:
    interfaces: Mapping[str, int]        # tau -> r_tau (host-decided budget)
    own_dims: Mapping[str, int]          # tau -> dim of o_v (host input width)
    width_D: int = 128                   # shared core working width
    m: int = 256                         # sketch output dim (loss test dim)
    d_c: int = 32                        # context feature dim
    d_f: int = 32                        # future feature dim
    lambda_kf: float = 1.0               # weight of the component loss in the total
    alphas: Dict[str, float] = field(default_factory=dict)  # tau -> weight; default 1
    ridge_eps: float = 1e-4              # relative ridge (x tr(Sigma)/dim)
    delta_t_scale: float = 1e6           # fixed scale for continuous delta_t RFF
    neg_per_cut: int = 4                 # stage-1 negative rows per cut
    min_cuts_per_type: int = 32          # skip a tau's term below this many rows
    rpbe_seed: int = 0                   # fixed-measurement seed (independent of host)

    def __post_init__(self):
        self.interfaces = dict(self.interfaces)
        self.own_dims = dict(self.own_dims)
        if not self.interfaces:
            raise ValueError("RPBConfig requires at least one interface")
        if set(self.interfaces) != set(self.own_dims):
            raise ValueError("interfaces and own_dims must share keys")
        for name, r in self.interfaces.items():
            if int(r) <= 0:
                raise ValueError("r_tau must be positive for {}".format(name))
        if self.width_D <= 0 or self.m <= 0 or self.d_c <= 0 or self.d_f <= 0:
            raise ValueError("width/m/d_c/d_f must be positive")
        if self.ridge_eps <= 0 or self.delta_t_scale <= 0:
            raise ValueError("ridge_eps and delta_t_scale must be positive")
        if self.min_cuts_per_type < 1:
            raise ValueError("min_cuts_per_type must be >= 1")

    def alpha(self, tau: str) -> float:
        return float(self.alphas.get(tau, 1.0))

    def as_dict(self):
        return {
            "interfaces": dict(self.interfaces),
            "own_dims": dict(self.own_dims),
            "width_D": self.width_D,
            "m": self.m,
            "d_c": self.d_c,
            "d_f": self.d_f,
            "lambda_kf": self.lambda_kf,
            "alphas": dict(self.alphas),
            "ridge_eps": self.ridge_eps,
            "delta_t_scale": self.delta_t_scale,
            "neg_per_cut": self.neg_per_cut,
            "min_cuts_per_type": self.min_cuts_per_type,
            "rpbe_seed": self.rpbe_seed,
        }
