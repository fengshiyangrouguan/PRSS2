"""RPBE configuration.

State widths ``d_tau`` come from the host interface alone — they are the
propagation width the host requires, NOT a matrix rank and NOT a compression
ratio hyperparameter.  The compression RPBE performs is information deletion
in the predictive-quotient sense, even at d_tau == host width.  Everything
else configures the fixed measurement and the Ky Fan training term.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional


@dataclass
class RPBConfig:
    state_dims: Mapping[str, int]        # tau -> d_tau (host-required width)
    own_dims: Mapping[str, int]          # tau -> dim of o_v (host input width)
    width_D: int = 128                   # shared core working width
    m: int = 64                          # sketch output dim (loss test dim)
    d_c: int = 32                        # context feature dim
    d_f: int = 32                        # future feature dim
    lambda_kf: float = 1e-3              # weight of the component loss in the total
    alphas: Dict[str, float] = field(default_factory=dict)  # tau -> weight; default 1
    ridge_eps: float = 1e-4              # relative ridge (x tr(Sigma)/dim)
    delta_t_scale: float = 1e6           # fixed scale for continuous delta_t RFF
    cuts_per_tau: int = 32              # depth-balanced cuts per interface/batch
    kf_min_ratio: float = 2.0            # window: unique cuts >= ratio * d_tau
    kf_min_abs: int = 1024               # gate: effective cut count >= this floor
    # Macro-group length in batches.  Default None = ceil(kf_min_abs /
    # trace_roots), which assumes every traced root contributes a cut;
    # the real valid-root rate is lower (strict-future masking), so runs
    # that observe below_threshold_groups > 0 should set this explicitly
    # (e.g. 40 for 32 roots at ~80% validity).
    kf_group_batches: Optional[int] = None
    kf_taus: Optional[List[str]] = None  # optional whitelist; the host adapter
                                         # always intersects it with internal
                                         # compressible interfaces (never leaf/root)
    # Table-2 ablation switches (paper spec section 4):
    #   full_balancing  J_KF = ||S_ZZ^-1/2 S_ZP S_PP^-1/2||_F^2 (the method)
    #   diagonal        J_diag = ||D_Z^-1/2 S_ZP D_P^-1/2||_F^2
    #   reconstruction  J_rec = tr(S_UZ S_ZZ^-1 S_ZU) (PCA-equivalent when
    #                   the encoder is linear; needs CutCandidate.u)
    kf_variant: str = "full_balancing"
    # 1 = one local task observation (Y1 only); 2 = the two-observation
    # pullback refinement (Y1 and Y2).  Both keep per-tree total weight 1.
    n_observations: int = 2
    rpbe_seed: int = 0                   # fixed-measurement seed (independent of host)

    def __post_init__(self):
        self.state_dims = dict(self.state_dims)
        self.own_dims = dict(self.own_dims)
        if not self.state_dims:
            raise ValueError("RPBConfig requires at least one interface")
        if set(self.state_dims) != set(self.own_dims):
            raise ValueError("state_dims and own_dims must share keys")
        for name, r in self.state_dims.items():
            if int(r) <= 0:
                raise ValueError("r_tau must be positive for {}".format(name))
        if self.width_D <= 0 or self.m <= 0 or self.d_c <= 0 or self.d_f <= 0:
            raise ValueError("width/m/d_c/d_f must be positive")
        if self.ridge_eps <= 0 or self.delta_t_scale <= 0:
            raise ValueError("ridge_eps and delta_t_scale must be positive")
        if self.kf_min_ratio <= 0 or self.kf_min_abs < 2:
            raise ValueError("kf_min_ratio > 0 and kf_min_abs >= 2")
        if self.cuts_per_tau < 1:
            raise ValueError("cuts_per_tau must be >= 1")
        if self.kf_variant not in ("full_balancing", "diagonal",
                                   "reconstruction"):
            raise ValueError("unknown kf_variant {}".format(self.kf_variant))
        if self.n_observations not in (1, 2):
            raise ValueError("n_observations must be 1 or 2")
        if self.kf_taus is not None:
            bad = set(self.kf_taus) - set(self.state_dims)
            if bad:
                raise ValueError("kf_taus not in state_dims: {}".format(
                    sorted(bad)))

    def alpha(self, tau: str) -> float:
        return float(self.alphas.get(tau, 1.0))

    def as_dict(self):
        return {
            "state_dims": dict(self.state_dims),
            "own_dims": dict(self.own_dims),
            "width_D": self.width_D,
            "m": self.m,
            "d_c": self.d_c,
            "d_f": self.d_f,
            "lambda_kf": self.lambda_kf,
            "alphas": dict(self.alphas),
            "ridge_eps": self.ridge_eps,
            "delta_t_scale": self.delta_t_scale,
            "cuts_per_tau": self.cuts_per_tau,
            "kf_min_ratio": self.kf_min_ratio,
            "kf_min_abs": self.kf_min_abs,
            "kf_group_batches": self.kf_group_batches,
            "kf_taus": list(self.kf_taus) if self.kf_taus is not None else None,
            "kf_variant": self.kf_variant,
            "n_observations": self.n_observations,
            "rpbe_seed": self.rpbe_seed,
        }
