"""EmbodiedRPBConfig — frozen method configuration for the MemoryVLA host.

Field names mirror RPBConfig (src/rpbe/config.py) where semantics overlap.
lambda_rpbe is written in AFTER the Task-8 calibration run (r_eff rule);
the frozen value must be recorded here, not silently read from defaults.
"""
from dataclasses import dataclass, field


@dataclass
class EmbodiedRPBConfig:
    # --- fixed measurement ---
    m: int = 64                  # CountSketch output dim for P
    d_c: int = 64                # context feature dim
    d_cv: int = 32               # vision fixed-projection dim (raw 2176 -> 32)
    d_f: int = 64                # action RFF dim
    action_rff_scale: float = 0.16
    num_counter_bins: int = 4096
    rpbe_seed: int = 0

    # --- window / statistics ---
    ridge_eps: float = 1e-4
    kf_variant: str = "full_dual"   # "full_dual" | "diag"
    kf_min_ratio: float = 2.0
    kf_min_abs: int = 128           # min unique merges per window
    horizon_weights: tuple = (0.5, 0.5)

    # --- gamma ---
    gamma_rank: int = 64
    gamma_alpha_init: float = 0.0

    # --- objective ---
    lambda_rpbe: float = 0.0        # frozen after calibration (Task 8)

    # --- temporal metadata ---
    delta_s_scale: float = 1.0

    def freeze_lambda(self, value: float) -> None:
        self.lambda_rpbe = value
