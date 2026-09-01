"""CCM-merge host integration for RPBE (plan v2 L2+).

- ``gamma_residual``: the Gamma residual R_theta attached to the arithmetic
  mean merge (M_t = mean + R_theta(M_{t-1}, h_t, t)).
- ``ccm_patch``: the attach hook + per-arm paired seed hash.

The merge-side forward edits live in the vendored official repo
(``third_party/ccm/src/arch/ccm_llama.py``), marked with RPBE
modification markers; this package holds the learnable pieces.
"""

from .ccm_patch import N_TOK_LOCK, attach_gamma, paired_seed_hash
from .gamma_residual import GammaResidual, time_features

__all__ = ["GammaResidual", "time_features", "attach_gamma",
           "paired_seed_hash", "N_TOK_LOCK"]
