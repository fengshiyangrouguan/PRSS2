"""rpbe_embodied — RPBE for the MemoryVLA host (LIBERO-Mem).

Independent package (plan §31.12): no imports from src.rpbe, no imports
from the MemoryVLA vendored tree.  The TGN implementation stays frozen.
"""
from .boundary import apply_gamma_boundary_update  # noqa: F401
from .config import EmbodiedRPBConfig  # noqa: F401
from .loss import (  # noqa: F401
    EmbodiedRPBEWindow,
    active_set_feasibility_projection,
    diag_latent_z_adjoint,
    diag_score,
    dual_full_score,
    dual_latent_z_adjoint,
    dual_latent_z_adjoint_modes,
    gamma_replay_loss,
    interface_influence_rows,
    kappa_at,
    treewise_feasibility_projection,
)
from .maps import EmbodiedFixedMaps  # noqa: F401
from .records import (  # noqa: F401
    EmbodiedCutRow,
    MergeRecord,
    PendingMerge,
    PendingMergeQueue,
)
from .resume import boundary_config, verify_resume_config  # noqa: F401
