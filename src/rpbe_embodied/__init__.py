"""rpbe_embodied — RPBE for the MemoryVLA host (LIBERO-Mem).

Independent package (plan §31.12): no imports from src.rpbe, no imports
from the MemoryVLA vendored tree.  The TGN implementation stays frozen.
"""
from .config import EmbodiedRPBConfig  # noqa: F401
from .loss import (  # noqa: F401
    EmbodiedRPBEWindow,
    diag_latent_z_adjoint,
    diag_score,
    dual_full_score,
    dual_latent_z_adjoint,
    gamma_replay_loss,
)
from .maps import EmbodiedFixedMaps  # noqa: F401
from .records import (  # noqa: F401
    EmbodiedCutRow,
    MergeRecord,
    PendingMerge,
    PendingMergeQueue,
)
