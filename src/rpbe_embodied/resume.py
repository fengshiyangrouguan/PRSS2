"""rpbe_embodied.resume — checkpoint config contract for --resume-full.

The Stage8 boundary recipe (rpbe_mode + the projection knobs) must not change
silently across a resume, so it is written into the checkpoint config and
verified on load.  Stage7 checkpoints predate ``rpbe_mode``; they are only
loadable with an explicit migration flag, and the caller is then required to
RESET the Gamma optimizer state -- Adam's first/second moments from the
retired additive ``-lambda*J`` objective must never seed a feasibility-
projection run.
"""
from __future__ import annotations

from typing import Dict, Tuple

# the Stage8 boundary/projection recipe written into every checkpoint config
BOUNDARY_CONFIG_KEYS = (
    "rpbe_mode", "kappa", "kappa_anneal_start", "kappa_anneal_end",
    "kappa_anneal_to", "proj_iters", "proj_tau", "proj_max_active",
    "proj_max_rounds", "proj_add_per_round",
)


def boundary_config(args) -> Dict[str, object]:
    """The subset of run args the boundary protocol depends on."""
    return {k: getattr(args, k) for k in BOUNDARY_CONFIG_KEYS}


def verify_resume_config(ck_config: dict, want: dict,
                         *, allow_legacy_gamma: bool = False
                         ) -> Tuple[Dict[str, tuple], bool]:
    """Compare a checkpoint's config block against the requested run.

    Returns ``(bad, legacy)``.  ``bad`` maps key -> (checkpoint, requested) for
    every mismatch (including a key absent from a Stage8-format checkpoint).
    A Stage7 checkpoint has no ``rpbe_mode`` at all: that is tolerated only
    with ``allow_legacy_gamma``, which returns ``legacy=True`` and an empty
    ``bad`` so the caller can proceed with a Gamma-optimizer reset.
    """
    if "rpbe_mode" not in ck_config and allow_legacy_gamma:
        return {}, True
    bad = {k: (ck_config.get(k), v) for k, v in want.items()
           if ck_config.get(k) != v}
    return bad, False
