"""rpbe_embodied.resume — checkpoint config contract for --resume-full.

Two layers, checked differently:

  * ``COMMON_CONFIG_KEYS`` -- the training recipe that Stage7 and Stage8 both
    already carry.  ALWAYS compared strictly; a migration must never silently
    skip ``batch_size`` / ``grad_accum`` / ``mem_length`` / ...
  * ``BOUNDARY_CONFIG_KEYS`` -- the Stage8-only boundary/projection recipe.
    Compared strictly on a Stage8 checkpoint; on a Stage7 migration these keys
    simply do not exist, so they are the ONLY thing the migration flag exempts.

A Stage7 checkpoint has no ``rpbe_mode``; loading one requires
``--migrate-legacy-gamma-state``, and the caller must then RESET the Gamma
optimizer state -- Adam's first/second moments from the retired additive
``-lambda*J`` objective must never seed a feasibility-projection run.
"""
from __future__ import annotations

import warnings
from typing import Dict, Tuple

# training recipe shared by Stage7 and Stage8 -- always strictly enforced
COMMON_CONFIG_KEYS = (
    "sched", "batch_size", "grad_accum", "gamma_replay_batch_size",
    "gamma_task_boundary_episodes", "rpbe_stats_episodes", "lambda_rpbe",
    "mem_length", "kf_min_abs",
)

# Stage8-only boundary/projection recipe; absent from Stage7 checkpoints
BOUNDARY_CONFIG_KEYS = (
    "rpbe_mode", "kappa", "proj_iters", "proj_iters_max", "proj_tau",
    "proj_max_active", "proj_max_rounds", "proj_add_per_round",
    "rpbe_proposal_space",
)


def boundary_config(args) -> Dict[str, object]:
    """The subset of run args the Stage8 boundary protocol depends on."""
    return {k: getattr(args, k) for k in BOUNDARY_CONFIG_KEYS}


def common_config(args) -> Dict[str, object]:
    """The shared training recipe that every resume must reproduce exactly."""
    return {k: getattr(args, k) for k in COMMON_CONFIG_KEYS}


def verify_resume_config(ck_config: dict, want: dict,
                         *, allow_legacy_gamma: bool = False
                         ) -> Tuple[Dict[str, tuple], bool]:
    """Compare a checkpoint's config block against the requested run.

    Returns ``(bad, legacy)``.  ``bad`` maps key -> (checkpoint, requested) for
    every mismatch.  A Stage7 checkpoint (no ``rpbe_mode``) is tolerated only
    with ``allow_legacy_gamma``; that exempts ONLY the Stage8-only boundary
    keys -- every COMMON key present in ``want`` is still compared, so a
    migration cannot silently change the shared training recipe.
    """
    legacy = "rpbe_mode" not in ck_config
    if legacy and allow_legacy_gamma:
        bad = {k: (ck_config.get(k), v) for k, v in want.items()
               if k not in BOUNDARY_CONFIG_KEYS and ck_config.get(k) != v}
        return bad, True
    bad = {k: (ck_config.get(k), v) for k, v in want.items()
           if ck_config.get(k) != v}
    return bad, False


def realign_lambda_scheduler(sched, step: int):
    """Move a LambdaLR to the state it would have after ``step`` step() calls.

    Needed on the legacy migration path: the Gamma scheduler is deliberately
    NOT loaded from the checkpoint (its optimizer state is reset), so without
    this it would sit at step 0 and hand out the warmup learning rate however
    far the run had already progressed.  Returns the resulting LR.
    """
    if sched is None:
        return None
    sched.last_epoch = int(step) - 1
    with warnings.catch_warnings():
        # realignment is deliberately a step() with no preceding optimizer step
        warnings.simplefilter("ignore", UserWarning)
        sched.step()
    return sched.get_last_lr()
