"""rpbe_embodied.boundary — one Gamma macro-boundary update.

The VLA trainer calls this instead of hand-rolling the boundary.  Keeping the
protocol here makes the invariant directly testable without the 30 GB host:

    gamma-task and gamma-rpbe assemble EXACTLY ONE Gamma gradient per
    boundary and take EXACTLY ONE clip + ONE optimizer.step(); they differ
    only by the feasibility projection applied to the Gamma rows.

RPBE is a constraint, not an objective: the accumulated task direction
t = -g_task is projected onto the interface-wise feasible set
g_i.d >= -kappa*||g_i||*||t|| and only the Gamma gradient is replaced.  If the
projection cannot reach feasibility within tolerance the whole boundary is
ABORTED (no Gamma step) rather than applied with a violated preservation
constraint.
"""
from __future__ import annotations

import copy
from typing import Optional, Sequence, Tuple

import torch

from .loss import (active_set_feasibility_projection, gamma_replay_loss,
                   interface_influence_rows, rows_backend_stats)
from .loss import _row_dot, _stack_to


def _stream_replay(gamma, pairs, cotangents, dtype, device, minibatch):
    """Accumulate sum_v <a_v, gamma(m_a_v, m_b_v)> over minibatches.

    Gradients ACCUMULATE (no zero_grad inside the loop), so the boundary gets
    ONE gradient over every interface -- the property that makes the task
    control and the projected arm differ by exactly the projection.

    Pairs may hold tensors on MIXED devices (a merged child lives on CUDA, a
    leaf child on CPU), so every element is cast before the stack."""
    n = len(pairs)
    for i in range(0, n, minibatch):
        sl = list(range(i, min(i + minibatch, n)))
        m_a = _stack_to([pairs[j][0] for j in sl], device, dtype)
        m_b = _stack_to([pairs[j][1] for j in sl], device, dtype)
        gamma_replay_loss(gamma, m_a, m_b, {j: cotangents[j] for j in sl},
                          sl).backward()


def _counterfactual_task_step(optimizer, gamma_params, g_task, grad_clip):
    """The AdamW displacement this boundary WOULD have taken with the PURE
    task gradient, from the SAME pre-step optimizer state.

    Executed on clones, so real parameters and optimizer state are untouched.
    Returns the per-parameter displacement, or None if the optimizer type
    cannot be reconstructed.  No forward pass is involved, so no RNG is
    consumed and the training stream is unchanged."""
    try:
        # unwrap thin optimizer wrappers (e.g. step counters) so the clone is
        # built from the real optimizer class
        base = getattr(optimizer, "opt", optimizer)
        g0 = {k: v for k, v in base.param_groups[0].items() if k != "params"}
        state = copy.deepcopy(base.state_dict())
        clones = [torch.nn.Parameter(p.detach().clone()) for p in gamma_params]
        cf = type(base)(clones, **g0)
        cf.load_state_dict(state)
        for c, gt in zip(clones, g_task):
            c.grad = None if gt is None else gt.detach().clone()
        torch.nn.utils.clip_grad_norm_(clones, grad_clip)
        cf.step()
        return [c.detach() - p.detach() for c, p in zip(clones, gamma_params)]
    except Exception as e:  # noqa: BLE001 -- diagnostics must never break training
        print(f"[gamma proj] WARNING: counterfactual AdamW step unavailable "
              f"({type(e).__name__}: {e}); realized metrics skipped.", flush=True)
        return None


def _realized_metrics(pj, gamma_params, theta_before, d_task, kappa):
    """Compare the RAW certified direction d* with what AdamW actually did.

    Uses the PRE-STEP interface gradients g_i(theta_t) -- never recomputed at
    theta_{t+1}, which would be a different local geometry.  Reports

      real_cos_dstar        cos(d*, d_theta_real)   -- how far AdamW twisted
                                                      the projected raw direction
      real_cos_{min,median} cos_i = g_i.d_theta_real / (||g_i|| ||d_theta_real||)
      real_vmax             the ORIGINAL half-space, but with the right-hand
                            side taken from the counterfactual TASK-ONLY AdamW
                            displacement:
                              g_i.d_real >= -kappa ||g_i|| ||d_task_AdamW||
                            i.e. "if there had been no projection, AdamW would
                            have moved this far -- did the realized step stay
                            inside the relaxed half-space?"
      real_dtheta_ratio     ||d_real|| / ||d_task_AdamW||

    PRECISION NOTE: d_real is recovered by subtracting fp32 parameters whose
    magnitude is O(1) while a step is O(lr) ~ 1e-5, so its MAGNITUDE is known
    to only a few digits and ||d_real|| is biased slightly high (the rounding
    noise adds energy).  The DIRECTION quantities -- cos(d*, d_real) and
    cos_i -- average that per-coordinate noise over ~0.6M dimensions and are
    therefore the trustworthy ones.  Read real_dtheta_ratio as indicative.
    """
    G_cpu = pj.get("_G_cpu")
    ng = pj.get("_ng")
    valid = pj.get("_valid")
    corr = pj.get("_corr")
    t = pj.get("_t")
    if G_cpu is None or ng is None or valid is None:
        return {}
    d_real = torch.cat([p.detach().flatten().float() - p0.flatten().float()
                        for p, p0 in zip(gamma_params, theta_before)])
    nr_real = float(d_real.norm())
    m = {"real_dtheta_norm": nr_real}
    Gd = _row_dot(G_cpu, d_real.cpu())
    cos_i = torch.where(valid, Gd / (ng * max(nr_real, 1e-30) + 1e-30),
                        torch.zeros_like(Gd))
    m["real_cos_min"] = float(cos_i[valid].min()) if bool(valid.any()) else 0.0
    m["real_cos_median"] = (float(cos_i[valid].median())
                            if bool(valid.any()) else 0.0)
    m["real_n_below_neg_kappa"] = int((cos_i[valid] < -kappa).sum())
    if d_task is not None:
        dt = torch.cat([x.flatten().float() for x in d_task])
        nr_task = float(dt.norm())
        m["real_dtheta_task_norm"] = nr_task
        m["real_dtheta_ratio"] = nr_real / max(nr_task, 1e-30)
        if nr_task > 1e-30:
            scale = ng * nr_task
            vv = torch.where(valid,
                             torch.clamp(-kappa * scale - Gd, min=0.0)
                             / (scale + 1e-30), torch.zeros_like(Gd))
            m["real_vmax"] = float(vv.max())
        else:
            m["real_vmax"] = float("nan")
    if corr is not None and t is not None:
        dstar = (t + corr).detach().cpu().flatten().float()
        m["real_cos_dstar"] = float(
            (dstar * d_real).sum()
            / max(float(dstar.norm()) * nr_real, 1e-30))
    return m


def apply_gamma_boundary_update(
    *, gamma, gamma_params, optimizer, scheduler=None,
    task_pairs: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    task_cotangents: Sequence[torch.Tensor],
    rpbe_pairs: Optional[Sequence[Tuple[torch.Tensor, torch.Tensor]]] = None,
    rpbe_cotangents: Optional[Sequence[torch.Tensor]] = None,
    kappa: float = 0.05, proj_iters: int = 400, proj_iters_max: int = 1600,
    tau_feas: float = 1e-3,
    max_rounds: int = 8, max_active: int = 2048, add_per_round: int = 512,
    grad_clip: float = 1.0, minibatch: int = 64, device: str = "cuda",
    row_chunk: int = 128,
) -> dict:
    """One boundary: accumulate task grad -> (optionally) project -> clip -> step.

    Returns diagnostics with ``gamma_steps`` in {0, 1} and, when a projection
    ran, the full ``proj_*`` block plus ``proj_feasible``.  On an infeasible
    projection ``gamma_steps == 0`` and ``gamma_aborted`` is set; the caller
    must NOT substitute a task-only step (that is exactly the update the
    feasibility set forbids).
    """
    diag: dict = {"gamma_steps": 0, "gamma_aborted": False,
                  "n_task": len(task_pairs),
                  "n_rpbe": len(rpbe_pairs or ())}
    if not gamma_params or not task_pairs:
        diag["gamma_aborted"] = True
        diag["gamma_abort_reason"] = "no_task_keys"
        return diag
    dtype = gamma_params[0].dtype

    optimizer.zero_grad()
    _stream_replay(gamma, task_pairs, task_cotangents, dtype, device, minibatch)
    # p.grad now holds the accumulated g_task -- also the fallback value when
    # no constraint is written (empty / all-zero rows).
    g_task = [p.grad.detach().clone() if p.grad is not None
              else torch.zeros_like(p) for p in gamma_params]
    g_task_flat = torch.cat([g.flatten().float() for g in g_task])
    diag["g_gamma_task"] = float(g_task_flat.norm())
    if not bool(torch.isfinite(g_task_flat).all()):
        optimizer.zero_grad()
        diag["gamma_aborted"] = True
        diag["gamma_abort_reason"] = "nonfinite_task_gradient"
        return diag

    if rpbe_pairs:
        n = len(rpbe_pairs)
        be0 = rows_backend_stats()
        G_cpu, used, n_nonfinite = interface_influence_rows(
            gamma,
            {j: rpbe_cotangents[j] for j in range(n)},
            {j: rpbe_pairs[j] for j in range(n)},
            list(range(n)), params=gamma_params, device=device, chunk=row_chunk)
        be1 = rows_backend_stats()
        diag["rows_batched_chunks"] = be1["batched_chunks"] - be0["batched_chunks"]
        diag["rows_fallback_chunks"] = (be1["fallback_chunks"]
                                        - be0["fallback_chunks"])
        if diag["rows_fallback_chunks"]:
            # visible in the training log: the slow path is never silent
            diag["rows_backend_error"] = be1["last_error"]
        diag["proj_n_rows_used"] = len(used)
        diag["proj_n_nonfinite"] = n_nonfinite
        if n_nonfinite:
            # FAIL CLOSED: one bad interface gradient must not silently reduce
            # the constraint set the boundary claims to satisfy.
            optimizer.zero_grad()
            diag["gamma_aborted"] = True
            diag["gamma_abort_reason"] = "nonfinite_constraint_row"
            return diag
        pj = active_set_feasibility_projection(
            g_task, gamma_params, G_cpu, kappa, iters=proj_iters,
            iters_max=proj_iters_max, tau_feas=tau_feas,
            max_rounds=max_rounds, max_active=max_active,
            add_per_round=add_per_round)
        diag.update(pj)
        if not pj.get("proj_feasible", True):
            optimizer.zero_grad()
            diag["gamma_aborted"] = True
            diag["gamma_abort_reason"] = (
                pj.get("proj_abort")
                or ("active_budget_exhausted"
                    if pj.get("proj_n_active", 0) >= max_active
                    else "projection_not_certified"))
            return diag

    # --- realized-displacement diagnostics -------------------------------
    # theta_t and the counterfactual task-only AdamW step must both be taken
    # from the PRE-STEP state.  The counterfactual runs on clones, so it does
    # not touch the real parameters, the real optimizer state, or the RNG.
    theta_before = [p.detach().clone() for p in gamma_params]
    d_task = (_counterfactual_task_step(optimizer, gamma_params, g_task,
                                        grad_clip) if rpbe_pairs else None)

    diag["g_gamma_clip"] = float(
        torch.nn.utils.clip_grad_norm_(gamma_params, grad_clip))
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    diag["gamma_steps"] = 1
    if rpbe_pairs:
        diag.update(_realized_metrics(diag, gamma_params, theta_before,
                                      d_task, kappa))
    return diag
