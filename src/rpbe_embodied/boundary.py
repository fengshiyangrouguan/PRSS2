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

from typing import Optional, Sequence, Tuple

import torch

from .loss import (active_set_feasibility_projection, gamma_replay_loss,
                   interface_influence_rows)


def _stream_replay(gamma, pairs, cotangents, dtype, device, minibatch):
    """Accumulate sum_v <a_v, gamma(m_a_v, m_b_v)> over minibatches.

    Gradients ACCUMULATE (no zero_grad inside the loop), so the boundary gets
    ONE gradient over every interface -- the property that makes the task
    control and the projected arm differ by exactly the projection."""
    n = len(pairs)
    for i in range(0, n, minibatch):
        sl = list(range(i, min(i + minibatch, n)))
        m_a = torch.stack([pairs[j][0] for j in sl]).to(device, dtype=dtype)
        m_b = torch.stack([pairs[j][1] for j in sl]).to(device, dtype=dtype)
        gamma_replay_loss(gamma, m_a, m_b, {j: cotangents[j] for j in sl},
                          sl).backward()


def apply_gamma_boundary_update(
    *, gamma, gamma_params, optimizer, scheduler=None,
    task_pairs: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    task_cotangents: Sequence[torch.Tensor],
    rpbe_pairs: Optional[Sequence[Tuple[torch.Tensor, torch.Tensor]]] = None,
    rpbe_cotangents: Optional[Sequence[torch.Tensor]] = None,
    kappa: float = 0.05, proj_iters: int = 400, tau_feas: float = 1e-3,
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
        G_cpu, used, n_nonfinite = interface_influence_rows(
            gamma,
            {j: rpbe_cotangents[j] for j in range(n)},
            {j: rpbe_pairs[j] for j in range(n)},
            list(range(n)), params=gamma_params, device=device, chunk=row_chunk)
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
            tau_feas=tau_feas, max_rounds=max_rounds, max_active=max_active,
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

    diag["g_gamma_clip"] = float(
        torch.nn.utils.clip_grad_norm_(gamma_params, grad_clip))
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    diag["gamma_steps"] = 1
    return diag
