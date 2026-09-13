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
                   interface_influence_rows, normalized_violation,
                   rows_backend_stats)
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
        # param_groups may carry keys the constructor does not accept (torch
        # 2.8 adds `decoupled_weight_decay`), so keep only real signature args
        try:
            import inspect
            sig = inspect.signature(type(base).__init__)
            if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
                allowed = None                     # takes **kwargs: pass all
            else:
                allowed = set(sig.parameters) - {"self", "params"}
        except (TypeError, ValueError):
            allowed = None
        g0 = {k: v for k, v in base.param_groups[0].items()
              if allowed is None or k in allowed}
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
    d_real_cpu = d_real.detach().cpu()
    nr_real = float(d_real_cpu.norm())
    m = {"real_dtheta_norm": nr_real}
    Gd = _row_dot(G_cpu, d_real_cpu)
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
        # both sides on CPU: d_real lives on the Gamma params' device.
        # cos is clamped: fp32 rounding can push a parallel pair to 1+1e-7.
        dstar = (t + corr).detach().flatten().float().cpu()
        c = float((dstar * d_real_cpu).sum()
                  / max(float(dstar.norm()) * nr_real, 1e-30))
        m["real_cos_dstar"] = max(-1.0, min(1.0, c))
    return m


def _proposal_space_update(optimizer, scheduler, gamma_params, G_cpu, g_task,
                           kappa, tau_feas, grad_clip, proj_kw, diag):
    """Project the AdamW PROPOSAL, not the raw gradient.

    The raw-gradient projection certifies a direction the optimizer is then
    free to rotate: measured on this host, cos(d*, d_theta_real) was 0.56-0.63
    and the realized violation was ~5x tau.  So instead:

      1. snapshot Gamma_old
      2. clip + AdamW.step() ONCE  -- the optimizer proposes Delta_adam
         (its m/v keep their normal task history; no pseudo-gradient)
      3. solve the SAME QP with t = Delta_adam:
             Delta* = argmin 1/2||Delta - Delta_adam||^2
                      s.t. g_i.Delta >= -kappa||g_i|| ||Delta_adam||
         using the PRE-STEP interface gradients g_i(theta_t)
      4. write Gamma <- Gamma_old + Delta*
      5. HARD GATE: the full-interface certificate on Delta* itself
         (v_max(Delta*) <= tau, all interfaces, not just the active set).  On
         failure roll the parameters AND the optimizer state back.
         The fp32 writeback is then audited (r_wb = Delta_stored - Delta*,
         its relative size, cos(Delta_stored, Delta*), v_max(Delta_stored))
         but is NOT a gate: at these step sizes |r_fp32| is comparable to
         ||g_i|| ||Delta_adam||, so v_max(Delta_stored) measures fp32
         quantisation rather than a violated constraint.

    In words: the projection certifies the optimizer proposal that is applied
    to Gamma, up to finite-precision parameter writeback.

    There is no second optimizer step: the proposal is projected, not
    re-derived.
    """
    theta_old = [p.detach().clone() for p in gamma_params]
    sizes = [p.numel() for p in gamma_params]

    # 4. ONE global clip.  The CLIPPED gradient is used for BOTH the preview
    #    and the real moment update -- never preview-unclipped / step-clipped.
    diag["g_gamma_clip"] = float(
        torch.nn.utils.clip_grad_norm_(gamma_params, grad_clip))
    g_clipped = [p.grad.detach().clone() if p.grad is not None
                 else torch.zeros_like(p) for p in gamma_params]

    # 5. STATELESS preview of the AdamW proposal: a real AdamW on CLONES with a
    #    deep copy of the optimizer state.  The real parameters and the real
    #    m/v/step are untouched, and the preview is the exact analytic proposal
    #    (no parameter-write rounding).
    delta_task = _counterfactual_task_step(optimizer, gamma_params,
                                           g_clipped, grad_clip)
    if delta_task is None:
        diag["gamma_aborted"] = True
        diag["gamma_abort_reason"] = "adamw_preview_unavailable"
        return diag
    delta_adam = torch.cat([d.flatten().float() for d in delta_task])
    norm_adam = float(delta_adam.norm())
    diag["delta_adam_norm"] = norm_adam

    # 6. proposal-space QP.  7. HARD GATE = full-interface certificate on
    #    Delta* itself (all interfaces, not just the active set).  Nothing has
    #    been written yet, so an abort needs no rollback at all.
    pj = active_set_feasibility_projection(
        gamma_params=gamma_params, G_cpu=G_cpu, kappa=kappa,
        t_override=delta_adam, write_grad=False, **proj_kw)
    diag.update(pj)
    if not pj.get("proj_feasible", True):
        diag["gamma_aborted"] = True
        diag["gamma_abort_reason"] = (
            pj.get("proj_abort")
            or ("active_budget_exhausted"
                if pj.get("proj_n_active", 0) >= proj_kw.get("max_active", 2048)
                else "proposal_not_certified"))
        return diag
    diag["vmax_proj"] = pj.get("proj_max_viol_after")

    # 8. advance the moments EXACTLY ONCE, with the SAME clipped gradient
    for p, g in zip(gamma_params, g_clipped):
        p.grad = g.detach().clone()
    optimizer.step()

    # 9. Gamma <- Gamma_t + Delta*  -- a FULL overwrite, never += on the
    #    parameter AdamW just wrote.
    #    If the QP did NOTHING (no interface binds), keep AdamW's own write
    #    byte-for-byte: an unconstrained boundary is then EXACTLY the plain
    #    task step, not a re-rounding of it.
    d_star = pj["_d_star"].to(gamma_params[0].device)
    if pj.get("proj_n_active", 0) == 0:
        diag["proj_noop_kept_adamw_write"] = 1
        d_star = delta_adam.to(d_star.device)
    else:
        with torch.no_grad():
            for p, o, dv in zip(gamma_params, theta_old,
                                torch.split(d_star, sizes)):
                p.data.copy_(o + dv.reshape(p.shape).to(p.dtype))

    # 10. scheduler exactly once
    if scheduler is not None:
        scheduler.step()
    diag["gamma_steps"] = 1

    # 11. finite-precision writeback AUDIT (never a gate) -------------------
    # Gamma_{t+1} = fl32(Gamma_t + Delta*), so Delta_stored = Delta* + r_fp32.
    # At these step sizes |r_fp32| is comparable to the normalising scale
    # ||g_i|| ||Delta_adam||, so v_max(Delta_stored) measures fp32
    # QUANTISATION, not a violated constraint.  Reported, never enforced.
    delta_stored = torch.cat([p.detach().flatten().float() - o.flatten().float()
                              for p, o in zip(gamma_params, theta_old)])
    ds = d_star.detach().cpu().flatten().float()
    dsc = delta_stored.detach().cpu()
    r_wb = dsc - ds
    n_ds = float(ds.norm())
    n_rwb = float(r_wb.norm())
    diag["dstar_norm"] = n_ds
    diag["wb_resid_norm"] = n_rwb
    diag["wb_resid_ratio"] = n_rwb / max(n_ds, 1e-30)
    diag["stored_delta_norm"] = float(dsc.norm())
    diag["stored_cos_dstar"] = max(-1.0, min(1.0, float(
        (ds * dsc).sum() / max(n_ds * float(dsc.norm()), 1e-30))))
    diag["stored_vmax"] = normalized_violation(
        G_cpu, pj["_ng"], pj["_valid"], delta_stored, norm_adam, kappa)[0]
    diag["proj_shift_ratio"] = float(
        (ds - delta_adam.detach().cpu()).norm() / max(norm_adam, 1e-30))
    d_task_split = [dv.reshape(p.shape) for p, dv in
                    zip(gamma_params, torch.split(delta_adam, sizes))]
    diag.update(_realized_metrics(diag, gamma_params, theta_old,
                                  d_task_split, kappa))
    return diag


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
    row_chunk: int = 128, proposal_space: bool = True,
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
        proj_kw = dict(iters=proj_iters, iters_max=proj_iters_max,
                        tau_feas=tau_feas, max_rounds=max_rounds,
                        max_active=max_active, add_per_round=add_per_round)
        if proposal_space:
            return _proposal_space_update(
                optimizer, scheduler, gamma_params, G_cpu, g_task, kappa,
                tau_feas, grad_clip, proj_kw, diag)
        pj = active_set_feasibility_projection(
            g_task, gamma_params, G_cpu, kappa, **proj_kw)
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
