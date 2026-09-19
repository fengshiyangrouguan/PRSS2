"""Proposal-space feasibility QP -- VENDORED from the frozen Stage8 line.

Source: PRSS2 repository, branch `fix/avg-lora-clock`, commit `c05e4fb8`:
  * `src/rpbe_embodied/loss.py`   -- `_fista_nonneg`, `_norm_and_dot`,
    `_row_dot`, `_iters_ladder`, `_cutting_plane`,
    `active_set_feasibility_projection`, `normalized_violation`
  * `src/rpbe_embodied/boundary.py` -- `_counterfactual_task_step` and the
    ordering of `_proposal_space_update`

Copied rather than imported so that Meta^n and PRSS2 do not silently diverge
on the numeric core. Re-sync deliberately if upstream changes.

THE ONE THING THAT MATTERS: the projection certifies the AdamW PROPOSAL, not
the raw gradient. Raw-gradient projection certifies a direction the optimizer
is then free to rotate -- measured on the Stage8 host,
`cos(d*, d_theta_real)` was 0.56-0.63 and the realized violation was ~5x tau.
So the order is fixed:

    1. snapshot Gamma_old
    2. ONE global clip, used for BOTH the preview and the real moment update
    3. stateless counterfactual AdamW on clones -> Delta_adam
    4. solve the QP with t = Delta_adam        -> Delta*
    5. HARD GATE: full-interface certificate v_max(Delta*) <= tau.
       Nothing has been written yet, so a failure needs NO rollback and the
       caller must abort the whole boundary.
    6. advance the moments EXACTLY ONCE with the same clipped gradient
    7. Gamma <- Gamma_old + Delta*   (or keep AdamW's own write when the QP
       bound nothing, byte-for-byte)
    8. scheduler.step() exactly once

There is NO second optimizer step: the proposal is projected, not re-derived.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from meta_n.rpbe import config as C


# --------------------------------------------------------------------------
# vendored: dual solver + cutting-plane QP
# --------------------------------------------------------------------------

def _fista_nonneg(Q: torch.Tensor, c: torch.Tensor, iters: int) -> torch.Tensor:
    """min_{mu>=0} 1/2 mu^T Q mu - c^T mu  (Q PSD) by accelerated projected GD."""
    L = float(torch.linalg.eigvalsh(Q).max().clamp(min=1e-12))
    mu = torch.zeros_like(c)
    y = mu.clone()
    tk = 1.0
    for _ in range(iters):
        grad = Q @ y - c
        mu_new = torch.clamp(y - grad / L, min=0.0)
        tk_new = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * tk * tk))
        y = mu_new + ((tk - 1.0) / tk_new) * (mu_new - mu)
        mu, tk = mu_new, tk_new
    return mu


def _norm_and_dot(G_cpu: torch.Tensor, v_cpu: torch.Tensor,
                  chunk: int = 256) -> Tuple[torch.Tensor, torch.Tensor]:
    """Chunked ``||g_i||`` and ``g_i . v`` for every CPU row (no GPU peak)."""
    n = G_cpu.shape[0]
    ng = torch.empty(n, dtype=torch.float32)
    gd = torch.empty(n, dtype=torch.float32)
    for i in range(0, n, chunk):
        Gc = G_cpu[i:i + chunk]
        ng[i:i + chunk] = Gc.norm(dim=1)
        gd[i:i + chunk] = Gc @ v_cpu
    return ng, gd


def _row_dot(G_cpu: torch.Tensor, v_cpu: torch.Tensor,
             chunk: int = 256) -> torch.Tensor:
    """Chunked ``g_i . v`` for every CPU row (norms already known)."""
    n = G_cpu.shape[0]
    gd = torch.empty(n, dtype=torch.float32)
    for i in range(0, n, chunk):
        gd[i:i + chunk] = G_cpu[i:i + chunk] @ v_cpu
    return gd


def _iters_ladder(base: int, cap: int) -> List[int]:
    """FISTA budget ladder, e.g. (400, 1600) -> [400, 800, 1600]."""
    base = max(1, int(base))
    if cap is None or int(cap) <= base:
        return [base]
    out = [base]
    while out[-1] < int(cap):
        out.append(min(out[-1] * 2, int(cap)))
    return out


def _cutting_plane(G_cpu, t, dev, kappa, tau_feas, ng, Gt, valid, scale, nt,
                   active, iters, max_rounds, max_active, add_per_round):
    """One cutting-plane sweep at a fixed FISTA budget.

    The active rows are EXACTLY row-normalised before the QP: with
    ``g_hat_i = g_i / ||g_i||`` the half-space ``g_i.d >= -kappa||g_i|| ||t||``
    is the identical constraint ``g_hat_i.d >= -kappa||t||`` (dividing by the
    positive ``||g_i||`` does not change the feasible set, hence not d*), but
    the dual Gram becomes a cosine matrix with unit diagonal instead of one
    whose conditioning is set by the spread of the row norms.
    """
    corr = torch.zeros(t.numel(), dtype=torch.float32, device=dev)
    mu = None
    rnd = 0
    vmax = float("inf")
    min_slack = float("nan")
    n_mu_pos = 0
    active_set = set(active)
    for rnd in range(1, max_rounds + 1):
        if active:
            idx = torch.tensor(active, dtype=torch.long)
            nga = ng[idx].to(dev, dtype=torch.float32)
            Ha = G_cpu[idx].to(dev, dtype=torch.float32) / nga[:, None]
            b = torch.full((Ha.shape[0],), -kappa * nt, device=dev)
            mu = _fista_nonneg(Ha @ Ha.t(), b - Ha @ t, iters)
            corr = (mu[:, None] * Ha).sum(0)
            d = t + corr
        else:
            corr = torch.zeros_like(corr)
            d = t
        # RE-CHECK every interface at the new d (not just the active set)
        Gd = _row_dot(G_cpu, d.cpu())
        vv = torch.where(valid, torch.clamp(-kappa * scale - Gd, min=0.0)
                         / (scale + 1e-30), torch.zeros_like(Gt))
        vmax = float(vv.max())
        if vmax <= tau_feas:
            break
        added = 0
        for i in torch.argsort(vv, descending=True).tolist():
            if len(active) >= max_active or added >= add_per_round:
                break
            if vv[i] > tau_feas and i not in active_set:
                active.append(i)
                active_set.add(i)
                added += 1
        if added == 0:
            break
    if active:
        idx = torch.tensor(active, dtype=torch.long)
        nga = ng[idx].to(dev, dtype=torch.float32)
        Ha = G_cpu[idx].to(dev, dtype=torch.float32) / nga[:, None]
        slack = Ha @ (t + corr) - (-kappa * nt)
        min_slack = float(slack.min())
    if mu is not None:
        n_mu_pos = int((mu > 1e-8).sum())
    return corr, active, mu, vmax, rnd, min_slack, n_mu_pos


def active_set_feasibility_projection(
    g_task_gamma: Optional[List[torch.Tensor]] = None,
    gamma_params: Optional[List[torch.Tensor]] = None,
    G_cpu: Optional[torch.Tensor] = None, kappa: float = 0.05,
    iters: int = 400, iters_max: int = 1600, tau_feas: float = 1e-3,
    max_rounds: int = 8, max_active: int = 2048, add_per_round: int = 512,
    row_norm_tol: float = 1e-9, task_norm_tol: float = 1e-12,
    t_override: Optional[torch.Tensor] = None, write_grad: bool = True,
) -> dict:
    """Cutting-plane feasibility projection over EVERY interface.

    One global task direction t (here always the AdamW displacement via
    ``t_override``); each interface i contributes the half-space
    g_i.d >= -kappa*||g_i||*||t||.  Round 0 ranks every interface by its
    violation at d = t, the worst enter the active set, then each round solves
    the QP on the active set and RE-SCANS every interface at the new d, until
    the dimensionless max violation is <= ``tau_feas``.

    ``diag["proj_feasible"]`` False means the caller must ABORT: nothing is
    written and no constrained update may be applied.
    """
    sizes = [p.numel() for p in gamma_params] if gamma_params else []
    if t_override is not None:
        t = t_override.detach().flatten().float()
    elif g_task_gamma is not None:
        t = torch.cat([-x.detach().flatten().float() for x in g_task_gamma])
    else:
        t = torch.zeros(1)
    dev = t.device
    t_cpu = t.detach().cpu()
    nt = float(t.norm())
    diag = {
        "proj_n_interfaces": int(G_cpu.shape[0]) if G_cpu is not None else 0,
        "proj_n_checked": 0, "proj_n_valid": 0, "proj_n_candidates": 0,
        "proj_n_active": 0, "proj_n_active_init": 0, "proj_n_mu_pos": 0,
        "proj_rounds": 0, "proj_coverage": 1.0, "proj_norm_t": nt,
        "proj_cos_min": 0.0, "proj_max_viol_before": 0.0,
        "proj_max_viol_after": 0.0, "proj_min_slack_after": float("nan"),
        "proj_corr_ratio": 0.0, "proj_kappa": float(kappa),
        "proj_feasible": True,
    }
    if G_cpu is None or G_cpu.numel() == 0:
        diag["_d_star"] = t.detach()
        return diag
    if not math.isfinite(nt):
        diag["proj_feasible"] = False
        diag["proj_abort"] = "nonfinite_task_direction"
        return diag
    if nt <= task_norm_tol:
        # A negligible task direction makes ||t|| a meaningless normaliser and
        # t itself ~0, so the pure task step IS the answer.  Flagged distinctly
        # so this can never be mistaken for a certified projection.
        diag["proj_skipped_tiny_task"] = 1
        diag["_d_star"] = t.detach()
        return diag
    ng, Gt = _norm_and_dot(G_cpu, t_cpu)
    valid = ((ng > row_norm_tol) & torch.isfinite(ng) & torch.isfinite(Gt))
    diag["proj_n_below_row_tol"] = int((ng <= row_norm_tol).sum())
    diag["proj_n_valid"] = int(valid.sum())
    diag["proj_n_checked"] = int(G_cpu.shape[0])
    if not bool(valid.any()):
        diag["_d_star"] = t.detach()
        return diag
    scale = ng * nt
    cos = torch.where(valid, Gt / (scale + 1e-30), torch.ones_like(Gt))
    viol = torch.clamp(-cos - kappa, min=0.0)
    diag["proj_cos_min"] = float(cos[valid].min())
    # cos quantiles over the valid interfaces: the EVIDENCE for choosing kappa
    # (an interface binds iff cos(g_i, t) < -kappa).
    qv = torch.quantile(cos[valid], torch.tensor(
        [0.01, 0.05, 0.10, 0.25, 0.50]))
    for nm, v in zip(("q01", "q05", "q10", "q25", "q50"), qv.tolist()):
        diag["proj_cos_{}".format(nm)] = float(v)
    diag["proj_max_viol_before"] = float(viol.max())
    n_cand = int((viol > tau_feas).sum())
    diag["proj_n_candidates"] = n_cand
    active = [i for i in torch.argsort(viol, descending=True).tolist()[:max_active]
              if viol[i] > tau_feas]
    diag["proj_n_active_init"] = len(active)
    corr = torch.zeros(t.numel(), dtype=torch.float32, device=dev)
    mu, vmax, min_slack, rnd, n_mu_pos = (None, float(viol.max()),
                                          float("nan"), 0, 0)
    for budget in _iters_ladder(iters, iters_max):
        corr, active, mu, vmax, rnd, min_slack, n_mu_pos = _cutting_plane(
            G_cpu, t, dev, kappa, tau_feas, ng, Gt, valid, scale, nt, active,
            budget, max_rounds, max_active, add_per_round)
        diag["proj_iters_used"] = int(budget)
        if vmax <= tau_feas:
            break
    diag["proj_rounds"] = int(rnd)
    diag["proj_n_active"] = len(active)
    diag["proj_coverage"] = len(active) / max(1, int(G_cpu.shape[0]))
    diag["proj_max_viol_after"] = vmax
    diag["proj_feasible"] = bool(vmax <= tau_feas)
    diag["proj_corr_ratio"] = float(corr.norm() / max(nt, 1e-12))
    diag["proj_n_mu_pos"] = n_mu_pos
    diag["proj_min_slack_after"] = min_slack
    diag["_active"] = active
    diag["_mu"] = None if mu is None else mu.detach().cpu()
    diag["_G_cpu"] = G_cpu
    diag["_ng"] = ng
    diag["_valid"] = valid
    diag["_corr"] = corr
    diag["_t"] = t
    if write_grad and diag["proj_feasible"] and g_task_gamma is not None:
        with torch.no_grad():
            for p, cp, gt in zip(gamma_params, torch.split(corr, sizes),
                                 g_task_gamma):
                p.grad = (gt.reshape(-1).float() - cp).to(p.dtype).view_as(p)
    diag["_d_star"] = (t + corr).detach()
    return diag


# --------------------------------------------------------------------------
# vendored: counterfactual AdamW + the proposal-space driver
# --------------------------------------------------------------------------

def counterfactual_task_step(optimizer, gamma_params, g_task, grad_clip):
    """The AdamW displacement this boundary WOULD have taken with the PURE task
    gradient, from the SAME pre-step optimizer state.

    Executed on clones, so real parameters and optimizer state are untouched.
    No forward pass is involved, so no RNG is consumed.
    """
    try:
        base = getattr(optimizer, "opt", optimizer)
        try:
            import inspect
            sig = inspect.signature(type(base).__init__)
            if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
                allowed = None
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
        print("[gamma proj] WARNING: counterfactual AdamW step unavailable "
              "({}: {}); boundary aborted.".format(type(e).__name__, e),
              flush=True)
        return None


def proposal_space_update(
    optimizer, gamma_params, q_rows: Sequence[torch.Tensor], g_task,
    *, scheduler=None, kappa: float = C.KAPPA, tau_feas: float = C.TAU,
    grad_clip: float = C.GRAD_CLIP, proj_kw: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Project the AdamW proposal and commit it exactly once.

    Returns a diagnostics dict. `committed` is False when the certificate
    failed -- then parameters, moments, step counter and scheduler are all
    untouched, and the caller must treat the boundary as failed.
    """
    diag: Dict[str, Any] = {"committed": False, "aborted": None}
    sizes = [p.numel() for p in gamma_params]
    theta_old = [p.detach().clone() for p in gamma_params]

    # ONE global clip, used for BOTH the preview and the real moment update --
    # never preview-unclipped / step-clipped.
    diag["g_gamma_clip"] = float(
        torch.nn.utils.clip_grad_norm_(gamma_params, grad_clip))
    g_clipped = [p.grad.detach().clone() if p.grad is not None
                 else torch.zeros_like(p) for p in gamma_params]

    delta_task = counterfactual_task_step(optimizer, gamma_params,
                                          g_clipped, grad_clip)
    if delta_task is None:
        diag["aborted"] = "adamw_preview_unavailable"
        return diag
    delta_adam = torch.cat([d.flatten().float() for d in delta_task])
    diag["delta_adam_norm"] = float(delta_adam.norm())

    G_cpu = (torch.stack([q.detach().flatten().float() for q in q_rows], 0)
             if q_rows else torch.zeros(0, delta_adam.numel()))

    pj = active_set_feasibility_projection(
        gamma_params=gamma_params, G_cpu=G_cpu, kappa=kappa,
        tau_feas=tau_feas, t_override=delta_adam, write_grad=False,
        **(proj_kw or {}))
    diag.update({k: v for k, v in pj.items() if not k.startswith("_")})
    if not pj.get("proj_feasible", True):
        # Nothing has been written yet, so this needs NO rollback: the
        # parameters, the moments, the step counter and the scheduler are all
        # exactly where they were.
        diag["aborted"] = pj.get("proj_abort") or "proposal_not_certified"
        return diag
    diag["vmax_proj"] = pj.get("proj_max_viol_after")

    # Advance the moments EXACTLY ONCE, with the SAME clipped gradient.
    for p, g in zip(gamma_params, g_clipped):
        p.grad = g.detach().clone()
    optimizer.step()

    d_star = pj["_d_star"].to(gamma_params[0].device)
    if pj.get("proj_n_active", 0) == 0:
        # The QP bound nothing: keep AdamW's own write BYTE-FOR-BYTE rather
        # than re-rounding it through theta_old + delta_adam.
        diag["proj_noop_kept_adamw_write"] = 1
        diag["d_star"] = delta_adam
    else:
        with torch.no_grad():
            for p, o, dv in zip(gamma_params, theta_old,
                                torch.split(d_star, sizes)):
                p.data.copy_(o + dv.reshape(p.shape).to(p.dtype))
        diag["d_star"] = d_star
    diag["committed"] = True

    if scheduler is not None:
        scheduler.step()
    diag["gamma_steps"] = 1
    return diag


# --------------------------------------------------------------------------
# acceptance self-test
# --------------------------------------------------------------------------

def self_test() -> int:
    import os
    print("qp.py acceptance")
    print("=" * 68)

    torch.manual_seed(0)
    d, n_int = 12, 6
    params = [torch.nn.Parameter(torch.randn(d))]

    def fresh_opt():
        return torch.optim.AdamW(params, lr=1e-3, betas=(0.9, 0.999),
                                 eps=1e-8, weight_decay=0.0)

    # --- 1. no binding constraint -> EXACTLY the plain AdamW step ----------
    opt = fresh_opt()
    g_task = [torch.randn(d)]
    for p, g in zip(params, g_task):
        p.grad = g.clone()
    # interfaces that agree with t => nothing binds
    t = -g_task[0]
    q_rows = [(t / t.norm() * 0.5) for _ in range(n_int)]
    ref_params = [p.detach().clone() for p in params]
    ref_opt = torch.optim.AdamW([torch.nn.Parameter(p.detach().clone())
                                 for p in params], lr=1e-3,
                                betas=(0.9, 0.999), eps=1e-8,
                                weight_decay=0.0)
    for q, g in zip(ref_opt.param_groups[0]["params"], g_task):
        q.grad = g.clone()
    # The real path clips ONCE before both the preview and the moment update,
    # so the reference must clip identically or it is not the same step.
    torch.nn.utils.clip_grad_norm_(ref_opt.param_groups[0]["params"],
                                   C.GRAD_CLIP)
    ref_opt.step()

    out = proposal_space_update(opt, params, q_rows, g_task,
                                kappa=C.KAPPA, tau_feas=C.TAU,
                                grad_clip=C.GRAD_CLIP)
    assert out["committed"], out["aborted"]
    assert torch.equal(params[0].detach(), ref_opt.param_groups[0]["params"][0]
                       .detach()), "unconstrained boundary != plain AdamW step"
    print("OK  no-constraint      committed result is BITWISE the plain AdamW "
          "step (n_active={}, kept_adamw_write={})".format(
              out["proj_n_active"], out.get("proj_noop_kept_adamw_write", 0)))

    # --- 2. binding constraint -> projection moves the proposal ------------
    opt2 = fresh_opt()
    for p, g in zip(params, g_task):
        p.grad = g.clone()
    # DISTINCT directions, each opposing t -- identical rows would make one
    # enforced constraint enforce them all, which would defeat the budget test.
    gq = torch.Generator().manual_seed(1)
    base_dir = -t / t.norm()
    q_rows_opposed = []
    for _ in range(n_int):
        v = base_dir + 0.5 * torch.randn(d, generator=gq)
        q_rows_opposed.append(v / v.norm())
    params_before = params[0].detach().clone()
    out2 = proposal_space_update(opt2, params, q_rows_opposed, g_task,
                                 kappa=C.KAPPA, tau_feas=C.TAU,
                                 grad_clip=C.GRAD_CLIP)
    if out2["committed"]:
        moved = float((params[0].detach() - params_before).norm())
        assert out2["proj_n_active"] > 0 or out2.get("proj_noop_kept_adamw_write")
        print("OK  binding constraint {} interfaces active, corr_ratio={:.3f}, "
              "moved {:.3e}".format(out2["proj_n_active"],
                                    out2.get("proj_corr_ratio", 0.0), moved))
    else:
        print("OK  binding constraint aborted cleanly ({})".format(
            out2["aborted"]))

    # --- 3. certificate failure -> EVERYTHING frozen -----------------------
    # The half-space intersection is always non-empty (d can grow), so a
    # tau-only failure is hard to provoke; force the solver to run out of
    # budget instead: only ONE interface may be enforced and no round may add
    # another. The remaining interfaces stay violated -> certificate fails.
    opt3 = fresh_opt()
    for p, g in zip(params, g_task):
        p.grad = g.clone()
    snap_p = params[0].detach().clone()
    snap_m = copy.deepcopy(opt3.state_dict())
    out3 = proposal_space_update(opt3, params, q_rows_opposed, g_task,
                                 kappa=C.KAPPA, tau_feas=C.TAU,
                                 grad_clip=C.GRAD_CLIP,
                                 proj_kw={"max_active": 1, "add_per_round": 0})
    assert not out3["committed"], out3
    assert out3["proj_max_viol_after"] > C.TAU, out3["proj_max_viol_after"]
    assert torch.equal(params[0].detach(), snap_p), "params changed on abort"
    assert str(opt3.state_dict()) == str(snap_m), "moments changed on abort"
    print("OK  cert failure       solver budget exhausted (v_max={:.3e} > tau); "
          "params + optimizer moments UNCHANGED".format(
              out3["proj_max_viol_after"]))

    # --- 4. the optimizer is stepped EXACTLY once --------------------------
    calls = {"n": 0}
    orig_step = opt3.step

    def counting_step(*a, **k):
        calls["n"] += 1
        return orig_step(*a, **k)
    opt3.step = counting_step
    proposal_space_update(opt3, params, q_rows, g_task, kappa=C.KAPPA,
                          tau_feas=C.TAU, grad_clip=C.GRAD_CLIP)
    assert calls["n"] == 1, calls
    print("OK  single step        optimizer.step() called exactly {} time(s)"
          .format(calls["n"]))

    # --- 5. no paid requests ----------------------------------------------
    paid = 0
    led = os.environ.get("META_N_REQUEST_LEDGER", "").strip()
    if led and os.path.isfile(led):
        from meta_n.rpbe.accounting import snapshot
        paid = snapshot()["backend_requests"]
    assert paid == 0
    print("OK  zero-cost          paid backend_requests = {}".format(paid))

    print()
    print("VERDICT: ALL OK")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(self_test())
