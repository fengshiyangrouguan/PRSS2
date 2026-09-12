"""test_projection.py — interface-wise RPBE feasibility projection (TGN port).

CPU-only.  Run:
  PYTHONPATH=src python -m pytest src/rpbe_embodied/test_projection.py -q
  (or plain `python src/rpbe_embodied/test_projection.py`)

Covers the properties the Stage8 migration relies on:
  1. no interfaces            -> Gamma keeps the pure task gradient (no-op);
  2. feasible + REAL KKT      -> every interface satisfies g_i.d >= b_i and
                                 mu_j (g_i.d - b_i) == 0 for the active set;
  3. opposite interfaces      -> g_2 = -g_1 (aggregate == 0) is still repaired,
                                 which the retired sum-then-project path could
                                 not do;
  4. all interfaces checked   -> 1000+ rows, only a small active set, coverage
                                 reported, row order does not matter;
  5. non-finite rows          -> dropped from Q, but the count is reported so
                                 the BOUNDARY fails closed (never silently
                                 enforces one constraint fewer);
  6. uncertified projection   -> the boundary is ABORTED (0 Gamma steps);
  7. kappa >= 1               -> never binds, identical to the task control
                                 (one step, same parameters);
  8. production vs oracle     -> the active-set d matches an INDEPENDENT
                                 full-QP oracle defined in this file;
  9. migration                -> the Gamma scheduler is realigned to
                                 param_version, not left at warmup step 0.
"""
import math

import torch
import torch.nn as nn

from rpbe_embodied.boundary import apply_gamma_boundary_update
from rpbe_embodied.loss import (_fista_nonneg, _iters_ladder, _stack_to,
                                active_set_feasibility_projection,
                                interface_influence_rows,
                                reset_rows_backend_stats, rows_backend_stats)
from rpbe_embodied.resume import (BOUNDARY_CONFIG_KEYS, COMMON_CONFIG_KEYS,
                                  boundary_config, realign_lambda_scheduler,
                                  verify_resume_config)

D = 6
torch.manual_seed(0)


class TinyGamma(nn.Module):
    """Non-linear per-interface merge, tiny enough to reason about."""

    def __init__(self, dim=D):
        super().__init__()
        self.w = nn.Parameter(torch.randn(dim, dim) * 0.2)
        self.u = nn.Parameter(torch.randn(dim) * 0.1)

    def forward(self, a, b):
        m = 0.5 * (a + b)
        h = torch.tanh(a @ self.w) * torch.tanh(b @ self.w)
        return m + h + self.u


def _rows(gamma, cot, inp):
    keys = list(cot.keys())
    G, used, n_bad = interface_influence_rows(
        gamma, cot, inp, keys, params=list(gamma.parameters()), device="cpu")
    return G, used, n_bad


def _full_qp_reference(g_task_gamma, G, kappa, iters=800, tau_feas=1e-6):
    """Independent small-scale oracle: ALL interfaces enter ONE full QP.

    Pure function -- returns (d, mu, diag) and never touches p.grad, so it
    cannot mask a bug in the production active-set solver.
    """
    t = torch.cat([-x.flatten().float() for x in g_task_gamma])
    nt = float(t.norm())
    ng = G.norm(dim=1)
    Gt = G @ t
    b = -kappa * ng * nt
    mu = _fista_nonneg(G @ G.t(), b - Gt, iters)
    d = t + G.t() @ mu
    v = torch.clamp(-(G @ d - b), min=0.0) / (ng * nt + 1e-30)
    return d, mu, {"vmax": float(v.max()), "min_slack": float((G @ d - b).min()),
                   "feasible": float(v.max()) <= tau_feas}


def _flat_task_grad(gamma, flat):
    out, i = [], 0
    for p in gamma.parameters():
        out.append(flat[i:i + p.numel()].reshape(p.shape).clone())
        i += p.numel()
    return out


def _set_task_grad(gamma, g_task):
    for p, g in zip(gamma.parameters(), g_task):
        p.grad = g.clone()


def _adversarial(gamma, G, scale=0.5):
    """g_task aligned with the SUM of the rows -> many constraints bind."""
    return _flat_task_grad(gamma, scale * G.sum(0))


def _d_from_grad(gamma):
    """d = -p.grad (since p.grad = g_task - corr and d = t + corr)."""
    return -torch.cat([p.grad.flatten() for p in gamma.parameters()])


def _rand_problem(gamma, n):
    cot = {i: torch.randn(D) for i in range(n)}
    inp = {i: (torch.randn(D), torch.randn(D)) for i in range(n)}
    return cot, inp


def test_fista_matches_slow_pgd():
    n = 12
    A = torch.randn(n, n)
    Q = A @ A.t() + 0.1 * torch.eye(n)
    c = torch.randn(n)
    mu = _fista_nonneg(Q, c, 2000)
    assert mu.min() >= 0
    L = float(torch.linalg.eigvalsh(Q).max())
    m = torch.zeros(n)
    for _ in range(20000):
        m = torch.clamp(m - (Q @ m - c) / L, min=0.0)
    assert torch.allclose(mu, m, atol=1e-4), (mu - m).abs().max().item()
    print("test_fista_matches_slow_pgd OK  gap=%.2e" % (mu - m).abs().max())


def test_no_interfaces_is_pure_task():
    gamma = TinyGamma()
    g_task = [torch.randn_like(p) for p in gamma.parameters()]
    _set_task_grad(gamma, g_task)
    diag = active_set_feasibility_projection(
        [g.clone() for g in g_task], list(gamma.parameters()), None, 0.05)
    for p, g in zip(gamma.parameters(), g_task):
        assert torch.equal(p.grad, g), "empty G must leave the task grad"
    assert diag["proj_feasible"] and diag["proj_n_active"] == 0
    print("test_no_interfaces_is_pure_task OK")


def test_real_kkt_complementarity():
    gamma = TinyGamma()
    cot, inp = _rand_problem(gamma, 20)
    G, _, _ = _rows(gamma, cot, inp)
    g_task = _adversarial(gamma, G)
    _set_task_grad(gamma, g_task)
    kappa, tau = 0.1, 1e-4
    diag = active_set_feasibility_projection(
        [g.clone() for g in g_task], list(gamma.parameters()), G, kappa,
        tau_feas=tau)
    assert diag["proj_feasible"]
    assert diag["proj_n_candidates"] >= 1
    t = torch.cat([-g.flatten() for g in g_task])
    d = _d_from_grad(gamma)
    ng = G.norm(dim=1)
    b = -kappa * ng * float(t.norm())
    slack = G @ d - b
    # primal feasibility on EVERY interface, not just the active set
    v = torch.clamp(-slack, min=0.0) / (ng * float(t.norm()) + 1e-30)
    assert float(v.max()) <= tau + 1e-9, float(v.max())
    # KKT complementarity: active constraints tight, inactive slack >= 0
    act, mu = diag["_active"], diag["_mu"]
    assert mu is not None and len(mu) == len(act)
    for j, i in enumerate(act):
        if float(mu[j]) > 1e-6:
            assert abs(float(slack[i])) < 1e-3, (i, float(slack[i]))
        assert float(slack[i]) > -1e-3
    # inactive rows must have mu == 0: |active| == number of positive mu
    assert int((mu > 1e-8).sum()) == diag["proj_n_mu_pos"]
    assert len(act) <= G.shape[0] and 0.0 < diag["proj_coverage"] <= 1.0
    print("test_real_kkt_complementarity OK  vmax=%.2e active=%d/%d"
          % (float(v.max()), len(act), diag["proj_n_candidates"]))


def test_opposite_interfaces_still_repaired():
    """g_2 = -g_1: the AGGREGATE gradient is zero, so a sum-then-project path
    would see no violation and do nothing.  The per-interface QP must repair."""
    gamma = TinyGamma()
    a1 = (torch.randn(D), torch.randn(D))
    cot = {0: torch.randn(D), 1: torch.randn(D)}
    inp = {0: a1, 1: a1}
    G, _, _ = _rows(gamma, cot, inp)
    G = torch.stack([G[0], -G[0]])
    assert float(G.sum(0).norm()) < 1e-6, "test setup: aggregate must cancel"
    g_task = _flat_task_grad(gamma, 0.5 * G[0])
    _set_task_grad(gamma, g_task)
    diag = active_set_feasibility_projection(
        [g.clone() for g in g_task], list(gamma.parameters()), G, kappa=0.0)
    assert diag["proj_feasible"]
    assert diag["proj_corr_ratio"] > 1e-6, "opposite pair not repaired"
    slack = G @ _d_from_grad(gamma) - torch.zeros(2)
    assert float(slack.min()) > -1e-3, float(slack.min())
    print("test_opposite_interfaces_still_repaired OK  corr=%.3f"
          % diag["proj_corr_ratio"])


def test_all_interfaces_checked_and_order_invariant():
    gamma = TinyGamma()
    n = 1200
    cot, inp = _rand_problem(gamma, n)
    G, used, _ = _rows(gamma, cot, inp)
    assert G.shape[0] == n and len(used) == n, "no interface may be dropped"
    g_task = _adversarial(gamma, G, scale=0.05)
    _set_task_grad(gamma, g_task)
    diag = active_set_feasibility_projection(
        [g.clone() for g in g_task], list(gamma.parameters()), G, kappa=0.05)
    assert diag["proj_feasible"]
    assert diag["proj_n_checked"] == n
    assert diag["proj_n_active"] <= n
    assert diag["proj_n_active"] < n, "active set should be a small subset"
    assert 0.0 < diag["proj_coverage"] <= 1.0
    d1 = _d_from_grad(gamma).clone()
    # permutation of the row order must not change the projected direction
    perm = torch.randperm(n)
    _set_task_grad(gamma, g_task)
    diag2 = active_set_feasibility_projection(
        [g.clone() for g in g_task], list(gamma.parameters()), G[perm],
        kappa=0.05)
    assert diag2["proj_feasible"]
    d2 = _d_from_grad(gamma)
    rel = float((d1 - d2).norm() / max(float(d1.norm()), 1e-12))
    assert rel < 1e-4, f"row order changed the projection: rel={rel}"
    print("test_all_interfaces_checked_and_order_invariant OK  "
          "n=%d active=%d coverage=%.3f" % (n, diag["proj_n_active"],
                                            diag["proj_coverage"]))


def test_batched_rows_match_per_interface_vjp():
    """Correctness of the backend switch: the batched torch.func rows must
    equal an INDEPENDENT per-interface VJP, row by row."""
    gamma = TinyGamma()
    cot, inp = _rand_problem(gamma, 7)
    G, used, n_bad = _rows(gamma, cot, inp)
    assert n_bad == 0 and len(used) == 7
    for i in range(7):
        a, b = inp[i]
        z = gamma(a.unsqueeze(0), b.unsqueeze(0))[0]
        ref = torch.cat([g.reshape(-1) for g in torch.autograd.grad(
            (cot[i] * z.float()).sum(), list(gamma.parameters()))])
        assert torch.allclose(G[i], ref, atol=1e-5), \
            (i, float((G[i] - ref).abs().max()))
    print("test_batched_rows_match_per_interface_vjp OK")


def test_rows_are_detached_constants():
    """The QP treats rows as constants: carrying a graph would retain it
    across every cutting-plane round."""
    gamma = TinyGamma()
    cot, inp = _rand_problem(gamma, 6)
    G, used, _ = _rows(gamma, cot, inp)
    assert not G.requires_grad, "influence rows must be detached"
    diag = active_set_feasibility_projection(
        [torch.randn_like(p) for p in gamma.parameters()],
        list(gamma.parameters()), G, 0.05)
    assert diag["proj_feasible"] is not None
    print("test_rows_are_detached_constants OK")


def test_stack_to_handles_mixed_inputs():
    """Replay inputs are NOT uniform: the window's fixed-trace rebuild returns
    a CUDA merged state for a merged child and the raw CPU leaf for a leaf
    child, and leaves are stored fp32.  ``torch.stack`` demands a uniform
    dtype AND device, so every element must be cast first -- this is the bug
    the first Stage8 smoke run hit."""
    a = torch.zeros(3, dtype=torch.float32)
    b = torch.zeros(3, dtype=torch.bfloat16)
    out = _stack_to([a, b], "cpu", torch.float32)
    assert out.dtype == torch.float32 and out.shape == (2, 3)
    if torch.cuda.is_available():
        mixed = [torch.zeros(3), torch.zeros(3, device="cuda")]
        out2 = _stack_to(mixed, "cuda", torch.float32)
        assert out2.device.type == "cuda" and out2.shape == (2, 3)
    # and the boundary must survive mixed-dtype pairs end to end
    g, tp, tc, rp, rc = _boundary_fixture()
    mixed_tp = [(p[0].to(torch.float32), p[1].to(torch.float32))
                for p in tp]
    mixed_tp[::2] = [(p[0].to(torch.bfloat16), p[1].to(torch.bfloat16))
                     for p in mixed_tp[::2]]
    opt, diag, _ = _run_boundary(g, mixed_tp, tc, rp, rc, kappa=0.05)
    assert diag["gamma_steps"] == 1 and opt.steps == 1
    print("test_stack_to_handles_mixed_inputs OK")


def test_iters_ladder():
    assert _iters_ladder(400, 1600) == [400, 800, 1600]
    assert _iters_ladder(400, 400) == [400]
    assert _iters_ladder(400, 0) == [400]
    assert _iters_ladder(300, 1000) == [300, 600, 1000]
    print("test_iters_ladder OK")


def _ill_conditioned_problem(gamma, n=60, spread=4.0):
    """Rows whose norms span `spread` orders of magnitude -- exactly what real
    bf16 Gamma gradients do, and what made the RAW dual Gram (cond ~1e12)
    un-solvable by FISTA."""
    P = sum(p.numel() for p in gamma.parameters())
    G = (torch.randn(n, P)
         * torch.logspace(-spread / 2, spread / 2, n)[:, None])
    t = torch.randn(P) * 1e-2
    return G, t


def test_ill_conditioned_rows_certify_at_the_base_budget():
    """The row-normalised QP must certify an ill-conditioned problem at the
    FIRST ladder rung; before the fix this aborted (FISTA stalled)."""
    gamma = TinyGamma()
    G, t = _ill_conditioned_problem(gamma)
    g_task = _flat_task_grad(gamma, t)
    _set_task_grad(gamma, g_task)
    kappa = 0.02
    diag = active_set_feasibility_projection(
        [g.clone() for g in g_task], list(gamma.parameters()), G, kappa,
        iters=400, iters_max=1600, tau_feas=1e-3)
    assert diag["proj_feasible"], diag
    assert diag["proj_iters_used"] == 400, diag["proj_iters_used"]
    assert diag["proj_max_viol_after"] <= 1e-3
    # and the returned d must satisfy the ORIGINAL (un-normalised) half-spaces
    d = _d_from_grad(gamma)
    ng = G.norm(dim=1)
    b = -kappa * ng * float(t.norm())
    v = torch.clamp(-(G @ d - b), min=0.0) / (ng * float(t.norm()) + 1e-30)
    assert float(v.max()) <= 1e-3, float(v.max())
    print("test_ill_conditioned_rows_certify_at_the_base_budget OK  "
          "vmax=%.2e iters=%d" % (diag["proj_max_viol_after"],
                                  diag["proj_iters_used"]))


def test_row_normalisation_preserves_the_optimum():
    """Dividing a constraint by the positive ||g_i|| does not change the
    feasible set, so d* is unchanged.  Checked on a WELL-conditioned problem,
    where the raw formulation also converges -- so the comparison is
    meaningful (on an ill-conditioned one the raw dual never converges at
    all, which is the defect the normalisation fixes)."""
    gamma = TinyGamma()
    P = sum(p.numel() for p in gamma.parameters())
    G = torch.randn(10, P)                      # norms O(1): well conditioned
    t = torch.randn(P) * 1e-2
    # the driver's task direction is t_driver = -g_task, so pass -t
    g_task = _flat_task_grad(gamma, -t)
    _set_task_grad(gamma, g_task)
    kappa = 0.02
    ng = G.norm(dim=1)
    nt = float(t.norm())
    b = -kappa * ng * nt
    # reference: RAW formulation, huge budget (converges here)
    mu = _fista_nonneg(G @ G.t(), b - G @ t, 40000)
    d_raw = t + G.t() @ mu
    assert float((torch.clamp(-(G @ d_raw - b), min=0)
                  / (ng * nt)).max()) < 1e-6, "raw reference did not converge"
    # production: normalised QP, driven to the solver's practical floor
    # (fp32 FISTA bottoms out around 1e-8, so tau must stay above that)
    active_set_feasibility_projection(
        [g.clone() for g in g_task], list(gamma.parameters()), G, kappa,
        iters=40000, iters_max=40000, tau_feas=1e-7)
    d_prod = _d_from_grad(gamma)
    rel = float((d_prod - d_raw).norm() / d_raw.norm())
    assert rel < 1e-3, f"normalisation changed d*: rel={rel}"
    print("test_row_normalisation_preserves_the_optimum OK  rel=%.2e" % rel)


def test_solver_budget_escalates_instead_of_aborting():
    """A boundary that cannot be certified at the base rung must escalate."""
    gamma = TinyGamma()
    G, t = _ill_conditioned_problem(gamma, n=50)
    g_task = _flat_task_grad(gamma, t)
    _set_task_grad(gamma, g_task)
    diag = active_set_feasibility_projection(
        [g.clone() for g in g_task], list(gamma.parameters()), G, 0.02,
        iters=5, iters_max=4000, tau_feas=1e-4)
    assert diag["proj_feasible"], diag
    assert diag["proj_iters_used"] > 5, diag["proj_iters_used"]
    print("test_solver_budget_escalates_instead_of_aborting OK  "
          "iters_used=%d" % diag["proj_iters_used"])


def test_nonfinite_rows_dropped():
    gamma = TinyGamma()
    cot, inp = _rand_problem(gamma, 8)
    G, _, _ = _rows(gamma, cot, inp)
    G[2] = float("inf")
    G[5, 1] = float("nan")
    g_task = _adversarial(gamma, torch.nan_to_num(G, posinf=0.0, neginf=0.0))
    _set_task_grad(gamma, g_task)
    diag = active_set_feasibility_projection(
        [g.clone() for g in g_task], list(gamma.parameters()), G, 0.05)
    assert diag["proj_n_interfaces"] == 8
    assert diag["proj_n_valid"] == 6, diag["proj_n_valid"]
    assert diag["proj_feasible"]
    gp = torch.cat([p.grad.flatten() for p in gamma.parameters()])
    assert torch.isfinite(gp).all(), "Q was poisoned by a non-finite row"
    print("test_nonfinite_rows_dropped OK  valid=%d/8" % diag["proj_n_valid"])


def test_infeasible_projection_is_dropped_by_row_filter():
    """A gamma with non-finite weights yields no usable row at all, and the
    count is reported so the caller can FAIL CLOSED."""
    gamma = TinyGamma()
    with torch.no_grad():
        gamma.w[0, 0] = float("nan")
    cot, inp = _rand_problem(gamma, 5)
    G, used, n_bad = _rows(gamma, cot, inp)
    assert G is None, "non-finite rows must not be returned as a G matrix"
    assert n_bad == 5 and used == [], (n_bad, used)
    print("test_infeasible_projection_is_dropped_by_row_filter OK")


def test_rows_backend_is_the_batched_vmap_path():
    """The chunked VJP must use the batched torch.func path, not silently
    degrade to the per-interface loop."""
    reset_rows_backend_stats()
    gamma = TinyGamma()
    cot, inp = _rand_problem(gamma, 40)
    G, used, n_bad = _rows(gamma, cot, inp)
    st = rows_backend_stats()
    assert G.shape[0] == 40 and n_bad == 0
    assert st["fallback_chunks"] == 0, st
    assert st["batched_chunks"] >= 1, st
    print("test_rows_backend_is_the_batched_vmap_path OK  %s" % st)


def test_kappa_ge_one_never_binds():
    gamma = TinyGamma()
    cot, inp = _rand_problem(gamma, 7)
    G, _, _ = _rows(gamma, cot, inp)
    g_task = _adversarial(gamma, G)
    _set_task_grad(gamma, g_task)
    diag = active_set_feasibility_projection(
        [g.clone() for g in g_task], list(gamma.parameters()), G, kappa=1.0)
    assert diag["proj_n_active_init"] == 0
    assert diag["proj_n_active"] == 0
    assert diag["proj_corr_ratio"] == 0.0
    for p, g in zip(gamma.parameters(), g_task):
        assert torch.equal(p.grad, g), "kappa>=1 must be a pure task step"
    print("test_kappa_ge_one_never_binds OK")


def test_round0_admits_only_tolerance_breaches():
    """Round 0 must admit interfaces with v_i > tau_feas, not v_i > 0:
    interfaces already inside the tolerance would otherwise burn active-set
    slots and could push a later real violator out of the budget."""
    gamma = TinyGamma()
    cot, inp = _rand_problem(gamma, 15)
    G, _, _ = _rows(gamma, cot, inp)
    g_task = _adversarial(gamma, G)
    kappa = 0.1
    t = torch.cat([-g.flatten() for g in g_task])
    ng = G.norm(dim=1)
    v = torch.clamp(-(G @ t) / (ng * float(t.norm())) - kappa, min=0.0)
    vmax = float(v.max())
    # tau just above every violation -> nothing breaches -> pure task step
    _set_task_grad(gamma, g_task)
    diag = active_set_feasibility_projection(
        [g.clone() for g in g_task], list(gamma.parameters()), G, kappa,
        tau_feas=vmax + 1e-6)
    assert diag["proj_feasible"]
    assert diag["proj_n_candidates"] == 0, diag["proj_n_candidates"]
    assert diag["proj_n_active"] == 0
    assert diag["proj_corr_ratio"] == 0.0
    for p, g in zip(gamma.parameters(), g_task):
        assert torch.equal(p.grad, g), "no breach must mean a pure task step"
    # tau just below the worst violation -> that interface IS admitted
    _set_task_grad(gamma, g_task)
    diag2 = active_set_feasibility_projection(
        [g.clone() for g in g_task], list(gamma.parameters()), G, kappa,
        tau_feas=max(vmax - 1e-3, 1e-9))
    assert diag2["proj_n_candidates"] >= 1
    assert diag2["proj_n_active"] >= 1
    print("test_round0_admits_only_tolerance_breaches OK  vmax=%.3e" % vmax)


def test_amp_scale_invariance():
    """Scaling every gradient by S scales d by S (mu unchanged), because the
    half-space is scale-homogeneous.  The projection is therefore invariant to
    the GradScaler factor and AMP can never change the projected direction."""
    gamma = TinyGamma()
    cot, inp = _rand_problem(gamma, 12)
    G, _, _ = _rows(gamma, cot, inp)
    g_task = _adversarial(gamma, G)
    _set_task_grad(gamma, g_task)
    active_set_feasibility_projection(
        [g.clone() for g in g_task], list(gamma.parameters()), G, 0.1)
    grad0 = torch.cat([p.grad.flatten() for p in gamma.parameters()]).clone()
    S = 512.0
    scaled = [g * S for g in g_task]
    _set_task_grad(gamma, scaled)
    active_set_feasibility_projection(
        [g.clone() for g in scaled], list(gamma.parameters()), G * S, 0.1)
    gradS = torch.cat([p.grad.flatten() for p in gamma.parameters()])
    assert torch.allclose(gradS, grad0 * S, rtol=1e-3, atol=1e-3), \
        float((gradS - grad0 * S).abs().max())
    print("test_amp_scale_invariance OK  S=%.0f" % S)


# --- boundary integration ---------------------------------------------------

class _CountingOpt:
    def __init__(self, opt):
        self.opt = opt
        self.steps = 0

    def zero_grad(self, *a, **k):
        return self.opt.zero_grad(*a, **k)

    def step(self, *a, **k):
        self.steps += 1
        return self.opt.step(*a, **k)


def _boundary_fixture(n_task=30, n_rpbe=40):
    gamma = TinyGamma()
    cot, inp = _rand_problem(gamma, max(n_task, n_rpbe) + 1)
    task_pairs = [inp[i] for i in range(n_task)]
    task_cots = [cot[i] for i in range(n_task)]
    rpbe_pairs = [inp[i] for i in range(n_rpbe)]
    rpbe_cots = [cot[i] for i in range(n_rpbe)]
    return gamma, task_pairs, task_cots, rpbe_pairs, rpbe_cots


def _run_boundary(gamma, task_pairs, task_cots, rpbe_pairs, rpbe_cots, **kw):
    opt = _CountingOpt(torch.optim.AdamW(list(gamma.parameters()), lr=1e-3))
    before = [p.detach().clone() for p in gamma.parameters()]
    diag = apply_gamma_boundary_update(
        gamma=gamma, gamma_params=list(gamma.parameters()), optimizer=opt,
        scheduler=None, task_pairs=task_pairs, task_cotangents=task_cots,
        rpbe_pairs=rpbe_pairs, rpbe_cotangents=rpbe_cots, device="cpu", **kw)
    moved = sum(float((p.detach() - b0).pow(2).sum())
                for p, b0 in zip(gamma.parameters(), before)) ** 0.5
    return opt, diag, moved


def test_boundary_takes_exactly_one_step_per_arm():
    # task control: no rpbe rows
    g1, tp, tc, rp, rc = _boundary_fixture()
    opt1, d1, _ = _run_boundary(g1, tp, tc, [], [])
    assert opt1.steps == 1 and d1["gamma_steps"] == 1
    assert not d1["gamma_aborted"]
    # projected arm: same task data + rpbe constraints
    g2, tp, tc, rp, rc = _boundary_fixture()
    opt2, d2, _ = _run_boundary(g2, tp, tc, rp, rc, kappa=0.05)
    assert opt2.steps == 1 and d2["gamma_steps"] == 1
    assert d2["proj_n_interfaces"] == len(rp)
    assert d2["proj_n_checked"] == len(rp)
    print("test_boundary_takes_exactly_one_step_per_arm OK  "
          "checked=%d active=%d" % (d2["proj_n_checked"], d2["proj_n_active"]))


def test_boundary_kappa_one_equals_task_control():
    """kappa >= 1 never binds, so the projected arm must reproduce the task
    control bit-for-bit (same init, same data, fresh optimizer)."""
    import copy
    g1, tp, tc, rp, rc = _boundary_fixture()
    g2 = copy.deepcopy(g1)
    _run_boundary(g1, tp, tc, [], [])
    _run_boundary(g2, tp, tc, rp, rc, kappa=1.0)
    for p1, p2 in zip(g1.parameters(), g2.parameters()):
        assert torch.allclose(p1, p2, atol=0, rtol=0), "kappa=1 changed the step"
    print("test_boundary_kappa_one_equals_task_control OK")


def test_infeasible_boundary_aborts_without_stepping():
    g1, tp, tc, rp, rc = _boundary_fixture()
    opt, diag, moved = _run_boundary(
        g1, tp, tc, rp, rc, kappa=0.0, tau_feas=1e-12, max_rounds=1,
        max_active=1, add_per_round=0, proj_iters_max=400)
    assert not diag["proj_feasible"]
    assert diag["gamma_aborted"] and diag["gamma_steps"] == 0
    assert diag["gamma_abort_reason"] in ("active_budget_exhausted",
                                          "projection_not_certified")
    assert opt.steps == 0, "an uncertified projection must not step"
    assert moved == 0.0, "parameters moved on an aborted boundary"
    print("test_infeasible_boundary_aborts_without_stepping OK  "
          "reason=%s vmax=%.2e" % (diag["gamma_abort_reason"],
                                   diag["proj_max_viol_after"]))


def test_nonfinite_constraint_row_fails_closed():
    """A non-finite interface gradient must ABORT the boundary, not silently
    enforce one constraint fewer.  (Poisoning Gamma makes BOTH the task
    gradient and the constraint rows non-finite, so either guard may fire
    first; both are fail-closed and both give 0 steps.)"""
    g1, tp, tc, rp, rc = _boundary_fixture()
    g1.w.data[0, 0] = float("nan")       # poisons the task grad AND every g_i
    opt, diag, moved = _run_boundary(g1, tp, tc, rp, rc, kappa=0.05)
    assert diag["gamma_aborted"] and diag["gamma_steps"] == 0
    assert diag["gamma_abort_reason"] in ("nonfinite_constraint_row",
                                          "nonfinite_task_gradient"), diag
    assert opt.steps == 0, "a NaN boundary must not step"
    print("test_nonfinite_constraint_row_fails_closed OK  reason=%s"
          % diag["gamma_abort_reason"])


def test_nonfinite_task_direction_is_refused():
    """A non-finite task direction can never be feasibly projected: the driver
    must refuse to write a gradient even when the rows are fine."""
    gamma = TinyGamma()
    cot, inp = _rand_problem(gamma, 6)
    G, _, _ = _rows(gamma, cot, inp)
    # finite entries whose fp32 norm overflows to inf
    g_task = [torch.full_like(p, 1e38) for p in gamma.parameters()]
    _set_task_grad(gamma, g_task)
    diag = active_set_feasibility_projection(
        [g.clone() for g in g_task], list(gamma.parameters()), G, 0.05)
    assert not diag["proj_feasible"]
    assert diag["proj_abort"] == "nonfinite_task_direction"
    for p, g in zip(gamma.parameters(), g_task):
        assert torch.equal(p.grad, g), "driver must not write on abort"
    print("test_nonfinite_task_direction_is_refused OK")


def test_boundary_aborts_without_task_keys():
    g1, _, _, _, _ = _boundary_fixture()
    opt, diag, moved = _run_boundary(g1, [], [], [], [])
    assert diag["gamma_aborted"] and diag["gamma_steps"] == 0
    assert diag["gamma_abort_reason"] == "no_task_keys"
    assert opt.steps == 0
    print("test_boundary_aborts_without_task_keys OK")


# --- resume contract --------------------------------------------------------

class _Args:
    rpbe_mode = "project"
    kappa = 0.05
    proj_iters = 400
    proj_iters_max = 1600
    proj_tau = 1e-3
    proj_max_active = 2048
    proj_max_rounds = 8
    proj_add_per_round = 512


def test_resume_config_contract():
    want = boundary_config(_Args())
    assert set(want) == {"rpbe_mode", "kappa", "proj_iters", "proj_iters_max",
                         "proj_tau", "proj_max_active", "proj_max_rounds",
                         "proj_add_per_round"}
    # kappa is a FIXED constant on the formal method -- no annealing knobs
    assert not any("anneal" in k for k in COMMON_CONFIG_KEYS
                   + BOUNDARY_CONFIG_KEYS)
    # Stage8 ckpt, matching -> ok
    bad, legacy = verify_resume_config(dict(want), want)
    assert not bad and not legacy
    # Stage8 ckpt, silently changed kappa -> refused
    bad, legacy = verify_resume_config(dict(want, kappa=0.2), want)
    assert bad["kappa"] == (0.2, 0.05) and not legacy
    # Stage7 ckpt (no rpbe_mode) -> refused by default
    legacy_cfg = {k: v for k, v in want.items() if k != "rpbe_mode"}
    bad, legacy = verify_resume_config(legacy_cfg, want)
    assert "rpbe_mode" in bad and not legacy
    # ... allowed explicitly, exempting ONLY the Stage8-only keys
    bad, legacy = verify_resume_config(legacy_cfg, want, allow_legacy_gamma=True)
    assert not bad and legacy, "legacy migration path must be explicit"
    print("test_resume_config_contract OK")


def test_legacy_migration_still_checks_the_common_recipe():
    """Migration exempts the Stage8-only keys, NOT the shared training recipe:
    a Stage7 checkpoint with a different batch_size/grad_accum/mem_length must
    still be refused."""
    want = dict(boundary_config(_Args()), batch_size=4, grad_accum=2,
                mem_length=16, gamma_replay_batch_size=64,
                gamma_task_boundary_episodes=1, rpbe_stats_episodes=1,
                lambda_rpbe=0.0, kf_min_abs=64, sched="const")
    legacy_cfg = {k: v for k, v in want.items() if k != "rpbe_mode"}
    bad, legacy = verify_resume_config(legacy_cfg, want, allow_legacy_gamma=True)
    assert not bad and legacy
    for key, wrong in (("batch_size", 8), ("grad_accum", 4),
                       ("mem_length", 32), ("kf_min_abs", 128)):
        drifted = dict(legacy_cfg, **{key: wrong})
        bad, legacy = verify_resume_config(drifted, want,
                                           allow_legacy_gamma=True)
        assert legacy and key in bad, (key, bad)
        assert bad[key] == (wrong, want[key])
    print("test_legacy_migration_still_checks_the_common_recipe OK")


def test_production_solver_matches_full_qp_oracle():
    """The active-set solver must agree with an INDEPENDENT full-QP oracle
    (written here, not in production) on the projected direction."""
    gamma = TinyGamma()
    cot, inp = _rand_problem(gamma, 15)
    G, _, _ = _rows(gamma, cot, inp)
    g_task = _adversarial(gamma, G)
    _set_task_grad(gamma, g_task)
    act = active_set_feasibility_projection(
        [g.clone() for g in g_task], list(gamma.parameters()), G, 0.1)
    assert act["proj_feasible"]
    d_prod = _d_from_grad(gamma)
    d_ref, mu_ref, ref = _full_qp_reference(g_task, G, kappa=0.1)
    assert ref["feasible"], ref
    rel = float((d_prod - d_ref).norm() / max(float(d_ref.norm()), 1e-12))
    assert rel < 1e-3, f"active-set d disagrees with the full-QP oracle: {rel}"
    print("test_production_solver_matches_full_qp_oracle OK  rel=%.2e" % rel)


def test_scheduler_realignment_after_migration():
    """Legacy migration resets the Gamma optimizer, so the scheduler must be
    moved to param_version instead of sitting at step 0 (warmup LR)."""
    p = nn.Parameter(torch.zeros(3))
    opt = torch.optim.AdamW([p], lr=1e-3)
    warm, total = 30, 1200

    def lam(rs):
        if rs < warm:
            return rs / warm
        return 0.5 * (1.0 + math.cos(math.pi * min(
            1.0, (rs - warm) / max(1, total - warm))))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lam)
    assert sched.get_last_lr()[0] == 0.0            # fresh scheduler
    pv = 500
    lr = realign_lambda_scheduler(sched, pv)
    assert abs(lr[0] - 1e-3 * lam(pv)) < 1e-12, (lr, lam(pv))
    assert sched.last_epoch == pv
    # and a fresh scheduler would have been wrong there
    assert abs(1e-3 * lam(0) - 1e-3 * lam(pv)) > 1e-9
    print("test_scheduler_realignment_after_migration OK  lr=%.3e" % lr[0])


if __name__ == "__main__":
    test_fista_matches_slow_pgd()
    test_no_interfaces_is_pure_task()
    test_real_kkt_complementarity()
    test_opposite_interfaces_still_repaired()
    test_all_interfaces_checked_and_order_invariant()
    test_nonfinite_rows_dropped()
    test_infeasible_projection_is_dropped_by_row_filter()
    test_rows_backend_is_the_batched_vmap_path()
    test_batched_rows_match_per_interface_vjp()
    test_rows_are_detached_constants()
    test_stack_to_handles_mixed_inputs()
    test_iters_ladder()
    test_ill_conditioned_rows_certify_at_the_base_budget()
    test_row_normalisation_preserves_the_optimum()
    test_solver_budget_escalates_instead_of_aborting()
    test_kappa_ge_one_never_binds()
    test_round0_admits_only_tolerance_breaches()
    test_amp_scale_invariance()
    test_boundary_takes_exactly_one_step_per_arm()
    test_boundary_kappa_one_equals_task_control()
    test_infeasible_boundary_aborts_without_stepping()
    test_nonfinite_constraint_row_fails_closed()
    test_nonfinite_task_direction_is_refused()
    test_boundary_aborts_without_task_keys()
    test_resume_config_contract()
    test_legacy_migration_still_checks_the_common_recipe()
    test_production_solver_matches_full_qp_oracle()
    test_scheduler_realignment_after_migration()
    print("ALL_PROJECTION_TESTS_PASS")
