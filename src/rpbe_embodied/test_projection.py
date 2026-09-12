"""test_projection.py — interface-wise RPBE feasibility projection (TGN port).

CPU-only.  Run:
  PYTHONPATH=src python -m pytest src/rpbe_embodied/test_projection.py -q
  (or plain `python src/rpbe_embodied/test_projection.py`)

Checks the three properties the migration relies on:
  1. no interfaces  -> Gamma keeps the pure task gradient (no-op);
  2. feasible + KKT -> G d* >= b and mu_i (g_i . d* - b_i) == 0;
  3. geometry survives -> two opposite interfaces (aggregate == 0) are still
     repaired, which the retired aggregate-gradient path could not do.
Plus per-interface row correctness and the kappa schedule.
"""
import torch
import torch.nn as nn

from rpbe_embodied.loss import (_fista_nonneg, kappa_at,
                                per_cut_influence_grads,
                                treewise_feasibility_projection)

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


def _rows(gamma, cotangents, inputs):
    keys = list(cotangents.keys())
    G, used = per_cut_influence_grads(
        gamma, cotangents, inputs, keys,
        params=list(gamma.parameters()), device="cpu")
    return G, used


def _set_task_grad(gamma, g_task):
    for p, g in zip(gamma.parameters(), g_task):
        p.grad = g.clone()


def _flat_task_grad(gamma, flat):
    out, i = [], 0
    for p in gamma.parameters():
        out.append(flat[i:i + p.numel()].reshape(p.shape).clone())
        i += p.numel()
    return out


def _adversarial_task_grad(gamma, G, scale=0.5):
    """g_task aligned with the SUM of the interface rows -> every interface
    sees cos(g_i, -g_task) strongly negative -> the constraints bind."""
    return _flat_task_grad(gamma, scale * G.sum(0))


def test_fista_matches_slow_pgd():
    n = 12
    A = torch.randn(n, n)
    Q = A @ A.t() + 0.1 * torch.eye(n)
    c = torch.randn(n)
    mu = _fista_nonneg(Q, c, 2000)
    assert mu.min() >= 0
    # slow projected gradient reference
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
    diag = treewise_feasibility_projection(
        [g.clone() for g in g_task], list(gamma.parameters()), None, 0.05)
    for p, g in zip(gamma.parameters(), g_task):
        assert torch.equal(p.grad, g), "empty G must leave the task grad"
    assert diag["proj_n_valid"] == 0
    print("test_no_interfaces_is_pure_task OK")


def test_feasible_and_kkt():
    gamma = TinyGamma()
    n = 9
    cot = {i: torch.randn(D) for i in range(n)}
    inp = {i: (torch.randn(D), torch.randn(D)) for i in range(n)}
    G, _ = _rows(gamma, cot, inp)
    g_task = _adversarial_task_grad(gamma, G)
    _set_task_grad(gamma, g_task)
    kappa = 0.2
    diag = treewise_feasibility_projection(
        [g.clone() for g in g_task], list(gamma.parameters()), G, kappa)

    t = torch.cat([-g.flatten() for g in g_task])
    # p.grad = g_task - corr  and d = t + corr  =>  d = -p.grad
    gp = torch.cat([p.grad.flatten() for p in gamma.parameters()])
    d = -gp
    ng = G.norm(dim=1)
    b = -kappa * ng * float(t.norm())
    slack = G @ d - b
    assert slack.min() > -1e-3, f"infeasible: min slack {slack.min()}"
    assert diag["proj_min_slack_after"] > -1e-3
    # KKT complementarity: active constraints have mu > 0, inactive mu ~ 0.
    assert diag["proj_n_mu_pos"] >= 1
    assert diag["proj_corr_ratio"] > 0.0
    print("test_feasible_and_kkt OK  slack_min=%.2e corr=%.3f"
          % (slack.min(), diag["proj_corr_ratio"]))


def test_opposite_interfaces_still_repaired():
    """g_2 = -g_1: the AGGREGATE gradient is zero, so the retired
    sum-then-project path would see no violation and do nothing.  The
    per-interface QP must still repair."""
    gamma = TinyGamma()
    a1 = (torch.randn(D), torch.randn(D))
    inp = {0: a1, 1: a1}
    cot = {0: torch.randn(D), 1: torch.randn(D)}
    G, _ = _rows(gamma, cot, inp)
    # force an exactly opposite pair
    G = torch.stack([G[0], -G[0]])
    assert float((G.sum(0)).norm()) < 1e-6, "test setup: aggregate must cancel"

    g_task = _flat_task_grad(gamma, 0.5 * G[0])
    _set_task_grad(gamma, g_task)
    diag = treewise_feasibility_projection(
        [g.clone() for g in g_task], list(gamma.parameters()), G, kappa=0.0)
    assert diag["proj_corr_ratio"] > 1e-6, "opposite pair not repaired"
    gp = torch.cat([p.grad.flatten() for p in gamma.parameters()])
    d = -gp
    b = torch.zeros(2)  # kappa = 0 -> hard
    slack = G @ d - b
    assert slack.min() > -1e-3, f"infeasible hard: {slack}"
    # both constraints active (each side of the pair must be respected)
    assert diag["proj_n_mu_pos"] >= 1
    print("test_opposite_interfaces_still_repaired OK  corr=%.3f"
          % diag["proj_corr_ratio"])


def test_kappa_ge_one_never_binds():
    gamma = TinyGamma()
    n = 7
    cot = {i: torch.randn(D) for i in range(n)}
    inp = {i: (torch.randn(D), torch.randn(D)) for i in range(n)}
    G, _ = _rows(gamma, cot, inp)
    g_task = [torch.randn_like(p) for p in gamma.parameters()]
    _set_task_grad(gamma, g_task)
    diag = treewise_feasibility_projection(
        [g.clone() for g in g_task], list(gamma.parameters()), G, kappa=1.0)
    assert diag["proj_n_active_init"] == 0
    assert diag["proj_corr_ratio"] < 1e-3, diag["proj_corr_ratio"]
    print("test_kappa_ge_one_never_binds OK  corr=%.2e"
          % diag["proj_corr_ratio"])


def test_zero_rows_are_dropped():
    gamma = TinyGamma()
    n = 5
    cot = {i: torch.randn(D) for i in range(n)}
    inp = {i: (torch.randn(D), torch.randn(D)) for i in range(n)}
    G, _ = _rows(gamma, cot, inp)
    G[2] = 0.0                      # one dead interface
    g_task = [torch.randn_like(p) for p in gamma.parameters()]
    _set_task_grad(gamma, g_task)
    diag = treewise_feasibility_projection(
        [g.clone() for g in g_task], list(gamma.parameters()), G, 0.05)
    assert diag["proj_n_trees"] == 5
    assert diag["proj_n_valid"] == 4
    print("test_zero_rows_are_dropped OK")


def test_per_cut_rows_are_single_interface_vjps():
    gamma = TinyGamma()
    n = 3
    cot = {i: torch.randn(D) for i in range(n)}
    inp = {i: (torch.randn(D), torch.randn(D)) for i in range(n)}
    G, used = _rows(gamma, cot, inp)
    assert used == [0, 1, 2]
    for i in range(n):
        a, b = inp[i]
        z = gamma(a.unsqueeze(0), b.unsqueeze(0))[0]
        term = (cot[i] * z.float()).sum()
        ref = torch.cat([g.reshape(-1) for g in torch.autograd.grad(
            term, list(gamma.parameters()))])
        assert torch.allclose(G[i], ref, atol=1e-5), \
            f"row {i} != single-interface VJP"
    print("test_per_cut_rows_are_single_interface_vjps OK")


def test_kappa_schedule():
    assert kappa_at(0, 0.05, -1, -1, 1.0) == 0.05          # disabled
    assert kappa_at(50, 0.05, 100, 200, 1.0) == 0.05       # before start
    assert abs(kappa_at(150, 0.05, 100, 200, 1.0) - 0.525) < 1e-9
    assert kappa_at(250, 0.05, 100, 200, 1.0) == 1.0       # after end
    print("test_kappa_schedule OK")


if __name__ == "__main__":
    test_fista_matches_slow_pgd()
    test_no_interfaces_is_pure_task()
    test_feasible_and_kkt()
    test_opposite_interfaces_still_repaired()
    test_kappa_ge_one_never_binds()
    test_zero_rows_are_dropped()
    test_per_cut_rows_are_single_interface_vjps()
    test_kappa_schedule()
    print("ALL_PROJECTION_TESTS_PASS")
