"""Unit tests for the Tree-wise RPBE Feasibility Projection.

Pure torch; the helper sources are pulled out of scripts/train_lamp.py with ast
so the test does not import the training module.  Run with pytest or directly:

    python test/test_lamp_projection.py
"""
import ast
import pathlib

import torch
from torch import nn

_SRC = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "train_lamp.py"


def _load(names):
    tree = ast.parse(_SRC.read_text(encoding="utf-8"))
    keep = [n for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name in names]
    ns = {"torch": torch, "math": __import__("math")}
    exec(compile(ast.Module(body=keep, type_ignores=[]), "proj", "exec"), ns)
    return ns


_NS = _load({"treewise_feasibility_projection", "_fista_nonneg", "kappa_at"})
PROJ = _NS["treewise_feasibility_projection"]
KAPPA_AT = _NS["kappa_at"]


def _mk(n_trees, dim, seed, conflict):
    """Gamma-like params, per-tree g_i with target cos(i, t)=conflict."""
    torch.manual_seed(seed)
    sizes = [dim // 2, dim - dim // 2]
    ps = [nn.Parameter(torch.zeros(s)) for s in sizes]
    gt = [torch.randn(s, generator=torch.Generator().manual_seed(seed + 1))
          for s in sizes]
    t = torch.cat([-g for g in gt]).float()
    t = t / t.norm()
    rows = []
    for i in range(n_trees):
        gi = torch.randn(dim, generator=torch.Generator().manual_seed(100 + i))
        gi = gi - gi.dot(t) * t
        gi = gi / gi.norm()
        gi = conflict * t + (1.0 - conflict ** 2) ** 0.5 * gi
        rows.append(gi * (1.0 + 0.1 * i))
    return ps, gt, torch.stack(rows).float(), t


def _run(ps, gt, G, kappa, iters=500):
    for p, g in zip(ps, gt):
        p.grad = g.clone()
    return PROJ(gt, ps, G, kappa, iters=iters)


def test_no_trees_is_noop():
    ps, gt, G, t = _mk(3, 8, 0, 0.5)
    diag = PROJ(gt, ps, None, 0.05)
    assert diag["proj_n_trees"] == 0
    for p, g in zip(ps, gt):
        assert p.grad is None or torch.allclose(p.grad, g)


def test_kkt_complementarity_and_feasibility():
    for seed in range(3):
        ps, gt, G, t = _mk(6, 12, seed, conflict=-0.6)
        _run(ps, gt, G, kappa=0.05)
        d = -torch.cat([p.grad.reshape(-1) for p in ps])
        tt = torch.cat([-g.reshape(-1) for g in gt])
        mu = torch.linalg.lstsq(G.t(), d - tt).solution
        b = -0.05 * G.norm(dim=1) * tt.norm()
        slack = G @ d - b
        assert float(slack.min()) > -1e-4, "all constraints feasible"
        tight = mu > 1e-5
        if tight.any():
            assert float(slack[tight].abs().max()) < 1e-3, "active => tight"
        assert float(tt @ d) > 0.0, "task direction still descends"


def test_opposite_trees_do_not_cancel():
    """g1 = -g2 (aggregate zero), both conflicting: sum-then-project would do
    nothing; the tree-wise QP must repair both."""
    dim = 10
    torch.manual_seed(1)
    ps = [nn.Parameter(torch.zeros(dim))]
    gt = [torch.randn(dim, generator=torch.Generator().manual_seed(2))]
    tt = -gt[0].clone()                        # PROJ uses t = -g_task (unnorm)
    t_unit = tt / tt.norm()
    p1 = -0.9 * t_unit + (1 - 0.81) ** 0.5 * torch.randn(
        dim, generator=torch.Generator().manual_seed(3))
    g1 = p1 / p1.norm()
    G = torch.stack([g1, -g1]).float()
    assert torch.allclose(G.sum(0), torch.zeros(dim), atol=1e-6)
    ps[0].grad = gt[0].clone()
    diag = PROJ(gt, ps, G, kappa=0.05, iters=800)
    d = -ps[0].grad
    assert not torch.allclose(d, tt, atol=1e-4), "must move off the task dir"
    b = -0.05 * G.norm(dim=1) * tt.norm()
    slack = G @ d - b
    assert float(slack.min()) > -1e-4, "both trees protected"
    assert diag["proj_n_mu_pos"] >= 1


def test_amp_scale_invariance():
    """Scaling all gradients by S scales d by S (mu unchanged): the constraint
    is invariant to the GradScaler factor."""
    ps, gt, G, t = _mk(5, 10, 7, conflict=-0.6)
    _run(ps, gt, G, 0.05)
    grad0 = torch.cat([p.grad.reshape(-1) for p in ps]).clone()
    S = 512.0
    ps2 = [nn.Parameter(torch.zeros_like(p)) for p in ps]
    PROJ([g * S for g in gt], ps2, G * S, 0.05, iters=500)
    gradS = torch.cat([p.grad.reshape(-1) for p in ps2])
    assert torch.allclose(gradS, grad0 * S, rtol=1e-3, atol=1e-3)


def test_kappa_zero_is_hard_per_tree():
    ps, gt, G, t = _mk(4, 10, 9, conflict=-0.8)
    _run(ps, gt, G, kappa=0.0, iters=800)
    d = -torch.cat([p.grad.reshape(-1) for p in ps])
    tt = torch.cat([-g.reshape(-1) for g in gt])
    active = (G @ tt) < 0
    slack = G @ d
    assert float(slack.min()) > -1e-4
    if active.any():
        assert float(slack[active].abs().max()) < 1e-3, "hard: g.d = 0"


def test_kappa_anneal_schedule():
    """kappa holds until `start`, then goes LINEARLY to `to` at `end`.
    Direction reminder: larger kappa = looser (>=1 never binds = pure task)."""
    assert KAPPA_AT(0, 0.05, -1, -1) == 0.05          # no anneal
    assert KAPPA_AT(69, 0.05, 70, 120, 1.0) == 0.05   # before start
    assert abs(KAPPA_AT(95, 0.05, 70, 120, 1.0)
               - (0.05 + 0.5 * 0.95)) < 1e-9          # midpoint
    assert KAPPA_AT(120, 0.05, 70, 120, 1.0) == 1.0   # at end
    assert KAPPA_AT(300, 0.05, 70, 120, 1.0) == 1.0   # after end
    assert KAPPA_AT(95, 0.05, 70, 120, 0.0) < 0.05    # annealing DOWN is stricter
    assert KAPPA_AT(200, 0.05, 70, 60, 1.0) == 1.0    # degenerate end<=start


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print("PASS", fn.__name__)
    print("all", len(fns), "tree-wise projection tests passed")
