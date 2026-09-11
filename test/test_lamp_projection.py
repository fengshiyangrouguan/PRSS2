"""Unit tests for the Plan-B RPBE projection (project_fp_guardrail).

Pure torch; the helper source is pulled out of scripts/train_lamp.py with ast so
the test does not import the training module (which chdirs to the CCM root and
pulls heavy deps).  Run either with pytest or directly:

    python test/test_lamp_projection.py
"""
import ast
import math
import pathlib

import torch
from torch import nn

_SRC = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "train_lamp.py"


def _load_pfg():
    tree = ast.parse(_SRC.read_text(encoding="utf-8"))
    fn = [n for n in tree.body
          if isinstance(n, ast.FunctionDef)
          and n.name == "project_fp_guardrail"][0]
    ns = {"torch": torch}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "pfg", "exec"), ns)
    return ns["project_fp_guardrail"]


PFG = _load_pfg()


def _vec(seed, n, mag, cos, t):
    """A vector g with ||g||=mag and cos(g, t)=cos (t a unit vector)."""
    g = torch.randn(n, generator=torch.Generator().manual_seed(seed))
    t = t / t.norm()
    perp = g - g.dot(t) / t.dot(t) * t
    perp = perp / perp.norm()
    return mag * (cos * t + math.sqrt(max(0.0, 1.0 - cos ** 2)) * perp)


def _mk(seed, gmag, cos, n=6):
    g_t = torch.randn(n, generator=torch.Generator().manual_seed(seed + 99))
    t = -g_t                                   # task descent direction
    g = _vec(seed, n, gmag, cos, t)
    param = nn.Parameter(torch.zeros(n))
    param.grad = g_t.clone()                   # p.grad holds g_task
    return [param], [g_t.clone()], [g.clone()], g, t


def test_inactive_noop():
    p, gt, gr, g, t = _mk(0, 2.0, cos=0.5)
    p[0].grad = gt[0].clone()
    b = -0.05 * g.norm() * t.norm()
    d = PFG(gt, p, gr, float(b))
    assert not d["proj_active"]
    assert torch.allclose(p[0].grad, gt[0]), "inactive must not touch grad"


def test_active_hits_boundary_and_keeps_task_descent():
    p, gt, gr, g, t = _mk(1, 2.0, cos=-0.9)
    p[0].grad = gt[0].clone()
    b = -0.05 * g.norm() * t.norm()
    d = PFG(gt, p, gr, float(b))
    assert d["proj_active"]
    mu = d["proj_mu"]
    dd = t + mu * g                            # d = t + mu*g
    assert abs(float(g.dot(dd)) - float(b)) < 1e-5, "g.d must equal boundary"
    assert float(t.dot(dd)) > 0.0, "task direction must still descend"
    assert d["proj_corr_ratio"] > 0.0


def test_extreme_cos_minus_one_gives_zero_update():
    p, gt, gr, g, t = _mk(2, 3.0, cos=-1.0)
    p[0].grad = gt[0].clone()
    d = PFG(gt, p, gr, 0.0)                    # hard constraint, c = -1
    assert d["proj_active"]
    assert abs(d["proj_corr_ratio"] - 1.0) < 1e-6
    assert float(p[0].grad.abs().max()) < 1e-6, "d must be ~0 when c=-1"


def test_amp_scale_invariance_with_s2_scaled_floor():
    """With GradScaler scale S the grads are S-scaled; an absolute floor
    boundary must be scaled by S^2 so the written (scaled) grad is exactly S x
    the true projection (i.e. unscale-invariant)."""
    n, S = 6, 1024.0
    p, gt, gr, g, t = _mk(3, 2.0, cos=-0.9, n=n)
    lr = 3e-5
    j_floor, j_now = 10.0, 5.0
    b_true = (j_floor - j_now) / lr
    # true-scale projection
    p[0].grad = gt[0].clone()
    d_true = PFG(gt, p, gr, float(b_true))
    grad_true = p[0].grad.clone()
    # scaled run: grads * S, boundary * S^2
    gt_s = [x * S for x in gt]
    gr_s = [x * S for x in gr]
    p[0].grad = gt_s[0].clone()
    d_s = PFG(gt_s, p, gr_s, float(S * S * b_true))
    assert d_s["proj_active"] == d_true["proj_active"]
    assert torch.allclose(p[0].grad, S * grad_true, rtol=1e-4, atol=1e-6), \
        "S^2-scaled floor boundary must be scale-invariant"
    assert abs(d_s["proj_mu"] - d_true["proj_mu"]) < 1e-6


def test_unscaled_floor_boundary_is_wrong():
    """The pre-fix behaviour (boundary NOT scaled by S^2) collapses the floor
    by S^2; this documents why the fix is needed."""
    n, S = 6, 1024.0
    p, gt, gr, g, t = _mk(4, 2.0, cos=-0.9, n=n)
    lr = 3e-5
    b_true = (10.0 - 5.0) / lr
    gt_s = [x * S for x in gt]
    gr_s = [x * S for x in gr]
    p[0].grad = gt_s[0].clone()
    d_bug = PFG(gt_s, p, gr_s, float(b_true))          # unscaled (bug)
    # effective g.d after unscale is b_true / S^2 -> collapsed, not b_true
    eff = d_bug["proj_gJ_dot_d_after"] / (S * S)
    assert abs(eff - b_true / (S * S)) < 1e-3
    assert abs(abs(eff) - abs(b_true)) > 1.0, "floor is collapsed by S^2"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print("PASS", fn.__name__)
    print("all", len(fns), "projection tests passed")
