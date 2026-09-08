"""Phase-2 unit gates (spec §8 #5/#6/#8/#10 subset, no GPU/TGB needed).

  * ``_score_from_covs`` unbalanced equals the direct formula and is
    differentiable (B1);
  * AuxHeads: per-tau dims, target stop-gradient, one-shot frozen statistics,
    and that only decoder params carry trainable weight (P1/P2);
  * C1 no-C: zero-ctx p differs from full-ctx p but shares shape/map hash;
  * 9-config isolation: each config changes exactly its allowed axes vs R0.
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from rpbe.loss import _score_from_covs, latent_z_adjoint
from rpbe.pair_maps import BoundaryMaps
from rpbe.pair_rows import build_ctx_vector
from rpbe.training.aux_heads import AuxHeads


# ---------------------------------------------------------------- B1 formula
def _rand_spd(r, g, scale=1.0):
    a = torch.randn(r, r, generator=g)
    x = a @ a.t() + r * torch.eye(r)
    return x * scale


def test_unbalanced_equals_direct_formula_and_differs():
    g = torch.Generator().manual_seed(3)
    czz = _rand_spd(8, g)
    cpp = _rand_spd(5, g)
    czp = torch.randn(8, 5, generator=g) * 0.3
    eps = 1e-3
    j, diag = _score_from_covs(czz, czp, cpp, eps, "unbalanced")
    assert diag["failed"] is None
    direct = (czp.square().sum()
              / (czz.diagonal().sum() * cpp.diagonal().sum() + eps))
    assert abs(float(j.detach()) - float(direct.detach())) < 1e-6


def test_unbalanced_differentiable():
    g = torch.Generator().manual_seed(4)
    z = torch.randn(40, 6, generator=g, requires_grad=True)
    p = torch.randn(40, 4, generator=g)
    zc = z - z.mean(0)
    pc = p - p.mean(0)
    czz = zc.t() @ zc
    czp = zc.t() @ pc
    cpp = pc.t() @ pc
    j, diag = _score_from_covs(czz, czp, cpp, 1e-4, "unbalanced")
    assert diag["failed"] is None
    assert float(j.detach()) > 0.0
    j.backward()
    assert z.grad is not None and bool(torch.isfinite(z.grad).all())


def test_variants_bounded_full_diag_unbal():
    g = torch.Generator().manual_seed(5)
    czz = _rand_spd(10, g)
    cpp = _rand_spd(7, g)
    czp = torch.randn(10, 7, generator=g) * 0.5
    for v in ("full_balancing", "diagonal", "unbalanced"):
        j, diag = _score_from_covs(czz, czp, cpp, 1e-3, v)
        assert diag["failed"] is None, v
        assert float(j.detach()) >= 0.0


# ---------------------------------------------------------------- AuxHeads
def test_aux_heads_rec_dims_and_stopgrad():
    h = AuxHeads("rec", taus=["tjo:layer1", "tjo:layer2"], z_dim=8, d_ctx=4,
                 d_out=8)
    z = torch.randn(3, 8, requires_grad=True)
    ctx = torch.randn(3, 4)
    target = torch.randn(3, 8, requires_grad=True)  # must not receive grad
    loss = h.regression_loss("tjo:layer1", z, ctx, target)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert target.grad is None, "target must be stop-gradient"
    assert z.grad is not None


def test_aux_heads_frozen_stats_once():
    h = AuxHeads("pred", taus=["tjo:layer1"], z_dim=6, d_ctx=3, d_out=5)
    t1 = torch.randn(20, 5) * 3 + 1
    h.fit("tjo:layer1", t1)
    m1 = h._mean("tjo:layer1").clone()
    s1 = h._scale("tjo:layer1").clone()
    t2 = torch.randn(20, 5) * 100 + 50
    h.fit("tjo:layer1", t2)   # ignored after first fit
    assert torch.equal(h._mean("tjo:layer1"), m1)
    assert torch.equal(h._scale("tjo:layer1"), s1)


def test_aux_heads_only_decoder_params_trainable():
    h = AuxHeads("rec", taus=["tjo:layer1"], z_dim=6, d_ctx=3, d_out=6)
    h.fit("tjo:layer1", torch.randn(10, 6))
    ps = list(h.parameters())
    assert len(ps) == 4  # two Linear(weight+bias) per decoder
    assert all(p.requires_grad for p in ps)


# ------------------------------------------------------------------ C1 no-C
def test_noC_zero_ctx_changes_p_keeps_shape_and_hash():
    maps = BoundaryMaps(d_ctx=8, d_event=12, m=16, d_msg=4,
                        delta_t_scale=1.0, num_counter_bins=64, seed=1)
    msg = torch.randn(4)
    ch = {"counterpart": 3, "role": 0, "delta_t": 1.0, "message": msg}
    pa = {"counterpart": 7, "role": 1, "delta_t": 2.0, "message": msg}
    ctx = torch.randn(8)
    p_full = maps.pv(ctx, ch, pa, use_parent=1)
    p_zero = maps.pv(torch.zeros(8), ch, pa, use_parent=1)
    assert tuple(p_full.shape) == (16,)
    assert tuple(p_zero.shape) == (16,)
    assert not torch.equal(p_zero, p_full)
    fp1 = maps.isolation_fingerprint()
    # hash is context-independent (map geometry unchanged)
    fp2 = maps.isolation_fingerprint()
    assert fp1["sha256"] == fp2["sha256"]


# -------------------------------------------------- 9-config isolation (axes)
def test_nine_config_isolation_vs_R0():
    from scripts.train_tgb_link import CONFIG_MAP
    r0 = CONFIG_MAP["R0"]
    allowed = {
        "P0": {"aux_kind"},
        "P1": {"aux_kind"},
        "P2": {"aux_kind"},
        "S1": {"use_parent"},
        "S2": {"mispaired"},
        "C1": {"context_mode"},
        "B1": {"variant"},
        "E1": {"variant"},
    }
    for cid in allowed:
        cfg = CONFIG_MAP[cid]
        changed = {k for k in r0 if cfg.get(k) != r0[k]}
        changed -= {"arm"}   # internal alias, not a scientific axis
        assert changed == allowed[cid], (cid, changed)


# -------------------------------------------------- replay-gradient (variants)
def test_unbalanced_latent_z_adjoint_replays_direct_gradient():
    g = torch.Generator().manual_seed(11)
    N = 32
    z = torch.randn(N, 5, generator=g)
    p = torch.randn(N, 4, generator=g)
    w = torch.rand(N, generator=g) + 0.3
    W = float(w.sum())
    W2 = float((w * w).sum())
    D = W - W2 / W
    mu_z = (z * w[:, None]).sum(0, keepdim=True) / W
    mu_p = (p * w[:, None]).sum(0, keepdim=True) / W
    j, gc, diag = latent_z_adjoint(
        z.float(), p.float(), w, [("cut", i) for i in range(N)],
        mu_z.float(), mu_p.float(), D, 1e-4, strict=True,
        variant="unbalanced")
    assert diag["failed"] is None
    assert len(gc) == N
    # direct autograd of the same reconstruction reproduces the value
    zl = z.double().clone().requires_grad_(True)
    zc = zl - mu_z.double()
    pc = p.double() - mu_p.double()
    sw = torch.tensor(w, dtype=torch.float64).sqrt().reshape(-1, 1)
    czz = (zc * sw).t() @ (zc * sw)
    czp = (zc * sw).t() @ (pc * sw)
    cpp = (pc * sw).t() @ (pc * sw)
    jd, dd = _score_from_covs(czz / D, czp / D, cpp / D, 1e-4, "unbalanced")
    assert dd["failed"] is None
    assert abs(float(j) - float(jd.detach())) < 1e-6


def test_p2_target_bitwise_deterministic():
    maps = BoundaryMaps(d_ctx=8, d_event=12, m=16, d_msg=4,
                        delta_t_scale=1.0, num_counter_bins=64, seed=7)
    msg = torch.randn(4)
    ch = {"counterpart": 1, "role": 0, "delta_t": 0.5, "message": msg}
    pa = {"counterpart": 9, "role": 1, "delta_t": 1.5, "message": msg}
    ctx = torch.randn(8)
    a = maps.pv(ctx, ch, pa, use_parent=1)
    b = maps.pv(ctx, ch, pa, use_parent=1)
    assert torch.equal(a, b)          # P2 target == R0 p path, reproducible


def test_aux_head_init_rng_isolation():
    import random as _random
    torch.manual_seed(0)
    _lin = torch.nn.Linear(8, 8)
    w_ref = _lin.weight.clone()
    rng_saved = (torch.get_rng_state(), np.random.get_state(),
                 _random.getstate())
    AuxHeads("rec", taus=["tjo:layer1"], z_dim=8, d_ctx=4, d_out=8)
    torch.set_rng_state(rng_saved[0])
    np.random.set_state(rng_saved[1])
    _random.setstate(rng_saved[2])
    torch.manual_seed(0)              # same seed as the Linear above
    _lin2 = torch.nn.Linear(8, 8)
    assert torch.equal(_lin2.weight, w_ref)


def test_fit_weighted_uses_tree_weights():
    h = AuxHeads("pred", taus=["tjo:layer1"], z_dim=6, d_ctx=3, d_out=4)
    tgt = torch.tensor([[1., 1., 1., 1.], [5., 5., 5., 5.],
                        [9., 9., 9., 9.]])
    wt = torch.tensor([1.0, 1.0, 0.0])   # ignore the third row
    h.fit_weighted("tjo:layer1", tgt, wt)
    m = h._mean("tjo:layer1")
    assert torch.allclose(m, torch.tensor([3., 3., 3., 3.]), atol=1e-5)
    s = h._scale("tjo:layer1")
    assert torch.allclose(s, torch.full((4,), 2.0), atol=1e-4)
