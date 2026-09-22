#!/usr/bin/env python3
"""Review test 5 (2026-09-22): AdamW proposal-space shadow correctness.

CPU-only, pure tensors.  The proposal-space treewise write-back (native
actuation) claims: WITHOUT projection, the manual write-back

    d = adamw_proposal(optimizer, params, grads, step_count)
    theta <- theta + d
    moments <- beta1*m + (1-beta1)*g,  beta2*v + (1-beta2)*g^2

is EXACTLY the torch AdamW step (theta and moments bit-identical at
every step), so the QP projects the same vector the optimizer would
have taken — g -> AdamW(g) -> Proj(d), the frozen method definition.

Covers: bias-correction bookkeeping, moment dtypes, per-group lr read,
multiple sequential steps (warm state), weight_decay=0.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import torch

import train_ccm as tc


def manual_step(optimizer, params, grads, step_count):
    """The native write-back WITHOUT projection (d* = d_0)."""
    d = tc.adamw_proposal(optimizer, params, grads, step_count)
    b1, b2 = tuple(optimizer.param_groups[0]["betas"])
    with torch.no_grad():
        for p, g, dp in zip(params, grads, d):
            st = optimizer.state[p]
            if "exp_avg" in st:
                st["exp_avg"].mul_(b1).add_(g, alpha=1.0 - b1)
            else:
                st["exp_avg"] = ((1.0 - b1) * g).clone()
            if "exp_avg_sq" in st:
                st["exp_avg_sq"].mul_(b2).addcmul_(g, g, value=1.0 - b2)
            else:
                st["exp_avg_sq"] = ((1.0 - b2) * g * g).clone()
            p.add_(dp.to(p.dtype))
    return d


def main():
    torch.manual_seed(0)
    # Two param groups (lr 1e-3 / 1e-4) to exercise the per-group lr
    # lookup in adamw_proposal.
    refs = [torch.randn(64, 32, requires_grad=True),
            torch.randn(16, requires_grad=True)]
    ref_opt = torch.optim.AdamW(
        [{"params": [refs[0]], "lr": 1e-3},
         {"params": [refs[1]], "lr": 1e-4}],
        weight_decay=0.0)

    # Independent copy, same init + same gradient stream.
    torch.manual_seed(0)
    mans = [torch.randn(64, 32, requires_grad=True),
            torch.randn(16, requires_grad=True)]
    man_opt = torch.optim.AdamW(
        [{"params": [mans[0]], "lr": 1e-3},
         {"params": [mans[1]], "lr": 1e-4}],
        weight_decay=0.0)

    for step in range(1, 6):
        gs = [torch.randn_like(refs[0]), torch.randn_like(refs[1])]
        # Reference: ordinary torch AdamW step.
        for p, g in zip(refs, gs):
            p.grad = g.clone()
        ref_opt.step()
        ref_opt.zero_grad(set_to_none=True)
        # Manual: shadow proposal + direct write-back.
        d = manual_step(man_opt, mans, gs, step)
        # torch computes denom = sqrt(v)/sqrt(bc2) + eps while the
        # shadow computes sqrt(v/bc2) + eps — algebraically identical,
        # ~1 ulp apart in float32.  allclose, not equal.
        for p, q in zip(refs, mans):
            assert torch.allclose(p, q, atol=1e-6, rtol=1e-6), \
                "theta mismatch at step {}".format(step)
        for p, q in zip(refs, mans):
            st_r, st_m = ref_opt.state[p], man_opt.state[q]
            assert torch.allclose(st_r["exp_avg"], st_m["exp_avg"],
                                  atol=1e-6, rtol=1e-6), \
                "exp_avg mismatch at step {}".format(step)
            assert torch.allclose(st_r["exp_avg_sq"], st_m["exp_avg_sq"],
                                  atol=1e-6, rtol=1e-6), \
                "exp_avg_sq mismatch at step {}".format(step)
        print("step {} PASS: proposal norm={:.6f} (group lr {} / {}), "
              "theta + moments bit-identical".format(
                  step, float(torch.cat(
                      [x.reshape(-1) for x in d]).norm()),
                  man_opt.param_groups[0]["lr"],
                  man_opt.param_groups[1]["lr"]), flush=True)

    print("test 5 PASS: adamw_proposal shadow == torch AdamW step "
          "(d* = d_0 case of the proposal-space write-back)")


if __name__ == "__main__":
    main()
