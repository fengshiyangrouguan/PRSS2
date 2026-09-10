#!/usr/bin/env python3
"""RPBE gradient probe: measure task-vs-aux compressor grad norms on UCI.

Loads the ours (2obs_aligned) config through train_uci_link.main() with
gauge_comp patched so the zero-VALUED straight-line surrogate (value 0,
gradient = adjoint) is still measured.  Runs 1 epoch x 80 batches
(2 macro groups, >= 1 window close each).  Prints task/aux grad norms and
r_eff = aux_norm / task_norm — the lambda-calibration ratio (target band
[0.05, 0.30] per the CCM-calibration protocol).

Usage:  python scripts/diag_rpbe_grad.py
"""
import importlib.util
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

spec = importlib.util.spec_from_file_location(
    "train_uci_link", os.path.join(HERE, "train_uci_link.py"))
TU = importlib.util.module_from_spec(spec)
spec.loader.exec_module(TU)

from rpbe.training.tgb_link_loop import TGBPairLinkLoop  # noqa: E402

_orig_gauge = TGBPairLinkLoop.gauge_comp


def patched_gauge(self, link_loss, aux_scalar):
    """Same as the original but measures aux even when the surrogate's VALUE
    is 0 (its gradient is nonzero — the straight-line estimator)."""
    if not self._comp_params:
        return 0.0, 0.0
    params = self._comp_params
    g_t = torch.autograd.grad(link_loss, params, retain_graph=True,
                              allow_unused=True)
    task_n = self._g_norm(g_t)
    aux_n = 0.0
    if aux_scalar is not None:
        g_a = torch.autograd.grad(aux_scalar, params, retain_graph=True,
                                  allow_unused=True)
        aux_n = self._g_norm(g_a)
    return float(task_n), float(aux_n)


TGBPairLinkLoop.gauge_comp = patched_gauge

OUT = "outputs/diag_grad_probe"
sys.argv = [
    "train_uci_link",
    "--arm", "2obs_aligned",
    "--data-dir", "/root/autodl-tmp/benchtemp/data_uci",
    "--gpu", "0",
    "--epochs", "1",
    "--budget-cap", "1",
    "--max-batches", "80",
    "--kf-group-batches", "40",
    "--n-neighbors", "10",
    "--n-layers", "3",
    "--seed", "0",
    "--output", OUT,
]
print("== diag_rpbe_grad: patched gauge, ours arm, 1 epoch x 80 batches ==",
      flush=True)
TU.main()

import json  # noqa: E402
rows = [json.loads(l) for l in open(os.path.join(OUT, "metrics.jsonl"))]
print("== metrics ==", flush=True)
for r in rows:
    print({k: r.get(k) for k in ("epoch", "aux_comp_grad_norm",
                                 "task_comp_grad_norm", "n_closed",
                                 "n_aux_batches", "train_aux")}, flush=True)
