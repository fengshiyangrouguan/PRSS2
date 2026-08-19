#!/usr/bin/env python3
"""Empirical check: replay-then-test (ours) vs official carry-over test protocol.

Both paths start from the same trained checkpoint.  The question: at the moment
test evaluation starts, is the TGN memory state identical, and do the two
protocols produce the same test MRR?
"""

import argparse
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SRC = ROOT / "src"
for path in (str(SRC), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from prss.data.tgb_link import TGBLinkDataset
from prss.monitoring import MonitorWriter
from prss.training.event_loop import TGBLinkPredictionLoop


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="path to best.pt")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--output", default="/tmp/protocol_ab.json")
    args = p.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    ds = TGBLinkDataset(name="tgbl-wiki", device=device)
    time_stats = ds.time_stats()

    import scripts.train as T

    class A:
        pass

    a = A()
    a.variant = "vanilla"
    a.dataset = "tgbl-wiki"
    a.seed = 0
    a.gpu = args.gpu
    a.emb_dim = 100
    a.mem_dim = 100
    a.time_dim = 100
    a.n_neighbors = 10
    a.candidate_dim = 256
    a.candidate_hidden = 128
    a.context_dim = 64
    a.reader_hidden = 128
    a.lambda_resp = 1.0
    a.lambda_spec = 0.1
    a.gram_ema = 0.05
    a.spectral_warmup = 100
    a.spectral_interval = 100
    a.spectral_step_size = 0.25
    a.trace_roots = 8
    a.lr = 1e-4
    a.freeze_host = False

    components, _ = T.build_components(a, device, ds, time_stats)
    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    for name in ("memory", "gnn", "link_pred"):
        components[name].load_state_dict(ck["model"][name])

    loop = TGBLinkPredictionLoop(
        dataset=ds, memory=components["memory"], gnn=components["gnn"],
        link_pred=components["link_pred"], neighbor_loader=components["neighbor_loader"],
        adapter=None, bridge=None, prss_core=None,
        optimizer=components["optimizer"],
        unrestricted_optimizer=None,
        criterion=components["criterion"], device=device, batch_size=200,
        n_neighbors=10, grad_clip=0.0, lambda_resp=0.0, lambda_spec=0.0,
        trace_roots=0, spectral_warmup=10 ** 9, spectral_interval=10 ** 9,
        monitor=MonitorWriter("/tmp/protocol_ab_mon", fail_on_error=False),
        seed=0)

    # ---- Path B (ours): reset -> replay train (train-mode) -> replay val -> test
    loop.reset_stream()
    loop.replay_split("train")
    loop.replay_split("val")
    mem_b = loop.memory.memory.detach().cpu().clone()
    result_b = loop.evaluate_split("test")

    # ---- Path A (official-style): reset -> replay train -> val pass (eval-mode
    #      advancement, as the official baseline leaves it) -> test without reset
    loop.reset_stream()
    loop.replay_split("train")
    loop.evaluate_split("val")  # advances memory with official eval semantics
    mem_a = loop.memory.memory.detach().cpu().clone()
    result_a = loop.evaluate_split("test")

    import json
    import numpy as np

    max_diff = float((mem_a - mem_b).abs().max())
    mean_diff = float((mem_a - mem_b).abs().mean())
    out = {
        "protocol_ours_replay": result_b,
        "protocol_official_carryover": result_a,
        "memory_max_abs_diff_at_test_start": max_diff,
        "memory_mean_abs_diff_at_test_start": mean_diff,
        "memory_allclose_1e-4": bool(torch.allclose(mem_a, mem_b, atol=1e-4, rtol=1e-4)),
        "memory_allclose_1e-6": bool(torch.allclose(mem_a, mem_b, atol=1e-6, rtol=1e-6)),
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, allow_nan=True)
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
