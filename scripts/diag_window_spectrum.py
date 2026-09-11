#!/usr/bin/env python3
"""Window spectrum diagnostic (reviewer items 3/4).

Answers "are the window rows too similar / is the factorization degenerate"
with direct measurements on the first two macro groups at fixed init
(no weight updates — reuses the --calibrate path):

* C_ZZ eigenvalue spectrum + participation-ratio effective rank (vs d=172);
* C_ZP singular-value distribution + top-3 concentration;
* mean |cosine| between pairs of sampled z rows and p rows.

Usage (server):
    python scripts/diag_window_spectrum.py \
        --data-dir /root/autodl-tmp/benchtemp/data_uci \
        [--n-layers 3] [--n-neighbors 10] [--groups 2] [--gpu 0]

Writes outputs/window_spectrum_diag/spectrum.json.
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
for _p in (str(SRC), str(HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch

from rpbe.pair_window import PairKFWindow

_spectrum_lines = []
_orig_close = PairKFWindow.close_replay


def _patched_close(self, maps):
    records = self._records
    if records:
        z = torch.stack([r.z for r in records]).double()
        p = torch.stack([maps.pv_row(r) for r in records]).double()
        w = torch.tensor([float(r.weight) for r in records],
                         dtype=torch.float64, device=z.device)
        W = float(w.sum())
        W2 = float((w * w).sum())
        D = W - W2 / W
        sw = w.sqrt().reshape(-1, 1)
        zc = z - (z * w[:, None]).sum(0, keepdim=True) / W
        pc = p - (p * w[:, None]).sum(0, keepdim=True) / W
        czz = ((zc * sw).t() @ (zc * sw)) / D
        czp = ((zc * sw).t() @ (pc * sw)) / D
        # 1) C_ZZ spectrum / effective rank (participation ratio)
        ev = np.linalg.eigvalsh(czz.detach().cpu().numpy())[::-1]
        pr = float((ev.sum() ** 2) / max(float((ev ** 2).sum()), 1e-30))
        # 2) C_ZP singular values + top-3 concentration
        s = np.linalg.svd(czp.detach().cpu().numpy(), compute_uv=False)
        conc3 = float(s[:3].sum()) / max(float(s.sum()), 1e-30)
        # 3) row similarity on a subsample
        idx = np.linspace(0, len(z) - 1, min(300, len(z))).astype(int)
        zs = zc[idx]
        zn = zs / zs.norm(dim=1, keepdim=True).clamp(min=1e-12)
        Gz = zn @ zn.t()
        iu = np.triu_indices(len(idx), 1)
        cos_z = float(Gz[iu].abs().mean())
        ps_ = pc[idx]
        pn = ps_ / ps_.norm(dim=1, keepdim=True).clamp(min=1e-12)
        Gp = pn @ pn.t()
        cos_p = float(Gp[iu].abs().mean())
        row = {
            "tau": self.tau,
            "M": len(records),
            "ev_top5": [round(float(x), 3) for x in ev[:5]],
            "ev_bottom": round(float(ev[-1]), 5),
            "eff_rank": round(float(pr), 1),
            "d_z": int(z.shape[1]),
            "sv_top8": [round(float(x), 4) for x in s[:8]],
            "sv_conc_top3": round(float(conc3), 3),
            "mean_abs_cos_z": round(float(cos_z), 4),
            "mean_abs_cos_p": round(float(cos_p), 4),
        }
        _spectrum_lines.append(row)
        print("[spectrum] tau=%s M=%d eff_rank=%.1f/%d sv_conc3=%.2f "
              "|cos_z|=%.3f |cos_p|=%.3f"
              % (self.tau, len(records), pr, int(z.shape[1]),
                 conc3, cos_z, cos_p), flush=True)
    return _orig_close(self, maps)


def main():
    ap = argparse.ArgumentParser("window spectrum diagnostic")
    ap.add_argument("--data-dir", default="/root/autodl-tmp/benchtemp/data_uci")
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--n-neighbors", type=int, default=10)
    ap.add_argument("--groups", type=int, default=2)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default="outputs/window_spectrum_diag")
    args = ap.parse_args()

    PairKFWindow.close_replay = _patched_close

    sys.argv = ["train_uci_link",
                "--arm", "2obs_aligned",
                "--data-dir", args.data_dir,
                "--gpu", str(args.gpu),
                "--calibrate", "--calib-groups", str(args.groups),
                "--kf-group-batches", "40",
                "--n-neighbors", str(args.n_neighbors),
                "--n-layers", str(args.n_layers),
                "--seed", str(args.seed),
                "--output", args.output]
    from scripts import train_uci_link  # noqa: E402 (patch lives in-process)
    train_uci_link.main()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "spectrum.json", "w") as f:
        json.dump(_spectrum_lines, f, indent=2)
    print("WROTE", out / "spectrum.json")


if __name__ == "__main__":
    main()
