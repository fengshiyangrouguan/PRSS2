#!/usr/bin/env python3
"""Review test 1 (2026-09-21 ruling): artificial 5-turn dialogue.

L = 3 (3 history + context + target).  Verifies the cut collector:

  - parse_meta reports L = 3 with 3 [C0 C1 S0 S1] blocks;
  - collect_rows extracts z_t from block_idx t-1 for t = 1, 2, 3;
  - chi/phi come from the RAW tokenized turns (u_{t+1}, u_{t+2} for the
    2Obs pair; (c, y) for the terminal cut), never from spans of the
    collated ids.

Runs on CPU with fake adapter / fake embedding sketches — no model, no
tokenizer, no GPU.
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
CCM = HERE.parent / "third_party" / "ccm"
for p in (str(SRC), str(CCM)):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch

import train_ccm as tc


class FakeAdapter:
    def __init__(self):
        self.calls = []

    def extract_z(self, sum_positions):
        self.calls.append([tuple(int(x) for x in r)
                           for r in sum_positions.cpu().tolist()])
        return torch.zeros(1, 8)

    def clear(self):
        pass


class FakeEmb:
    """chi sketch that leaks its input: sum of token ids (dim 64)."""
    def __call__(self, embed_tokens, token_ids, tag=0):
        s = token_ids.float().sum(-1, keepdim=True)
        return s.expand(token_ids.shape[0], 64)

    def combine(self, embed_tokens, token_ids_a, token_ids_b, tag=1):
        s = (token_ids_a.float().sum(-1)
             + 10.0 * token_ids_b.float().sum(-1))
        return s.unsqueeze(0).expand(token_ids_a.shape[0], 64)


class FakeBuilder:
    def __init__(self):
        self.calls = []  # (v, chi1_sum, chi2_sum, phi1_sum, phi2_sum)

    def build(self, dm, z_v, chi_1, chi_2, phi_1, phi_2, stats=None,
              v=None, skip_context_obs=True):
        self.calls.append((int(v), float(chi_1.sum()), float(chi_2.sum()),
                           float(phi_1.sum()), float(phi_2.sum())))
        return []


def main():
    comp_ids = [101, 102]
    sum_ids = [201, 202]
    # 3 history turns (each followed by [C0 C1 S0 S1]) + context + target.
    row = ([101, 102, 201, 202] * 3) + [5, 6, 7, 8]
    batch = {
        "input_ids": torch.tensor([row]),
        "labels": torch.tensor([[-100] * 12 + [7, 8]]),
    }
    # raw tokenized turns: [u_1, u_2, u_3, c, y] — ids leak through the
    # fake sketch, so the chi source is identifiable.
    raw_dialogs = [[[1], [2, 2], [3], [5, 6], [7, 8]]]

    metas = tc.parse_meta(batch, comp_ids, sum_ids, 0, orig_ids=[0],
                          raw_dialogs=raw_dialogs)
    assert len(metas) == 1
    meta = metas[0]
    assert meta["ok"], "block parse failed"
    assert meta["L"] == 3, meta["L"]
    assert meta["k"] == 4, meta["k"]
    assert meta["blocks"] == [(0, 2), (4, 6), (8, 10)], meta["blocks"]
    assert len(meta["raw_dialog"]) == 5, len(meta["raw_dialog"])

    adapter = FakeAdapter()
    builder = FakeBuilder()
    utter = FakeEmb()
    phi = FakeEmb()
    rows = tc.collect_rows(meta, adapter, builder, utter, phi,
                           None, batch, torch.device("cpu"))

    # z_t extracted from block_idx t-1 for t = 1, 2, 3 (S0, S0+1).
    assert adapter.calls == [[(2, 3)], [(6, 7)], [(10, 11)]], \
        adapter.calls
    # three cuts, block indices 0, 1, 2.
    assert [c[0] for c in builder.calls] == [0, 1, 2], builder.calls

    # t = 1 (v=0): chi1 = u_2 = [2,2] (sum 4), phi1 = u_3 = [3] (sum 3),
    # chi2 = combine(u_2, u_3) = 4 + 10*3 = 34, phi2 = y = [7,8] (15).
    # (FakeEmb expands to dim 64, so the recorded sums are 64x the id sum.)
    v0 = builder.calls[0]
    assert (v0[1], v0[2], v0[3], v0[4]) == (256.0, 2176.0, 192.0, 960.0), v0
    # t = 2 (v=1): chi1 = u_3 (3), phi1 = u_4 = c (11),
    # chi2 = 3 + 10*11 = 113, phi2 = y (15).
    v1 = builder.calls[1]
    assert (v1[1], v1[2], v1[3], v1[4]) == (192.0, 7232.0, 704.0, 960.0), v1
    # t = 3 (v=2, TERMINAL): chi = c = [5,6] (11), phi = y (15);
    # chi2/phi2 are the same objects (unused on the single-row path).
    v2 = builder.calls[2]
    assert (v2[1], v2[2], v2[3], v2[4]) == (704.0, 704.0, 960.0, 960.0), v2

    print("test 1 PASS: L=3 -> blocks 0/1/2, terminal cut (c, y), "
          "chi/phi from raw turns")


if __name__ == "__main__":
    main()
