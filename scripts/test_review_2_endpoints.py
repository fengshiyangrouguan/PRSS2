#!/usr/bin/env python3
"""Review test 2 (2026-09-21 ruling): the five endpoint depths.

L in {1, 2, 4, 8, 13}.  For each endpoint:

  - total turns = L + 2 (raw_dialog length);
  - parse_meta sees exactly L [C0 C1 S0 S1] blocks (the Gamma scan runs
    t = 1..L, one call per block);
  - collect_rows emits exactly L cuts (real DialogueCutBuilder + real
    Llmmaps depth buckets): 2 rows per t < L and the single terminal
    row (c, y, w = 1.0) for t = L.

Runs on CPU — no model, no tokenizer, no GPU.
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
from rpbe.llm.dialogue_records import DialogueCutBuilder, Llmmaps


class FakeAdapter:
    def __init__(self, z_dim):
        self.z_dim = z_dim
        self.n_calls = 0

    def extract_z(self, sum_positions):
        self.n_calls += 1
        return torch.zeros(1, self.z_dim)

    def clear(self):
        pass


class FakeEmb:
    """Returns constant sketches of the right dims (values irrelevant)."""
    def __init__(self, dim):
        self.dim = dim

    def __call__(self, embed_tokens, token_ids, tag=0):
        return torch.ones(token_ids.shape[0], self.dim)

    def combine(self, embed_tokens, token_ids_a, token_ids_b, tag=1):
        return torch.ones(token_ids_a.shape[0], self.dim)


def make_batch(L):
    """One row: L blocks of [C0 C1 S0 S1] + context + target tokens."""
    row = [101, 102, 201, 202] * L + [5, 6, 7, 8]
    prompt_len = 4 * L + 2
    batch = {
        "input_ids": torch.tensor([row]),
        "labels": torch.tensor([[-100] * prompt_len + [7, 8]]),
    }
    raw = [[10 + t] for t in range(L)] + [[5, 6], [7, 8]]
    assert len(raw) == L + 2
    return batch, [raw]


def main():
    z_dim = 8
    maps = Llmmaps(d_chi=64, d_phi=32, m=32, seed=0).to(torch.device("cpu"))
    utter = FakeEmb(64)
    phi = FakeEmb(32)

    for L in (1, 2, 4, 8, 13):
        batch, raw_dialogs = make_batch(L)
        metas = tc.parse_meta(batch, [101, 102], [201, 202], 0,
                              orig_ids=[0], raw_dialogs=raw_dialogs)
        meta = metas[0]
        assert meta["ok"], "block parse failed at L={}".format(L)
        assert meta["L"] == L, (meta["L"], L)
        assert meta["k"] == L + 1, (meta["k"], L)
        assert len(meta["blocks"]) == L, (len(meta["blocks"]), L)
        assert len(meta["raw_dialog"]) == L + 2, (
            len(meta["raw_dialog"]), L)

        adapter = FakeAdapter(z_dim)
        builder = DialogueCutBuilder(maps, z_dim=z_dim, seed=0)
        rows = tc.collect_rows(meta, adapter, builder, utter, phi,
                               None, batch, torch.device("cpu"))
        # one z extraction per compression layer, one cut per layer.
        assert adapter.n_calls == L, (adapter.n_calls, L)
        cut_turns = sorted({int(r.context["cut_turn"]) for r in rows})
        assert cut_turns == list(range(L)), (cut_turns, L)
        # 2 rows per t < L + the single terminal row (w = 1.0).
        assert len(rows) == 2 * L - 1, (len(rows), L)
        weights = [float(r.weight) for r in rows]
        assert weights.count(1.0) == 1, (weights, L)
        terminal = [r for r in rows if r.weight == 1.0][0]
        assert int(terminal.context["cut_turn"]) == L - 1, (
            terminal.context, L)
        print("L={:2d} PASS: {} turns, {} blocks, {} cuts, "
              "{} rows".format(L, L + 2, L, L, len(rows)))

    print("test 2 PASS: five endpoints L in {1,2,4,8,13} all correct")


if __name__ == "__main__":
    main()
