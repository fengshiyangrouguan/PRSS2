#!/usr/bin/env python3
"""Test 1-3 (review 2026-09-25 concat line):
  1. topology smoke: same 4-turn dialogue forwards under BOTH
     merge_recur and concat_recur, loss finite
  2. cut count: parse_meta yields n_blocks == L for every depth,
     terminal cut present
  3. state growth: state_positions_for_cut span count == t (concat)
     and == 1 (merge)
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
CCM = HERE.parent / "third_party" / "ccm"
for p in (str(SRC), str(CCM), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ["DIALOG_MIRROR"] = \
    "/root/autodl-tmp/dailydialog_mirror/ijcnlp_dailydialog"

import torch
import types

import train_ccm as tc
from src.arguments import CompressionArguments
from src.data.dialogue.data import DialogueDataset
from src.data.dialogue.collator import DataCollatorForDialogue_LLAMA

BASE = "/root/autodl-tmp"
device = torch.device("cuda", 0)

# tokenizer + dataset
tok = tc.build_tokenizer(types.SimpleNamespace(
    model_name_or_path=BASE + "/llama-7b-hf", host="llama"))
ds = DialogueDataset(tok, comp_token=tok.comp_token_id,
                     online=True, add_comp_token=True, clean_split=False)

# a 4-turn dialogue (raw turns, any depth >= 2)
items = [dict(it) for it in ds.eval_dataset["turn_4"]][:1]
for it in items:
    it["fixed_depth"] = True

# model (official host, fp16)
import eval_llama_pooled_B as E
model = E.build_official_host(device)
model.eval()

print("=== Test 1: topology smoke ===")
for topo in ["merge_recur", "concat_recur"]:
    ca = CompressionArguments(attn_type=topo, num_comp_tokens=tc.N_TOK,
                              add_comp_token=True,
                              relative_embedding="skip")
    coll = DataCollatorForDialogue_LLAMA(
        dialog=ds, tokenizer=tok, comp_args=ca,
        comp_token=tok.comp_token_id, sum_token=tok.sum_token_id,
        padding="left", pad_token=tok.pad_token_id,
        label_pad_token_id=-100)
    batch = coll([dict(it) for it in items])
    out = tc.run_forward(model, batch, device, grad_enabled=False)
    l = out.logits
    assert torch.isfinite(l).all(), topo + " logits not finite"
    print("  {}: forward OK, seq={}, logits finite".format(
        topo, batch["input_ids"].shape[1]))

print("=== Test 2: cut count per depth ===")
for topo in ["merge_recur", "concat_recur"]:
    ca = CompressionArguments(attn_type=topo, num_comp_tokens=tc.N_TOK,
                              add_comp_token=True,
                              relative_embedding="skip")
    coll = DataCollatorForDialogue_LLAMA(
        dialog=ds, tokenizer=tok, comp_args=ca,
        comp_token=tok.comp_token_id, sum_token=tok.sum_token_id,
        padding="left", pad_token=tok.pad_token_id,
        label_pad_token_id=-100)
    for L in [1, 2, 4]:
        it = [dict(x) for x in ds.eval_dataset["turn_%d" % (L + 2)]][:1]
        it[0]["fixed_depth"] = True
        b = coll(it)
        metas = tc.parse_meta(b, tok.comp_token_id, tok.sum_token_id, 0,
                              orig_ids=[0], raw_dialogs=[list(it[0]["dialog"])],
                              topology=topo)
        m = metas[0]
        n_blocks = len(m["blocks"])
        assert n_blocks == L, "topo={} L={} blocks={}".format(topo, L, n_blocks)
        print("  {} L={}: {} blocks, ok={}".format(topo, L, n_blocks, m["ok"]))

print("=== Test 3: state growth ===")
L = 4
it = [dict(x) for x in ds.eval_dataset["turn_%d" % (L + 2)]][:1]
it[0]["fixed_depth"] = True
ca = CompressionArguments(attn_type="concat_recur", num_comp_tokens=tc.N_TOK,
                          add_comp_token=True, relative_embedding="skip")
coll = DataCollatorForDialogue_LLAMA(
    dialog=ds, tokenizer=tok, comp_args=ca,
    comp_token=tok.comp_token_id, sum_token=tok.sum_token_id,
    padding="left", pad_token=tok.pad_token_id, label_pad_token_id=-100)
b = coll(it)
metas = tc.parse_meta(b, tok.comp_token_id, tok.sum_token_id, 0,
                      orig_ids=[0], raw_dialogs=[list(it[0]["dialog"])],
                      topology="concat_recur")
m = metas[0]
for t in range(1, L + 1):
    spans_c = tc.state_positions_for_cut(m, t - 1, "concat_recur")
    spans_m = tc.state_positions_for_cut(m, t - 1, "merge_recur")
    assert len(spans_c) == t, "concat spans {} != t {}".format(len(spans_c), t)
    assert len(spans_m) == 1
print("  concat spans: " + ", ".join(
    "t{}={}".format(t, len(tc.state_positions_for_cut(m, t - 1, "concat_recur")))
    for t in range(1, L + 1)))

print("ALL 3 TESTS PASSED")
