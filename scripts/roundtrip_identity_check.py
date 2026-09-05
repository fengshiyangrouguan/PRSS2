#!/usr/bin/env python3
"""Review 2026-09-05 (most necessary verification): the REAL save/load
path must reconstruct the SAME model — foundation merge + LoRA + Gamma +
frozen COMP/SUM rows.  Compare logits on identical inputs before save
and after rebuild; any mismatch means eval results are not trustworthy.

Usage: python scripts/roundtrip_identity_check.py
Requires the same env as train (LLAMA2 dirs on disk).
"""
import os, sys, types
sys.path.insert(0, "/root/autodl-tmp")
sys.path.insert(0, "/root/autodl-tmp/src")
sys.path.insert(0, "/root/autodl-tmp/third_party/ccm")

import torch
import train_ccm as tc

device = torch.device("cuda", 0)
ckpt_path = "/root/autodl-tmp/roundtrip_probe.pt"
foundation = "/root/autodl-tmp/result/dialog/llama-7b-no"
model_name = "/root/autodl-tmp/llama-7b-hf"


def build_fresh():
    args = types.SimpleNamespace(arm="ours", model_name_or_path=model_name,
                                 relative_embedding="skip", lora_r=8,
                                 gamma_hidden=64)
    model = tc.build_model(args, device)
    from src.model import load_lora_weight
    load_lora_weight(foundation, model, merge=True)
    model = tc.wrap_lora(model, args.lora_r)
    model.update_comp_token([32000 + k for k in range(tc.N_TOK)],
                            [32000 + tc.N_TOK + k for k in range(tc.N_TOK)])
    tc.attach_gamma(model, hidden=args.gamma_hidden)
    return model


def probe_logits(model):
    """Small fixed input exercising comp tokens + a normal prefix."""
    tok = model.config.pad_token_id or 0
    comp_ids = [32000, 32001]
    ids = torch.tensor([[tok, tok, tok, comp_ids[0], comp_ids[1], tok, tok]],
                       device=device)
    mask = torch.ones_like(ids)
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=mask)
    return out.logits[0].detach().clone()


print("== build A ==", flush=True)
model_a = build_fresh()
logits_a = probe_logits(model_a)

dummy = torch.optim.AdamW([p for p in model_a.parameters()
                           if p.requires_grad], lr=1e-3)
tc.save_trainable(ckpt_path, model_a, step=1)
print("saved:", ckpt_path, flush=True)
del model_a
torch.cuda.empty_cache()

print("== rebuild B (save -> load path) ==", flush=True)
model_b = build_fresh()
payload = tc.load_trainable(ckpt_path, model_b, dummy, device)
assert "new_token_rows" in payload, "new_token_rows missing!"
n = 2 * tc.N_TOK
model_b.get_input_embeddings().weight[-n:] = \
    payload["new_token_rows"]["input_embed_rows"].to(device)
model_b.lm_head.weight[-n:] = \
    payload["new_token_rows"]["lm_head_rows"].to(device)
logits_b = probe_logits(model_b)

diff = (logits_a - logits_b).abs()
print("max abs logit diff: {:.3e}".format(diff.max().item()), flush=True)
print("mean abs logit diff: {:.3e}".format(diff.mean().item()), flush=True)
if diff.max().item() < 1e-4:
    print("ROUNDTRIP_IDENTITY_OK", flush=True)
else:
    print("ROUNDTRIP_MISMATCH", flush=True)
    raise SystemExit(1)
