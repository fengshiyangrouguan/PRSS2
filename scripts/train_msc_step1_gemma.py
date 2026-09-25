#!/usr/bin/env python3
"""gemma MSC Step-1: default LoRA pretraining on the MSC train split
(user ruling 2026-09-25), output named gemma_MSC_no (llama-7b-no
convention).  Plain-text LM over episode utterances joined with the
official sep; 1000 steps, cosine, official LoRA r8/alpha16/dropout0.05.
"""
import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
CCM = HERE.parent / "third_party" / "ccm"
for p in (str(SRC), str(CCM), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
from torch.nn import functional as F

BASE = "/root/autodl-tmp"
SEP = "a\nA: "


def load_episodes(path, limit=0):
    rows = []
    with open(path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                rows.append(json.loads(ln))
    if limit:
        rows = rows[:limit]
    return rows


def episode_text(ep):
    parts = []
    for p in ep.get("init_personas", []):
        for t in p.get("text", []):
            parts.append(t)
    for s in ep.get("sessions", []):
        for p in s.get("personas", []):
            for t in p.get("text", []):
                parts.append(t)
        for d in s.get("dialogue", []):
            parts.append(d.get("text", ""))
    return SEP.join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--grad-accum", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--checkpoint-every", type=int, default=100)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-length", type=int, default=2048,
                    help="per-episode token cap (MSC episodes reach "
                         "~2.8k tokens; truncation keeps one forward on "
                         "a 40G card)")
    ap.add_argument("--resume-from", default="")
    a = ap.parse_args()

    random.seed(a.seed)
    torch.manual_seed(a.seed)
    device = torch.device("cuda", a.gpu)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer
    from transformers.models.gemma4.configuration_gemma4 import \
        Gemma4TextConfig
    from src.arch.ccm_gemma4 import Gemma4ForCausalLM_CCM
    from peft import LoraConfig
    from src import peft_custom

    tok = AutoTokenizer.from_pretrained(BASE + "/gemma-4-E4B")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    text_cfg = Gemma4TextConfig.from_pretrained(BASE + "/gemma-4-E4B")
    text_cfg.comp_relative_embedding = "skip"
    model = Gemma4ForCausalLM_CCM(text_cfg)
    from transformers.models.gemma4.modeling_gemma4 import (
        Gemma4ForConditionalGeneration)
    full = Gemma4ForConditionalGeneration.from_pretrained(
        BASE + "/gemma-4-E4B", torch_dtype=torch.bfloat16)
    prefix = "model.language_model."
    text_sd = {}
    for k, v in full.state_dict().items():
        if k.startswith(prefix):
            text_sd["model." + k[len(prefix):]] = v
        elif k == "lm_head.weight":
            text_sd[k] = v
    del full
    torch.cuda.empty_cache()
    model.load_state_dict(text_sd, strict=False)
    model.to(device, torch.bfloat16)

    cfg = LoraConfig(r=int(a.lora_r), lora_alpha=16,
                     lora_dropout=float(a.lora_dropout), bias="none",
                     task_type="CAUSAL_LM",
                     target_modules=["q_proj", "k_proj", "v_proj",
                                     "o_proj"])
    model = peft_custom.get_peft_model(model, cfg)
    for _p in model.parameters():
        _p.requires_grad_(False)
    for _n, _p in model.named_parameters():
        if "lora_" in _n:
            _p.requires_grad_(True)
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    try:
        model.gradient_checkpointing_enable()
        print("[msc-step1] gradient checkpointing ON", flush=True)
    except Exception:
        pass
    print("[msc-step1] trainable:", n_tr, flush=True)

    eps = load_episodes(
        BASE + "/third_party/ccm/dataset/msc/train.jsonl", a.limit)
    n_items = len(eps)
    print("[msc-step1] episodes:", n_items, flush=True)

    def encode(ep):
        ids = tok(episode_text(ep), add_special_tokens=True)["input_ids"]
        return ids

    def make_batch(items):
        seqs = [encode(it)[:a.max_length] for it in items]
        L = max(len(x) for x in seqs)
        pad = tok.pad_token_id
        ids = [[pad] * (L - len(x)) + x for x in seqs]
        labs = [[-100] * (L - len(x)) + x for x in seqs]
        return {
            "input_ids": torch.tensor(ids, device=device),
            "attention_mask": torch.ones((len(ids), L), device=device,
                                         dtype=torch.long),
            "labels": torch.tensor(labs, device=device)}

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr)
    warmup = int(0.03 * a.max_steps)
    total = a.max_steps

    def _lr(step):
        if step < warmup:
            return float(a.lr) * (step + 1) / warmup
        p = float(step - warmup) / float(max(1, total - warmup))
        return max(0.0, a.lr * 0.5 * (1.0 + math.cos(math.pi * p)))

    order = list(range(n_items))
    random.shuffle(order)
    pos = 0
    step = 0
    if a.resume_from:
        ck = torch.load(a.resume_from, map_location=device,
                        weights_only=False)
        model.load_state_dict(ck.get("model", ck), strict=False)
        step = int(ck.get("step", 0))
        pos = int(ck.get("cursor", 0))
        print("[msc-step1] resume step={} cursor={}".format(step, pos),
              flush=True)

    while step < a.max_steps:
        for g in opt.param_groups:
            g["lr"] = _lr(step)
        opt.zero_grad()
        acc_ce = 0.0
        acc_tok = 0
        for _ in range(a.grad_accum):
            items = []
            if pos >= n_items:
                random.shuffle(order)
                pos = 0
            items.append(eps[order[pos]])
            pos += 1
            batch = make_batch(items)
            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]).logits
            sh = logits[..., :-1, :].contiguous()
            sl = batch["labels"][..., 1:].contiguous()
            ce = F.cross_entropy(sh.view(-1, sh.shape[-1]),
                                 sl.reshape(-1), ignore_index=-100,
                                 reduction="sum")
            n_valid = int((sl != -100).sum())
            (ce / float(a.grad_accum)).backward()
            acc_ce += float(ce.detach())
            acc_tok += n_valid
        torch.nn.utils.clip_grad_norm_(params, a.grad_clip)
        opt.step()
        step += 1
        if step % a.log_every == 0 or step == a.max_steps:
            print("step={} ce_token={:.4f} lr={:.2e}".format(
                step, acc_ce / max(acc_tok, 1), _lr(step)), flush=True)
        if step % a.checkpoint_every == 0:
            torch.save({"model": {k: v.cpu() for k, v in
                                  model.state_dict().items()
                                  if "lora_" in k},
                        "step": step, "cursor": pos},
                       out / "checkpoint_step{}.pt".format(step))
    torch.save({"model": {k: v.cpu() for k, v in
                          model.state_dict().items()
                          if "lora_" in k},
                "step": step, "cursor": pos}, out / "final.pt")
    print("MSC_STEP1 DONE -> {}".format(out / "final.pt"), flush=True)


if __name__ == "__main__":
    main()
