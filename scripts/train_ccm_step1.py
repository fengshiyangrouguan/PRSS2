#!/usr/bin/env python3
"""Gemma-4 Step-1: default LoRA fine-tune (official two-stage protocol).

Strictly follows the official Step-1 recipe (paper Table 13/14 and the
official hydra config dialog/llama-7b.yaml) on the Gemma-4-E4B base:
  1000 steps, batch 128 (default; --micro-batch x --grad-accum), lr 3e-4,
  cosine schedule with warmup_ratio 0.03, LoRA on q/k/v/o_proj with
  r=8 / alpha=16 / dropout=0.05, grad clip 1.0.

The official hydra/Trainer entry (src/train.py) is NOT usable on
transformers 5.x (omegaconf 2.3 cannot parse the new TrainingArguments
trackio fields), so this self-contained loop reproduces the recipe
semantics bit-for-bit: same data construction (official sample_dialog
plain-text layout, random-k truncation, last-turn CE), same scheduler
math (cosine with warmup_ratio), same LoRA config.  Mixed precision is
bf16 (gemma weights are natively bf16; the official FP16 is llama
specific).

Stage-1 output: adapter-only checkpoints + a merged foundation
(full text-only weights) under --output/foundation.pt for the Step-2
--foundation path.

Usage:
    python -u train_ccm_step1.py \
        --model-name-or-path /root/autodl-tmp/gemma-4-E4B \
        --dialog-mirror /root/autodl-tmp/dailydialog_mirror/ijcnlp_dailydialog \
        --output /root/autodl-tmp/outputs/ccm_gemma4/seed0_gemma4_step1
"""
import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, "/root/autodl-tmp")
sys.path.insert(0, "/root/autodl-tmp/third_party/ccm")
sys.path.insert(0, "/root/autodl-tmp/src")
sys.path.insert(0, "/root/autodl-tmp/scripts")

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence


# ---------------------------------------------------------------------
# tokenizer
# ---------------------------------------------------------------------
def build_tokenizer(model_path):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path)
    # Gemma4 ships without a pad token; config default pad id is 0
    # (distinct from eos=1/bos=2, keeps EOS exclusion semantics clean).
    if tok.pad_token_id is None:
        tok.pad_token_id = 0
        tok.pad_token = "<pad>"
    tok.padding_side = "left"
    return tok


# ---------------------------------------------------------------------
# model: text-only load + official Step-1 LoRA (dropout 0.05)
# ---------------------------------------------------------------------
def build_model(model_path, device):
    import torch as _t
    from transformers.models.gemma4.configuration_gemma4 import (
        Gemma4TextConfig)
    from transformers.models.gemma4.modeling_gemma4 import (
        Gemma4ForConditionalGeneration)
    from src.arch.ccm_gemma4 import Gemma4ForCausalLM_CCM
    from src import peft_custom
    from peft import LoraConfig

    text_cfg = Gemma4TextConfig.from_pretrained(model_path)
    model = Gemma4ForCausalLM_CCM(text_cfg)
    full = Gemma4ForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16)
    prefix = "model.language_model."
    text_sd = {}
    for k, v in full.state_dict().items():
        if k.startswith(prefix):
            text_sd["model." + k[len(prefix):]] = v
        elif k == "lm_head.weight":
            text_sd[k] = v
    del full
    _t.cuda.empty_cache() if device.type == "cuda" else None
    missing, unexpected = model.load_state_dict(text_sd, strict=False)
    print("[step1] text-only load: {} missing / {} unexpected".format(
        len(missing), len(unexpected)), flush=True)
    if unexpected:
        raise RuntimeError("unexpected keys: {}".format(unexpected[:8]))
    model.to(device, torch.bfloat16)  # cast persistent=False buffers too (embed_scale fp32 -> bf16, G1 parity)

    # Official Table 14 LoRA (r8/alpha16/dropout0.05, q/k/v/o_proj).
    lora_cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                          bias="none", task_type="CAUSAL_LM",
                          target_modules=["q_proj", "k_proj", "v_proj",
                                          "o_proj"])
    model = peft_custom.get_peft_model(model, lora_cfg)
    trainable = [n for n, p in model.named_parameters()
                 if p.requires_grad]
    print("[step1] trainable params: {} ({})".format(
        sum(p.numel() for n, p in model.named_parameters()
            if p.requires_grad), trainable[:4]), flush=True)
    return model


# ---------------------------------------------------------------------
# data: official plain-text layout WITHOUT comp tokens
# ---------------------------------------------------------------------
def build_data(tokenizer, mirror):
    from src.data.dialogue.gemma4_data import (
        Gemma4DialogueDataset, _read_mirror, _preprocess)
    ds = Gemma4DialogueDataset(tokenizer, mirror=mirror)
    return ds


def sample_no_comp(ds, item, seed_rng):
    """Official sample_dialog layout with comp_ids=() (default LoRA
    stage): bos + history turns each + sep + context turn + sep;
    target = last turn + eos.  random-k truncation, train only."""
    dialog = item["dialog"]
    if item["is_train"]:
        k = int(seed_rng.integers(3, len(dialog) + 1))
        dialog = dialog[:k]
    prompt = []
    for i in range(len(dialog) - 2):
        prompt += list(dialog[i]) + list(ds.sep_token)
    prompt += list(dialog[-2]) + list(ds.sep_token)
    prompt = list(ds.bos_token) + prompt
    target = list(dialog[-1]) + list(ds.eos_token)
    return prompt, target


def collate_batch(ds, items, pad_token_id, rng):
    inputs, labels = [], []
    for item in items:
        p, t = sample_no_comp(ds, item, rng)
        inputs.append(torch.tensor(p + t))
        labels.append(torch.tensor([-100] * len(p) + t))
    input_ids = pad_sequence(inputs, batch_first=True,
                             padding_value=pad_token_id)
    labels = pad_sequence(labels, batch_first=True,
                          padding_value=-100)
    return {"input_ids": input_ids,
            "attention_mask": (input_ids != pad_token_id).long(),
            "labels": labels}


# ---------------------------------------------------------------------
# main loop
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name-or-path", required=True)
    ap.add_argument("--dialog-mirror", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--micro-batch", type=int, default=2,
                    help="dialogues per forward (official batch 128 = "
                         "micro x accum)")
    ap.add_argument("--grad-accum", type=int, default=128,
                    help="gradient accumulation steps "
                         "(2 x 128 = 256 macro, 2x official per user "
                         "ruling 2026-09-20)")
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--checkpoint-every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--resume-from", default="",
                    help="adapter checkpoint to resume from")
    a = ap.parse_args()

    device = torch.device("cuda", a.gpu)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    rng = np.random.default_rng(a.seed)
    os.makedirs(a.output, exist_ok=True)

    tok = build_tokenizer(a.model_name_or_path)
    ds = build_data(tok, a.dialog_mirror)
    model = build_model(a.model_name_or_path, device)

    train_items = ds.trainset
    n_items = len(train_items)
    print("[step1] train dialogues: {}".format(n_items), flush=True)

    opt_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(opt_params, lr=a.lr,
                                  betas=(0.9, 0.999), eps=1e-8)

    # cosine with warmup_ratio (official dialog/llama-7b.yaml)
    total = a.max_steps
    warmup_steps = int(total * a.warmup_ratio)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # resume
    if a.resume_from:
        ck = torch.load(a.resume_from, map_location=device,
                        weights_only=False)
        sd = ck.get("trainable_state_dict", ck)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print("[step1] resume from {}: {} missing / {} unexpected".format(
            a.resume_from, len(missing), len(unexpected)), flush=True)

    def save_ckpt(step):
        path = os.path.join(a.output, "checkpoint_step{}.pt".format(step))
        tsd = {n: p.data for n, p in model.named_parameters()
               if p.requires_grad}
        torch.save({"step": step, "trainable_state_dict": tsd}, path)
        print("[step1] saved {}".format(path), flush=True)

    # official Trainer-style: step counts GLOBAL steps (each step does
    # grad_accum microbatches regardless of AMP skips); warmup on steps.
    step = 0
    t_start = time.time()
    while step < total:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        n_tok_accum = 0
        for _mb in range(a.grad_accum):
            idxs = rng.integers(0, n_items, size=a.micro_batch)
            items = [train_items[int(i)] for i in idxs]
            batch = collate_batch(ds, items, tok.pad_token_id, rng)
            n_tok = int((batch["labels"] != -100).sum())
            with torch.autocast(device_type="cuda",
                                dtype=torch.bfloat16):
                out = model(input_ids=batch["input_ids"].to(device),
                            attention_mask=batch["attention_mask"]
                            .to(device),
                            labels=batch["labels"].to(device))
                loss = out.loss / a.grad_accum
            loss.backward()
            loss_accum += float(loss.detach()) * a.grad_accum
            n_tok_accum += n_tok
        torch.nn.utils.clip_grad_norm_(opt_params, a.grad_clip)
        optimizer.step()
        scheduler.step()
        step += 1

        if step % a.log_every == 0 or step == 1:
            nll = loss_accum * a.grad_accum / max(n_tok_accum, 1)
            print("[step1] step {}/{} loss={:.4f} tok={} lr={:.2e} "
                  "({:.0f}s)".format(
                      step, total, loss_accum, n_tok_accum,
                      scheduler.get_last_lr()[0], time.time() - t_start),
                  flush=True)
        if step % a.checkpoint_every == 0 or step == total:
            save_ckpt(step)

    # Adapter-only artifact: Step-2 loads it with
    # load_lora_weight(..., merge=True) the way the Llama line merges
    # llama-7b-no into the base (official "adapters are then merged").
    model.eval()
    print("[step1] DONE: adapter at {}".format(a.output), flush=True)
    save_ckpt("final")
    json.dump({"seed": a.seed, "cli": vars(a)},
              open(os.path.join(a.output, "config.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
