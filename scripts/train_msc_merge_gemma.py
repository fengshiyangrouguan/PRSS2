#!/usr/bin/env python3
"""gemma MSC merge (official adapter training) — the official msc recipe
(1000 steps, save every 50, lr 3e-4 cosine + 3% warmup, macro 128) on
the gemma host, from the aligned gemma_MSC_no foundation.  Mirrors the
llama official run.py msc path (merge_recur, online comp tokens) with
the gemma merge chain (build_model_merge + wrap_lora_merge, bf16,
seq cap 1024 for the 40GB card).

--gpus "0,1" enables manual data parallelism: the 128 accumulation
micro-batches are split across the two cards (64 each), gradients are
averaged across cards before the optimizer step (mathematically equal
to single-card accumulation up to float rounding; the two cards see
independent dropout masks, standard DDP behavior).  The optimizer
steps on the FIRST gpu and weights are synced back to the others.
"""
import argparse
import json
import math
import os
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (str(HERE), str(HERE.parent / "src"),
          str(HERE.parent / "third_party" / "ccm")):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
from torch.nn import functional as F

import train_ccm as tc
import train_ccm_merge as tcm

BASE = "/root/autodl-tmp"


def build_model(a, device):
    import types as _t
    args = _t.SimpleNamespace(host="gemma4",
                              model_name_or_path=BASE + "/gemma-4-E4B",
                              relative_embedding="skip", foundation="")
    tok = tc.build_tokenizer(args)
    model = tcm.build_model_merge(args, device)
    ck = torch.load(a.foundation, map_location=device, weights_only=False)
    tsd = ck["model"]
    n_merged = 0
    with torch.no_grad():
        for k in list(tsd):
            if ".lora_A." not in k:
                continue
            b_key = k.replace(".lora_A.", ".lora_B.")
            base_key = re.sub(
                r"^base_model\.model\.model\.layers\.(\d+)\."
                r"self_attn\.(\w+_proj)\.lora_A\.default\.weight$",
                r"model.layers.\1.self_attn.\2.weight", k)
            if base_key == k:
                continue
            A = tsd[k].float().to(device)
            B = tsd[b_key].float().to(device)
            delta = (B @ A) * (16.0 / 8.0)
            tgt = dict(model.named_parameters())[base_key]
            tgt.data += delta.to(tgt.dtype)
            n_merged += 1
    print("[msc-merge] gpu{} foundation merged: {} LoRA modules".format(
        device.index, n_merged), flush=True)
    model = tcm.wrap_lora_merge(model, a.lora_r, a.lora_dropout)
    model.update_comp_token(
        [tok.comp_token_id[k] for k in range(tc.N_TOK)],
        [tok.sum_token_id[k] for k in range(tc.N_TOK)])
    return tok, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--gpus", default="0",
                    help='comma list, e.g. "0,1" for dual-card data '
                         "parallel")
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--grad-accum", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--checkpoint-every", type=int, default=50)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--foundation",
                    default=BASE + "/result/msc/gemma_MSC_no/final.pt")
    ap.add_argument("--msc-dir",
                    default=BASE + "/third_party/ccm/dataset/msc")
    ap.add_argument("--resume-from", default="")
    a = ap.parse_args()

    gpu_ids = [int(x) for x in a.gpus.split(",")]
    n_gpu = len(gpu_ids)
    assert a.grad_accum % n_gpu == 0, "grad-accum must divide by n_gpu"
    per_gpu = a.grad_accum // n_gpu

    random.seed(a.seed)
    torch.manual_seed(a.seed)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)

    tok, model0 = build_model(a, torch.device("cuda", gpu_ids[0]))
    models = [model0]
    for g in gpu_ids[1:]:
        _, m = build_model(a, torch.device("cuda", g))
        models.append(m)

    from src.arguments import CompressionArguments
    from src.data.dialogue.data_msc import DialogueDataset
    from src.data.dialogue.collator import DataCollatorForDialogue_LLAMA
    ds = DialogueDataset(tok, comp_token=tok.comp_token_id, online=True,
                         add_comp_token=True, eval_source="val",
                         max_length=a.max_length, msc_dir=a.msc_dir)
    comp_args = CompressionArguments(attn_type="merge_recur",
                                     num_comp_tokens=tc.N_TOK,
                                     add_comp_token=True,
                                     relative_embedding="skip")
    collator = DataCollatorForDialogue_LLAMA(
        dialog=ds, tokenizer=tok, comp_args=comp_args,
        comp_token=tok.comp_token_id, sum_token=tok.sum_token_id,
        padding="left", pad_token=tok.pad_token_id,
        label_pad_token_id=-100)

    rows = list(ds.train_dataset)
    n_items = len(rows)
    print("[msc-merge] train rows:", n_items, flush=True)

    params = [p for p in model0.parameters() if p.requires_grad]
    other_params = [[p for p in m.parameters() if p.requires_grad]
                    for m in models[1:]]
    print("[msc-merge] trainable:", sum(p.numel() for p in params),
          "on", n_gpu, "gpu(s)", flush=True)
    opt = torch.optim.AdamW(params, lr=a.lr)
    warmup = max(1, int(0.03 * a.max_steps))
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
        rck = torch.load(a.resume_from, map_location="cpu",
                         weights_only=False)
        model0.load_state_dict(rck["model"], strict=False)
        step = int(rck.get("step", 0))
        pos = int(rck.get("cursor", 0))
        for m, op in zip(models[1:], other_params):
            for p, p0 in zip(op, params):
                p.data.copy_(p0.data)
        print("[msc-merge] resume step={} cursor={}".format(step, pos),
              flush=True)

    def micro(model, dev, row):
        batch = collator([row])
        out_f = tc.run_forward(model, batch, dev, grad_enabled=True)
        logits = out_f.logits
        labs = batch["labels"].to(dev)
        sh = logits[..., :-1, :].contiguous()
        sl = labs[..., 1:].contiguous()
        ce = F.cross_entropy(sh.view(-1, sh.shape[-1]),
                             sl.reshape(-1), ignore_index=-100,
                             reduction="sum")
        n_valid = (sl != -100).sum()
        (ce / float(a.grad_accum)).backward()
        # NO float()/int() here — those force GPU sync and serialize the
        # two cards.  Detached tensors are summed at the step end.
        return ce.detach(), n_valid.detach()

    def draw_row():
        nonlocal pos
        if pos >= n_items:
            random.shuffle(order)
            pos = 0
        row = dict(rows[order[pos]])
        pos += 1
        return row

    while step < a.max_steps:
        for g in opt.param_groups:
            g["lr"] = _lr(step)
        opt.zero_grad()
        for op in other_params:
            for p in op:
                p.grad = None
        acc_ce = 0.0
        acc_tok = 0
        # interleave: enqueue one micro-batch per card alternately so
        # the two GPUs run their async kernels concurrently
        ce_t = []
        for _ in range(per_gpu):
            ce_t.append(micro(model0, torch.device("cuda", gpu_ids[0]),
                              draw_row()))
            for gi, m in enumerate(models[1:]):
                ce_t.append(micro(m, torch.device("cuda", gpu_ids[1 + gi]),
                                  draw_row()))
        acc_ce += sum(float(t[0]) for t in ce_t)
        acc_tok += sum(int(t[1]) for t in ce_t)
        for op in other_params:
            # gradient add onto GPU0 (one sync, at the step end)
            for p0, p in zip(params, op):
                p0.grad.add_(p.grad.to(p0.device))
                p.grad = None
        # average: sum of the two halves is the full gradient; each
        # micro-batch was already scaled by 1/grad_accum, so the sum
        # equals the single-card gradient exactly.
        torch.nn.utils.clip_grad_norm_(params, a.grad_clip)
        opt.step()
        for m, op in zip(models[1:], other_params):
            for p0, p in zip(params, op):
                p.data.copy_(p0.data)
        step += 1
        if step % a.log_every == 0 or step == a.max_steps:
            print("step={} ce_token={:.4f} lr={:.2e}".format(
                step, acc_ce / max(acc_tok, 1), _lr(step)), flush=True)
        if step % a.checkpoint_every == 0:
            torch.save({"model": {k: v.cpu() for k, v in
                                  model0.state_dict().items()
                                  if ("lora_" in k)
                                  or ("comp_embeddings" in k)},
                        "step": step, "cursor": pos},
                       out / "checkpoint_step{}.pt".format(step))
            print("[msc-merge] saved step {}".format(step), flush=True)
    torch.save({"model": {k: v.cpu() for k, v in
                          model0.state_dict().items()
                          if ("lora_" in k)
                          or ("comp_embeddings" in k)},
                "step": step, "cursor": pos}, out / "final.pt")
    print("MSC_MERGE_DONE -> {}".format(out / "final.pt"), flush=True)


if __name__ == "__main__":
    main()
