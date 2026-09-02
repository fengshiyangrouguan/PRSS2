#!/usr/bin/env python3
"""L6.5 gate 2: one-step parity between the official HF Trainer protocol
and the train_ccm single-pass loop (ccm_merge arm).

Same model init, same 128 pre-collated microbatches, same recipe:
AdamW(weight_decay=0), fp16 autocast + GradScaler, mean-CE / 128
accumulation, max_grad_norm=1.0.  warmup_ratio=0 pins the scheduler at
full lr for this single step (the warmup curve itself is formula-level
identical in both paths).  Batch ORDER is irrelevant: the 128 batches
are pre-collated from one shared RNG stream and Adam starts from zero
state, so the per-parameter update deltas must match.

Usage (cloud):
    python -m scripts.ccm_parity --model-name-or-path /root/autodl-tmp/llama-7b-hf \
        --dialog-mirror /root/autodl-tmp/dailydialog_mirror/ijcnlp_dailydialog \
        --out outputs/ccm_parity
"""

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np
import torch

torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
CCM = HERE.parent / "third_party" / "ccm"
for p in (str(SRC), str(CCM)):
    if p not in sys.path:
        sys.path.insert(0, p)

from scripts.train_ccm import (build_dataset, build_model, build_tokenizer,
                               run_forward, task_ce_sum, wrap_lora)


def parse_args():
    p = argparse.ArgumentParser("CCM one-step parity check (L6.5 gate 2)")
    p.add_argument("--model-name-or-path", required=True)
    p.add_argument("--dialog-mirror", required=True)
    p.add_argument("--out", default="outputs/ccm_parity")
    p.add_argument("--n-microbatches", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--relative-embedding", default="skip",
                   choices=["skip", "base"])
    return p.parse_args()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available()
                          else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- shared data stream: pre-collate N microbatches once ----
    tokenizer = build_tokenizer(args)
    dialog, collator = build_dataset(args, tokenizer)
    items = dialog.train_dataset
    batches = [collator([items[i % len(items)]])
               for i in range(args.n_microbatches)]

    class BatchList:
        def __len__(self):
            return len(batches)

        def __getitem__(self, i):
            return batches[i]

    def identity_collator(list_of_batches):
        return list_of_batches[0]

    # ---- arm A: official HF Trainer ----
    seed_all(args.seed)
    model_a = wrap_lora(build_model(args, device), r=8)
    model_a.update_comp_token([32000, 32001], [32002, 32003])
    params_a = [p for p in model_a.parameters() if p.requires_grad]
    before_a = {n: p.detach().float().cpu().clone()
                for n, p in model_a.named_parameters() if p.requires_grad}

    from transformers import Trainer, TrainingArguments
    trainer = Trainer(
        model=model_a,
        args=TrainingArguments(
            output_dir=str(out / "trainer"),
            per_device_train_batch_size=1,
            gradient_accumulation_steps=args.n_microbatches,
            max_steps=1,
            learning_rate=args.lr,
            weight_decay=0.0,
            lr_scheduler_type="cosine",
            warmup_ratio=0.0,
            fp16=(device.type == "cuda"),
            fp16_full_eval=False,
            max_grad_norm=1.0,
            seed=args.seed,
            logging_steps=1,
            report_to=[],
            remove_unused_columns=False,
            dataloader_drop_last=True,
        ),
        train_dataset=BatchList(),
        data_collator=identity_collator,
    )
    trainer.train()
    deltas_a = {}
    for n, p in model_a.named_parameters():
        if p.requires_grad:
            deltas_a[n] = (p.detach().float().cpu() - before_a[n])

    # ---- arm B: the train_ccm single-pass loop ----
    seed_all(args.seed)
    model_b = wrap_lora(build_model(args, device), r=8)
    model_b.update_comp_token([32000, 32001], [32002, 32003])
    params_b = [p for p in model_b.parameters() if p.requires_grad]
    before_b = {n: p.detach().float().cpu().clone()
                for n, p in model_b.named_parameters() if p.requires_grad}
    optimizer_b = torch.optim.AdamW(params_b, lr=args.lr, weight_decay=0.0)
    scaler_b = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    optimizer_b.zero_grad(set_to_none=True)
    for b in batches:
        out = run_forward(model_b, b, device, grad_enabled=True)
        task_sum, n_valid = task_ce_sum(out, b["labels"], device)
        task_mean = task_sum / max(n_valid, 1)
        scaler_b.scale(task_mean / float(len(batches))).backward()
    scaler_b.unscale_(optimizer_b)
    torch.nn.utils.clip_grad_norm_(params_b, 1.0)
    scaler_b.step(optimizer_b)
    scaler_b.update()
    deltas_b = {}
    for n, p in model_b.named_parameters():
        if p.requires_grad:
            deltas_b[n] = (p.detach().float().cpu() - before_b[n])

    # ---- comparison ----
    names = sorted(set(deltas_a) & set(deltas_b))
    assert names, "no common trainable parameters"
    missing = (set(deltas_a) ^ set(deltas_b))
    worst = None
    for n in names:
        da, db = deltas_a[n], deltas_b[n]
        abs_diff = float((da - db).abs().max())
        scale = float(da.abs().max())
        rel = abs_diff / max(scale, 1e-30)
        if worst is None or rel > worst[1]:
            worst = (n, rel, abs_diff, scale)
    report = {
        "n_params": len(names),
        "missing_params": sorted(missing),
        "worst_param": worst[0],
        "worst_rel_diff": worst[1],
        "worst_abs_diff": worst[2],
        "param_scale": worst[3],
        "pass": worst[1] < 0.05 and not missing,
    }
    (out / "parity.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
