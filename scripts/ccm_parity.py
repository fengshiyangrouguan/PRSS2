#!/usr/bin/env python3
"""L6.5 gate 2: one-step parity between the official HF Trainer protocol
and the train_ccm single-pass loop (ccm_merge arm).

Same model init, same 128 pre-collated microbatches, same recipe:
AdamW(weight_decay=0), fp16 autocast + GradScaler, per-microbatch
mean-CE divided by the accumulation count (accelerate 1.14 does
`loss = loss / gradient_accumulation_steps` inside Accelerator.backward,
so the official gradient is the sum of loss_b / 128), max_grad_norm=1.0.
The LR schedule is pinned CONSTANT (full lr) on both sides: replicating
the warmup/cosine formula inside a single-step parity is a
scheduler-phase trap (train_ccm's LambdaLR with max_steps=1 pins
warmup_steps=max(1,0)=1, so its first optimizer step would run at lr=0
while the Trainer arm runs at full lr).  The scheduler curve itself is
formula-level identical in both paths and verified by review; what this
gate verifies is the optimizer / scaling / clipping / accumulation
protocol.  Batch ORDER is irrelevant: the 128 batches are pre-collated
from one shared RNG stream and Adam starts from zero state, so the
per-parameter update deltas must match.

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
                               run_forward, wrap_lora)


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
            lr_scheduler_type="constant",
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
    # Probe: did the Trainer's GradScaler see inf/nan (skip) and what
    # is its growth factor after the step?
    scaler_a = getattr(trainer, "scaler", None)
    if scaler_a is not None and device.type == "cuda":
        print("PROBE armA scaler scale={:.1f} growth={} "
              "found_inf={}".format(
                  float(scaler_a.get_scale()),
                  "nan" if scaler_a._growth_factor != scaler_a._growth_factor
                  else float(scaler_a._growth_factor),
                  "nan" if scaler_a._found_inf_per_device
                  != scaler_a._found_inf_per_device else
                  {k: bool(v) for k, v in
                   scaler_a._found_inf_per_device.items()}),
              flush=True)
    deltas_a = {}
    for n, p in model_a.named_parameters():
        if p.requires_grad:
            deltas_a[n] = (p.detach().float().cpu() - before_a[n])
    n_nan_a = sum(1 for d in deltas_a.values()
                  if torch.isnan(d.float()).any())
    print(f"PROBE armA delta_nan_params={n_nan_a}/"
          f"{len(deltas_a)} nonzero="
          f"{sum(1 for d in deltas_a.values() if float(d.abs().max()) > 0)}",
          flush=True)

    # ---- arm B: the train_ccm single-pass loop ----
    seed_all(args.seed)
    model_b = wrap_lora(build_model(args, device), r=8)
    model_b.update_comp_token([32000, 32001], [32002, 32003])
    params_b = [p for p in model_b.parameters() if p.requires_grad]
    before_b = {n: p.detach().float().cpu().clone()
                for n, p in model_b.named_parameters() if p.requires_grad}
    # No scheduler: LR stays constant at full value, matching the
    # Trainer arm's "constant" type (see module docstring).
    optimizer_b = torch.optim.AdamW(params_b, lr=args.lr, weight_decay=0.0)
    scaler_b = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    optimizer_b.zero_grad(set_to_none=True)
    for b in batches:
        fwd_out = run_forward(model_b, b, device, grad_enabled=True)
        # Official Trainer loss, replicated exactly: the vendored model
        # shifts logits/labels by one before the CE and reduces with the
        # MEAN over valid tokens (ccm_llama.py ~line 922).  The /128 is
        # NOT a free choice: accelerate 1.14's Accelerator.backward()
        # does `loss = loss / gradient_accumulation_steps` before
        # scaling, so the official gradient is sum of (loss_b / 128).
        # Omitting it overflows the fp16 backward (caught here: 220/256
        # nan grads -> GradScaler skips the step) and shifts the update
        # scale by 128x.
        labels_b = b["labels"].to(device)
        sl = fwd_out.logits[..., :-1, :].contiguous()
        sy = labels_b[..., 1:].contiguous()
        # Loss under autocast too: the official Trainer computes the CE
        # inside its autocast context, which affects accumulation dtype.
        with torch.autocast(device_type="cuda", dtype=torch.float16,
                            enabled=(device.type == "cuda")):
            ce = torch.nn.functional.cross_entropy(
                sl.view(-1, sl.shape[-1]), sy.view(-1),
                ignore_index=-100, reduction="mean")
        scaler_b.scale(ce / float(len(batches))).backward()
    if device.type == "cuda":
        nan_grads = [n for n, p in model_b.named_parameters()
                     if p.requires_grad and p.grad is not None
                     and torch.isnan(p.grad.float()).any()]
        print(f"PROBE armB pre-unscale nan_grad_params={len(nan_grads)} "
              f"first={nan_grads[:3]}", flush=True)
    scaler_b.unscale_(optimizer_b)
    torch.nn.utils.clip_grad_norm_(params_b, 1.0)
    scale_before = float(scaler_b.get_scale())
    scaler_b.step(optimizer_b)
    scaler_b.update()
    scale_after = float(scaler_b.get_scale())
    print(f"PROBE armB scale {scale_before:.0f} -> {scale_after:.0f} "
          f"(halved => step skipped on inf/nan)", flush=True)
    deltas_b = {}
    for n, p in model_b.named_parameters():
        if p.requires_grad:
            deltas_b[n] = (p.detach().float().cpu() - before_b[n])
    n_nan_b = sum(1 for d in deltas_b.values()
                  if torch.isnan(d.float()).any())
    print(f"PROBE armB delta_nan_params={n_nan_b}/"
          f"{len(deltas_b)} nonzero="
          f"{sum(1 for d in deltas_b.values() if float(d.abs().max()) > 0)}",
          flush=True)

    # ---- comparison ----
    names = sorted(set(deltas_a) & set(deltas_b))
    assert names, "no common trainable parameters"
    missing = (set(deltas_a) ^ set(deltas_b))
    rows = []
    for n in names:
        da, db = deltas_a[n], deltas_b[n]
        abs_diff = float((da - db).abs().max())
        scale = max(float(da.abs().max()), float(db.abs().max()))
        rel = abs_diff / max(scale, 1e-30)
        rows.append((n, rel, abs_diff, scale))
    rows.sort(key=lambda r: -r[1])
    buckets = {"exact": 0, "lt_1e-4": 0, "lt_1e-2": 0, "lt_5e-2": 0,
               "ge_5e-2": 0}
    for n, rel, abs_diff, scale in rows:
        if rel == 0.0:
            buckets["exact"] += 1
        elif rel < 1e-4:
            buckets["lt_1e-4"] += 1
        elif rel < 1e-2:
            buckets["lt_1e-2"] += 1
        elif rel < 5e-2:
            buckets["lt_5e-2"] += 1
        else:
            buckets["ge_5e-2"] += 1
    top = rows[:10]
    report = {
        "n_params": len(names),
        "missing_params": sorted(missing),
        "rel_diff_buckets": buckets,
        "top10": [{"param": n, "rel_diff": rel, "abs_diff": abs_diff,
                   "scale": scale} for n, rel, abs_diff, scale in top],
        "worst_param": top[0][0],
        "worst_rel_diff": top[0][1],
        "worst_abs_diff": top[0][2],
        "param_scale": top[0][3],
        "pass": (buckets["ge_5e-2"] == 0 and not missing),
    }
    (out / "parity.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
