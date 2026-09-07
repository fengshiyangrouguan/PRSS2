#!/usr/bin/env python3
"""Diagnose the lora_B gradient asymmetry between the two parity arms.

Same model instance, same pre-collated microbatches:
  path_a: model(**batch with labels) under autocast  -> internal shifted
          mean CE (the official Trainer path)
  path_b: run_forward (no labels) + manual shifted mean CE under autocast
          (the parity arm-B path)
Backward each path separately and compare per-parameter gradients, the
logits, and the losses.  If logits match and losses match but lora_B
grads do not, the problem is in the graph/backward; if logits differ,
the forward inputs differ.

Usage (cloud):
    python -m scripts.ccm_parity_diag --model-name-or-path /root/autodl-tmp/llama-7b-hf \
        --dialog-mirror /root/autodl-tmp/dailydialog_mirror/ijcnlp_dailydialog
"""

import argparse
import os
import sys
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import torch

torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
CCM = HERE.parent / "third_party" / "ccm"
for p in (str(SRC), str(CCM)):
    if p not in sys.path:
        sys.path.insert(0, p)

from scripts.train_ccm import build_dataset, build_model, build_tokenizer, \
    wrap_lora


def parse_args():
    p = argparse.ArgumentParser("parity arm asymmetry diagnostic")
    p.add_argument("--model-name-or-path", required=True)
    p.add_argument("--dialog-mirror", required=True)
    p.add_argument("--n-microbatches", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--relative-embedding", default="skip",
                   choices=["skip", "base"])
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available()
                          else "cpu")
    tokenizer = build_tokenizer(args)
    dialog, collator = build_dataset(args, tokenizer)
    items = dialog.train_dataset
    batches = [collator([items[i % len(items)]])
               for i in range(args.n_microbatches)]

    torch.manual_seed(args.seed)
    model = wrap_lora(build_model(args, device), r=8)
    model.update_comp_token([32000, 32001], [32002, 32003])
    params = [p for p in model.parameters() if p.requires_grad]
    amc = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    use_cuda = device.type == "cuda"

    def shifted_ce(fwd_out, labels):
        sl = fwd_out.logits[..., :-1, :].contiguous()
        sy = labels.to(device)[..., 1:].contiguous()
        with torch.autocast(device_type="cuda", dtype=torch.float16,
                            enabled=use_cuda):
            return torch.nn.functional.cross_entropy(
                sl.view(-1, sl.shape[-1]), sy.view(-1),
                ignore_index=-100, reduction="mean")

    # ---- path A: full batch (labels) through the model, official loss ----
    model.zero_grad(set_to_none=True)
    loss_a = 0.0
    logits_a = None
    for b in batches:
        kw = {k: v.to(device) for k, v in b.items()
              if isinstance(v, torch.Tensor)}
        with torch.autocast(device_type="cuda", dtype=torch.float16,
                            enabled=use_cuda):
            out = model(**kw)
        if logits_a is None:
            logits_a = out.logits.detach().clone()
        amc.scale(out.loss).backward()
        loss_a += float(out.loss.detach())
    grads_a = {n: p.grad.detach().clone() if p.grad is not None else None
               for n, p in model.named_parameters() if p.requires_grad}

    # ---- path B: no labels, manual shifted CE ----
    model.zero_grad(set_to_none=True)
    loss_b = 0.0
    logits_b = None
    for b in batches:
        with torch.autocast(device_type="cuda", dtype=torch.float16,
                            enabled=use_cuda):
            fwd = model(input_ids=b["input_ids"].to(device),
                        attention_mask=b["attention_mask"].to(device),
                        attention_mask_comp=b.get("attention_mask_comp")
                        .to(device) if b.get("attention_mask_comp")
                        is not None else None)
        if logits_b is None:
            logits_b = fwd.logits.detach().clone()
        ce = shifted_ce(fwd, b["labels"])
        amc.scale(ce).backward()
        loss_b += float(ce.detach())
    grads_b = {n: p.grad.detach().clone() if p.grad is not None else None
               for n, p in model.named_parameters() if p.requires_grad}

    # ---- compare ----
    print(f"loss_a={loss_a:.6f} loss_b={loss_b:.6f} "
          f"logits_maxdiff={float((logits_a - logits_b).abs().max()):.6e}",
          flush=True)
    n_mismatch = 0
    n_exact = 0
    for n in sorted(grads_a):
        ga, gb = grads_a[n], grads_b[n]
        if ga is None or gb is None:
            tag = "NONE" if (ga is None) == (gb is None) else "ASYMM-NONE"
        else:
            d = float((ga - gb).abs().max())
            s = max(float(ga.abs().max()), float(gb.abs().max()))
            tag = f"exact" if d == 0 else \
                (f"diff d={d:.3e} s={s:.3e}" if s > 0 else "both-zero")
        if "diff" in tag and "ASYMM" not in tag:
            n_mismatch += 1
            print(f"  {n}: {tag}", flush=True)
        elif "exact" in tag:
            n_exact += 1
    print(f"summary: exact_or_zero={n_exact} mismatch={n_mismatch} "
          f"of {len(grads_a)}", flush=True)

    # Full lora_B statistics: nan / finite-max / zero counts per path.
    def stats(grads):
        n_nan = n_none = n_zero = 0
        finite_max = 0.0
        for n, g in grads.items():
            if "lora_B" not in n:
                continue
            if g is None:
                n_none += 1
                continue
            g = g.float()
            if torch.isnan(g).any():
                n_nan += 1
            elif float(g.abs().max()) == 0.0:
                n_zero += 1
            else:
                finite_max = max(finite_max, float(g.abs().max()))
        return n_nan, n_none, n_zero, finite_max

    sa = stats(grads_a)
    sb = stats(grads_b)
    print(f"lora_B stats pathA(nan={sa[0]} none={sa[1]} zero={sa[2]} "
          f"finite_max={sa[3]:.3e})", flush=True)
    print(f"lora_B stats pathB(nan={sb[0]} none={sb[1]} zero={sb[2]} "
          f"finite_max={sb[3]:.3e})", flush=True)
    # Per-layer-0..2 sample of finite grads for scale comparison.
    shown = 0
    for n in sorted(grads_a):
        if "lora_B" not in n or shown >= 6:
            continue
        ga, gb = grads_a[n], grads_b[n]
        f = (lambda g: "None" if g is None else
             (f"nan" if torch.isnan(g.float()).any() else
              f"{float(g.float().abs().max()):.3e}"))
        print(f"  {n}: A={f(ga)} B={f(gb)}", flush=True)
        shown += 1


if __name__ == "__main__":
    main()
