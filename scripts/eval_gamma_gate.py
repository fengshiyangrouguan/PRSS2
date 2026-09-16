#!/usr/bin/env python3
"""Gamma wiring gate (review 2026-09-16 P0-C).

Loads an R10-style checkpoint through the official host and compares,
on ONE L=13 dialogue batch:

  logits_R10   (loaded checkpoint)
  logits_G0    (all Gamma U.weight zeroed)
  logits_G100  (all Gamma U.weight x 100)

If Delta(logits_R10, logits_G0) is machine-zero, the eval path never
touches Gamma (loading or wiring bug).  If it is only ~1e-6..1e-5, the
wiring is fine and Gamma itself is nearly inert.  A large Delta under
100x proves the eval forward really routes through Gamma.
"""
import argparse
import math
import os
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
CCM = HERE.parent / "third_party" / "ccm"
for p in (str(SRC), str(CCM), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch

import train_ccm as tc
import eval_ccm_depth as ed


def gamma_u_params(model):
    out = []
    for n, p in model.named_parameters():
        if ".gamma.U.weight" in n:
            out.append(p)
    return out


def nll_of(out, labels, device):
    loss, n = ed.eval_ce_shifted_noeos(out, labels.to(device), device)
    return float(loss.detach()) / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ref-ckpt", default="",
                    help="R9 counterpart checkpoint for ||U^R10-U^R9||")
    ap.add_argument("--model-name-or-path",
                    default="/root/autodl-tmp/llama-7b-hf")
    ap.add_argument("--dialog-mirror",
                    default="/root/autodl-tmp/dailydialog_mirror/"
                            "ijcnlp_dailydialog")
    ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args()
    device = torch.device("cuda", a.gpu)
    os.environ["DIALOG_MIRROR"] = a.dialog_mirror
    args = types.SimpleNamespace(
        arm="ours", model_name_or_path=a.model_name_or_path,
        dialog_mirror=a.dialog_mirror, relative_embedding="skip",
        lora_r=8, z_dim=128, rpbe_seed=0, sketch_dim=64, gamma_hidden=64,
        foundation="/root/autodl-tmp/result/dialog/llama-7b-no",
        official_adapter="/root/autodl-tmp/result/dialog/"
                        "llama-7b-no-online-merge_recur-ntok2",
        official_host=True)

    tokenizer, model = ed.load_official_host_arm(args, device, a.ckpt)
    us = gamma_u_params(model)
    print("Gamma U params:", len(us), flush=True)
    for p in us:
        assert p.requires_grad, "Gamma U not trainable"

    # P0-B strict: checkpoint tensor == loaded model tensor (bit-for-bit
    # on every Gamma/LoRA/COMP key) + missing/unexpected audit.
    payload = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    sd = payload["model"]
    model_sd = model.state_dict()
    # checkpoint carries ONLY trainable params; frozen base weights are
    # legitimately absent.  unexpected = checkpoint keys the model does
    # not have (true error); missing trainable = trainable params the
    # checkpoint lacks.
    missing = [k for k in model_sd if k not in sd]
    unexpected = [k for k in sd if k not in model_sd]
    trainable_names = {n for n, p in model.named_parameters()
                       if p.requires_grad}
    missing_trainable = [k for k in missing if k in trainable_names]
    mismatch = [k for k in sd
                if k in model_sd
                and not torch.equal(model_sd[k].cpu(), sd[k].cpu())]
    print("missing (any):", len(missing), flush=True)
    print("missing TRAINABLE:", missing_trainable[:5], flush=True)
    print("unexpected:", unexpected[:5], flush=True)
    print("bit-mismatch tensors:", len(mismatch), flush=True)
    assert not missing_trainable and not unexpected and not mismatch, \
        "P0-B FAIL"
    print("P0-B PASS: checkpoint == loaded model bit-for-bit", flush=True)

    if a.ref_ckpt:
        ref_sd = torch.load(a.ref_ckpt, map_location="cpu",
                            weights_only=False)["model"]
        d10 = 0.0
        for k in sd:
            if ".gamma." in k and k in ref_sd:
                d10 += float((sd[k].double() - ref_sd[k].double()
                              ).norm().pow(2))
        print("||U^R10 - U^R9|| over ALL gamma tensors = {:.6f}"
              .format(math.sqrt(d10)), flush=True)

    # One L=13 batch (protocol C: turns = 15)
    dialog, comp_args = ed.build_eval_dataset(
        args, tokenizer, False, online=True, comp_type="online")
    collator = ed.build_collator(dialog, tokenizer, comp_args,
                                 comp_type="online", sum_recur=True)
    dialogs = dialog.testset["dialog"]
    pool = [d for d in dialogs if len(d) >= 15][:4]
    items = [{"dialog": d[:15], "act": [], "is_train": False}
             for d in pool]
    batch = collator(items)
    labels = batch["labels"]

    def forward():
        return tc.run_forward(model, batch, device, grad_enabled=False)

    with torch.no_grad():
        out_r10 = forward()
        nll_r10 = nll_of(out_r10, labels, device)

        saved = [p.data.clone() for p in us]
        for p in us:
            p.data.zero_()
        out_g0 = forward()
        nll_g0 = nll_of(out_g0, labels, device)
        for p, s in zip(us, saved):
            p.data.copy_(s * 100.0)
        out_g100 = forward()
        nll_g100 = nll_of(out_g100, labels, device)
        for p, s in zip(us, saved):
            p.data.copy_(s)

    d10 = float((out_r10.logits - out_g0.logits).abs().max())
    d100 = float((out_g100.logits - out_g0.logits).abs().max())
    dd_r10 = (out_r10.logits.double() - out_g0.logits.double())
    dd_100 = (out_g100.logits.double() - out_g0.logits.double())
    mean_abs = float(dd_r10.abs().mean())
    rms = float(dd_r10.pow(2).mean().sqrt())
    rel = float(dd_r10.norm() / out_g0.logits.double().norm())
    # target-position log-prob delta (the part that actually moves PPL)
    shift = out_g0.logits.double()[..., :-1, :]
    sl = labels.to(device)[..., 1:]
    m = (sl != -100) & (sl != 2)
    lp0 = torch.nn.functional.log_softmax(shift, dim=-1)
    lp1 = torch.nn.functional.log_softmax(
        out_r10.logits.double()[..., :-1, :], dim=-1)
    tgt0 = (lp0.gather(-1, sl.unsqueeze(-1)).squeeze(-1) * m).sum() \
        / m.sum().clamp(min=1)
    tgt1 = (lp1.gather(-1, sl.unsqueeze(-1)).squeeze(-1) * m).sum() \
        / m.sum().clamp(min=1)
    print("L=13 batch ({} dialogues)".format(len(items)), flush=True)
    print("NLL R10      = {:.6f}".format(nll_r10), flush=True)
    print("NLL Gamma=0  = {:.6f}".format(nll_g0), flush=True)
    print("NLL Gamma100 = {:.6f}".format(nll_g100), flush=True)
    print("Delta logits R10 vs G0: "
          "max={:.3e} mean_abs={:.3e} rms={:.3e} rel_L2={:.3e}".format(
              d10, mean_abs, rms, rel), flush=True)
    print("Delta logits G100 vs G0: max={:.3e} rel_L2={:.3e}".format(
        d100, float(dd_100.norm() / out_g0.logits.double().norm())),
        flush=True)
    print("Delta target logprob R10 vs G0 = {:.3e}".format(
        float(tgt1 - tgt0)), flush=True)
    print("Delta NLL   R10 vs G0    = {:.6f}".format(nll_r10 - nll_g0),
          flush=True)
    print("Delta NLL   G100 vs G0   = {:.6f}".format(
        nll_g100 - nll_g0), flush=True)
    if d10 <= 1e-9:
        print("GATE FAIL: eval path never touches Gamma", flush=True)
        sys.exit(1)
    if d100 <= d10 * 10:
        print("GATE FAIL: 100x Gamma barely moves logits", flush=True)
        sys.exit(1)
    print("GATE PASS: eval forward routes through Gamma; "
          "loaded Gamma is weak (d10 ~ {:.2e})".format(d10), flush=True)


if __name__ == "__main__":
    main()
