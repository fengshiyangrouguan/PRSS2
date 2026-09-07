#!/usr/bin/env python3
"""Review round 7: depth-curve eval of a trained official-host checkpoint.

Loads an official-host + Gamma model, applies a train_ccm checkpoint
(checkpoint_stepN.pt), and evaluates the 5-L x 51-dialog record-mean
curve with the VERIFIED turn14_audit_val.evaluate_arm machinery (same
as gate_official_host.py).  Compare against the clean official
reference: L1=1.920893 L2=1.814313 L4=1.742679 L8=1.728306 L13=1.700960.
"""
import importlib.util
import json
import os
import sys
import types

import numpy as np
import torch

sys.path.insert(0, "/root/autodl-tmp")
sys.path.insert(0, "/root/autodl-tmp/src")
sys.path.insert(0, "/root/autodl-tmp/third_party/ccm")
sys.path.insert(0, "/root/autodl-tmp/scripts")
os.environ["DIALOG_MIRROR"] = "/root/autodl-tmp/dailydialog_mirror/ijcnlp_dailydialog"
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import train_ccm as tc  # noqa: E402

BASE = "/root/autodl-tmp"
ADAPTER = BASE + "/result/dialog/llama-7b-no-online-merge_recur-ntok2"
CKPT = os.environ.get(
    "CKPT",
    BASE + "/outputs/ccm_pilot/seed0_R7_ours_cont_e300/checkpoint_step50.pt")
OUT = os.environ.get("OUT", "/tmp/eval_official_ckpt.json")


def build_host():
    args = types.SimpleNamespace(
        arm="ours", model_name_or_path=BASE + "/llama-7b-hf",
        relative_embedding="skip", lora_r=8, gamma_hidden=64,
        foundation=BASE + "/result/dialog/llama-7b-no",
        official_adapter=ADAPTER, official_host=True)
    model = tc.build_official_host(args, torch.device("cuda", 0))
    tc.attach_gamma(model, hidden=args.gamma_hidden)
    return model


def main():
    spec = importlib.util.spec_from_file_location(
        "turn14_audit_val", "/root/autodl-tmp/scripts/turn14_audit_val.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    tok, collator, dialogs = mod.build_stuff()
    max_pos = 2048

    model = build_host()
    dummy = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3)
    payload = tc.load_trainable(CKPT, model, dummy, torch.device("cuda", 0))
    model.eval()
    print("EVAL ckpt={} step={}".format(CKPT, payload.get("step")),
          flush=True)

    recs, skipped = mod.evaluate_arm(model, collator, dialogs, max_pos)
    out = {}
    for L in [1, 2, 4, 8, 13]:
        sub = [r for r in recs if r["L"] == L]
        out[L] = float(np.mean([r["nll"] for r in sub]))
    print("EVAL_DONE", flush=True)
    for L in [1, 2, 4, 8, 13]:
        print("L={:>2}: {:.6f}".format(L, out[L]), flush=True)
    with open(OUT, "w") as f:
        json.dump({"ckpt": CKPT, "nll": out, "skipped": skipped,
                   "step": payload.get("step")}, f, indent=2)
    print("wrote {}".format(OUT), flush=True)


if __name__ == "__main__":
    main()
