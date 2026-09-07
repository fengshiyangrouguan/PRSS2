#!/usr/bin/env python3
"""Gate round-7 AB diag: same collated batch fed to ref and host models,
per-(dialog,L) logits max-diff + official-NLL. If weights are truly equal
and forward deterministic, all diffs must be ~0."""
import importlib.util
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
DEV = torch.device("cuda", 0)
LS = [1, 2, 4, 8, 13]

spec = importlib.util.spec_from_file_location(
    "turn14_audit_val", BASE + "/scripts/turn14_audit_val.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def build_host(gamma: bool):
    args = types.SimpleNamespace(
        arm="ours", model_name_or_path=BASE + "/llama-7b-hf",
        relative_embedding="skip", lora_r=8, gamma_hidden=64,
        foundation=BASE + "/result/dialog/llama-7b-no",
        official_adapter=ADAPTER, official_host=True)
    model = tc.build_official_host(args, DEV)
    if gamma:
        tc.attach_gamma(model, hidden=args.gamma_hidden)
    model.eval()
    return model


def nll_one(model, batch):
    ids = batch["input_ids"].to(DEV)
    mask = batch["attention_mask"].to(DEV)
    labels = batch["labels"].to(DEV)
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=mask,
                    attention_mask_comp=batch["attention_mask_comp"].to(DEV),
                    labels=labels)
    logits = out.logits[0].float()
    shift_logits = logits[:-1]
    shift_labels = labels[0][1:]
    eos = model.config.eos_token_id
    valid = (shift_labels != -100) & (shift_labels != eos)
    loss_per = torch.nn.functional.cross_entropy(
        shift_logits, shift_labels, reduction="none")
    return float(loss_per[valid].sum()), int(valid.sum()), logits


def main():
    tok, collator, dialogs = mod.build_stuff()
    print("cohort:", len(dialogs), flush=True)
    ref = mod.build_official_merge()
    host = build_host(gamma=False)
    # quick weight fingerprint on non-frozen-relevant keys
    diffs = []
    rp = dict(ref.named_parameters())
    hp = dict(host.named_parameters())
    for n in rp:
        if n not in hp:
            diffs.append((n, "missing-in-host")); continue
        if rp[n].data.shape != hp[n].data.shape:
            diffs.append((n, "shape")); continue
        d = (rp[n].data.double() - hp[n].data.double()).abs().max().item()
        if d > 1e-9:
            diffs.append((n, f"maxdiff={d:.3e}"))
    for n in hp:
        if n not in rp:
            diffs.append((n, "extra-in-host"))
    print("weight diffs:", len(diffs), flush=True)
    for n, why in diffs[:30]:
        print("  ", n, why, flush=True)
    agg = {L: {"ref_sum": 0.0, "host_sum": 0.0, "n": 0,
               "max_logit": 0.0, "max_nll": 0.0} for L in LS}
    n_bad = 0
    for di, (dname, dialog) in enumerate(dialogs):
        for L in LS:
            item = {"dialog": dialog[13 - L:13] + [dialog[13], dialog[14]],
                    "is_train": False, "act": []}
            batch = collator([item])
            if batch["input_ids"].shape[1] > 2048:
                continue
            rs, rn, rlog = nll_one(ref, batch)
            hs, hn, hlog = nll_one(host, batch)
            md = (rlog - hlog).abs().max().item()
            agg[L]["ref_sum"] += rs; agg[L]["host_sum"] += hs
            agg[L]["n"] += rn
            agg[L]["max_logit"] = max(agg[L]["max_logit"], md)
            nll_d = abs(rs / rn - hs / hn)
            agg[L]["max_nll"] = max(agg[L]["max_nll"], nll_d)
            if md > 1e-6 or nll_d > 1e-6:
                n_bad += 1
                if n_bad <= 8:
                    print(f"  DIVERGE di={di} dname={dname} L={L} "
                          f"len={batch['input_ids'].shape[1]} "
                          f"max_logit={md:.3e} nll_d={nll_d:.3e}", flush=True)
            del rlog, hlog
            torch.cuda.empty_cache()
        print(f"  dialog {di}/{len(dialogs)} done (bad so far {n_bad})",
              flush=True)
    print("== per-L ==", flush=True)
    for L in LS:
        a = agg[L]
        if a["n"] == 0:
            continue
        rnll = a["ref_sum"] / a["n"]; hnll = a["host_sum"] / a["n"]
        print(f"L={L:>2}: ref={rnll:.6f} host={hnll:.6f} "
              f"d={abs(rnll-hnll):.2e} max_logit={a['max_logit']:.2e} "
              f"max_nll={a['max_nll']:.2e} tokens={a['n']}", flush=True)
    print("BAD_ROWS:", n_bad, flush=True)
    print("AB_GATE_PASS" if n_bad == 0 else "AB_GATE_FAIL", flush=True)


if __name__ == "__main__":
    main()
