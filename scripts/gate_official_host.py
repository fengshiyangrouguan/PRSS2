#!/usr/bin/env python3
"""Review round 7 gate: step-0 identity — task@step0 == cond@step0 ==
official merge, within numerical error, on all five L."""
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


def build_host(gamma: bool):
    args = types.SimpleNamespace(
        arm="ours", model_name_or_path=BASE + "/llama-7b-hf",
        relative_embedding="skip", lora_r=8, gamma_hidden=64,
        foundation=BASE + "/result/dialog/llama-7b-no",
        official_adapter=ADAPTER, official_host=True)
    model = tc.build_official_host(args, torch.device("cuda", 0))
    if gamma:
        tc.attach_gamma(model, hidden=args.gamma_hidden)
    model.eval()
    return model



def nlls(model, collator, dialogs, max_pos):
    """Use the VERIFIED evaluate_arm machinery from turn14_audit_val."""
    recs, skipped = mod.evaluate_arm(model, collator, dialogs, max_pos)
    out = {}
    for L in [1, 2, 4, 8, 13]:
        sub = [r for r in recs if r["L"] == L]
        out[L] = np.mean([r["nll"] for r in sub])
    return out

if __name__ == "__main__":
    import torch as _t
    spec = importlib.util.spec_from_file_location(
        "turn14_audit_val", "/root/autodl-tmp/scripts/turn14_audit_val.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    tok, collator, dialogs = mod.build_stuff()
    max_pos = 2048
    print("== official merge reference ==", flush=True)
    ref_model = mod.build_official_merge()
    ref = nlls(ref_model, collator, dialogs, max_pos)
    del ref_model; _t.cuda.empty_cache()
    print("== task step0 (official host, no Gamma) ==", flush=True)
    task_model = build_host(gamma=False)
    task = nlls(task_model, collator, dialogs, max_pos)
    del task_model; _t.cuda.empty_cache()
    print("== cond step0 (official host + zero-output Gamma) ==", flush=True)
    cond_model = build_host(gamma=True)
    cond = nlls(cond_model, collator, dialogs, max_pos)
    del cond_model; _t.cuda.empty_cache()
    ok = True
    for L in [1, 2, 4, 8, 13]:
        d_task = abs(ref[L] - task[L])
        d_cond = abs(ref[L] - cond[L])
        status = "OK" if max(d_task, d_cond) < 1e-4 else "FAIL"
        if status == "FAIL":
            ok = False
        print(f"L={L:>2}: ref={ref[L]:.6f} task={task[L]:.6f} "
              f"cond={cond[L]:.6f} d_task={d_task:.2e} d_cond={d_cond:.2e} "
              f"{status}", flush=True)
    print("GATE_PASS" if ok else "GATE_FAIL", flush=True)
