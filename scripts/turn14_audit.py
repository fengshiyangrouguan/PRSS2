#!/usr/bin/env python3
"""Turn-14 mechanism audit (review round 3, 2026-09-06).

Fixed-target suffix-depth test: instead of the official turn_14 bucket
(which switches cohort, depth, AND target at once), hold the cohort and
target fixed:
  - cohort: clean TEST dialogues with >= 15 turns (the special
    subpopulation the official n_turn_list=[15] bucket uses)
  - current = u14 (index 13), target = u15 (index 14) ALWAYS
  - compressed history = the LAST L turns of u1..u13, L in {1,2,4,8,13}
Then evaluate BOTH arms (ours lambda=0.3, taskonly lambda=0) at their
checkpoint_step400 snapshots with the exact official model construction
(foundation merge + LoRA + Gamma + new_token_rows restore).

Outputs (JSON): per-(dialogue, L, arm) loss_sum/token_count/NLL plus
per-layer Gamma stats (gate s, ||residual||/||mean-merge||) and the
number of Gamma recursion calls.
"""
import json
import os
import sys
import types
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, "/root/autodl-tmp")
sys.path.insert(0, "/root/autodl-tmp/src")
sys.path.insert(0, "/root/autodl-tmp/third_party/ccm")

os.environ["DIALOG_MIRROR"] = os.environ.get(
    "DIALOG_MIRROR",
    "/root/autodl-tmp/dailydialog_mirror/ijcnlp_dailydialog")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import train_ccm as tc  # noqa: E402

BASE = "/root/autodl-tmp"
DEVICE = torch.device("cuda", 0)
L_SUFFIXES = [1, 2, 4, 8, 13]
ARMS = {
    "ours": BASE + "/outputs/ccm_pilot/seed0_CCM_Diag_ours_fnd_l03_e50/checkpoint_step400.pt",
    "taskonly": BASE + "/outputs/ccm_pilot/seed0_CCM_Diag_taskonly_fnd_e50/checkpoint_step400.pt",
}
OUT = BASE + "/outputs/ccm_pilot/turn14_audit.json"


def build_tokenizer():
    return tc.build_tokenizer(types.SimpleNamespace(
        model_name_or_path=BASE + "/llama-7b-hf"))


def build_arm(ckpt_path, tok):
    """Exact eval-side reconstruction (same as eval_ccm_official)."""
    from src.model import load_lora_weight
    args = types.SimpleNamespace(arm="ours",
                                 model_name_or_path=BASE + "/llama-7b-hf",
                                 relative_embedding="skip",
                                 lora_r=8, gamma_hidden=64)
    model = tc.build_model(args, DEVICE)
    load_lora_weight(BASE + "/result/dialog/llama-7b-no", model, merge=True)
    model = tc.wrap_lora(model, args.lora_r)
    model.update_comp_token([32000 + k for k in range(tc.N_TOK)],
                            [32000 + tc.N_TOK + k for k in range(tc.N_TOK)])
    gammas = tc.attach_gamma(model, hidden=args.gamma_hidden)
    for g in gammas:
        g.half()
    dummy = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3)
    payload = tc.load_trainable(ckpt_path, model, dummy, DEVICE)
    assert "new_token_rows" in payload, "pre-fix checkpoint!"
    n = 2 * tc.N_TOK
    model.get_input_embeddings().weight[-n:] = \
        payload["new_token_rows"]["input_embed_rows"].to(DEVICE)
    model.lm_head.weight[-n:] = \
        payload["new_token_rows"]["lm_head_rows"].to(DEVICE)
    model.eval()
    # register Gamma stats hooks: record gate s and ||res||/||mean||
    stats = {"calls": 0, "layers": defaultdict(list)}
    for li, g in enumerate(gammas):
        def hook(module, inp, out, layer=li):
            prev, cur = inp[0], inp[1]
            base = 0.5 * (prev + cur)
            bnorm = base.detach().float().norm().item()
            rnorm = out.detach().float().norm().item()
            stats["calls"] += 1
            stats["layers"][layer].append({
                "gate_s": float(module.s.detach().float().item()),
                "res_norm": rnorm,
                "base_norm": bnorm,
                "ratio": (rnorm / bnorm) if bnorm > 0 else 0.0,
            })
        g.register_forward_hook(hook)
    return model, stats


def load_testset(tok):
    """Official clean TEST dialogues, tokenized, >=15 turns."""
    from src.data.dialogue.data import DialogueDataset
    ds = DialogueDataset(tok, comp_token=tok.comp_token_id,
                         online=True, add_comp_token=True,
                         clean_split=True)
    dialogs = []
    for i, d in enumerate(ds.testset["dialog"]):
        if len(d) >= 15:
            dialogs.append((i, d))
    return dialogs


def build_input(dialog, L):
    """fixed current=u14, target=u15; compressed history = last L of
    u1..u13.  Official token stream: turn + COMP + SUM + sep per history
    turn, then current + sep; output = target + eos."""
    sep = tok.encode("a\nA:", add_special_tokens=False)[1:]
    comp = tok.comp_token_id
    sumt = tok.sum_token_id
    bos = [tok.bos_token_id] if tok.bos_token_id is not None else []
    eos = [tok.eos_token_id]
    hist = dialog[13 - L:13]          # last L of u1..u13
    context = []
    for t in hist:
        context += list(t) + comp + sumt + sep
    context += list(dialog[13]) + sep   # current u14
    input_ids = bos + context
    output_ids = list(dialog[14]) + eos  # target u15
    full = input_ids + output_ids
    if len(full) > 512:                  # model max length: left-truncate
        full = full[-512:]
    labels = [-100] * (len(full) - len(output_ids)) + output_ids
    return full, labels


def evaluate_arm(model, stats, dialogs):
    records = []
    with torch.no_grad():
        for di, (dname, dialog) in enumerate(dialogs):
            for L in L_SUFFIXES:
                ids, labels = build_input(dialog, L)
                ids_t = torch.tensor([ids], device=DEVICE)
                labels_t = torch.tensor([labels], device=DEVICE)
                mask = torch.ones_like(ids_t)
                stats["calls"] = 0
                stats["layers"].clear()
                out = model(input_ids=ids_t, attention_mask=mask,
                            labels=labels_t)
                n_valid = int((labels_t != -100).sum())
                loss_sum = float(out.loss.detach()) * n_valid
                lay = stats["layers"]
                all_entries = [e for vs in lay.values() for e in vs]
                gate_mean = (float(np.mean([e["gate_s"] for e in all_entries]))
                             if all_entries else 0.0)
                ratio_mean = float(np.mean(
                    [v["ratio"] for vs in lay.values() for v in vs])
                    if any(lay.values()) else 0.0)
                ratio_max = float(np.max(
                    [v["ratio"] for vs in lay.values() for v in vs])
                    if any(lay.values()) else 0.0)
                records.append({
                    "dialogue": int(di),
                    "testset_idx": int(dname),
                    "L": L,
                    "loss_sum": loss_sum,
                    "token_count": n_valid,
                    "nll": loss_sum / n_valid,
                    "gamma_calls": stats["calls"],
                    "gamma_gate_mean": gate_mean,
                    "gamma_ratio_mean": ratio_mean,
                    "gamma_ratio_max": ratio_max,
                })
    return records


if __name__ == "__main__":
    tok = build_tokenizer()
    dialogs = load_testset(tok)
    print("cohort: {} dialogues with >=15 turns".format(len(dialogs)),
          flush=True)
    results = {"cohort_size": len(dialogs), "suffixes": L_SUFFIXES,
               "arms": {}}
    for arm_name, ckpt in ARMS.items():
        print("== arm {} ==".format(arm_name), flush=True)
        model, stats = build_arm(ckpt, tok)
        records = evaluate_arm(model, stats, dialogs)
        results["arms"][arm_name] = records
        print("  {} records".format(len(records)), flush=True)
        del model
        torch.cuda.empty_cache()
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print("done -> {}".format(OUT), flush=True)
