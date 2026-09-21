#!/usr/bin/env python3
"""Protocol B reference arms (no_ctx / full_ctx) for the Qwen3 CCM port.

Same five-bucket pooled protocol as eval_qwen3_pooled.py (val+test
merged, turn_3/4/6/10/14 buckets with >= n turns truncated, target
turn CE, EOS excluded, token-weighted PPL), but on the RAW pretrained
Qwen3 backbone with NO compression tokens:

  - full_ctx: history turns + sep concatenated with the context turn
    (online=False, comp_ids empty)
  - no_ctx  : only the context turn (neg_control)

Usage:
  python -u eval_qwen3_pooled_ref.py --gpu 1 --out eval_B_ref.json
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, "/root/autodl-tmp")
sys.path.insert(0, "/root/autodl-tmp/third_party/ccm")
sys.path.insert(0, "/root/autodl-tmp/src")
sys.path.insert(0, "/root/autodl-tmp/scripts")

import torch

import train_ccm as tc


def eval_ce_shifted_noeos(logits, labels, device, eos_id):
    """Shifted CE ignoring padding (-100) AND EOS (protocol B)."""
    labs = labels.to(device)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labs[..., 1:].contiguous()
    shift_labels = shift_labels.masked_fill(shift_labels == eos_id, -100)
    n_valid = int((shift_labels != -100).sum())
    loss = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]),
        shift_labels.view(-1), ignore_index=-100, reduction="sum")
    return loss, n_valid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name-or-path",
                    default="/root/autodl-tmp/qwen3-4b-instruct")
    ap.add_argument("--dialog-mirror",
                    default="/root/autodl-tmp/dailydialog_mirror/"
                            "ijcnlp_dailydialog")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", default="eval_qwen3_pooled_ref.json")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    os.environ["DIALOG_MIRROR"] = a.dialog_mirror
    device = torch.device(f"cuda:{a.gpu}")

    import types
    args = types.SimpleNamespace(
        arm="ours", model_name_or_path=a.model_name_or_path,
        dialog_mirror=a.dialog_mirror, relative_embedding="skip",
        lora_r=8, z_dim=128, rpbe_seed=0, sketch_dim=64, gamma_hidden=64,
        host="qwen3", official_host=False, foundation="",
        official_adapter="", micro_batch=1)

    tokenizer = tc.build_tokenizer(args)
    eos_id = int(tokenizer.eos_token_id)

    # RAW pretrained Qwen3 (no comp tokens, no ckpt)
    from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
    model = Qwen3ForCausalLM.from_pretrained(
        args.model_name_or_path, torch_dtype=torch.bfloat16).to(device)
    model.eval()
    print("ref arms use RAW pretrained Qwen3", flush=True)

    from src.arguments import CompressionArguments
    from src.data.dialogue.qwen3_data import (
        Qwen3DialogueDataset, Qwen3DialogueCollator)
    comp_args = CompressionArguments(attn_type="merge_recur",
                                     num_comp_tokens=tc.N_TOK,
                                     add_comp_token=True,
                                     relative_embedding="skip")
    dialog = Qwen3DialogueDataset(tokenizer, mirror=a.dialog_mirror,
                                  pooled=True)

    results = {}
    for name, online, neg_control in [
            ("full_ctx", False, False),
            ("no_ctx", False, True)]:
        collator = Qwen3DialogueCollator(
            dataset=dialog, tokenizer=tokenizer, comp_args=comp_args,
            comp_token=[], sum_token=[],
            pad_token=tokenizer.pad_token_id, label_pad_token_id=-100,
            online=online, neg_control=neg_control)
        per_bucket = {}
        for bname, items in dialog.eval_dataset.items():
            if a.limit:
                items = items[:a.limit]
            items = [dict(it) for it in items]
            for it in items:
                it["fixed_depth"] = True
            total = 0.0
            n_tok = 0
            with torch.no_grad():
                for i in range(0, len(items), 8):
                    batch = collator(items[i:i + 8])
                    out = model(
                        input_ids=batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device))
                    s, n = eval_ce_shifted_noeos(out.logits,
                                                 batch["labels"],
                                                 device, eos_id)
                    total += float(s.detach())
                    n_tok += n
            nll = total / max(n_tok, 1)
            per_bucket[bname] = {"nll": nll, "perplexity": math.exp(nll),
                                "tokens": n_tok,
                                "dialogues": len(items)}
            print("{} {}: perplexity={:.4f} nll={:.4f} ({} tok, {} dlg)"
                  .format(name, bname, math.exp(nll), nll, n_tok,
                          len(items)), flush=True)
        results[name] = per_bucket

    with open(a.out, "w") as f:
        json.dump(results, f, indent=2)
    print("ref done -> {}".format(a.out), flush=True)


if __name__ == "__main__":
    main()
