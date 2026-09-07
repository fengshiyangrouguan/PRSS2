#!/usr/bin/env python3
"""Fixed-target VALIDATION audit (review round 6).

Same official-collator + official-NLL machinery as turn14_audit.py v2,
but on the clean VALIDATION cohort (>=15 turns) and across FOUR
checkpoints: conditional @200/@250 and taskonly @200/@250.

Cohort/target fixed: current = u14, target = u15; compressed history
suffix L in {1,2,4,8,13}. No Gamma audit (speed).
"""
import json
import os
import sys
import types

import numpy as np
import torch

sys.path.insert(0, "/root/autodl-tmp")
sys.path.insert(0, "/root/autodl-tmp/src")
sys.path.insert(0, "/root/autodl-tmp/third_party/ccm")
os.environ["DIALOG_MIRROR"] = "/root/autodl-tmp/dailydialog_mirror/ijcnlp_dailydialog"
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import train_ccm as tc  # noqa: E402
from src.arguments import CompressionArguments  # noqa: E402
from src.data.dialogue.data import DialogueDataset  # noqa: E402
from src.data.dialogue.collator import DataCollatorForDialogue_LLAMA  # noqa: E402

BASE = "/root/autodl-tmp"
DEVICE = torch.device("cuda", 0)
L_SUFFIXES = [1, 2, 4, 8, 13]
ARMS = {
    "cond200": BASE + "/outputs/ccm_pilot/seed0_CCM_Diag_ours_cond_e50/checkpoint_step200.pt",
    "cond250": BASE + "/outputs/ccm_pilot/seed0_CCM_Diag_ours_cond_e50/checkpoint_step250.pt",
    "task200": BASE + "/outputs/ccm_pilot/seed0_CCM_Diag_taskonly_fnd_e50/checkpoint_step200.pt",
    "task250": BASE + "/outputs/ccm_pilot/seed0_CCM_Diag_taskonly_fnd_e50/checkpoint_step250.pt",
}
OUT = BASE + "/outputs/ccm_pilot/turn14_audit_val.json"


def build_stuff():
    tok = tc.build_tokenizer(types.SimpleNamespace(
        model_name_or_path=BASE + "/llama-7b-hf"))
    comp_args = CompressionArguments(attn_type="merge_recur",
                                     num_comp_tokens=tc.N_TOK,
                                     add_comp_token=True,
                                     relative_embedding="skip")
    ds = DialogueDataset(tok, comp_token=tok.comp_token_id,
                         online=True, add_comp_token=True, clean_split=True)
    collator = DataCollatorForDialogue_LLAMA(
        dialog=ds, tokenizer=tok, comp_args=comp_args,
        comp_token=tok.comp_token_id, sum_token=tok.sum_token_id,
        padding="left", pad_token=tok.pad_token_id, label_pad_token_id=-100)
    dialogs = [(i, d) for i, d in enumerate(ds.valset["dialog"])
               if len(d) >= 15]
    return tok, collator, dialogs


def build_arm(ckpt_path):
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
    tc.attach_gamma(model, hidden=args.gamma_hidden)
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
    return model


def build_official_merge():
    """Official CCM-merge, EXACT official loading path: base LLaMA (no
    embed resize) -> Step-1 foundation merged -> SeparatedEmbedding for
    the 4 COMP/SUM tokens -> conditional LoRA wrap (peft_custom) ->
    released Step-2 compression adapter state."""
    from transformers.models.llama.configuration_llama import LlamaConfig
    from src.arch.ccm_llama import LlamaForCausalLM_CCM
    from src.model import load_lora_weight, peft_custom
    from src.utils import SeparatedEmbedding
    from peft import LoraConfig
    model_path = BASE + "/llama-7b-hf"
    config = LlamaConfig.from_pretrained(model_path)
    config.comp_relative_embedding = "skip"
    model = LlamaForCausalLM_CCM.from_pretrained(
        model_path, config=config, torch_dtype=torch.float32)
    model = model.to(DEVICE)
    load_lora_weight(BASE + "/result/dialog/llama-7b-no", model, merge=True)
    model.update_comp_token([32000, 32001], [32002, 32003])
    n_tok = 4  # COMP0/1 + SUM0/1
    model.model.embed_tokens = SeparatedEmbedding(model.model.embed_tokens,
                                                  n_tok)
    # fp32 throughout (model is fp32); no dtype mixing
    adapter_dir = BASE + "/result/dialog/llama-7b-no-online-merge_recur-ntok2"
    lora_cfg = LoraConfig().from_pretrained(adapter_dir)
    model = peft_custom.get_peft_model(model, lora_cfg)
    load_lora_weight(adapter_dir, model, merge=False)
    model.eval()
    return model


def evaluate_arm(model, collator, dialogs, max_pos):
    records = []
    skipped = 0
    with torch.no_grad():
        for di, (dname, dialog) in enumerate(dialogs):
            for L in L_SUFFIXES:
                item = {"dialog": dialog[13 - L:13]
                        + [dialog[13], dialog[14]],
                        "is_train": False, "act": []}
                batch = collator([item])
                ids = batch["input_ids"].to(DEVICE)
                mask = batch["attention_mask"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)
                if ids.shape[1] > max_pos:
                    skipped += 1
                    continue
                out = model(input_ids=ids, attention_mask=mask,
                            attention_mask_comp=batch["attention_mask_comp"].to(DEVICE),
                            labels=labels)
                logits = out.logits[0]
                shift_logits = logits[:-1]
                shift_labels = labels[0][1:]
                eos = model.config.eos_token_id
                valid = (shift_labels != -100) & (shift_labels != eos)
                loss_per = torch.nn.functional.cross_entropy(
                    shift_logits, shift_labels, reduction="none")
                loss_sum = float(loss_per[valid].sum())
                n_valid = int(valid.sum())
                records.append({
                    "dialogue": int(di),
                    "valset_idx": int(dname),
                    "L": L,
                    "loss_sum": loss_sum,
                    "token_count": n_valid,
                    "nll": loss_sum / max(n_valid, 1),
                })
    return records, skipped


if __name__ == "__main__":
    tok, collator, dialogs = build_stuff()
    print("VAL cohort: {} dialogues with >=15 turns".format(len(dialogs)),
          flush=True)
    results = {"cohort_size": len(dialogs), "suffixes": L_SUFFIXES,
               "arms": {}}
    for arm_name, ckpt in ARMS.items():
        print("== arm {} ==".format(arm_name), flush=True)
        model = build_arm(ckpt)
        max_pos = model.config.max_position_embeddings
        records, skipped = evaluate_arm(model, collator, dialogs, max_pos)
        results["arms"][arm_name] = records
        print("  {} records, {} skipped".format(len(records), skipped),
              flush=True)
        del model
        torch.cuda.empty_cache()
    with open(OUT, "w") as f:
        json.dump(results, f)
    print("done -> {}".format(OUT), flush=True)
