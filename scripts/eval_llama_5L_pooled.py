#!/usr/bin/env python3
"""Llama protocol A (5L record-mean) VAL+TEST pooled cohort, for the
llama v2 comp-trainable experiment.

Same construction as turn14_audit_val.py but the cohort is val + test
merged (official clean_split=False口径, ~102 dialogues with >=15 turns):
for each dialogue and L in {1,2,4,8,13}: history = dialog[13-L:13],
context = dialog[13], target = dialog[14]; CE on the target only, EOS
excluded, record-mean aggregation.

Model build = the official merged host (Step-1 foundation merged +
released Step-2 adapter via peft_custom + SeparatedEmbedding); the
checkpoint (from train_ccm --official-host) supplies the trainable
state (LoRA + comp rows).
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
CCM = HERE.parent / "third_party" / "ccm"
for p in (str(SRC), str(CCM), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ["DIALOG_MIRROR"] = os.environ.get(
    "DIALOG_MIRROR",
    "/root/autodl-tmp/dailydialog_mirror/ijcnlp_dailydialog")

import torch
from torch.nn import functional as F

import train_ccm as tc

L_SUFFIXES = [1, 2, 4, 8, 13]
BASE = "/root/autodl-tmp"


def build_official_host(device):
    """Official merged host: LLaMA + Step-1 merged + Step-2 adapter +
    SeparatedEmbedding (fp32 throughout, official semantics)."""
    from transformers.models.llama.configuration_llama import LlamaConfig
    from src.arch.ccm_llama import LlamaForCausalLM_CCM
    from src.model import load_lora_weight, peft_custom
    from src.utils import SeparatedEmbedding
    from peft import LoraConfig
    config = LlamaConfig.from_pretrained(BASE + "/llama-7b-hf")
    config.comp_relative_embedding = "skip"
    model = LlamaForCausalLM_CCM.from_pretrained(
        BASE + "/llama-7b-hf", config=config, torch_dtype=torch.float32)
    model = model.to(device)
    load_lora_weight(BASE + "/result/dialog/llama-7b-no", model, merge=True)
    model.update_comp_token([32000, 32001], [32002, 32003])
    model.model.embed_tokens = SeparatedEmbedding(model.model.embed_tokens, 4)
    adapter_dir = BASE + "/result/dialog/llama-7b-no-online-merge_recur-ntok2"
    lora_cfg = LoraConfig().from_pretrained(adapter_dir)
    model = peft_custom.get_peft_model(model, lora_cfg)
    load_lora_weight(adapter_dir, model, merge=False)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ccm-topology", default="merge_recur",
                    choices=["merge_recur", "concat_recur"],
                    help="CCM topology override (concat line, "
                         "review 2026-09-25)")
    ap.add_argument("--ckpt", required=True,
                    help="checkpoint from train_ccm --official-host "
                         "(or NONE for the un-trained official merge)")
    ap.add_argument("--out", default="eval_llama_5L_pooled.json")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    device = torch.device("cuda", 0)

    import types as _t
    tok = tc.build_tokenizer(_t.SimpleNamespace(
        model_name_or_path=BASE + "/llama-7b-hf", host="llama"))
    from src.arguments import CompressionArguments
    from src.data.dialogue.data import DialogueDataset
    from src.data.dialogue.collator import DataCollatorForDialogue_LLAMA
    comp_args = CompressionArguments(attn_type=a.ccm_topology,
                                     num_comp_tokens=tc.N_TOK,
                                     add_comp_token=True,
                                     relative_embedding="skip")
    ds = DialogueDataset(tok, comp_token=tok.comp_token_id,
                         online=True, add_comp_token=True,
                         clean_split=True)
    collator = DataCollatorForDialogue_LLAMA(
        dialog=ds, tokenizer=tok, comp_args=comp_args,
        comp_token=tok.comp_token_id, sum_token=tok.sum_token_id,
        padding="left", pad_token=tok.pad_token_id,
        label_pad_token_id=-100)
    # val+test merged cohort (official clean_split=False口径)
    cohort = [(i, d) for i, d in enumerate(ds.valset["dialog"])
              if len(d) >= 15] \
        + [(i, d) for i, d in enumerate(ds.testset["dialog"])
           if len(d) >= 15]
    if a.limit:
        cohort = cohort[:a.limit]
    print("[eval] pooled cohort: {} dialogues >=15 turns".format(
        len(cohort)), flush=True)

    model = build_official_host(device)
    # joint-training checkpoints carry Gamma params: attach the Gamma
    # structure before loading (same as the R8 official-host eval path)
    tc.attach_gamma(model, hidden=64)
    if a.ckpt != "NONE":
        payload = torch.load(a.ckpt, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(payload["model"],
                                                    strict=False)
        if unexpected:
            raise RuntimeError("unexpected keys: {}"
                               .format(sorted(unexpected)[:5]))
        print("[eval] checkpoint loaded: {} (missing {})".format(
            a.ckpt, len(missing)), flush=True)
    else:
        print("[eval] NONE: official merge as released", flush=True)
    model.eval()

    eos = model.config.eos_token_id
    results = {}
    for L in L_SUFFIXES:
        nlls = []
        with torch.no_grad():
            for _i, d in cohort:
                item = {"dialog": d[13 - L:13] + [d[13], d[14]],
                        "is_train": False, "act": []}
                batch = collator([item])
                ids = batch["input_ids"].to(device)
                out = model(
                    input_ids=ids,
                    attention_mask=batch["attention_mask"].to(device),
                    attention_mask_comp=batch["attention_mask_comp"]
                    .to(device))
                logits = out.logits[0]
                shift_logits = logits[:-1]
                shift_labels = batch["labels"][0][1:].to(device)
                valid = (shift_labels != -100) & (shift_labels != eos)
                loss_per = F.cross_entropy(
                    shift_logits, shift_labels, reduction="none")
                n_valid = int(valid.sum())
                if n_valid == 0:
                    continue
                nlls.append(float(loss_per[valid].sum()) / n_valid)
        mean_nll = sum(nlls) / max(len(nlls), 1)
        results["L{}".format(L)] = {
            "ppl": math.exp(mean_nll), "mean_nll": mean_nll,
            "dialogues": len(nlls)}
        print("L={}: PPL={:.4f} mean_nll={:.4f} ({} dlg)".format(
            L, results["L{}".format(L)]["ppl"], mean_nll, len(nlls)),
            flush=True)

    with open(a.out, "w") as f:
        json.dump(results, f, indent=2)
    print("done -> {}".format(a.out), flush=True)


if __name__ == "__main__":
    main()
