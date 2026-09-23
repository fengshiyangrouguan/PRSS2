#!/usr/bin/env python3
"""Llama protocol B (official pooled evaluate_perp semantics) for the
llama v2 joint run: val+test merged (clean_split=False), five turn
buckets, token-weighted PPL with EOS excluded, record-mean loss field.

Model build = official merged host (Step-1 merged + Step-2 adapter +
SeparatedEmbedding) + Gamma (joint checkpoints carry Gamma params).
--ckpt NONE evaluates the released official merge as-is.
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

BASE = "/root/autodl-tmp"


def build_official_host(device):
    from transformers.models.llama.configuration_llama import LlamaConfig
    from src.arch.ccm_llama import LlamaForCausalLM_CCM
    from src.model import load_lora_weight, peft_custom
    from src.utils import SeparatedEmbedding
    from peft import LoraConfig
    config = LlamaConfig.from_pretrained(BASE + "/llama-7b-hf")
    config.comp_relative_embedding = "skip"
    # official evaluation protocol is fp16 (fp16_full_eval=True)
    model = LlamaForCausalLM_CCM.from_pretrained(
        BASE + "/llama-7b-hf", config=config, torch_dtype=torch.float16)
    model = model.to(device)
    load_lora_weight(BASE + "/result/dialog/llama-7b-no", model, merge=True)
    model.update_comp_token([32000, 32001], [32002, 32003])
    model.model.embed_tokens = SeparatedEmbedding(model.model.embed_tokens, 4)
    adapter_dir = BASE + "/result/dialog/llama-7b-no-online-merge_recur-ntok2"
    lora_cfg = LoraConfig().from_pretrained(adapter_dir)
    model = peft_custom.get_peft_model(model, lora_cfg)
    load_lora_weight(adapter_dir, model, merge=False)
    # fp16 consistency: comp rows and Gamma (plain attribute modules are
    # not traversed by model.to) must be cast explicitly
    model.base_model.model.model.embed_tokens.comp_embeddings.weight.data = \
        model.base_model.model.model.embed_tokens.comp_embeddings.weight \
        .data.half()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="eval_llama_pooled_B.json")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    device = torch.device("cuda", 0)

    import types as _t
    tok = tc.build_tokenizer(_t.SimpleNamespace(
        model_name_or_path=BASE + "/llama-7b-hf", host="llama"))
    from src.arguments import CompressionArguments
    from src.data.dialogue.data import DialogueDataset
    from src.data.dialogue.collator import DataCollatorForDialogue_LLAMA
    comp_args = CompressionArguments(attn_type="merge_recur",
                                     num_comp_tokens=tc.N_TOK,
                                     add_comp_token=True,
                                     relative_embedding="skip")
    # official pooled protocol: clean_split=False (val + test merged)
    ds = DialogueDataset(tok, comp_token=tok.comp_token_id,
                         online=True, add_comp_token=True,
                         clean_split=False)
    collator = DataCollatorForDialogue_LLAMA(
        dialog=ds, tokenizer=tok, comp_args=comp_args,
        comp_token=tok.comp_token_id, sum_token=tok.sum_token_id,
        padding="left", pad_token=tok.pad_token_id,
        label_pad_token_id=-100)

    model = build_official_host(device)
    # NO gamma attach (native S4 line, 2026-09-24): the train side runs
    # the official host WITHOUT Gamma; attaching it here triggers the
    # ccm_llama.py recur-scan block whose n_heads binding only happens
    # under t_max>=2 — turn_3 dialogues (L=1, t_max=1) then raise
    # UnboundLocalError n_heads.  The merge baseline and the native
    # treewise checkpoints are all Gamma-free.
    if a.ckpt != "NONE":
        payload = torch.load(a.ckpt, map_location=device,
                             weights_only=False)
        missing, unexpected = model.load_state_dict(payload["model"],
                                                    strict=False)
        if unexpected:
            raise RuntimeError("unexpected keys: {}"
                               .format(sorted(unexpected)[:5]))
        print("[eval] checkpoint loaded: {} (missing {})".format(
            a.ckpt, len(missing)), flush=True)
    model.eval()
    eos = model.config.eos_token_id

    results = {}
    for name, ds_items in ds.eval_dataset.items():
        items = list(ds_items)
        if a.limit:
            items = items[:a.limit]
        items = [dict(it) for it in items]
        for it in items:
            it["fixed_depth"] = True
        tot_nll = 0.0
        tot_tok = 0
        sum_dlg = 0.0
        n_dlg = 0
        with torch.no_grad():
            for i in range(0, len(items), 8):
                batch = collator(items[i:i + 8])
                # tc.run_forward matches the train-side forward contract
                # (comp/sum mask kwargs) — the raw model(...) call with
                # the old attention_mask_comp signature drifted from
                # ccm_llama.py and raised UnboundLocalError n_heads.
                out = tc.run_forward(model, batch, device,
                                     grad_enabled=False)
                logits = out.logits
                labs = batch["labels"].to(device)
                sh = logits[..., :-1, :].contiguous()
                sl = labs[..., 1:].contiguous().clone()
                sl = sl.masked_fill(sl == eos, -100)
                B = sh.shape[0]
                row_n = (sl != -100).sum(-1)
                row_loss = F.cross_entropy(
                    sh.view(-1, sh.shape[-1]), sl.reshape(-1),
                    ignore_index=-100, reduction="none").view(
                        B, -1).sum(-1)
                tot_nll += float(row_loss.detach().sum())
                tot_tok += int(row_n.sum())
                row_mean = row_loss / row_n.clamp(min=1)
                sum_dlg += float(row_mean.detach().sum())
                n_dlg += B
        results[name] = {
            "perplexity": math.exp(tot_nll / max(tot_tok, 1)),
            "loss": sum_dlg / max(n_dlg, 1),
            "tokens": tot_tok, "dialogues": n_dlg}
        print("{}: perplexity={:.4f} loss={:.4f}".format(
            name, results[name]["perplexity"], results[name]["loss"]),
            flush=True)

    with open(a.out, "w") as f:
        json.dump(results, f, indent=2)
    print("done -> {}".format(a.out), flush=True)


if __name__ == "__main__":
    main()
