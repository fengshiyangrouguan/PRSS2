#!/usr/bin/env python3
"""Protocol A (5L record-mean) for the Qwen3 CCM port.

Same protocol as the Llama-line turn14_audit_val.py: the clean
validation cohort of dialogues with >= 15 turns; for each dialogue and
each depth L in {1, 2, 4, 8, 13}: history = dialog[13-L:13] (L turns),
context = dialog[13], target = dialog[14] (fixed 15th turn).  The CE is
scored on the target turn only, EOS excluded, record-mean aggregation
(dialogue-equal weights): PPL_L = exp(mean_dialogue NLL_L).

Model build matches train_ccm (SeparatedEmbedding + conditional LoRA +
optional Gamma); --no-gamma for merge-stage checkpoints.
"""
import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
CCM = HERE.parent / "third_party" / "ccm"
for p in (str(SRC), str(CCM), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
from torch.nn import functional as F

import train_ccm as tc

L_SUFFIXES = [1, 2, 4, 8, 13]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name-or-path",
                    default="/root/autodl-tmp/qwen3-4b-instruct")
    ap.add_argument("--dialog-mirror",
                    default="/root/autodl-tmp/dailydialog_mirror/"
                            "ijcnlp_dailydialog")
    ap.add_argument("--ckpt", required=True,
                    help="checkpoint.pt (merge-stage or ours)")
    ap.add_argument("--no-gamma", action="store_true",
                    help="merge-stage checkpoint (no Gamma params)")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", default="eval_qwen3_protocolA.json")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--pooled", action="store_true",
                    help="val+test merged cohort (official clean_split=False "
                         "口径, ~102 dialogues) instead of val-only")
    a = ap.parse_args()

    import os
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

    # model build: official layout + conditional LoRA + optional Gamma
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
    from src.arch.ccm_qwen3 import Qwen3ForCausalLM_CCM
    from src.utils import SeparatedEmbedding
    config = Qwen3Config.from_pretrained(args.model_name_or_path)
    config.comp_relative_embedding = "skip"
    model = Qwen3ForCausalLM_CCM.from_pretrained(
        args.model_name_or_path, config=config,
        torch_dtype=torch.bfloat16).to(device)
    model.model.embed_tokens = SeparatedEmbedding(
        model.model.embed_tokens, 2 * tc.N_TOK)
    model.update_comp_token(
        [config.vocab_size + k for k in range(tc.N_TOK)],
        [config.vocab_size + tc.N_TOK + k for k in range(tc.N_TOK)])
    model = tc.wrap_lora(model, args.lora_r)
    model.update_comp_token(
        [tokenizer.comp_token_id[k] for k in range(tc.N_TOK)],
        [tokenizer.sum_token_id[k] for k in range(tc.N_TOK)])
    for _p in model.parameters():
        _p.requires_grad_(False)
    for _n, _p in model.named_parameters():
        if "lora_" in _n:
            _p.requires_grad_(True)
    model.base_model.model.model.embed_tokens.comp_embeddings.weight \
        .requires_grad_(True)
    if not a.no_gamma:
        tc.attach_gamma(model, hidden=args.gamma_hidden)
    dummy = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3)
    tc.load_trainable(a.ckpt, model, dummy, device, load_optimizer=False)
    model.eval()
    print("[eval] checkpoint loaded: {}".format(a.ckpt), flush=True)

    from src.arguments import CompressionArguments
    from src.data.dialogue.qwen3_data import (
        Qwen3DialogueDataset, Qwen3DialogueCollator)
    comp_args = CompressionArguments(attn_type="merge_recur",
                                     num_comp_tokens=tc.N_TOK,
                                     add_comp_token=True,
                                     relative_embedding="skip")
    dialog = Qwen3DialogueDataset(tokenizer, mirror=a.dialog_mirror)
    collator = Qwen3DialogueCollator(
        dataset=dialog, tokenizer=tokenizer, comp_args=comp_args,
        comp_token=tokenizer.comp_token_id,
        sum_token=tokenizer.sum_token_id,
        pad_token=tokenizer.pad_token_id, label_pad_token_id=-100)

    # cohort: >= 15-turn dialogues (val only, or val+test pooled)
    src_items = dialog.valset + dialog.testset if a.pooled \
        else dialog.valset
    cohort = [(i, d["dialog"]) for i, d in enumerate(src_items)
              if len(d["dialog"]) >= 15]
    if a.limit:
        cohort = cohort[:a.limit]
    print("[eval] cohort: {} dialogues >= 15 turns".format(len(cohort)),
          flush=True)

    results = {}
    for L in L_SUFFIXES:
        nlls = []
        with torch.no_grad():
            for _i, d in cohort:
                # L history turns + context (turn 14) + fixed target
                # (turn 15): L+2 turns so the collator compresses exactly
                # the L history turns and predicts dialog[14].
                item = {"dialog": list(d[13 - L:13]) + [list(d[13]),
                                                        list(d[14])],
                        "act": [], "orig": [], "split": "validation",
                        "is_train": False, "fixed_depth": True}
                batch = collator([item])
                out = tc.run_forward(model, batch, device,
                                     grad_enabled=False)
                logits = out.logits
                labs = batch["labels"].to(device)
                sh = logits[..., :-1, :].contiguous()
                sl = labs[..., 1:].contiguous().clone()
                sl = sl.masked_fill(sl == eos_id, -100)
                n = int((sl != -100).sum())
                if n == 0:
                    continue
                loss = F.cross_entropy(
                    sh.view(-1, sh.shape[-1]), sl.reshape(-1),
                    ignore_index=-100, reduction="sum")
                nlls.append(float(loss) / n)
        mean_nll = sum(nlls) / max(len(nlls), 1)
        results["L{}".format(L)] = {
            "ppl": math.exp(mean_nll), "mean_nll": mean_nll,
            "dialogues": len(nlls)}
        print("L={}: PPL={:.4f} mean_nll={:.4f} ({} dlg)".format(
            L, results["L{}".format(L)]["ppl"], mean_nll, len(nlls)),
            flush=True)

    with open(a.out, "w") as f:
        json.dump(results, f, indent=2)
    print("protocol A done -> {}".format(a.out), flush=True)


if __name__ == "__main__":
    main()
