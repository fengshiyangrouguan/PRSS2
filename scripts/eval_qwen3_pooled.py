#!/usr/bin/env python3
"""Protocol B (official pooled evaluate_perp semantics) for the Qwen3
CCM port (feature_QWEN).

Mirrors the Llama-line eval_ccm_official.py OUR_CKPT build-mode path on
the official protocol: val+test MERGED (clean_split=False), five turn
buckets (turn_3/4/6/10/14 with the turn-14 bucket using 15 turns),
target-turn-only CE, EOS excluded, token-weighted perplexity as the
primary field plus the record-mean loss field.

Usage:
  python scripts/eval_qwen3_pooled.py --ckpt /path/checkpoint_step50.pt \
      --out eval_pooled_step50.json
  (--ckpt INIT keeps the random trainable init: step-0 baseline.)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name-or-path",
                    default="/root/autodl-tmp/qwen3-4b-instruct")
    ap.add_argument("--dialog-mirror",
                    default="/root/autodl-tmp/dailydialog_mirror/"
                            "ijcnlp_dailydialog")
    ap.add_argument("--ckpt", required=True,
                    help="checkpoint.pt (or INIT for the step-0 baseline)")
    ap.add_argument("--no-gamma", action="store_true",
                    help="ccm_merge-arm checkpoints carry no Gamma params "
                         "(skip attach_gamma)")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", default="eval_qwen3_pooled.json")
    ap.add_argument("--limit", type=int, default=0,
                    help="per-bucket dialogue cap (debug)")
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

    # Exact training build chain (same as load_ccm_arm on the qwen3 host)
    model = tc.build_model(args, device)
    model = tc.wrap_lora(model, args.lora_r)
    model.update_comp_token(
        [tokenizer.comp_token_id[k] for k in range(tc.N_TOK)],
        [tokenizer.sum_token_id[k] for k in range(tc.N_TOK)])
    if not a.no_gamma:
        tc.attach_gamma(model, hidden=args.gamma_hidden)
    dummy = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3)
    if a.ckpt != "INIT":
        payload = tc.load_trainable(a.ckpt, model, dummy, device)
        if "new_token_rows" in payload:
            n = 2 * tc.N_TOK
            model.get_input_embeddings().weight[-n:] = \
                payload["new_token_rows"]["input_embed_rows"].to(device)
            model.lm_head.weight[-n:] = \
                payload["new_token_rows"]["lm_head_rows"].to(device)
            print("[eval] new_token_rows restored", flush=True)
        else:
            raise RuntimeError(
                "checkpoint predates new_token_rows save; exact model "
                "reconstruction impossible")
        print("[eval] checkpoint loaded: {}".format(a.ckpt), flush=True)
    else:
        print("[eval] INIT: random trainable init kept", flush=True)
    model.eval()

    from src.arguments import CompressionArguments
    from src.data.dialogue.qwen3_data import (
        Qwen3DialogueDataset, Qwen3DialogueCollator)
    comp_args = CompressionArguments(attn_type="merge_recur",
                                     num_comp_tokens=tc.N_TOK,
                                     add_comp_token=True,
                                     relative_embedding="skip")
    dialog = Qwen3DialogueDataset(tokenizer, mirror=a.dialog_mirror,
                                  pooled=True)
    collator = Qwen3DialogueCollator(
        dataset=dialog, tokenizer=tokenizer, comp_args=comp_args,
        comp_token=tokenizer.comp_token_id,
        sum_token=tokenizer.sum_token_id,
        pad_token=tokenizer.pad_token_id, label_pad_token_id=-100)

    results = {}
    for name, items in dialog.eval_dataset.items():
        if a.limit:
            items = items[:a.limit]
        items = [dict(it) for it in items]
        for it in items:
            it["fixed_depth"] = True  # bucket turn count is authoritative
        tot_nll = 0.0
        tot_tok = 0
        sum_dlg_mean = 0.0
        n_dlg = 0
        with torch.no_grad():
            for i in range(0, len(items), 8):
                batch = collator(items[i:i + 8])
                out = tc.run_forward(model, batch, device,
                                     grad_enabled=False)
                logits = out.logits
                labs = batch["labels"].to(device)
                sh = logits[..., :-1, :].contiguous()
                sl = labs[..., 1:].contiguous().clone()
                # official _loglikelihood_clm: EOS excluded
                sl = sl.masked_fill(sl == eos_id, -100)
                B = sh.shape[0]
                row_n = (sl != -100).sum(-1)  # [B]
                row_loss = F.cross_entropy(
                    sh.view(-1, sh.shape[-1]), sl.reshape(-1),
                    ignore_index=-100, reduction="none").view(
                        B, -1).sum(-1)
                tot_nll += float(row_loss.detach().sum())
                tot_tok += int(row_n.sum())
                row_mean = row_loss / row_n.clamp(min=1)
                sum_dlg_mean += float(row_mean.detach().sum())
                n_dlg += B
        results[name] = {
            "perplexity": math.exp(tot_nll / max(tot_tok, 1)),
            "loss": sum_dlg_mean / max(n_dlg, 1),
            "tokens": tot_tok,
            "dialogues": n_dlg,
        }
        print("{}: perplexity={:.4f} loss={:.4f} ({} tok, {} dlg)".format(
            name, results[name]["perplexity"], results[name]["loss"],
            tot_tok, n_dlg), flush=True)

    with open(a.out, "w") as f:
        json.dump(results, f, indent=2)
    print("pooled protocol B done -> {}".format(a.out), flush=True)


if __name__ == "__main__":
    main()
