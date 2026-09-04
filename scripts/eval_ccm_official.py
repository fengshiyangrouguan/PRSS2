#!/usr/bin/env python3
"""THIN wrapper over the official CCM DailyDialog evaluator (review
ruling): data construction, collator, and the PPL metric all come from
the UNMODIFIED official code (DialogueDataset.eval_dataset /
DataCollatorForDialogue_LLAMA / CompSeq2SeqTrainer.evaluate_perp).
Only the model is swapped:

  --official-adapter <dir>   load an official adapter through the
                             official load path (llama-7b-no foundation
                             LoRA or a released compression adapter).
  --our-ckpt <file>          load OUR trained arm (LoRA + Gamma).

The official eval_dataset buckets are turn_3/4/6/10/14 (released
protocol, pooled val+test); evaluate_perp returns the official
perplexity = exp(token-normalized log-likelihood, EOS excluded).
"""
import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
CCM = HERE.parent / "third_party" / "ccm"
for p in (str(SRC), str(CCM)):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ["DIALOG_MIRROR"] = os.environ.get(
    "DIALOG_MIRROR",
    "/root/autodl-tmp/dailydialog_mirror/ijcnlp_dailydialog")

import torch
from hydra import initialize, compose

import train_ccm as tc


def build_official_args(model_name_or_path, attn_type="merge_recur",
                        num_comp_tokens=2, official_adapter="",
                        load_path=""):
    overrides = [
        "+dialog=llama-7b",
        "model.model_name_or_path={}".format(model_name_or_path),
        "training.comp.attn_type={}".format(attn_type),
        "training.comp.num_comp_tokens={}".format(num_comp_tokens),
        "training.comp.comp_type=online",
        "training.eval_path={}".format(official_adapter),
        "training.load_path={}".format(load_path),
        "training.do_train=false",
        "training.do_eval=true",
        "wandb.log=false",
        "training.output_dir=/tmp/ccm_official_eval",
    ]
    with initialize(config_path=str(CCM / "src" / "config"),
                    version_base="1.1"):
        cfg = compose(config_name="config", overrides=overrides)
    from src.arguments import global_setup
    return global_setup(cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name-or-path",
                    default="/root/autodl-tmp/llama-7b-hf")
    ap.add_argument("--official-adapter", default="")
    ap.add_argument("--our-ckpt", default="")
    ap.add_argument("--out", default="eval_official.json")
    a = ap.parse_args()

    args = build_official_args(
        a.model_name_or_path,
        official_adapter=a.official_adapter)
    args.training.fp16 = True
    args.training.fp16_full_eval = True
    args.training.do_eval = True
    args.training.do_train = False

    from src.model import load_model, load_pretrained
    model, tokenizer = load_model(args)
    if a.official_adapter:
        load_pretrained(args.training.eval_path, model,
                        lora=args.training.peft)
        print("official adapter loaded via official path: {}"
              .format(a.official_adapter), flush=True)
    elif a.our_ckpt:
        # Model swap: rebuild through our pipeline (same official arch,
        # conditional LoRA, Gamma) and restore our trainable state.
        our_args = tc.parse_args().parse_args([])
        our_args.arm = "ours"
        our_args.model_name_or_path = a.model_name_or_path
        our_args.relative_embedding = "skip"
        our_args.lora_r = 8
        our_args.gamma_hidden = 64
        model = tc.build_model(our_args, torch.device("cuda", 0))
        model = tc.wrap_lora(model, our_args.lora_r)
        model.update_comp_token(
            [32000 + k for k in range(tc.N_TOK)],
            [32000 + tc.N_TOK + k for k in range(tc.N_TOK)])
        tc.attach_gamma(model, hidden=our_args.gamma_hidden)
        dummy = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=1e-3)
        tc.load_trainable(a.our_ckpt, model, dummy, torch.device("cuda", 0))
        print("our checkpoint loaded: {}".format(a.our_ckpt), flush=True)

    from src.data.load import load_dataset_metric_collator
    _, eval_dataset, _, collator = load_dataset_metric_collator(
        args, model, tokenizer)

    from src.trainer_seq2seq import CompSeq2SeqTrainer
    trainer = CompSeq2SeqTrainer(
        model=model,
        args=args.training,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=None,
    )

    results = {}
    for eval_name, to_eval in eval_dataset.items():
        metrics = trainer.evaluate_perp(to_eval)
        results[eval_name] = {"perplexity": float(metrics["perplexity"]),
                              "loss": float(metrics["loss"])}
        print("{}: perplexity={:.4f} loss={:.4f}".format(
            eval_name, metrics["perplexity"], metrics["loss"]), flush=True)

    with open(a.out, "w") as f:
        json.dump(results, f, indent=2)
    print("official evaluator done -> {}".format(a.out), flush=True)


if __name__ == "__main__":
    main()
