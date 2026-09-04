#!/usr/bin/env python3
"""THIN wrapper over the official CCM DailyDialog evaluator (review
ruling): data construction, collator, and the PPL metric all come from
the UNMODIFIED official code (DialogueDataset.eval_dataset /
DataCollatorForDialogue_LLAMA / CompSeq2SeqTrainer.evaluate_perp).
Only the model is swapped.

Runs as a hydra application exactly like the official train.py (the
config_path is resolved relative to the vendored src directory, so we
chdir there first).  Usage:

  # official adapter (foundation LoRA or released compression adapter)
  python scripts/eval_ccm_official.py +dialog=llama-7b \\
      model.model_name_or_path=/path/llama-7b-hf \\
      training.eval_path=/path/to/adapter training.do_train=false \\
      wandb.log=false

  # OUR trained arm (LoRA + Gamma), same official evaluator
  OUR_CKPT=/path/final.pt python scripts/eval_ccm_official.py \\
      +dialog=llama-7b model.model_name_or_path=/path/llama-7b-hf \\
      training.do_train=false wandb.log=false
"""
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

# hydra 1.3/1.4 resolves config_path relative to THIS SCRIPT's
# directory, not the cwd; point at the vendored config tree with a
# relative path.
import hydra  # noqa: E402
import torch  # noqa: E402

import train_ccm as tc  # noqa: E402
# module-level import: registers the hydra ConfigStore entry
# "base_config" BEFORE the @hydra.main decorator resolves the config.
from src.arguments import global_setup  # noqa: E402


@hydra.main(config_path="../third_party/ccm/src/config",
            config_name="config", version_base="1.1")
def main(args) -> None:
    args = global_setup(args)
    args.training.fp16 = True
    args.training.fp16_full_eval = True
    args.training.do_train = False
    args.training.do_eval = True

    from src.model import load_model, load_pretrained
    # The official foundation adapter (llama-7b-no, Step-1 default LoRA)
    # is a MERGED model: it goes through training.load_path, which the
    # official load_model merges into the base weights (merge=True).
    # Only the compression adapters go through training.eval_path.
    model, tokenizer = load_model(args)

    our_ckpt = os.environ.get("OUR_CKPT", "")
    if args.training.eval_path != '':
        load_pretrained(args.training.eval_path, model,
                        lora=args.training.peft)
        print("official adapter loaded via official path: {}"
              .format(args.training.eval_path), flush=True)
    elif our_ckpt:
        # Build-mode swap: construct OUR model object exactly as in
        # training (same arch/config/conditional-LoRA/Gamma pipeline)
        # and hand it to the OFFICIAL data+collator+metric.  The
        # official load_model result is discarded.
        import types
        del model
        torch.cuda.empty_cache()
        our_args = types.SimpleNamespace(
            arm="ours",
            model_name_or_path=args.model.model_name_or_path,
            relative_embedding="skip",
            lora_r=8, gamma_hidden=64)
        device = torch.device("cuda", 0)
        model = tc.build_model(our_args, device)
        model = tc.wrap_lora(model, our_args.lora_r)
        model.update_comp_token(
            [32000 + k for k in range(tc.N_TOK)],
            [32000 + tc.N_TOK + k for k in range(tc.N_TOK)])
        _gammas = tc.attach_gamma(model, hidden=our_args.gamma_hidden)
        # Gamma modules are plain attribute assignments (not registered
        # submodules), so model.to(fp16) does NOT traverse them and the
        # official evaluate_perp path (fp16, no autocast) would hit a
        # dtype mismatch.  Cast them explicitly.
        for _g in _gammas:
            _g.half()
        dummy = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=1e-3)
        tc.load_trainable(our_ckpt, model, dummy, device)
        print("our checkpoint loaded (build mode): {}".format(our_ckpt),
              flush=True)

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

    import json
    out = os.environ.get("EVAL_OUT", "eval_official.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print("official evaluator done -> {}".format(out), flush=True)


if __name__ == "__main__":
    main()
