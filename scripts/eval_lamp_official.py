#!/usr/bin/env python
"""LaMP-2 official merge host + official candidate-classification eval.

L1 (LaMP line): reproduce official CCM-merge on LaMP-2 (k=16, n_tok=4).

Host construction mirrors the dialog line build_official_host
(review round 7 protocol, fp32):
    llama-7b-hf (fp32 base)
      + llama-7b-no            (Step-1 LaMP LoRA, MERGED into base)
      + merge-ntok4 adapter    (Step-2 cond-LoRA + 8 GIST embeddings)
attn_type=merge: the model builds comp/sum masks internally from
input_ids (get_comp_sum_mask); the collator supplies the block-isolated
merge attention mask (attention_mask_comp).

Eval replicates the official trainer.evaluate_lamp_cls:
  - per sample, 15 candidate sequences [prompt + candidate]; per-sequence
    label logprob sum; argmax -> accuracy
  - BaseRetriever eval path: seed 1337 random 16-profile selection,
    reset once before the dataloader loop (deterministic)
  - split_batch forward (15 at a time) like _loglikelihood_clm

Run from /root/autodl-tmp/third_party/ccm (official relative paths):
  python scripts/eval_lamp_official.py --split dev
Official reference (paper Table 24, test @ k=16): CCM-merge = 83.9%.
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

CCM_ROOT = "/root/autodl-tmp/third_party/ccm"
sys.path.insert(0, CCM_ROOT)
os.chdir(CCM_ROOT)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["dev", "test"], default="dev")
    p.add_argument("--eval_batch_size", type=int, default=2,
                   help="samples per batch (each sample -> 15 candidate rows)")
    p.add_argument("--model_name_or_path", default="/root/autodl-tmp/llama-7b-hf")
    p.add_argument("--foundation", default="result/lamp/llama-7b-no")
    p.add_argument("--adapter",
                   default="result/lamp/finetune/llama-7b-no-online-merge-ntok4")
    p.add_argument("--n_tok", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--fp16_eval", action="store_true",
                   help="cast the whole host to fp16 before eval (official "
                        "fp16_full_eval protocol; ~3-4x faster on A800)")
    p.add_argument("--ckpt", default="",
                   help="train_lamp.py checkpoint: load its trainable state "
                        "(comp_embeddings + cond LoRA) over the official "
                        "adapter before eval")
    p.add_argument("--fp16_weights", action="store_true",
                   help="load the base in fp16 (matches train_lamp.py "
                        "--fp16_weights host)")
    p.add_argument("--with_gamma", action="store_true",
                   help="attach GammaOnetime (ours arm): the checkpoint's "
                        "gamma params are overlaid too, so eval sees the "
                        "exact training-time merge structure")
    p.add_argument("--gamma_hidden", type=int, default=64)
    return p.parse_args()


N_TOK = 4  # COMP tokens; merge doubles it to 8 (COMP + SUM)


def build_host(args, device):
    """Official merge host for LaMP-2 (dialog-line build_official_host
    pattern; see review round 7).  fp32 throughout; comp_embeddings and
    the Step-2 conditional LoRA are the only trainable parts (frozen for
    eval here)."""
    from transformers.models.llama.configuration_llama import LlamaConfig
    from src.arch.ccm_llama import LlamaForCausalLM_CCM
    from src.model import load_lora_weight, peft_custom
    from src.utils import SeparatedEmbedding
    from peft import LoraConfig

    config = LlamaConfig.from_pretrained(args.model_name_or_path)
    config.comp_relative_embedding = "skip"
    config.attn_type = "merge"
    base_dtype = (torch.float16 if args.fp16_weights else torch.float32)
    model = LlamaForCausalLM_CCM.from_pretrained(
        args.model_name_or_path, config=config, torch_dtype=base_dtype)
    model = model.to(device)

    # Step-1 LaMP LoRA merged into base
    load_lora_weight(args.foundation, model, merge=True)

    comp_ids = [32000 + k for k in range(args.n_tok)]
    sum_ids = [32000 + args.n_tok + k for k in range(args.n_tok)]
    model.update_comp_token(comp_ids, sum_ids)

    model.model.embed_tokens = SeparatedEmbedding(model.model.embed_tokens,
                                                  2 * args.n_tok)

    lora_cfg = LoraConfig().from_pretrained(args.adapter)
    model = peft_custom.get_peft_model(model, lora_cfg)
    load_lora_weight(args.adapter, model, merge=False)

    if args.with_gamma:
        # ours arm: attach GammaOnetime so the merge structure matches
        # training; the checkpoint overlay covers the gamma params too.
        sys.path.insert(0, "/root/autodl-tmp/src")
        from rpbe.hosts.ccm.ccm_patch import attach_gamma_onetime
        attach_gamma_onetime(model, hidden=args.gamma_hidden)

    for _p in model.parameters():
        _p.requires_grad_(False)
    if args.ckpt:
        # Overlay a train_lamp.py checkpoint (trainable state only:
        # comp_embeddings + conditional LoRA [+ Gamma when --with_gamma])
        # over the official adapter.
        payload = torch.load(args.ckpt, map_location="cpu",
                             weights_only=False)
        state = payload["model"]
        model_names = set(dict(model.named_parameters()).keys())
        # Review fix: fail on UNUSED checkpoint keys instead of silently
        # dropping them (e.g. gamma params loaded into a no-gamma host
        # used to evaluate silently with the wrong structure).
        unused = [k for k in state if k not in model_names]
        if unused:
            raise RuntimeError(
                "checkpoint has {} params the host does not own (gamma "
                "present but --with_gamma missing?): {}".format(
                    len(unused), unused[:4]))
        missing, applied = [], 0
        for _n, _p in model.named_parameters():
            if _n in state:
                _p.data.copy_(state[_n])
                applied += 1
            elif _p.requires_grad:
                missing.append(_n)
        if missing:
            raise RuntimeError(
                "checkpoint missing trainable params: " + str(missing))
        print(f"[lamp-host] ckpt overlay applied to {applied} params "
              f"(step={payload.get('step')})", flush=True)
    if args.fp16_eval:
        # Official fp16_full_eval: the whole model (comp_embeddings
        # included) is cast to fp16 right before evaluation; logprobs are
        # still computed in fp32 (logits.to(float) in loglikelihood_clm).
        model = model.half()
    model.eval()
    print("[lamp-official-host] merge-ntok%d host built (fp16_eval=%s)"
          % (args.n_tok, args.fp16_eval), flush=True)
    return model


def build_tokenizer(model_name_or_path, n_tok):
    from transformers import LlamaTokenizer
    tok = LlamaTokenizer.from_pretrained(model_name_or_path)
    tok.add_special_tokens(
        {"additional_special_tokens": [f"<COMP{k}>" for k in range(2 * n_tok)]})
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    tok.comp_token_id = tok.additional_special_tokens_ids[-2 * n_tok:-n_tok]
    tok.sum_token_id = tok.additional_special_tokens_ids[-n_tok:]
    return tok


def build_dataset_and_collator(tokenizer, model, args):
    from datasets import load_dataset
    from src.arguments import CompressionArguments
    from src.data.lamp import retriever, collator as lamp_collator

    # Only generate the requested split (train_questions.json is 540MB;
    # lamp.py's GeneratorBasedBuilder would materialize it otherwise).
    split_name = "validation" if args.split == "dev" else "test"
    ds = load_dataset("src/data/lamp/lamp.py", name="lamp2",
                      cache_dir=None, split=split_name)
    comp_args = CompressionArguments(attn_type="merge",
                                     num_comp_tokens=args.n_tok,
                                     add_comp_token=True,
                                     relative_embedding="skip")
    col = lamp_collator.LLaMACollatorForLaMP(
        tokenizer=tokenizer,
        retriever=retriever.BaseRetriever(k=16),
        comp_args=comp_args,
        model=model,
        padding="longest",
        comp_token=tokenizer.comp_token_id,
        sum_token=tokenizer.sum_token_id,
        pad_token=tokenizer.pad_token_id,
        k=16,
    )
    return ds, col


def loglikelihood_clm(model, batch, device, eos_id, split_batch=-1):
    """Replicate trainer_seq2seq._loglikelihood_clm (llama path).

    Unlike the official version, logprobs are reduced to per-sequence
    scalars group-by-group (no giant [B, T, vocab] fp32 cat), which is
    math-equivalent (softmax is per-row) and avoids OOM at large
    candidate batches.
    """
    batch = {k: v.to(device) for k, v in batch.items()}
    labels = batch["labels"]
    n = len(labels)
    ll = np.zeros(n)
    if split_batch > 0:
        assert n % split_batch == 0
        step = n // split_batch
        ranges = range(0, n, step)
    else:
        step = n
        ranges = range(0, n, step)
    with torch.no_grad():
        for i in ranges:
            batch_i = {k: v[i:i + step] for k, v in batch.items()}
            out = model(**batch_i)
            log_probs = F.log_softmax(out.logits.to(dtype=torch.float), dim=-1)
            labels_i = labels[i:i + step]
            for j in range(len(labels_i)):
                y = labels_i[j].unsqueeze(-1)
                y_shift = y[1:, :]
                lp_shift = log_probs[j][:-1, :]
                valid = (y_shift != eos_id) & (y_shift != -100)
                valid = valid.squeeze(-1)
                lp = torch.gather(lp_shift[valid], 1, y_shift[valid]).squeeze(-1)
                ll[i + j] = lp.sum().item()
    return {"loglikelihoods": ll}


def evaluate_lamp_cls(model, dataset, collator, lamp_index, device,
                      eval_batch_size):
    from src.data.lamp.utils import classification_candidates
    eos_id = collator.tokenizer.eos_token_id

    collator.retriever.reset_eval_seed()
    collate_fn = collator.collate_for_classification
    dataloader = DataLoader(dataset, collate_fn=collate_fn,
                            batch_size=eval_batch_size)

    # Base prior: candidate likelihoods without any profile context
    candidates_batch = collator.get_candidate_batch(lamp_index)
    base_ll = loglikelihood_clm(model, candidates_batch, device, eos_id)
    base_ll = base_ll["loglikelihoods"]
    print("[lamp-eval] base candidate loglik:", np.round(base_ll, 3), flush=True)

    preds_list, answers_list = [], []
    n_batches = len(dataloader)
    for bi, batch in enumerate(dataloader):
        answers = batch.pop("answers").numpy()
        bsz = len(answers)
        split_batch = len(classification_candidates[lamp_index])
        results = loglikelihood_clm(model, batch, device, eos_id,
                                    split_batch=split_batch)
        ll = results["loglikelihoods"].reshape(bsz, split_batch)
        preds = ll.argmax(axis=-1)
        preds_list += preds.tolist()
        answers_list += answers.tolist()
        if (bi + 1) % 50 == 0 or bi + 1 == n_batches:
            acc_sofar = (np.array(preds_list) == np.array(answers_list)).mean()
            print(f"[lamp-eval] batch {bi + 1}/{n_batches} "
                  f"(partial acc {acc_sofar * 100:.2f}%)", flush=True)

    preds = np.array(preds_list)
    answers = np.array(answers_list)
    acc = (preds == answers).mean() * 100
    return acc


def main():
    args = parse_args()
    device = args.device
    model = build_host(args, device)
    tokenizer = build_tokenizer(args.model_name_or_path, args.n_tok)
    ds, collator = build_dataset_and_collator(tokenizer, model, args)

    dataset = ds
    print(f"[lamp-eval] split={args.split} n={len(dataset)}", flush=True)

    acc = evaluate_lamp_cls(model, dataset, collator, 2, device,
                            args.eval_batch_size)
    print(f"ACC_{args.split.upper()} = {acc:.2f}%")
    if args.split == "test":
        print("official reference (Table 24, k=16): CCM-merge 83.9%")


if __name__ == "__main__":
    main()
