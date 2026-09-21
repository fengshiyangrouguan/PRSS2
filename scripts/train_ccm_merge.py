#!/usr/bin/env python3
"""Qwen3-line Stage A: STRICT official CCM Step-2 merge training.

SELF-CONTAINED entry point (user ruling 2026-09-17) — no dependency on
train_ccm.py, no RPBE machinery.  The official protocol, verbatim where
the host allows it:

  - random_k truncation per sample (k ~ U[3, len] on train only);
  - shuffled epoch-loop data stream (each dialogue exactly once per
    epoch, ~12 epochs over the 1000-step budget);
  - fixed accum-128-dialogues cadence (official
    per_device_train_batch_size=1 x gradient_accumulation_steps=128);
  - AdamW lr 3e-4, weight_decay 0, cosine over max_steps with 3%
    warmup, max_grad_norm 1.0;
  - trainable set = conditional LoRA (r=8, alpha=16, lora_dropout=0.05
    like the official LoraConfig) + the COMP/SUM input embeddings via
    the official SeparatedEmbedding (5,898,240 + 10,240 = 5,908,480,
    matches the port spec allowlist bit-for-bit); the lm_head is NOT
    extended (comp tokens have no output rows, official behavior);
  - loss = shifted CE on the target turn (EOS included in training,
    excluded only at evaluation — official behavior).

Qwen3-host deviations (recorded, not silent): bf16 instead of fp16,
Qwen chat-template block layout instead of the Llama sep-token layout,
--micro-batch 4 with bit-preserved equal-weight-per-dialogue
normalization (kernel efficiency only).
"""
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
CCM = HERE.parent / "third_party" / "ccm"
for p in (str(SRC), str(CCM), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

N_TOK = 2  # official dialog line: n_tok = 2 (2 COMP + 2 SUM)


def parse_args():
    p = argparse.ArgumentParser(
        "CCM merge training (official Step-2 protocol, self-contained)")
    p.add_argument("--model-name-or-path", required=True)
    p.add_argument("--host", default="qwen3",
                   choices=["llama", "qwen3", "gemma4"])
    p.add_argument("--foundation", default="",
                   help="gemma4 Stage-1 adapter checkpoint to merge into the base before the conditional LoRA (official two-stage)")
    p.add_argument("--dialog-mirror", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--grad-accum", type=int, default=128,
                   help="official gradient_accumulation_steps (dialogues "
                        "per optimizer step)")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--relative-embedding", default="skip",
                   choices=["skip", "base"])
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-dropout", type=float, default=0.05,
                   help="official LoraConfig dropout")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--micro-batch", type=int, default=4,
                   help="dialogues per collated batch (equal-weight "
                        "normalization preserved bit-for-bit)")
    p.add_argument("--checkpoint-every", type=int, default=50)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--resume-from", default="")
    return p.parse_args()


# ---------------------------------------------------------------------------
# builders (shared logic, local copies so this entry stays standalone)
# ---------------------------------------------------------------------------

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def build_tokenizer(args):
    if args.host == "qwen3":
        from transformers import AutoTokenizer
        from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
        tok = AutoTokenizer.from_pretrained(args.model_name_or_path)
        if tok.pad_token_id is None:
            tok.pad_token = "<|endoftext|>"
        tok.padding_side = "left"
        # Align the tokenizer vocab to config.vocab_size so comp ids land
        # >= 151936 and the SeparatedEmbedding routes them to the
        # TRAINABLE comp rows (see train_ccm.build_tokenizer).
        cfg_vocab = Qwen3Config.from_pretrained(
            args.model_name_or_path).vocab_size
        if len(tok) < cfg_vocab:
            tok.add_tokens(
                ["<|extra_{}|>".format(i)
                 for i in range(cfg_vocab - len(tok))])
        added = [f"<COMP{k}>" for k in range(N_TOK)] \
            + [f"<SUM{k}>" for k in range(N_TOK)]
        tok.add_special_tokens({"additional_special_tokens": added})
        ids = [tok.convert_tokens_to_ids(f"<COMP{k}>") for k in range(N_TOK)]  + [tok.convert_tokens_to_ids(f"<SUM{k}>") for k in range(N_TOK)]
        assert ids[0] >= cfg_vocab
        tok.comp_token_id = ids[:N_TOK]
        tok.sum_token_id = ids[N_TOK:]
        return tok
    if args.host == "gemma4":
        from transformers import AutoTokenizer
        from transformers.models.gemma4.configuration_gemma4 import (
            Gemma4TextConfig)
        tok = AutoTokenizer.from_pretrained(args.model_name_or_path)
        if tok.pad_token_id is None:
            tok.pad_token_id = 0
            tok.pad_token = "<pad>"
        tok.padding_side = "left"
        cfg_vocab = Gemma4TextConfig.from_pretrained(
            args.model_name_or_path).vocab_size
        if len(tok) < cfg_vocab:
            tok.add_tokens(
                ["<|extra_{}|>".format(i)
                 for i in range(cfg_vocab - len(tok))])
        added = [f"<COMP{k}>" for k in range(N_TOK)]             + [f"<SUM{k}>" for k in range(N_TOK)]
        tok.add_special_tokens({"additional_special_tokens": added})
        ids = [tok.convert_tokens_to_ids(f"<COMP{k}>")
               for k in range(N_TOK)] + [tok.convert_tokens_to_ids(
                   f"<SUM{k}>") for k in range(N_TOK)]
        assert ids[0] >= cfg_vocab
        tok.comp_token_id = ids[:N_TOK]
        tok.sum_token_id = ids[N_TOK:]
        return tok
    from transformers import LlamaTokenizer
    tok = LlamaTokenizer.from_pretrained(args.model_name_or_path)
    tok.pad_token = tok.eos_token
    tok.pad_token_id = tok.pad_token_id if tok.pad_token_id is not None \
        else tok.eos_token_id
    tok.bos_token_id = tok.bos_token_id or 1
    tok.eos_token_id = tok.eos_token_id or 2
    tok.padding_side = "left"
    added = [f"<COMP{k}>" for k in range(N_TOK)] \
        + [f"<SUM{k}>" for k in range(N_TOK)]
    tok.add_special_tokens({"additional_special_tokens": added})
    ids = [tok.convert_tokens_to_ids(f"<COMP{k}>") for k in range(N_TOK)]  + [tok.convert_tokens_to_ids(f"<SUM{k}>") for k in range(N_TOK)]
    tok.comp_token_id = ids[:N_TOK]
    tok.sum_token_id = ids[N_TOK:]
    return tok


def build_model_merge(args, device):
    """Official-host semantics: SeparatedEmbedding with TRAINABLE
    COMP/SUM rows; the lm_head stays at the base vocab (comp tokens have
    no output rows); no resize_token_embeddings."""
    if args.host == "gemma4":
        import re as _re
        from transformers.models.gemma4.configuration_gemma4 import (
            Gemma4TextConfig)
        from transformers.models.gemma4.modeling_gemma4 import (
            Gemma4ForConditionalGeneration)
        from src.arch.ccm_gemma4 import Gemma4ForCausalLM_CCM
        text_cfg = Gemma4TextConfig.from_pretrained(
            args.model_name_or_path)
        text_cfg.comp_relative_embedding = args.relative_embedding
        model = Gemma4ForCausalLM_CCM(text_cfg)
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        full = Gemma4ForConditionalGeneration.from_pretrained(
            args.model_name_or_path, torch_dtype=dtype)
        prefix = "model.language_model."
        text_sd = {}
        for k, v in full.state_dict().items():
            if k.startswith(prefix):
                text_sd["model." + k[len(prefix):]] = v
            elif k == "lm_head.weight":
                text_sd[k] = v
        del full
        torch.cuda.empty_cache() if device.type == "cuda" else None
        missing, unexpected = model.load_state_dict(text_sd, strict=False)
        if unexpected:
            raise RuntimeError("unexpected keys: {}".format(
                unexpected[:8]))
        model.to(device, dtype)
        if args.foundation:
            ck = torch.load(args.foundation, map_location=device,
                            weights_only=False)
            tsd = ck.get("trainable_state_dict", ck)
            n_merged = 0
            with torch.no_grad():
                for k in list(tsd):
                    if ".lora_A." not in k:
                        continue
                    b_key = k.replace(".lora_A.", ".lora_B.")
                    base_key = _re.sub(
                        r"^base_model\.model\.model\.layers\.(\d+)\."
                        r"self_attn\.(\w+_proj)\.lora_A\.default\."
                        r"weight$",
                        r"model.layers.\1.self_attn.\2.weight", k)
                    if base_key == k:
                        continue
                    A = tsd[k].float().to(device)
                    B = tsd[b_key].float().to(device)
                    delta = (B @ A) * (16.0 / 8.0)
                    tgt = dict(model.named_parameters())[base_key]
                    tgt.data += delta.to(tgt.dtype)
                    n_merged += 1
            print("[merge] foundation merged: {} LoRA modules".format(
                n_merged), flush=True)
        # SeparatedEmbedding (official-host semantics, same as qwen3):
        # the COMP/SUM rows live in a TRAINABLE separate table routed by
        # id >= vocab_size; the lm_head keeps the base vocab (comp
        # tokens have no output rows).  No resize on gemma either.
        from src.utils import SeparatedEmbedding
        model.model.embed_tokens = SeparatedEmbedding(
            model.model.embed_tokens, 2 * N_TOK)
        # The PLE table is looked up by the SAME input_ids — without a
        # resize its 262144 rows cannot serve the comp ids (device
        # assert on the lookup).  Grow it to match (zero rows, frozen).
        model.model.resize_ple_embeddings(
            text_cfg.vocab_size + 2 * N_TOK)
        model.update_comp_token(
            [text_cfg.vocab_size + k for k in range(N_TOK)],
            [text_cfg.vocab_size + N_TOK + k for k in range(N_TOK)])
        return model.to(device)
    if args.host != "qwen3":
        raise NotImplementedError(
            "train_ccm_merge currently targets the qwen3 host")
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
    from src.arch.ccm_qwen3 import Qwen3ForCausalLM_CCM
    from src.utils import SeparatedEmbedding
    config = Qwen3Config.from_pretrained(args.model_name_or_path)
    config.comp_relative_embedding = "skip"
    model = Qwen3ForCausalLM_CCM.from_pretrained(
        args.model_name_or_path, config=config,
        torch_dtype=torch.bfloat16 if device.type == "cuda"
        else torch.float32)
    model.model.embed_tokens = SeparatedEmbedding(
        model.model.embed_tokens, 2 * N_TOK)
    model.update_comp_token(
        [config.vocab_size + k for k in range(N_TOK)],
        [config.vocab_size + N_TOK + k for k in range(N_TOK)])
    return model.to(device)


def wrap_lora_merge(model, r, dropout):
    """Conditional LoRA with the OFFICIAL dropout + exact trainable set:
    LoRA params + the SeparatedEmbedding comp rows."""
    from peft import LoraConfig
    from src import peft_custom
    cfg = LoraConfig(r=int(r), lora_alpha=16, lora_dropout=float(dropout),
                     bias="none", task_type="CAUSAL_LM",
                     target_modules=["q_proj", "k_proj", "v_proj",
                                     "o_proj"])
    model = peft_custom.get_peft_model(model, cfg)
    for _p in model.parameters():
        _p.requires_grad_(False)
    for _n, _p in model.named_parameters():
        if "lora_" in _n:
            _p.requires_grad_(True)
    # Unified path: both qwen3 and gemma4 use SeparatedEmbedding (the
    # gemma slice hack was invalid — nn.Parameter requires_grad is
    # tensor-level, a row slice assignment does nothing).
    model.base_model.model.model.embed_tokens.comp_embeddings.weight \
        .requires_grad_(True)
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("[merge] trainable params:", n_tr, flush=True)
    return model


def build_dataset(args, tokenizer):
    from src.arguments import CompressionArguments
    os.environ["DIALOG_MIRROR"] = args.dialog_mirror
    comp_args = CompressionArguments(attn_type="merge_recur",
                                     num_comp_tokens=N_TOK,
                                     add_comp_token=True,
                                     relative_embedding=args.relative_embedding)
    if args.host == "gemma4":
        from src.data.dialogue.gemma4_data import (
            Gemma4DialogueDataset, Gemma4DialogueCollator)
        dialog = Gemma4DialogueDataset(tokenizer,
                                       mirror=args.dialog_mirror)
        collator = Gemma4DialogueCollator(
            dataset=dialog, tokenizer=tokenizer, comp_args=comp_args,
            comp_token=tokenizer.comp_token_id,
            sum_token=tokenizer.sum_token_id,
            pad_token=tokenizer.pad_token_id,
            label_pad_token_id=-100)
        return dialog, collator
    if args.host == "qwen3":
        from src.data.dialogue.qwen3_data import (
            Qwen3DialogueDataset, Qwen3DialogueCollator)
        dialog = Qwen3DialogueDataset(tokenizer, mirror=args.dialog_mirror)
        collator = Qwen3DialogueCollator(
            dataset=dialog, tokenizer=tokenizer, comp_args=comp_args,
            comp_token=tokenizer.comp_token_id,
            sum_token=tokenizer.sum_token_id,
            pad_token=tokenizer.pad_token_id,
            label_pad_token_id=-100)
        return dialog, collator
    from src.data.dialogue.data import DialogueDataset
    from src.data.dialogue.collator import DataCollatorForDialogue_LLAMA
    dialog = DialogueDataset(tokenizer, comp_token=tokenizer.comp_token_id,
                             online=True, add_comp_token=True,
                             clean_split=True)
    collator = DataCollatorForDialogue_LLAMA(
        dialog=dialog, tokenizer=tokenizer, comp_args=comp_args,
        comp_token=tokenizer.comp_token_id, sum_token=tokenizer.sum_token_id,
        padding="left", pad_token=tokenizer.pad_token_id,
        label_pad_token_id=-100)
    return dialog, collator


def run_forward(model, batch, device, grad_enabled):
    """Forward with the visibility mask folded in (merge_recur).  The
    autocast dtype follows the backbone weight dtype (bf16 Qwen3)."""
    ctx = torch.enable_grad() if grad_enabled else torch.no_grad()
    amc = batch.get("attention_mask_comp")
    _dt = next(model.parameters()).dtype
    _acast_dt = _dt if _dt in (torch.float16, torch.bfloat16) \
        else torch.float16
    with ctx:
        with torch.autocast(device_type="cuda", dtype=_acast_dt,
                            enabled=(device.type == "cuda")):
            return model(input_ids=batch["input_ids"].to(device),
                         attention_mask=batch["attention_mask"].to(device),
                         attention_mask_comp=amc.to(device)
                         if amc is not None else None)


def task_ce_rows(out, labels, device):
    """Per-dialogue-row shifted CE: (row_sum [B], row_n [B]).  The
    row-wise reduction keeps the equal-weight-per-dialogue objective
    intact when several dialogues share one collated batch."""
    logits = out.logits
    labs = labels.to(device)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labs[..., 1:].contiguous()
    B = shift_logits.shape[0]
    row_n = (shift_labels != -100).sum(-1)
    loss = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]),
        shift_labels.reshape(-1), ignore_index=-100, reduction="none")
    row_sum = loss.view(B, -1).sum(-1)
    return row_sum, row_n


def task_ce_shifted(out, labels, device):
    """Official CCM task CE: shifted sum + valid count."""
    logits = out.logits
    labs = labels.to(device)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labs[..., 1:].contiguous()
    n_valid = int((shift_labels != -100).sum())
    loss = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]),
        shift_labels.view(-1), ignore_index=-100, reduction="sum")
    return loss, n_valid


def save_trainable(path, model, **extra):
    """Trainable params only (LoRA + the SeparatedEmbedding comp rows;
    the frozen backbone is not stored)."""
    payload = {
        "model": {n: p.detach().cpu() for n, p in model.named_parameters()
                  if p.requires_grad},
    }
    payload.update(extra)
    torch.save(payload, path)


# ---------------------------------------------------------------------------
# training loop (official protocol)
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    seed_all(args.seed)
    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    tokenizer = build_tokenizer(args)
    model = build_model_merge(args, device)
    model = wrap_lora_merge(model, args.lora_r, args.lora_dropout)
    model.update_comp_token(
        [tokenizer.comp_token_id[k] for k in range(N_TOK)],
        [tokenizer.sum_token_id[k] for k in range(N_TOK)])
    dialog, collator = build_dataset(args, tokenizer)
    train_items = dialog.trainset
    n_items = len(train_items)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    total_steps = max(1, args.max_steps)
    warmup_steps = max(1, int(0.03 * total_steps))

    def _lr_lambda(s):
        if s < warmup_steps:
            return float(s) / float(warmup_steps)
        progress = float(s - warmup_steps) / float(
            max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

    # official data stream: shuffled epoch loop
    shuf_order = list(range(n_items))
    random.shuffle(shuf_order)
    epoch_pos = 0
    n_epochs = 0

    def next_batch():
        nonlocal epoch_pos, shuf_order, n_epochs
        items = []
        for _ in range(args.micro_batch):
            if epoch_pos >= len(shuf_order):
                random.shuffle(shuf_order)
                epoch_pos = 0
                n_epochs += 1
            items.append(dict(train_items[int(shuf_order[epoch_pos])]))
            epoch_pos += 1
        # NO fixed_depth -> the collator applies the official random_k
        return collator(items)

    step = 0
    last_logged = -1
    t_start = time.time()
    pending = []
    total_tokens = 0
    total_loss = 0.0
    model.train()

    while step < args.max_steps:
        batch = next_batch()
        pending.append(batch)
        n_dial = sum(int(b["input_ids"].shape[0]) for b in pending)
        if n_dial >= args.grad_accum:
            optimizer.zero_grad(set_to_none=True)
            task_sum = 0.0
            n_tokens = 0
            for b in pending:
                fwd_out = run_forward(model, b, device, grad_enabled=True)
                if args.micro_batch > 1:
                    row_sum, row_n = task_ce_rows(fwd_out, b["labels"],
                                                  device)
                    row_mean = row_sum / row_n.clamp(min=1)
                    loss_scale = row_mean.sum() / float(n_dial)
                    task_sum += float(row_sum.detach().sum())
                    n_tokens += int(row_n.sum())
                else:
                    task_raw, n_valid = task_ce_shifted(
                        fwd_out, b["labels"], device)
                    # official accelerate semantics: loss /
                    # gradient_accumulation_steps
                    loss_scale = task_raw / max(n_valid, 1) \
                        / float(len(pending))
                    task_sum += float(task_raw.detach())
                    n_tokens += n_valid
                loss_scale.backward()
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            optimizer.step()
            scheduler.step()
            step += 1
            total_loss += task_sum
            total_tokens += n_tokens
            pending = []
            if step % args.log_every == 0 and step != last_logged:
                last_logged = step
                print("step={} ce_token={:.4f} mbs={} sec={:.1f} "
                      "epoch={}".format(
                          step, total_loss / max(total_tokens, 1),
                          step * args.grad_accum,
                          time.time() - t_start, n_epochs), flush=True)
            if step % args.checkpoint_every == 0:
                save_trainable(out / f"checkpoint_step{step}.pt", model,
                               step=step)
                print("[ckpt] saved step {}".format(step), flush=True)

    save_trainable(out / "final.pt", model, step=step)
    save_json(out / "summary.json", {
        "arm": "ccm_merge_official", "protocol": "official Step-2 merge",
        "seed": args.seed, "steps": step,
        "mean_task_ce_per_token": total_loss / max(total_tokens, 1),
        "task_valid_tokens": total_tokens, "epochs": n_epochs,
        "lr": args.lr, "lora_r": args.lora_r,
        "lora_dropout": args.lora_dropout, "grad_accum": args.grad_accum,
        "micro_batch": args.micro_batch,
        "trainable_comp_embeddings": True,
    })
    print("official merge training done -> {}".format(out), flush=True)


if __name__ == "__main__":
    main()
