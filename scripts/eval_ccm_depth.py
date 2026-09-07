#!/usr/bin/env python3
"""L9 depth evaluation (official protocol rebuild, review ruling).

Everything runs through the OFFICIAL data construction (DialogueDataset
sample_dialog + DataCollatorForDialogue_LLAMA): the CE is scored on the
LAST turn only, history = turns[:-2] concatenated (+turn[-2] as the
immediate context).  The self-made EOS-concatenation reference arms are
DELETED (superseded by the official constructions below).

Conditions:
  ccm mode   : compressed arms through the official merge_recur
               collator (ours / taskonly / official merge adapter).
  ref mode   : official no_ctx (neg_control: only turn[-2]) and
               full_ctx (online=False dataset: pure concatenation),
               each with an optionally supplied fine-tuned model
               (--official-adapter llama-7b-no, or --lora-ckpt, or the
               raw LLaMA).

--pooled switches the dataset to the official pooled val+test protocol
(clean_split=False); the default keeps the clean test split.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
CCM = HERE.parent / "third_party" / "ccm"
for p in (str(SRC), str(CCM), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch

import train_ccm as tc

# official bucket grouping (DialogueDataset._subsample n_turn lists)
TURN_BUCKETS = {"turn_3": [3], "turn_4": [4], "turn_6": [6],
                "turn_10": [10], "turn_14": [15]}


def bucket_of(dialog):
    n = len(dialog)
    for b, lst in TURN_BUCKETS.items():
        if n in lst:
            return b
    return None


def build_eval_dataset(args, tokenizer, pooled, online, comp_type):
    """Official DialogueDataset under the requested protocol."""
    from src.arguments import CompressionArguments
    from src.data.dialogue.data import DialogueDataset
    comp_args = CompressionArguments(attn_type="merge_recur",
                                     num_comp_tokens=tc.N_TOK,
                                     add_comp_token=True,
                                     relative_embedding=args.relative_embedding,
                                     comp_type=comp_type)
    dialog = DialogueDataset(tokenizer, comp_token=tokenizer.comp_token_id,
                             online=online, add_comp_token=True,
                             clean_split=not pooled)
    return dialog, comp_args


def build_collator(dialog, tokenizer, comp_args, comp_type, sum_recur):
    from src.data.dialogue.collator import DataCollatorForDialogue_LLAMA
    comp_args.comp_type = comp_type
    return DataCollatorForDialogue_LLAMA(
        dialog=dialog, tokenizer=tokenizer, comp_args=comp_args,
        comp_token=tokenizer.comp_token_id, sum_token=tokenizer.sum_token_id,
        padding="left", pad_token=tokenizer.pad_token_id,
        label_pad_token_id=-100)


def eval_split(model, collator, dialogs, device, limit, name,
               use_ccm=True):
    """One condition over bucketed dialogues -> per-bucket NLL.

    use_ccm=False forwards the raw LLaMA path (no attention_mask_comp
    kwarg); the uncompressed constructions carry no comp tokens so the
    plain forward is exact."""
    per_bucket = {}
    for b in sorted(TURN_BUCKETS):
        items = [d for d in dialogs if bucket_of(d) == b]
        if limit:
            items = items[:limit]
        items = [{"dialog": d, "act": [], "is_train": False}
                 for d in items]
        total = 0.0
        n_tok = 0
        with torch.no_grad():
            for i in range(0, len(items), 8):
                batch = collator(items[i:i + 8])
                if use_ccm:
                    out = tc.run_forward(model, batch, device,
                                         grad_enabled=False)
                else:
                    out = model(
                        input_ids=batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device))
                s, n = tc.task_ce_shifted(out, batch["labels"], device)
                total += float(s.detach())
                n_tok += n
        nll = total / max(n_tok, 1)
        per_bucket[b] = {"nll": nll, "tokens": n_tok,
                         "dialogues": len(items)}
        print("{} {}: nll={:.4f} ({} tok, {} dlg)".format(
            name, b, nll, n_tok, len(items)), flush=True)
    return per_bucket


def load_ccm_arm(args, device, ckpt):
    tokenizer = tc.build_tokenizer(args)
    model = tc.build_model(args, device)
    model = tc.wrap_lora(model, args.lora_r)
    model.update_comp_token([32000 + k for k in range(tc.N_TOK)],
                            [32000 + tc.N_TOK + k for k in range(tc.N_TOK)])
    tc.attach_gamma(model, hidden=args.gamma_hidden)
    dummy = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3)
    tc.load_trainable(ckpt, model, dummy, device)
    return tokenizer, model


def load_official_adapter(args, device, adapter_path):
    """Official llama-7b-no (default LoRA) onto the raw LLaMA."""
    from transformers.models.llama.modeling_llama import LlamaForCausalLM
    from peft import LoraConfig, get_peft_model
    _cfg = json.load(open(str(Path(adapter_path) / "adapter_config.json")))
    base = LlamaForCausalLM.from_pretrained(
        args.model_name_or_path, torch_dtype=torch.float16)
    base.resize_token_embeddings(32000 + 2 * tc.N_TOK)
    lora = LoraConfig(
        r=_cfg.get("r", 8), lora_alpha=_cfg.get("lora_alpha", 16),
        lora_dropout=_cfg.get("lora_dropout", 0.0),
        target_modules=_cfg.get("target_modules",
                                ["q_proj", "k_proj", "v_proj", "o_proj"]),
        task_type="CAUSAL_LM")
    model = get_peft_model(base, lora)
    sd = torch.load(str(Path(adapter_path) / "pytorch_model.bin"),
                    map_location="cpu")
    sd = {k.replace("lora_A.weight", "lora_A.default.weight")
            .replace("lora_B.weight", "lora_B.default.weight"): v
          for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model = model.to(device)
    model.eval()
    print("official adapter loaded ({} tensors remapped)".format(len(sd)),
          flush=True)
    return model


# official Table 25 time steps (truncation protocol); t=1 omitted: the
# official t=1 construction is not recoverable from the released code
# (prepare_input's n=1 branch degenerates), and the paper numbers for
# t=1 can only be cited, not reproduced.
TIME_STEPS = [2, 4, 8, 12, 20]


def eval_truncated(model, collator, dialogs, device, limit, name,
                   use_ccm=True):
    """Official truncation protocol: every test dialogue with
    len >= t is truncated to dialog[:t] and the CE is scored on the
    t-th turn only (sample_dialog predicts dialog[-1]).  The sample set
    is identical across time steps up to the len >= t filter (matching
    the official Table 23 note)."""
    per_t = {}
    for t in TIME_STEPS:
        items = [{"dialog": d[:t], "act": [], "is_train": False}
                 for d in dialogs if len(d) >= t]
        if limit:
            items = items[:limit]
        total = 0.0
        n_tok = 0
        with torch.no_grad():
            for i in range(0, len(items), 8):
                batch = collator(items[i:i + 8])
                if use_ccm:
                    out = tc.run_forward(model, batch, device,
                                         grad_enabled=False)
                else:
                    out = model(
                        input_ids=batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device))
                s, n = tc.task_ce_shifted(out, batch["labels"], device)
                total += float(s.detach())
                n_tok += n
        nll = total / max(n_tok, 1)
        per_t[str(t)] = {"nll": nll, "tokens": n_tok,
                         "dialogues": len(items)}
        print("{} t={}: nll={:.4f} ({} tok, {} dlg)".format(
            name, t, nll, n_tok, len(items)), flush=True)
    return per_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["ccm", "ref"], required=True)
    ap.add_argument("--taskonly-ckpt", default="")
    ap.add_argument("--ours-ckpt", default="")
    ap.add_argument("--official-adapter", default="")
    ap.add_argument("--lora-ckpt", default="")
    ap.add_argument("--model-name-or-path",
                    default="/root/autodl-tmp/llama-7b-hf")
    ap.add_argument("--dialog-mirror",
                    default="/root/autodl-tmp/dailydialog_mirror/ijcnlp_dailydialog")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--pooled", action="store_true",
                    help="official pooled val+test protocol (default: "
                         "clean test split)")
    ap.add_argument("--truncate", action="store_true",
                    help="official time-step truncation protocol "
                         "(matched cohort, Table 25 style) instead of "
                         "turn-count buckets")
    ap.add_argument("--out", default="eval_depth.json")
    a = ap.parse_args()

    device = torch.device("cuda", a.gpu)
    import os
    os.environ["DIALOG_MIRROR"] = a.dialog_mirror
    import types
    args = types.SimpleNamespace(
        arm="ours", model_name_or_path=a.model_name_or_path,
        dialog_mirror=a.dialog_mirror, relative_embedding="skip",
        lora_r=8, z_dim=128, rpbe_seed=0, sketch_dim=64, gamma_hidden=64)

    tokenizer = tc.build_tokenizer(args)
    results = {}

    if a.mode == "ccm":
        # compressed arms through the official merge_recur collator
        dialog, comp_args = build_eval_dataset(
            args, tokenizer, a.pooled, online=True, comp_type="online")
        collator = build_collator(dialog, tokenizer, comp_args,
                                  comp_type="online", sum_recur=True)
        eval_dialogs = (dialog.valset["dialog"] if a.pooled
                        else dialog.testset["dialog"])
        for name, ckpt in ([("ours", a.ours_ckpt),
                            ("taskonly", a.taskonly_ckpt)]):
            if not ckpt:
                continue
            _, model = load_ccm_arm(args, device, ckpt)
            if a.truncate:
                results[name] = eval_truncated(model, collator,
                                               eval_dialogs, device,
                                               a.limit, name,
                                               use_ccm=True)
            else:
                results[name] = eval_split(model, collator, eval_dialogs,
                                           device, a.limit, name)
            del model
            torch.cuda.empty_cache()
        with open(a.out, "w") as f:
            json.dump(results, f, indent=2)
        print("ccm done -> {}".format(a.out), flush=True)
        return

    # ref mode: official no_ctx / full_ctx constructions
    if a.official_adapter:
        model = load_official_adapter(args, device, a.official_adapter)
    elif a.lora_ckpt:
        _, model = load_ccm_arm(args, device, a.lora_ckpt)
    else:
        from transformers.models.llama.modeling_llama import LlamaForCausalLM
        model = LlamaForCausalLM.from_pretrained(
            a.model_name_or_path, torch_dtype=torch.float16).to(device)
        model.eval()
        print("ref arms use RAW pretrained LLaMA", flush=True)

    # full_ctx: online=False dataset (no comp token injection) ->
    # _concat_dialog is pure concatenation; CE on the last turn only.
    dialog_nc, comp_args_nc = build_eval_dataset(
        args, tokenizer, a.pooled, online=False, comp_type="online")
    collator_nc = build_collator(dialog_nc, tokenizer, comp_args_nc,
                                 comp_type="online", sum_recur=False)
    eval_dialogs = (dialog_nc.valset["dialog"] if a.pooled
                    else dialog_nc.testset["dialog"])
    results["full_ctx"] = (eval_truncated(model, collator_nc, eval_dialogs,
                                          device, a.limit, "full_ctx",
                                          use_ccm=False) if a.truncate
                           else eval_split(model, collator_nc,
                                           eval_dialogs, device, a.limit,
                                           "full_ctx", use_ccm=False))

    # no_ctx: neg_control -> context = dialog[-2] only.
    dialog_nx, comp_args_nx = build_eval_dataset(
        args, tokenizer, a.pooled, online=True, comp_type="neg_control")
    collator_nx = build_collator(dialog_nx, tokenizer, comp_args_nx,
                                 comp_type="neg_control", sum_recur=True)
    results["no_ctx"] = (eval_truncated(model, collator_nx, eval_dialogs,
                                        device, a.limit, "no_ctx",
                                        use_ccm=False) if a.truncate
                         else eval_split(model, collator_nx, eval_dialogs,
                                         device, a.limit, "no_ctx",
                                         use_ccm=False))

    with open(a.out, "w") as f:
        json.dump(results, f, indent=2)
    print("ref done -> {}".format(a.out), flush=True)


if __name__ == "__main__":
    main()
