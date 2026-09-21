"""MSC Stage-1 gradient smoke: verify the OFFICIAL training path.

Builds the official model through load_model() (SeparatedEmbedding +
conditional LoRA + peft_custom, exactly what src.train uses), collates
one MSC cut window with the official collator, runs ONE forward +
backward, and reports per-parameter-group gradient norms:

  - comp_embeddings  (the COMP/SUM rows) must receive a NONZERO gradient
    (the Qwen3 line once shipped a tokenizer-vocab misalignment that
    silently zeroed this gradient — this smoke exists to catch that
    class of bug on the Llama/MSC path);
  - lora_*           must be nonzero;
  - everything else  must be None (frozen backbone stays frozen).
"""
import os
import sys

sys.path.insert(0, "/root/autodl-tmp/third_party/ccm")
os.chdir("/root/autodl-tmp/third_party/ccm")

import torch
from hydra import compose, initialize

from src.arguments import global_setup
from src.model import load_model
from src.data.load import load_dataset_metric_collator

OVER = [
    "+msc=llama-7b",
    "model.model_name_or_path=/root/autodl-tmp/llama-7b-hf",
    "training.do_train=true",
    "training.comp.attn_type=merge_recur",
    "training.comp.num_comp_tokens=2",
    "training.comp.comp_type=online",
    "data.max_length=2048",
    "wandb.log=false",
]

with initialize(config_path="../third_party/ccm/src/config",
                version_base="1.1"):
    cfg = compose(config_name="config", overrides=OVER)
args = global_setup(cfg)
print("[smoke] comp: {} {} ntok={}".format(
    args.training.comp.comp_type, args.training.comp.attn_type,
    args.training.comp.num_comp_tokens), flush=True)

model, tokenizer = load_model(args)
model = model.to("cuda:0")
model.train()
print("[smoke] model loaded", flush=True)

train_dataset, eval_dataset, compute_metrics, collator = \
    load_dataset_metric_collator(args, model, tokenizer)
print("[smoke] dataset + collator OK, n_train={}".format(
    len(train_dataset)), flush=True)

# one real cut window through the OFFICIAL collator
row = dict(train_dataset[0])
batch = collator([row])
for k, v in batch.items():
    if hasattr(v, "to"):
        batch[k] = v.to("cuda:0")

comp_ids = tokenizer.comp_token_id
sum_ids = tokenizer.sum_token_id
with torch.autocast(device_type="cuda", dtype=torch.float16):
    out = model(input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                attention_mask_comp=batch["attention_mask_comp"])
logits = out.logits
labs = batch["labels"]
shift_logits = logits[..., :-1, :].contiguous()
shift_labels = labs[..., 1:].contiguous()
n_valid = int((shift_labels != -100).sum())
loss = torch.nn.functional.cross_entropy(
    shift_logits.view(-1, shift_logits.shape[-1]),
    shift_labels.reshape(-1), ignore_index=-100, reduction="sum")
loss.backward()

print("[smoke] loss={:.4f} n_valid_tokens={}".format(
    float(loss.detach()) / max(n_valid, 1), n_valid), flush=True)

g_comp = g_lora = g_other = g_frozen = 0.0
n_frozen = 0
for n, p in model.named_parameters():
    if not p.requires_grad:
        if p.grad is not None:
            print("[smoke] !! frozen param has grad:", n, flush=True)
            n_frozen += 1
        continue
    g = p.grad
    if g is None:
        print("[smoke] !! trainable param has NO grad:", n, flush=True)
        continue
    gn = float(g.detach().float().norm())
    if "comp_embeddings" in n:
        g_comp += gn
    elif "lora_" in n:
        g_lora += gn
    else:
        g_other += gn

print("[smoke] grad norms: comp={:.4e} lora={:.4e} other={:.4e} "
      "frozen_with_grad={}".format(g_comp, g_lora, g_other, n_frozen),
      flush=True)
ok = g_comp > 0 and g_lora > 0 and g_other == 0.0 and n_frozen == 0
print("[smoke] RESULT: {}".format("PASS" if ok else "FAIL"), flush=True)
sys.exit(0 if ok else 1)
