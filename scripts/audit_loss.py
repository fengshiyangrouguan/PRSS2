"""audit_loss.py — deterministic per-dim / per-group audit on one val batch.

Loads an arm checkpoint, runs one val-split batch through the model in train
mode, and reports:
  * coverage assertion (every trainable param in an optimizer)
  * per-dimension diffusion MSE
  * gradient norm per component group after backward
"""
import os, sys, argparse
import numpy as np
os.environ.setdefault("HF_ENDPOINT","https://hf-mirror.com"); os.environ.setdefault("HF_HUB_OFFLINE","1")
os.environ.setdefault("LLAMA2_LOCAL_PATH","/root/autodl-tmp/Llama-2-7b-hf")
sys.path.insert(0,"/root/autodl-tmp/vla"); sys.path.insert(0,"/root/autodl-tmp/libero-mem-code")
import torch
from peft import LoraConfig, get_peft_model
from vla import load_vla
from vla.datasets.hdf5_dataset import get_hdf5_decision_stream_dataset_and_collator

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--data-root", default="/root/autodl-tmp/libero-mem")
ap.add_argument("--base", default="/root/autodl-tmp/openvla-7b-prismatic/checkpoints/step-295000-epoch-40-loss=0.2200.pt")
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
arm = ck["arm"]; mem = ck.get("mem_length", 16)
is_gamma = arm in ("gamma-task","gamma-rpbe")
torch.manual_seed(args.seed); np.random.seed(args.seed)

vla = load_vla(model_id_or_path=args.base, hf_token=None, load_for_training=True,
    use_bf16=True, action_dim=7, future_action_window_size=15, action_model_type="DiT-L",
    use_ema=False, dataloader_type="stream", mem_length=mem, retrieval_layers=2,
    use_timestep_pe=True, fusion_type="gate", consolidate_type="tome", update_fused=False,
    per_token_size=256, use_rpbe_gamma=is_gamma, gamma_rank=64, gamma_alpha_init=1.0,
    rpbe_merge_records=is_gamma, rpbe_task_grad=is_gamma, rpbe_seed=args.seed)
vla.vlm.requires_grad_(False)
lc = ck["lora_config"]
vla.vlm.llm_backbone.llm = get_peft_model(vla.vlm.llm_backbone.llm, LoraConfig(
    r=lc["r"], lora_alpha=lc["lora_alpha"], lora_dropout=lc["lora_dropout"],
    target_modules="all-linear", task_type="CAUSAL_LM"))
named = dict(vla.named_parameters())
for n, t in ck["model"].items():
    if n in named and named[n].requires_grad:
        named[n].data.copy_(t.to(named[n].dtype))
vla.train()

# ---- coverage assertion ----
lora_params = [p for n,p in vla.named_parameters() if p.requires_grad and "lora_" in n]
gamma_params = list(vla.gamma.parameters()) if vla.gamma is not None else []
gamma_ids = {id(p) for p in gamma_params}
task_modules = [vla.cog_mem_bank, vla.per_mem_bank, vla.per_compr, vla.action_model]
lora_ids = {id(x) for x in lora_params}
task_params = [p for m in task_modules for p in m.parameters()
               if p.requires_grad and id(p) not in lora_ids and id(p) not in gamma_ids]
task_params += lora_params
covered = {id(p) for p in task_params} | gamma_ids
all_tr = {id(p) for p in vla.parameters() if p.requires_grad}
miss = all_tr - covered
print("coverage: all_trainable=%d covered=%d missing=%d" % (len(all_tr), len(covered), len(miss)))
assert not miss, "PARAM COVERAGE GAP: %d params not in any optimizer" % len(miss)
print("COVERAGE_OK arm=%s" % arm)

# ---- build 4 rows of val demo_0 ----
tok = vla.vlm.llm_backbone.get_tokenizer(); itr = vla.vlm.vision_backbone.get_image_transform()
val_ds, _, collator = get_hdf5_decision_stream_dataset_and_collator(
    data_root=args.data_root, tokenizer=tok, image_transform=itr,
    prompt_builder_fn=vla.vlm.llm_backbone.prompt_builder_fn,
    future_action_window_size=15, seed=args.seed, split="val",
    pad_token_id=tok.pad_token_id, task_filter="KITCHEN_SCENE1_3")
rows = [r for i, r in zip(range(4), val_ds.iter_episode(0))]
batch = collator(rows)

def pv_cuda(b):
    pv = b["pixel_values"]
    if isinstance(pv, dict):
        return {k: v.to("cuda", dtype=torch.bfloat16) for k, v in pv.items()}
    return pv.to("cuda", dtype=torch.bfloat16)

# ---- monkeypatch action_model.loss to record per-dim MSE ----
am = vla.action_model
orig_loss = am.loss
per_dim_store = {}
def loss_rec(x, z, per_token, action_mask=None, dim_weight=None):
    noise = torch.randn_like(x)
    timestep = torch.randint(0, am.diffusion.num_timesteps, (x.size(0),), device=x.device)
    x_t = am.diffusion.q_sample(x, timestep, noise)
    noise_pred = am.net(x_t, timestep, z, per_token=per_token)
    sq = (noise_pred - noise).pow(2)                  # [B,T,7]
    if action_mask is not None:
        valid = action_mask[..., None].to(sq.dtype)
    else:
        valid = torch.ones_like(sq)
    if dim_weight is not None:
        dw = dim_weight.view(1, 1, -1)
    else:
        dw = torch.ones_like(sq)
    per_dim_store["valid_count"] = (valid > 0).sum().float().item()
    # mean over B,T of valid entries, per dim
    valid_dim = valid.squeeze(-1)                     # [B,T] bool
    per_dim = []
    for d in range(sq.size(-1)):
        sel = sq[..., d] * valid.squeeze(-1)
        per_dim.append(sel.sum() / (valid.squeeze(-1).sum() + 1e-8))
    per_dim_store["per_dim"] = per_dim
    return (sq * valid * dw).sum() / ((valid * dw).sum() + 1e-8)
am.loss = loss_rec

with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
    loss, _ = vla(input_ids=batch["input_ids"].to("cuda"),
        attention_mask=batch["attention_mask"].to("cuda"),
        actions=batch["actions"].to("cuda", dtype=torch.bfloat16),
        action_masks=batch["action_masks"].to("cuda"),
        pixel_values=pv_cuda(batch), labels=batch["labels"].to("cuda"),
        timesteps=batch["timesteps"], episode_ids=batch["episode_ids"],
        output_hidden_states=True, repeated_diffusion_steps=4)
print("weighted loss on val batch: %.4f" % loss.item())
pd = [float(x) for x in per_dim_store["per_dim"]]
names = ["x","y","z","rx","ry","rz","gripper"]
print("per-dim MSE (valid-only):")
for n, v in zip(names, pd):
    print("  %-6s %.5f" % (n, v))
am.loss = orig_loss

loss.backward()
def gnorm(params, name):
    gs = [p.grad.detach().float().norm()**2 for p in params if p.grad is not None]
    n = torch.sqrt(sum(gs)).item() if gs else 0.0
    ng = sum(1 for p in params if p.grad is not None)
    print("grad_norm[%s]=%.4e (%d/%d)" % (name, n, ng, len(params)))
gnorm([vla.action_model], "action_model")
gnorm(lora_params, "lora")
gnorm([vla.cog_mem_bank], "cog_memory")
gnorm([vla.per_mem_bank], "per_memory")
gnorm([vla.per_compr], "per_compr")
gnorm(gamma_params, "gamma")
print("AUDIT_DONE")
