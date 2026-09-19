"""Stage 2 of the T1 checkpoint-restore audit — NO ROLLOUT, NO FILE WRITES.

Builds the model with the EXACT T1 training architecture (same load_vla kwargs
as build_vla(), plus the recorded LoRA config), then verifies that every one of
the delta's 880 trainable tensors maps onto a live parameter with an identical
shape, and that applying them works (strict load per module, in memory only).

Nothing is saved to disk; no simulator, no policy server, no GPU rollout.
"""
from __future__ import annotations

import collections
import json
import os
import sys
from pathlib import Path

import torch

MEMVLA = Path("/root/autodl-tmp/PRSS2/third_party/memoryvla")
BASE = Path("/root/autodl-tmp/openvla-7b-prismatic/checkpoints/"
            "step-295000-epoch-40-loss=0.2200.pt")
DELTA = Path("/root/autodl-tmp/runs/t1_avg_seed42/best.pt")

sys.path.insert(0, str(MEMVLA))
sys.path.insert(0, "/root/autodl-tmp/PRSS2/src")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("LLAMA2_LOCAL_PATH", "/root/autodl-tmp/Llama-2-7b-hf")

print("=" * 74)
print("STAGE 2 — live-architecture restore audit (read-only)")
print("=" * 74, flush=True)

d = torch.load(DELTA, map_location="cpu", weights_only=False)
delta = d["model"]
lora_cfg = d["lora_config"]
print(f"delta: {len(delta)} tensors | arm={d['arm']!r} step={d['step']} "
      f"seed={d['seed']}", flush=True)
print(f"lora_config: {lora_cfg}", flush=True)

# ---------------------------------------------------------------- build
from vla import load_vla          # noqa: E402
from peft import LoraConfig, get_peft_model   # noqa: E402

# Mirror build_vla() in train_libero_mem_rpbe.py for the `avg` arm.
# load_for_training=False -> empty LLM skeleton, then the base checkpoint's
# llm_backbone weights are loaded by MemoryVLA.from_pretrained.
print("\nbuilding model (load_vla, load_for_training=False) ...", flush=True)
vla = load_vla(
    model_id_or_path=str(BASE),
    hf_token=None,
    load_for_training=False,
    use_bf16=True,
    action_dim=7,
    future_action_window_size=15,
    action_model_type="DiT-L",
    use_ema=False,
    dataloader_type="stream",
    mem_length=16,
    retrieval_layers=2,
    use_timestep_pe=True,
    fusion_type="gate",
    consolidate_type="tome",
    update_fused=False,
    per_token_size=256,
    use_rpbe_gamma=False,        # arm == avg
    gamma_rank=64,
    gamma_alpha_init=1.0,
    rpbe_merge_records=False,
    rpbe_task_grad=False,
    rpbe_seed=int(d["seed"]),
)
print("model built", flush=True)

# ------------------------------------------------------------ attach LoRA
print("\nattaching LoRA (same recipe as build_vla) ...", flush=True)
peft_cfg = LoraConfig(
    r=lora_cfg["r"], lora_alpha=lora_cfg["lora_alpha"],
    lora_dropout=lora_cfg["lora_dropout"], target_modules="all-linear",
)
vla.vlm.llm_backbone.llm = get_peft_model(vla.vlm.llm_backbone.llm, peft_cfg)
print("LoRA attached", flush=True)

# --------------------------------------------------------- build the index
model_params = dict(vla.named_parameters())
print(f"\nmodel has {len(model_params)} named parameters", flush=True)


def classify(k: str) -> str:
    if ".lora_A." in k or ".lora_B." in k:
        return "LoRA"
    if k.startswith("vlm."):
        return "vlm(frozen)"
    if k.startswith("action_model."):
        return "DiT/action"
    if k.startswith("cog_mem_bank.") or k.startswith("per_mem_bank."):
        return "memory banks"
    if k.startswith("per_compr."):
        return "per_compr"
    if k.startswith("reg_head."):
        return "reg_head"
    return "other"


cat = collections.Counter()
matched, bad_shape, missing = [], [], []
delta_total_bytes = 0
for k, t in delta.items():
    cat[classify(k)] += 1
    delta_total_bytes += t.numel() * t.element_size()
    p = model_params.get(k)
    if p is None:
        missing.append(k)
    elif tuple(p.shape) != tuple(t.shape):
        bad_shape.append((k, tuple(t.shape), tuple(p.shape)))
    else:
        matched.append(k)

print("\n" + "=" * 74)
print("PER-CATEGORY MATCH REPORT")
print("=" * 74)
print(f"{'category':16s} {'delta':>7s} {'matched':>8s}  {'shape-mismatch':>14s}  {'missing':>8s}")
for c in sorted(cat):
    n_m = sum(1 for k in matched if classify(k) == c)
    n_s = sum(1 for k, _, _ in bad_shape if classify(k) == c)
    n_x = sum(1 for k in missing if classify(k) == c)
    print(f"{c:16s} {cat[c]:7d} {n_m:8d}  {n_s:14d}  {n_x:8d}")

print(f"\nTOTAL delta tensors : {len(delta)}")
print(f"  matched           : {len(matched)}")
print(f"  shape mismatch    : {len(bad_shape)}")
print(f"  not found in model: {len(missing)}")
print(f"delta payload size  : {delta_total_bytes / 1e9:.2f} GB")

if bad_shape:
    print("\nSHAPE MISMATCHES (first 10):")
    for k, a, b in bad_shape[:10]:
        print(f"   {k}\n      delta={a}  model={b}")
if missing:
    print("\nNOT FOUND IN MODEL (first 15):")
    for k in missing[:15]:
        print(f"   {k}")

# ------------------------------------------------- strict per-module load test
print("\n" + "=" * 74)
print("STRICT PER-MODULE LOAD TEST (in memory, nothing saved)")
print("=" * 74)

groups: dict[str, dict[str, torch.Tensor]] = collections.defaultdict(dict)
for k, t in delta.items():
    parts = k.split(".")
    if k.startswith("vlm."):
        # vlm.llm_backbone.<rest>  -> target module vlm.llm_backbone
        mod = "vlm." + parts[1]
        sub = ".".join(parts[2:])
    else:
        mod = parts[0]
        sub = ".".join(parts[1:])
    groups[mod][sub] = t

ok = True
for mod, sub in sorted(groups.items()):
    target = vla
    for seg in mod.split("."):
        target = getattr(target, seg, None)
        if target is None:
            break
    if target is None:
        print(f"  FAIL  {mod:28s} module not found on the model")
        ok = False
        continue
    try:
        res = target.load_state_dict(sub, strict=False)
        n = len(sub)
        extra_loaded = n - len(res.unexpected_keys)
        missing_here = [k for k in res.missing_keys]
        # missing_keys here are the frozen ones we intentionally do not supply
        print(f"  OK    {mod:28s} supplied={n:<4d} "
              f"applied={extra_loaded:<4d} unexpected={len(res.unexpected_keys)} "
              f"untouched(frozen)={len(missing_here)}")
        if res.unexpected_keys:
            print(f"        !! unexpected: {res.unexpected_keys[:5]}")
            ok = False
    except Exception as e:                       # noqa: BLE001
        print(f"  FAIL  {mod:28s} {type(e).__name__}: {e}")
        ok = False

print("\n" + "=" * 74)
verdict = (len(missing) == 0 and len(bad_shape) == 0 and ok)
print(f"AUDIT VERDICT: {'PASS' if verdict else 'FAIL'}")
print(f"  every delta tensor maps to a model param with identical shape : "
      f"{len(missing) == 0 and len(bad_shape) == 0}")
print(f"  every module group applies cleanly (no unexpected keys)       : {ok}")
print("=" * 74)
print("No file written, no simulator started, no rollout.")
